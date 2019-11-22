#region

using System.Collections.Concurrent;
using System.Collections.Generic;
using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;
using NavJob.Components;

#endregion

namespace NavJob.Systems
{

    class SetDestinationBarrier : EntityCommandBufferSystem { }
    class PathSuccessBarrier : EntityCommandBufferSystem { }
    class PathErrorBarrier : EntityCommandBufferSystem { }

    //[DisableAutoCreation]
    public class NavAgentSystem : JobComponentSystem
    {

        private struct AgentData
        {
            public int index;
            public Entity entity;
            public NavAgent agent;
        }

        private NativeQueue<AgentData> needsWaypoint;
        private ConcurrentDictionary<int, Vector3[]> waypoints = new ConcurrentDictionary<int, Vector3[]> ();
        private NativeHashMap<int, AgentData> pathFindingData;

        [BurstCompile]
        private struct DetectNextWaypointJob : IJobParallelFor
        {
            public int navMeshQuerySystemVersion;

            [ReadOnly]
            public NativeArray<Entity> entities;
            [NativeDisableParallelForRestriction]
            public ComponentDataFromEntity<NavAgent> agents;
            public NativeQueue<AgentData>.ParallelWriter needsWaypoint;

            public void Execute (int index)
            {
                var entity = entities[index];
                var agent = agents[entity];
                if (agent.remainingDistance - agent.stoppingDistance > 0 || agent.status != AgentStatus.Moving)
                {
                    return;
                }

                if (agent.nextWaypointIndex != agent.totalWaypoints)
                {
                    needsWaypoint.Enqueue (new AgentData { agent = agent, entity = entity, index = index });
                }
                else if (navMeshQuerySystemVersion != agent.queryVersion || agent.nextWaypointIndex == agent.totalWaypoints)
                {
                    agent.totalWaypoints = 0;
                    agent.currentWaypoint = 0;
                    agent.status = AgentStatus.Idle;
                    agents[entity] = agent;
                }
            }
        }

        private struct SetNextWaypointJob : IJob
        {
            public ComponentDataFromEntity<NavAgent> agents;
            public NativeQueue<AgentData> needsWaypoint;
            public void Execute ()
            {
                // TODO: Don't like how this one converted...
                while (needsWaypoint.TryDequeue (out AgentData item))
                {
                    var entity = item.entity;
                    if (NavAgentSystem.instance.waypoints.TryGetValue (entity.Index, out Vector3[] currentWaypoints))
                    {
                        var agent = item.agent;
                        agent.currentWaypoint = currentWaypoints[agent.nextWaypointIndex];
                        agent.remainingDistance = Vector3.Distance (agent.position, agent.currentWaypoint);
                        agent.nextWaypointIndex++;
                        agents[entity] = agent;
                    }
                }
            }
        }

        [BurstCompile]
        private struct MovementJob : IJobParallelFor
        {
            public float dt;
            public float3 up;
            public float3 one;

            [ReadOnly]
            [DeallocateOnJobCompletion]
            public NativeArray<Entity> entities;
            [NativeDisableParallelForRestriction]
            public ComponentDataFromEntity<NavAgent> agents;

            public void Execute (int index)
            {
                var entity = entities[index];

                var agent = agents[entity];
                if (agent.status != AgentStatus.Moving)
                {
                    return;
                }

                if (agent.remainingDistance > 0)
                {
                    agent.currentMoveSpeed = Mathf.Lerp (agent.currentMoveSpeed, agent.moveSpeed, dt * agent.acceleration);
                    // todo: deceleration
                    if (agent.nextPosition.x != Mathf.Infinity)
                    {
                        agent.position = agent.nextPosition;
                    }
                    var heading = (Vector3) (agent.currentWaypoint - agent.position);
                    agent.remainingDistance = heading.magnitude;
                    if (agent.remainingDistance > 0.001f)
                    {
                        var targetRotation = Quaternion.LookRotation (heading, up).eulerAngles;
                        targetRotation.x = targetRotation.z = 0;
                        if (agent.remainingDistance < 1)
                        {
                            agent.rotation = Quaternion.Euler (targetRotation);
                        }
                        else
                        {
                            agent.rotation = Quaternion.Slerp (agent.rotation, Quaternion.Euler (targetRotation), dt * agent.rotationSpeed);
                        }
                    }
                    var forward = math.forward (agent.rotation) * agent.currentMoveSpeed * dt;
                    agent.nextPosition = agent.position + forward;
                    agents[entity] = agent;
                }
                else if (agent.nextWaypointIndex == agent.totalWaypoints)
                {
                    agent.nextPosition = new float3 { x = Mathf.Infinity, y = Mathf.Infinity, z = Mathf.Infinity };
                    agent.status = AgentStatus.Idle;
                    agents[entity] = agent;
                }
            }
        }

        private EntityQuery agentQuery;
        private NavMeshQuerySystem querySystem;
        SetDestinationBarrier setDestinationBarrier;
        PathSuccessBarrier pathSuccessBarrier;
        PathErrorBarrier pathErrorBarrier;

        protected override JobHandle OnUpdate (JobHandle inputDeps)
        {
            var entityCnt = agentQuery.CalculateEntityCount();
            var entities = agentQuery.ToEntityArray(Allocator.TempJob);
            
            var dt = Time.deltaTime;
            inputDeps = new DetectNextWaypointJob {
                entities = entities,
                agents = GetComponentDataFromEntity<NavAgent>(),
                needsWaypoint = needsWaypoint.AsParallelWriter(),
                navMeshQuerySystemVersion = querySystem.Version
            }.Schedule (entityCnt, 64, inputDeps);

            inputDeps = new SetNextWaypointJob
            {
                agents = GetComponentDataFromEntity<NavAgent>(),
                needsWaypoint = needsWaypoint
            }.Schedule (inputDeps);

            inputDeps = new MovementJob
            {
                dt = dt,
                up = Vector3.up,
                one = Vector3.one,
                entities = entities,
                agents = GetComponentDataFromEntity<NavAgent>()
            }.Schedule (entityCnt, 64, inputDeps);

            return inputDeps;
        }

        /// <summary>
        /// Used to set an agent destination and start the pathfinding process
        /// </summary>
        /// <param name="entity"></param>
        /// <param name="agent"></param>
        /// <param name="destination"></param>
        public void SetDestination (Entity entity, NavAgent agent, Vector3 destination, int areas = -1)
        {
            if (pathFindingData.TryAdd (entity.Index, new AgentData { index = entity.Index, entity = entity, agent = agent }))
            {
                var command = setDestinationBarrier.CreateCommandBuffer ();
                agent.status = AgentStatus.PathQueued;
                agent.destination = destination;
                agent.queryVersion = querySystem.Version;
                command.SetComponent<NavAgent> (entity, agent);
                querySystem.RequestPath (entity.Index, agent.position, agent.destination, areas);
            }
        }

        /// <summary>
        /// Static counterpart of SetDestination
        /// </summary>
        /// <param name="entity"></param>
        /// <param name="agent"></param>
        /// <param name="destination"></param>
        public static void SetDestinationStatic (Entity entity, NavAgent agent, Vector3 destination, int areas = -1)
        {
            instance.SetDestination (entity, agent, destination, areas);
        }

        protected static NavAgentSystem instance;

        protected override void OnCreate()
        {
            instance = this;

            querySystem = World.Active.GetOrCreateSystem<NavMeshQuerySystem>();
            setDestinationBarrier = World.Active.GetOrCreateSystem<SetDestinationBarrier>();
            pathSuccessBarrier = World.Active.GetOrCreateSystem<PathSuccessBarrier>();
            pathErrorBarrier = World.Active.GetOrCreateSystem<PathErrorBarrier>();

            querySystem.RegisterPathResolvedCallback (OnPathSuccess);
            querySystem.RegisterPathFailedCallback (OnPathError);

            var agentQueryDesc = new EntityQueryDesc
            {
                All = new ComponentType[] { typeof(NavAgent) }
            };
            agentQuery = GetEntityQuery(agentQueryDesc);

            needsWaypoint = new NativeQueue<AgentData>(Allocator.Persistent);
            pathFindingData = new NativeHashMap<int, AgentData> (0, Allocator.Persistent);
        }

        protected override void OnDestroy()
        {
            needsWaypoint.Dispose ();
            pathFindingData.Dispose ();
        }

        private void SetWaypoint (Entity entity, NavAgent agent, Vector3[] newWaypoints)
        {
            waypoints[entity.Index] = newWaypoints;
            var command = pathSuccessBarrier.CreateCommandBuffer ();
            agent.status = AgentStatus.Moving;
            agent.nextWaypointIndex = 1;
            agent.totalWaypoints = newWaypoints.Length;
            agent.currentWaypoint = newWaypoints[0];
            agent.remainingDistance = Vector3.Distance (agent.position, agent.currentWaypoint);
            command.SetComponent<NavAgent> (entity, agent);
        }

        private void OnPathSuccess (int index, Vector3[] waypoints)
        {
            if (pathFindingData.TryGetValue (index, out AgentData entry))
            {
                SetWaypoint (entry.entity, entry.agent, waypoints);
                pathFindingData.Remove (index);
            }
        }

        private void OnPathError (int index, PathfindingFailedReason reason)
        {
            if (pathFindingData.TryGetValue (index, out AgentData entry))
            {
                entry.agent.status = AgentStatus.Idle;
                var command = pathErrorBarrier.CreateCommandBuffer ();
                command.SetComponent<NavAgent> (entry.entity, entry.agent);
                pathFindingData.Remove (index);
            }
        }
    }
}
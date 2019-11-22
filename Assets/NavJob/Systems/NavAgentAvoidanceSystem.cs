using System.Collections.Concurrent;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.Experimental.AI;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using NavJob.Components;

namespace NavJob.Systems
{
    //[DisableAutoCreation]
    public class NavAgentAvoidanceSystem : JobComponentSystem
    {

        public NativeMultiHashMap<int, int> indexMap;
        public NativeMultiHashMap<int, float3> nextPositionMap;
        NavMeshQuery navMeshQuery;

        [BurstCompile]
        struct NavAgentAvoidanceJob : IJobNativeMultiHashMapMergedSharedKeyIndices
        {
            // This job goes last and should do the deallocation
            [ReadOnly]
            [DeallocateOnJobCompletion]
            public NativeArray<Entity> entities;
            public ComponentDataFromEntity<NavAgent> agents;
            //public ComponentDataFromEntity<NavAgentAvoidance> avoidances;

            [ReadOnly] public NativeMultiHashMap<int, int> indexMap;
            [ReadOnly] public NativeMultiHashMap<int, float3> nextPositionMap;
            [ReadOnly] public NavMeshQuery navMeshQuery;
            public float dt;
            public void ExecuteFirst (int index) { }

            public void ExecuteNext (int firstIndex, int index)
            {
                var entity = entities[index];
                var agent = agents[entity];
                //var avoidance = avoidances[entity];
                var move = Vector3.left;
                if (index % 2 == 1)
                {
                    move = Vector3.right;
                }
                float3 drift = agent.rotation * (Vector3.forward + move) * agent.currentMoveSpeed * dt;
                if (agent.nextWaypointIndex != agent.totalWaypoints)
                {
                    var offsetWaypoint = agent.currentWaypoint + drift;
                    var waypointInfo = navMeshQuery.MapLocation (offsetWaypoint, Vector3.one * 3f, 0, agent.areaMask);
                    if (navMeshQuery.IsValid (waypointInfo))
                    {
                        agent.currentWaypoint = waypointInfo.position;
                    }
                }
                agent.currentMoveSpeed = Mathf.Max (agent.currentMoveSpeed / 2f, 0.5f);
                var positionInfo = navMeshQuery.MapLocation (agent.position + drift, Vector3.one * 3f, 0, agent.areaMask);
                if (navMeshQuery.IsValid (positionInfo))
                {
                    agent.nextPosition = positionInfo.position;
                }
                else
                {
                    agent.nextPosition = agent.position;
                }
                agents[entity] = agent;
            }
        }

        [BurstCompile]
        struct HashPositionsJob : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<Entity> entities;
            [ReadOnly]
            public ComponentDataFromEntity<NavAgent> agents;
            public ComponentDataFromEntity<NavAgentAvoidance> avoidances;
            public NativeMultiHashMap<int, int>.ParallelWriter indexMap;
            public NativeMultiHashMap<int, float3>.ParallelWriter nextPositionMap;
            public int mapSize;

            public void Execute (int index)
            {
                var entity = entities[index];
                var agent = agents[entity];
                var avoidance = avoidances[entity];
                var hash = Hash (agent.position, avoidance.radius);
                indexMap.Add (hash, index);
                nextPositionMap.Add (hash, agent.nextPosition);
                avoidance.partition = hash;
                avoidances[entity] = avoidance;
            }

            public int Hash (float3 position, float radius)
            {
                int ix = Mathf.RoundToInt ((position.x / radius) * radius);
                int iz = Mathf.RoundToInt ((position.z / radius) * radius);
                return ix * mapSize + iz;
            }
        }

        /*struct InjectData
        {
            public readonly int Length;
            [ReadOnly] public EntityArray Entities;
            public ComponentDataArray<NavAgent> Agents;
            public ComponentDataArray<NavAgentAvoidance> Avoidances;
        }*/

        private EntityQuery agentQuery;

        //[Inject] InjectData agent;
        /*[Inject]*/ NavMeshQuerySystem querySystem;
        protected override JobHandle OnUpdate (JobHandle inputDeps)
        {
            // In theory, JobComponentSystem will by default not run OnUpdate if the agentQuery is empty to begin with.
            var agentCnt = agentQuery.CalculateEntityCount();

            if (agentCnt > 0)
            {
                var agentEntities = agentQuery.ToEntityArray(Allocator.TempJob);

                indexMap.Clear ();
                nextPositionMap.Clear ();

                var hashPositionsJob = new HashPositionsJob
                {
                    mapSize = querySystem.MaxMapWidth,
                    entities = agentEntities,
                    agents = GetComponentDataFromEntity<NavAgent>(true),
                    avoidances = GetComponentDataFromEntity<NavAgentAvoidance>(),
                    indexMap = indexMap.AsParallelWriter(),
                    nextPositionMap = nextPositionMap.AsParallelWriter()
                };
                var dt = Time.deltaTime;
                var hashPositionsJobHandle = hashPositionsJob.Schedule (agentCnt, 64, inputDeps);
                var avoidanceJob = new NavAgentAvoidanceJob
                {
                    dt = dt,
                    indexMap = indexMap,
                    nextPositionMap = nextPositionMap,
                    agents = GetComponentDataFromEntity<NavAgent>(),
                    entities = agentEntities, // Set to deallocate
                    navMeshQuery = navMeshQuery
                };
                var avoidanceJobHandle = avoidanceJob.Schedule (indexMap, 64, hashPositionsJobHandle);


                return avoidanceJobHandle;
            }
            return inputDeps;
        }

        protected override void OnCreate()
        {
            var agentQueryDesc = new EntityQueryDesc
            {
                All = new ComponentType[] {typeof(NavAgent), typeof(NavAgentAvoidance)}
            };
            agentQuery = GetEntityQuery(agentQueryDesc);
            navMeshQuery = new NavMeshQuery (NavMeshWorld.GetDefaultWorld (), Allocator.Persistent, 128);
            indexMap = new NativeMultiHashMap<int, int> (100 * 1024, Allocator.Persistent);
            nextPositionMap = new NativeMultiHashMap<int, float3> (100 * 1024, Allocator.Persistent);
            querySystem = World.Active.GetOrCreateSystem<NavMeshQuerySystem>();
        }

        protected override void OnDestroy()
        {

            if (indexMap.IsCreated) indexMap.Dispose ();
            if (nextPositionMap.IsCreated) nextPositionMap.Dispose ();
            navMeshQuery.Dispose ();
        }
    }

}
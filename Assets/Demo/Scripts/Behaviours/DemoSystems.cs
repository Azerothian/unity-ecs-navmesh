#region

using UnityEngine;
using UnityEngine.UI;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;
using Demo.Behaviours;
using NavJob.Components;
using NavJob.Systems;
using Unity.Collections.LowLevel.Unsafe;

#endregion

namespace Demo
{
    public class SpawnSystem : ComponentSystem
    {
        public int pendingSpawn;
        private EntityManager _manager;

        private PopulationSpawner _spawner;
        private int _lastSpawned;
        private float _nextUpdate;

        private Vector3 one = Vector3.one;
        private EntityArchetype agent;

        private int spawned;

        private Text spawnedText;

        private Text SpawnedText
        {
            get
            {
                if (spawnedText == null)
                {
                    spawnedText = GameObject.Find ("SpawnedText").GetComponent<Text> ();
                }

                return spawnedText;
            }
        }

        private PopulationSpawner GetSpawner ()
        {
            if (_spawner == null)
            {
                _spawner = Object.FindObjectOfType<PopulationSpawner> ();
            }

            return _spawner;
        }

        private EntityQuery spawnQuery;
        protected override void OnCreate()
        {
            base.OnCreate();

            // create the system
            World.Active.CreateSystem<NavAgentSystem> ();
            World.Active.GetOrCreateSystem<NavAgentToTransfomMatrixSyncSystem> ();
            buildings = World.Active.GetOrCreateSystem<BuildingCacheSystem>();

            var spawnQueryDesc = new EntityQueryDesc
            {
                All = new ComponentType[] { typeof(PendingSpawn) }
            };
            spawnQuery = GetEntityQuery(spawnQueryDesc);

            agent = EntityManager.CreateArchetype (
                typeof (NavAgent),
                // optional avoidance
                // typeof(NavAgentAvoidance),
                // optional
                // typeof (Position),
                // typeof (Rotation),
                // typeof (SyncPositionToNavAgent),
                // typeof (SyncRotationToNavAgent),
                // typeof (SyncPositionFromNavAgent),
                // typeof (SyncRotationFromNavAgent),
                typeof (LocalToWorld)
            );


        }

        private BuildingCacheSystem buildings;

        //[Inject] private InjectData data;

        protected override void OnUpdate()
        {
            if (Time.time > _nextUpdate && _lastSpawned != spawned)
            {
                _nextUpdate = Time.time + 0.5f;
                _lastSpawned = spawned;
                SpawnedText.text = $"Spawned: {spawned} people";
            }

            if (GetSpawner().Renderers.Length == 0)
            {
                return;
            }

            if (buildings.ResidentialBuildings.Length == 0)
            {
                return;
            }

            var pendings = GetComponentDataFromEntity<PendingSpawn>();
            var entities = spawnQuery.ToEntityArray(Allocator.TempJob);
            var rootEntity = entities[0];

            var spawnData = pendings[rootEntity];
            pendingSpawn = spawnData.Quantity;
            spawnData.Quantity = 0;
            pendings[rootEntity] = spawnData;
            var manager = EntityManager;
            for (var i = 0; i < pendingSpawn; i++)
            {
                spawned++;
                var position = buildings.GetResidentialBuilding ();
                var entity = manager.CreateEntity (agent);
                var navAgent = new NavAgent (
                    position,
                    Quaternion.identity,
                    spawnData.AgentStoppingDistance,
                    spawnData.AgentMoveSpeed,
                    spawnData.AgentAcceleration,
                    spawnData.AgentRotationSpeed,
                    spawnData.AgentAreaMask
                );
                // optional if set on the archetype
                // manager.SetComponentData (entity, new Position { Value = position });
                manager.SetComponentData (entity, navAgent);
                // optional for avoidance
                // var navAvoidance = new NavAgentAvoidance(2f);
                // manager.SetComponentData(entity, navAvoidance);
                manager.AddSharedComponentData (entity, GetSpawner ().Renderers[UnityEngine.Random.Range (0, GetSpawner ().Renderers.Length)].Value);
            }

            entities.Dispose();
        }

        /*private struct InjectData
        {
            public readonly int Length;
            public ComponentDataArray<PendingSpawn> Spawn;
        }*/
    }

    public class DetectIdleAgentSystem : ComponentSystem
    {
        public struct AgentData
        {
            public int index;
            public Entity entity;
            public NavAgent agent;
        }

        private Text awaitingNavmeshText;

        private Text AwaitingNavmeshText
        {
            get
            {
                if (awaitingNavmeshText == null)
                {
                    awaitingNavmeshText = GameObject.Find ("AwaitingNavmeshText").GetComponent<Text> ();
                }

                return awaitingNavmeshText;
            }
        }

        private Text cachedPathText;

        private Text CachedPathText
        {
            get
            {
                if (cachedPathText == null)
                {
                    cachedPathText = GameObject.Find ("CachedPathText").GetComponent<Text> ();
                }

                return cachedPathText;
            }
        }

        private float _nextUpdate;

        private NativeQueue<AgentData> needsPath = new NativeQueue<AgentData> (Allocator.Persistent);

        [BurstCompile]
        private struct DetectIdleAgentJob : IJobParallelFor
        {
            [ReadOnly]
            [DeallocateOnJobCompletion]
            public NativeArray<Entity> entities;
            [NativeDisableParallelForRestriction]
            public ComponentDataFromEntity<NavAgent> agents;
            public NativeQueue<AgentData>.ParallelWriter needsPath;

            public void Execute (int index)
            {
                var entity = entities[index];
                var agent = agents[entity];
                if (agent.status == AgentStatus.Idle)
                {
                    needsPath.Enqueue (new AgentData { index = index, agent = agent, entity = entity });
                    agent.status = AgentStatus.PathQueued;
                    agents[entity] = agent;
                }
            }
        }

        private struct SetNextPathJob : IJob
        {
            public NativeQueue<AgentData> needsPath;
            public void Execute ()
            {
                while (needsPath.TryDequeue (out AgentData item))
                {
                    var destination = BuildingCacheSystem.GetCommercialBuilding ();
                    NavAgentSystem.SetDestinationStatic (item.entity, item.agent, destination, item.agent.areaMask);
                }
            }
        }

        private EntityQuery agentQuery;
        NavMeshQuerySystem navQuery;

        protected override void OnUpdate ()
        {
            if (Time.time > _nextUpdate)
            {
                AwaitingNavmeshText.text = $"Awaiting Path: {navQuery.PendingCount} people";
                CachedPathText.text = $"Cached Paths: {navQuery.CachedCount}";
                _nextUpdate = Time.time + 0.5f;
            }

            var entityCnt = agentQuery.CalculateEntityCount();
            var entities = agentQuery.ToEntityArray(Allocator.TempJob);

            var inputDeps = new DetectIdleAgentJob
            {
                entities = entities,
                agents = GetComponentDataFromEntity<NavAgent>(),
                needsPath = needsPath.AsParallelWriter()
            }.Schedule (entityCnt, 64);
            inputDeps = new SetNextPathJob
            {
                needsPath = needsPath
            }.Schedule (inputDeps);
            inputDeps.Complete ();
        }

        protected override void OnCreate()
        {
            base.OnCreate();
            var agentQueryDesc = new EntityQueryDesc
            {
                All = new ComponentType[] { typeof(NavAgent) }
            };
            agentQuery = GetEntityQuery(agentQueryDesc);
            navQuery = World.Active.GetOrCreateSystem<NavMeshQuerySystem>();
        }

        protected override void OnDestroy()
        {
            needsPath.Dispose ();
        }
    }

    public class BuildingCacheSystem : ComponentSystem
    {
        public NativeList<Vector3> CommercialBuildings = new NativeList<Vector3> (Allocator.Persistent);
        public NativeList<Vector3> ResidentialBuildings = new NativeList<Vector3> (Allocator.Persistent);
        private PopulationSpawner spawner;
        private int nextCommercial = 0;
        private int nextResidential = 0;
        //private EntityQuery buildingQuery;
        private static BuildingCacheSystem instance;

        protected override void OnCreate()
        {
            instance = this;
        }

        private PopulationSpawner Spawner
        {
            get
            {
                if (spawner == null)
                {
                    spawner = Object.FindObjectOfType<PopulationSpawner> ();
                }

                return spawner;
            }
        }

        public Vector3 GetResidentialBuilding ()
        {
            nextResidential++;
            if (nextResidential >= ResidentialBuildings.Length)
            {
                nextResidential = 0;
            }

            return ResidentialBuildings[nextResidential];
        }

        public static Vector3 GetCommercialBuilding ()
        {
            var building = instance.CommercialBuildings[0];
            try
            {
                if (instance.nextCommercial < instance.CommercialBuildings.Length)
                {
                    building = instance.CommercialBuildings[instance.nextCommercial];
                    instance.nextCommercial++;
                }
                else
                {
                    instance.nextCommercial = 0;
                }
                return building;
            }
            catch
            {
                return building;
            }
        }

        protected override void OnUpdate ()
        {

            Entities.WithAll<BuildingData>().ForEach((ref BuildingData building) =>
            {
                if (building.Type == BuildingType.Residential)
                {
                    ResidentialBuildings.Add(building.Position);
                }
                else
                {
                    CommercialBuildings.Add(building.Position);
                }

                PostUpdateCommands.RemoveComponent<BuildingData>(building.Entity);
            });

            /*var buildings = buildingQuery.ToComponentDataArray<BuildingData>(Allocator.Temp);
            
            for (var i = 0; i < buildings.Length; i++)
            {
                var building = buildings[i];
                if (building.Type == BuildingType.Residential)
                {
                    ResidentialBuildings.Add (building.Position);
                }
                else
                {
                    CommercialBuildings.Add (building.Position);
                }

                PostUpdateCommands.RemoveComponent<BuildingData> (building.Entity);
            }

            buildings.Dispose();*/
        }

        protected override void OnDestroy()
        {
            ResidentialBuildings.Dispose ();
            CommercialBuildings.Dispose ();
        }
    }
}
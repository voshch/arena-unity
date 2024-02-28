using System;
using System.Threading.Tasks;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using Unity.Robotics.Core;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using System.Collections.Generic;



// this job creates a batch of RaycastCommands we are going to use for
// collision detection against the world. these can be sent to PhysX
// as a batch that will be executed in a job, rather than us having to
// call Physics.Raycast in a loop just on the main thread! 
[BurstCompile(CompileSynchronously = true)]
struct ZOPrepareRaycastCommands : IJobParallelFor
{
    public Vector3 Position;
    public Quaternion Orientation;
    public float Distance;
    public NativeArray<RaycastCommand> RaycastCommands;
    [ReadOnly]
    public NativeArray<Vector3> RaycastDirections;

    public void Execute(int i)
    {
        RaycastCommands[i] = new RaycastCommand(Position, Orientation * RaycastDirections[i], QueryParameters.Default, Distance);
    }
};

[BurstCompile(CompileSynchronously = true)]
struct ZOTransformHits : IJobParallelFor
{
    [ReadOnly] public ZOLIDAR3D.ReferenceFrame ReferenceFrame;
    [ReadOnly] public Vector3 Position;

    [ReadOnly] public NativeArray<RaycastHit> RaycastHits;

    public NativeArray<Vector3> HitPositions;
    public NativeArray<Vector3> HitNormals;


    public void Execute(int i)
    {

        RaycastHit hit = RaycastHits[i];
        if (ReferenceFrame == ZOLIDAR3D.ReferenceFrame.LeftHanded_XRight_YUp_ZForward)
        {
            HitNormals[i] = hit.normal;
            HitPositions[i] = hit.point - Position;
        }
        else if (ReferenceFrame == ZOLIDAR3D.ReferenceFrame.RightHanded_XBackward_YLeft_ZUp)
        {
            HitNormals[i] = new Vector3(-hit.normal.z, -hit.normal.x, hit.normal.y);
            Vector3 pos = hit.point - Position;
            HitPositions[i] = new Vector3(-pos.z, -pos.x, pos.y);
        }
    }
};

public class ZOLIDAR3D : MonoBehaviour
{
    /// <summary>
    /// A generic LIDAR sensor.  
    /// </summary>

    public enum ReferenceFrame
    {
        RightHanded_XBackward_YLeft_ZUp,
        LeftHanded_XRight_YUp_ZForward // Unity Standard
    };

    public ReferenceFrame _referenceFrame = ReferenceFrame.LeftHanded_XRight_YUp_ZForward;



    [Header("FOV")]
    public float _verticalUpFovDegrees = 21.0f;
    public float _verticalDownFovDegrees = 74.0f;
    public float _horizontalFovDegrees = 360.0f;

    [Header("Resolution")]
    public float _verticalResolutionDegrees = 0.76f;
    public float _horizontalResolutionDegrees = 1.2f;
    public float _minRange = 0.0f;
    public float _maxRange = 20.0f;

    // Property Accessors
    public float HorizontalFOVDegrees { get => _horizontalFovDegrees; }
    public float HorizontalResolutionDegrees { get => _horizontalResolutionDegrees; }
    public float MinRange { get => _minRange; }
    public float MaxRange { get => _maxRange; }


    private int _horizontalRayCount = -1;
    private int _verticalScanCount = -1;
    private int _totalRayCount = -1;

    private ZORaycastJobBatch _raycastBatchJob;
    private RaycastHit[] _rayCastHits;

    private NativeArray<Vector3> _hitPositions;
    private NativeArray<Vector3> _hitNormals;

    /// Pre-calculated rays
    NativeArray<Vector3> _rayDirections;

    ///
    JobHandle _transformHitsJobHandle = default(JobHandle);


    private double TimeNextScanSeconds = 0;
    private double TimeLastScanBeganSeconds = 0;
    private int NumMeasurementsTaken = 0;
    private double PublishPeriodSeconds = 0.1;
    private ROSConnection rosConnection;
    public string topic = "lidar3d";
    public string topicNamespace = "";
    public string frame_id = "lidar_frame";
    string PublishTopic => topicNamespace + "/" + topic;

    public void ConfigureLidar(Dictionary<string, object> configDict, string robotName, string jointName)
    {
        if (configDict.ContainsKey("topic"))
            topic = (string)configDict["topic"];

        if (configDict.ContainsKey("verticalUpFovDegrees"))
            float.TryParse((string)configDict["verticalUpFovDegrees"], out _verticalUpFovDegrees);

        if (configDict.ContainsKey("verticalDownFovDegrees"))
            float.TryParse((string)configDict["verticalDownFovDegrees"], out _verticalDownFovDegrees);

        if (configDict.ContainsKey("horizontalFovDegrees"))
            float.TryParse((string)configDict["horizontalFovDegrees"], out _horizontalFovDegrees);

        if (configDict.ContainsKey("verticalResolutionDegrees"))
            float.TryParse((string)configDict["verticalResolutionDegrees"], out _verticalResolutionDegrees);

        if (configDict.ContainsKey("horizontalResolutionDegrees"))
            float.TryParse((string)configDict["horizontalResolutionDegrees"], out _horizontalResolutionDegrees);

        if (configDict.ContainsKey("minRange"))
            float.TryParse((string)configDict["minRange"], out _minRange);

        if (configDict.ContainsKey("maxRange"))
            float.TryParse((string)configDict["maxRange"], out _maxRange);

    }
    // Start is called before the first frame update
    public void Start()
    {
        rosConnection = ROSConnection.GetOrCreateInstance();
        rosConnection.RegisterPublisher<PointCloud2Msg>(PublishTopic);

        Debug.Log("INFO: ZOLIDAR::Start");

        _horizontalRayCount = Mathf.RoundToInt(_horizontalFovDegrees / _horizontalResolutionDegrees);
        _verticalScanCount = Mathf.RoundToInt((_verticalDownFovDegrees + _verticalUpFovDegrees) / _verticalResolutionDegrees);
        _totalRayCount = _horizontalRayCount * _verticalScanCount;

        _raycastBatchJob = new ZORaycastJobBatch(_totalRayCount, Allocator.TempJob);


        _rayCastHits = new RaycastHit[_totalRayCount];

        // build up the ray directions
        _rayDirections = new NativeArray<Vector3>(_totalRayCount, Allocator.Persistent);
        Vector3 rayDirection = Quaternion.AngleAxis(_verticalUpFovDegrees, transform.right) * transform.forward;
        Quaternion horizontalRotationStep = Quaternion.AngleAxis(_horizontalResolutionDegrees, transform.up);
        Quaternion verticalRotationStep = Quaternion.AngleAxis(-_verticalResolutionDegrees, transform.right);
        int rayIndex = 0;
        for (int verticalStep = 0; verticalStep < _verticalScanCount; verticalStep++)
        {
            for (int horizontalStep = 0; horizontalStep < _horizontalRayCount; horizontalStep++)
            {
                _rayDirections[rayIndex] = rayDirection;
                rayIndex++;
                rayDirection = horizontalRotationStep * rayDirection;
            }

            // BUGBUG:??? transform.right may change so do we need to have a rightDirection that moves with the rayDirection
            rayDirection = verticalRotationStep * rayDirection;
        }

        _hitPositions = new NativeArray<Vector3>(_totalRayCount, Allocator.Persistent);
        _hitNormals = new NativeArray<Vector3>(_totalRayCount, Allocator.Persistent);

    }


    private void OnDestroy()
    {
        _transformHitsJobHandle.Complete();
        _rayDirections.Dispose();
        _hitNormals.Dispose();
        _hitPositions.Dispose();
        _raycastBatchJob.Dispose();
    }


    public void Update()
    {

        if (Clock.NowTimeInSeconds < TimeNextScanSeconds)
        {
            return;
        }

        TimeLastScanBeganSeconds = Clock.Now;
        TimeNextScanSeconds = TimeLastScanBeganSeconds + PublishPeriodSeconds;
        FixedHzUpdate();

    }

    protected void FixedHzUpdate()
    {
        UnityEngine.Profiling.Profiler.BeginSample("ZOLIDAR::ZOUpdateHzSynchronized");
        if (_transformHitsJobHandle.IsCompleted == true)
        {

            _transformHitsJobHandle.Complete();



            Publish();


            // Ref: https://github.com/LotteMakesStuff/SimplePhysicsDemo/blob/master/Assets/SimpleJobifiedPhysics.cs
            // create new raycast job
            _raycastBatchJob.Dispose();
            _raycastBatchJob = new ZORaycastJobBatch(_totalRayCount, Allocator.TempJob);
            // SetupRaycasts();
            var setupRaycastsJob = new ZOPrepareRaycastCommands()
            {
                Position = transform.position,
                Orientation = transform.rotation,
                Distance = _maxRange,
                RaycastDirections = _rayDirections,
                RaycastCommands = _raycastBatchJob.RaycastCommands
            };
            JobHandle setupRaycastsJobHandle = setupRaycastsJob.Schedule(_totalRayCount, 32);

            _raycastBatchJob.Schedule(32, setupRaycastsJobHandle);

            var transformHitJob = new ZOTransformHits()
            {
                ReferenceFrame = _referenceFrame,
                Position = transform.position,
                RaycastHits = _raycastBatchJob.RaycastHitResults,
                HitPositions = _hitPositions,
                HitNormals = _hitNormals
            };

            _transformHitsJobHandle = transformHitJob.Schedule(_totalRayCount, 32, _raycastBatchJob.RaycastBatchJobHandle);


        }
        UnityEngine.Profiling.Profiler.EndSample();

    }





    void Publish()
    {
        UnityEngine.Profiling.Profiler.BeginSample("ZOLIDAR::ZOUpdateHzSynchronized::Publish");
        // Publish the data
        PointCloud2Msg pointCloud = new PointCloud2Msg
        {
            header = new HeaderMsg((uint)NumMeasurementsTaken++, new TimeStamp(Unity.Robotics.Core.Clock.time), frame_id),
            fields = new PointFieldMsg[]{
                new PointFieldMsg("x", 0, PointFieldMsg.FLOAT32, 1),
                new PointFieldMsg("y", 4, PointFieldMsg.FLOAT32, 1),
                new PointFieldMsg("z", 8, PointFieldMsg.FLOAT32, 1)
            },
            height = 1,
            width = (uint)_totalRayCount,
            is_bigendian = false,
            point_step = 12,
            row_step = (uint)(_totalRayCount * 12),
            is_dense = false,
        };

        byte[] data = new byte[pointCloud.row_step * pointCloud.height];

        // initialized to -sizeof(float) so that the first increment is 0, because there is no post-addition-assignment analogous to the post-increment ; sadge
        // i'll make my own programming language just to have that operator, and it will be called "C+=+"
        // we could also make some preprocessor macros to make as much pluses as we need to reach the size in bytes of a float, which would be 💀
        int byteDataOffset = -sizeof(float);
        foreach (Vector3 hit in _hitPositions)
        {
            var transformedHit = transform.InverseTransformVector(hit);
            Buffer.BlockCopy(BitConverter.GetBytes(transformedHit.z), 0, data, byteDataOffset += sizeof(float), sizeof(float));
            Buffer.BlockCopy(BitConverter.GetBytes(-transformedHit.x), 0, data, byteDataOffset += sizeof(float), sizeof(float));
            Buffer.BlockCopy(BitConverter.GetBytes(transformedHit.y), 0, data, byteDataOffset += sizeof(float), sizeof(float));
        }

        pointCloud.data = data;

        rosConnection.Publish(PublishTopic, pointCloud);

        UnityEngine.Profiling.Profiler.EndSample();
    }


}

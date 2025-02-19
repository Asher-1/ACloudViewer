from typing import Iterable, Iterator, Tuple

from typing import overload
import numpy
CV_LOCAL_MODEL_MIN_SIZE: int
NAN_VALUE: float
PC_NAN: float
PC_ONE: float
POINT_HIDDEN: int
POINT_OUT_OF_FOV: int
POINT_OUT_OF_RANGE: int
POINT_VISIBLE: int
SQRT_3: float
ZERO_TOLERANCE_D: float
ZERO_TOLERANCE_F: float
ZERO_TOLERANCE_POINT_COORDINATE: float
ZERO_TOLERANCE_SCALAR: float
c_FastMarchingNeighbourPosShift: int
c_erfRelativeError: float

def DegreesToRadians(arg0: float) -> float: ...
def GreaterThanEpsilon(arg0: float) -> bool: ...
def LessThanEpsilon(arg0: float) -> bool: ...
def RadiansToDegrees(arg0: float) -> float: ...
def delete(arg0: ReferenceCloud) -> None: ...

class AutoSegmentationTools:
    def __init__(self, *args, **kwargs) -> None: ...
    def extractConnectedComponents(self, *args, **kwargs) -> Any: ...
    def frontPropagationBasedSegmentation(self, *args, **kwargs) -> Any: ...
    def labelConnectedComponents(self, *args, **kwargs) -> Any: ...

class BoundingBox:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: CCVector3, arg1: CCVector3) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def add(self, aPoint: CCVector3) -> None: ...
    def clear(self) -> None: ...
    def computeVolume(self) -> float: ...
    def contains(self, P: CCVector3) -> bool: ...
    def getCenter(self) -> CCVector3: ...
    def getDiagNorm(self) -> float: ...
    def getDiagNormd(self) -> float: ...
    def getDiagVec(self) -> CCVector3: ...
    def getMaxBoxDim(self) -> float: ...
    def getMinBoxDim(self) -> float: ...
    def isValid(self) -> bool: ...
    def maxCorner(self) -> CCVector3: ...
    def minCorner(self) -> CCVector3: ...
    def minDistTo(self, box: BoundingBox) -> float: ...
    def setValidity(self, state: bool) -> None: ...
    def __add__(self, arg0: BoundingBox) -> BoundingBox: ...
    @overload
    def __iadd__(self, aBBox: BoundingBox) -> BoundingBox: ...
    @overload
    def __iadd__(self, aVector: CCVector3) -> BoundingBox: ...
    @overload
    def __iadd__(*args, **kwargs) -> Any: ...
    @overload
    def __imul__(self, scaleFactor: float) -> BoundingBox: ...
    @overload
    def __imul__(self, aMatrix) -> BoundingBox: ...
    @overload
    def __imul__(*args, **kwargs) -> Any: ...
    def __isub__(self, arg0: CCVector3) -> BoundingBox: ...

class CCMiscTools:
    def __init__(self, *args, **kwargs) -> None: ...
    def ComputeBaseVectors(self, *args, **kwargs) -> Any: ...
    def EnlargeBox(self, *args, **kwargs) -> Any: ...
    def MakeMinAndMaxCubical(self, *args, **kwargs) -> Any: ...
    def TriBoxOverlap(self, *args, **kwargs) -> Any: ...
    def TriBoxOverlapd(self, *args, **kwargs) -> Any: ...

class CCShareable:
    def __init__(self) -> None: ...
    def getLinkCount(self) -> int: ...
    def link(self) -> None: ...
    def release(self) -> None: ...

class CCVector2:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: float, arg1: float) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def cross(self, arg0: CCVector2) -> float: ...
    def dot(self, arg0: CCVector2) -> float: ...
    def norm(self) -> float: ...
    def norm2(self) -> float: ...
    def normalize(self) -> None: ...
    def __getitem__(self, arg0: int) -> float: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, val: float) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, val: float) -> None: ...

class CCVector2d:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: float, arg1: float) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def cross(self, arg0: CCVector2d) -> float: ...
    def dot(self, arg0: CCVector2d) -> float: ...
    def norm(self) -> float: ...
    def norm2(self) -> float: ...
    def normalize(self) -> None: ...
    def __getitem__(self, arg0: int) -> float: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, val: float) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, val: float) -> None: ...

class CCVector3:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def __add__(self, arg0: CCVector3) -> CCVector3: ...
    def __div__(self, arg0: float) -> CCVector3: ...
    def __mul__(self, arg0: float) -> CCVector3: ...
    def __sub__(self, arg0: CCVector3) -> CCVector3: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, val: float) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, val: float) -> None: ...
    @property
    def z(self) -> float: ...
    @z.setter
    def z(self, val: float) -> None: ...

class CCVector3d:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def __add__(self, arg0: CCVector3d) -> CCVector3d: ...
    def __div__(self, arg0: float) -> CCVector3d: ...
    def __mul__(self, arg0: float) -> CCVector3d: ...
    def __sub__(self, arg0: CCVector3d) -> CCVector3d: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, val: float) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, val: float) -> None: ...
    @property
    def z(self) -> float: ...
    @z.setter
    def z(self, val: float) -> None: ...

class CHAMFER_DISTANCE_TYPE:
    CHAMFER_111: Any = ...
    CHAMFER_345: Any = ...
    __entries: Any = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> Any: ...
    @property
    def __doc__(self) -> Any: ...
    @property
    def __members__(self) -> Any: ...

class ChamferDistanceTransform(Grid3Dus):
    def __init__(self, *args, **kwargs) -> None: ...
    def init(self, gridSize: Tuple3ui) -> bool: ...
    def propagateDistance(self, type: CHAMFER_DISTANCE_TYPE, progressCb: GenericProgressCallback = ...) -> int: ...
    @property
    def MAX_DIST(self) -> Any: ...

class CloudSamplingTools:
    CELL_CENTER: Any = ...
    CELL_GRAVITY_CENTER: Any = ...
    NEAREST_POINT_TO_CELL_CENTER: Any = ...
    RANDOM_POINT: Any = ...
    RESAMPLING_CELL_METHOD: Any = ...
    SFModulationParams: Any = ...
    SUBSAMPLING_CELL_METHODS: Any = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def noiseFilter(self, *args, **kwargs) -> Any: ...
    def resampleCloudSpatially(self, *args, **kwargs) -> Any: ...
    def resampleCloudWithOctree(self, *args, **kwargs) -> Any: ...
    def resampleCloudWithOctreeAtLevel(self, *args, **kwargs) -> Any: ...
    def sorFilter(self, *args, **kwargs) -> Any: ...
    def subsampleCloudRandomly(self, *args, **kwargs) -> Any: ...
    def subsampleCloudWithOctree(self, *args, **kwargs) -> Any: ...
    def subsampleCloudWithOctreeAtLevel(self, *args, **kwargs) -> Any: ...

class Delaunay2dMesh(GenericIndexedMesh):
    def __init__(self) -> None: ...
    def Available(self, *args, **kwargs) -> Any: ...
    def TesselateContour(self, *args, **kwargs) -> Any: ...
    @overload
    def buildMesh(self, points2D, std, pointCountToUse: int, outputErrorStr: str) -> bool: ...
    @overload
    def buildMesh(*args, **kwargs) -> Any: ...
    def getAssociatedCloud(self) -> GenericIndexedCloud: ...
    @overload
    def getTriangleVertIndexesArray(self) -> int: ...
    @overload
    def getTriangleVertIndexesArray(self, maxEdgeLength: float) -> bool: ...
    @overload
    def getTriangleVertIndexesArray(*args, **kwargs) -> Any: ...
    def linkMeshWith(self, aCloud: GenericIndexedCloud, passOwnership: bool = ...) -> None: ...
    def removeOuterTriangles(self, *args, **kwargs) -> Any: ...
    @property
    def USE_ALL_POINTS(self) -> Any: ...

class DgmOctree(GenericOctree):
    CellDescriptor: Any = ...
    NearestNeighboursSearchStruct: Any = ...
    NeighbourCellsSet: Any = ...
    NeighboursSet: Any = ...
    PointDescriptor: Any = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def GET_BIT_SHIFT(self, *args, **kwargs) -> Any: ...
    def OCTREE_LENGTH(self, *args, **kwargs) -> Any: ...
    @overload
    def build(self, progressCb: GenericProgressCallback = ...) -> int: ...
    @overload
    def build(self, octreeMin: CCVector3, octreeMax: CCVector3, pointsMinFilter: CCVector3 = ..., pointsMaxFilter: CCVector3 = ..., progressCb: GenericProgressCallback = ...) -> int: ...
    @overload
    def build(*args, **kwargs) -> Any: ...
    def clear(self) -> None: ...
    def getBoundingBox(self, bbMin: CCVector3, bbMax: CCVector3) -> None: ...
    def getCellSize(self, level: int) -> float: ...
    def getMaxFillIndexes(self, level: int) -> int: ...
    def getMinFillIndexes(self, level: int) -> int: ...
    def getNumberOfProjectedPoints(self) -> int: ...
    def getOctreeMaxs(self) -> CCVector3: ...
    def getOctreeMins(self) -> CCVector3: ...
    @property
    def INVALID_CELL_CODE(self) -> Any: ...
    @property
    def MAX_OCTREE_LENGTH(self) -> Any: ...
    @property
    def MAX_OCTREE_LEVEL(self) -> Any: ...

class DgmOctreeReferenceCloud(GenericIndexedCloudPersist):
    def __init__(self, *args, **kwargs) -> None: ...
    def forwardIterator(self) -> None: ...

class DistanceComputationTools:
    ERRPOR_MEASURES: Any = ...
    MAX_DIST: Any = ...
    MAX_DIST_68_PERCENT: Any = ...
    MAX_DIST_95_PERCENT: Any = ...
    MAX_DIST_99_PERCENT: Any = ...
    RMS: Any = ...
    def __init__(self, *args, **kwargs) -> None: ...

class ErrorFunction:
    def __init__(self, *args, **kwargs) -> None: ...
    def erf(self, *args, **kwargs) -> Any: ...
    def erfc(self, *args, **kwargs) -> Any: ...

class FPCSRegistrationTools(RegistrationTools):
    def __init__(self, *args, **kwargs) -> None: ...
    def RegisterClouds(self, *args, **kwargs) -> Any: ...

class FastMarching:
    def __init__(self, *args, **kwargs) -> None: ...
    def cleanLastPropagation(self) -> None: ...
    def getTime(self, pos: Tuple3i, absoluteCoordinates: bool = ...) -> float: ...
    def propagate(self, arg0: Tuple3i) -> bool: ...
    def setExtendedConnectivity(self, state: bool) -> None: ...
    def setSeedCell(self, pos: Tuple3i) -> bool: ...

class GenericCloud:
    def __init__(self, *args, **kwargs) -> None: ...
    def enableScalarField(self) -> bool: ...
    def forEach(self, action: Callable[[CCVector3,float],None]) -> None: ...
    def getBoundingBox(self, arg0: CCVector3, arg1: CCVector3) -> None: ...
    def getNextPoint(self) -> CCVector3: ...
    def getPointScalarValue(self, arg0: int) -> float: ...
    def isScalarFieldEnabled(self) -> bool: ...
    def placeIteratorAtBeginning(self) -> None: ...
    def setPointScalarValue(self, arg0: int) -> float: ...
    def size(self) -> int: ...
    def testVisibility(self, arg0: CCVector3) -> int: ...

class GenericDistribution:
    ScalarContainer: Any = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def computeChi2Dist(self, Yk: GenericCloud, numberOfClasses: int, histo: int = ...) -> float: ...
    @overload
    def computeP(self, x: float) -> float: ...
    @overload
    def computeP(self, x1: float, x2: float) -> float: ...
    @overload
    def computeP(*args, **kwargs) -> Any: ...
    def computeParameters(self, values, std) -> bool: ...
    def computePfromZero(self, x: float) -> float: ...
    def getName(self) -> str: ...
    def isValid(self) -> bool: ...

class GenericIndexedCloud(GenericCloud):
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    def getPoint(self, arg0: int) -> CCVector3: ...
    @overload
    def getPoint(self, arg0: int, arg1: CCVector3) -> None: ...
    @overload
    def getPoint(*args, **kwargs) -> Any: ...

class GenericIndexedCloudPersist(GenericIndexedCloud):
    def __init__(self, *args, **kwargs) -> None: ...
    def getPointPersistentPtr(self, index: int) -> CCVector3: ...

class GenericIndexedMesh(GenericMesh):
    def __init__(self, *args, **kwargs) -> None: ...
    def _getTriangle(self, triangleIndex: int) -> GenericTriangle: ...
    def getNextTriangleVertIndexes(self) -> VerticesIndexes: ...
    def getTriangleVertIndexes(self, triangleIndex: int) -> VerticesIndexes: ...
    def getTriangleVertices(self, triangleIndex: int, A: CCVector3, B: CCVector3, C: CCVector3) -> None: ...

class GenericMesh:
    def __init__(self, *args, **kwargs) -> None: ...
    def _getNextTriangle(self) -> GenericTriangle: ...
    def forEach(self, action) -> None: ...
    def getBoundingBox(self, bbMin: CCVector3, bbMax: CCVector3) -> None: ...
    def placeIteratorAtBeginning(self) -> None: ...
    def size(self) -> int: ...

class GenericOctree:
    def __init__(self, *args, **kwargs) -> None: ...

class GenericProgressCallback:
    def __init__(self, *args, **kwargs) -> None: ...
    def isCancelRequested(self) -> bool: ...
    def setInfo(self, arg0: str) -> None: ...
    def setMethodTitle(self, arg0: str) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def textCanBeEdited(self) -> bool: ...
    def update(self, arg0: float) -> None: ...

class GenericTriangle:
    def __init__(self, *args, **kwargs) -> None: ...
    def _getA(self) -> CCVector3: ...
    def _getB(self) -> CCVector3: ...
    def _getC(self) -> CCVector3: ...

class GeometricalAnalysisTools:
    ApproxLocalDensity: Any = ...
    Curvature: Any = ...
    DENSITY_2D: Any = ...
    DENSITY_3D: Any = ...
    DENSITY_KNN: Any = ...
    Density: Any = ...
    ErrorCode: Any = ...
    Feature: Any = ...
    GeomCharacteristic: Any = ...
    InvalidInput: Any = ...
    LocalDensity: Any = ...
    MomentOrder1: Any = ...
    NoError: Any = ...
    NotEnoughMemory: Any = ...
    NotEnoughPoints: Any = ...
    OctreeComputationFailed: Any = ...
    ProcessCancelledByUser: Any = ...
    ProcessFailed: Any = ...
    Roughness: Any = ...
    UnhandledCharacteristic: Any = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def ComputeCharactersitic(self, *args, **kwargs) -> Any: ...
    def ComputeCrossCovarianceMatrix(self, *args, **kwargs) -> Any: ...
    def ComputeGravityCenter(self, *args, **kwargs) -> Any: ...
    def ComputeLocalDensityApprox(self, *args, **kwargs) -> Any: ...
    def ComputeSphereFrom4(self, *args, **kwargs) -> Any: ...
    def ComputeWeightedCrossCovarianceMatrix(self, *args, **kwargs) -> Any: ...
    def ComputeWeightedGravityCenter(self, *args, **kwargs) -> Any: ...
    def DetectSphereRobust(self, *args, **kwargs) -> Any: ...
    def FlagDuplicatePoints(self, *args, **kwargs) -> Any: ...

class Grid3Dus:
    def __init__(self) -> None: ...
    def init(self, arg0: int, arg1: int, arg2: int, arg3: int, arg4: int) -> bool: ...
    def isInitialized(self) -> bool: ...
    def size(self) -> Tuple3ui: ...

class HornRegistrationTools(RegistrationTools):
    def __init__(self, *args, **kwargs) -> None: ...
    def ComputeRMS(self, *args, **kwargs) -> Any: ...
    def FindAbsoluteOrientation(self, *args, **kwargs) -> Any: ...

class ICPRegistrationTools(RegistrationTools):
    CONVERGENCE_TYPE: Any = ...
    ICP_APPLY_TRANSFO: Any = ...
    ICP_ERROR: Any = ...
    ICP_ERROR_CANCELED_BY_USER: Any = ...
    ICP_ERROR_DIST_COMPUTATION: Any = ...
    ICP_ERROR_INVALID_INPUT: Any = ...
    ICP_ERROR_NOT_ENOUGH_MEMORY: Any = ...
    ICP_ERROR_REGISTRATION_STEP: Any = ...
    ICP_NOTHING_TO_DO: Any = ...
    MAX_ERROR_CONVERGENCE: Any = ...
    MAX_ITER_CONVERGENCE: Any = ...
    Parameters: Any = ...
    RESULT_TYPE: Any = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def Register(self, *args, **kwargs) -> Any: ...

class KDTree:
    IndicesVector: Any = ...
    def __init__(self) -> None: ...
    def buildFromCloud(self, cloud: GenericIndexedCloud, progressCb: GenericProgressCallback = ...) -> bool: ...
    def findNearestNeighbour(self, queryPoint: sequence, maxDist: float) -> object: ...
    def findPointBelowDistance(self, queryPoint: sequence, maxDist: float) -> bool: ...
    def findPointsLyingToDistance(self, queryPoint: sequence, distance: float, tolerance: float, points: KDTree.IndicesVector) -> int: ...
    def getAssociatedCloud(self) -> GenericIndexedCloud: ...

class KMeanClass:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def maxValue(self) -> float: ...
    @maxValue.setter
    def maxValue(self, val: float) -> None: ...
    @property
    def mean(self) -> float: ...
    @mean.setter
    def mean(self, val: float) -> None: ...
    @property
    def minValue(self) -> float: ...
    @minValue.setter
    def minValue(self, val: float) -> None: ...

class CV_LOCAL_MODEL_TYPES:
    LS: Any = ...
    NO_MODEL: Any = ...
    QUADRIC: Any = ...
    TRI: Any = ...
    __entries: Any = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> Any: ...
    @property
    def __doc__(self) -> Any: ...
    @property
    def __members__(self) -> Any: ...

class LocalModel:
    def __init__(self, *args, **kwargs) -> None: ...
    def computeDistanceFromModelToPoint(self, P: CCVector3, nearestPoint: CCVector3 = ...) -> float: ...
    def getCenter(self) -> CCVector3: ...
    def getSquareSize(self) -> float: ...
    def getType(self) -> CV_LOCAL_MODEL_TYPES: ...

class ManualSegmentationTools:
    MeshCutterParams: Any = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def isPointInsidePoly(self, *args, **kwargs) -> Any: ...
    def segment(self, *args, **kwargs) -> Any: ...
    def segmentMesh(self, *args, **kwargs) -> Any: ...
    def segmentMeshWithAABox(self, *args, **kwargs) -> Any: ...
    def segmentMeshWithAAPlane(self, *args, **kwargs) -> Any: ...
    def segmentReferenceCloud(self, *args, **kwargs) -> Any: ...

class NormalDistribution(GenericDistribution):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, _mu: float, _sigma2: float) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def computeParameters(self, cloud: GenericCloud) -> bool: ...
    def computeRobustParameters(self, values: GenericDistribution.ScalarContainer, nSigma: float) -> bool: ...
    def getMu(self) -> float: ...
    def getParameters(self) -> Tuple[float,float]: ...
    def getSigma2(self) -> float: ...
    def setParameters(self, _mu: float, _sigma2: float) -> bool: ...

class NormalizedProgress:
    def __init__(self, callback: GenericProgressCallback, totalSteps: int, totalPercentage: int = ...) -> None: ...
    def oneStep(self) -> bool: ...
    def reset(self) -> None: ...
    def scale(self, totalSteps: int, totalPercentage: int = ..., updateCurrentProgress: bool = ...) -> None: ...
    def steps(self, arg0: int) -> bool: ...

class NumpyCloud(GenericIndexedCloud):
    def __init__(self, arg0: numpy.ndarray) -> None: ...

class PointCloud(GenericIndexedCloudPersist):
    def __init__(self) -> None: ...
    def addPoint(self, P: CCVector3) -> None: ...
    def reserve(self, newCapacity: int) -> bool: ...

class PointProjectionTools:
    IndexedCCVector2: Any = ...
    Transformation: Any = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def applyTransformation(self, *args, **kwargs) -> Any: ...
    def computeTriangulation(self, *args, **kwargs) -> Any: ...
    def developCloudOnCone(self, *args, **kwargs) -> Any: ...
    def developCloudOnCylinder(self, *args, **kwargs) -> Any: ...

class Polyline(ReferenceCloud):
    def __init__(self, associatedCloud: GenericIndexedCloudPersist) -> None: ...
    def clear(self, unusedParam: bool = ...) -> None: ...
    def isClosed(self) -> bool: ...
    def setClosed(self, state: bool) -> None: ...

class ReferenceCloud(GenericIndexedCloudPersist):
    def __init__(self, associatedCloud: GenericIndexedCloudPersist) -> None: ...
    def add(self, arg0: ReferenceCloud) -> bool: ...
    @overload
    def addPointIndex(self, arg0: int, arg1: int) -> None: ...
    @overload
    def addPointIndex(self, arg0: int) -> None: ...
    @overload
    def addPointIndex(*args, **kwargs) -> Any: ...
    def capacity(self) -> int: ...
    def clear(self, releaseMemory: bool = ...) -> None: ...
    def forwardIterator(self) -> None: ...
    def getAssociatedCloud(self) -> GenericIndexedCloudPersist: ...
    def getCurrentPointGlobalIndex(self) -> int: ...
    def getCurrentPointScalarValue(self) -> float: ...
    def getPointGlobalIndex(self, arg0: int) -> int: ...
    def invalidateBoundingBox(self) -> None: ...
    def removeCurrentPointGlobalIndex(self) -> None: ...
    def removePointGlobalIndex(self, arg0: int) -> None: ...
    def reserve(self, arg0: int) -> bool: ...
    def resize(self, arg0: int) -> bool: ...
    def setAssociatedCloud(self, arg0: GenericIndexedCloudPersist) -> None: ...
    def setCurrentPointScalarValue(self, arg0: float) -> None: ...
    def setPointIndex(self, arg0: int, arg1: int) -> None: ...
    def swap(self, arg0: int, arg1: int) -> None: ...

class ReferenceCloudContainer:
    __hash__: Any = ...
    __pybind11_module_local_v4_msvc__: Any = ...
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: ReferenceCloudContainer) -> None: ...
    @overload
    def __init__(self, arg0: Iterable) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def append(self, x) -> None: ...
    def clear(self) -> None: ...
    def count(self, x) -> int: ...
    @overload
    def extend(self, L: ReferenceCloudContainer) -> None: ...
    @overload
    def extend(self, L: Iterable) -> None: ...
    @overload
    def extend(*args, **kwargs) -> Any: ...
    def insert(self, i: int, x) -> None: ...
    def pop(*args, **kwargs) -> Any: ...
    def remove(self, x) -> None: ...
    def __bool__(self) -> bool: ...
    def __contains__(self, x) -> bool: ...
    @overload
    def __delitem__(self, arg0: int) -> None: ...
    @overload
    def __delitem__(self, arg0: slice) -> None: ...
    @overload
    def __delitem__(*args, **kwargs) -> Any: ...
    def __eq__(self, arg0: ReferenceCloudContainer) -> bool: ...
    @overload
    def __getitem__(self, s: slice) -> ReferenceCloudContainer: ...
    @overload
    def __getitem__(*args, **kwargs) -> Any: ...
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: ReferenceCloudContainer) -> bool: ...
    @overload
    def __setitem__(self, arg0: int, arg1) -> None: ...
    @overload
    def __setitem__(self, arg0: slice, arg1: ReferenceCloudContainer) -> None: ...
    @overload
    def __setitem__(*args, **kwargs) -> Any: ...

class RegistrationTools:
    SKIP_NONE: Any = ...
    SKIP_ROTATION: Any = ...
    SKIP_RXY: Any = ...
    SKIP_RXZ: Any = ...
    SKIP_RYZ: Any = ...
    SKIP_TRANSLATION: Any = ...
    SKIP_TX: Any = ...
    SKIP_TY: Any = ...
    SKIP_TZ: Any = ...
    TRANSFORMATION_FILTERS: Any = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def FilterTransformation(self, *args, **kwargs) -> Any: ...

class ScalarField(CCShareable):
    def __init__(self, name: str) -> None: ...
    def NaN(self, *args, **kwargs) -> Any: ...
    def ValidValue(self, *args, **kwargs) -> Any: ...
    def asArray(self) -> numpy.ndarray[numpy.float32]: ...
    def computeMeanAndVariance(self, mean: float, variance: float = ...) -> None: ...
    def computeMinAndMax(self) -> None: ...
    def fill(self, fillValue: float = ...) -> None: ...
    def flagValueAsInvalid(self, index: int) -> None: ...
    def getMax(self) -> float: ...
    def getMin(self) -> float: ...
    def getName(self) -> str: ...
    def reserveSafe(self, count: int) -> bool: ...
    def resizeSafe(self, count: int, initNewElements: bool = ..., valueForNewElements: float = ...) -> bool: ...
    def setName(self, arg0: str) -> None: ...
    def size(self) -> int: ...
    def __getitem__(self, arg0: int) -> float: ...
    def __setitem__(self, arg0: int, arg1: float) -> None: ...

class ScalarFieldTools:
    def __init__(self, *args, **kwargs) -> None: ...
    def SetScalarValueInverted(self, *args, **kwargs) -> Any: ...
    def SetScalarValueToNaN(self, *args, **kwargs) -> Any: ...
    def applyScalarFieldGaussianFilter(self, *args, **kwargs) -> Any: ...
    def computeKmeans(self, *args, **kwargs) -> Any: ...
    def computeMeanScalarValue(self, *args, **kwargs) -> Any: ...
    def computeMeanSquareScalarValue(self, *args, **kwargs) -> Any: ...
    def computeScalarFieldExtremas(self, *args, **kwargs) -> Any: ...
    def computeScalarFieldGradient(self, *args, **kwargs) -> Any: ...
    def computeScalarFieldHistogram(self, *args, **kwargs) -> Any: ...
    def countScalarFieldValidValues(self, *args, **kwargs) -> Any: ...
    def multiplyScalarFields(self, *args, **kwargs) -> Any: ...

class SimpleMesh(GenericIndexedMesh):
    def __init__(self, theVertices: GenericIndexedCloud, linkVerticesWithMesh: bool = ...) -> None: ...
    def addTriangle(self, i1: int, i2: int, i3: int) -> None: ...
    def capacity(self) -> int: ...
    def clear(self) -> None: ...
    def reserve(self, n: int) -> bool: ...
    def resize(self, n: int) -> bool: ...
    def vertices(self) -> GenericIndexedCloud: ...

class SimpleRefTriangle(GenericTriangle):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, _A: CCVector3, _B: CCVector3, _C: CCVector3) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    @property
    def A(self) -> cccorelib.CCVector3: ...
    @property
    def B(self) -> cccorelib.CCVector3: ...
    @property
    def C(self) -> cccorelib.CCVector3: ...

class SimpleTriangle(GenericTriangle):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: CCVector3, arg1: CCVector3, arg2: CCVector3) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    @property
    def A(self) -> cccorelib.CCVector3: ...
    @property
    def B(self) -> cccorelib.CCVector3: ...
    @property
    def C(self) -> cccorelib.CCVector3: ...

class StatisticalTestingTools:
    def __init__(self, *args, **kwargs) -> None: ...
    def computeAdaptativeChi2Dist(self, *args, **kwargs) -> Any: ...
    def computeChi2Fractile(self, *args, **kwargs) -> Any: ...
    def computeChi2Probability(self, *args, **kwargs) -> Any: ...
    def testCloudWithStatisticalModel(self, *args, **kwargs) -> Any: ...

class TRIANGULATION_TYPES:
    DELAUNAY_2D_AXIS_ALIGNED: Any = ...
    DELAUNAY_2D_BEST_LS_PLANE: Any = ...
    __entries: Any = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> Any: ...
    @property
    def __doc__(self) -> Any: ...
    @property
    def __members__(self) -> Any: ...

class TrueKdTree:
    BaseNode: Any = ...
    Leaf: Any = ...
    LeafVector: Any = ...
    Node: Any = ...
    def __init__(self, cloud: GenericIndexedCloudPersist) -> None: ...
    def associatedCloud(self) -> GenericIndexedCloudPersist: ...
    def build(self, maxError: float, errorMeasure: DistanceComputationTools.ERRPOR_MEASURES = ..., minPointCountPerCell: int = ..., maxPointCountPerCell: int = ..., progressCb: GenericProgressCallback = ...) -> bool: ...
    def clear(self) -> None: ...
    def getLeaves(self, leaves: TrueKdTree.LeafVector) -> bool: ...
    def getMaxErrorType(self) -> DistanceComputationTools.ERRPOR_MEASURES: ...
    @property
    def LEAF_TYPE(self) -> Any: ...
    @property
    def NODE_TYPE(self) -> Any: ...
    @property
    def X_DIM(self) -> Any: ...
    @property
    def Y_DIM(self) -> Any: ...
    @property
    def Z_DIM(self) -> Any: ...

class Tuple3i:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def __getitem__(self, arg0: int) -> int: ...
    @property
    def x(self) -> int: ...
    @x.setter
    def x(self, val: int) -> None: ...
    @property
    def y(self) -> int: ...
    @y.setter
    def y(self, val: int) -> None: ...
    @property
    def z(self) -> int: ...
    @z.setter
    def z(self, val: int) -> None: ...

class Tuple3s:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def __getitem__(self, arg0: int) -> int: ...
    @property
    def x(self) -> int: ...
    @x.setter
    def x(self, val: int) -> None: ...
    @property
    def y(self) -> int: ...
    @y.setter
    def y(self, val: int) -> None: ...
    @property
    def z(self) -> int: ...
    @z.setter
    def z(self, val: int) -> None: ...

class Tuple3ub:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def __getitem__(self, arg0: int) -> int: ...
    @property
    def x(self) -> int: ...
    @x.setter
    def x(self, val: int) -> None: ...
    @property
    def y(self) -> int: ...
    @y.setter
    def y(self, val: int) -> None: ...
    @property
    def z(self) -> int: ...
    @z.setter
    def z(self, val: int) -> None: ...

class Tuple3ui:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def __getitem__(self, arg0: int) -> int: ...
    @property
    def x(self) -> int: ...
    @x.setter
    def x(self, val: int) -> None: ...
    @property
    def y(self) -> int: ...
    @y.setter
    def y(self, val: int) -> None: ...
    @property
    def z(self) -> int: ...
    @z.setter
    def z(self, val: int) -> None: ...

class VerticesIndexes:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, _i1: int, _i2: int, _i3: int) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def __getitem__(self, arg0: int) -> int: ...
    @property
    def i1(self) -> int: ...
    @i1.setter
    def i1(self, val: int) -> None: ...
    @property
    def i2(self) -> int: ...
    @i2.setter
    def i2(self, val: int) -> None: ...
    @property
    def i3(self) -> int: ...
    @i3.setter
    def i3(self, val: int) -> None: ...

class WeibullDistribution(GenericDistribution):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, a: float, b: float, valueShift: float = ...) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def computeMode(self) -> float: ...
    def computeSkewness(self) -> float: ...
    def getOtherParameters(self, mu: float, sigma2: float) -> bool: ...
    def getParameters(self, a: float, b: float) -> bool: ...
    def getValueShift(self) -> float: ...
    def setParameters(self, a: float, b: float, valueshift: float = ...) -> bool: ...
    def setValueShift(self, vs: float) -> None: ...

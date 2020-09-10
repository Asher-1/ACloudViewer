//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#                    COPYRIGHT: CLOUDVIEWER  project                     #
//#                                                                        #
//##########################################################################

#include "PCLModules.h"

// CV_CORE_LIB
#include <CVConst.h>
#include <CVTools.h>

// PCL COMMON
#include <pcl/io/pcd_io.h>

// PCL FEATURES
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/impl/normal_3d_omp.hpp>

// PCL SEARCH
#include <pcl/kdtree/impl/kdtree_flann.hpp>

// PCL SEGMENTATION
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>

// PCL RECOGNITION
#include <pcl/recognition/impl/cg/geometric_consistency.hpp>

// PCL SURFACE
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/grid_projection.h>

// normal function
namespace PCLModules
{
	/**
	 * SEGMENTATION
	 */
	int GetRegionGrowing(
		const PointCloudT::ConstPtr cloud, 
		std::vector<pcl::PointIndices>& clusters,
		PointCloudRGB::Ptr cloud_segmented,
		int k, int min_cluster_size, int max_cluster_size,
		unsigned int neighbour_number, float smoothness_theta, float curvature) 
	{

		// 和triangulation中的同理，需要用到法向量，先用Normalesimation去计算
		CloudNormal::Ptr normals(new CloudNormal);
		ComputeNormals<PointT, NormalT>(cloud, normals, k, true);

		pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
		kdtree->setInputCloud(cloud);

		// 增长对象
		pcl::RegionGrowing<PointT, NormalT> regionGrowing;
		// 最少点
		regionGrowing.setMinClusterSize(min_cluster_size); // example default: 50
		// 最大点，一般希望无穷大
		regionGrowing.setMaxClusterSize(max_cluster_size);
		// 用设置好的kd树
		regionGrowing.setSearchMethod(kdtree);
		// 参考的领域点数，即多少个点决定一个平面，决定了容错率
		// 如果设置很小，检测到的平面也很小，很大的话，可能有的点很歪
		regionGrowing.setNumberOfNeighbours(neighbour_number); // example default: 30
		// 输入检测的点云
		regionGrowing.setInputCloud(cloud);
		// regionGrowing.setIndices(indicesPtr);
		// 输入点法线
		regionGrowing.setInputNormals(normals);
		// 设置弯曲阈值，决定了是否要继续探索
		// 假设每个点都是平稳弯曲的，夹角也很小，随着探索区域增大，变化一定会超过这两个阈值
		regionGrowing.setSmoothnessThreshold(smoothness_theta);
		regionGrowing.setCurvatureThreshold(curvature);

		regionGrowing.extract(clusters);
		CVLog::Print(QString("[Basic Region Growing] Number of clusters is equal to %1").arg(clusters.size()));
		if (clusters.size() != 0)
		{
			CVLog::Print(QString("[Basic Region Growing] First cluster has %1 points.").arg(clusters[0].indices.size()));
		}

		pcl::copyPointCloud(*regionGrowing.getColoredCloud(), *cloud_segmented);

		return 1;
	}

	int GetRegionGrowingRGB(
		const PointCloudRGB::ConstPtr cloud,
		std::vector<pcl::PointIndices>& clusters,
		PointCloudRGB::Ptr cloud_segmented,
		int min_cluster_size, float neighbors_distance,
		float point_color_diff, float region_color_diff)
	{
		// color-based region growing segmentation
		// kd-tree object for searches.
		pcl::search::KdTree<PointRGB>::Ptr kdtree(new pcl::search::KdTree<PointRGB>);
		kdtree->setInputCloud(cloud);

		// Color-based region growing clustering object.
		pcl::RegionGrowingRGB<PointRGB> regionGrowing;
		regionGrowing.setInputCloud(cloud);
		regionGrowing.setSearchMethod(kdtree);
		// Here, the minimum cluster size affects also the postprocessing step:
		// clusters smaller than this will be merged with their neighbors.
		regionGrowing.setMinClusterSize(min_cluster_size);
		// Set the distance threshold, to know which points will be considered neighbors.
		regionGrowing.setDistanceThreshold(neighbors_distance);
		// Color threshold for comparing the RGB color of two points.
		regionGrowing.setPointColorThreshold(point_color_diff);
		// Region color threshold for the postprocessing step: clusters with colors
		// within the threshold will be merged in one.
		regionGrowing.setRegionColorThreshold(region_color_diff);

		regionGrowing.extract(clusters);
		CVLog::Print(QString("[RGB Region Growing] Number of clusters is equal to %1").arg(clusters.size()));
		if (clusters.size() != 0)
		{
			CVLog::Print(QString("[RGB Region Growing] First cluster has %1 points.").arg(clusters[0].indices.size()));
		}

		pcl::copyPointCloud(*regionGrowing.getColoredCloud(), *cloud_segmented);

		return 1;
	}

	int	GetSACSegmentation(
			const PointCloudT::ConstPtr cloud, 
			pcl::PointIndices::Ptr inliers,
			pcl::ModelCoefficients::Ptr coefficients/* = nullptr*/,
			const int &methodType/* = pcl::SAC_RANSAC*/,
			const int &modelType/* = pcl::SACMODEL_PLANE*/,
			float distanceThreshold/* = 0.02*/,
			float probability/* = 0.95*/,
			int maxIterations/* = 100*/,
			float minRadiusLimits/* = -10000.0f*/,
			float maxRadiusLimits/* =  10000.0f*/,
			float normalDisWeight/* = 0.1f*/)
	{
		// Build the model
		switch (modelType)
		{
		case pcl::SACMODEL_CYLINDER:
		case pcl::SACMODEL_NORMAL_PLANE:
		case pcl::SACMODEL_NORMAL_PARALLEL_PLANE:
		case pcl::SACMODEL_CONE:
		case pcl::SACMODEL_NORMAL_SPHERE:
		{
			pcl::SACSegmentationFromNormals<PointT, NormalT> seg;
			seg.setOptimizeCoefficients(coefficients ? true : false);

			CloudNormal::Ptr cloudNormas(new CloudNormal);
			if (!ComputeNormals<PointT, NormalT>(cloud, cloudNormas, -1, false) ||
				cloudNormas->width * cloudNormas->height == 0)
			{
				return -1;
			}

			float normalWeight = normalDisWeight > 0 ? normalDisWeight : 0.1/*default*/;
			seg.setModelType(modelType);
			seg.setMethodType(methodType);
			seg.setNormalDistanceWeight(normalWeight);
			seg.setDistanceThreshold(distanceThreshold);
			seg.setRadiusLimits(minRadiusLimits, maxRadiusLimits);
			seg.setMaxIterations(maxIterations);
			seg.setProbability(probability);
			seg.setInputCloud(cloud);
			seg.setInputNormals(cloudNormas);

			if (coefficients)
				seg.segment(*inliers, *coefficients);
            else
            {
                pcl::ModelCoefficients coeff;
                seg.segment(*inliers, coeff);
            }
		}
		break;
		// If nothing else, try SACSegmentation
		default:
		{
			pcl::SACSegmentation<PointT> seg;
			seg.setOptimizeCoefficients(coefficients ? true : false);

			seg.setInputCloud(cloud);
			seg.setModelType(modelType);
			seg.setMethodType(methodType);
			seg.setDistanceThreshold(distanceThreshold);
			seg.setRadiusLimits(minRadiusLimits, maxRadiusLimits);
			seg.setMaxIterations(maxIterations);
			seg.setProbability(probability);

			if (coefficients)
				seg.segment(*inliers, *coefficients);
			else
            {
                pcl::ModelCoefficients coeff;
                seg.segment(*inliers, coeff);
            }
		}
		break;
		}

		return 1;
	}

	/**
	 * SURFACE
	 */
	int GridProjection(
		const PointCloudNormal::ConstPtr &cloudWithNormals,
		PCLMesh &outMesh,
		float resolution, int paddingSize,
		int maxSearchLevel)
	{
		// another kd-tree for reconstruction
		pcl::search::KdTree<PointNT>::Ptr kdtree(new pcl::search::KdTree<PointNT>);
		kdtree->setInputCloud(cloudWithNormals);

		// reconstruction
		pcl::GridProjection<PointNT> gp;
		gp.setInputCloud(cloudWithNormals);
		gp.setSearchMethod(kdtree);
		gp.setResolution(resolution);
		gp.setPaddingSize(paddingSize);
		gp.setMaxBinarySearchLevel(maxSearchLevel);
		gp.reconstruct(outMesh);
		return 1;
	}

	int GetPoissonReconstruction(
		const PointCloudNormal::ConstPtr &cloudWithNormals,
		PCLMesh &outMesh,
		int degree, int treeDepth, int isoDivideDepth,
		int solverDivideDepth, float scale, float samplesPerNode,
		bool useConfidence, bool useManifold, bool outputPolygons)
	{
		// another kd-tree for reconstruction
		pcl::search::KdTree<PointNT>::Ptr
			kdtree(new pcl::search::KdTree<PointNT>);
		kdtree->setInputCloud(cloudWithNormals);
		pcl::Poisson<PointNT> pn;
		pn.setConfidence(useConfidence); // normalize normals[false]
		pn.setDegree(degree); // degree[1,5], the bigger the more time needed
		pn.setDepth(treeDepth); // tree maximum depth, calculate 2^d x 2 ^d x 2^d
		pn.setIsoDivide(isoDivideDepth); // extract ISO depth
		pn.setManifold(useManifold); // add triangle gravity
		pn.setOutputPolygons(outputPolygons); // output polygon mesh
		pn.setSamplesPerNode(samplesPerNode); // minimum sample points number no noise[1.0-5.0], noise[15.-20.]
		pn.setScale(scale); // rate
		pn.setSolverDivide(solverDivideDepth); // Gauss-Seidel depth
												 // pn.setIndices();
		pn.setSearchMethod(kdtree);
		pn.setInputCloud(cloudWithNormals);
		pn.performReconstruction(outMesh);

		return 1;
	}

	int GetGreedyTriangulation(
		const PointCloudNormal::ConstPtr &cloudWithNormals,
		PCLMesh &outMesh,
		int trigulationSearchRadius,
		float weightingFactor,
		int maxNearestNeighbors,
		int maxSurfaceAngle,
		int minAngle,
		int maxAngle,
		bool normalConsistency)
	{
		// another kd-tree for reconstruction
		pcl::search::KdTree<PointNT>::Ptr kdtree(new pcl::search::KdTree<PointNT>);
		kdtree->setInputCloud(cloudWithNormals);
		pcl::GreedyProjectionTriangulation<PointNT> gp3;
		// options
		gp3.setSearchRadius(trigulationSearchRadius);
		gp3.setMu(weightingFactor);
		gp3.setMaximumNearestNeighbors(maxNearestNeighbors);
		gp3.setMaximumSurfaceAngle(CV_DEG_TO_RAD * maxSurfaceAngle);
		gp3.setMinimumAngle(CV_DEG_TO_RAD * minAngle);
		gp3.setMaximumAngle(CV_DEG_TO_RAD * maxAngle);
		gp3.setNormalConsistency(normalConsistency);
		gp3.setInputCloud(cloudWithNormals);
		gp3.setSearchMethod(kdtree);
		gp3.reconstruct(outMesh);

		return 1;
	}

	int GetProjection(
		const PointCloudT::ConstPtr &originCloud,
		PointCloudT::Ptr &projectedCloud,
		const pcl::ModelCoefficients::ConstPtr coefficients,
		const int &modelType/* = pcl::SACMODEL_PLANE*/)
	{
		// Create the filtering object
		pcl::ProjectInliers<PointT> projectInliers;
		// set object projection model
		projectInliers.setModelType((pcl::SacModel)modelType);
		// input point cloud
		projectInliers.setInputCloud(originCloud);
		// set mode coefficients
		projectInliers.setModelCoefficients(coefficients);
		// execute projection filter result
		projectInliers.filter(*projectedCloud);

		return 1;
	}

	int GetProjection(
		const PointCloudT::ConstPtr &originCloud,
		PointCloudT::Ptr &projectedCloud,
		float coefficientA/* = 0.0f*/,
		float coefficientB/* = 0.0f*/,
		float coefficientC/* = 1.0f*/,
		float coefficientD/* = 0.0f*/,
		const int &modelType/* = pcl::SACMODEL_PLANE*/)
	{
		// define model object -> With X = Y= 0, Z=1
		// we use a plane model, with ax + by + cz + d = 0, 
		// where default : a= b = d =0, and c=1, or said differently, the X-Y plane.
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
		coefficients->values.resize(4);
		coefficients->values[0] = coefficientA;
		coefficients->values[1] = coefficientB;
		coefficients->values[2] = coefficientC;
		coefficients->values[3] = coefficientD;
		
		return GetProjection(originCloud, projectedCloud, coefficients, modelType);
	}

	template int GetMarchingCubes<PointNT>(
		const PointCloudNormal::ConstPtr inCloud,
		const MarchingMethod &marchingMethod,
		PCLMesh &outMesh,
		float epsilon,
		float isoLevel,
		float gridResolution,
		float percentageExtendGrid);

}

// Template function instance
namespace PCLModules
{
	//INSTANTIATING TEMPLATED FUNCTIONS
	template int ComputeNormals<PointT, NormalT>(
		const PointCloudT::ConstPtr inCloud,
		CloudNormal::Ptr outcloud,
		const float radius, 
		const bool useKnn, // true if use knn, false if radius search
		bool normalConsistency,
		int maxThreadCount);
	template int ComputeNormals<PointRGB, NormalT>(
		const PointCloudRGB::ConstPtr inCloud,
		CloudNormal::Ptr outcloud,
		const float radius,
		const bool useKnn, // true if use knn, false if radius search
		bool normalConsistency,
		int maxThreadCount);
	template int ComputeNormals<PointRGBA, NormalT>(
		const PointCloudRGBA::ConstPtr inCloud,
		CloudNormal::Ptr outcloud,
		const float radius,
		const bool useKnn, // true if use knn, false if radius search
		bool normalConsistency,
		int maxThreadCount);
	template int ComputeNormals<PointT, PointNT>(
		const PointCloudT::ConstPtr inCloud,
		PointCloudNormal::Ptr outcloud,
		const float radius,
		const bool useKnn, // true if use knn, false if radius search
		bool normalConsistency,
		int maxThreadCount);
	template int ComputeNormals<PointRGB, PointNT>(
		const PointCloudRGB::ConstPtr inCloud,
		PointCloudNormal::Ptr outcloud,
		const float radius,
		const bool useKnn, // true if use knn, false if radius search
		bool normalConsistency,
		int maxThreadCount);
	template int ComputeNormals<PointRGBA, PointNT>(
		const PointCloudRGBA::ConstPtr inCloud,
		PointCloudNormal::Ptr outcloud,
		const float radius,
		const bool useKnn, // true if use knn, false if radius search
		bool normalConsistency,
		int maxThreadCount);

	template int RemoveOutliersStatistical<PCLCloud>(
		const PCLCloud::ConstPtr inCloud,
		PCLCloud::Ptr outCloud,
		int knn, double nSigma);

	template int GetConvexHullReconstruction<PointT>(
		const PointCloudT::ConstPtr inCloud,
		PCLMesh &outMesh,
		int dimension);
	template int GetConvexHullReconstruction<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		PCLMesh &outMesh,
		int dimension);
	template int GetConcaveHullReconstruction<PointT>(
		const PointCloudT::ConstPtr inCloud,
		PCLMesh &outMesh,
		int dimension/* = 3*/,
		float alpha/* = 0.05*/);
	template int GetConcaveHullReconstruction<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		PCLMesh &outMesh,
		int dimension/* = 3*/,
		float alpha/* = 0.05*/);

	template int CropHullFilter<PointT>(
		const PointCloudT::ConstPtr inCloud,
		const PointCloudT::ConstPtr boundingBox,
		PointCloudT::Ptr outCloud,
		int dimensions);
	template int CropHullFilter<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		const PointCloudT::ConstPtr boundingBox,
		PointCloudRGB::Ptr outCloud,
		int dimensions);

	template int NurbsSurfaceFitting<PointT>(
		const PointCloudT::ConstPtr inCloud,
		const NurbsParameters &nurbsParams,
		PCLMesh &outMesh,
		PointCloudRGB::Ptr outCurve);
	template int NurbsSurfaceFitting<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		const NurbsParameters &nurbsParams,
		PCLMesh &outMesh,
		PointCloudRGB::Ptr outCurve);

	template int BSplineCurveFitting3D<PointT>(
		const PointCloudT::ConstPtr inCloud,
		PointCloudRGB::Ptr outCurve,
		int order,
		int controlPointsNum,
		unsigned curveResolution,
		double smoothness,
		double rScale,
		bool closed);
	template int BSplineCurveFitting3D<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		PointCloudRGB::Ptr outCurve,
		int order,
		int controlPointsNum,
		unsigned curveResolution,
		double smoothness,
		double rScale,
		bool closed);
	template int BSplineCurveFitting2D<PointT>(
		const PointCloudT::ConstPtr inCloud,
		const CurveFittingMethod &fittingMethod,
		PointCloudRGB::Ptr outCurve,
		int order,
		int controlPointsNum,
		unsigned curveResolution,
		double smoothness,
		double rScale,
		bool closed);
	template int BSplineCurveFitting2D<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		const CurveFittingMethod &fittingMethod,
		PointCloudRGB::Ptr outCurve,
		int order,
		int controlPointsNum,
		unsigned curveResolution,
		double smoothness,
		double rScale,
		bool closed);

	template int EuclideanCluster<PointT>(
		const PointCloudT::ConstPtr inCloud,
		std::vector<pcl::PointIndices> &cluster_indices,
		float clusterTolerance/* = 0.02*/,
		int minClusterSize/* = 100*/,
		int maxClusterSize/* = 250000*/);
	template int EuclideanCluster<PointNT>(
		const PointCloudNormal::ConstPtr inCloud,
		std::vector<pcl::PointIndices> &cluster_indices,
		float clusterTolerance/* = 0.02*/,
		int minClusterSize/* = 100*/,
		int maxClusterSize/* = 250000*/);

	template int ProgressiveMpFilter<PointT>(
		const PointCloudT::ConstPtr inCloud,
		pcl::PointIndicesPtr groundIndices,
		int maxWindowSize,
		float slope,
		float initialDistance,
		float maxDistance);
	template int ProgressiveMpFilter<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		pcl::PointIndicesPtr groundIndices,
		int maxWindowSize,
		float slope,
		float initialDistance,
		float maxDistance);

	template int ConditionalRemovalFilter<PointNT>(
		const PointCloudNormal::ConstPtr inCloud,
		const ConditionParameters &params,
		PointCloudNormal::Ptr outCloud,
		bool keepOrganized);

	template int SmoothMls<PointT, PointNT>(
		const PointCloudT::ConstPtr &inCloud,
		const MLSParameters &params,
		PointCloudNormal::Ptr &outcloud
#ifdef LP_PCL_PATCH_ENABLED
		, pcl::PointIndicesPtr &used_ids
#endif
		);

	template int EstimateSIFT<pcl::PointXYZI, PointT>(
		const pcl::PointCloud<pcl::PointXYZI>::ConstPtr inCloud,
		PointCloudT::Ptr outcloud,
		int nr_octaves, float min_scale, 
		int nr_scales_per_octave, float min_contrast);
	template int EstimateSIFT<PointRGB, PointT>(
		const PointCloudRGB::ConstPtr inCloud,
		PointCloudT::Ptr outcloud,
		int nr_octaves, float min_scale, 
		int nr_scales_per_octave, float min_contrast);
	template int EstimateSIFT<PointRGBA, PointT>(
		const PointCloudRGBA::ConstPtr inCloud,
		PointCloudT::Ptr outcloud,
		int nr_octaves, float min_scale,
		int nr_scales_per_octave, float min_contrast);

	template int EstimateShot<PointT, NormalT, pcl::SHOT352>(
		const PointCloudT::ConstPtr inCloud,
		const PointCloudT::ConstPtr keyPoints,
		const CloudNormal::ConstPtr normals,
		pcl::PointCloud<pcl::SHOT352>::Ptr outDescriptors,
		float searchRadius,
		int maxThreadCount);
	template int EstimateShot<PointRGB, NormalT, pcl::SHOT352>(
		const PointCloudRGB::ConstPtr inCloud,
		const PointCloudRGB::ConstPtr keyPoints,
		const CloudNormal::ConstPtr normals,
		pcl::PointCloud<pcl::SHOT352>::Ptr outDescriptors,
		float searchRadius,
		int maxThreadCount);
	template int EstimateShot<PointRGBA, NormalT, pcl::SHOT352>(
		const PointCloudRGBA::ConstPtr inCloud,
		const PointCloudRGBA::ConstPtr keyPoints,
		const CloudNormal::ConstPtr normals,
		pcl::PointCloud<pcl::SHOT352>::Ptr outDescriptors,
		float searchRadius,
		int maxThreadCount);

	template int DONEstimation<PointT, PointNT, PointNT>(
		const PointCloudT::ConstPtr inCloud,
		const PointCloudNormal::ConstPtr normalsLargeScale,
		const PointCloudNormal::ConstPtr normalsSmallScale,
		PointCloudNormal::Ptr outCloud);
	template int DONEstimation<PointRGB, PointNT, PointNT>(
		const PointCloudRGB::ConstPtr inCloud,
		const PointCloudNormal::ConstPtr normalsLargeScale,
		const PointCloudNormal::ConstPtr normalsSmallScale,
		PointCloudNormal::Ptr outCloud);

	template int EstimateLocalReferenceFrame<PointT, NormalT, pcl::ReferenceFrame>(
		const PointCloudT::ConstPtr inCloud,
		const PointCloudT::ConstPtr keyPoints,
		const CloudNormal::ConstPtr normals,
		pcl::PointCloud<pcl::ReferenceFrame>::Ptr outRF,
		float searchRadius);
	template int EstimateLocalReferenceFrame<PointRGB, NormalT, pcl::ReferenceFrame>(
		const PointCloudRGB::ConstPtr inCloud,
		const PointCloudRGB::ConstPtr keyPoints,
		const CloudNormal::ConstPtr normals,
		pcl::PointCloud<pcl::ReferenceFrame>::Ptr outRF,
		float searchRadius);
	template int EstimateLocalReferenceFrame<PointRGBA, NormalT, pcl::ReferenceFrame>(
		const PointCloudRGBA::ConstPtr inCloud,
		const PointCloudRGBA::ConstPtr keyPoints,
		const CloudNormal::ConstPtr normals,
		pcl::PointCloud<pcl::ReferenceFrame>::Ptr outRF,
		float searchRadius);

	template int EstimateHough3DGrouping<PointT, PointT, pcl::ReferenceFrame, pcl::ReferenceFrame>(
		const PointCloudT::ConstPtr modelKeypoints,
		const PointCloudT::ConstPtr sceneKeypoints,
		const pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr modelRF,
		const pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr sceneRF,
		const pcl::CorrespondencesConstPtr modelSceneCorrs,
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &rotoTranslations,
		std::vector<pcl::Correspondences> &clusteredCorrs,
		float houghBinSize,
		float houghThreshold);
	template int EstimateHough3DGrouping<PointRGB, PointRGB, pcl::ReferenceFrame, pcl::ReferenceFrame>(
		const PointCloudRGB::ConstPtr modelKeypoints,
		const PointCloudRGB::ConstPtr sceneKeypoints,
		const pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr modelRF,
		const pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr sceneRF,
		const pcl::CorrespondencesConstPtr modelSceneCorrs,
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& rotoTranslations,
		std::vector<pcl::Correspondences>& clusteredCorrs,
		float houghBinSize,
		float houghThreshold);
	template int EstimateHough3DGrouping<PointRGBA, PointRGBA, pcl::ReferenceFrame, pcl::ReferenceFrame>(
		const PointCloudRGBA::ConstPtr modelKeypoints,
		const PointCloudRGBA::ConstPtr sceneKeypoints,
		const pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr modelRF,
		const pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr sceneRF,
		const pcl::CorrespondencesConstPtr modelSceneCorrs,
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& rotoTranslations,
		std::vector<pcl::Correspondences>& clusteredCorrs,
		float houghBinSize,
		float houghThreshold);

	template int EstimateGeometricConsistencyGrouping<PointT, PointT>(
		const PointCloudT::ConstPtr modelKeypoints,
		const PointCloudT::ConstPtr sceneKeypoints,
		const pcl::CorrespondencesConstPtr modelSceneCorrs,
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &rotoTranslations,
		std::vector<pcl::Correspondences> &clusteredCorrs,
		float gcSize, float gcThreshold);
	template int EstimateGeometricConsistencyGrouping<PointRGB, PointRGB>(
		const PointCloudRGB::ConstPtr modelKeypoints,
		const PointCloudRGB::ConstPtr sceneKeypoints,
		const pcl::CorrespondencesConstPtr modelSceneCorrs,
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& rotoTranslations,
		std::vector<pcl::Correspondences>& clusteredCorrs,
		float gcSize, float gcThreshold);
	template int EstimateGeometricConsistencyGrouping<PointRGBA, PointRGBA>(
		const PointCloudRGBA::ConstPtr modelKeypoints,
		const PointCloudRGBA::ConstPtr sceneKeypoints,
		const pcl::CorrespondencesConstPtr modelSceneCorrs,
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& rotoTranslations,
		std::vector<pcl::Correspondences>& clusteredCorrs,
		float gcSize, float gcThreshold);

	template int EstimateHarris3D<PointT>(
		const PointCloudT::ConstPtr inCloud,
		pcl::PointCloud<pcl::PointXYZI>::Ptr outcloud,
		float normalRadius, float searchRadius);
	template int EstimateHarris3D<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		pcl::PointCloud<pcl::PointXYZI>::Ptr outcloud,
		float normalRadius, float searchRadius);
	template int EstimateHarris3D<PointRGBA>(
		const PointCloudRGBA::ConstPtr inCloud,
		pcl::PointCloud<pcl::PointXYZI>::Ptr outcloud,
		float normalRadius, float searchRadius);

	template int GetUniformSampling<PointT>(
		const PointCloudT::ConstPtr inCloud,
		PointCloudT::Ptr outcloud,
		const float &searchRadius);
	template int GetUniformSampling<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		PointCloudRGB::Ptr outcloud,
		const float &searchRadius);
	template int GetUniformSampling<PointRGBA>(
		const PointCloudRGBA::ConstPtr inCloud,
		PointCloudRGBA::Ptr outcloud,
		const float &searchRadius);

	template int PassThroughFilter<PointT>(
		const PointCloudT::ConstPtr inCloud,
		PointCloudT::Ptr outcloud,
		const QString& filterFieldName,
		const float &limit_min,
		const float &limit_max);
	template int PassThroughFilter<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		PointCloudRGB::Ptr outcloud,
		const QString& filterFieldName,
		const float &limit_min,
		const float &limit_max);
	template int PassThroughFilter<PointRGBA>(
		const PointCloudRGBA::ConstPtr inCloud,
		PointCloudRGBA::Ptr outcloud,
		const QString& filterFieldName,
		const float &limit_min,
		const float &limit_max);

	template int VoxelGridFilter<PointT>(
		const PointCloudT::ConstPtr inCloud,
		PointCloudT::Ptr outcloud,
		const float	&leafSizeX,
		const float &leafSizeY,
		const float &leafSizeZ);
	template int VoxelGridFilter<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		PointCloudRGB::Ptr outcloud,
		const float	&leafSizeX,
		const float &leafSizeY,
		const float &leafSizeZ);
	template int VoxelGridFilter<PointRGBA>(
		const PointCloudRGBA::ConstPtr inCloud,
		PointCloudRGBA::Ptr outcloud,
		const float	&leafSizeX,
		const float &leafSizeY,
		const float &leafSizeZ);

	template int ExtractIndicesFilter<PointT>(
		const PointCloudT::ConstPtr inCloud,
		const pcl::PointIndices::ConstPtr inliers,
		PointCloudT::Ptr outcloud,
		PointCloudT::Ptr outcloud2);
	template int ExtractIndicesFilter<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		const pcl::PointIndices::ConstPtr inliers,
		PointCloudRGB::Ptr outcloud,
		PointCloudRGB::Ptr outcloud2);
	template int ExtractIndicesFilter<PointRGBA>(
		const PointCloudRGBA::ConstPtr inCloud,
		const pcl::PointIndices::ConstPtr inliers,
		PointCloudRGBA::Ptr outcloud,
		PointCloudRGBA::Ptr outcloud2);

	template int GetMinCutSegmentation<PointT>(
		const PointCloudT::ConstPtr inCloud,
		std::vector <pcl::PointIndices> &outClusters,
		PointCloudRGB::Ptr cloud_segmented,
		const PointT foregroundPoint,
		int neighboursNumber,
		float smoothSigma,
		float backWeightRadius,
		float foreWeight,
		const pcl::IndicesConstPtr indices);

	template int GetMinCutSegmentation<PointRGB>(
		const PointCloudRGB::ConstPtr inCloud,
		std::vector <pcl::PointIndices> &outClusters,
		PointCloudRGB::Ptr cloud_segmented,
		const PointRGB foregroundPoint,
		int neighboursNumber,
		float smoothSigma,
		float backWeightRadius,
		float foreWeight,
		const pcl::IndicesConstPtr indices);

	template int GetBoundaryCloud<PointT, NormalT>(
		const PointCloudT::ConstPtr inCloud,
		const CloudNormal::ConstPtr normals,
		PointCloudT::Ptr boundaryCloud,
		const float angleThreshold,
		const float radius,
		const bool useKnn);
	template int GetBoundaryCloud<PointT, PointNT>(
		const PointCloudT::ConstPtr inCloud,
		const PointCloudNormal::ConstPtr normals,
		PointCloudT::Ptr boundaryCloud,
		const float angleThreshold,
		const float radius,
		const bool useKnn);
	template int GetBoundaryCloud<PointRGBA, NormalT>(
		const PointCloudRGBA::ConstPtr inCloud,
		const CloudNormal::ConstPtr normals,
		PointCloudRGBA::Ptr boundaryCloud,
		const float angleThreshold,
		const float radius,
		const bool useKnn);
	template int GetBoundaryCloud<PointRGBA, PointNT>(
		const PointCloudRGBA::ConstPtr inCloud,
		const PointCloudNormal::ConstPtr normals,
		PointCloudRGBA::Ptr boundaryCloud,
		const float angleThreshold,
		const float radius,
		const bool useKnn);

	template bool ICPRegistration<PointT, PointT>(
		const PointCloudT::ConstPtr targetCloud,
		const PointCloudT::ConstPtr sourceCloud,
		PointCloudT::Ptr outRegistered,
		int ipcMaxIterations,
		float icpCorrDistance);
	template bool ICPRegistration<PointRGB, PointRGB>(
		const PointCloudRGB::ConstPtr targetCloud,
		const PointCloudRGB::ConstPtr sourceCloud,
		PointCloudRGB::Ptr outRegistered,
		int ipcMaxIterations,
		float icpCorrDistance);

	template int GetHypothesesVerification<PointT, PointT>(
		const PointCloudT::Ptr sceneCloud,
		std::vector<PointCloudT::ConstPtr> modelClouds,
		std::vector<bool> &hypothesesMask,
		float clusterReg,
		float inlierThreshold,
		float occlusionThreshold,
		float radiusClutter,
		float regularizer,
		float radiusNormals,
		bool detectClutter);
	template int GetHypothesesVerification<PointRGB, PointRGB>(
		const PointCloudRGB::Ptr sceneCloud,
		std::vector<PointCloudRGB::ConstPtr> modelClouds,
		std::vector<bool> &hypothesesMask,
		float clusterReg,
		float inlierThreshold,
		float occlusionThreshold,
		float radiusClutter,
		float regularizer,
		float radiusNormals,
		bool detectClutter);
}


// class
namespace PCLModules
{
	/* ############################## FeatureCloud Class ################################# */
	FeatureCloud::FeatureCloud() :
		m_searchMethod(new SearchMethod),
		m_normalRadius(0.02f),
		m_featureRadius(0.02f),
		m_maxThreadCount( QThread::idealThreadCount() )
	{}

	// Load and process the cloud in the given PCD file
	void FeatureCloud::loadInputCloud(const std::string &pcd_file)
	{
		m_xyz = PointCloudT::Ptr(new PointCloudT);
		pcl::io::loadPCDFile(pcd_file, *m_xyz);
		processInput();
	}

	// Compute the surface normals
	void FeatureCloud::computeSurfaceNormals()
	{
		m_normals = SurfaceNormals::Ptr(new SurfaceNormals);
		ComputeNormals<PointT, NormalT>(m_xyz, m_normals, m_normalRadius, false);
	}

	// Compute the local feature descriptors
	void FeatureCloud::computeLocalFeatures()
	{
		m_features = LocalFeatures::Ptr(new LocalFeatures);
		pcl::FPFHEstimationOMP<PointT, NormalT, pcl::FPFHSignature33> fpfh_est;
		fpfh_est.setNumberOfThreads(m_maxThreadCount);
		fpfh_est.setInputCloud(m_xyz);
		fpfh_est.setInputNormals(m_normals);
		fpfh_est.setSearchMethod(m_searchMethod);
		fpfh_est.setRadiusSearch(m_featureRadius);
		fpfh_est.compute(*m_features);
	}

	/* ############################## FeatureCloud Class ################################# */


	/* ############################## TemplateMatching Class ################################# */

	TemplateMatching::TemplateMatching() :
		m_minSampleDistance(0.05f),
		m_maxCorrespondenceDistance(0.01f*0.01f),
		m_nr_iterations(500)
	{
		// Initialize the parameters in the Sample Consensus Initial Alignment (SAC-IA) algorithm
		m_sac_ia.setMinSampleDistance(m_minSampleDistance);
		m_sac_ia.setMaxCorrespondenceDistance(m_maxCorrespondenceDistance);
		m_sac_ia.setMaximumIterations(m_nr_iterations);
	}

	// Set the given cloud as the target to which the templates will be aligned
	void TemplateMatching::setTargetCloud(FeatureCloud &target_cloud)
	{
		m_target = target_cloud;
		m_sac_ia.setInputTarget(target_cloud.getPointCloud());
		m_sac_ia.setTargetFeatures(target_cloud.getLocalFeatures());
	}

	// Align the given template cloud to the target specified by setTargetCloud ()
	void TemplateMatching::align(FeatureCloud &template_cloud, TemplateMatching::Result &result)
	{
		m_sac_ia.setInputSource(template_cloud.getPointCloud());
		m_sac_ia.setSourceFeatures(template_cloud.getLocalFeatures());

		pcl::PointCloud<PointT> registration_output;

		m_sac_ia.align(registration_output);
		result.fitness_score = (float)m_sac_ia.getFitnessScore(m_maxCorrespondenceDistance);
		result.final_transformation = m_sac_ia.getFinalTransformation();
	}

	// Align all of template clouds set by addTemplateCloud to the target specified by setTargetCloud ()
	void TemplateMatching::alignAll(std::vector< TemplateMatching::Result, Eigen::aligned_allocator<Result> > &results)
	{
		results.resize(m_templates.size());
		for (std::size_t i = 0; i < m_templates.size(); ++i)
		{
			align(m_templates[i], results[i]);
		}
	}

	// Align all of template clouds to the target cloud to find the one with best alignment score
	int TemplateMatching::findBestAlignment(TemplateMatching::Result &result)
	{
		// Align all of the templates to the target cloud
		std::vector< Result, Eigen::aligned_allocator<Result> > results;
		alignAll(results);

		// Find the template with the best (lowest) fitness score
		float lowest_score = std::numeric_limits<float>::infinity();
		int best_template = 0;
		for (std::size_t i = 0; i < results.size(); ++i)
		{
			const Result &r = results[i];
			if (r.fitness_score < lowest_score)
			{
				lowest_score = r.fitness_score;
				best_template = (int)i;
			}
		}

		// Output the best alignment
		result = results[best_template];
		return (best_template);
	}
	/* ############################## TemplateMatching Class ################################# */

}

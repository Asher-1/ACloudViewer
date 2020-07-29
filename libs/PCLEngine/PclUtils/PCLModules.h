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

#ifndef QPCL_PCLMODULES_HEADER
#define QPCL_PCLMODULES_HEADER

#ifdef _MSC_VER
#pragma warning(disable:4996)
#pragma warning(disable:4290)
#pragma warning(disable:4819)
#endif

// LOCAL
#include "../qPCL.h"
#include "PCLConv.h"
#include "PCLCloud.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVConst.h>

// PCL COMMON
#include <pcl/Vertices.h>
#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>

// PCL KEYPOINTS
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/uniform_sampling.h>

// PCL FILTERS
#include <pcl/filters/crop_hull.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

// PCL RECOGNITION
#include <pcl/recognition/hv/hv_go.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>

// PCL REGISTRATION
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_ransac.h>

// PCL SEARCH
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

// PCL SURFACE
#include <pcl/surface/mls.h>
#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>

#include <pcl/surface/on_nurbs/triangulation.h>
#include <pcl/surface/on_nurbs/fitting_curve_pdm.h>
#include <pcl/surface/on_nurbs/fitting_surface_tdm.h>

#include <pcl/surface/on_nurbs/fitting_curve_2d.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_pdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_tdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_sdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_apdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_asdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_atdm.h>

// PCL FEATURES
#include <pcl/features/don.h>
#include <pcl/features/board.h>
#include <pcl/features/boundary.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/normal_3d_omp.h>

// PCL SEGMENTATION
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/impl/extract_clusters.hpp>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/segmentation/progressive_morphological_filter.h>

// QT
#include <QThread>

// SYSTEM
#include <limits>
#include <vector>
#include <Eigen/Core>

// normal function
namespace PCLModules
{
	// for grid projection
	int QPCL_ENGINE_LIB_API GridProjection(
		const PointCloudNormal::ConstPtr &cloudWithNormals,
		PCLMesh &outMesh,
		float resolution = 0.2f, 
		int paddingSize = 2,
		int maxSearchLevel = 8
	);

	// for poisson reconstruction
	int QPCL_ENGINE_LIB_API GetPoissonReconstruction(
		const PointCloudNormal::ConstPtr &cloudWithNormals,
		PCLMesh &outMesh,
		int degree = 2, 
		int treeDepth = 8, 
		int isoDivideDepth = 8,
		int solverDivideDepth = 8, 
		float scale = 1.25f, 
		float samplesPerNode = 3.0f,
		bool useConfidence = true, 
		bool useManifold = true, 
		bool outputPolygons = false
	);

	// for greedy projection triangulation
	int QPCL_ENGINE_LIB_API GetGreedyTriangulation(
		const PointCloudNormal::ConstPtr &cloudWithNormals,
		PCLMesh &outMesh,
		int trigulationSearchRadius = 25,
		float weightingFactor = 2.5f,
		int maxNearestNeighbors = 100,
		int maxSurfaceAngle = 45,
		int minAngle = 10,
		int maxAngle = 120,
		bool normalConsistency = false
	);

	int QPCL_ENGINE_LIB_API GetProjection(
		const PointCloudT::ConstPtr &originCloud,
		PointCloudT::Ptr &projectedCloud,
		const pcl::ModelCoefficients::ConstPtr coefficients,
		const int &modelType = 0 /*pcl::SACMODEL_PLANE*/
	);

	// for pcl projection filter
	int QPCL_ENGINE_LIB_API GetProjection(
		const PointCloudT::ConstPtr &originCloud,
		PointCloudT::Ptr &projectedCloud,
		float coefficientA = 0.0f,
		float coefficientB = 0.0f,
		float coefficientC = 1.0f,
		float coefficientD = 0.0f,
		const int &modelType = 0 /*pcl::SACMODEL_PLANE*/
	);

	/**
	  * @brief Basic Region Growing
	  * @param cloud             原始输入点云
	  * @param k                 k近邻参数
	  * @param min_cluster_size  一个region最少点数量
	  * @param max_cluster_size  一个region最大点数量，通常我们希望无穷大，选一个足够大的值就够了
	  * @param neighbour_number  多少个点来决定一个平面
	  * @param smoothness_theta  夹角阈值
	  * @param curvature         曲率阈值
	  * @return
	  */
	int QPCL_ENGINE_LIB_API GetRegionGrowing(
			const PointCloudT::ConstPtr cloud, 
			std::vector<pcl::PointIndices>& clusters,
			PointCloudRGB::Ptr cloud_segmented,
			int k, int min_cluster_size, int max_cluster_size,
			unsigned int neighbour_number, float smoothness_theta, float curvature);

	/**
	 * @brief Color based Region Growing
	 * @param cloud                 输入RGB点云
	 * @param min_cluster_size      region的最少点数
	 * @param neighbors_distance    近邻检测阈值
	 * @param point_color_diff      两点RGB差别阈值检测
	 * @param region_color_diff     两域RGB差别阈值检测
	 * @return
	 */
	int QPCL_ENGINE_LIB_API GetRegionGrowingRGB(
		const PointCloudRGB::ConstPtr cloud,
		std::vector<pcl::PointIndices> &clusters,
		PointCloudRGB::Ptr cloud_segmented,
		int min_cluster_size, float neighbors_distance,
		float point_color_diff, float region_color_diff);

	int QPCL_ENGINE_LIB_API	GetSACSegmentation(
		const PointCloudT::ConstPtr cloud,
		pcl::PointIndices::Ptr inliers,
		pcl::ModelCoefficients::Ptr coefficients = nullptr,
		const int &methodType = 0/* = pcl::SAC_RANSAC*/,
		const int &modelType = 0/* = pcl::SACMODEL_PLANE*/,
        float distanceThreshold = 0.02f,
        float probability = 0.95f,
		int maxIterations = 100,
		float minRadiusLimits = -10000.0f,
		float maxRadiusLimits = 10000.0f,
		float normalDisWeight = 0.1f);
}

// surface reconstruction
namespace PCLModules
{
	enum MarchingMethod { HOPPE, RBF };
	template <typename PointInT>
	inline int GetMarchingCubes(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		const MarchingMethod &marchingMethod,
		PCLMesh &outMesh,
		float epsilon = 0.01f,
		float isoLevel = 0.0f,
		float gridResolution = 50,
		float percentageExtendGrid = 0.0f
	)
	{
		// Create a search tree, use KDTreee for non-organized data.
        typename pcl::search::Search<PointInT>::Ptr tree;
		if (inCloud->isOrganized())
		{
			tree.reset(new pcl::search::OrganizedNeighbor<PointInT>());
		}
		else
		{
			tree.reset(new pcl::search::KdTree<PointInT>());
		}
		tree->setInputCloud(inCloud);

        typename pcl::MarchingCubes<PointInT>::Ptr mc;
		switch (marchingMethod)
		{
		  case MarchingMethod::HOPPE:
		  {
			  mc.reset(new pcl::MarchingCubesHoppe<PointInT>());
			  break;
		  }
		  case MarchingMethod::RBF:
		  {
              typename pcl::MarchingCubesRBF<PointInT>::Ptr rbf(new pcl::MarchingCubesRBF<PointInT>());
			  rbf->setOffSurfaceDisplacement (epsilon);
			  mc = rbf;
			  break;
		  }
		  default:
			  return -1;
		}

		mc->setIsoLevel(isoLevel);
		mc->setGridResolution(gridResolution, gridResolution, gridResolution);
		mc->setPercentageExtendGrid(percentageExtendGrid);
		mc->setInputCloud(inCloud);
		mc->reconstruct(outMesh);
		return 1;
	}
}

// template function
namespace PCLModules
{
	template <typename PointInOutT>
	inline int SwapAxis(
		const typename pcl::PointCloud<PointInOutT>::ConstPtr inCloud,
		typename pcl::PointCloud<PointInOutT>::Ptr outcloud,
		const std::string &flag
	)
	{
        pcl::copyPointCloud<PointInOutT, PointInOutT>(*inCloud, *outcloud);
        typename pcl::PointCloud<PointInOutT>::iterator it;
		for (it = outcloud->begin(); it != outcloud->end();)
		{
			float x = it->x;
			float y = it->y;
			float z = it->z;

			if ("zxy" == flag)
			{
				it->x = z;
				it->y = x;
				it->z = y;
			}
			else if ("zyx" == flag)
			{
				it->x = z;
				it->z = x;
			}
			else if ("xzy" == flag)
			{
				it->y = z;
				it->z = y;
			}
			else if ("yxz" == flag)
			{
				it->x = y;
				it->y = x;
			}
			else if ("yzx" == flag)
			{
				it->x = y;
				it->y = z;
				it->z = x;
			}
			++it;
		}
		return 1;
	}

	template <typename PointInT>
	inline int RemoveNaN(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		typename pcl::PointCloud<PointInT>::Ptr outcloud,
        std::vector<int> &index
	)
	{
		pcl::removeNaNFromPointCloud(*inCloud, *outcloud, index);
		return 1;
	}

	struct QPCL_ENGINE_LIB_API NurbsParameters
	{
		NurbsParameters()
			: order_(3)
			, twoDim_(true)
			, iterations_(10)
			, refinements_(4)
			, curveResolution_(4)
			, meshResolution_(128)
		{
		}

		int order_;
		bool twoDim_;
		int iterations_;
		int refinements_;
		unsigned curveResolution_;
        unsigned meshResolution_;
		pcl::on_nurbs::FittingSurface::Parameter fittingParams_;
			
	};
	template <typename PointInT>
	inline int NurbsSurfaceFitting(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		const NurbsParameters &nurbsParams,
		PCLMesh &outMesh,
		PointCloudRGB::Ptr outCurve = nullptr
	)
	{
		// step 1. convert point cloud to nurbs data vector3d
		pcl::on_nurbs::NurbsDataSurface data;
		for (unsigned i = 0; i < inCloud->size(); i++)
		{
			const PointInT &p = inCloud->at(i);
			if (!pcl_isnan(p.x) && !pcl_isnan(p.y) && !pcl_isnan(p.z))
				data.interior.push_back(Eigen::Vector3d(p.x, p.y, p.z));
		}

		// step 2. initialize and get first inital surface reconstruction
		ON_NurbsSurface nurbs = pcl::on_nurbs::FittingSurface::initNurbsPCABoundingBox(nurbsParams.order_, &data);
		pcl::on_nurbs::FittingSurface fit(&data, nurbs);
		//  fit.setQuiet (false); // enable/disable debug output

		// step 3. surface refinement
		for (int i = 0; i < nurbsParams.refinements_; i++)
		{
			fit.refine(0);
			if (nurbsParams.twoDim_) fit.refine(1);
			fit.assemble(nurbsParams.fittingParams_);
			fit.solve();
		}

		// step 4. surface fitting with final refinement level
		for (int i = 0; i < nurbsParams.iterations_; i++)
		{
			fit.assemble(nurbsParams.fittingParams_);
			fit.solve();
		}

		// step 5. initialization (circular)
		if (outCurve)
		{
			pcl::on_nurbs::FittingCurve2dAPDM::FitParameter curve_params;
			{
				curve_params.addCPsAccuracy = 5e-2;
				curve_params.addCPsIteration = 3;
				curve_params.maxCPs = 200;
				curve_params.accuracy = 1e-3;
				curve_params.iterations = 100;

				curve_params.param.closest_point_resolution = 0;
				curve_params.param.closest_point_weight = 1.0;
				curve_params.param.closest_point_sigma2 = 0.1;
				curve_params.param.interior_sigma2 = 0.00001;
				curve_params.param.smooth_concavity = 1.0;
				curve_params.param.smoothness = 1.0;
			}

			CVLog::Print("[PCLModules::NurbsSurfaceReconstruction] Start curve fitting ...");
			pcl::on_nurbs::NurbsDataCurve2d curve_data;
			curve_data.interior = data.interior_param;
			curve_data.interior_weight_function.push_back(true);
			ON_NurbsCurve curve_nurbs = pcl::on_nurbs::FittingCurve2dAPDM::initNurbsCurve2D(nurbsParams.order_, curve_data.interior);
			// curve fitting
			pcl::on_nurbs::FittingCurve2dASDM curve_fit(&curve_data, curve_nurbs);
			// curve_fit.setQuiet (false); // enable/disable debug output
			curve_fit.fitting(curve_params);

			pcl::on_nurbs::Triangulation::convertCurve2PointCloud(curve_fit.m_nurbs, fit.m_nurbs, outCurve, nurbsParams.curveResolution_);
			CVLog::Print("[PCLModules::NurbsSurfaceReconstruction] Triangulate trimmed surface ...");
			pcl::on_nurbs::Triangulation::convertTrimmedSurface2PolygonMesh(
				fit.m_nurbs, curve_fit.m_nurbs, outMesh, nurbsParams.meshResolution_);
		}
		else
		{
			pcl::on_nurbs::Triangulation::convertSurface2PolygonMesh(fit.m_nurbs, outMesh, nurbsParams.meshResolution_);
		}

		CVLog::Print(QString("[PCLModules::NurbsSurfaceReconstruction] Refines: %1, Iterations: %2")
			.arg(nurbsParams.refinements_).arg(nurbsParams.iterations_));
		return 1;
	}

    enum CurveFittingMethod { PD, SD, APD, TD, ASD };
    template <typename PointInT>
    inline int BSplineCurveFitting2D(
        const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
        const CurveFittingMethod &fittingMethod,
        PointCloudRGB::Ptr outCurve,
        int order = 3,
        int controlPointsNum = 10,
        unsigned curveResolution = 8,
        double smoothness = 0.000001,
        double rScale = 1.0,
        bool closed = false
        )
    {
        // initialize curve
        ON_NurbsCurve curve;
        // curve fitting
        ON_NurbsCurve nurbs;

        // convert to NURBS data 2D structure
        pcl::on_nurbs::NurbsDataCurve2d data;
        for (unsigned i = 0; i < inCloud->size(); i++)
        {
            const PointInT &p = inCloud->at(i);
            if (!pcl_isnan(p.x) && !pcl_isnan(p.y))
                data.interior.push_back(Eigen::Vector2d(p.x, p.y));
        }

        if (closed)
        {
            curve = pcl::on_nurbs::FittingCurve2dSDM::initNurbsCurve2D(order, data.interior, controlPointsNum);
            switch (fittingMethod)
            {
            case CurveFittingMethod::PD:
            {
                pcl::on_nurbs::FittingCurve2dPDM::Parameter curve_params;
                curve_params.smoothness = smoothness;
                curve_params.rScale = rScale;

                pcl::on_nurbs::FittingCurve2dPDM fit(&data, curve);
                fit.assemble(curve_params);
                fit.solve();
                nurbs = fit.m_nurbs;
                break;
            }
            case CurveFittingMethod::TD:
            {
                pcl::on_nurbs::FittingCurve2dTDM::Parameter curve_params;
                curve_params.smoothness = smoothness;
                curve_params.rScale = rScale;

                pcl::on_nurbs::FittingCurve2dTDM fit(&data, curve);
                fit.assemble(curve_params);
                fit.solve();
                nurbs = fit.m_nurbs;
                break;
            }
            case CurveFittingMethod::SD:
            {
                pcl::on_nurbs::FittingCurve2dSDM::Parameter curve_params;
                curve_params.smoothness = smoothness;
                curve_params.rScale = rScale;

                pcl::on_nurbs::FittingCurve2dSDM fit(&data, curve);
                fit.assemble(curve_params);
                fit.solve();
                nurbs = fit.m_nurbs;
                break;
            }
            case CurveFittingMethod::APD:
            {
                pcl::on_nurbs::FittingCurve2dAPDM::Parameter curve_params;
                curve_params.smoothness = smoothness;
                curve_params.rScale = rScale;

                pcl::on_nurbs::FittingCurve2dAPDM fit(&data, curve);
                fit.assemble(curve_params);
                fit.solve();
                nurbs = fit.m_nurbs;
                break;
            }
            case CurveFittingMethod::ASD:
            {
                pcl::on_nurbs::FittingCurve2dASDM::Parameter curve_params;
                curve_params.smoothness = smoothness;
                curve_params.rScale = rScale;

                pcl::on_nurbs::FittingCurve2dASDM fit(&data, curve);
                fit.assemble(curve_params);
                fit.solve();
                nurbs = fit.m_nurbs;
                break;
            }
            default:
                break;

            }
        }
        else
        {
            curve = pcl::on_nurbs::FittingCurve2d::initNurbsPCA(order, &data, controlPointsNum);
            pcl::on_nurbs::FittingCurve2d::Parameter curve_params;
            curve_params.smoothness = smoothness;
            curve_params.rScale = rScale;
            pcl::on_nurbs::FittingCurve2d fit(&data, curve);

            fit.assemble(curve_params);
            fit.solve();

            //for (int i = 0; i < 3; i++)
            //{
            //	fit.refine(0);
            //	fit.refine(1);
            //	fit.assemble(curve_params);
            //	fit.solve();
            //}

            //for (int i = 0; i < 3; i++)
            //{
            //	fit.assemble(curve_params);
            //	fit.solve();
            //}

            nurbs = fit.m_nurbs;
        }

        pcl::on_nurbs::Triangulation::convertCurve2PointCloud(nurbs, outCurve, curveResolution);
        return 1;
    }

	
	template <typename PointInT>
	inline int BSplineCurveFitting3D(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		PointCloudRGB::Ptr outCurve,
		int order = 3,
		int controlPointsNum = 10,
		unsigned curveResolution = 8,
		double smoothness = 0.000001,
		double rScale = 1.0,
		bool closed = false
	)
	{
		if (closed)
		{
			// convert to NURBS data 3D structure
			pcl::on_nurbs::NurbsDataCurve data;
			for (unsigned i = 0; i < inCloud->size(); i++)
			{
				const PointInT &p = inCloud->at(i);
				if (!pcl_isnan(p.x) && !pcl_isnan(p.y) && !pcl_isnan(p.z))
				{
					data.interior.push_back(Eigen::Vector3d(p.x, p.y, p.z));
				}
			}

			ON_NurbsCurve curve = pcl::on_nurbs::FittingCurve::initNurbsCurvePCA(order, data.interior, controlPointsNum);
			pcl::on_nurbs::FittingCurve::Parameter curve_params;
			curve_params.smoothness = smoothness;

			pcl::on_nurbs::FittingCurve fit(&data, curve);
			fit.assemble(curve_params);
			fit.solve();
			//fit.m_nurbs.Trim(ON_Interval(
			//	fit.m_nurbs.m_knot[0],
			//	fit.m_nurbs.m_knot[fit.m_nurbs.KnotCount() - fit.m_nurbs.Order()]
			//));
			pcl::on_nurbs::Triangulation::convertCurve2PointCloud(fit.m_nurbs, outCurve, curveResolution);
		}
		else
		{
			// step 1. xoy plane 2d curve fitting
			PointCloudRGB::Ptr xoyCurve(new PointCloudRGB);
			PCLModules::CurveFittingMethod fittingMethod = PCLModules::CurveFittingMethod::PD;
			if (!PCLModules::BSplineCurveFitting2D<PointInT>(
				inCloud, fittingMethod, xoyCurve, order, controlPointsNum,
				curveResolution, smoothness, rScale, closed) || xoyCurve->size() == 0)
			{
				return -1;
			}
			
			PointInT minPt;
			PointInT maxPt;
			pcl::getMinMax3D<PointInT>(*inCloud, minPt, maxPt);
			double dis_x = maxPt.x - minPt.x;
			double dis_y = maxPt.y - minPt.y;

			PointCloudRGB::Ptr xozCurve(new PointCloudRGB);
			PointCloudRGB::Ptr zoyCurve(new PointCloudRGB);
			if (dis_x > dis_y)
			{
				// xoz plane 2d curve fitting
                typename pcl::PointCloud<PointInT>::Ptr xozCloud(new pcl::PointCloud<PointInT>);
                SwapAxis<PointInT>(inCloud, xozCloud, "xzy");
                if (!BSplineCurveFitting2D<PointInT>(
					xozCloud, fittingMethod, xozCurve, order, controlPointsNum,
					curveResolution, smoothness, rScale, closed) || xozCurve->size() == 0)
				{
					return -1;
				}
				if (xoyCurve->size() != xozCurve->size())
				{
					return -53;
				}

			}
			else
			{
				// zoy plane 2d curve fitting
                typename pcl::PointCloud<PointInT>::Ptr zoyCloud(new pcl::PointCloud<PointInT>);
                SwapAxis<PointInT>(inCloud, zoyCloud, "zyx");
                if (!BSplineCurveFitting2D<PointInT>(
					zoyCloud, fittingMethod, zoyCurve, order, controlPointsNum,
					curveResolution, smoothness, rScale, closed) || zoyCurve->size() == 0)
				{
					return -1;
				}
				if (xoyCurve->size() != zoyCurve->size())
				{
					return -53;
				}
			}

			// out result
			outCurve->resize(xoyCurve->size());
			for (unsigned i = 0; i < xoyCurve->size(); ++i)
			{
				const PointRGB &xoy_p = xoyCurve->at(i);
				if (!pcl_isnan(xoy_p.x) && !pcl_isnan(xoy_p.y))
				{
					outCurve->points[i].x = xoy_p.x;
					outCurve->points[i].y = xoy_p.y;
					if (dis_x > dis_y)
					{
						const PointRGB &xoz_p = xozCurve->at(i);
						outCurve->points[i].z = xoz_p.y;
					}
					else
					{
						const PointRGB &zoy_p = zoyCurve->at(i);
						outCurve->points[i].z = zoy_p.x;
					}
				}
			}
		}
		return 1;
	}

	//! Extract SIFT keypoints
	/** if only the point cloud is given PCL default parameters are used (that are not really good, so please give parameters)
		\note Different types can be passed as input for this function:
			- PointXYZI
			- PointNormal
			- PointXYZRGB
		\note If a PointType with a scale field is passed as output type, scales will be returned together with the return cloud
	**/
	template <typename PointInT, typename PointOutT>
	inline int EstimateSIFT(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		typename pcl::PointCloud<PointOutT>::Ptr outcloud,
		int nr_octaves = 0, float min_scale = 0,
		int nr_scales_per_octave = 0, float min_contrast = 0)
	{
        typename pcl::SIFTKeypoint< PointInT, PointOutT > sift_detector;
        typename pcl::search::KdTree<PointInT>::Ptr tree(new pcl::search::KdTree<PointInT>());
		sift_detector.setSearchMethod(tree);

		if (nr_octaves != 0 && min_scale != 0 && nr_scales_per_octave != 0)
		{
			sift_detector.setScales(min_scale, nr_octaves, nr_scales_per_octave);
		}

		// DGM the min_contrast must be positive
		sift_detector.setMinimumContrast(min_contrast > 0 ? min_contrast : 0);

		sift_detector.setInputCloud(inCloud);
		sift_detector.compute(*outcloud);
		return 1;
	}

	template <typename PointInT, typename NormalType, typename DescriptorType = pcl::SHOT352>
	inline int EstimateShot(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		const typename pcl::PointCloud<PointInT>::ConstPtr keyPoints,
		const typename pcl::PointCloud<NormalType>::ConstPtr normals,
		typename pcl::PointCloud<DescriptorType>::Ptr outDescriptors,
		float searchRadius = 0.03f,
		int maxThreadCount = QThread::idealThreadCount())
	{
		pcl::SHOTEstimationOMP< PointInT, NormalType, DescriptorType > shot_detector;
		shot_detector.setRadiusSearch(searchRadius);
		shot_detector.setNumberOfThreads(maxThreadCount);

		shot_detector.setInputCloud(keyPoints);
		shot_detector.setSearchSurface(inCloud);
		shot_detector.setInputNormals(normals);
		shot_detector.compute(*outDescriptors);
		return 1;
	}

	template <typename PointInT>
	inline double ComputeCloudResolution(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud)
	{
		int nres;
		double res = 0.0;
		std::vector<int> indices(2);
		std::vector<float> sqr_distances(2);

		// Create a search tree, use KDTreee for non-organized data.
        typename pcl::search::Search<PointInT>::Ptr tree;
		if (inCloud->isOrganized())
		{
			tree.reset(new pcl::search::OrganizedNeighbor<PointInT>());
		}
		else
		{
			tree.reset(new pcl::search::KdTree<PointInT>());
		}
		tree->setInputCloud(inCloud);

		int size_cloud = static_cast<int>(inCloud->size());

		std::vector<float> nn_dis(size_cloud);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
		for (int i = 0; i < size_cloud; ++i)
		{
			if (!pcl_isfinite((*inCloud)[i].x))
			{
				continue;
			}

			//Considering the second neighbor since the first is the point itself.
			nres = tree->nearestKSearch(i, 2, indices, sqr_distances);
			if (nres == 2)
			{
				nn_dis[i] = std::sqrt(sqr_distances[1]);
				//res += sqr_distances[1];
				//++n_points;
			}
			else
			{
				CVLog::WarningDebug(
					"[ComputeCloudResolution] Found a point without neighbors.");
				nn_dis[i] = 0.0f;
			}
		}

		//if (n_points != 0)
		//{
		//	//res /= n_points;
		//	res = sqrt(res / n_points);
		//}

		res = std::accumulate(std::begin(nn_dis), std::end(nn_dis), 0.0f) / size_cloud;
		return res;
	}

	template <typename PointOutT>
	inline int RemoveOutliersStatistical(
		const typename PointOutT::ConstPtr inCloud,
		typename PointOutT::Ptr outCloud,
		int knn, double nSigma)
	{
		pcl::StatisticalOutlierRemoval<PointOutT> remover;
		remover.setInputCloud(inCloud);
		remover.setMeanK(knn);
		remover.setStddevMulThresh(nSigma);
		remover.filter(*outCloud);
		return 1;
	}

	template <typename PointInT>
	inline int EuclideanCluster(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		std::vector<pcl::PointIndices> &cluster_indices,
        float clusterTolerance = 0.02f,
		int minClusterSize = 100,
		int maxClusterSize = 250000)
	{
		// Creating the KdTree object for the search method of the extraction
        typename pcl::search::KdTree<PointInT>::Ptr tree(new pcl::search::KdTree<PointInT>);
		tree->setInputCloud(inCloud); // 创建点云索引向量，用于存储实际的点云信息

		pcl::EuclideanClusterExtraction<PointInT> ece;
		ece.setClusterTolerance(clusterTolerance);	// 设置近邻搜索的搜索半径为2cm
		ece.setMinClusterSize(minClusterSize);		// 设置一个聚类需要的最少点数目为100
		ece.setMaxClusterSize(maxClusterSize);		// 设置一个聚类需要的最大点数目为25000
		ece.setSearchMethod(tree);					// 设置点云的搜索机制
		ece.setInputCloud(inCloud);
		ece.extract(cluster_indices);				// 从点云中提取聚类，并将点云索引保存在cluster_indices中

		return 1;
	}

	template <typename PointInT>
	inline int ProgressiveMpFilter(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		pcl::PointIndicesPtr groundIndices,
		int maxWindowSize = 20,
		float slope = 1.0f,
		float initialDistance = 0.5f,
		float maxDistance = 3.0f
	)
	{
		// Create the filtering object
		pcl::ProgressiveMorphologicalFilter<PointInT> pmf;
		pmf.setInputCloud(inCloud);
		pmf.setMaxWindowSize(maxWindowSize);
		pmf.setSlope(slope);
		pmf.setInitialDistance(initialDistance);
		pmf.setMaxDistance(maxDistance);
		pmf.extract(groundIndices->indices);

		return 1;
	}

	template <typename PointInT, typename NormalType, typename PointOutT>
	inline int DONEstimation(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		const typename pcl::PointCloud<NormalType>::ConstPtr normalsLargeScale,
		const typename pcl::PointCloud<NormalType>::ConstPtr normalsSmallScale,
		typename pcl::PointCloud<PointOutT>::Ptr outCloud)
	{
		// Create DoN operator
		pcl::DifferenceOfNormalsEstimation<PointInT, NormalType, PointOutT> don;
		don.setInputCloud(inCloud);
		don.setNormalScaleLarge(normalsLargeScale);
		don.setNormalScaleSmall(normalsSmallScale);

		if (!don.initCompute())
		{
			return -1;
		}

		// Compute DoN
		don.computeFeature(*outCloud);
		return 1;
	}

	template <typename PointInT, typename NormalType, typename RFType = pcl::ReferenceFrame>
	inline int EstimateLocalReferenceFrame(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		const typename pcl::PointCloud<PointInT>::ConstPtr keyPoints,
		const typename pcl::PointCloud<NormalType>::ConstPtr normals,
		typename pcl::PointCloud<RFType>::Ptr outRF,
		float searchRadius = 0.015f)
	{
		pcl::BOARDLocalReferenceFrameEstimation< PointInT, NormalType, RFType > rf_detector;
		rf_detector.setRadiusSearch(searchRadius);
		rf_detector.setFindHoles(true);
		rf_detector.setInputCloud(keyPoints);
		rf_detector.setInputNormals(normals);
		rf_detector.setSearchSurface(inCloud);
		rf_detector.compute(*outRF);
		return 1;
	}

	template <typename PointModelT, typename PointSceneT, typename PointModelRfT = pcl::ReferenceFrame, typename PointSceneRfT = pcl::ReferenceFrame>
	inline int EstimateHough3DGrouping(
		const typename pcl::PointCloud<PointModelT>::ConstPtr modelKeypoints,
		const typename pcl::PointCloud<PointSceneT>::ConstPtr sceneKeypoints,
		const typename pcl::PointCloud<PointModelRfT>::ConstPtr modelRF,
		const typename pcl::PointCloud<PointSceneRfT>::ConstPtr sceneRF,
		const typename pcl::CorrespondencesConstPtr modelSceneCorrs,
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& rotoTranslations,
		std::vector<pcl::Correspondences>& clusteredCorrs,
		float houghBinSize = 0.01f,
		float houghThreshold = 5.0f)
	{
		pcl::Hough3DGrouping<PointModelT, PointSceneT, PointModelRfT, PointSceneRfT> clusterer;
		clusterer.setHoughBinSize(houghBinSize); // Hough空间的采样间隔
		clusterer.setHoughThreshold(houghThreshold); // 在Hough空间确定是否有实例存在的最少票数阈值
		clusterer.setUseInterpolation(true); // 设置是否对投票在Hough空间进行插值计算
		clusterer.setUseDistanceWeight(false); // 设置在投票时是否将对应点之间的距离作为权重参与计算

		clusterer.setInputCloud(modelKeypoints);
		clusterer.setInputRf(modelRF);
		clusterer.setSceneCloud(sceneKeypoints);
		clusterer.setSceneRf(sceneRF);
		clusterer.setModelSceneCorrespondences(modelSceneCorrs);

		bool flag = clusterer.recognize(rotoTranslations, clusteredCorrs);

		return flag ? 1 : -1;
	}


	template <typename PointModelT, typename PointSceneT>
	inline int EstimateGeometricConsistencyGrouping(
		const typename pcl::PointCloud<PointModelT>::ConstPtr modelKeypoints,
		const typename pcl::PointCloud<PointSceneT>::ConstPtr sceneKeypoints,
		const typename pcl::CorrespondencesConstPtr modelSceneCorrs,
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& rotoTranslations,
		std::vector<pcl::Correspondences>& clusteredCorrs,
		float gcSize = 0.01f,
		float gcThreshold = 20.0f)
	{
		pcl::GeometricConsistencyGrouping<PointModelT, PointSceneT> gc_clusterer;
		gc_clusterer.setGCSize(gcSize); // 设置检查几何一致性时的空间分辨率
		gc_clusterer.setGCThreshold(gcThreshold); // 设置最小的聚类数量

		gc_clusterer.setInputCloud(modelKeypoints);
		gc_clusterer.setSceneCloud(sceneKeypoints);
		gc_clusterer.setModelSceneCorrespondences(modelSceneCorrs);

		bool flag = gc_clusterer.recognize(rotoTranslations, clusteredCorrs);

		return flag ? 1 : -1;
	}


	template <typename PointInT>
	inline int EstimateHarris3D(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		pcl::PointCloud<pcl::PointXYZI>::Ptr outcloud,
		float normalRadius = 0.1f, float searchRadius = 0.1f)
	{
		pcl::HarrisKeypoint3D< PointInT, pcl::PointXYZI, NormalT > harris_detector;

		harris_detector.setRadius(normalRadius); // 设置法向量估算的半径
		harris_detector.setRadiusSearch(searchRadius); // 设置关键点估计的近邻搜索半径

		harris_detector.setInputCloud(inCloud);
		harris_detector.compute(*outcloud);
		return 1;
	}

	template <typename PointInOut>
	inline int GetUniformSampling(
		const typename pcl::PointCloud<PointInOut>::ConstPtr inCloud,
		typename pcl::PointCloud<PointInOut>::Ptr outcloud,
		const float &searchRadius = -1.0f /*0.03f*/)
	{
		pcl::UniformSampling< PointInOut > uniform_sampling;

		if (searchRadius < 0)
		{
			float cloudResolution = 
				static_cast<float>(ComputeCloudResolution<PointInOut>(inCloud));
			assert(cloudResolution);
			uniform_sampling.setRadiusSearch(10 * cloudResolution);
		}
		else
		{
			uniform_sampling.setRadiusSearch(searchRadius);
		}

		uniform_sampling.setInputCloud(inCloud);
		uniform_sampling.filter(*outcloud);
		return 1;
	}

	struct QPCL_ENGINE_LIB_API ConditionParameters
	{
		///NOTE: DISTINCT CLOUD METHOD NOT IMPLEMENTED
		enum ConditionType { CONDITION_OR, CONDITION_AND };
		enum ComparisonType { GT, GE, LT, LE, EQ };

		struct QPCL_ENGINE_LIB_API ComparisonParam
		{
			ComparisonType comparison_type_;
			std::string fieldName_;
			double min_threshold_;
			double max_threshold_;
		};

		ConditionParameters()
			: condition_type_(CONDITION_OR)
		{
		}

		ConditionType condition_type_;
		std::vector<ComparisonParam> condition_params_;
	};

	template <typename PointInOut>
	inline int ConditionalRemovalFilter(
		const typename pcl::PointCloud<PointInOut>::ConstPtr inCloud,
		const ConditionParameters &params,
		typename pcl::PointCloud<PointInOut>::Ptr outCloud,
		bool keepOrganized = false
    )
	{
        typename pcl::ConditionBase<PointInOut>::Ptr condition;
		switch (params.condition_type_)
        {
		case ConditionParameters::ConditionType::CONDITION_OR:
		{
			// Build the condition or for filtering
            typename pcl::ConditionOr<PointInOut>::Ptr range_cond(
				new pcl::ConditionOr<PointInOut>()
			);
			condition = range_cond;
			break;
		}
		case ConditionParameters::ConditionType::CONDITION_AND:
		{
            typename pcl::ConditionAnd<PointInOut>::Ptr range_cond(
				new pcl::ConditionAnd<PointInOut>()
			);
			condition = range_cond;
			break;
		}
		default:
			break;
		}

		for (size_t i = 0; i < params.condition_params_.size(); ++i)
		{
			ConditionParameters::ComparisonParam cp = params.condition_params_[i];
            pcl::ComparisonOps::CompareOp ops = static_cast<pcl::ComparisonOps::CompareOp>(static_cast<int>(cp.comparison_type_));
			double threshod = 0.0;
			if (pcl::ComparisonOps::CompareOp::GT == ops ||
				pcl::ComparisonOps::CompareOp::GE == ops)
			{
				threshod = cp.min_threshold_;
			}
			else if (pcl::ComparisonOps::CompareOp::LT == ops ||
				pcl::ComparisonOps::CompareOp::LE == ops)
			{
				threshod = cp.max_threshold_;
			}
			else
			{
				threshod = (cp.max_threshold_ + cp.min_threshold_) * 0.5;
			}
            condition->addComparison(typename pcl::FieldComparison<PointInOut>::ConstPtr(
				new pcl::FieldComparison<PointInOut>(cp.fieldName_, ops, threshod))
			);
		}

		if (!condition->isCapable())
		{
			return -1;
		}

		// Build the conditionRemoval Filter
        typename pcl::ConditionalRemoval<PointInOut> condrem(false);
		condrem.setCondition(condition);
		condrem.setInputCloud(inCloud);
		condrem.setKeepOrganized(keepOrganized);
		// Apply filter
		condrem.filter(*outCloud);

		return 1;
	}


	struct QPCL_ENGINE_LIB_API MLSParameters
	{
		///NOTE: DISTINCT CLOUD METHOD NOT IMPLEMENTED
		enum UpsamplingMethod { NONE, SAMPLE_LOCAL_PLANE, RANDOM_UNIFORM_DENSITY, VOXEL_GRID_DILATION };

		MLSParameters()
			: order_(0)
			, polynomial_fit_(false)
			, search_radius_(0)
			, sqr_gauss_param_(0)
			, compute_normals_(false)
			, upsample_method_(NONE)
			, upsampling_radius_(0)
			, upsampling_step_(0)
			, step_point_density_(0)
			, dilation_voxel_size_(0)
			, dilation_iterations_(0)
		{
		}

		int order_;
		bool polynomial_fit_;
		double search_radius_;
		double sqr_gauss_param_;
		bool compute_normals_;
		UpsamplingMethod upsample_method_;
		double upsampling_radius_;
		double upsampling_step_;
		int step_point_density_;
		double dilation_voxel_size_;
		int dilation_iterations_;
	};

	// for smooth filter
	template <typename PointInT, typename PointOutT>
	inline int SmoothMls(
		const typename pcl::PointCloud<PointInT>::ConstPtr &inCloud,
		const MLSParameters &params,
		typename pcl::PointCloud<PointOutT>::Ptr &outcloud
#ifdef LP_PCL_PATCH_ENABLED
		, pcl::PointIndicesPtr &mapping_ids
#endif
	)
	{
		typename pcl::search::KdTree<PointInT>::Ptr tree(new pcl::search::KdTree<PointInT>);

#ifdef _OPENMP
		//create the smoothing object
		pcl::MovingLeastSquaresOMP< PointInT, PointOutT > smoother;
		int n_threads = omp_get_max_threads();
		smoother.setNumberOfThreads(n_threads);
#else
		pcl::MovingLeastSquares< PointInT, PointOutT > smoother;
#endif
		smoother.setInputCloud(inCloud);
		smoother.setSearchMethod(tree);
		smoother.setSearchRadius(params.search_radius_);
		smoother.setComputeNormals(params.compute_normals_);
		smoother.setPolynomialFit(params.polynomial_fit_);

		if (params.polynomial_fit_)
		{
			smoother.setPolynomialOrder(params.order_);
			smoother.setSqrGaussParam(params.sqr_gauss_param_);
		}

		switch (params.upsample_method_)
		{
		case (MLSParameters::NONE):
		{
			smoother.setUpsamplingMethod(pcl::MovingLeastSquares<PointInT, PointOutT>::NONE);
			//no need to set other parameters here!
			break;
		}

		case (MLSParameters::SAMPLE_LOCAL_PLANE):
		{
			smoother.setUpsamplingMethod(pcl::MovingLeastSquares<PointInT, PointOutT>::SAMPLE_LOCAL_PLANE);
			smoother.setUpsamplingRadius(params.upsampling_radius_);
			smoother.setUpsamplingStepSize(params.upsampling_step_);
			break;
		}

		case (MLSParameters::RANDOM_UNIFORM_DENSITY):
		{
			smoother.setUpsamplingMethod(pcl::MovingLeastSquares<PointInT, PointOutT>::RANDOM_UNIFORM_DENSITY);
			smoother.setPointDensity(params.step_point_density_);
			break;
		}

		case (MLSParameters::VOXEL_GRID_DILATION):
		{
			smoother.setUpsamplingMethod(pcl::MovingLeastSquares<PointInT, PointOutT>::VOXEL_GRID_DILATION);
			smoother.setDilationVoxelSize(static_cast<float>(params.dilation_voxel_size_));
			smoother.setDilationIterations(params.dilation_iterations_);
			break;
		}
		}

		smoother.process(*outcloud);

#ifdef LP_PCL_PATCH_ENABLED
		mapping_ids = smoother.getCorrespondingIndices();
#endif
		return 1;
	}

	// for normal estimation
	template <typename PointInT, typename PointOutT>
	inline int ComputeNormals(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		typename pcl::PointCloud<PointOutT>::Ptr outcloud,
		const float radius = -1.0f /*10.0f*/,
		const bool useKnn = true, // true if use knn, false if radius search
		bool normalConsistency = false,
		int maxThreadCount = QThread::idealThreadCount()
	)
	{
		typename pcl::NormalEstimationOMP<PointInT, PointOutT> normal_estimator;

		// Create a search tree, use KDTreee for non-organized data.
        typename pcl::search::Search<PointInT>::Ptr tree;
		if (inCloud->isOrganized())
		{
			tree.reset(new pcl::search::OrganizedNeighbor<PointInT>());
		}
		else
		{
			tree.reset(new pcl::search::KdTree<PointInT>());
		}
		tree->setInputCloud(inCloud);

		normal_estimator.setSearchMethod(tree);

		if (useKnn) //use knn
		{
            int knn_radius = static_cast<int>(radius); //cast to int
			normal_estimator.setKSearch(knn_radius);
		}
		else //use radius search
		{
			if (radius < 0)
			{
				float cloudResolution = 
					static_cast<float>(ComputeCloudResolution<PointInT>(inCloud));
				assert(cloudResolution);
				normal_estimator.setRadiusSearch(10 * cloudResolution);
			}
			else
			{
				normal_estimator.setRadiusSearch(radius);
			}
		}

		if (normalConsistency)
		{
			/**
			 * NOTE: setting viewpoint is very important, so that we can ensure
			 * normals are all pointed in the same direction!
			 */
			normal_estimator.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
		}

		normal_estimator.setInputCloud(inCloud);
		normal_estimator.setNumberOfThreads(maxThreadCount);
		normal_estimator.compute(*outcloud);

		return 1;
	}

	template <typename PointInOut>
	inline int PassThroughFilter(
		const typename pcl::PointCloud<PointInOut>::ConstPtr inCloud,
		typename pcl::PointCloud<PointInOut>::Ptr outcloud,
		const QString& filterFieldName = "z",
        const float &limit_min = 0.1f,
        const float &limit_max = 1.1f)
	{
		pcl::PassThrough<PointInOut> pass;
		pass.setInputCloud(inCloud);
		pass.setFilterFieldName(filterFieldName.toStdString());
		pass.setFilterLimits(limit_min, limit_max);
		pass.filter(*outcloud);
		return 1;
	}

	template <typename PointInOut>
	inline int VoxelGridFilter(
		const typename pcl::PointCloud<PointInOut>::ConstPtr inCloud,
		typename pcl::PointCloud<PointInOut>::Ptr outcloud,
		const float	&leafSizeX = -0.01f,
		const float &leafSizeY = -0.01f,
		const float &leafSizeZ = -0.01f)
	{
		// Create the filtering object: downsample the dataset using a leaf size of 1cm
		pcl::VoxelGrid<PointInOut> vg;
		vg.setInputCloud(inCloud);
		if (leafSizeX < 0 || leafSizeY < 0 || leafSizeZ < 0)
		{
			float cloudResolution = 
				static_cast<float>(ComputeCloudResolution<PointInOut>(inCloud));
			assert(cloudResolution);
			float newLeafSize = 10 * cloudResolution;
			vg.setLeafSize(newLeafSize, newLeafSize, newLeafSize);
		}
		else
		{
			vg.setLeafSize(leafSizeX, leafSizeY, leafSizeZ);
		}

		vg.filter(*outcloud);
		if (outcloud->points.size() == inCloud->points.size())
		{
			CVLog::Warning("[PCLModules::VoxelGridFilter] leaf size is too small, voxel grid filter failed!");
		}
		else
		{
			CVLog::Print(
				QString("[PCLModules::VoxelGridFilter] Filter original size[%1] to size[%2]!").
				arg(inCloud->size()).arg(outcloud->size()));
		}
		return 1;
	}

	template <typename PointInOut>
	inline int ExtractIndicesFilter(
		const typename pcl::PointCloud<PointInOut>::ConstPtr inCloud,
		const pcl::PointIndices::ConstPtr inliers,
		typename pcl::PointCloud<PointInOut>::Ptr outcloud = nullptr,
		typename pcl::PointCloud<PointInOut>::Ptr outcloud2 = nullptr)
	{
		pcl::ExtractIndices<PointInOut> extract;
		extract.setInputCloud(inCloud);
		extract.setIndices(inliers);

		if (outcloud)
		{
			extract.setNegative(false);
			extract.filter(*outcloud);
		}

		if (outcloud2)
		{
			extract.setNegative(true);
			extract.filter(*outcloud2);
		}
		return 1;
	}

	// for convexHull reconstruction
	template <typename PointInT>
	inline int GetConvexHullReconstruction(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		PCLMesh &outMesh,
		int dimension = 3
	)
	{
		pcl::ConvexHull<PointInT> hull;
		hull.setInputCloud(inCloud);
		hull.setDimension(dimension);

        typename pcl::PointCloud<PointInT>::Ptr surface_hull(new pcl::PointCloud<PointInT>());
		hull.reconstruct(*surface_hull, outMesh.polygons);

		CVLog::Print(QString("convex hull area [1%], convex hull volume [%2]").
			arg(hull.getTotalArea()).arg(hull.getTotalVolume()));

		// Convert the PointCloud into a PCLPointCloud2
		TO_PCL_CLOUD(*surface_hull, outMesh.cloud);

		return 1;
	}

	// for concaveHull reconstruction
	template <typename PointInT>
	inline int GetConcaveHullReconstruction(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		PCLMesh &outMesh,
		int dimension = 3,
        float alpha = 0.05f)
	{
		pcl::ConcaveHull<PointInT> chull;
		chull.setInputCloud(inCloud);
		chull.setAlpha(alpha);
		chull.setDimension(dimension);
        typename pcl::PointCloud<PointInT>::Ptr surface_hull(new pcl::PointCloud<PointInT>());
		chull.reconstruct(*surface_hull, outMesh.polygons);

		// Convert the PointCloud into a PCLPointCloud2
		TO_PCL_CLOUD(*surface_hull, outMesh.cloud);

		return 1;
	}

	template <typename PointInT>
	inline int CropHullFilter(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		const PointCloudT::ConstPtr boundingBox,
		typename pcl::PointCloud<PointInT>::Ptr outCloud,
		int dimensions = 2
	)
	{
		// reconstruction
		PCLMesh mesh;
		if (!PCLModules::GetConvexHullReconstruction<PointT>(boundingBox, mesh, dimensions))
		{
			return -1;
		}

		pcl::CropHull<PointInT> bb_filter;
		bb_filter.setInputCloud(inCloud);
		bb_filter.setDim(dimensions);
		bb_filter.setHullIndices(mesh.polygons);
        typename pcl::PointCloud<PointInT>::Ptr surface_hull(new pcl::PointCloud<PointInT>);
		FROM_PCL_CLOUD(mesh.cloud, *surface_hull);
		if (surface_hull->width * surface_hull->height == 0)
		{
			return -1;
		}
		bb_filter.setHullCloud(surface_hull);
		bb_filter.filter(*outCloud);
		return 1;
	}

	template <typename PointInT>
	inline int GetMinCutSegmentation(
		const typename pcl::PointCloud<PointInT>::ConstPtr inCloud,
		std::vector <pcl::PointIndices> &outClusters,
		PointCloudRGB::Ptr cloud_segmented,
        const PointInT foregroundPoint,
		int neighboursNumber = 14,
        float smoothSigma = 0.25f,
        float backWeightRadius = 0.8f,
        float foreWeight = 0.5f,
		const pcl::IndicesConstPtr indices = nullptr
		)
	{
        typename pcl::MinCutSegmentation<PointInT> seg;
		seg.setInputCloud(inCloud);
        if (indices)
		{
			seg.setIndices(indices);
		}

        typename pcl::PointCloud<PointInT>::Ptr foreground_points(new pcl::PointCloud<PointInT> ());
		foreground_points->points.push_back(foregroundPoint);
		seg.setForegroundPoints(foreground_points);

		seg.setSigma(smoothSigma);
		seg.setRadius(backWeightRadius);
		seg.setNumberOfNeighbours(neighboursNumber);
		seg.setSourceWeight(foreWeight);
		seg.extract(outClusters);
		pcl::copyPointCloud(*seg.getColoredCloud(), *cloud_segmented);
		
		return 1;
	}


	template <typename PointInOut, typename NormalType>
	inline int GetBoundaryCloud(
		const typename pcl::PointCloud<PointInOut>::ConstPtr inCloud,
		const typename pcl::PointCloud<NormalType>::ConstPtr normals,
		typename pcl::PointCloud<PointInOut>::Ptr boundaryCloud,
		const float angleThreshold = 90.0f, // 45.0f for radius search && 90.0f for knn search
		const float radius = -1.0f,  // 0.05f for radius search && 20.0f for knn search
		const bool useKnn = true // true if use knn, false if radius search
		)
	{
		pcl::BoundaryEstimation<PointInOut, NormalType, pcl::Boundary> boundEst;
		pcl::PointCloud<pcl::Boundary> boundaries;
		boundEst.setInputCloud(inCloud);
		boundEst.setInputNormals(normals);
		if (useKnn)
		{
			boundEst.setKSearch(int(radius)); // the bigger the more accurate
		}
		else // slow
		{
			if (radius < 0)
			{
				float cloudResolution = 
					static_cast<float>(ComputeCloudResolution<PointInOut>(inCloud));
				assert(cloudResolution);
				boundEst.setRadiusSearch(10 * cloudResolution);
			}
			else
			{
				boundEst.setRadiusSearch(radius);
			}
		}
		
        typename pcl::search::KdTree<PointInOut>::Ptr searchTree(new pcl::search::KdTree<PointInOut>());
        boundEst.setSearchMethod(searchTree);
		boundEst.setAngleThreshold(angleThreshold * CV_DEG_TO_RAD);
		boundEst.compute(boundaries);

		boundaryCloud->clear();
        for (size_t i = 0; i < inCloud->size(); i++)
		{
			if (boundaries[i].boundary_point > 0)
			{
				boundaryCloud->push_back(inCloud->points[i]);
			}
		}

		return 1;
	}

	template <typename PointInT, typename PointOutT>
	inline bool ICPRegistration(
		const typename pcl::PointCloud<PointInT>::ConstPtr targetCloud,
		const typename pcl::PointCloud<PointOutT>::ConstPtr sourceCloud,
		typename pcl::PointCloud<PointOutT>::Ptr outRegistered,
		int ipcMaxIterations = 5,
		float icpCorrDistance = 0.005f)
	{
		pcl::IterativeClosestPoint<PointInT, PointOutT> icp;
		icp.setMaximumIterations(ipcMaxIterations);
		icp.setMaxCorrespondenceDistance(icpCorrDistance);
		icp.setInputTarget(targetCloud);
		icp.setInputSource(sourceCloud);
		icp.align(*outRegistered);

		return icp.hasConverged();
	}

	template <typename PointSceneT, typename PointModelT>
	inline int GetHypothesesVerification(
		const typename pcl::PointCloud<PointSceneT>::Ptr sceneCloud,
		std::vector<typename pcl::PointCloud<PointModelT>::ConstPtr> modelClouds,
		std::vector<bool> &hypothesesMask,
		float clusterReg = 5.0f,
		float inlierThreshold = 0.005f,
		float occlusionThreshold = 0.01f,
		float radiusClutter = 0.03f,
		float regularizer = 3.0f,
		float radiusNormals = 0.05f,
		bool detectClutter = true)
	{
		pcl::GlobalHypothesesVerification<PointModelT, PointSceneT> GoHv;
		GoHv.setSceneCloud(sceneCloud);  // Scene Cloud
		GoHv.addModels(modelClouds, true);  // Models to verify

		GoHv.setInlierThreshold(inlierThreshold);
		GoHv.setOcclusionThreshold(occlusionThreshold);
		GoHv.setRadiusClutter(radiusClutter);
		GoHv.setClutterRegularizer(clusterReg);
		GoHv.setRegularizer(regularizer);
		GoHv.setRadiusNormals(radiusNormals);
		GoHv.setDetectClutter(detectClutter);

		GoHv.verify();
		GoHv.getMask(hypothesesMask);  // i-element TRUE if hvModels[i] verifies hypotheses

		return 1;
	}
}

// class
namespace PCLModules
{
	class QPCL_ENGINE_LIB_API FeatureCloud
	{
	public:
		// A bit of shorthand
		typedef pcl::PointCloud<NormalT> SurfaceNormals;
		typedef pcl::PointCloud<pcl::FPFHSignature33> LocalFeatures;
		typedef pcl::search::KdTree<PointT> SearchMethod;

		FeatureCloud();
		~FeatureCloud() {}

		void setNormalRadius(float normalRadius) { m_normalRadius = normalRadius; }
		void setFeatureRadius(float featureRadius) { m_featureRadius = featureRadius; }
		void setmaxThreadCount(int maxThreadCount) { m_maxThreadCount = maxThreadCount; }

		// Process the given cloud
		inline void setInputCloud(PointCloudT::Ptr xyz) { m_xyz = xyz; processInput(); }

		// Load and process the cloud in the given PCD file
		void loadInputCloud(const std::string &pcd_file);

		// Get a pointer to the cloud 3D points
		inline PointCloudT::Ptr getPointCloud() const { return (m_xyz); }

		// Get a pointer to the cloud of 3D surface normals
		inline SurfaceNormals::Ptr getSurfaceNormals() const { return (m_normals); }

		// Get a pointer to the cloud of feature descriptors
		inline LocalFeatures::Ptr getLocalFeatures() const { return (m_features); }

	protected:
		// Compute the surface normals and local features
		inline void processInput()
		{
			computeSurfaceNormals();
			computeLocalFeatures();
		}

		// Compute the surface normals
		void computeSurfaceNormals();

		// Compute the local feature descriptors
		void computeLocalFeatures();

	private:
		// Point cloud data
		PointCloudT::Ptr m_xyz;
		SurfaceNormals::Ptr m_normals;
		LocalFeatures::Ptr m_features;
		SearchMethod::Ptr m_searchMethod;

		// Parameters
		float m_normalRadius;
		float m_featureRadius;
		int m_maxThreadCount;
	};

	class QPCL_ENGINE_LIB_API TemplateMatching
	{
	public:

		// A struct for storing alignment results
		struct Result
		{
			float fitness_score;
			Eigen::Matrix4f final_transformation;
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		};

		TemplateMatching();
		~TemplateMatching() {}

		void setminSampleDis(float minSampleDistance) { m_minSampleDistance = minSampleDistance; }
		void setmaxCorrespondenceDis(float maxCorrespondenceDistance) { m_maxCorrespondenceDistance = maxCorrespondenceDistance; }
		void setmaxIterations(int maxIterations) { m_nr_iterations = maxIterations; }

		// Set the given cloud as the target to which the templates will be aligned
		void setTargetCloud(FeatureCloud &target_cloud);

		// Add the given cloud to the list of template clouds
		inline void addTemplateCloud(FeatureCloud &template_cloud) { m_templates.push_back(template_cloud); }
		
		// get the template cloud by the given index
		inline FeatureCloud* getTemplateCloud(int index) {
            if (static_cast<size_t>(index) > m_templates.size())
				return nullptr;
            return &m_templates[static_cast<size_t>(index)];
		}

		// Align the given template cloud to the target specified by setTargetCloud ()
		void align(FeatureCloud &template_cloud, TemplateMatching::Result &result);

		// Align all of template clouds set by addTemplateCloud to the target specified by setTargetCloud ()
		void alignAll(std::vector<TemplateMatching::Result, Eigen::aligned_allocator<Result> > &results);

		// Align all of template clouds to the target cloud to find the one with best alignment score
		int findBestAlignment(TemplateMatching::Result &result);

		inline void clear() { m_templates.clear(); }

	private:
		// A list of template clouds and the target to which they will be aligned
		std::vector<FeatureCloud> m_templates;
		FeatureCloud m_target;

		// The Sample Consensus Initial Alignment (SAC-IA) registration routine and its parameters
		pcl::SampleConsensusInitialAlignment<PointT, PointT, pcl::FPFHSignature33> m_sac_ia;
		float m_minSampleDistance;
		float m_maxCorrespondenceDistance;
		int m_nr_iterations;
	};

}


#endif // QPCL_PCLMODULES_HEADER

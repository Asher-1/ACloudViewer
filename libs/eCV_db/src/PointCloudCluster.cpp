// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPointCloud.h"

// CV_CORE_LIB
#include <Logging.h>
#include <Parallel.h>
#include <ScalarField.h>

// ECV_DB_LIB
#include "ecvCone.h"
#include "ecvCylinder.h"
#include "ecvGenericMesh.h"
#include "ecvGenericPointCloud.h"
#include "ecvKDTreeFlann.h"
#include "ecvMesh.h"
#include "ecvPlane.h"
#include "ecvSphere.h"
#include "ecvTorus.h"

#ifdef CV_RANSAC_SUPPORT
// PrimitiveShapes/MiscLib
#include <ConePrimitiveShape.h>
#include <ConePrimitiveShapeConstructor.h>
#include <CylinderPrimitiveShape.h>
#include <CylinderPrimitiveShapeConstructor.h>
#include <PlanePrimitiveShape.h>
#include <PlanePrimitiveShapeConstructor.h>
#include <RansacShapeDetector.h>
#include <SpherePrimitiveShape.h>
#include <SpherePrimitiveShapeConstructor.h>
#include <TorusPrimitiveShape.h>
#include <TorusPrimitiveShapeConstructor.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>
#include <unordered_set>

using namespace cloudViewer;

std::vector<int> ccPointCloud::ClusterDBSCAN(double eps,
                                             size_t min_points,
                                             bool print_progress) const {
    geometry::KDTreeFlann kdtree(*this);

    // precompute all neighbours
    utility::LogDebug("Precompute Neighbours");
    utility::ConsoleProgressBar progress_bar(
            this->size(), "Precompute Neighbours", print_progress);
    std::vector<std::vector<int>> nbs(this->size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
#endif
    for (int idx = 0; idx < int(this->size()); ++idx) {
        std::vector<double> dists2;
        kdtree.SearchRadius(getEigenPoint(static_cast<size_t>(idx)), eps,
                            nbs[idx], dists2);

#ifdef _OPENMP
#pragma omp critical(ClusterDBSCAN)
#endif
        {
            ++progress_bar;
        }
    }
    utility::LogDebug("Done Precompute Neighbours");

    // set all labels to undefined (-2)
    utility::LogDebug("Compute Clusters");
    progress_bar.reset(this->size(), "Clustering", print_progress);
    std::vector<int> labels(this->size(), -2);
    int cluster_label = 0;
    for (size_t idx = 0; idx < this->size(); ++idx) {
        if (labels[idx] != -2) {  // label is not undefined
            continue;
        }

        // check density
        if (nbs[idx].size() < min_points) {
            labels[idx] = -1;
            continue;
        }

        std::unordered_set<int> nbs_next(nbs[idx].begin(), nbs[idx].end());
        std::unordered_set<int> nbs_visited;
        nbs_visited.insert(int(idx));

        labels[idx] = cluster_label;
        ++progress_bar;
        while (!nbs_next.empty()) {
            int nb = *nbs_next.begin();
            nbs_next.erase(nbs_next.begin());
            nbs_visited.insert(nb);

            if (labels[nb] == -1) {  // noise label
                labels[nb] = cluster_label;
                ++progress_bar;
            }
            if (labels[nb] != -2) {  // not undefined label
                continue;
            }
            labels[nb] = cluster_label;
            ++progress_bar;

            if (nbs[nb].size() >= min_points) {
                for (int qnb : nbs[nb]) {
                    if (nbs_visited.count(qnb) == 0) {
                        nbs_next.insert(qnb);
                    }
                }
            }
        }

        cluster_label++;
    }

    utility::LogDebug("Done Compute Clusters: {:d}", cluster_label);
    return labels;
}

#ifdef CV_RANSAC_SUPPORT

std::string geometry::RansacResult::getTypeName() const {
    if (!this->primitive) {
        return "";
    }

    std::shared_ptr<ccGenericPrimitive> prim =
            std::static_pointer_cast<ccGenericPrimitive>(this->primitive);

    if (!prim) {
        return "";
    }

    return prim->getTypeName().toStdString();
}

unsigned geometry::RansacResult::getDrawingPrecision() const {
    if (!this->primitive) {
        return -1;
    }

    std::shared_ptr<ccGenericPrimitive> prim =
            std::static_pointer_cast<ccGenericPrimitive>(this->primitive);

    if (!prim) {
        return -1;
    }

    return prim->getDrawingPrecision();
}

bool geometry::RansacResult::setDrawingPrecision(unsigned steps) {
    if (!this->primitive) {
        return false;
    }

    std::shared_ptr<ccGenericPrimitive> prim =
            std::static_pointer_cast<ccGenericPrimitive>(this->primitive);

    if (!prim) {
        return false;
    }

    return prim->setDrawingPrecision(steps);
}

geometry::RansacResults ccPointCloud::ExecuteRANSAC(
        const geometry::RansacParams& params,
        bool print_progress /* = false*/) {
    geometry::RansacResults group;
    // consistency check
    {
        if (params.primEnabled.size() == 0) {
            utility::LogError(
                    "[ccPointCloud::ExecuteRANSAC] No primitive type "
                    "selected!");
            return group;
        }
    }

    unsigned count = this->size();
    bool hasNorms = this->hasNormals();
    CCVector3 bbMin, bbMax;
    this->getBoundingBox(bbMin, bbMax);
    const CCVector3d& globalShift = this->getGlobalShift();
    double globalScale = this->getGlobalScale();
    ransac::RansacPointCloud cloud;
    {
        try {
            cloud.reserve(count);
        } catch (...) {
            utility::LogError(
                    "[ccPointCloud::ExecuteRANSAC] Could not create temporary "
                    "cloud, Not enough memory!");
            return group;
        }

        // default point & normal
        ransac::RansacPoint Pt;
        Pt.normal[0] = 0.0;
        Pt.normal[1] = 0.0;
        Pt.normal[2] = 0.0;
        for (unsigned i = 0; i < count; ++i) {
            const CCVector3* P = this->getPoint(i);
            Pt.pos[0] = static_cast<float>(P->x);
            Pt.pos[1] = static_cast<float>(P->y);
            Pt.pos[2] = static_cast<float>(P->z);
            if (hasNorms) {
                const CCVector3& N = this->getPointNormal(i);
                Pt.normal[0] = static_cast<float>(N.x);
                Pt.normal[1] = static_cast<float>(N.y);
                Pt.normal[2] = static_cast<float>(N.z);
            }
            Pt.index = i;
            cloud.push_back(Pt);
        }

        // manually set bounding box!
        Vec3f cbbMin, cbbMax;
        cbbMin[0] = static_cast<float>(bbMin.x);
        cbbMin[1] = static_cast<float>(bbMin.y);
        cbbMin[2] = static_cast<float>(bbMin.z);
        cbbMax[0] = static_cast<float>(bbMax.x);
        cbbMax[1] = static_cast<float>(bbMax.y);
        cbbMax[2] = static_cast<float>(bbMax.z);
        cloud.setBBox(cbbMin, cbbMax);
    }

    RansacShapeDetector::Options ransacOptions;
    {
        ransacOptions.m_epsilon = params.epsilon;
        ransacOptions.m_bitmapEpsilon = params.bitmapEpsilon;
        ransacOptions.m_normalThresh = static_cast<float>(
                cos(cloudViewer::DegreesToRadians(params.maxNormalDev_deg)));
        assert(ransacOptions.m_normalThresh >= 0);
        ransacOptions.m_probability = params.probability;
        ransacOptions.m_minSupport = params.supportPoints;
    }
    const float scale = cloud.getScale();

    if (!hasNorms) {
        cloud.calcNormals(.01f * scale);

        if (this->reserveTheNormsTable()) {
            for (unsigned i = 0; i < count; ++i) {
                Vec3f& Nvi = cloud[i].normal;
                CCVector3 Ni = CCVector3::fromArray(Nvi);
                // normalize the vector in case of
                Ni.normalize();
                this->addNorm(Ni);
            }
            this->showNormals(true);

            // currently selected entities appearance may have changed!
            this->setRedrawFlagRecursive(true);
        } else {
            utility::LogError(
                    "[ccPointCloud::ExecuteRANSAC] Not enough memory to "
                    "compute normals!");
            return group;
        }
    }

    RansacShapeDetector detector(ransacOptions);  // the detector object

    const int preserve_detectors_number = 5;
    bool detectors_enabled[preserve_detectors_number] = {false, false, false,
                                                         false, false};
    for (auto object_type : params.primEnabled) {
        switch (object_type) {
            case geometry::RansacParams::RPT_PLANE:
                if (!detectors_enabled[geometry::RansacParams::RPT_PLANE]) {
                    detector.Add(new PlanePrimitiveShapeConstructor());
                    detectors_enabled[geometry::RansacParams::RPT_PLANE] = true;
                }
                break;
            case geometry::RansacParams::RPT_SPHERE:
                if (!detectors_enabled[geometry::RansacParams::RPT_SPHERE]) {
                    detector.Add(new SpherePrimitiveShapeConstructor());
                    detectors_enabled[geometry::RansacParams::RPT_SPHERE] =
                            true;
                }
                break;
            case geometry::RansacParams::RPT_CYLINDER:
                if (!detectors_enabled[geometry::RansacParams::RPT_CYLINDER]) {
                    detector.Add(new CylinderPrimitiveShapeConstructor());
                    detectors_enabled[geometry::RansacParams::RPT_CYLINDER] =
                            true;
                }
                break;
            case geometry::RansacParams::RPT_CONE:
                if (!detectors_enabled[geometry::RansacParams::RPT_CONE]) {
                    detector.Add(new ConePrimitiveShapeConstructor());
                    detectors_enabled[geometry::RansacParams::RPT_CONE] = true;
                }
                break;
            case geometry::RansacParams::RPT_TORUS:
                if (!detectors_enabled[geometry::RansacParams::RPT_TORUS]) {
                    detector.Add(new TorusPrimitiveShapeConstructor());
                    detectors_enabled[geometry::RansacParams::RPT_TORUS] = true;
                }
                break;
            default:
                utility::LogWarning("unsupported detector object!");
                break;
        }
    }

    unsigned remaining = count;
    typedef std::pair<MiscLib::RefCountPtr<PrimitiveShape>, size_t>
            DetectedShape;
    MiscLib::Vector<DetectedShape> shapes;  // stores the detected shapes

    // run detection
    // returns number of unassigned points
    // the array shapes is filled with pointers to the detected shapes
    // the second element per shapes gives the number of points assigned to that
    // primitive (the support) the points belonging to the first shape
    // (shapes[0]) have been sorted to the end of pc, i.e. into the range [
    // pc.size() - shapes[0].second, pc.size() ) the points of shape i are found
    // in the range [ pc.size() - \sum_{j=0..i} shapes[j].second, pc.size() -
    // \sum_{j=0..i-1} shapes[j].second )
    remaining = static_cast<unsigned>(
            detector.Detect(cloud, 0, cloud.size(), &shapes));

#if 0  // def _DEBUG
	FILE* fp = fopen("RANS_SD_trace.txt", "wt");

	fprintf(fp, "[Options]\n");
	fprintf(fp, "epsilon=%f\n", ransacOptions.m_epsilon);
	fprintf(fp, "bitmap epsilon=%f\n", ransacOptions.m_bitmapEpsilon);
	fprintf(fp, "normal thresh=%f\n", ransacOptions.m_normalThresh);
	fprintf(fp, "min support=%i\n", ransacOptions.m_minSupport);
	fprintf(fp, "probability=%f\n", ransacOptions.m_probability);

	fprintf(fp, "\n[Statistics]\n");
	fprintf(fp, "input points=%i\n", count);
	fprintf(fp, "segmented=%i\n", count - remaining);
	fprintf(fp, "remaining=%i\n", remaining);

	if (shapes.size() > 0)
	{
		fprintf(fp, "\n[Shapes]\n");
		for (unsigned i = 0; i < shapes.size(); ++i)
		{
			PrimitiveShape* shape = shapes[i].first;
			size_t shapePointsCount = shapes[i].second;

			std::string desc;
			shape->Description(&desc);
			fprintf(fp, "#%i - %s - %i points\n", i + 1, desc.c_str(), shapePointsCount);
		}
	}
	fclose(fp);
#endif

    if (remaining == count && shapes.size() == 0) {
        utility::LogWarning(
                "[ccPointCloud::ExecuteRANSAC] Segmentation failed...");
        return group;
    }

    if (shapes.size() > 0) {
        unsigned planeCount = 1;
        unsigned sphereCount = 1;
        unsigned cylinderCount = 1;
        unsigned coneCount = 1;
        unsigned torusCount = 1;

        for (MiscLib::Vector<DetectedShape>::const_iterator it = shapes.begin();
             it != shapes.end(); ++it) {
            const PrimitiveShape* shape = it->first;
            unsigned shapePointsCount = static_cast<unsigned>(it->second);

            // too many points?!
            if (shapePointsCount > count) {
                utility::LogError(
                        "[ccPointCloud::ExecuteRANSAC] Inconsistent result!");
                break;
            }

            if (shapePointsCount < params.supportPoints) {
                utility::LogWarning(
                        "[ccPointCloud::ExecuteRANSAC] Skipping shape, {:d} "
                        "did not meet minimum point requirement",
                        shapePointsCount);
                count -= shapePointsCount;
                continue;
            }

            std::string desc;
            shape->Description(&desc);

            // points to current shapes last point in cloud
            const auto shapeCloudIndex = count - 1;

            // convert detected primitive into a CC primitive type
            std::shared_ptr<ccGenericPrimitive> prim = nullptr;
            switch (shape->Identifier()) {
                case geometry::RansacParams::RPT_PLANE:  // plane
                {
                    const PlanePrimitiveShape* plane =
                            static_cast<const PlanePrimitiveShape*>(shape);
                    Vec3f G = plane->Internal().getPosition();
                    Vec3f N = plane->Internal().getNormal();
                    Vec3f X = plane->getXDim();
                    Vec3f Y = plane->getYDim();

                    // we look for real plane extents
                    float minX, maxX, minY, maxY;
                    for (unsigned j = 0; j < shapePointsCount; ++j) {
                        std::pair<float, float> param;
                        plane->Parameters(cloud[shapeCloudIndex - j].pos,
                                          &param);
                        if (j != 0) {
                            if (minX < param.first)
                                minX = param.first;
                            else if (maxX > param.first)
                                maxX = param.first;
                            if (minY < param.second)
                                minY = param.second;
                            else if (maxY > param.second)
                                maxY = param.second;
                        } else {
                            minX = maxX = param.first;
                            minY = maxY = param.second;
                        }
                    }

                    // we recenter plane (as it is not always the case!)
                    float dX = maxX - minX;
                    float dY = maxY - minY;
                    G += X * (minX + dX / 2);
                    G += Y * (minY + dY / 2);

                    // we build matrix from these vectors
                    ccGLMatrix glMat(CCVector3::fromArray(X.getValue()),
                                     CCVector3::fromArray(Y.getValue()),
                                     CCVector3::fromArray(N.getValue()),
                                     CCVector3::fromArray(G.getValue()));

                    // plane primitive
                    prim = std::make_shared<ccPlane>(std::abs(dX), std::abs(dY),
                                                     &glMat);
                    prim->setSelectionBehavior(ccHObject::SELECTION_FIT_BBOX);
                    prim->enableStippling(true);
                    PointCoordinateType dip = 0.0f;
                    PointCoordinateType dipDir = 0.0f;
                    ccNormalVectors::ConvertNormalToDipAndDipDir(
                            CCVector3::fromArray(N.getValue()), dip, dipDir);
                    QString dipAndDipDirStr =
                            ccNormalVectors::ConvertDipAndDipDirToString(
                                    dip, dipDir);
                    prim->setName("Plane (" + dipAndDipDirStr + ")");
                    planeCount++;
                } break;

                case geometry::RansacParams::RPT_SPHERE:  // sphere
                {
                    const SpherePrimitiveShape* sphere =
                            static_cast<const SpherePrimitiveShape*>(shape);
                    float radius = sphere->Internal().Radius();
                    if (radius < params.minRadius ||
                        radius > params.maxRadius) {
                        count -= shapePointsCount;
                        continue;
                    }

                    Vec3f CC = sphere->Internal().Center();

                    // we build matrix from these vectors
                    ccGLMatrix glMat;
                    glMat.setTranslation(CC.getValue());
                    // sphere primitive
                    prim = std::make_shared<ccSphere>(radius, &glMat);
                    prim->setEnabled(true);
                    prim->setName(QString("Sphere (r=%1)").arg(radius, 0, 'f'));
                    sphereCount++;
                } break;

                case geometry::RansacParams::RPT_CYLINDER:  // cylinder
                {
                    const CylinderPrimitiveShape* cyl =
                            static_cast<const CylinderPrimitiveShape*>(shape);
                    float radius = cyl->Internal().Radius();
                    if (radius < params.minRadius ||
                        radius > params.maxRadius) {
                        count -= shapePointsCount;
                        continue;
                    }

                    Vec3f G = cyl->Internal().AxisPosition();
                    Vec3f N = cyl->Internal().AxisDirection();
                    Vec3f X = cyl->Internal().AngularDirection();
                    Vec3f Y = N.cross(X);

                    float hMin = cyl->MinHeight();
                    float hMax = cyl->MaxHeight();
                    float h = hMax - hMin;
                    G += N * (hMin + h / 2);

                    // we build matrix from these vectors
                    ccGLMatrix glMat(CCVector3::fromArray(X.getValue()),
                                     CCVector3::fromArray(Y.getValue()),
                                     CCVector3::fromArray(N.getValue()),
                                     CCVector3::fromArray(G.getValue()));

                    // cylinder primitive
                    prim = std::make_shared<ccCylinder>(radius, h, &glMat);
                    prim->setEnabled(true);
                    prim->setName(QString("Cylinder (r=%1/h=%2)")
                                          .arg(radius, 0, 'f')
                                          .arg(h, 0, 'f'));
                    cylinderCount++;
                } break;

                case geometry::RansacParams::RPT_CONE:  // cone
                {
                    const ConePrimitiveShape* cone =
                            static_cast<const ConePrimitiveShape*>(shape);
                    Vec3f CC = cone->Internal().Center();
                    Vec3f CA = cone->Internal().AxisDirection();
                    float alpha = cone->Internal().Angle();

                    // compute max height
                    Vec3f minP, maxP;
                    float minHeight, maxHeight;
                    minP = maxP = cloud[shapeCloudIndex].pos;
                    minHeight = maxHeight =
                            cone->Internal().Height(cloud[shapeCloudIndex].pos);
                    for (size_t j = 1; j < shapePointsCount; ++j) {
                        float h = cone->Internal().Height(
                                cloud[shapeCloudIndex - j].pos);
                        if (h < minHeight) {
                            minHeight = h;
                            minP = cloud[shapeCloudIndex - j].pos;
                        } else if (h > maxHeight) {
                            maxHeight = h;
                            maxP = cloud[shapeCloudIndex - j].pos;
                        }
                    }

                    float minRadius = tan(alpha) * minHeight;
                    float maxRadius = tan(alpha) * maxHeight;
                    if (minRadius < params.minRadius ||
                        maxRadius > params.maxRadius) {
                        count -= shapePointsCount;
                        continue;
                    }

                    // let's build the cone primitive
                    {
                        // the bottom should be the largest part so we inverse
                        // the axis direction
                        CCVector3 Z = -CCVector3::fromArray(CA.getValue());
                        Z.normalize();

                        // the center is halfway between the min and max height
                        float midHeight = (minHeight + maxHeight) / 2;
                        CCVector3 C = CCVector3::fromArray(
                                (CC + CA * midHeight).getValue());

                        // radial axis
                        CCVector3 X = CCVector3::fromArray(
                                (maxP - (CC + maxHeight * CA)).getValue());
                        X.normalize();

                        // orthogonal radial axis
                        CCVector3 Y = Z * X;

                        // we build the transformation matrix from these vectors
                        ccGLMatrix glMat(X, Y, Z, C);

                        // eventually create the cone primitive
                        prim = std::make_shared<ccCone>(maxRadius, minRadius,
                                                        maxHeight - minHeight,
                                                        0, 0, &glMat);
                        prim->setEnabled(true);
                        prim->setName(
                                QString("Cone (alpha=%1/h=%2)")
                                        .arg(alpha, 0, 'f')
                                        .arg(maxHeight - minHeight, 0, 'f'));
                        coneCount++;
                    }

                } break;

                case geometry::RansacParams::RPT_TORUS:  // torus
                {
                    const TorusPrimitiveShape* torus =
                            static_cast<const TorusPrimitiveShape*>(shape);
                    if (torus->Internal().IsAppleShaped()) {
                        utility::LogWarning(
                                "[ccPointCloud::ExecuteRANSAC] Apple-shaped "
                                "torus are not handled by CloudViewer!");
                    } else {
                        Vec3f CC = torus->Internal().Center();
                        Vec3f CA = torus->Internal().AxisDirection();
                        float minRadius = torus->Internal().MinorRadius();
                        float maxRadius = torus->Internal().MajorRadius();

                        if (minRadius < params.minRadius ||
                            maxRadius > params.maxRadius) {
                            count -= shapePointsCount;
                            continue;
                        }

                        CCVector3 Z = CCVector3::fromArray(CA.getValue());
                        CCVector3 C = CCVector3::fromArray(CC.getValue());
                        // construct remaining of base
                        CCVector3 X = Z.orthogonal();
                        CCVector3 Y = Z * X;

                        // we build matrix from these vectors
                        ccGLMatrix glMat(X, Y, Z, C);

                        // torus primitive
                        prim = std::make_shared<ccTorus>(
                                maxRadius - minRadius, maxRadius + minRadius,
                                M_PI * 2.0, false, 0, &glMat);
                        prim->setEnabled(true);
                        prim->setName(QString("Torus (r=%1/R=%2)")
                                              .arg(minRadius, 0, 'f')
                                              .arg(maxRadius, 0, 'f'));
                        torusCount++;
                    }

                } break;
            }

            // is there a primitive to add to part cloud?
            if (prim) {
                geometry::RansacResult result;
                // for solving phongShader bugs
                prim->clearTriNormals();
                prim->applyGLTransformation_recursive();
                prim->setVisible(true);
                if (params.randomColor) {
                    ecvColor::Rgb col = ecvColor::Generator::Random();
                    prim->setColor(col);
                    prim->showColors(true);
                }
                result.primitive = prim;

                // new cloud for sub-part
                result.indices.resize(shapePointsCount);
                {
                    for (unsigned j = 0; j < shapePointsCount; ++j) {
                        result.indices[j] = static_cast<size_t>(
                                cloud[shapeCloudIndex - j].index);
                    }
                }
                group.push_back(result);
            }

            count -= shapePointsCount;
        }

        if (print_progress) {
            utility::LogDebug(
                    "[ccPointCloud::ExecuteRANSAC] Detect Plane Shape number: "
                    "{:d}",
                    planeCount - 1);
            utility::LogDebug(
                    "[ccPointCloud::ExecuteRANSAC] Detect Sphere Shape number: "
                    "{:d}",
                    sphereCount - 1);
            utility::LogDebug(
                    "[ccPointCloud::ExecuteRANSAC] Detect Cylinder Shape "
                    "number: {:d}",
                    cylinderCount - 1);
            utility::LogDebug(
                    "[ccPointCloud::ExecuteRANSAC] Detect Cone Shape number: "
                    "{:d}",
                    coneCount - 1);
            utility::LogDebug(
                    "[ccPointCloud::ExecuteRANSAC] Detect Torus Shape number: "
                    "{:d}",
                    torusCount - 1);
        }
    }

    if (print_progress) {
        utility::LogDebug(
                "[ccPointCloud::ExecuteRANSAC] Total Shape Detection "
                "Instances: {:d}",
                group.size());
    }

    return group;
}
#endif

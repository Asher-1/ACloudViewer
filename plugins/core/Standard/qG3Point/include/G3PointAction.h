// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <AnglesCustomPlot.h>
#include <DgmOctree.h>
#include <G3PointDialog.h>
#include <G3PointPlots.h>
#include <GrainsAsEllipsoids.h>
#include <Neighbourhood.h>
#include <ecvOctree.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#include <QObject>
#include <nanoflann.hpp>
#include <vector>

#include "Eigen/Dense"

class ecvMainAppInterface;

namespace G3Point {
class G3PointAction : public QObject {
    Q_OBJECT

    typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> XXb;
    typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> Xb;

public:
    explicit G3PointAction(ccPointCloud* cloud,
                           ecvMainAppInterface* app = nullptr);
    ~G3PointAction();
    static void createAction(ecvMainAppInterface* appInterface);
    static void GetG3PointAction(ccPointCloud* cloud,
                                 ecvMainAppInterface* app = nullptr);
    void segment();
    void clusterAndOrClean();
    void getBorders();
    bool cluster();
    void fit();
    void exportResults();
    void plots();
    void showWolman(const Eigen::ArrayXf& d_sample,
                    const Eigen::Array3d& dq_final,
                    const Eigen::Array3d& edq);
    bool wolman();
    bool angles();
    bool processNewStacks(std::vector<std::vector<int>>& newStacks,
                          int pointCount);
    bool buildStacksFromG3PointLabelSF(cloudViewer::ScalarField* g3PointLabel);
    bool merge(XXb& condition);
    bool keep(Xb& condition);
    bool cleanLabels();
    void clean();

    template <typename T>
    static bool EigenArrayToFile(QString name, T array);

    // A small adaptor to let nanoflann access ccPointCloud data
    struct CloudAdaptor {
        const ccPointCloud* cloud;

        CloudAdaptor(const ccPointCloud* c) : cloud(c) {}

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const { return cloud->size(); }

        // Returns the dim'th component of the idx'th point
        inline float kdtree_get_pt(const size_t idx, int dim) const {
            if (dim == 0)
                return cloud->getPoint(static_cast<unsigned>(idx))->x;
            else if (dim == 1)
                return cloud->getPoint(static_cast<unsigned>(idx))->y;
            else
                return cloud->getPoint(static_cast<unsigned>(idx))->z;
        }

        // Optional bounding-box computation: return false to default to a
        // standard bbox computation loop.
        template <class BBOX>
        bool kdtree_get_bbox(BBOX&) const {
            return false;
        }
    };

    // Typedef for a 3D KD-tree index
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, CloudAdaptor>,
            CloudAdaptor,
            3 /* dim */>;

private:
    bool sfConvertToRandomRGB(const ccHObject::Container& selectedEntities,
                              QWidget* parent);
    void addToStack(int index,
                    const Eigen::ArrayXi& n_donors,
                    const Eigen::ArrayXXi& donors,
                    std::vector<int>& stack);
    int segmentLabels(bool useParallelStrategy = true);
    double angleRot2VecMat(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
    Eigen::ArrayXXd computeMeanAngleBetweenNormalsAtBorders();
    bool exportLocalMaximaAsCloud(const Eigen::ArrayXi& localMaximumIndexes);
    bool updateLocalMaximumIndexes();
    bool updateLabelsAndColors();
    bool checkStacks(const std::vector<std::vector<int>>& stacks, int count);
    void addToStackBraunWillett(int index,
                                const Eigen::ArrayXi& delta,
                                const Eigen::ArrayXi& Di,
                                std::vector<int>& stack,
                                int local_maximum);
    int segmentLabelsBraunWillett();
    void getNeighborsDistancesSlopes(unsigned index,
                                     std::vector<char>& duplicates);
    void computeNodeSurfaces();
    bool computeNormalsAndOrientThem();
    void orientNormals(const Eigen::Vector3d& sensorCenter);
    bool findNearestNeighborsNanoFlann(const unsigned int globalIndex,
                                       cloudViewer::ReferenceCloud* points,
                                       const KDTree* kdTree);
    bool computeNormWithFlann(unsigned int index,
                              NormsTableType* theNorms,
                              const KDTree* kdTree);
    bool computeNormals();
    bool queryNeighbors(ccPointCloud* cloud,
                        ecvMainAppInterface* appInterface,
                        bool useParallelStrategy = true);
    void init();
    void showDlg();
    void resetDlg();
    bool setCloud(ccPointCloud* cloud);
    void setKNN();

    int m_kNN = 20;
    double m_radiusFactor = 0.6;
    double m_maxAngle1 = 60;
    double m_maxAngle2 = 10;
    int m_nMin = 50;
    double m_minFlatness = 0.1;

    ccPointCloud* m_cloud;
    ecvMainAppInterface* m_app;
    G3PointDialog* m_dlg;

    Eigen::ArrayXXi m_neighborsIndexes;
    Eigen::ArrayXXd m_neighborsDistances;
    Eigen::ArrayXXd m_neighborsSlopes;
    Eigen::ArrayXXd m_normals;

    Eigen::ArrayXi m_initial_labels;
    Eigen::ArrayXi m_initial_labelsnpoint;
    Eigen::ArrayXi m_initial_localMaximumIndexes;
    Eigen::ArrayXi m_labels;
    Eigen::ArrayXi m_labelsnpoint;
    Eigen::ArrayXi m_localMaximumIndexes;
    Eigen::ArrayXi m_ndon;
    Eigen::ArrayXd m_area;

    QSharedPointer<RGBAColorsTableType> m_grainColors;

    std::vector<std::vector<int>> m_initialStacks;
    std::vector<std::vector<int>> m_stacks;

    ccOctree::Shared m_octree;
    unsigned char m_bestOctreeLevel = 0;
    cloudViewer::DgmOctree::NearestNeighboursSearchStruct m_nNSS;

    static std::shared_ptr<G3PointAction> s_g3PointAction;

    static QPointer<G3PointPlots> s_g3PointPlots;

    GrainsAsEllipsoids* m_grainsAsEllipsoids;

    int m_currentNumberOfSteps;
};
}  // namespace G3Point

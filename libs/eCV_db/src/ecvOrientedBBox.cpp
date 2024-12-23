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
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

// LOCAL
#include "ecvOrientedBBox.h"

#include "ecvDisplayTools.h"
#include "ecvGLMatrix.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"
#include "ecvQhull.h"

// CV_CORE_LIB
#include <GeometricalAnalysisTools.h>

// EIGEN
#include <Eigen/Eigenvalues>

// SYSTEM
#include <numeric>

void ecvOrientedBBox::draw(CC_DRAW_CONTEXT& context, const ecvColor::Rgb& col) {
    if (!ecvDisplayTools::GetMainWindow()) {
        return;
    }

    context.viewID = QString("BBox-") + context.viewID;
    setColor(ecvColor::Rgb::ToEigen(col));
    ecvDisplayTools::DrawOrientedBBox(context, this);
}

Eigen::Vector3d ecvOrientedBBox::getMinBound() const {
    auto points = getBoxPoints();
    return ComputeMinBound(points);
}

Eigen::Vector3d ecvOrientedBBox::getMaxBound() const {
    auto points = getBoxPoints();
    return ComputeMaxBound(points);
}

Eigen::Vector3d ecvOrientedBBox::getGeometryCenter() const { return center_; }

ccBBox ecvOrientedBBox::getOwnBB(bool withGLFeatures) {
    return getAxisAlignedBoundingBox();
}

ccBBox ecvOrientedBBox::getAxisAlignedBoundingBox() const {
    return ccBBox::CreateFromPoints(getBoxPoints());
}

ecvOrientedBBox ecvOrientedBBox::getOrientedBoundingBox() const {
    return *this;
}

ecvOrientedBBox& ecvOrientedBBox::transform(
        const Eigen::Matrix4d& transformation) {
    const Eigen::Matrix3d rotation = transformation.block<3, 3>(0, 0);
    const Eigen::Vector3d translation = transformation.block<3, 1>(0, 3);
    this->rotate(rotation, Eigen::Vector3d(0.0, 0.0, 0.0));
    this->translate(translation, true);
    return *this;
}

ecvOrientedBBox& ecvOrientedBBox::translate(const Eigen::Vector3d& translation,
                                            bool relative) {
    if (relative) {
        center_ += translation;
    } else {
        center_ = translation;
    }
    return *this;
}

ecvOrientedBBox& ecvOrientedBBox::scale(const double scale,
                                        const Eigen::Vector3d& center) {
    extent_ *= scale;
    center_ = scale * (center_ - center) + center;
    return *this;
}

ecvOrientedBBox& ecvOrientedBBox::rotate(const Eigen::Matrix3d& R,
                                         const Eigen::Vector3d& center) {
    R_ = R * R_;
    center_ = R * (center_ - center) + center;
    return *this;
}

const ecvOrientedBBox ecvOrientedBBox::operator*(const ccGLMatrix& mat) {
    ecvOrientedBBox rotatedBox(*this);
    rotatedBox.rotate(ccGLMatrixd::ToEigenMatrix3(mat),
                      Eigen::Vector3d(0.0, 0.0, 0.0));
    rotatedBox.translate(CCVector3d::fromArray(mat.getTranslationAsVec3D()),
                         true);
    return rotatedBox;
}

const ecvOrientedBBox ecvOrientedBBox::operator*(const ccGLMatrixd& mat) {
    ecvOrientedBBox rotatedBox(*this);
    rotatedBox.rotate(ccGLMatrixd::ToEigenMatrix3(mat),
                      Eigen::Vector3d(0.0, 0.0, 0.0));
    rotatedBox.translate(CCVector3d::fromArray(mat.getTranslationAsVec3D()),
                         true);
    return rotatedBox;
}

ecvOrientedBBox ecvOrientedBBox::CreateFromPoints(
        const std::vector<Eigen::Vector3d>& points) {
    return CreateFromPoints(CCVector3::fromArrayContainer(points));
}

ecvOrientedBBox ecvOrientedBBox::CreateFromPoints(
        const std::vector<CCVector3>& points) {
    auto mesh =
            std::get<0>(cloudViewer::utility::Qhull::ComputeConvexHull(points));
    ccGenericPointCloud* hull_pcd = mesh->getAssociatedCloud();
    assert(hull_pcd);

    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;
    std::tie(mean, cov) = hull_pcd->computeMeanAndCovariance();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d evals = es.eigenvalues();
    Eigen::Matrix3d R = es.eigenvectors();
    R.col(0) /= R.col(0).norm();
    R.col(1) /= R.col(1).norm();
    R.col(2) /= R.col(2).norm();

    if (evals(1) > evals(0)) {
        std::swap(evals(1), evals(0));
        Eigen::Vector3d tmp = R.col(1);
        R.col(1) = R.col(0);
        R.col(0) = tmp;
    }
    if (evals(2) > evals(0)) {
        std::swap(evals(2), evals(0));
        Eigen::Vector3d tmp = R.col(2);
        R.col(2) = R.col(0);
        R.col(0) = tmp;
    }
    if (evals(2) > evals(1)) {
        std::swap(evals(2), evals(1));
        Eigen::Vector3d tmp = R.col(2);
        R.col(2) = R.col(1);
        R.col(1) = tmp;
    }

    hull_pcd->placeIteratorAtBeginning();
    CCVector3* pt = nullptr;
    while ((pt = const_cast<CCVector3*>(hull_pcd->getNextPoint()))) {
        Eigen::Vector3d vc = CCVector3d::fromArray(*pt - mean);
        Eigen::Vector3d data = R.transpose() * vc;
        *pt = data;
    }

    const auto aabox = hull_pcd->getAxisAlignedBoundingBox();

    ecvOrientedBBox obox;
    obox.center_ = R * aabox.getGeometryCenter() + mean;
    obox.R_ = R;
    obox.extent_ = aabox.getExtent();

    return obox;
}

ecvOrientedBBox ecvOrientedBBox::CreateFromAxisAlignedBoundingBox(
        const ccBBox& aabox) {
    ecvOrientedBBox obox;
    obox.center_ = aabox.getGeometryCenter();
    obox.extent_ = aabox.getExtent();
    obox.R_ = Eigen::Matrix3d::Identity();
    return obox;
}

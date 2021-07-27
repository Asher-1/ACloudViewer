// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <Logging.h>
#include <Parallel.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>

#include "ecvHObjectCaster.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"

using namespace cloudViewer;

std::shared_ptr<ccMesh> ccMesh::deformAsRigidAsPossible(
        const std::vector<int>& constraint_vertex_indices,
        const std::vector<Eigen::Vector3d>& constraint_vertex_positions,
        size_t max_iter,
        DeformAsRigidAsPossibleEnergy energy_model,
        double smoothed_alpha) const {
    ccPointCloud* baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    auto prime = cloudViewer::make_shared<ccMesh>(baseVertices);
    prime->addChild(baseVertices);
    baseVertices->addPoints(this->getVertices());
    prime->addTriangles(this->getTriangles());

    utility::LogDebug("[DeformAsRigidAsPossible] setting up S'");
    prime->computeAdjacencyList();
    auto edges_to_vertices = prime->getEdgeToVerticesMap();
    auto edge_weights =
            prime->computeEdgeWeightsCot(edges_to_vertices, /*min_weight=*/0);
    utility::LogDebug("[DeformAsRigidAsPossible] done setting up S'");

    std::unordered_map<int, Eigen::Vector3d> constraints;
    for (size_t idx = 0; idx < constraint_vertex_indices.size() &&
                         idx < constraint_vertex_positions.size();
         ++idx) {
        constraints[constraint_vertex_indices[idx]] =
                constraint_vertex_positions[idx];
    }

    double surface_area = -1;
    // std::vector<Eigen::Matrix3d> Rs(vertices_.size(),
    // Eigen::Matrix3d::Identity());
    std::vector<Eigen::Matrix3d> Rs(getVerticeSize());
    std::vector<Eigen::Matrix3d> Rs_old;
    if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed) {
        surface_area = prime->getSurfaceArea();
        Rs_old.resize(getVerticeSize());
    }

    // Build system matrix L and its solver
    utility::LogDebug("[DeformAsRigidAsPossible] setting up system matrix L");
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < int(getVerticeSize()); ++i) {
        if (constraints.count(i) > 0) {
            triplets.push_back(Eigen::Triplet<double>(i, i, 1));
        } else {
            double W = 0;
            for (int j : prime->adjacency_list_[i]) {
                double w = edge_weights[GetOrderedEdge(i, j)];
                triplets.push_back(Eigen::Triplet<double>(i, j, -w));
                W += w;
            }
            if (W > 0) {
                triplets.push_back(Eigen::Triplet<double>(i, i, W));
            }
        }
    }
    Eigen::SparseMatrix<double> L(getVerticeSize(), getVerticeSize());
    L.setFromTriplets(triplets.begin(), triplets.end());
    utility::LogDebug(
            "[DeformAsRigidAsPossible] done setting up system matrix L");

    utility::LogDebug("[DeformAsRigidAsPossible] setting up sparse solver");
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(L);
    solver.factorize(L);
    if (solver.info() != Eigen::Success) {
        utility::LogError(
                "[DeformAsRigidAsPossible] Failed to build solver (factorize)");
    } else {
        utility::LogDebug(
                "[DeformAsRigidAsPossible] done setting up sparse solver");
    }

    std::vector<Eigen::VectorXd> b = {Eigen::VectorXd(getVerticeSize()),
                                      Eigen::VectorXd(getVerticeSize()),
                                      Eigen::VectorXd(getVerticeSize())};

    for (size_t iter = 0; iter < max_iter; ++iter) {
        if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed) {
            std::swap(Rs, Rs_old);
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
#endif
        for (int i = 0; i < int(getVerticeSize()); ++i) {
            // Update rotations
            Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
            Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
            int n_nbs = 0;
            for (int j : prime->adjacency_list_[i]) {
                Eigen::Vector3d e0 = getVertice(static_cast<size_t>(i)) -
                                     getVertice(static_cast<size_t>(j));
                Eigen::Vector3d e1 = prime->getVertice(static_cast<size_t>(i)) -
                                     prime->getVertice(static_cast<size_t>(j));
                double w = edge_weights[GetOrderedEdge(i, j)];
                S += w * (e0 * e1.transpose());
                if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed) {
                    R += Rs_old[j];
                }
                n_nbs++;
            }
            if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed &&
                iter > 0 && n_nbs > 0) {
                S = 2 * S +
                    (4 * smoothed_alpha * surface_area / n_nbs) * R.transpose();
            }
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                    S, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();
            Eigen::Vector3d D(1, 1, (V * U.transpose()).determinant());
            // ensure rotation:
            // http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
            Rs[i] = V * D.asDiagonal() * U.transpose();
            if (Rs[i].determinant() <= 0) {
                utility::LogError(
                        "[DeformAsRigidAsPossible] something went wrong with "
                        "updateing R");
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
#endif
        for (int i = 0; i < int(getVerticeSize()); ++i) {
            // Update Positions
            Eigen::Vector3d bi(0, 0, 0);
            if (constraints.count(i) > 0) {
                bi = constraints[i];
            } else {
                for (int j : prime->adjacency_list_[i]) {
                    double w = edge_weights[GetOrderedEdge(i, j)];
                    bi += w / 2 *
                          ((Rs[i] + Rs[j]) *
                           (getVertice(static_cast<size_t>(i)) -
                            getVertice(static_cast<size_t>(j))));
                }
            }
            b[0](i) = bi(0);
            b[1](i) = bi(1);
            b[2](i) = bi(2);
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
#endif
        for (int comp = 0; comp < 3; ++comp) {
            Eigen::VectorXd p_prime = solver.solve(b[comp]);
            if (solver.info() != Eigen::Success) {
                utility::LogError(
                        "[DeformAsRigidAsPossible] Cholesky solve failed");
            }
            for (int i = 0; i < int(getVerticeSize()); ++i) {
                baseVertices->getPointPtr(static_cast<size_t>(i))->u[comp] =
                        static_cast<PointCoordinateType>(p_prime(i));
            }
        }

        // Compute energy and log
        double energy = 0;
        double reg = 0;
        for (int i = 0; i < int(getVerticeSize()); ++i) {
            for (int j : prime->adjacency_list_[i]) {
                double w = edge_weights[GetOrderedEdge(i, j)];
                Eigen::Vector3d e0 = getVertice(static_cast<size_t>(i)) -
                                     getVertice(static_cast<size_t>(j));
                Eigen::Vector3d e1 = prime->getVertice(static_cast<size_t>(i)) -
                                     prime->getVertice(static_cast<size_t>(j));
                Eigen::Vector3d diff = e1 - Rs[i] * e0;
                energy += w * diff.squaredNorm();
                if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed) {
                    reg += (Rs[i] - Rs[j]).squaredNorm();
                }
            }
        }
        if (energy_model == DeformAsRigidAsPossibleEnergy::Smoothed) {
            energy = energy + smoothed_alpha * surface_area * reg;
        }
        utility::LogDebug("[DeformAsRigidAsPossible] iter={}, energy={:e}",
                          iter, energy);
    }

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        prime->shrinkToFit();
        NormsIndexesTableType* normals = prime->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }
    return prime;
}

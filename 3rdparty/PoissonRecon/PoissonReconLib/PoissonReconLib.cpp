// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Unified in-memory wrapper around the vendored PoissonRecon (Adaptive
// Multigrid Solvers v18.75, the upstream COLMAP vendored tree). Implemented on
// the header-defined Reconstructor::Implicit facade so this wrapper compiles
// against the same single source tree the CLI-style consumers use, and both
// in-memory consumers (libs/CV_db and the qPoissonRecon plugin) share one
// implementation instead of hand-wiring the FEMTree internals per module.

#include "PoissonReconLib.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

#include "PreProcessor.h"
#include "MyMiscellany.h"
#include "PPolynomial.h"
#include "FEMTree.h"
#include "Reconstructors.h"

// The v18.75 tree wraps everything in namespace PoissonRecon (the same
// using-directive upstream COLMAP's own PoissonRecon.cpp relies on).
using namespace PoissonRecon;

namespace {

// Every PoissonReconLib consumer reconstructs surfaces in 3D.
constexpr unsigned int kDim = 3;

// Translates the public boundary enum into the v18.75 boundary type.
BoundaryType ToFEMBoundary(PoissonReconLib::Parameters::BoundaryType boundary) {
    switch (boundary) {
        case PoissonReconLib::Parameters::FREE:
            return BOUNDARY_FREE;
        case PoissonReconLib::Parameters::DIRICHLET:
            return BOUNDARY_DIRICHLET;
        case PoissonReconLib::Parameters::NEUMANN:
        default:
            return BOUNDARY_NEUMANN;
    }
}

// In-memory oriented sample stream (position + normal) fed by the
// PoissonReconLib::ICloud interface. InputDataStream's default thread-aware
// read() serializes on an internal mutex, so a plain cursor suffices.
template <typename Real>
class CloudSampleStream : public Reconstructor::InputOrientedSampleStream<
                              Real, kDim> {
public:
    explicit CloudSampleStream(const PoissonReconLib::ICloud<Real>& cloud)
        : cloud_(cloud) {}

    void reset(void) override { index_ = 0; }

    bool read(Reconstructor::Position<Real, kDim>& p,
              Reconstructor::Normal<Real, kDim>& n) override {
        if (index_ >= cloud_.size()) {
            return false;
        }
        Real coords[kDim];
        Real normals[kDim];
        cloud_.getPoint(index_, coords);
        cloud_.getNormal(index_, normals);
        for (unsigned int k = 0; k < kDim; ++k) {
            p[k] = coords[k];
            n[k] = normals[k];
        }
        ++index_;
        return true;
    }

private:
    const PoissonReconLib::ICloud<Real>& cloud_;
    size_t index_ = 0;
};

// Same stream with the per-sample color auxiliary channel attached.
template <typename Real>
class ColoredCloudSampleStream
    : public Reconstructor::InputOrientedSampleStream<Real, kDim,
                                                      Point<Real, 3>> {
public:
    explicit ColoredCloudSampleStream(
            const PoissonReconLib::ICloud<Real>& cloud)
        : cloud_(cloud) {}

    void reset(void) override { index_ = 0; }

    bool read(Reconstructor::Position<Real, kDim>& p,
              Reconstructor::Normal<Real, kDim>& n,
              Point<Real, 3>& color) override {
        if (index_ >= cloud_.size()) {
            return false;
        }
        Real coords[kDim];
        Real normals[kDim];
        Real rgb[3] = {0, 0, 0};
        cloud_.getPoint(index_, coords);
        cloud_.getNormal(index_, normals);
        cloud_.getColor(index_, rgb);
        for (unsigned int k = 0; k < kDim; ++k) {
            p[k] = coords[k];
            n[k] = normals[k];
            color[k] = rgb[k];
        }
        ++index_;
        return true;
    }

private:
    const PoissonReconLib::ICloud<Real>& cloud_;
    size_t index_ = 0;
};

// In-memory output vertex stream forwarding each extracted level-set vertex
// (position/gradient/density/color) to the IMesh interface. WithColors=false
// drops the aux slot from the stream signature entirely (partial
// specializations, since the stream type is part of the facade contract). The
// old wrapper never emitted normals (addNormal was commented out upstream), so
// gradients are extracted but discarded unless a future consumer asks for
// them.
template <typename Real, bool WithColors>
class MeshVertexStream;

template <typename Real>
class MeshVertexStream<Real, false>
    : public Reconstructor::OutputLevelSetVertexStream<Real, kDim> {
public:
    MeshVertexStream(PoissonReconLib::IMesh<Real>& mesh, bool outputDensity)
        : mesh_(mesh), outputDensity_(outputDensity) {}

    size_t size(void) const override { return count_; }

    size_t write(const Reconstructor::Position<Real, kDim>& p,
                 const Reconstructor::Gradient<Real, kDim>&,
                 const Reconstructor::Weight<Real>& w) override {
        Real coords[kDim];
        for (unsigned int k = 0; k < kDim; ++k) {
            coords[k] = p[k];
        }
        mesh_.addVertex(coords);
        if (outputDensity_) {
            mesh_.addDensity(static_cast<double>(w));
        }
        ++count_;
        return count_;
    }

protected:
    PoissonReconLib::IMesh<Real>& mesh_;
    bool outputDensity_ = false;
    size_t count_ = 0;
};

template <typename Real>
class MeshVertexStream<Real, true>
    : public Reconstructor::OutputLevelSetVertexStream<Real, kDim,
                                                       Point<Real, 3>> {
public:
    MeshVertexStream(PoissonReconLib::IMesh<Real>& mesh, bool outputDensity)
        : mesh_(mesh), outputDensity_(outputDensity) {}

    size_t size(void) const override { return count_; }

    size_t write(const Reconstructor::Position<Real, kDim>& p,
                 const Reconstructor::Gradient<Real, kDim>&,
                 const Reconstructor::Weight<Real>& w,
                 const Point<Real, 3>& color) override {
        Real coords[kDim];
        for (unsigned int k = 0; k < kDim; ++k) {
            coords[k] = p[k];
        }
        mesh_.addVertex(coords);
        if (outputDensity_) {
            mesh_.addDensity(static_cast<double>(w));
        }
        Real rgb[3];
        for (unsigned int k = 0; k < kDim; ++k) {
            rgb[k] = color[k];
        }
        mesh_.addColor(rgb);
        ++count_;
        return count_;
    }

protected:
    PoissonReconLib::IMesh<Real>& mesh_;
    bool outputDensity_ = false;
    size_t count_ = 0;
};

// In-memory output face stream collecting the extracted triangles.
template <typename Real>
class MeshFaceStream : public Reconstructor::OutputFaceStream<2> {
public:
    explicit MeshFaceStream(PoissonReconLib::IMesh<Real>& mesh)
        : mesh_(mesh) {}

    size_t size(void) const override { return count_; }

    size_t write(const Reconstructor::Face<2>& face) override {
        mesh_.addTriangle(static_cast<size_t>(face[0]),
                          static_cast<size_t>(face[1]),
                          static_cast<size_t>(face[2]));
        ++count_;
        return count_;
    }

private:
    PoissonReconLib::IMesh<Real>& mesh_;
    size_t count_ = 0;
};

// The reconstruction pipeline shared by the float and double instantiations.
// WithColors selects whether the color auxiliary channel takes part in the
// solve and extraction signatures.
template <typename Real, bool WithColors, BoundaryType BType>
bool ReconstructImpl(const PoissonReconLib::Parameters& params,
                     const PoissonReconLib::ICloud<Real>& inCloud,
                     PoissonReconLib::IMesh<Real>& outMesh) {
    using FEMSigs = IsotropicUIntPack<
            kDim,
            FEMDegreeAndBType<Reconstructor::Poisson::DefaultFEMDegree,
                              BType>::Signature>;

    typename Reconstructor::Poisson::SolutionParameters<Real> sParams;
    Reconstructor::LevelSetExtractionParameters meParams;

    sParams.verbose = false;
    // v18.75 reduced confidence to a boolean flag (the |normal|-as-weight
    // interpretation); normalConfidenceBias has no facade equivalent and is
    // treated as enabling the same flag.
    sParams.confidence = params.normalConfidence > 0 ||
                         params.normalConfidenceBias > 0;
    sParams.scale = static_cast<Real>(params.scale);
    sParams.width = static_cast<Real>(params.finestCellWidth);
    sParams.samplesPerNode = static_cast<Real>(params.samplesPerNode);
    sParams.pointWeight = static_cast<Real>(params.pointWeight);
    sParams.cgSolverAccuracy = static_cast<Real>(params.cgAccuracy);
    sParams.perLevelDataScaleFactor =
            static_cast<Real>(params.colorPullFactor);
    sParams.depth = static_cast<unsigned int>(params.depth);
    sParams.fullDepth = static_cast<unsigned int>(params.fullDepth);
    sParams.baseDepth = static_cast<unsigned int>(params.baseDepth);
    sParams.baseVCycles = static_cast<unsigned int>(params.baseVCycles);
    sParams.iters = static_cast<unsigned int>(params.iters);

    meParams.linearFit = params.linearFit;
    meParams.forceManifold = true;
    meParams.outputGradients = false;
    meParams.outputDensity = params.density;

    std::unique_ptr<Reconstructor::Implicit<Real, kDim, FEMSigs>> implicitNoAux;
    std::unique_ptr<Reconstructor::Implicit<Real, kDim, FEMSigs, Point<Real, 3>>>
            implicitAux;

    if constexpr (WithColors) {
        ColoredCloudSampleStream<Real> sampleStream(inCloud);
        implicitAux.reset(
                Reconstructor::Poisson::Solver<Real, kDim, FEMSigs,
                                               Point<Real, 3>>::Solve(
                        sampleStream, sParams, Point<Real, 3>(0, 0, 0)));
        if (!implicitAux) {
            return false;
        }

        MeshVertexStream<Real, true> vertexStream(outMesh, params.density);
        MeshFaceStream<Real> faceStream(outMesh);
        Reconstructor::TransformedOutputLevelSetVertexStream<Real, kDim,
                                                             Point<Real, 3>>
                transformedVertex(implicitAux->unitCubeToModel, vertexStream);
        implicitAux->extractLevelSet(transformedVertex, faceStream, meParams);
    } else {
        CloudSampleStream<Real> sampleStream(inCloud);
        implicitNoAux.reset(Reconstructor::Poisson::Solver<Real, kDim, FEMSigs>::
                            Solve(sampleStream, sParams));
        if (!implicitNoAux) {
            return false;
        }

        MeshVertexStream<Real, false> vertexStream(outMesh, params.density);
        MeshFaceStream<Real> faceStream(outMesh);
        Reconstructor::TransformedOutputLevelSetVertexStream<Real, kDim>
                transformedVertex(implicitNoAux->unitCubeToModel,
                                  vertexStream);
        implicitNoAux->extractLevelSet(transformedVertex, faceStream,
                                       meParams);
    }
    return true;
}

template <typename Real, bool WithColors>
bool ReconstructWithBoundary(const PoissonReconLib::Parameters& params,
                             const PoissonReconLib::ICloud<Real>& inCloud,
                             PoissonReconLib::IMesh<Real>& outMesh) {
    switch (params.boundary) {
        case PoissonReconLib::Parameters::FREE:
            return ReconstructImpl<Real, WithColors, BOUNDARY_FREE>(
                    params, inCloud, outMesh);
        case PoissonReconLib::Parameters::DIRICHLET:
            return ReconstructImpl<Real, WithColors, BOUNDARY_DIRICHLET>(
                    params, inCloud, outMesh);
        case PoissonReconLib::Parameters::NEUMANN:
        default:
            return ReconstructImpl<Real, WithColors, BOUNDARY_NEUMANN>(
                    params, inCloud, outMesh);
    }
}

template <typename Real>
bool ReconstructDispatch(const PoissonReconLib::Parameters& params,
                         const PoissonReconLib::ICloud<Real>& inCloud,
                         PoissonReconLib::IMesh<Real>& outMesh) {
    if (params.depth < 2) {
        // The facade derives kernelDepth = depth - 2; shallower depths cannot
        // run (mirrors the previous wrapper's guard).
        return false;
    }
    if (params.withColors && inCloud.hasColors()) {
        return ReconstructWithBoundary<Real, true>(params, inCloud, outMesh);
    }
    return ReconstructWithBoundary<Real, false>(params, inCloud, outMesh);
}

}  // namespace

int PoissonReconLib::Parameters::GetMaxThreadCount() {
    const unsigned int hardware = std::thread::hardware_concurrency();
    return hardware > 0 ? static_cast<int>(hardware) : 1;
}

PoissonReconLib::Parameters::Parameters() : threads(GetMaxThreadCount()) {}

bool PoissonReconLib::Reconstruct(const Parameters& params,
                                  const PoissonReconLib::ICloud<float>& inCloud,
                                  PoissonReconLib::IMesh<float>& ouMesh) {
    return ReconstructDispatch<float>(params, inCloud, ouMesh);
}

bool PoissonReconLib::Reconstruct(
        const Parameters& params, const PoissonReconLib::ICloud<double>& inCloud,
        PoissonReconLib::IMesh<double>& ouMesh) {
    return ReconstructDispatch<double>(params, inCloud, ouMesh);
}

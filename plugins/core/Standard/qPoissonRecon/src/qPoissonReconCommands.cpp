// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qPoissonReconCommands.h"

#include <PoissonReconLib.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#include <QObject>
#include <algorithm>
#include <cassert>

static const char COMMAND_POISSON_RECON[] = "POISSON_RECON";
static const char COMMAND_PR_DEPTH[] = "DEPTH";
static const char COMMAND_PR_SAMPLES_PER_NODE[] = "SAMPLES_PER_NODE";
static const char COMMAND_PR_POINT_WEIGHT[] = "POINT_WEIGHT";
static const char COMMAND_PR_BOUNDARY[] = "BOUNDARY";
static const char COMMAND_PR_LINEAR_FIT[] = "LINEAR_FIT";
static const char COMMAND_PR_WITH_COLORS[] = "WITH_COLORS";
static const char COMMAND_PR_DENSITY[] = "DENSITY";

template <typename Real>
class PointCloudWrapper : public PoissonReconLib::ICloud<Real> {
public:
    explicit PointCloudWrapper(const ccPointCloud& cloud) : m_cloud(cloud) {}

    size_t size() const override { return m_cloud.size(); }
    bool hasNormals() const override { return m_cloud.hasNormals(); }
    bool hasColors() const override { return m_cloud.hasColors(); }
    void getPoint(size_t index, Real* coords) const override {
        if (index >= m_cloud.size()) {
            assert(false);
            return;
        }
        const CCVector3* P = m_cloud.getPoint(static_cast<unsigned>(index));
        coords[0] = static_cast<Real>(P->x);
        coords[1] = static_cast<Real>(P->y);
        coords[2] = static_cast<Real>(P->z);
    }

    void getNormal(size_t index, Real* coords) const override {
        if (index >= m_cloud.size() || !m_cloud.hasNormals()) {
            assert(false);
            return;
        }
        const CCVector3& N =
                m_cloud.getPointNormal(static_cast<unsigned>(index));
        coords[0] = static_cast<Real>(N.x);
        coords[1] = static_cast<Real>(N.y);
        coords[2] = static_cast<Real>(N.z);
    }

    void getColor(size_t index, Real* rgb) const override {
        if (index >= m_cloud.size() || !m_cloud.hasColors()) {
            assert(false);
            return;
        }
        const ecvColor::Rgb& color =
                m_cloud.getPointColor(static_cast<unsigned>(index));
        rgb[0] = static_cast<Real>(color.r);
        rgb[1] = static_cast<Real>(color.g);
        rgb[2] = static_cast<Real>(color.b);
    }

protected:
    const ccPointCloud& m_cloud;
};

template <typename Real>
class MeshWrapper : public PoissonReconLib::IMesh<Real> {
public:
    explicit MeshWrapper(ccMesh& mesh,
                         ccPointCloud& vertices,
                         cloudViewer::ScalarField* densitySF = nullptr)
        : m_mesh(mesh),
          m_vertices(vertices),
          m_densitySF(densitySF),
          m_error(false) {}

    bool checkMeshCapacity() {
        if (m_error) {
            return false;
        }
        if (m_mesh.size() == m_mesh.capacity() &&
            !m_mesh.reserve(m_mesh.size() + 1024)) {
            m_error = true;
            return false;
        }
        return true;
    }

    bool checkVertexCapacity() {
        if (m_error) {
            return false;
        }
        if (m_vertices.size() == m_vertices.capacity() &&
            !m_vertices.reserve(m_vertices.size() + 4096)) {
            m_error = true;
            return false;
        }
        return true;
    }

    void addVertex(const Real* coords) override {
        if (!checkVertexCapacity()) {
            return;
        }
        CCVector3 P = CCVector3::fromArray(coords);
        m_vertices.addPoint(P);
    }

    void addNormal(const Real* coords) override {
        if (!checkVertexCapacity()) {
            return;
        }
        if (!m_vertices.hasNormals() && !m_vertices.reserveTheNormsTable()) {
            m_error = true;
            return;
        }
        CCVector3 N = CCVector3::fromArray(coords);
        m_vertices.addNorm(N);
    }

    void addColor(const Real* rgb) override {
        if (!checkVertexCapacity()) {
            return;
        }
        if (!m_vertices.hasColors()) {
            if (!m_vertices.reserveTheRGBTable()) {
                m_error = true;
                return;
            }
        }
        m_vertices.addRGBColor(static_cast<ColorCompType>(std::min(
                                       (Real)255, std::max((Real)0, rgb[0]))),
                               static_cast<ColorCompType>(std::min(
                                       (Real)255, std::max((Real)0, rgb[1]))),
                               static_cast<ColorCompType>(std::min(
                                       (Real)255, std::max((Real)0, rgb[2]))));
    }

    void addDensity(double d) override {
        if (!m_densitySF) {
            return;
        }
        if (m_densitySF->size() == m_densitySF->capacity() &&
            !m_densitySF->reserveSafe(m_densitySF->size() + 4096)) {
            m_error = true;
            return;
        }
        m_densitySF->addElement(static_cast<ScalarType>(d));
    }

    void addTriangle(size_t i1, size_t i2, size_t i3) override {
        if (!checkMeshCapacity()) {
            return;
        }
        m_mesh.addTriangle(static_cast<unsigned>(i1), static_cast<unsigned>(i2),
                           static_cast<unsigned>(i3));
    }

    bool isInErrorState() const { return m_error; }

protected:
    ccMesh& m_mesh;
    ccPointCloud& m_vertices;
    bool m_error;
    cloudViewer::ScalarField* m_densitySF;
};

CommandPoissonRecon::CommandPoissonRecon()
    : ccCommandLineInterface::Command("Poisson Recon", COMMAND_POISSON_RECON) {}

/**
 * @brief Processes the Poisson surface reconstruction command.
 *
 * This method reads the currently loaded point clouds from the command-line
 * interface, parses optional parameters (e.g., octree depth, samples per node,
 * point weight, boundary conditions, linear fit, color interpolation, and
 * density output), and then invokes the Poisson reconstruction library to
 * generate an output mesh (and optionally a density scalar field).
 *
 * The function reports any parsing or processing errors through @p cmd and
 * returns @c true on success and @c false if an error occurred.
 *
 * @param cmd Command-line interface providing input clouds, arguments and
 *            helper routines for reporting messages and errors.
 */
bool CommandPoissonRecon::process(ccCommandLineInterface& cmd) {
    // Print command header in the console to delimit this operation.
    cmd.print("[POISSON_RECON]");

    // Ensure that at least one point cloud is available before running
    // the reconstruction.
    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr("No point cloud loaded (use \"-O "
                                     "[filename]\" before \"-%1\")")
                                 .arg(COMMAND_POISSON_RECON));
    }

    // Default Poisson reconstruction parameters (can be overridden by
    // command-line options parsed below).
    int depth = 8;
    float samplesPerNode = 1.5f;
    float pointWeight = 2.0f;
    PoissonReconLib::Parameters::BoundaryType boundary =
            PoissonReconLib::Parameters::NEUMANN;
    bool linearFit = false;
    bool withColors = false;
    bool useDensity = false;

    // Parse remaining command-line arguments and update reconstruction
    // parameters accordingly (e.g., -DEPTH, -SAMPLES_PER_NODE, ...).
    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_PR_DEPTH)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_PR_DEPTH));
            bool ok;
            depth = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok || depth < 1) return cmd.error("Invalid value for -DEPTH");
        } else if (ccCommandLineInterface::IsCommand(
                           arg, COMMAND_PR_SAMPLES_PER_NODE)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_PR_SAMPLES_PER_NODE));
            bool ok;
            samplesPerNode = cmd.arguments().takeFirst().toFloat(&ok);
            if (!ok) return cmd.error("Invalid value for -SAMPLES_PER_NODE");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_PR_POINT_WEIGHT)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_PR_POINT_WEIGHT));
            bool ok;
            pointWeight = cmd.arguments().takeFirst().toFloat(&ok);
            if (!ok) return cmd.error("Invalid value for -POINT_WEIGHT");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_PR_BOUNDARY)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_PR_BOUNDARY));
            QString b = cmd.arguments().takeFirst().toUpper();
            if (b == "FREE")
                boundary = PoissonReconLib::Parameters::FREE;
            else if (b == "DIRICHLET")
                boundary = PoissonReconLib::Parameters::DIRICHLET;
            else if (b == "NEUMANN")
                boundary = PoissonReconLib::Parameters::NEUMANN;
            else
                return cmd.error(
                        "Invalid -BOUNDARY (use FREE, DIRICHLET, or NEUMANN)");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_PR_LINEAR_FIT)) {
            cmd.arguments().pop_front();
            linearFit = true;
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_PR_WITH_COLORS)) {
            cmd.arguments().pop_front();
            withColors = true;
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_PR_DENSITY)) {
            cmd.arguments().pop_front();
            useDensity = true;
        } else {
            break;
        }
    }

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* pc = desc.pc;
        if (!pc) continue;

        if (!pc->hasNormals()) {
            return cmd.error(QObject::tr("[POISSON_RECON] Cloud '%1' has no "
                                         "normals (compute normals "
                                         "on the cloud first)")
                                     .arg(pc->getName()));
        }

        PoissonReconLib::Parameters params;
        params.depth = depth;
        params.finestCellWidth = 0.0f;
        params.samplesPerNode = samplesPerNode;
        params.pointWeight = pointWeight;
        params.boundary = boundary;
        params.linearFit = linearFit;
        params.withColors = withColors && pc->hasColors();
        params.density = useDensity;

        cmd.print(
                QObject::tr("[POISSON_RECON] Processing cloud '%1' (%2 points)")
                        .arg(pc->getName())
                        .arg(pc->size()));

        ccScalarField* densitySF = nullptr;
        ccPointCloud* newPC = new ccPointCloud("vertices");
        ccMesh* newMesh = new ccMesh(newPC);
        newMesh->addChild(newPC);

        if (params.density) {
            densitySF = new ccScalarField("Density");
        }

        MeshWrapper<PointCoordinateType> meshWrapper(*newMesh, *newPC,
                                                     densitySF);
        PointCloudWrapper<PointCoordinateType> cloudWrapper(*pc);

        bool reconstructed = PoissonReconLib::Reconstruct(params, cloudWrapper,
                                                          meshWrapper) &&
                             !meshWrapper.isInErrorState();

        // Retry with single thread if multi-threaded attempt failed (known
        // race condition in PoissonRecon v12 IsoSurfaceExtractor).
        if (!reconstructed && params.threads > 1) {
            cmd.warning(
                    "[POISSON_RECON] Multi-threaded attempt failed, "
                    "retrying with single thread...");

            newPC->clear();
            newMesh->clear();
            if (densitySF) {
                densitySF->clear();
            }

            MeshWrapper<PointCoordinateType> retryWrapper(*newMesh, *newPC,
                                                          densitySF);
            PoissonReconLib::Parameters retryParams = params;
            retryParams.threads = 1;

            reconstructed = PoissonReconLib::Reconstruct(
                                    retryParams, cloudWrapper, retryWrapper) &&
                            !retryWrapper.isInErrorState();
        }

        if (!reconstructed) {
            if (densitySF) {
                densitySF->release();
            }
            delete newMesh;
            return cmd.error(
                    QObject::tr("[POISSON_RECON] Reconstruction failed "
                                "for cloud '%1'")
                            .arg(pc->getName()));
        }

        const bool cloudHasColors = pc->hasColors();
        newMesh->setName(QString("Mesh[%1] (level %2)")
                                 .arg(pc->getName())
                                 .arg(params.depth));
        newPC->setEnabled(false);
        newMesh->setVisible(true);
        newMesh->computeNormals(true);
        if (!cloudHasColors || !params.withColors) {
            newPC->unallocateColors();
            newPC->showColors(false);
        }
        newMesh->showColors(newPC->hasColors());

        if (densitySF) {
            densitySF->computeMinAndMax();
            densitySF->showNaNValuesInGrey(false);
            int sfIdx = newPC->addScalarField(densitySF);
            newPC->setCurrentDisplayedScalarField(sfIdx);
            newPC->showSF(true);
            newMesh->showColors(newPC->colorsShown());
            newMesh->showSF(true);
        }

        newPC->setGlobalShift(pc->getGlobalShift());
        newPC->setGlobalScale(pc->getGlobalScale());

        CLMeshDesc meshDesc(newMesh, desc.basename + "_POISSON_RECON",
                            desc.path);
        cmd.meshes().push_back(meshDesc);

        if (cmd.autoSaveMode()) {
            QString errorStr = cmd.exportEntity(cmd.meshes().back());
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}

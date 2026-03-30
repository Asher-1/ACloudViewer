// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PCV.h"

#include "PCVContext.h"

// Qt
#include <QString>

// System
#include <algorithm>
#include <cassert>
#include <cstring>

using namespace cloudViewer;

static int gcd(int num1, int num2) {
    int remainder = (num2 % num1);
    return (remainder != 0 ? gcd(remainder, num1) : num1);
}

//! Sample points on the unit sphere
/** Transcripted from MATLAB's script "partsphere.m" by Paul Leopardi,
    2003-10-13, for UNSW School of Mathematics.
**/
static bool SampleSphere(unsigned N, std::vector<CCVector3d>& dirs) {
    static const double c_eps = 2.2204e-16;
    static const double c_twist = 4.0;

    if (N == 0) {
        assert(false);
        return false;
    }

    try {
        dirs.resize(N, CCVector3d(0, 0, 1));
    } catch (const std::bad_alloc&) {
        return false;
    }

    if (N == 1) {
        return true;
    }

    try {
        double area = (4 * M_PI) / N;
        double beta = acos(1.0 - 2.0 / N);
        double gamma = M_PI - 2 * beta;
        double fuzz = c_eps * 2 * N;

        int Ltemp = static_cast<int>(ceil(gamma / sqrt(area) - fuzz));
        int L = 2 + std::max(Ltemp, 1);

        std::vector<double> mbar;
        mbar.resize(L, 0);
        assert(L >= 3);
        {
            mbar[0] = 1.0;
            double theta = gamma / (L - 2);
            for (int i = 1; i < L - 1; ++i) {
                mbar[i] = N * (cos(theta * (i - 1) + beta) -
                               cos(theta * i + beta)) /
                          2;
            }
            mbar[L - 1] = 1.0;
        }

        std::vector<int> m;
        m.resize(L, 0);
        {
            m[0] = 1;
            double alpha = 0.0;
            for (int i = 1; i < L; ++i) {
                double f = floor(mbar[i] + alpha + fuzz);
                if ((mbar[i] - f) >= 0.5) {
                    f = ceil(mbar[i] + alpha - fuzz);
                }
                m[i] = static_cast<int>(f);
                alpha += mbar[i] - m[i];
            }
        }

        {
            std::vector<double> offset;
            offset.resize(L - 1, 0);

            double z = 1.0 - static_cast<double>(2 + m[1]) / N;

            unsigned int rayIndex = 1;
            for (int i = 1; i < L - 1; ++i) {
                if (m[i - 1] != 0 && m[i] != 0) {
                    offset[i] =
                            offset[i - 1] +
                            static_cast<double>(gcd(m[i], m[i - 1])) /
                                    (2 * m[i] * m[i - 1]) +
                            std::min<double>(c_twist,
                                             floor(m[i - 1] / c_twist)) /
                                    m[i - 1];
                } else {
                    offset[i] = 0.0;
                }

                double temp = static_cast<double>(m[i]) / N;
                double h = cos((acos(z + temp) + acos(z - temp)) / 2);
                double r = sqrt(1.0 - h * h);

                for (int j = 0; j < m[i]; ++j) {
                    double theta =
                            2.0 * M_PI *
                            (offset[i] + static_cast<double>(j) / m[i]);
                    dirs[rayIndex++] =
                            CCVector3d(r * cos(theta), r * sin(theta), h);
                }

                z -= static_cast<double>(m[i] + m[i + 1]) / N;
            }

            assert(rayIndex + 1 == N);
        }
    } catch (const std::bad_alloc&) {
        return false;
    }

    dirs[N - 1] = CCVector3d(0, 0, -1);
    return true;
}

bool PCV::GenerateRays(unsigned numberOfRays,
                       std::vector<CCVector3d>& rays,
                       bool mode360) {
    unsigned rayCount = numberOfRays * (mode360 ? 1 : 2);
    if (!SampleSphere(rayCount, rays)) {
        return false;
    }

    if (!mode360) {
        unsigned lastIndex = 0;
        for (size_t i = 0; i < rays.size(); ++i) {
            if (rays[i].z < 0) {
                if (lastIndex != i) {
                    rays[lastIndex] = rays[i];
                }
                ++lastIndex;
            }
        }
        rayCount = lastIndex;
        rays.resize(rayCount);
    }

    return true;
}

int PCV::Launch(unsigned numberOfRays,
                GenericCloud* vertices,
                GenericMesh* mesh,
                bool meshIsClosed,
                bool mode360,
                unsigned width,
                unsigned height,
                cloudViewer::GenericProgressCallback* progressCb,
                const QString& entityName) {
    std::vector<CCVector3d> rays;
    if (!GenerateRays(numberOfRays, rays, mode360)) {
        return -2;
    }

    if (!Launch(rays, vertices, mesh, meshIsClosed, width, height, progressCb,
                entityName)) {
        return -1;
    }

    return static_cast<int>(rays.size());
}

bool PCV::Launch(const std::vector<CCVector3d>& rays,
                 cloudViewer::GenericCloud* vertices,
                 cloudViewer::GenericMesh* mesh,
                 bool meshIsClosed,
                 unsigned width,
                 unsigned height,
                 cloudViewer::GenericProgressCallback* progressCb,
                 const QString& entityName) {
    if (rays.empty()) {
        return false;
    }

    if (!vertices || !vertices->enableScalarField()) {
        return false;
    }

    unsigned numberOfPoints = vertices->size();
    unsigned numberOfRays = static_cast<unsigned>(rays.size());

    std::vector<int> visibilityCount;
    try {
        visibilityCount.resize(numberOfPoints, 0);
    } catch (const std::bad_alloc&) {
        return false;
    }

    cloudViewer::NormalizedProgress nProgress(progressCb, numberOfRays);
    if (progressCb) {
        if (progressCb->textCanBeEdited()) {
            progressCb->setMethodTitle("ShadeVis");
            QString infoStr;
            if (!entityName.isEmpty()) {
                infoStr = entityName + "\n";
            }
            infoStr.append(QString("Rays: %1").arg(numberOfRays));
            if (mesh) {
                infoStr.append(QString("\nFaces: %1").arg(mesh->size()));
            } else {
                infoStr.append(
                        QString("\nVertices: %1").arg(numberOfPoints));
            }
            progressCb->setInfo(qPrintable(infoStr));
        }
        progressCb->update(0);
        progressCb->start();
    }

    PCVContext win;
    bool success = true;
    if (win.init(width, height, vertices, mesh, meshIsClosed)) {
        for (unsigned i = 0; i < numberOfRays; ++i) {
            int result = win.glAccumPixel(visibilityCount, rays[i]);
            if (result < 0) {
                success = false;
                break;
            }

            if (progressCb && !nProgress.oneStep()) {
                success = false;
                break;
            }
        }

        if (success) {
            for (unsigned j = 0; j < numberOfPoints; ++j) {
                ScalarType visValue =
                        static_cast<ScalarType>(visibilityCount[j]) /
                        numberOfRays;
                vertices->setPointScalarValue(j, visValue);
            }
        }
    } else {
        success = false;
    }

    return success;
}

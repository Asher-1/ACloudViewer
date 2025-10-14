// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CorePoints.h"

// qCC_db
#include <ecvPointCloud.h>

// CCLib
#include <CloudSamplingTools.h>

// system
#include <assert.h>

using namespace masc;

bool CorePoints::prepare(
        cloudViewer::GenericProgressCallback* progressCb /*=nullptr*/) {
    if (!origin) {
        assert(false);
        return false;
    }

    if (selection) {
        // nothing to do
        return true;
    }

    // now we can compute the subsampled version
    cloudViewer::ReferenceCloud* ref = nullptr;
    switch (selectionMethod) {
        case SPATIAL: {
            // we'll need an octree
            if (!origin->getOctree()) {
                if (!origin->computeOctree(progressCb)) {
                    CVLog::Warning(
                            "[CorePoints::prepare] Failed to compute the "
                            "octree");
                    return false;
                }
            }

            cloudViewer::CloudSamplingTools::SFModulationParams modParams;
            modParams.enabled = false;
            ref = cloudViewer::CloudSamplingTools::resampleCloudSpatially(
                    origin, static_cast<PointCoordinateType>(selectionParam),
                    modParams, origin->getOctree().data(), progressCb);

            break;
        }

        case RANDOM: {
            if (selectionParam <= 0.0 || selectionParam >= 1.0) {
                CVLog::Warning(
                        "[CorePoints::prepare] Random subsampling ration must "
                        "be between 0 and 1 (excluded)");
                return false;
            }
            int targetCount = static_cast<int>(origin->size() * selectionParam);
            ref = cloudViewer::CloudSamplingTools::subsampleCloudRandomly(
                    origin, targetCount, progressCb);
            break;
        }

        case NONE:
            // nothing to do
            cloud = origin;
            return true;

        default:
            assert(false);
            break;
    }

    // store the references
    if (!ref) {
        CVLog::Warning(
                "[CorePoints::prepare] Failed to subsampled the origin cloud");
        return false;
    }
    selection.reset(ref);

    // and create the subsampled version of the cloud
    cloud = origin->partialClone(ref);
    if (!cloud) {
        CVLog::Warning(
                "[CorePoints::prepare] Failed to subsampled the origin cloud "
                "(not enough memory)");
        return false;
    }

    return true;
}

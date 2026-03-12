// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file VtkPointCloudColorHandlerLUT.h
 * @brief Color handler mapping point labels to RGB via VtkCloudLUT.
 */

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

#include <cmath>

#include "VtkCloudLUT.h"
#include "qVTK.h"

/**
 * @class AnnotationColorHandler
 * @brief Color handler that maps point labels to RGB colors via a lookup table.
 *
 * Uses vtkPoints and an integer label array instead of PCL point clouds.
 * VTK points are assumed finite; points with NaN x coordinate use default
 * color.
 */
class QVTK_ENGINE_LIB_API AnnotationColorHandler {
public:
    AnnotationColorHandler()
        : points_(nullptr), label_(nullptr), capable_(false) {}

    ~AnnotationColorHandler() = default;

    /** @param points Input VTK points. */
    void setInputPoints(vtkPoints* points) { points_ = points; }

    /** @param value Label array (one int per point). */
    void setLabel(int* value) {
        label_ = value;
        capable_ = true;
    }

    /** Returns true if the handler has valid input (points and labels). */
    bool isCapable() const { return capable_; }

    /**
     * Produces an RGB color array for all points.
     *
     * Iterates over points and assigns colors from VtkCloudLUT based on
     * label[i]. Points with NaN x coordinate use the default color (label 0).
     * Returns nullptr if not capable or input is invalid.
     */
    vtkSmartPointer<vtkUnsignedCharArray> getColor() const {
        if (!capable_ || !points_ || !label_) {
            return nullptr;
        }

        vtkIdType nr_points = points_->GetNumberOfPoints();
        if (nr_points <= 0) {
            return nullptr;
        }

        vtkSmartPointer<vtkUnsignedCharArray> scalars =
                vtkSmartPointer<vtkUnsignedCharArray>::New();
        scalars->SetNumberOfComponents(3);
        scalars->SetNumberOfTuples(nr_points);
        unsigned char* colors = scalars->GetPointer(0);

        for (vtkIdType i = 0; i < nr_points; ++i) {
            double* pt = points_->GetPoint(i);
            if (std::isnan(pt[0])) {
                CloudRGB default_color = VtkCloudLUT::at(0);
                colors[i * 3] = default_color.r;
                colors[i * 3 + 1] = default_color.g;
                colors[i * 3 + 2] = default_color.b;
                continue;
            }
            CloudRGB color = VtkCloudLUT::at(label_[i]);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

        return scalars;
    }

private:
    vtkPoints* points_;
    int* label_;
    bool capable_;
};

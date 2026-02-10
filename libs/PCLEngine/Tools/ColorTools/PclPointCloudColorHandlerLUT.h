// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/point_tests.h>
#include <vtkDataArray.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

// PclUtils color handler base (replaces pcl::visualization::PointCloudColorHandler)
#include "base/CVVisualizerTypes.h"

#include "PclCloudLUT.h"

template <typename PointT>
class PclPointCloudColorHandlerLUT
        : public PclUtils::PointCloudColorHandler<PointT> {
public:
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef typename PointCloud::Ptr PointCloudPtr;
    typedef typename PointCloud::ConstPtr PointCloudConstPtr;

    typedef std::shared_ptr<PclPointCloudColorHandlerLUT<PointT>> Ptr;
    typedef std::shared_ptr<const PclPointCloudColorHandlerLUT<PointT>>
            ConstPtr;

    /** \brief Constructor. */
    PclPointCloudColorHandlerLUT()
        : PclUtils::PointCloudColorHandler<PointT>() {
        capable_ = false;
    }

    /** \brief Constructor. */
    PclPointCloudColorHandlerLUT(const PointCloudConstPtr& cloud)
        : PclUtils::PointCloudColorHandler<PointT>(cloud) {
        setInputCloud(cloud);
    }

    /** \brief Destructor. */
    virtual ~PclPointCloudColorHandlerLUT() {}

    /** \brief Check if this handler is capable of handling the input data or
     * not. */
    inline bool isCapable() const override { return (capable_); }

    /** \brief Abstract getName method. */
    std::string getName() const override { return "PclPointCloudColorHandlerLUT"; }

    /** \brief Abstract getFieldName method. */
    std::string getFieldName() const override { return ""; }

    /** \brief Obtain the actual color for the input dataset as vtk scalars.
     * \return smart pointer to VTK data array, or null on failure
     */
    vtkSmartPointer<vtkDataArray> getColor() const override {
        if (!capable_ || !cloud_) return nullptr;

        auto scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
        scalars->SetNumberOfComponents(3);

        vtkIdType nr_points = cloud_->points.size();
        scalars->SetNumberOfTuples(nr_points);
        unsigned char* colors = scalars->GetPointer(0);

        int j = 0;
        for (vtkIdType cp = 0; cp < nr_points; ++cp) {
            if (pcl::isFinite(cloud_->points[cp])) {
                const pcl::RGB& color = PclCloudLUT::at(label[cp]);
                colors[j] = color.r;
                colors[j + 1] = color.g;
                colors[j + 2] = color.b;
                j += 3;
            }
        }
        return scalars;
    }

    /** \brief Set the input cloud to be used.
     * \param[in] cloud the input cloud to be used by the handler
     */
    void setInputCloud(const PointCloudConstPtr& cloud) override {
        cloud_ = cloud;
    }

    void setLabel(int* value) {
        label = value;
        capable_ = true;
    }

private:
    /**
     * @brief array of cloud label
     */
    int* label;

    // Members derived from the base class
    using PclUtils::PointCloudColorHandler<PointT>::cloud_;
    using PclUtils::PointCloudColorHandler<PointT>::capable_;
    using PclUtils::PointCloudColorHandler<PointT>::field_idx_;
    using PclUtils::PointCloudColorHandler<PointT>::fields_;
};

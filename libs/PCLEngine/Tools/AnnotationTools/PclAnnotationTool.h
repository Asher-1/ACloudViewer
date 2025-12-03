// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "PclUtils/PCLCloud.h"
#include "Tools/ColorTools/PclPointCloudColorHandlerLUT.h"
#include "qPCL.h"

// ECV_DB_LIB
#include <ecvGenericAnnotationTool.h>

// QT
#include <QObject>
#include <memory>

namespace PclUtils {
class PCLVis;
}
class ecvGenericVisualizer3D;

class vtkActor;
class Annotation;
class Annotaions;

class QPCL_ENGINE_LIB_API PclAnnotationTool : public ecvGenericAnnotationTool {
    Q_OBJECT
public:
    explicit PclAnnotationTool(
            AnnotationMode mode = AnnotationMode::BOUNDINGBOX);
    explicit PclAnnotationTool(
            ecvGenericVisualizer3D* viewer,
            AnnotationMode mode = AnnotationMode::BOUNDINGBOX);
    virtual ~PclAnnotationTool() override;

public:  // implemented from ecvGenericAnnotationTool interface
    virtual void setVisualizer(
            ecvGenericVisualizer3D* viewer = nullptr) override;
    virtual bool loadClassesFromFile(const std::string& file) override;
    virtual void getAnnotationLabels(
            std::vector<std::string>& labelList) override;
    virtual bool getCurrentAnnotations(std::vector<int>& annos) const override;
    virtual void initAnnotationLabels(
            const std::vector<std::string>& labelList) override;

    virtual void toggleInteractor() override;
    virtual bool setInputCloud(ccPointCloud* pointCloud,
                               int viewport = 0) override;

    virtual void start() override;
    virtual void stop() override;

    virtual void intersectMode() override;
    virtual void unionMode() override;
    virtual void trimMode() override;
    virtual void resetMode() override;

    /**
     * @brief clear state before load new cloud and annotation
     */
    virtual void reset() override;
    virtual void clear() override;

    /**
     * @brief export annotations
     */
    virtual void exportAnnotations() override;

    virtual void changeAnnotationType(const std::string& type) override;
    virtual void selectExistedAnnotation(const std::string& type) override;
    virtual void updateCloud() override;

    // visibility
    virtual void showAnnotation() override;
    virtual void hideAnnotation() override;
    virtual void showOrigin() override;
    virtual void hideOrigin() override;
    virtual void removeAnnotation() override;

protected:  // implemented from ecvGenericAnnotationTool interface
    virtual void initialize(ecvGenericVisualizer3D* viewer) override;

protected slots:
    void pointPickingProcess(int index);
    void areaPickingEventProcess(const std::vector<int>& new_selected_slice);
    void pickedEventProcess(vtkActor* actor);
    void keyboardEventProcess(const std::string& symKey);

private:
    void showAnnotation(const Annotation* anno);
    void removeAnnotation(Annotation* anno);
    void hideAnnotation(Annotation* anno);

    void changeAnnotationType(Annotation* anno, const std::string& type);
    void setPointSize(const std::string& viewID, int viewport = 0);
    void highlightPoint(std::vector<int>& slice);
    void defaultColorPoint(std::vector<int>& slice);
    void createAnnotationFromSelectPoints(std::string type = "unknown");

    void labelCloudByAnnotations();
    void labelCloudByAnnotation(const Annotation* anno);
    void resetCloudByAnnotation(const Annotation* anno);
    void updateCloudLabel(const std::string& type);

    /**
     * @brief show loaded cloud and annotations
     */
    void refresh();
    void loadDefaultClasses();

    void filterPickedSlice(const std::vector<int>& inSlices,
                           std::vector<int>& outSlices,
                           bool skip = false);

private:
    void fastLabelCloud(const std::vector<int>& inSlices, int label = 0);

private:
    bool m_intersectMode;
    bool m_unionMode;
    bool m_trimMode;

    PclUtils::PCLVis* m_viewer;

    std::string m_pointcloudFileName;
    std::string m_annotationFileName;

    std::string m_annotationCloudId = "annotationCloud";
    std::string m_baseCloudId;

    /**
     * @brief the loaded cloud
     */
    PointCloudI::Ptr m_baseCloud;

    /**
     * @brief state of each point to identity color or selectable
     */
    int* m_cloudLabel;
    PclPointCloudColorHandlerLUT<PointIntensity> m_colorHandler;
    std::vector<int> m_last_selected_slice;

    // manage annotations
    std::shared_ptr<Annotaions> m_annoManager;

    // for pick
    Annotation* m_currPickedAnnotation;

    std::vector<Annotation*> m_lastSelectedAnnotations;
};

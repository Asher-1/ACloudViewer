// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file Annotaion.h
 * @brief Annotation and box label types for point cloud annotation.
 */

#include <vtkSmartPointer.h>

class vtkPoints;

#include <vector>

/// Bounding box label with center, dimensions, and type for annotation.
struct BoxLabel {
    BoxLabel() {
        type = "unknown";
        this->detail.center_x = this->detail.center_y = this->detail.center_z =
                0;
        this->detail.yaw = 2;
        this->detail.length = this->detail.width = this->detail.height = 1;
    }
    BoxLabel(const double p1[3],
             const double p2[3],
             std::string type_ = "unknown") {
        type = type_;
        this->detail.center_x = (p1[0] + p2[0]) / 2;
        this->detail.center_y = (p1[1] + p2[1]) / 2;
        this->detail.center_z = (p1[2] + p2[2]) / 2;
        this->detail.yaw = 0;
        this->detail.length = p2[0] - p1[0];
        this->detail.width = p2[1] - p1[1];
        this->detail.height = p2[2] - p1[2];
    }
    std::string type;
    union {
        double data[7];
        struct {
            double center_x;
            double center_y;
            double center_z;
            double length;
            double width;
            double height;
            double yaw;
        } detail;
    };

    std::string toString() {
        char buffer[200];
        sprintf(buffer, "%s %f %f %f %f %f %f %f", type.c_str(), data[0],
                data[1], data[2], data[3], data[4], data[5], data[6]);
        return std::string(buffer);
    }
};

class vtkBalloonWidget;
class vtkBoxWidgetCallback0;
class vtkBoxWidgetCallback1;
class vtkAnnotationBoxSource;
class vtkBoxWidgetRestricted;

class vtkTransform;
class vtkRenderWindowInteractor;
class vtkActor;
class vtkPolyDataMapper;

/**
 * @class Annotation
 * @brief Single 3D bounding box annotation with interactive editing.
 */
class Annotation {
    friend class Annotaions;

public:
    /**
     * @brief Annotation  construct from slice which load from label file
     * @param slice Point indices from label file.
     * @param type_ Annotation type string.
     */
    Annotation(const std::vector<int>& slice, std::string type_);

    /**
     * @brief Annotation  construct from boxlabel which load from label file
     * @param label Box label with center, dimensions, type.
     * @param visible_ Whether annotation is visible.
     * @param lock_ Whether annotation is locked.
     */
    Annotation(const BoxLabel& label, bool visible_ = true, bool lock_ = false);

    /**
     * @brief Annotation construct from part of cloud points
     * @param points VTK points.
     * @param slice Point indices to include.
     * @param type_ Annotation type string.
     */
    Annotation(vtkPoints* points, std::vector<int>& slice, std::string type_);

    ~Annotation();

    /**
     * @brief getBoxLabel get boxLabel from annotaion tranformation
     * @return BoxLabel with center, dimensions, and type.
     */
    BoxLabel getBoxLabel();

    /**
     * @brief apply transform to annotation
     * @param t Transform to apply.
     */
    void applyTransform(vtkSmartPointer<vtkTransform> t);

    /**
     * @brief enter picked state, show boxwidget which allow to adjust
     * annotation
     * @param interactor VTK interactor for the render window.
     */
    void picked(vtkRenderWindowInteractor* interactor);

    /**
     * @brief disable boxWidget
     */
    void unpicked();

    /**
     * @brief keep current orientation, re-compute the center and scale
     * to make annotation fit to selected point well enough
     */
    void adjustToAnchor();

    /**
     * @brief change the type of annotation, and color too
     * @param value New annotation type string.
     */
    void setType(const std::string value);
    /// @return VTK actor for the annotation box.
    vtkSmartPointer<vtkActor> getActor() const;
    /// @return Annotation type string.
    std::string getType() const;

    /// @return Point indices (slice) for this annotation.
    const std::vector<int>& getSlice() const;

protected:
    void initial();

    /**
     * @brief color the annotation with given color
     * @param color_index
     * if color_index>=0,refer to @ref pcl::GlasbeyLUT
     * otherwise use color already mapped by type
     */
    void colorAnnotation(int color_index = -1);

    /**
     * @brief copy selected points as anchor to current annotation
     * @param cloud
     * @param slice
     */
    void setAnchorPoint(vtkPoints* points, const std::vector<int>& slice);

    /**
     * @brief computeScaleAndCenterShift
     * @param o direction
     * @param scs ["scale", "center shift"]
     * @return scale
     */
    double computeScaleAndCenterShift(double o[3], double scs[2]);

private:
    std::string type;
    vtkSmartPointer<vtkAnnotationBoxSource> source;
    vtkSmartPointer<vtkActor> actor;
    vtkSmartPointer<vtkPolyDataMapper> mapper;
    vtkSmartPointer<vtkTransform> transform;

    vtkSmartPointer<vtkBoxWidgetRestricted> boxWidget;
    vtkSmartPointer<vtkBoxWidgetCallback0> boxWidgetCallback0;
    vtkSmartPointer<vtkBoxWidgetCallback1> boxWidgetCallback1;

    std::vector<double*> anchorPoints;

    std::vector<int> m_slice;

    double center[3];

    // NOTE not used
    bool visible;
    bool lock;

public:
    /**
     * @brief get types vector pointer
     * @return Pointer to the static types vector.
     */
    static std::vector<std::string>* GetTypes();

    /**
     * @brief GetTypeIndex  auto add to vector map if has not
     * @param type_ Type name string.
     * @return Index of the type in the types vector.
     */
    static std::size_t GetTypeIndex(std::string type_);

    /**
     * @brief GetTypeByIndex  auto add to vector map if has not
     * @param index Type index.
     * @return Type name string for the given index.
     */
    static std::string GetTypeByIndex(size_t index);

    /**
     * @brief ComputeOBB compute max,min [x,y,z] aligned to xyz axis
     * @param points Input point set.
     * @param slice Indices of points to include.
     * @param p1 Output min [x,y,z].
     * @param p2 Output max [x,y,z].
     */
    static void ComputeOBB(vtkPoints* points,
                           std::vector<int>& slice,
                           double p1[3],
                           double p2[3]);

protected:
    /**
     * @brief types all annotation type here
     */
    static std::vector<std::string>* types;
};

/**
 * @class Annotaions
 * @brief Manages a collection of annotations with load/save and balloon
 * display.
 */
class Annotaions {
public:
    /// @param interactor VTK interactor for balloon and box widgets.
    explicit Annotaions(vtkRenderWindowInteractor* interactor = nullptr);

    /// @param num Number of annotations to preserve (0 = all).
    void preserve(size_t num = 0);
    void release();

    /// @param anno Annotation to add.
    void add(Annotation* anno);
    /// @param anno Annotation to remove.
    void remove(Annotation* anno);
    void clear();
    /// @return Number of annotations in the collection.
    std::size_t getSize();

    /// @param anno Annotation to update labels for.
    /// @param resetFlag If true, reset labels.
    void updateLabels(Annotation* anno, bool resetFlag = false);

    /**
     * @brief load annotations from file
     * @param filename Path to annotation file.
     * @param mode Load mode.
     */
    void loadAnnotations(std::string filename, int mode);

    /**
     * @brief save annotations to file
     * @param filename Path to save file.
     * @param mode Save mode.
     */
    void saveAnnotations(std::string filename, int mode);

    /// @param annos Output vector of annotation indices.
    /// @return true if annotations were retrieved.
    bool getAnnotations(std::vector<int>& annos) const;

    /**
     * @brief from annotatin box actor to find annotation itself
     * @param actor VTK actor of the annotation box.
     * @return Pointer to the Annotation, or nullptr if not found.
     */
    Annotation* getAnnotation(vtkActor* actor);
    /// @param index Index of the annotation.
    /// @return Pointer to the Annotation at the given index.
    Annotation* getAnnotation(std::size_t index);
    /// @param type Filter by annotation type.
    /// @param annotations Output vector of matching annotations.
    void getAnnotations(const std::string& type,
                        std::vector<Annotation*>& annotations);
    /// @param anno Annotation to find index of.
    /// @return Index of the annotation, or -1 if not found.
    int getAnnotationIndex(Annotation* anno);
    /// @return Reference to the annotations vector.
    std::vector<Annotation*>& getAnnotations();

    /// @param index Cloud point index.
    /// @return Label at index, or -1 if out of range.
    inline int getLabelByIndex(std::size_t index) {
        if (index >= m_capacity) {
            return -1;
        }
        return m_labeledCloudIndex[index];
    }

    /// @param index Annotation index to update balloon for.
    void updateBalloonByIndex(std::size_t index);
    /// @param anno Annotation to update balloon for.
    void updateBalloonByAnno(Annotation* anno);

protected:
    /**
     * @brief keep all annotation from current cloud
     */
    std::vector<Annotation*> m_annotations;

    vtkRenderWindowInteractor* m_interactor;
    vtkSmartPointer<vtkBalloonWidget> m_balloonWidget;

    int* m_labeledCloudIndex = nullptr;
    size_t m_capacity = 0;
};

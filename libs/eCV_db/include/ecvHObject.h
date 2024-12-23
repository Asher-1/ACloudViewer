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

#ifndef ECV_HIERARCHY_OBJECT_HEADER
#define ECV_HIERARCHY_OBJECT_HEADER

// CV_CORE_LIB
#include "BoundingBox.h"

// Local
#include "ecvColorTypes.h"
#include "ecvDrawableObject.h"
#include "ecvGLMatrix.h"
#include "ecvGenericDisplayTools.h"
#include "ecvObject.h"

class QIcon;
class ccBBox;
class ecvOrientedBBox;

//! Hierarchical CLOUDVIEWER  Object
class ECV_DB_LIB_API ccHObject : public ccObject, public ccDrawableObject {
public:  // construction
    //! Default constructor
    /** \param name object name (optional)
     **/
    ccHObject(QString name = QString());
    //! Copy constructor
    ccHObject(const ccHObject& object);

    //! Default destructor
    virtual ~ccHObject() override;

    //! Static factory
    /** Warning: objects depending on other structures (such as meshes
            or polylines that should be linked with point clouds playing the
            role of vertices) are returned 'naked'.
            \param objectType object type
            \param name object name (optional)
            \return instantiated object (if type is valid) or 0
    **/
    static ccHObject* New(CV_CLASS_ENUM objectType, const char* name = nullptr);

    //! Static factory (version to be used by external plugin factories)
    /** Two strings are used as keys, one for the plugin name and one for the
    class name. Those strings will typically be saved as metadata of a custom
    object
    **/
    static ccHObject* New(const QString& pluginId,
                          const QString& classId,
                          const char* name = nullptr);

    /////////////////////// for python interface
    ////////////////////////////////////
    /// Compute min bound of a list points.
    static Eigen::Vector3d ComputeMinBound(
            const std::vector<Eigen::Vector3d>& points);
    /// Compute max bound of a list points.
    static Eigen::Vector3d ComputeMaxBound(
            const std::vector<Eigen::Vector3d>& points);
    /// Computer center of a list of points.
    static Eigen::Vector3d ComputeCenter(
            const std::vector<Eigen::Vector3d>& points);

    /// \brief Resizes the colors vector and paints a uniform color.
    ///
    /// \param colors An array of eigen vectors specifies colors in RGB.
    /// \param size The resultant size of the colors array.
    /// \param color The final color in which the colors will be painted.
    static void ResizeAndPaintUniformColor(std::vector<Eigen::Vector3d>& colors,
                                           std::size_t size,
                                           const Eigen::Vector3d& color);

    /// \brief Transforms all points with the transformation matrix.
    ///
    /// \param transformation 4x4 matrix for transformation.
    /// \param points A list of points to be transformed.
    static void TransformPoints(const Eigen::Matrix4d& transformation,
                                std::vector<Eigen::Vector3d>& points);
    /// \brief Transforms the normals with the transformation matrix.
    ///
    /// \param transformation 4x4 matrix for transformation.
    /// \param normals A list of normals to be transformed.
    static void TransformNormals(const Eigen::Matrix4d& transformation,
                                 std::vector<Eigen::Vector3d>& normals);

    /// \brief Transforms all covariance matrices with the transformation.
    ///
    /// \param transformation 4x4 matrix for transformation.
    /// \param covariances A list of covariance matrices to be transformed.
    static void TransformCovariances(const Eigen::Matrix4d& transformation,
                                     std::vector<Eigen::Matrix3d>& covariances);

    /// \brief Apply translation to the geometry coordinates.
    ///
    /// \param translation A 3D vector to transform the geometry.
    /// \param points A list of points to be transformed.
    /// \param relative If `true`, the \p translation is directly applied to the
    /// \points. Otherwise, the center of the \points is moved to the \p
    /// translation.
    static void TranslatePoints(const Eigen::Vector3d& translation,
                                std::vector<Eigen::Vector3d>& points,
                                bool relative);
    /// \brief Scale the coordinates of all points by the scaling factor \p
    /// scale.
    ///
    /// \param scale If `true`, the scale is applied relative to the center of
    /// the geometry. Otherwise, the scale is directly applied to the geometry,
    /// i.e. relative to the origin. \param points A list of points to be
    /// transformed. \param center If `true`, then the scale is applied to the
    /// centered geometry.
    static void ScalePoints(const double scale,
                            std::vector<Eigen::Vector3d>& points,
                            const Eigen::Vector3d& center);
    /// \brief Rotate all points with the rotation matrix \p R.
    ///
    /// \param R A 3x3 rotation matrix
    /// defines the axis of rotation and the norm the angle around this axis.
    /// \param points A list of points to be transformed.
    /// \param center Rotation center that is used for the rotation.
    static void RotatePoints(const Eigen::Matrix3d& R,
                             std::vector<Eigen::Vector3d>& points,
                             const Eigen::Vector3d& center);
    /// \brief Rotate all normals with the rotation matrix \p R.
    ///
    /// \param R A 3x3 rotation matrix
    /// \param normals A list of normals to be transformed.
    static void RotateNormals(const Eigen::Matrix3d& R,
                              std::vector<Eigen::Vector3d>& normals);

    /// \brief Rotate all covariance matrices with the rotation matrix \p R.
    ///
    /// \param R A 3x3 rotation matrix
    /// \param covariances A list of covariance matrices to be transformed.
    static void RotateCovariances(const Eigen::Matrix3d& R,
                                  std::vector<Eigen::Matrix3d>& covariances);

    virtual bool isEmpty() const { return true; }
    /// Returns min bounds for geometry coordinates.
    virtual Eigen::Vector3d getMinBound() const { return Eigen::Vector3d(); }
    virtual Eigen::Vector2d getMin2DBound() const { return Eigen::Vector2d(); }
    /// Returns max bounds for geometry coordinates.
    virtual Eigen::Vector3d getMaxBound() const { return Eigen::Vector3d(); }
    virtual Eigen::Vector2d getMax2DBound() const { return Eigen::Vector2d(); }
    /// Returns the center of the geometry coordinates.
    virtual Eigen::Vector3d getGeometryCenter() const {
        return Eigen::Vector3d();
    }

    /// Returns an axis-aligned bounding box of the geometry.
    virtual ccBBox getAxisAlignedBoundingBox() const;
    /// Returns an oriented bounding box of the geometry.
    virtual ecvOrientedBBox getOrientedBoundingBox() const;

    /// \brief Apply transformation (4x4 matrix) to the geometry coordinates.
    virtual ccHObject& transform(const Eigen::Matrix4d& transformation) {
        return *this;
    }
    /// \brief Apply translation to the geometry coordinates.
    ///
    /// \param translation A 3D vector to transform the geometry.
    /// \param relative If `true`, the \p translation is directly applied to the
    /// geometry. Otherwise, the geometry center is moved to the \p translation.
    virtual ccHObject& translate(const Eigen::Vector3d& translation,
                                 bool relative = true) {
        return *this;
    }
    /// \brief Apply scaling to the geometry coordinates.
    /// Given a scaling factor \f$s\f$, and center \f$c\f$, a given point
    /// \f$p\f$ is transformed according to \f$s (p - c) + c\f$.
    ///
    /// \param scale The scale parameter that is multiplied to the
    /// points/vertices of the geometry.
    /// \param center Scale center that is used to resize the geometry.
    virtual ccHObject& scale(const double s, const Eigen::Vector3d& center) {
        return *this;
    }
    virtual ccHObject& scale(const double s) {
        return scale(s, getGeometryCenter());
    }
    /// \brief Apply rotation to the geometry coordinates and normals.
    /// Given a rotation matrix \f$R\f$, and center \f$c\f$, a given point
    /// \f$p\f$ is transformed according to \f$R (p - c) + c\f$.
    ///
    /// \param R A 3x3 rotation matrix
    /// \param center Rotation center that is used for the rotation.
    virtual ccHObject& rotate(const Eigen::Matrix3d& R,
                              const Eigen::Vector3d& center) {
        return *this;
    }
    virtual ccHObject& rotate(const Eigen::Matrix3d& R) {
        return rotate(R, getGeometryCenter());
    }

    /// Get Rotation Matrix from XYZ RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromXYZ(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from YZX RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromYZX(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from ZXY RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromZXY(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from XZY RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromXZY(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from ZYX RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromZYX(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from YXZ RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromYXZ(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from AxisAngle RotationType.
    static Eigen::Matrix3d GetRotationMatrixFromAxisAngle(
            const Eigen::Vector3d& rotation);
    /// Get Rotation Matrix from Quaternion.
    static Eigen::Matrix3d GetRotationMatrixFromQuaternion(
            const Eigen::Vector4d& rotation);
    /// Get Rotation Matrix from Euler angle.
    static Eigen::Matrix3d GetRotationMatrixFromEulerAngle(
            const Eigen::Vector3d& rotation);
    /////////////////////// for python interface
    ////////////////////////////////////

public:  // base members access
    inline QString getViewId() const {
        return QString::number(this->getUniqueID(), 10);
    }

    //! Returns class ID
    /** \return class unique ID
     **/
    inline CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::HIERARCHY_OBJECT;
    }

    //! Returns whether the instance is a group
    inline bool isGroup() const {
        return getClassID() ==
               static_cast<CV_CLASS_ENUM>(CV_TYPES::HIERARCHY_OBJECT);
    }

    //! Returns parent object
    /** \return parent object (nullptr if no parent)
     **/
    inline ccHObject* getParent() const { return m_parent; }

    //! Returns the icon associated to this entity
    /** ccDBRoot will call this method: if an invalid icon is returned
            the default icon for that type will be used instead.
            \return invalid icon by default (to be re-implemented by child
    class)
    **/
    virtual QIcon getIcon() const;

public:  // dependencies management
    //! Dependency flags
    enum DEPENDENCY_FLAGS {
        DP_NONE = 0,                   /**< no dependency **/
        DP_NOTIFY_OTHER_ON_DELETE = 1, /**< notify 'other' when deleted (will
                                          call ccHObject::onDeletionOf) **/
        DP_NOTIFY_OTHER_ON_UPDATE =
                2, /**< notify 'other' when its geometry is modified (will call
                      ccHObject::onUpdateOf) **/
        // DP_NOTIFY_XXX				= 4,
        DP_DELETE_OTHER = 8,     /**< delete 'other' before deleting itself **/
        DP_PARENT_OF_OTHER = 24, /**< same as DP_DELETE_OTHER + declares itself
                                    as parent of 'other' **/
    };

    //! Adds a new dependence (additive or not)
    /** \param otherObject other object
            \param flags dependency flags (see DEPENDENCY_FLAGS)
            \param additive whether we should 'add' the flag(s) if there's
    already a dependence with the other object or not
    **/
    void addDependency(ccHObject* otherObject, int flags, bool additive = true);

    //! Returns the dependency flags with a given object
    /** \param otherObject other object
     **/
    int getDependencyFlagsWith(const ccHObject* otherObject);

    //! Removes any dependency flags with a given object
    /** \param otherObject other object
     **/
    void removeDependencyWith(ccHObject* otherObject);

    //! Removes a given dependency flag
    /** \param otherObject other object
            \param flag dependency flag to remove (see DEPENDENCY_FLAGS)
    **/
    void removeDependencyFlag(ccHObject* otherObject, DEPENDENCY_FLAGS flag);

public:  // children management
    //! Adds a child
    /** \warning by default (i.e. with the DP_PARENT_OF_OTHER flag) the child's
    parent will be automatically replaced by this instance. Moreover the child
    will be deleted

            \param child child instance
            \param dependencyFlags dependency flags
            \param insertIndex insertion index (if <0, child is simply appended
    to the children list) \return success
    **/
    virtual bool addChild(ccHObject* child,
                          int dependencyFlags = DP_PARENT_OF_OTHER,
                          int insertIndex = -1);

    //! Returns the number of children
    /** \return children number
     **/
    inline unsigned getChildrenNumber() const {
        return static_cast<unsigned>(m_children.size());
    }

    //! Returns the total number of children under this object recursively
    /** \return Number of children
     **/
    unsigned int getChildCountRecursive() const;

    //! Returns the ith child
    /** \param childPos child position
            \return child object (or nullptr if wrong position)
    **/
    inline ccHObject* getChild(unsigned childPos) const {
        return (childPos < getChildrenNumber() ? m_children[childPos]
                                               : nullptr);
    }

    //! Finds an entity in this object hierarchy
    /** \param uniqueID child unique ID
            \return child (or nullptr if not found)
    **/
    ccHObject* find(unsigned uniqueID);

    //! Standard instances container (for children, etc.)
    using Container = std::vector<ccHObject*>;

    //! Shared pointer
    using Shared = QSharedPointer<ccHObject>;

    //! Shared instances container (for children, etc.)
    using SharedContainer = std::vector<Shared>;

    //! Collects the children corresponding to a certain pattern
    /** \param filteredChildren result container
            \param recursive specifies if the search should be recursive
            \param filter pattern for children selection
            \param strict whether the search is strict on the type (i.e 'isA')
    or not (i.e. 'isKindOf') \return number of collected children
    **/
    unsigned filterChildren(Container& filteredChildren,
                            bool recursive = false,
                            CV_CLASS_ENUM filter = CV_TYPES::OBJECT,
                            bool strict = false) const;

    //! Detaches a specific child
    /** This method does not delete the child.
            Removes any dependency between the flag and this object
    **/
    void detachChild(ccHObject* child);
    //! Removes a specific child
    /** \warning This method may delete the child if the DP_PARENT_OF_OTHER
            dependency flag is set for this child (use detachChild if you
            want to avoid deletion).
    **/
    //! Detaches all children
    void detachAllChildren();

    void getTypeID_recursive(std::vector<removeInfo>& rmInfos, bool relative);
    void getTypeID_recursive(std::vector<hideInfo>& hdInfos, bool relative);

    void removeChild(ccHObject* child);
    //! Removes a specific child given its index
    /** \warning This method may delete the child if the DP_PARENT_OF_OTHER
            dependency flag is set for this child (use detachChild if you
            want to avoid deletion).
    **/
    void removeChild(int pos);
    //! Removes all children
    void removeAllChildren();
    //! Returns child index
    int getChildIndex(const ccHObject* aChild) const;
    //! Swaps two children
    void swapChildren(unsigned firstChildIndex, unsigned secondChildIndex);
    //! Returns index relatively to its parent or -1 if no parent
    int getIndex() const;

    //! Transfer a given child to another parent
    void transferChild(ccHObject* child, ccHObject& newParent);
    //! Transfer all children to another parent
    void transferChildren(ccHObject& newParent,
                          bool forceFatherDependent = false);

    //! Shortcut: returns first child
    ccHObject* getFirstChild() const {
        return (m_children.empty() ? nullptr : m_children.front());
    }
    //! Shortcut: returns last child
    ccHObject* getLastChild() const {
        return (m_children.empty() ? nullptr : m_children.back());
    }

    //! Returns true if the current object is an ancestor of the specified one
    bool isAncestorOf(const ccHObject* anObject) const;

    void removeFromRenderScreen(bool recursive = true);

public:  // bounding-box
    void hideObject_recursive(bool recursive);
    void hideBB(CC_DRAW_CONTEXT context);
    void showBB(CC_DRAW_CONTEXT context);

    //! Returns the entity's own bounding-box
    /** Children bboxes are ignored.
            \param withGLFeatures whether to take into account display-only
    elements (if any) \return bounding-box
    **/
    virtual ccBBox getOwnBB(bool withGLFeatures = false);

    void setRedrawFlagRecursive(bool redraw = false);
    void setForceRedrawRecursive(bool redraw = false);

    void setPointSizeRecursive(int pSize);
    void setLineWidthRecursive(PointCoordinateType width);

    //! Returns the bounding-box of this entity and it's children
    /** \param withGLFeatures whether to take into account display-only elements
    (if any) \param onlyEnabledChildren only consider the 'enabled' children
            \return bounding-box
    **/
    virtual ccBBox getBB_recursive(bool withGLFeatures = false,
                                   bool onlyEnabledChildren = true);

    //! Global (non-shifted) bounding-box
    using GlobalBoundingBox = cloudViewer::BoundingBoxTpl<double>;

    //! Returns the entity's own global bounding-box (with global/non-shifted
    //! coordinates - if relevant)
    /** Children bounding-boxes are ignored.
            May differ from the (local) bounding-box if the entity is shifted
            \param withGLFeatures whether to take into account display-only
    elements (if any) \return global bounding-box
    **/
    virtual GlobalBoundingBox getOwnGlobalBB(bool withGLFeatures = false);

    //! Returns the entity's own global bounding-box (with global/non-shifted
    //! coordinates - if relevant)
    /** Children bounding-boxes are ignored.
            By default this method returns the local bounding-box!
            But it may differ from the (local) bounding-box if the entity is
    shifted. \param[out] minCorner min global bounding-box corner \param[out]
    maxCorner max global bounding-box corner \return whether the bounding box is
    valid or not
    **/
    virtual bool getOwnGlobalBB(CCVector3d& minCorner, CCVector3d& maxCorner);

    //! Returns the global bounding-box of this entity and it's children
    /** \param withGLFeatures whether to take into account display-only elements
    (if any) \param onlyEnabledChildren only consider the 'enabled' children
            \return bounding-box
    **/
    virtual GlobalBoundingBox getGlobalBB_recursive(
            bool withGLFeatures = false, bool onlyEnabledChildren = true);

    //! Returns the bounding-box of this entity and it's children WHEN DISPLAYED
    /** Children's GL transformation is taken into account (if enabled).
            \param relative whether the bounding-box is relative (i.e. in the
    entity's local coordinate system) or absolute (in which case the parent's GL
    transformation will be taken into account) \param display if not null, this
    method will return the bounding-box of this entity (and its children) in the
    specified 3D view (i.e. potentially not visible) \return bounding-box
    **/
    virtual ccBBox getDisplayBB_recursive(bool relative);

    //! Returns best-fit bounding-box (if available)
    /** \warning Only suitable for leaf objects (i.e. without children)
            Therefore children bboxes are always ignored.
            \warning This method is not supported by all entities!
            (returns the axis-aligned bounding-box by default).
            \param[out] trans associated transformation (so that the
    bounding-box can be displayed in the right position!) \return fit
    bounding-box
    **/
    virtual ccBBox getOwnFitBB(ccGLMatrix& trans);

    //! Draws the entity (and its children) bounding-box
    virtual void drawBB(CC_DRAW_CONTEXT& context, const ecvColor::Rgb& col);

public:  // display
    // Inherited from ccDrawableObject
    void draw(CC_DRAW_CONTEXT& context) override;

    void updateNameIn3DRecursive();

    void setHideShowType(CC_DRAW_CONTEXT& context);
    void setRemoveType(CC_DRAW_CONTEXT& context);
    ENTITY_TYPE getEntityType() const;

    //! Redraws associated display
    virtual void redrawDisplay(bool forceRedraw = true, bool only2D = false);

    //! Returns the absolute transformation (i.e. the actual displayed GL
    //! transforamtion) of an entity
    /** \param[out] trans absolute transformation
            \return whether a GL transformation is actually enabled or not
    **/
    bool getAbsoluteGLTransformation(ccGLMatrix& trans) const;

    //! Returns whether the object is actually displayed (visible) or not
    virtual bool isDisplayed() const;

    //! Returns whether the object and all its ancestors are enabled
    virtual bool isBranchEnabled() const;

/*** RECURSIVE CALL SCRIPTS ***/

// 0 parameter
#define ccHObject_recursive_call0(baseName, recursiveName) \
    inline virtual void recursiveName() {                  \
        baseName();                                        \
        for (Container::iterator it = m_children.begin();  \
             it != m_children.end(); ++it)                 \
            (*it)->recursiveName();                        \
    }

// 1 parameter
#define ccHObject_recursive_call1(baseName, param1Type, recursiveName) \
    inline virtual void recursiveName(param1Type p) {                  \
        baseName(p);                                                   \
        for (Container::iterator it = m_children.begin();              \
             it != m_children.end(); ++it)                             \
            (*it)->recursiveName(p);                                   \
    }

    ccHObject_recursive_call1(redrawDisplay, bool, redrawDisplay_recursive) ccHObject_recursive_call1(
            redrawDisplay,
            bool,
            refreshDisplay_recursive) ccHObject_recursive_call1(setSelected,
                                                                bool,
                                                                setSelected_recursive)
            ccHObject_recursive_call0(toggleActivation,
                                      toggleActivation_recursive)
                    ccHObject_recursive_call0(toggleVisibility,
                                              toggleVisibility_recursive)
                            ccHObject_recursive_call0(toggleColors,
                                                      toggleColors_recursive)
                                    ccHObject_recursive_call0(
                                            resetGLTransformationHistory,
                                            resetGLTransformationHistory_recursive)
                                            ccHObject_recursive_call0(
                                                    toggleNormals,
                                                    toggleNormals_recursive)
                                                    ccHObject_recursive_call0(
                                                            toggleSF,
                                                            toggleSF_recursive)
                                                            ccHObject_recursive_call0(
                                                                    toggleShowName,
                                                                    toggleShowName_recursive);

    //! Returns the max 'unique ID' of this entity and its siblings
    unsigned findMaxUniqueID_recursive() const;

    //! Applies the active OpenGL transformation to the entity (recursive)
    /** The input ccGLMatrix should be left to 0, unless you want to apply
            a pre-transformation.
            \param trans a ccGLMatrix structure (reference to)
    **/
    void applyGLTransformation_recursive(const ccGLMatrix* trans = nullptr);

    //! Notifies all dependent entities that the geometry of this entity has
    //! changed
    virtual void notifyGeometryUpdate();

    // inherited from ccSerializableObject
    bool isSerializable() const override;
    bool toFile(QFile& out) const override;
    bool fromFile(QFile& in,
                  short dataVersion,
                  int flags,
                  LoadedIDMap& oldToNewIDMap) override;

    //! Custom version of ccSerializableObject::fromFile
    /** This is used to load only the object's part of a stream (and not its
    children) \param in input file (already opened) \param dataVersion file
    version \param flags deserialization flags (see
    ccSerializableObject::DeserializationFlags) \return success
    **/
    bool fromFileNoChildren(QFile& in,
                            short dataVersion,
                            int flags,
                            LoadedIDMap& oldToNewIDMap);

    //! Returns whether object is shareable or not
    /** If object is father dependent and 'shared', it won't
            be deleted but 'released' instead.
    **/
    virtual inline bool isShareable() const { return false; }

    //! Behavior when selected
    enum SelectionBehavior {
        SELECTION_AA_BBOX,
        SELECTION_FIT_BBOX,
        SELECTION_IGNORED
    };

    //! Sets selection behavior (when displayed)
    /** WARNING: SELECTION_FIT_BBOX relies on the
            'ccDrawableObject::getFitBB' method (which
            is not supported by all entities).
    **/
    virtual inline void setSelectionBehavior(SelectionBehavior mode) {
        m_selectionBehavior = mode;
    }

    //! Returns selection behavior
    virtual inline SelectionBehavior getSelectionBehavior() const {
        return m_selectionBehavior;
    }

    //! Returns object unqiue ID used for display
    virtual inline unsigned getUniqueIDForDisplay() const {
        return getUniqueID();
    }

    //! Returns the transformation 'history' matrix
    virtual inline const ccGLMatrix& getGLTransformationHistory() const {
        return m_glTransHistory;
    }
    //! Sets the transformation 'history' matrix (handle with care!)
    virtual inline void setGLTransformationHistory(const ccGLMatrix& mat) {
        m_glTransHistory = mat;
    }
    //! Resets the transformation 'history' matrix
    virtual inline void resetGLTransformationHistory() {
        m_glTransHistory.toIdentity();
    }

public:
    //! Pushes the current display state (overridden)
    bool pushDisplayState() override;

    //! Pops the last pushed display state (overridden)
    void popDisplayState(bool apply = true) override;

protected:
    //! Sets parent object
    virtual inline void setParent(ccHObject* anObject) { m_parent = anObject; }

    //! Draws the entity only (not its children)
    virtual void drawMeOnly(
            CC_DRAW_CONTEXT& context) { /*does nothing by default*/
    }

    //! Applies a GL transformation to the entity
    /** this = rotMat*(this-rotCenter)+(rotCenter+trans)
            \param trans a ccGLMatrix structure
    **/
    virtual void applyGLTransformation(const ccGLMatrix& trans);

    //! Save own object data
    /** Called by 'toFile' (recursive scheme)
            To be overloaded (but still called;) by subclass.
    **/
    virtual bool toFile_MeOnly(QFile& out) const;

    //! Loads own object data
    /** Called by 'fromFile' (recursive scheme)
            To be overloaded (but still called;) by subclass.
            \param in input file
            \param dataVersion file version
            \param flags deserialization flags (see
    ccSerializableObject::DeserializationFlags)
    **/
    virtual bool fromFile_MeOnly(QFile& in,
                                 short dataVersion,
                                 int flags,
                                 LoadedIDMap& oldToNewIDMap);

    //! Draws the entity name in 3D
    /** Names is displayed at the center of the bounding box by default.
     **/
    virtual void drawNameIn3D();

    //! This method is called when another object is deleted
    /** For internal use.
     **/
    virtual void onDeletionOf(const ccHObject* obj);

    //! This method is called when another object (geometry) is updated
    /** For internal use.
     **/
    virtual void onUpdateOf(ccHObject* obj) { /*does nothing by default*/
    }

    //! Parent
    ccHObject* m_parent;

    //! Children
    Container m_children;

    //! Selection behavior
    SelectionBehavior m_selectionBehavior;

    //! Dependencies map
    /** First parameter: other object
            Second parameter: dependency flags (see DEPENDENCY_FLAGS)
    **/
    std::map<ccHObject*, int> m_dependencies;

    //! Cumulative GL transformation
    /** History of all the applied transformations since the creation of the
    object as a single transformation.
    **/
    ccGLMatrix m_glTransHistory;

    //! Flag to safely handle dependencies when the object is being deleted
    bool m_isDeleting;
};

/*** Helpers ***/

//! Puts all entities inside a container in a group
/** Automatically removes siblings so as to get a valid hierarchy object.
        \param origin origin container
        \param dest destination group
        \param dependencyFlags default dependency link for the children added to
the group
**/
inline void ConvertToGroup(const ccHObject::Container& origin,
                           ccHObject& dest,
                           int dependencyFlags = ccHObject::DP_NONE) {
    size_t count = origin.size();
    for (size_t i = 0; i < count; ++i) {
        // we don't take objects that are siblings of others
        bool isSiblingOfAnotherOne = false;
        for (size_t j = 0; j < count; ++j) {
            if (i != j && origin[j]->isAncestorOf(origin[i])) {
                isSiblingOfAnotherOne = true;
                break;
            }
        }

        if (!isSiblingOfAnotherOne) {
            dest.addChild(origin[i], dependencyFlags);
        }
    }
}

#endif  // ECV_HIERARCHY_OBJECT_HEADER

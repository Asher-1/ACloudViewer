// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPropertiesTreeDelegate.h"

// Local
#include "../MainWindow.h"
#include "CommonSettings.h"
#include "ecvAxesGridDialog.h"
#include "ecvColorScaleEditorDlg.h"
#include "ecvColorScaleSelector.h"
#include "ecvFileUtils.h"
#include "ecvOptions.h"
#include "ecvPersistentSettings.h"
#include "ecvSettingManager.h"
#include "ecvTextureFileSelector.h"
#include "matrixDisplayDlg.h"
#include "sfEditDlg.h"

// Note: PCL Selection Tools includes removed. Selection properties are now
// handled by cvFindDataDockWidget in a standalone dock, not integrated into
// the properties tree delegate.

// CV_DB_LIB
#include <ecv2DLabel.h>
#include <ecv2DViewportLabel.h>
#include <ecv2DViewportObject.h>
#include <ecvAdvancedTypes.h>
#include <ecvCameraSensor.h>
#include <ecvCircle.h>
#include <ecvColorScalesManager.h>
#include <ecvCone.h>
#include <ecvCoordinateSystem.h>
#include <ecvDisc.h>
#include <ecvDisplayTools.h>
#include <ecvDrawContext.h>
#include <ecvFacet.h>
#include <ecvGBLSensor.h>
#include <ecvGenericPrimitive.h>
#include <ecvGuiParameters.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvImage.h>
#include <ecvIndexedTransformationBuffer.h>
#include <ecvKdTree.h>
#include <ecvMaterialSet.h>
#include <ecvMesh.h>
#include <ecvOctree.h>
#include <ecvOctreeProxy.h>
#include <ecvPlane.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>
#include <ecvSensor.h>
#include <ecvSphere.h>
#include <ecvSubMesh.h>

// Qt
#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImageReader>
#include <QLineEdit>
#include <QLocale>
#include <QPushButton>
#include <QScrollBar>
#include <QSet>
#include <QSlider>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QToolButton>
#include <QTreeView>

// STL
#include <algorithm>
#include <exception>
#include <functional>

// System
#include <assert.h>

#include <cmath>

// Default 'None' string
const char* ccPropertiesTreeDelegate::s_noneString = QT_TR_NOOP("None");

// Default color sources string
const char* ccPropertiesTreeDelegate::s_rgbColor = "RGB";
const char* ccPropertiesTreeDelegate::s_sfColor = QT_TR_NOOP("Scalar field");

// Other strings
const char* ccPropertiesTreeDelegate::s_defaultPointSizeString =
        QT_TR_NOOP("Default");
const char* ccPropertiesTreeDelegate::s_defaultPolyWidthSizeString =
        QT_TR_NOOP("Default Width");

// Default separator colors
constexpr const char* SEPARATOR_STYLESHEET(
        "QLabel { background-color : darkGray; color : white; }");

// Shortcut to create a delegate item
QStandardItem* ITEM(QString name,
                    Qt::ItemFlag additionalFlags = Qt::NoItemFlags,
                    ccPropertiesTreeDelegate::CC_PROPERTY_ROLE role =
                            ccPropertiesTreeDelegate::OBJECT_NO_PROPERTY) {
    QStandardItem* item = new QStandardItem(name);
    // flags
    item->setFlags(Qt::ItemIsEnabled | additionalFlags);
    // role (if any)
    if (role != ccPropertiesTreeDelegate::OBJECT_NO_PROPERTY)
        item->setData(role);

    return item;
}

// Shortcut to create a checkable delegate item
QStandardItem* CHECKABLE_ITEM(bool checkState,
                              ccPropertiesTreeDelegate::CC_PROPERTY_ROLE role) {
    QStandardItem* item = ITEM("", Qt::ItemIsUserCheckable, role);
    // check state
    item->setCheckState(checkState ? Qt::Checked : Qt::Unchecked);

    return item;
}

// Shortcut to create a persistent editor item
QStandardItem* PERSISTENT_EDITOR(
        ccPropertiesTreeDelegate::CC_PROPERTY_ROLE role) {
    return ITEM(QString(), Qt::ItemIsEditable, role);
}

ccPropertiesTreeDelegate::ccPropertiesTreeDelegate(QStandardItemModel* model,
                                                   QAbstractItemView* view,
                                                   QObject* parent)
    : QStyledItemDelegate(parent),
      m_currentObject(nullptr),
      m_model(model),
      m_view(view),
      m_viewer(nullptr),
      m_lastFocusItemRole(OBJECT_NO_PROPERTY) {
    // Note: Selection properties are now handled by cvFindDataDockWidget,
    // a standalone dock widget that is decoupled from the properties tree.
    assert(m_model && m_view);
}

ccPropertiesTreeDelegate::~ccPropertiesTreeDelegate() { unbind(); }

QSize ccPropertiesTreeDelegate::sizeHint(const QStyleOptionViewItem& option,
                                         const QModelIndex& index) const {
    assert(m_model);

    QStandardItem* item = m_model->itemFromIndex(index);

    if (item && item->data().isValid()) {
        switch (item->data().toInt()) {
            case OBJECT_CURRENT_DISPLAY:
            case OBJECT_CURRENT_SCALAR_FIELD:
            case OBJECT_OCTREE_TYPE:
            case OBJECT_COLOR_RAMP_STEPS:
            case OBJECT_CLOUD_POINT_SIZE:
            case OBJECT_OPACITY:
                return QSize(50, 24);
            case OBJECT_COLOR_SOURCE:
            case OBJECT_POLYLINE_WIDTH:
            case OBJECT_MESH_TEXTUREFILE:
            case OBJECT_CURRENT_COLOR_RAMP:
                return QSize(70, 24);
            case OBJECT_CLOUD_SF_EDITOR:
                return QSize(250, 200);
            case OBJECT_SENSOR_MATRIX_EDITOR:
            case OBJECT_HISTORY_MATRIX_EDITOR:
            case OBJECT_GLTRANS_MATRIX_EDITOR:
                return QSize(250, 140);
                // Note: OBJECT_SELECTION_PROPERTIES case removed - selection
                // properties are now in standalone cvFindDataDockWidget
        }
    }

    return QStyledItemDelegate::sizeHint(option, index);
}

void ccPropertiesTreeDelegate::unbind() {
    if (m_model) m_model->disconnect(this);
    // Clear texture path maps when unbinding
    // This ensures we don't keep stale references to removed objects
    m_meshTexturePathMaps.clear();
}

ccHObject* ccPropertiesTreeDelegate::getCurrentObject() {
    return m_currentObject;
}

void ccPropertiesTreeDelegate::fillModel(ccHObject* hObject) {
    if (!hObject) {
        CVLog::Print(
                "[ccPropertiesTreeDelegate::fillModel] Called with nullptr, "
                "clearing");
        unbind();
        if (m_model) {
            m_model->removeRows(0, m_model->rowCount());
        }
        m_currentObject = nullptr;
        return;
    }

    unbind();

    m_currentObject = hObject;

    // save current scroll position
    int scrollPos = (m_view && m_view->verticalScrollBar()
                             ? m_view->verticalScrollBar()->value()
                             : 0);

    if (m_model) {
        m_model->removeRows(0, m_model->rowCount());
        m_model->setColumnCount(2);
        m_model->setHeaderData(0, Qt::Horizontal, tr("Property"));
        m_model->setHeaderData(1, Qt::Horizontal, tr("State/Value"));
    }

    // Ensure header is visible when displaying normal properties
    // (it may have been hidden when showing only selection properties)
    if (m_view) {
        QTreeView* treeView = qobject_cast<QTreeView*>(m_view);
        if (treeView && treeView->header()) {
            treeView->header()->show();
        }
    }

    // Note: Selection properties are no longer shown in the properties tree.
    // They are now displayed in the standalone cvFindDataDockWidget.

    if (m_currentObject->isHierarchy())
        if (!m_currentObject->isA(
                    CV_TYPES::VIEWPORT_2D_LABEL))  // don't need to display this
                                                   // kind of info for viewport
                                                   // labels!
            fillWithHObject(m_currentObject);

    // View properties (ParaView-style) - added right after ECV Object section
    if (m_currentObject->getViewId().length() > 0) {
        fillWithViewProperties();
    }

    if (m_currentObject->isA(CV_TYPES::COORDINATESYSTEM)) {
        fillWithCoordinateSystem(
                ccHObjectCaster::ToCoordinateSystem(m_currentObject));
    } else if (m_currentObject->isKindOf(CV_TYPES::POINT_CLOUD)) {
        fillWithPointCloud(
                ccHObjectCaster::ToGenericPointCloud(m_currentObject));
    } else if (m_currentObject->isKindOf(CV_TYPES::MESH)) {
        fillWithMesh(ccHObjectCaster::ToGenericMesh(m_currentObject));

        if (m_currentObject->isKindOf(CV_TYPES::PRIMITIVE)) {
            fillWithPrimitive(ccHObjectCaster::ToPrimitive(m_currentObject));
        }
    } else if (m_currentObject->isA(CV_TYPES::FACET)) {
        fillWithFacet(ccHObjectCaster::ToFacet(m_currentObject));
    } else if (m_currentObject->isA(CV_TYPES::POLY_LINE)) {
        fillWithPolyline(ccHObjectCaster::ToPolyline(m_currentObject));
    } else if (m_currentObject->isA(CV_TYPES::POINT_OCTREE)) {
        fillWithPointOctree(ccHObjectCaster::ToOctree(m_currentObject));
    } else if (m_currentObject->isA(CV_TYPES::POINT_KDTREE)) {
        fillWithPointKdTree(ccHObjectCaster::ToKdTree(m_currentObject));
    } else if (m_currentObject->isKindOf(CV_TYPES::IMAGE)) {
        fillWithImage(ccHObjectCaster::ToImage(m_currentObject));
    } else if (m_currentObject->isA(CV_TYPES::LABEL_2D)) {
        fillWithLabel(ccHObjectCaster::To2DLabel(m_currentObject));
    } else if (m_currentObject->isKindOf(CV_TYPES::VIEWPORT_2D_OBJECT)) {
        fillWithViewportObject(
                ccHObjectCaster::To2DViewportObject(m_currentObject));
    } else if (m_currentObject->isKindOf(CV_TYPES::GBL_SENSOR)) {
        fillWithGBLSensor(ccHObjectCaster::ToGBLSensor(m_currentObject));
    } else if (m_currentObject->isKindOf(CV_TYPES::CAMERA_SENSOR)) {
        fillWithCameraSensor(ccHObjectCaster::ToCameraSensor(m_currentObject));
    } else if (m_currentObject->isA(CV_TYPES::MATERIAL_SET)) {
        fillWithMaterialSet(static_cast<ccMaterialSet*>(m_currentObject));
    } else if (m_currentObject->isA(CV_TYPES::NORMAL_INDEXES_ARRAY)) {
        fillWithCCArray(static_cast<NormsIndexesTableType*>(m_currentObject));
    } else if (m_currentObject->isA(CV_TYPES::TEX_COORDS_ARRAY)) {
        fillWithCCArray(static_cast<TextureCoordsContainer*>(m_currentObject));
    } else if (m_currentObject->isA(CV_TYPES::NORMALS_ARRAY)) {
        fillWithCCArray(static_cast<NormsTableType*>(m_currentObject));
    } else if (m_currentObject->isA(CV_TYPES::RGB_COLOR_ARRAY)) {
        fillWithCCArray(static_cast<ColorsTableType*>(m_currentObject));
    } else if (m_currentObject->isA(CV_TYPES::TRANS_BUFFER)) {
        fillWithTransBuffer(
                static_cast<ccIndexedTransformationBuffer*>(m_currentObject));
    }

    // transformation history
    if (m_currentObject->isKindOf(CV_TYPES::POINT_CLOUD) ||
        m_currentObject->isKindOf(CV_TYPES::MESH) ||
        m_currentObject->isKindOf(CV_TYPES::FACET) ||
        m_currentObject->isKindOf(CV_TYPES::POLY_LINE) ||
        m_currentObject->isKindOf(CV_TYPES::SENSOR)) {
        addSeparator(tr("Transformation history"));
        appendWideRow(PERSISTENT_EDITOR(OBJECT_HISTORY_MATRIX_EDITOR));

        if (m_currentObject->isGLTransEnabled()) {
            addSeparator(tr("Display transformation"));
            appendWideRow(PERSISTENT_EDITOR(OBJECT_GLTRANS_MATRIX_EDITOR));
        }
    }

    // meta-data
    fillWithMetaData(m_currentObject);

    // go back to original position
    if (scrollPos > 0)
        m_view->verticalScrollBar()->setSliderPosition(scrollPos);

    if (m_model) {
        connect(m_model, &QStandardItemModel::itemChanged, this,
                &ccPropertiesTreeDelegate::updateItem);
    }
}

void ccPropertiesTreeDelegate::appendRow(QStandardItem* leftItem,
                                         QStandardItem* rightItem,
                                         bool openPersistentEditor /*=false*/) {
    assert(leftItem && rightItem);
    assert(m_model);

    if (m_model) {
        // append row
        QList<QStandardItem*> rowItems;
        {
            rowItems.push_back(leftItem);
            rowItems.push_back(rightItem);
        }
        m_model->appendRow(rowItems);

        // the persistent editor (if any) is always the right one!
        if (openPersistentEditor)
            m_view->openPersistentEditor(
                    m_model->index(m_model->rowCount() - 1, 1));
    }
}

void ccPropertiesTreeDelegate::appendWideRow(
        QStandardItem* item, bool openPersistentEditor /*=true*/) {
    assert(item);
    assert(m_model);

    if (m_model) {
        m_model->appendRow(item);
        if (openPersistentEditor && m_view) {
            QModelIndex index = m_model->index(m_model->rowCount() - 1, 0);
            if (index.isValid()) {
                m_view->openPersistentEditor(index);
            } else {
                CVLog::Warning(
                        "[ccPropertiesTreeDelegate] Invalid index for "
                        "persistent editor");
            }
        }
    }
}

void ccPropertiesTreeDelegate::addSeparator(QString title) {
    if (m_model) {
        // DGM: we can't use the 'text' of the item as it will be displayed
        // under the associated editor (label)! So we simply use the 'accessible
        // description' field
        QStandardItem* leftItem = new QStandardItem(/*title*/);
        leftItem->setData(TREE_VIEW_HEADER);
        leftItem->setAccessibleDescription(title);
        m_model->appendRow(leftItem);
        m_view->openPersistentEditor(
                m_model->index(m_model->rowCount() - 1, 0));
    }
}

void ccPropertiesTreeDelegate::fillWithMetaData(ccObject* _obj) {
    assert(_obj && m_model);

    const QVariantMap& metaData = _obj->metaData();
    if (metaData.size() == 0) return;

    addSeparator(tr("Meta data"));

    for (QVariantMap::ConstIterator it = metaData.constBegin();
         it != metaData.constEnd(); ++it) {
        QVariant var = it.value();
        QString value;

        if (var.canConvert(QVariant::String)) {
            var.convert(QVariant::String);
            value = var.toString();
        } else {
            value = QString(QVariant::typeToName(static_cast<int>(var.type())));
        }

        appendRow(ITEM(it.key()), ITEM(value));
    }
}

void ccPropertiesTreeDelegate::fillWithViewProperties() {
    assert(m_model);

    // ParaView-style view properties section
    addSeparator(tr("View (Render View)"));

    // 1. Light Intensity - ParaView-style direct control (no checkbox)
    // ParaView uses LightIntensity property (0.0-2.0 range)
    appendRow(ITEM(tr("Light Intensity")),
              PERSISTENT_EDITOR(OBJECT_VIEW_LIGHT_KIT_INTENSITY), true);

    // 2. Opacity - moved from ECV Object section (ParaView has this in View
    // properties) Show for renderable objects and folders (to control all
    // children)
    if (m_currentObject) {
        bool isRenderable = (m_currentObject->isKindOf(CV_TYPES::POINT_CLOUD) ||
                             m_currentObject->isKindOf(CV_TYPES::MESH) ||
                             m_currentObject->isKindOf(CV_TYPES::PRIMITIVE) ||
                             m_currentObject->isKindOf(CV_TYPES::POLY_LINE) ||
                             m_currentObject->isKindOf(CV_TYPES::FACET));
        bool isFolder = (m_currentObject->getChildrenNumber() > 0);

        if (isRenderable || isFolder) {
            appendRow(ITEM(tr("Opacity")), PERSISTENT_EDITOR(OBJECT_OPACITY),
                      true);
        }
    }

    // 3. Data Axes Grid - ParaView-style: checkbox with integrated Edit button
    // This shows the coordinate axes for the data bounds of the current object
    {
        // Get current visibility from backend
        bool visible = false;
        if (m_currentObject) {
            QString viewID = m_currentObject->getViewId();
            AxesGridProperties props;
            ecvDisplayTools::TheInstance()->getDataAxesGridProperties(viewID,
                                                                      props);
            visible = props.visible;
        }

        // ParaView-style: Checkbox and Edit button in same row (like Opacity's
        // slider+spinbox) We create a custom editor that combines checkbox +
        // button using PERSISTENT_EDITOR
        appendRow(ITEM(tr("Show Axes Grid")),
                  PERSISTENT_EDITOR(OBJECT_VIEW_DATA_AXES_GRID_VISIBLE), true);
    }
}

// Note: fillWithSelectionProperties, setSelectionToolsActive, and
// showSelectionPropertiesOnly have been removed. Selection properties are now
// displayed in the standalone cvFindDataDockWidget, which is decoupled from
// the properties tree and selection tool state (following ParaView design).

void ccPropertiesTreeDelegate::clearModel() {
    if (!m_model) return;

    unbind();
    m_model->removeRows(0, m_model->rowCount());
    m_currentObject = nullptr;
}

void ccPropertiesTreeDelegate::fillWithHObject(ccHObject* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("ECV Object"));

    // name
    appendRow(ITEM(tr("Name")),
              ITEM(_obj->getName(), Qt::ItemIsEditable, OBJECT_NAME));

    // visibility
    if (!_obj->isVisibilityLocked())
        appendRow(ITEM(tr("Visible")),
                  CHECKABLE_ITEM(_obj->isVisible(), OBJECT_VISIBILITY));

    // normals
    if (_obj->hasNormals())
        appendRow(ITEM(tr("Normals")),
                  CHECKABLE_ITEM(_obj->normalsShown(), OBJECT_NORMALS_SHOWN));

    // name in 3D
    appendRow(ITEM(tr("Show name (in 3D)")),
              CHECKABLE_ITEM(_obj->nameShownIn3D(), OBJECT_NAME_IN_3D));

    // color source
    if (_obj->hasColors() || _obj->hasScalarFields())
        appendRow(ITEM(tr("Colors")), PERSISTENT_EDITOR(OBJECT_COLOR_SOURCE),
                  true);

    // Bounding-box
    {
        ccBBox box;
        bool fitBBox = false;
        if (_obj->getSelectionBehavior() == ccHObject::SELECTION_FIT_BBOX) {
            ccGLMatrix trans;
            box = _obj->getOwnFitBB(trans);
            box += trans.getTranslationAsVec3D();
            fitBBox = true;
        } else {
            box = _obj->getBB_recursive();
        }

        if (box.isValid()) {
            // Box dimensions
            CCVector3 bboxDiag = box.getDiagVec();
            appendRow(ITEM(fitBBox ? tr("Local box dimensions")
                                   : tr("Box dimensions")),
                      ITEM(QString("X: %0\nY: %1\nZ: %2")
                                   .arg(bboxDiag.x)
                                   .arg(bboxDiag.y)
                                   .arg(bboxDiag.z)));

            // Box center
            CCVector3 bboxCenter = box.getCenter();

            ccShiftedObject* shiftedObj = ccHObjectCaster::ToShifted(_obj);

            // local bounding box center
            appendRow(ITEM(shiftedObj ? tr("Shifted box center")
                                      : tr("Box center")),
                      ITEM(QStringLiteral("X: %0\nY: %1\nZ: %2")
                                   .arg(bboxCenter.x)
                                   .arg(bboxCenter.y)
                                   .arg(bboxCenter.z)));

            if (shiftedObj) {
                CCVector3d globalBBoxCenter =
                        shiftedObj->toGlobal3d(bboxCenter);

                // global bounding box center
                appendRow(ITEM(tr("Global box center")),
                          ITEM(QStringLiteral("X: %0\nY: %1\nZ: %2")
                                       .arg(globalBBoxCenter.x, 0, 'f')
                                       .arg(globalBBoxCenter.y, 0, 'f')
                                       .arg(globalBBoxCenter.z, 0, 'f')));
            }
        }
    }

    // infos (unique ID, children) //DGM: on the same line so as to gain space
    appendRow(ITEM(tr("Info")), ITEM(tr("Object ID: %1 - Children: %2")
                                             .arg(_obj->getUniqueID())
                                             .arg(_obj->getChildrenNumber())));

    // Note: Opacity has been moved to View (Render View) section
    // following ParaView's property panel layout

    // display window
    if (!_obj->isLocked())
        appendRow(ITEM(tr("Current Display")),
                  PERSISTENT_EDITOR(OBJECT_CURRENT_DISPLAY), true);
}

void ccPropertiesTreeDelegate::fillWithShifted(ccShiftedObject* _obj) {
    assert(_obj && m_model);

    // global shift & scale
    const CCVector3d& shift = _obj->getGlobalShift();
    appendRow(ITEM(tr("Global shift")), ITEM(QString("(%1;%2;%3)")
                                                     .arg(shift.x, 0, 'f', 2)
                                                     .arg(shift.y, 0, 'f', 2)
                                                     .arg(shift.z, 0, 'f', 2)));

    double scale = _obj->getGlobalScale();
    appendRow(ITEM(tr("Global scale")),
              ITEM(QString("%1").arg(scale, 0, 'f', 6)));
}

void ccPropertiesTreeDelegate::fillWithCoordinateSystem(
        const ccCoordinateSystem* _obj) {
    assert(_obj && m_model);
    if (!_obj || !m_model) {
        return;
    }

    CCVector3 origin = _obj->getOrigin();
    addSeparator(tr("Coordinate System"));
    appendRow(ITEM(tr("Origin")), ITEM(QStringLiteral("X: %0\nY: %1\nZ: %2")
                                               .arg(origin.x)
                                               .arg(origin.y)
                                               .arg(origin.z)));
    appendRow(ITEM(tr("Planes Visible")),
              CHECKABLE_ITEM(_obj->axisPlanesAreShown(),
                             OBJECT_COORDINATE_SYSTEM_DISP_PLANES));
    appendRow(
            ITEM(tr("Planes Stippled")),
            CHECKABLE_ITEM(static_cast<const ccMesh*>(_obj)->stipplingEnabled(),
                           OBJECT_MESH_STIPPLING));
    appendRow(ITEM(tr("Axis Lines Visible")),
              CHECKABLE_ITEM(_obj->axisLinesAreShown(),
                             OBJECT_COORDINATE_SYSTEM_DISP_AXES));
    appendRow(ITEM(tr("Axis width")),
              PERSISTENT_EDITOR(OBJECT_COORDINATE_SYSTEM_AXES_WIDTH), true);
    appendRow(ITEM(tr("Display scale")),
              PERSISTENT_EDITOR(OBJECT_COORDINATE_SYSTEM_DISP_SCALE), true);
}

void ccPropertiesTreeDelegate::fillWithPointCloud(ccGenericPointCloud* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Cloud"));

    // number of points
    appendRow(ITEM(tr("Points")),
              ITEM(QLocale(QLocale::English).toString(_obj->size())));

    // global shift & scale
    fillWithShifted(_obj);

    // custom point size
    appendRow(ITEM(tr("Point size")),
              PERSISTENT_EDITOR(OBJECT_CLOUD_POINT_SIZE), true);

    // scalar field
    fillSFWithPointCloud(_obj);

    // scan grid structure(s), waveform, etc.
    if (_obj->isA(CV_TYPES::POINT_CLOUD)) {
        ccPointCloud* cloud = static_cast<ccPointCloud*>(_obj);

        // scan grid(s)
        size_t gridCount = cloud->gridCount();
        if (gridCount != 0) {
            if (gridCount != 1)
                addSeparator(tr("Scan grids"));
            else
                addSeparator(tr("Scan grid"));

            for (size_t i = 0; i < gridCount; ++i) {
                // grid size + valid point count
                ccPointCloud::Grid::Shared grid = cloud->grid(i);
                appendRow(
                        ITEM(tr("Scan #%1").arg(i + 1)),
                        ITEM(tr("%1 x %2 (%3 points)")
                                     .arg(grid->w)
                                     .arg(grid->h)
                                     .arg(QLocale(QLocale::English)
                                                  .toString(
                                                          grid->validCount))));
            }
        }

        // waveform
        if (cloud->hasFWF()) {
            addSeparator(tr("Waveform"));
            appendRow(ITEM(tr("Waves")),
                      ITEM(QString::number(
                              cloud->waveforms()
                                      .size())));  // DGM: in fact some of them
                                                   // might be null/invalid!
            appendRow(ITEM(tr("Descriptors")),
                      ITEM(QString::number(cloud->fwfDescriptors().size())));

            double dataSize_mb =
                    (cloud->fwfData() ? cloud->fwfData()->size() : 0) /
                    static_cast<double>(1 << 20);
            appendRow(ITEM(tr("Data size")),
                      ITEM(QString("%1 Mb").arg(dataSize_mb, 0, 'f', 2)));
        }
    }
}

void ccPropertiesTreeDelegate::fillSFWithPointCloud(ccGenericPointCloud* _obj) {
    assert(m_model);

    // for "real" point clouds only
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(_obj);
    if (!cloud) return;

    // Scalar fields
    unsigned sfCount = cloud->getNumberOfScalarFields();
    if (sfCount != 0) {
        addSeparator(sfCount > 1 ? tr("Scalar Fields") : tr("Scalar Field"));

        // fields number
        appendRow(ITEM(tr("Count")), ITEM(QString::number(sfCount)));

        // fields list combo
        appendRow(ITEM(tr("Active")),
                  PERSISTENT_EDITOR(OBJECT_CURRENT_SCALAR_FIELD), true);

        // no need to go any further if no SF is currently active
        cloudViewer::ScalarField* sf = cloud->getCurrentDisplayedScalarField();
        if (sf) {
            addSeparator(tr("Color Scale"));

            // color scale selection combo box
            appendRow(ITEM(tr("Current")),
                      PERSISTENT_EDITOR(OBJECT_CURRENT_COLOR_RAMP), true);

            // color scale steps
            appendRow(ITEM(tr("Steps")),
                      PERSISTENT_EDITOR(OBJECT_COLOR_RAMP_STEPS), true);

            // scale visible?
            appendRow(ITEM(tr("Visible")),
                      CHECKABLE_ITEM(cloud->sfColorScaleShown(),
                                     OBJECT_SF_SHOW_SCALE));

            addSeparator(tr("SF display params"));

            // SF edit dialog (warning: 2 columns)
            appendWideRow(PERSISTENT_EDITOR(OBJECT_CLOUD_SF_EDITOR));
        }
    }
}

void ccPropertiesTreeDelegate::fillWithPrimitive(ccGenericPrimitive* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Primitive"));

    // type
    appendRow(ITEM(tr("Type")), ITEM(_obj->getTypeName()));

    // drawing steps
    if (_obj->hasDrawingPrecision()) {
        appendRow(ITEM(tr("Drawing precision")),
                  PERSISTENT_EDITOR(OBJECT_PRIMITIVE_PRECISION), true);
    }

    if (_obj->isA(CV_TYPES::SPHERE)) {
        appendRow(ITEM(tr("Radius")), PERSISTENT_EDITOR(OBJECT_SPHERE_RADIUS),
                  true);
    } else if (_obj->isKindOf(CV_TYPES::CONE))  // cylinders are also cones!
    {
        appendRow(ITEM(tr("Height")), PERSISTENT_EDITOR(OBJECT_CONE_HEIGHT),
                  true);
        if (_obj->isA(CV_TYPES::CYLINDER)) {
            appendRow(ITEM(tr("Radius")),
                      PERSISTENT_EDITOR(OBJECT_CONE_BOTTOM_RADIUS), true);
        } else {
            appendRow(ITEM(tr("Bottom radius")),
                      PERSISTENT_EDITOR(OBJECT_CONE_BOTTOM_RADIUS), true);
            appendRow(ITEM(tr("Top radius")),
                      PERSISTENT_EDITOR(OBJECT_CONE_TOP_RADIUS), true);
        }
    } else if (_obj->isKindOf(CV_TYPES::PLANE)) {
        // planar entity commons
        fillWithPlanarEntity(static_cast<ccPlane*>(_obj));
    } else if (_obj->isA(CV_TYPES::DISC)) {
        appendRow(ITEM(tr("Radius")), PERSISTENT_EDITOR(OBJECT_DISC_RADIUS),
                  true);
    }
}

void ccPropertiesTreeDelegate::fillWithFacet(ccFacet* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Facet"));

    // planar entity commons
    fillWithPlanarEntity(_obj);

    // surface
    appendRow(ITEM(tr("Surface")),
              ITEM(QLocale(QLocale::English).toString(_obj->getSurface())));

    // RMS
    appendRow(ITEM(tr("RMS")),
              ITEM(QLocale(QLocale::English).toString(_obj->getRMS())));

    // center
    appendRow(ITEM(tr("Center")), ITEM(QString("(%1 ; %2 ; %3)")
                                               .arg(_obj->getCenter().x)
                                               .arg(_obj->getCenter().y)
                                               .arg(_obj->getCenter().z)));

    // contour visibility
    if (_obj->getContour())
        appendRow(ITEM(tr("Show contour")),
                  CHECKABLE_ITEM(_obj->getContour()->isVisible(),
                                 OBJECT_FACET_CONTOUR));

    // polygon visibility
    if (_obj->getPolygon())
        appendRow(ITEM(tr("Show polygon")),
                  CHECKABLE_ITEM(_obj->getPolygon()->isVisible(),
                                 OBJECT_FACET_MESH));
}

void ccPropertiesTreeDelegate::fillWithPlanarEntity(
        ccPlanarEntityInterface* _obj) {
    // normal
    CCVector3 N = _obj->getNormal();
    appendRow(ITEM(tr("Normal")),
              ITEM(QString("(%1 ; %2 ; %3)").arg(N.x).arg(N.y).arg(N.z)));

    // Dip & Dip direction (in degrees)
    PointCoordinateType dip_deg, dipDir_deg;
    ccNormalVectors::ConvertNormalToDipAndDipDir(N, dip_deg, dipDir_deg);
    appendRow(ITEM(tr("Dip / Dip dir.")),
              ITEM(QString("(%1 ; %2) deg.")
                           .arg(static_cast<int>(dip_deg))
                           .arg(static_cast<int>(dipDir_deg))));

    // normal vector visibility
    appendRow(ITEM(tr("Show normal vector")),
              CHECKABLE_ITEM(_obj->normalVectorIsShown(),
                             OBJECT_PLANE_NORMAL_VECTOR));
}

void ccPropertiesTreeDelegate::fillWithMesh(ccGenericMesh* _obj) {
    assert(_obj && m_model);

    bool isSubMesh = _obj->isA(CV_TYPES::SUB_MESH);

    addSeparator(isSubMesh ? tr("Sub-mesh") : tr("Mesh"));

    // number of facets
    appendRow(ITEM(tr("Faces")),
              ITEM(QLocale(QLocale::English).toString(_obj->size())));

    // wireframe
    appendRow(ITEM(tr("Wireframe")),
              CHECKABLE_ITEM(_obj->isShownAsWire(), OBJECT_MESH_WIRE));

    // Pointsframe
    appendRow(ITEM(tr("Pointsframe")),
              CHECKABLE_ITEM(_obj->isShownAsPoints(), OBJECT_MESH_POINTS));

    // stippling (ccMesh only)
    // if (_obj->isA(CV_TYPES::MESH)) //DGM: can't remember why?
    appendRow(ITEM(tr("Stippling")),
              CHECKABLE_ITEM(static_cast<ccMesh*>(_obj)->stipplingEnabled(),
                             OBJECT_MESH_STIPPLING));

    // material/texture
    if (_obj->hasMaterials()) {
        appendRow(ITEM(tr("Materials/textures")),
                  CHECKABLE_ITEM(_obj->materialsShown(), OBJECT_MATERIALS));
        // texture file selection combo box
        appendRow(ITEM(tr("Texturefile")),
                  PERSISTENT_EDITOR(OBJECT_MESH_TEXTUREFILE), true);
    }

    // we also integrate vertices SF into mesh properties
    ccGenericPointCloud* vertices = _obj->getAssociatedCloud();
    if (vertices && (!vertices->isLocked() || _obj->isAncestorOf(vertices)))
        fillSFWithPointCloud(vertices);
}

void ccPropertiesTreeDelegate::fillWithPolyline(ccPolyline* _obj) {
    assert(_obj && m_model);
    if (!_obj || !m_model) {
        return;
    }

    if (_obj->isA(CV_TYPES::CIRCLE)) {
        addSeparator(tr("Circle"));

        appendRow(ITEM(tr("Drawing precision")),
                  PERSISTENT_EDITOR(OBJECT_CIRCLE_RESOLUTION), true);

        appendRow(ITEM(tr("Radius")), PERSISTENT_EDITOR(OBJECT_CIRCLE_RADIUS),
                  true);
    }

    addSeparator(tr("Polyline"));

    // number of vertices
    appendRow(ITEM(tr("Vertices")),
              ITEM(QLocale(QLocale::English).toString(_obj->size())));

    // polyline length
    appendRow(ITEM(tr("Length")),
              ITEM(QLocale(QLocale::English).toString(_obj->computeLength())));

    // custom line width
    appendRow(ITEM(tr("Line width")), PERSISTENT_EDITOR(OBJECT_POLYLINE_WIDTH),
              true);

    // global shift & scale
    fillWithShifted(_obj);
}

void ccPropertiesTreeDelegate::fillWithPointOctree(ccOctree* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Octree"));

    // display mode
    appendRow(ITEM(tr("Display mode")), PERSISTENT_EDITOR(OBJECT_OCTREE_TYPE),
              true);

    // level
    appendRow(ITEM(tr("Display level")), PERSISTENT_EDITOR(OBJECT_OCTREE_LEVEL),
              true);

    addSeparator(tr("Current level"));

    // current display level
    int level = _obj->getDisplayedLevel();
    assert(level > 0 && level <= ccOctree::MAX_OCTREE_LEVEL);

    // cell size
    PointCoordinateType cellSize =
            _obj->getCellSize(static_cast<unsigned char>(level));
    appendRow(ITEM(tr("Cell size")), ITEM(QString::number(cellSize)));

    // cell count
    unsigned cellCount = _obj->getCellNumber(static_cast<unsigned char>(level));
    appendRow(ITEM(tr("Cell count")),
              ITEM(QLocale(QLocale::English).toString(cellCount)));

    // total volume of filled cells
    appendRow(ITEM(tr("Filled volume")),
              ITEM(QString::number((double)cellCount *
                                   pow((double)cellSize, 3.0))));
}

void ccPropertiesTreeDelegate::fillWithPointKdTree(ccKdTree* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Kd-tree"));

    // max error
    appendRow(ITEM(tr("Max Error")),
              ITEM(QString::number(_obj->getMaxError())));
    // max error measure
    {
        QString errorMeasure;
        switch (_obj->getMaxErrorType()) {
            case cloudViewer::DistanceComputationTools::RMS:
                errorMeasure = tr("RMS");
                break;
            case cloudViewer::DistanceComputationTools::MAX_DIST_68_PERCENT:
                errorMeasure = tr("Max dist @ 68%");
                break;
            case cloudViewer::DistanceComputationTools::MAX_DIST_95_PERCENT:
                errorMeasure = tr("Max dist @ 95%");
                break;
            case cloudViewer::DistanceComputationTools::MAX_DIST_99_PERCENT:
                errorMeasure = tr("Max dist @ 99%");
                break;
            case cloudViewer::DistanceComputationTools::MAX_DIST:
                errorMeasure = tr("Max distance");
                break;
            default:
                assert(false);
                errorMeasure = tr("unknown");
                break;
        }
        appendRow(ITEM(tr("Error measure")), ITEM(errorMeasure));
    }
}

void ccPropertiesTreeDelegate::fillWithImage(ccImage* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Image"));

    // image width
    appendRow(ITEM(tr("Width")), ITEM(QString::number(_obj->getW())));

    // image height
    appendRow(ITEM(tr("Height")), ITEM(QString::number(_obj->getH())));

    // transparency
    appendRow(ITEM(tr("Alpha")), PERSISTENT_EDITOR(OBJECT_IMAGE_ALPHA), true);

    if (_obj->getAssociatedSensor()) {
        addSeparator(tr("Sensor"));
        //"Set Viewport" button (shortcut to associated sensor)
        appendRow(ITEM(tr("Apply Viewport")),
                  PERSISTENT_EDITOR(OBJECT_APPLY_IMAGE_VIEWPORT), true);
    }
}

void ccPropertiesTreeDelegate::fillWithLabel(cc2DLabel* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Label"));

    // Body
    QStringList body =
            _obj->getLabelContent(ecvGui::Parameters().displayedNumPrecision);
    appendRow(ITEM(tr("Body")), ITEM(body.join("\n")));

    // Show label in 2D
    appendRow(ITEM(tr("Show 2D label")),
              CHECKABLE_ITEM(_obj->isDisplayedIn2D(), OBJECT_LABEL_DISP_2D));

    // Show label in 3D
    appendRow(ITEM(tr("Show legend(s)")),
              CHECKABLE_ITEM(_obj->isPointLegendDisplayed(),
                             OBJECT_LABEL_POINT_LEGEND));
}

void ccPropertiesTreeDelegate::fillWithViewportObject(
        cc2DViewportObject* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Viewport"));

    // Name
    appendRow(ITEM(tr("Name")),
              ITEM(_obj->getName().isEmpty() ? tr("undefined")
                                             : _obj->getName()));

    //"Apply Viewport" button
    appendRow(ITEM(tr("Apply viewport")),
              PERSISTENT_EDITOR(OBJECT_APPLY_LABEL_VIEWPORT), true);

    //"Update Viewport" button
    appendRow(ITEM(tr("Update viewport")),
              PERSISTENT_EDITOR(OBJECT_UPDATE_LABEL_VIEWPORT), true);
}

void ccPropertiesTreeDelegate::fillWithTransBuffer(
        ccIndexedTransformationBuffer* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Trans. buffer"));

    // Associated positions
    appendRow(ITEM(tr("Count")), ITEM(QString::number(_obj->size())));

    // Show path as polyline
    appendRow(ITEM(tr("Show path")),
              CHECKABLE_ITEM(_obj->isPathShonwAsPolyline(),
                             OBJECT_SHOW_TRANS_BUFFER_PATH));

    // Show trihedrons
    appendRow(ITEM(tr("Show trihedrons")),
              CHECKABLE_ITEM(_obj->triherdonsShown(),
                             OBJECT_SHOW_TRANS_BUFFER_TRIHDERONS));

    // Trihedrons scale
    appendRow(ITEM(tr("Scale")),
              PERSISTENT_EDITOR(OBJECT_TRANS_BUFFER_TRIHDERONS_SCALE), true);
}

void ccPropertiesTreeDelegate::fillWithSensor(ccSensor* _obj) {
    assert(_obj && m_model);

    // Sensor drawing scale
    appendRow(ITEM(tr("Drawing scale")),
              PERSISTENT_EDITOR(OBJECT_SENSOR_DISPLAY_SCALE), true);

    //"Apply Viewport" button
    appendRow(ITEM(tr("Apply Viewport")),
              PERSISTENT_EDITOR(OBJECT_APPLY_SENSOR_VIEWPORT), true);

    // sensor aboslute orientation
    addSeparator(tr("Position/Orientation"));
    appendWideRow(PERSISTENT_EDITOR(OBJECT_SENSOR_MATRIX_EDITOR));

    // Associated positions
    addSeparator(tr("Associated positions"));

    // number of positions
    appendRow(
            ITEM(tr("Count")),
            ITEM(QString::number(
                    _obj->getPositions() ? _obj->getPositions()->size() : 0)));

    double minIndex, maxIndex;
    _obj->getIndexBounds(minIndex, maxIndex);
    if (minIndex != maxIndex) {
        // Index span
        appendRow(ITEM(tr("Indexes")),
                  ITEM(QString("%1 - %2").arg(minIndex).arg(maxIndex)));

        // Current index
        appendRow(ITEM(tr("Active index")),
                  PERSISTENT_EDITOR(OBJECT_SENSOR_INDEX), true);
    }
}

void ccPropertiesTreeDelegate::fillWithGBLSensor(ccGBLSensor* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("TLS/GBL Sensor"));

    // Uncertainty
    appendRow(ITEM(tr("Uncertainty")),
              PERSISTENT_EDITOR(OBJECT_SENSOR_UNCERTAINTY), true);

    // angles
    addSeparator(tr("Angular viewport (degrees)"));
    {
        // Angular range (yaw)
        PointCoordinateType yawMin = _obj->getMinYaw();
        PointCoordinateType yawMax = _obj->getMaxYaw();
        appendRow(ITEM(tr("Yaw span")),
                  ITEM(QString("[%1 ; %2]")
                               .arg(cloudViewer::RadiansToDegrees(yawMin), 0,
                                    'f', 2)
                               .arg(cloudViewer::RadiansToDegrees(yawMax), 0,
                                    'f', 2)));

        // Angular steps (yaw)
        PointCoordinateType yawStep = _obj->getYawStep();
        appendRow(ITEM(tr("Yaw step")),
                  ITEM(QString("%1").arg(cloudViewer::RadiansToDegrees(yawStep),
                                         0, 'f', 4)));

        // Angular range (pitch)
        PointCoordinateType pitchMin = _obj->getMinPitch();
        PointCoordinateType pitchMax = _obj->getMaxPitch();
        appendRow(ITEM(tr("Pitch span")),
                  ITEM(QString("[%1 ; %2]")
                               .arg(cloudViewer::RadiansToDegrees(pitchMin), 0,
                                    'f', 2)
                               .arg(cloudViewer::RadiansToDegrees(pitchMax), 0,
                                    'f', 2)));

        // Angular steps (pitch)
        PointCoordinateType pitchStep = _obj->getPitchStep();
        appendRow(
                ITEM(tr("Pitch step")),
                ITEM(QString("%1").arg(cloudViewer::RadiansToDegrees(pitchStep),
                                       0, 'f', 4)));
    }

    // Positions
    fillWithSensor(_obj);
}

void ccPropertiesTreeDelegate::fillWithCameraSensor(ccCameraSensor* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Camera Sensor"));

    const ccCameraSensor::IntrinsicParameters& params =
            _obj->getIntrinsicParameters();

    // Focal
    appendRow(ITEM(tr("Vert. focal")),
              ITEM(QString::number(params.vertFocal_pix) + tr(" pix.")));

    // Array size
    appendRow(ITEM(tr("Array size")), ITEM(QString("%1 x %2")
                                                   .arg(params.arrayWidth)
                                                   .arg(params.arrayHeight)));

    // Principal point
    appendRow(ITEM(tr("Principal point")),
              ITEM(QString("(%1 ; %2)")
                           .arg(params.principal_point[0])
                           .arg(params.principal_point[1])));

    // Pixel size
    if (params.pixelSize_mm[0] != 0 || params.pixelSize_mm[1] != 0) {
        appendRow(ITEM(tr("Pixel size")),
                  ITEM(QString("%1 x %2")
                               .arg(params.pixelSize_mm[0])
                               .arg(params.pixelSize_mm[1])));
    }

    // Field of view
    appendRow(ITEM(tr("Field of view")),
              ITEM(QString::number(
                           cloudViewer::RadiansToDegrees(params.vFOV_rad)) +
                   tr(" deg.")));

    // Skewness
    appendRow(ITEM(tr("Skew")), ITEM(QString::number(params.skew)));

    addSeparator(tr("Frustum display"));

    // Draw frustum
    appendRow(
            ITEM(tr("Show lines")),
            CHECKABLE_ITEM(_obj->frustumIsDrawn(), OBJECT_SENSOR_DRAW_FRUSTUM));
    appendRow(ITEM(tr("Show side planes")),
              CHECKABLE_ITEM(_obj->frustumPlanesAreDrawn(),
                             OBJECT_SENSOR_DRAW_FRUSTUM_PLANES));

    // Positions
    fillWithSensor(_obj);
}

void ccPropertiesTreeDelegate::fillWithMaterialSet(ccMaterialSet* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Material set"));

    // Count
    appendRow(ITEM(tr("Count")), ITEM(QString::number(_obj->size())));

    // ccMaterialSet objects are 'shareable'
    fillWithShareable(_obj);
}

void ccPropertiesTreeDelegate::fillWithShareable(CCShareable* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Array"));

    // Link count
    unsigned linkCount =
            _obj->getLinkCount();  // if we display it, it means it is a member
                                   // of the DB --> i.e. link is already >1
    appendRow(
            ITEM(tr("Shared")),
            ITEM(linkCount < 3 ? tr("No") : tr("Yes (%1)").arg(linkCount - 1)));
}

template <class Type, int N, class ComponentType>
void ccPropertiesTreeDelegate::fillWithCCArray(
        ccArray<Type, N, ComponentType>* _obj) {
    assert(_obj && m_model);

    addSeparator(tr("Array"));

    // Name
    appendRow(ITEM(tr("Name")),
              ITEM(_obj->getName().isEmpty() ? tr("undefined")
                                             : _obj->getName()));

    // Count
    appendRow(ITEM(tr("Elements")),
              ITEM(QLocale(QLocale::English)
                           .toString(static_cast<qulonglong>(_obj->size()))));

    // Capacity
    appendRow(
            ITEM(tr("Capacity")),
            ITEM(QLocale(QLocale::English)
                         .toString(static_cast<qulonglong>(_obj->capacity()))));

    // Memory
    appendRow(
            ITEM(tr("Memory")),
            ITEM(QString("%1 Mb").arg(
                    (_obj->capacity() * sizeof(Type)) / 1048576.0, 0, 'f', 2)));

    // ccArray objects are 'Shareable'
    fillWithShareable(_obj);
}

bool ccPropertiesTreeDelegate::isWideEditor(int itemData) const {
    switch (itemData) {
        case OBJECT_CLOUD_SF_EDITOR:
        case OBJECT_SENSOR_MATRIX_EDITOR:
        case OBJECT_HISTORY_MATRIX_EDITOR:
        case OBJECT_GLTRANS_MATRIX_EDITOR:
        // Note: OBJECT_SELECTION_PROPERTIES removed - selection properties
        // are now in standalone cvFindDataDockWidget
        case TREE_VIEW_HEADER:
            return true;
        default:
            break;
    }

    return false;
}

QWidget* ccPropertiesTreeDelegate::createEditor(
        QWidget* parent,
        const QStyleOptionViewItem& option,
        const QModelIndex& index) const {
    if (!m_model) return nullptr;

    QStandardItem* item = m_model->itemFromIndex(index);

    if (!item || !item->data().isValid()) return nullptr;

    int itemData = item->data().toInt();

    // All editors require a current object
    if (!m_currentObject) {
        return nullptr;
    }
    if (item->column() == 0 && !isWideEditor(itemData)) {
        // on the first column, only editors spanning on 2 columns are allowed
        return nullptr;
    }

    QWidget* outputWidget = nullptr;

    switch (itemData) {
        case OBJECT_CURRENT_DISPLAY: {
            QComboBox* comboBox = new QComboBox(parent);

            comboBox->addItem(s_noneString);

            connect(comboBox,
                    static_cast<void (QComboBox::*)(const QString&)>(
                            &QComboBox::currentTextChanged),
                    this, &ccPropertiesTreeDelegate::objectDisplayChanged);

            outputWidget = comboBox;
        } break;
        case OBJECT_CURRENT_SCALAR_FIELD: {
            ccPointCloud* cloud =
                    ccHObjectCaster::ToPointCloud(m_currentObject);
            assert(cloud);

            QComboBox* comboBox = new QComboBox(parent);

            comboBox->addItem(tr("None"));
            int nsf = static_cast<int>(cloud->getNumberOfScalarFields());
            for (int i = 0; i < nsf; ++i)
                comboBox->addItem(QString(cloud->getScalarFieldName(i)));

            connect(comboBox,
                    static_cast<void (QComboBox::*)(int)>(
                            &QComboBox::activated),
                    this, &ccPropertiesTreeDelegate::scalarFieldChanged);

            outputWidget = comboBox;
        } break;
        case OBJECT_MESH_TEXTUREFILE: {
            ecvTextureFileSelector* selector = new ecvTextureFileSelector(
                    parent,
                    QString::fromUtf8(":/Resources/images/ecvGear.png"));
            // Initialize with empty map - will be populated in setEditorData
            // based on current mesh object
            QMap<QString, QString> emptyMap;
            selector->init(emptyMap);

            connect(selector, &ecvTextureFileSelector::textureFileSelected,
                    this, &ccPropertiesTreeDelegate::textureFileChanged);
            connect(selector,
                    &ecvTextureFileSelector::textureFileEditorSummoned, this,
                    &ccPropertiesTreeDelegate::spawnTextureFileEditor);

            outputWidget = selector;
        } break;
        case OBJECT_CURRENT_COLOR_RAMP: {
            ccColorScaleSelector* selector = new ccColorScaleSelector(
                    ccColorScalesManager::GetUniqueInstance(), parent,
                    QString::fromUtf8(":/Resources/images/ecvGear.png"));
            // fill combobox box with Color Scales Manager
            selector->init();
            connect(selector, &ccColorScaleSelector::colorScaleSelected, this,
                    &ccPropertiesTreeDelegate::colorScaleChanged);
            connect(selector, &ccColorScaleSelector::colorScaleEditorSummoned,
                    this, &ccPropertiesTreeDelegate::spawnColorRampEditor);

            outputWidget = selector;
        } break;
        case OBJECT_COLOR_RAMP_STEPS: {
            QSpinBox* spinBox = new QSpinBox(parent);
            spinBox->setRange(ccColorScale::MIN_STEPS, ccColorScale::MAX_STEPS);
            spinBox->setSingleStep(4);

            connect(spinBox,
                    static_cast<void (QSpinBox::*)(int)>(
                            &QSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::colorRampStepsChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_CLOUD_SF_EDITOR: {
            sfEditDlg* sfd = new sfEditDlg(parent);

            // DGM: why does this widget can't follow its 'policy' ?!
            // QSizePolicy pol = sfd->sizePolicy();
            // QSizePolicy::Policy hpol = pol.horizontalPolicy();
            // sfd->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Maximum);
            // parent->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Maximum);

            connect(sfd, &sfEditDlg::entitySFHasChanged, this,
                    &ccPropertiesTreeDelegate::updateDisplay);

            outputWidget = sfd;
        } break;
        case OBJECT_HISTORY_MATRIX_EDITOR:
        case OBJECT_GLTRANS_MATRIX_EDITOR:
        case OBJECT_SENSOR_MATRIX_EDITOR: {
            MatrixDisplayDlg* mdd = new MatrixDisplayDlg(parent);

            // no signal connection, it's a display-only widget

            outputWidget = mdd;
        } break;
        case TREE_VIEW_HEADER: {
            QLabel* headerLabel = new QLabel(parent);
            headerLabel->setStyleSheet(SEPARATOR_STYLESHEET);

            // no signal connection, it's a display-only widget

            outputWidget = headerLabel;
        } break;
        case OBJECT_OCTREE_TYPE: {
            QComboBox* comboBox = new QComboBox(parent);

            comboBox->addItem(tr("Wire"), QVariant(ccOctree::WIRE));
            comboBox->addItem(tr("Points"), QVariant(ccOctree::MEAN_POINTS));
            comboBox->addItem(tr("Plain cubes"),
                              QVariant(ccOctree::MEAN_CUBES));

            connect(comboBox,
                    static_cast<void (QComboBox::*)(int)>(
                            &QComboBox::activated),
                    this, &ccPropertiesTreeDelegate::octreeDisplayModeChanged);

            outputWidget = comboBox;
        } break;
        case OBJECT_OCTREE_LEVEL: {
            QSpinBox* spinBox = new QSpinBox(parent);
            spinBox->setRange(1, cloudViewer::DgmOctree::MAX_OCTREE_LEVEL);

            connect(spinBox,
                    static_cast<void (QSpinBox::*)(int)>(
                            &QSpinBox::valueChanged),
                    this,
                    &ccPropertiesTreeDelegate::octreeDisplayedLevelChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_PRIMITIVE_PRECISION: {
            QSpinBox* spinBox = new QSpinBox(parent);
            spinBox->setRange(4, 360);
            spinBox->setSingleStep(4);

            connect(spinBox,
                    static_cast<void (QSpinBox::*)(int)>(
                            &QSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::primitivePrecisionChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_CIRCLE_RESOLUTION: {
            QSpinBox* spinBox = new QSpinBox(parent);
            spinBox->setRange(4, 1024);
            spinBox->setSingleStep(4);

            connect(spinBox,
                    static_cast<void (QSpinBox::*)(int)>(
                            &QSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::circleResolutionChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_SPHERE_RADIUS: {
            QDoubleSpinBox* spinBox = new QDoubleSpinBox(parent);
            spinBox->setDecimals(6);
            spinBox->setRange(0, 1.0e6);
            spinBox->setSingleStep(1.0);

            connect(spinBox,
                    static_cast<void (QDoubleSpinBox::*)(double)>(
                            &QDoubleSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::sphereRadiusChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_CIRCLE_RADIUS: {
            QDoubleSpinBox* spinBox = new QDoubleSpinBox(parent);
            spinBox->setDecimals(7);
            spinBox->setRange(1.0e-6, 1.0e6);
            spinBox->setSingleStep(1.0);

            connect(spinBox,
                    static_cast<void (QDoubleSpinBox::*)(double)>(
                            &QDoubleSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::circleRadiusChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_DISC_RADIUS: {
            QDoubleSpinBox* spinBox = new QDoubleSpinBox(parent);
            spinBox->setDecimals(7);
            spinBox->setRange(1.0e-6, 1.0e6);
            spinBox->setSingleStep(1.0);

            connect(spinBox,
                    static_cast<void (QDoubleSpinBox::*)(double)>(
                            &QDoubleSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::discRadiusChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_CONE_HEIGHT: {
            QDoubleSpinBox* spinBox = new QDoubleSpinBox(parent);
            spinBox->setDecimals(6);
            spinBox->setRange(0, 1.0e6);
            spinBox->setSingleStep(1.0);

            connect(spinBox,
                    static_cast<void (QDoubleSpinBox::*)(double)>(
                            &QDoubleSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::coneHeightChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_CONE_BOTTOM_RADIUS: {
            QDoubleSpinBox* spinBox = new QDoubleSpinBox(parent);
            spinBox->setDecimals(6);
            spinBox->setRange(0, 1.0e6);
            spinBox->setSingleStep(1.0);

            connect(spinBox,
                    static_cast<void (QDoubleSpinBox::*)(double)>(
                            &QDoubleSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::coneBottomRadiusChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_CONE_TOP_RADIUS: {
            QDoubleSpinBox* spinBox = new QDoubleSpinBox(parent);
            spinBox->setDecimals(6);
            spinBox->setRange(0, 1.0e6);
            spinBox->setSingleStep(1.0);

            connect(spinBox,
                    static_cast<void (QDoubleSpinBox::*)(double)>(
                            &QDoubleSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::coneTopRadiusChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_IMAGE_ALPHA: {
            QSlider* slider = new QSlider(Qt::Horizontal, parent);
            slider->setRange(0, 255);
            slider->setSingleStep(1);
            slider->setPageStep(16);
            slider->setTickPosition(QSlider::NoTicks);
            connect(slider, &QAbstractSlider::valueChanged, this,
                    &ccPropertiesTreeDelegate::imageAlphaChanged);

            outputWidget = slider;
        } break;
        case OBJECT_VIEW_LIGHT_KIT_INTENSITY: {
            // ParaView-style light intensity control: Slider + SpinBox
            // (0.0-1.0)
            QWidget* container = new QWidget(parent);
            QHBoxLayout* layout = new QHBoxLayout(container);
            layout->setContentsMargins(0, 0, 0, 0);
            layout->setSpacing(4);

            // Slider for quick adjustment (0-100 representing 0.0-1.0)
            QSlider* slider = new QSlider(Qt::Horizontal, container);
            slider->setRange(0, 100);  // 0% to 100% intensity
            slider->setSingleStep(1);
            slider->setPageStep(10);
            slider->setTickPosition(QSlider::NoTicks);

            // SpinBox for precise numeric input
            QDoubleSpinBox* spinBox = new QDoubleSpinBox(container);
            spinBox->setRange(0.0, 1.0);
            spinBox->setSingleStep(0.01);
            spinBox->setDecimals(2);
            spinBox->setMinimumWidth(60);

            // Synchronize slider and spinbox (visual sync only)
            connect(slider, &QSlider::valueChanged, this, [spinBox](int value) {
                spinBox->blockSignals(true);
                spinBox->setValue(value / 100.0);
                spinBox->blockSignals(false);
            });
            connect(spinBox,
                    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                    [slider](double value) {
                        slider->blockSignals(true);
                        slider->setValue(static_cast<int>(value * 100.0));
                        slider->blockSignals(false);
                    });

            // Connect BOTH slider and spinbox to light intensity handler (like
            // Opacity does) Slider: convert int [0, 100] to double [0.0, 1.0]
            // for handler
            ccPropertiesTreeDelegate* self =
                    const_cast<ccPropertiesTreeDelegate*>(this);
            connect(slider, &QAbstractSlider::valueChanged, self,
                    [self](int value) {
                        self->lightIntensityChanged(value / 100.0);
                    });
            // SpinBox: direct connection (already in [0.0, 1.0] range)
            connect(spinBox,
                    QOverload<double>::of(&QDoubleSpinBox::valueChanged), self,
                    [self](double value) {
                        self->lightIntensityChanged(value);
                    });

            layout->addWidget(slider, 1);
            layout->addWidget(spinBox, 0);

            outputWidget = container;
        } break;
        case OBJECT_OPACITY: {
            // ParaView-style opacity control: Slider + SpinBox combination
            // Creates a horizontal layout with slider and numeric input
            QWidget* container = new QWidget(parent);
            QHBoxLayout* layout = new QHBoxLayout(container);
            layout->setContentsMargins(0, 0, 0, 0);
            layout->setSpacing(4);

            // Slider for quick adjustment
            QSlider* slider = new QSlider(Qt::Horizontal, container);
            slider->setRange(0, 100);  // 0% to 100% opacity
            slider->setSingleStep(1);
            slider->setPageStep(10);
            slider->setTickPosition(QSlider::NoTicks);

            // SpinBox for precise numeric input (ParaView style: shows
            // 0.00-1.00)
            QDoubleSpinBox* spinBox = new QDoubleSpinBox(container);
            spinBox->setRange(0.0, 1.0);
            spinBox->setDecimals(2);
            spinBox->setSingleStep(0.01);
            spinBox->setFixedWidth(60);

            // Synchronize slider and spinbox (visual sync only, no opacity
            // update)
            connect(slider, &QSlider::valueChanged, this, [spinBox](int value) {
                spinBox->blockSignals(true);
                spinBox->setValue(value / 100.0);
                spinBox->blockSignals(false);
            });
            connect(spinBox,
                    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                    [slider](double value) {
                        slider->blockSignals(true);
                        slider->setValue(static_cast<int>(value * 100));
                        slider->blockSignals(false);
                    });

            // Connect BOTH slider and spinbox to opacity change handler
            // Slider: direct connection
            connect(slider, &QAbstractSlider::valueChanged, this,
                    &ccPropertiesTreeDelegate::opacityChanged);
            // SpinBox: convert double [0.0, 1.0] to int [0, 100] for handler
            // Use const_cast because createEditor is const but opacityChanged
            // is not
            ccPropertiesTreeDelegate* self =
                    const_cast<ccPropertiesTreeDelegate*>(this);
            connect(spinBox,
                    QOverload<double>::of(&QDoubleSpinBox::valueChanged), self,
                    [self](double value) {
                        self->opacityChanged(static_cast<int>(value * 100));
                    });

            layout->addWidget(slider, 1);   // Stretch factor 1
            layout->addWidget(spinBox, 0);  // Fixed size

            outputWidget = container;
        } break;
        case OBJECT_SENSOR_INDEX: {
            ccSensor* sensor = ccHObjectCaster::ToSensor(m_currentObject);
            assert(sensor);

            double minIndex, maxIndex;
            sensor->getIndexBounds(minIndex, maxIndex);

            QDoubleSpinBox* spinBox = new QDoubleSpinBox(parent);
            spinBox->setRange(minIndex, maxIndex);
            spinBox->setSingleStep((maxIndex - minIndex) / 1000.0);

            connect(spinBox,
                    static_cast<void (QDoubleSpinBox::*)(double)>(
                            &QDoubleSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::sensorIndexChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_TRANS_BUFFER_TRIHDERONS_SCALE: {
            QDoubleSpinBox* spinBox = new QDoubleSpinBox(parent);
            spinBox->setRange(1.0e-3, 1.0e6);
            spinBox->setDecimals(3);
            spinBox->setSingleStep(1.0);

            connect(spinBox,
                    static_cast<void (QDoubleSpinBox::*)(double)>(
                            &QDoubleSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::trihedronsScaleChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_APPLY_SENSOR_VIEWPORT: {
            QPushButton* button = new QPushButton(tr("Apply"), parent);
            connect(button, &QAbstractButton::clicked, this,
                    &ccPropertiesTreeDelegate::applySensorViewport);

            button->setMinimumHeight(30);

            outputWidget = button;
        } break;
        case OBJECT_APPLY_LABEL_VIEWPORT: {
            QPushButton* button = new QPushButton(tr("Apply"), parent);
            connect(button, &QAbstractButton::clicked, this,
                    &ccPropertiesTreeDelegate::applyLabelViewport);

            button->setMinimumHeight(30);
            outputWidget = button;
        } break;
        case OBJECT_UPDATE_LABEL_VIEWPORT: {
            QPushButton* button = new QPushButton(tr("Update"), parent);
            connect(button, &QAbstractButton::clicked, this,
                    &ccPropertiesTreeDelegate::updateLabelViewport);

            button->setMinimumHeight(30);
            outputWidget = button;
        } break;
        case OBJECT_SENSOR_UNCERTAINTY: {
            QLineEdit* lineEdit = new QLineEdit(parent);
            lineEdit->setValidator(
                    new QDoubleValidator(1.0e-8, 1.0, 8, lineEdit));
            connect(lineEdit, &QLineEdit::editingFinished, this,
                    &ccPropertiesTreeDelegate::sensorUncertaintyChanged);

            outputWidget = lineEdit;
        } break;
        case OBJECT_SENSOR_DISPLAY_SCALE: {
            QDoubleSpinBox* spinBox = new QDoubleSpinBox(parent);
            spinBox->setRange(1.0e-3, 1.0e6);
            spinBox->setDecimals(3);
            spinBox->setSingleStep(1.0e-1);

            connect(spinBox,
                    static_cast<void (QDoubleSpinBox::*)(double)>(
                            &QDoubleSpinBox::valueChanged),
                    this, &ccPropertiesTreeDelegate::sensorScaleChanged);

            outputWidget = spinBox;
        } break;
        case OBJECT_CLOUD_POINT_SIZE: {
            QComboBox* comboBox = new QComboBox(parent);

            comboBox->addItem(s_defaultPointSizeString);  // size = 0
            for (int i = static_cast<int>(MIN_POINT_SIZE_F);
                 i <= static_cast<int>(MAX_POINT_SIZE_F); ++i)
                comboBox->addItem(QString::number(i));

            connect(comboBox,
                    static_cast<void (QComboBox::*)(int)>(
                            &QComboBox::currentIndexChanged),
                    this, &ccPropertiesTreeDelegate::cloudPointSizeChanged);

            outputWidget = comboBox;
        } break;
        case OBJECT_POLYLINE_WIDTH: {
            QComboBox* comboBox = new QComboBox(parent);

            comboBox->addItem(s_defaultPolyWidthSizeString);  // size = 0
            for (int i = static_cast<int>(MIN_LINE_WIDTH_F);
                 i <= static_cast<int>(MAX_LINE_WIDTH_F); ++i)
                comboBox->addItem(QString::number(i));

            connect(comboBox,
                    static_cast<void (QComboBox::*)(int)>(
                            &QComboBox::currentIndexChanged),
                    this, &ccPropertiesTreeDelegate::polyineWidthChanged);

            outputWidget = comboBox;
        } break;
        case OBJECT_COLOR_SOURCE: {
            QComboBox* comboBox = new QComboBox(parent);

            comboBox->addItem(s_noneString);
            if (m_currentObject) {
                if (m_currentObject->hasColors()) {
                    comboBox->addItem(s_rgbColor);
                    comboBox->setItemIcon(
                            comboBox->count() - 1,
                            QIcon(QString::fromUtf8(
                                    ":/Resources/images/typeRgbCcolor.png")));
                }
                if (m_currentObject->hasScalarFields()) {
                    comboBox->addItem(s_sfColor);
                    comboBox->setItemIcon(
                            comboBox->count() - 1,
                            QIcon(QString::fromUtf8(
                                    ":/Resources/images/typeSF.png")));
                }
                connect(comboBox,
                        static_cast<void (QComboBox::*)(const QString&)>(
                                &QComboBox::currentTextChanged),
                        this, &ccPropertiesTreeDelegate::colorSourceChanged);
            }

            outputWidget = comboBox;
        } break;
        case OBJECT_COORDINATE_SYSTEM_AXES_WIDTH: {
            QComboBox* comboBox = new QComboBox(parent);

            comboBox->addItem(tr(s_defaultPolyWidthSizeString));  // size = 0

            for (int i = static_cast<int>(ccCoordinateSystem::MIN_AXIS_WIDTH_F);
                 i <= static_cast<int>(ccCoordinateSystem::MAX_AXIS_WIDTH_F);
                 ++i) {
                comboBox->addItem(QString::number(i));
            }
            ccCoordinateSystem* cs =
                    ccHObjectCaster::ToCoordinateSystem(m_currentObject);
            if (cs) {
                comboBox->setCurrentIndex(static_cast<int>(cs->getAxisWidth()));
            }
            connect(comboBox,
                    static_cast<void (QComboBox::*)(int)>(
                            &QComboBox::currentIndexChanged),
                    this,
                    &ccPropertiesTreeDelegate::
                            coordinateSystemAxisWidthChanged);

            outputWidget = comboBox;
        } break;
        case OBJECT_COORDINATE_SYSTEM_DISP_SCALE: {
            QDoubleSpinBox* spinBox = new QDoubleSpinBox(parent);
            spinBox->setRange(1.0e-3, 1.0e6);
            spinBox->setDecimals(3);
            spinBox->setSingleStep(1.0e-1);
            ccCoordinateSystem* cs =
                    ccHObjectCaster::ToCoordinateSystem(m_currentObject);
            if (cs) {
                spinBox->setValue(cs->getDisplayScale());
            }

            connect(spinBox,
                    static_cast<void (QDoubleSpinBox::*)(double)>(
                            &QDoubleSpinBox::valueChanged),
                    this,
                    &ccPropertiesTreeDelegate::
                            coordinateSystemDisplayScaleChanged);

            outputWidget = spinBox;
        } break;
        // ParaView-style Data Axes Grid: Checkbox + Edit button in same row
        case OBJECT_VIEW_DATA_AXES_GRID_VISIBLE: {
            // Create container widget with horizontal layout (like Opacity)
            QWidget* container = new QWidget(parent);
            QHBoxLayout* layout = new QHBoxLayout(container);
            layout->setContentsMargins(0, 0, 0, 0);
            layout->setSpacing(4);

            // Checkbox for visibility
            QCheckBox* checkbox = new QCheckBox(container);

            // Edit button (compact style)
            QPushButton* editButton = new QPushButton(tr("Edit..."), container);
            editButton->setMinimumHeight(22);
            editButton->setMaximumWidth(80);
            connect(editButton, &QPushButton::clicked, this,
                    &ccPropertiesTreeDelegate::dataAxesGridEditRequested);

            // Add to layout: checkbox (stretch) + button (fixed)
            layout->addWidget(checkbox, 1);
            layout->addWidget(editButton, 0);

            outputWidget = container;
        } break;
        // Note: OBJECT_VIEW_DATA_AXES_GRID_EDIT is now integrated into VISIBLE
        // case above
        case OBJECT_VIEW_DATA_AXES_GRID_EDIT: {
            // This case is no longer used - keeping for compatibility
            QPushButton* button = new QPushButton(tr("Edit..."), parent);
            connect(button, &QAbstractButton::clicked, this,
                    &ccPropertiesTreeDelegate::dataAxesGridEditRequested);
            button->setMinimumHeight(22);
            button->setMaximumWidth(80);
            outputWidget = button;
        } break;
        // Note: OBJECT_SELECTION_PROPERTIES case removed - selection
        // properties are now in standalone cvFindDataDockWidget
        default:
            return QStyledItemDelegate::createEditor(parent, option, index);
    }

    if (outputWidget) {
        // Qt doc: << The returned editor widget should have Qt::StrongFocus >>
        outputWidget->setFocusPolicy(Qt::StrongFocus);
    } else {
        // shouldn't happen
        assert(false);
    }

    return outputWidget;
}

void ccPropertiesTreeDelegate::updateEditorGeometry(
        QWidget* editor,
        const QStyleOptionViewItem& option,
        const QModelIndex& index) const {
    QStyledItemDelegate::updateEditorGeometry(editor, option, index);

    if (!m_model || !editor) return;

    QStandardItem* item = m_model->itemFromIndex(index);

    if (item && item->data().isValid() && item->column() == 0) {
        if (isWideEditor(item->data().toInt())) {
            QWidget* widget = qobject_cast<QWidget*>(editor);
            if (!widget) return;
            // we must resize the SF edit widget so that it spans on both
            // columns!
            QRect rect = m_view->visualRect(
                    m_model->index(item->row(), 1));  // second column width
            widget->resize(option.rect.width() + rect.width(),
                           widget->height());
        }
    }
}

void SetDoubleSpinBoxValue(QWidget* editor,
                           double value,
                           bool keyboardTracking = false) {
    QDoubleSpinBox* spinBox = qobject_cast<QDoubleSpinBox*>(editor);
    if (!spinBox) {
        assert(false);
        return;
    }
    spinBox->setKeyboardTracking(keyboardTracking);
    spinBox->setValue(value);
}

void SetSpinBoxValue(QWidget* editor,
                     int value,
                     bool keyboardTracking = false) {
    QSpinBox* spinBox = qobject_cast<QSpinBox*>(editor);
    if (!spinBox) {
        assert(false);
        return;
    }
    spinBox->setKeyboardTracking(keyboardTracking);
    spinBox->setValue(value);
}

void SetComboBoxIndex(QWidget* editor, int index) {
    QComboBox* comboBox = qobject_cast<QComboBox*>(editor);
    if (!comboBox) {
        assert(false);
        return;
    }
    assert(index < 0 || index < comboBox->maxCount());
    comboBox->setCurrentIndex(index);
}

void ccPropertiesTreeDelegate::setEditorData(QWidget* editor,
                                             const QModelIndex& index) const {
    if (!m_model) return;

    QStandardItem* item = m_model->itemFromIndex(index);
    if (!item || !item->data().isValid()) return;

    int itemData = item->data().toInt();

    // All properties require a current object
    if (!m_currentObject) return;

    if (item->column() == 0 && !isWideEditor(itemData)) return;

    switch (itemData) {
        case OBJECT_CURRENT_DISPLAY: {
            QComboBox* comboBox = qobject_cast<QComboBox*>(editor);
            if (!comboBox) {
                assert(false);
                return;
            }

            int pos = comboBox->findText(Settings::APP_TITLE);

            comboBox->setCurrentIndex(std::max(pos, 0));  // 0 = "NONE"
            break;
        }
        case OBJECT_CURRENT_SCALAR_FIELD: {
            ccPointCloud* cloud =
                    ccHObjectCaster::ToPointCloud(m_currentObject);
            assert(cloud);

            int pos = cloud->getCurrentDisplayedScalarFieldIndex();
            SetComboBoxIndex(editor, pos + 1);
            break;
        }
        case OBJECT_MESH_TEXTUREFILE: {
            QFrame* selectorFrame = qobject_cast<QFrame*>(editor);
            if (!selectorFrame) return;
            ecvTextureFileSelector* selector =
                    static_cast<ecvTextureFileSelector*>(selectorFrame);

            // get current material
            ccGenericMesh* mesh =
                    ccHObjectCaster::ToGenericMesh(m_currentObject);
            assert(mesh);

            // Get or create texture path map for current mesh
            QMap<QString, QString>& texturePathMap =
                    m_meshTexturePathMaps[m_currentObject];

            const ccMaterialSet* materialSet =
                    mesh ? mesh->getMaterialSet() : nullptr;
            if (materialSet) {
                if (!materialSet->empty()) {
                    // Always clear and repopulate selector when switching mesh
                    // objects This ensures we only show textures for the
                    // current mesh Clear the combo box by reinitializing with
                    // empty map
                    QMap<QString, QString> emptyMap;
                    selector->init(emptyMap);
                    texturePathMap.clear();

                    // Collect ALL DIFFUSE (map_Kd) textures from all materials
                    // Include all occurrences from the file, even if the same
                    // path appears multiple times
                    for (std::size_t i = 0; i < materialSet->size(); ++i) {
                        const ccMaterial::CShared& material =
                                materialSet->at(i);
                        if (!material) continue;

                        // Get ALL DIFFUSE textures for this material using the
                        // new method This returns all map_Kd textures,
                        // including duplicates
                        std::vector<QString> diffuseTextures =
                                material->getTextureFilenames(
                                        ccMaterial::TextureMapType::DIFFUSE);

                        // Add ALL DIFFUSE textures (map_Kd) - include all
                        // occurrences
                        for (const QString& texPath : diffuseTextures) {
                            if (!texPath.isEmpty()) {
                                QString texName = QFileInfo(texPath).fileName();
                                if (texName.isEmpty()) {
                                    texName = material->getName();
                                }
                                // Add all textures to show all map_Kd from the
                                // file
                                selector->addItem(texName, texPath);
                                texturePathMap[texName] = texPath;
                            }
                        }
                    }

                    // Don't auto-select - let user choose from all available
                    // textures The selector will show the first item but user
                    // can change it
                } else {
                    selector->setSelectedTexturefile(QString());
                }
            }
            break;
        }
        case OBJECT_CURRENT_COLOR_RAMP: {
            QFrame* selectorFrame = qobject_cast<QFrame*>(editor);
            if (!selectorFrame) return;
            ccColorScaleSelector* selector =
                    static_cast<ccColorScaleSelector*>(selectorFrame);

            ccPointCloud* cloud =
                    ccHObjectCaster::ToPointCloud(m_currentObject);
            assert(cloud);

            ccScalarField* sf = cloud->getCurrentDisplayedScalarField();
            if (sf) {
                if (sf->getColorScale())
                    selector->setSelectedScale(sf->getColorScale()->getUuid());
                else
                    selector->setSelectedScale(QString());
            }
            break;
        }
        case OBJECT_COLOR_RAMP_STEPS: {
            ccPointCloud* cloud =
                    ccHObjectCaster::ToPointCloud(m_currentObject);
            assert(cloud);
            ccScalarField* sf =
                    cloud ? cloud->getCurrentDisplayedScalarField() : nullptr;
            if (sf)
                SetSpinBoxValue(editor,
                                static_cast<int>(sf->getColorRampSteps()),
                                true);
            break;
        }
        case OBJECT_CLOUD_SF_EDITOR: {
            sfEditDlg* sfd = qobject_cast<sfEditDlg*>(editor);
            if (!sfd) return;

            ccPointCloud* cloud =
                    ccHObjectCaster::ToPointCloud(m_currentObject);
            assert(cloud);

            ccScalarField* sf = cloud->getCurrentDisplayedScalarField();
            if (sf) sfd->fillDialogWith(sf);
            break;
        }
        case OBJECT_HISTORY_MATRIX_EDITOR: {
            MatrixDisplayDlg* mdd = qobject_cast<MatrixDisplayDlg*>(editor);
            if (!mdd) return;

            mdd->fillDialogWith(m_currentObject->getGLTransformationHistory());
            break;
        }
        case OBJECT_GLTRANS_MATRIX_EDITOR: {
            MatrixDisplayDlg* mdd = qobject_cast<MatrixDisplayDlg*>(editor);
            if (!mdd) return;

            mdd->fillDialogWith(m_currentObject->getGLTransformation());
            break;
        }
        case OBJECT_SENSOR_MATRIX_EDITOR: {
            MatrixDisplayDlg* mdd = qobject_cast<MatrixDisplayDlg*>(editor);
            if (!mdd) return;

            ccSensor* sensor = ccHObjectCaster::ToSensor(m_currentObject);
            assert(sensor);

            ccIndexedTransformation trans;
            if (sensor->getActiveAbsoluteTransformation(trans)) {
                mdd->fillDialogWith(trans);
            } else {
                mdd->clear();
                mdd->setEnabled(false);
            }
            break;
        }
        case TREE_VIEW_HEADER: {
            QLabel* label = qobject_cast<QLabel*>(editor);
            if (label) label->setText(item->accessibleDescription());
            break;
        }
        case OBJECT_OCTREE_TYPE: {
            ccOctree* octree = ccHObjectCaster::ToOctree(m_currentObject);
            assert(octree);
            SetComboBoxIndex(editor,
                             static_cast<int>(octree->getDisplayMode()));
            break;
        }
        case OBJECT_OCTREE_LEVEL: {
            ccOctree* octree = ccHObjectCaster::ToOctree(m_currentObject);
            assert(octree);
            SetSpinBoxValue(editor, octree ? octree->getDisplayedLevel() : 0);
            break;
        }
        case OBJECT_PRIMITIVE_PRECISION: {
            ccGenericPrimitive* primitive =
                    ccHObjectCaster::ToPrimitive(m_currentObject);
            assert(primitive);
            SetSpinBoxValue(
                    editor,
                    primitive
                            ? static_cast<int>(primitive->getDrawingPrecision())
                            : 0);
            break;
        }
        case OBJECT_CIRCLE_RESOLUTION: {
            ccCircle* circle = ccHObjectCaster::ToCircle(m_currentObject);
            assert(circle);
            SetSpinBoxValue(editor, circle ? circle->getResolution() : 0);
            break;
        }
        case OBJECT_SPHERE_RADIUS: {
            ccSphere* sphere = ccHObjectCaster::ToSphere(m_currentObject);
            assert(sphere);
            SetDoubleSpinBoxValue(
                    editor,
                    sphere ? static_cast<double>(sphere->getRadius()) : 0.0);
            break;
        }
        case OBJECT_CIRCLE_RADIUS: {
            ccCircle* circle = ccHObjectCaster::ToCircle(m_currentObject);
            assert(circle);
            SetDoubleSpinBoxValue(editor, circle ? circle->getRadius() : 0.0);
            break;
        }
        case OBJECT_DISC_RADIUS: {
            ccDisc* disc = ccHObjectCaster::ToDisc(m_currentObject);
            assert(disc);
            SetDoubleSpinBoxValue(
                    editor,
                    disc ? static_cast<double>(disc->getRadius()) : 0.0);
            break;
        }
        case OBJECT_CONE_HEIGHT: {
            ccCone* cone = ccHObjectCaster::ToCone(m_currentObject);
            assert(cone);
            SetDoubleSpinBoxValue(
                    editor,
                    cone ? static_cast<double>(cone->getHeight()) : 0.0);
            break;
        }
        case OBJECT_CONE_BOTTOM_RADIUS: {
            ccCone* cone = ccHObjectCaster::ToCone(m_currentObject);
            assert(cone);
            SetDoubleSpinBoxValue(
                    editor,
                    cone ? static_cast<double>(cone->getBottomRadius()) : 0.0);
            break;
        }
        case OBJECT_CONE_TOP_RADIUS: {
            ccCone* cone = ccHObjectCaster::ToCone(m_currentObject);
            assert(cone);
            SetDoubleSpinBoxValue(
                    editor,
                    cone ? static_cast<double>(cone->getTopRadius()) : 0.0);
            break;
        }
        case OBJECT_IMAGE_ALPHA: {
            QSlider* slider = qobject_cast<QSlider*>(editor);
            if (!slider) return;

            ccImage* image = ccHObjectCaster::ToImage(m_currentObject);
            assert(image);
            slider->setValue(static_cast<int>(image->getAlpha() * 255.0f));
            // slider->setTickPosition(QSlider::NoTicks);
            break;
        }
        case OBJECT_VIEW_LIGHT_KIT_INTENSITY: {
            // ParaView-style: editor is a container with slider + spinbox
            QWidget* container = qobject_cast<QWidget*>(editor);
            if (!container) return;

            // Find the slider and spinbox in the container
            QSlider* slider = container->findChild<QSlider*>();
            QDoubleSpinBox* spinBox = container->findChild<QDoubleSpinBox*>();

            // Get current light intensity from backend (default 1.0 for normal
            // intensity)
            double intensity = 1.0;  // Default
            if (ecvDisplayTools::TheInstance()) {
                intensity = ecvDisplayTools::TheInstance()->getLightIntensity();
            }

            // Set both controls (slider triggers spinbox sync via signal)
            if (slider) {
                slider->setValue(static_cast<int>(intensity * 100.0));
            }
            if (spinBox) {
                spinBox->setValue(intensity);
            }
        } break;
        case OBJECT_OPACITY: {
            // ParaView-style: editor is a container with slider + spinbox
            QWidget* container = qobject_cast<QWidget*>(editor);
            if (!container) return;

            // Find the slider and spinbox in the container
            QSlider* slider = container->findChild<QSlider*>();
            QDoubleSpinBox* spinBox = container->findChild<QDoubleSpinBox*>();

            // Get current opacity from the object [0.0, 1.0]
            // For folders, calculate average opacity from all renderable
            // children
            float opacity = m_currentObject->getOpacity();

            if (m_currentObject->getChildrenNumber() > 0) {
                // This is a folder - calculate average opacity from renderable
                // children
                float totalOpacity = 0.0f;
                int renderableCount = 0;

                std::function<void(ccHObject*)> collectOpacity =
                        [&collectOpacity, &totalOpacity,
                         &renderableCount](ccHObject* obj) {
                            if (!obj || !obj->isEnabled()) return;

                            // Check if this is a renderable object
                            if (obj->isKindOf(CV_TYPES::POINT_CLOUD) ||
                                obj->isKindOf(CV_TYPES::MESH) ||
                                obj->isKindOf(CV_TYPES::PRIMITIVE) ||
                                obj->isKindOf(CV_TYPES::POLY_LINE) ||
                                obj->isKindOf(CV_TYPES::FACET)) {
                                totalOpacity += obj->getOpacity();
                                renderableCount++;
                            }

                            // Recursively process children
                            for (unsigned i = 0; i < obj->getChildrenNumber();
                                 ++i) {
                                collectOpacity(obj->getChild(i));
                            }
                        };

                collectOpacity(m_currentObject);

                if (renderableCount > 0) {
                    opacity = totalOpacity / renderableCount;
                }
            }

            // Set both controls (slider triggers spinbox sync via signal)
            if (slider) {
                slider->setValue(static_cast<int>(opacity * 100.0f));
            }
            if (spinBox) {
                spinBox->setValue(static_cast<double>(opacity));
            }
            break;
        }
        case OBJECT_VIEW_DATA_AXES_GRID_VISIBLE: {
            // ParaView-style: editor is a container with checkbox + edit button
            QWidget* container = qobject_cast<QWidget*>(editor);
            if (!container || !m_currentObject) return;

            // Find the checkbox in the container
            QCheckBox* checkbox = container->findChild<QCheckBox*>();
            if (!checkbox) return;

            // Get current visibility from backend
            QString viewID = m_currentObject->getViewId();
            AxesGridProperties props;
            ecvDisplayTools::TheInstance()->getDataAxesGridProperties(viewID,
                                                                      props);

            // Set checkbox state
            checkbox->setChecked(props.visible);

            // Connect checkbox signal to update handler (disconnect first to
            // avoid duplicates)
            disconnect(checkbox, nullptr, this, nullptr);
            connect(checkbox, &QCheckBox::toggled, this,
                    [this, viewID](bool checked) {
                        if (!ecvDisplayTools::TheInstance()) return;

                        // Get current properties using struct-based interface
                        AxesGridProperties props;
                        ecvDisplayTools::TheInstance()
                                ->getDataAxesGridProperties(viewID, props);

                        // Update visibility
                        props.visible = checked;

                        // Bounds will be automatically recalculated in
                        // SetDataAxesGridProperties if useCustomBounds is false
                        // For parent nodes/folders, bounds will be calculated
                        // from getDisplayBB_recursive(false) which includes all
                        // children
                        ecvDisplayTools::TheInstance()
                                ->setDataAxesGridProperties(viewID, props);

                        // Immediately update bbox visibility for all selected
                        // objects that use this viewID
                        if (MainWindow::TheInstance()) {
                            const ccHObject::Container& selectedEntities =
                                    MainWindow::TheInstance()
                                            ->getSelectedEntities();
                            const ecvGui::ParamStruct& params =
                                    ecvGui::Parameters();

                            for (ccHObject* entity : selectedEntities) {
                                if (entity && entity->getViewId() == viewID) {
                                    CC_DRAW_CONTEXT context;
                                    context.viewID = viewID;

                                    // If axes grid is now visible, immediately
                                    // hide bbox
                                    if (checked) {
                                        entity->hideBB(context);
                                    } else {
                                        // If axes grid is hidden, check if bbox
                                        // should be shown
                                        if (params.showBBOnSelected) {
                                            entity->showBB(context);
                                        } else {
                                            entity->hideBB(context);
                                        }
                                    }
                                }
                            }
                        }

                        ecvDisplayTools::UpdateScreen();
                    });
            break;
        }
        case OBJECT_SENSOR_INDEX: {
            ccSensor* sensor = ccHObjectCaster::ToSensor(m_currentObject);
            assert(sensor);
            SetDoubleSpinBoxValue(editor,
                                  sensor ? sensor->getActiveIndex() : 0.0);
            break;
        }
        case OBJECT_SENSOR_UNCERTAINTY: {
            QLineEdit* lineEdit = qobject_cast<QLineEdit*>(editor);
            if (!lineEdit) return;

            ccGBLSensor* sensor = ccHObjectCaster::ToGBLSensor(m_currentObject);
            assert(sensor);
            lineEdit->setText(QString::number(
                    sensor ? static_cast<double>(sensor->getUncertainty()) : 0,
                    'g', 8));
            break;
        }
        case OBJECT_SENSOR_DISPLAY_SCALE: {
            ccSensor* sensor = ccHObjectCaster::ToSensor(m_currentObject);
            assert(sensor);
            SetDoubleSpinBoxValue(
                    editor,
                    sensor ? static_cast<double>(sensor->getGraphicScale())
                           : 0.0);
            break;
        }
        case OBJECT_TRANS_BUFFER_TRIHDERONS_SCALE: {
            ccIndexedTransformationBuffer* buffer =
                    ccHObjectCaster::ToTransBuffer(m_currentObject);
            assert(buffer);
            SetDoubleSpinBoxValue(
                    editor, buffer ? static_cast<double>(
                                             buffer->triherdonsDisplayScale())
                                   : 0.0);
            break;
        }
        case OBJECT_CLOUD_POINT_SIZE: {
            ccGenericPointCloud* cloud =
                    ccHObjectCaster::ToGenericPointCloud(m_currentObject);
            assert(cloud);
            SetComboBoxIndex(editor, static_cast<int>(cloud->getPointSize()));
            break;
        }
        case OBJECT_POLYLINE_WIDTH: {
            ccPolyline* poly = ccHObjectCaster::ToPolyline(m_currentObject);
            assert(poly);
            SetComboBoxIndex(editor, static_cast<int>(poly->getWidth()));
            break;
        }
        case OBJECT_COLOR_SOURCE: {
            int currentIndex = 0;  // no color
            int lastIndex = currentIndex;
            if (m_currentObject->hasColors()) {
                ++lastIndex;
                if (m_currentObject->colorsShown()) currentIndex = lastIndex;
            }
            if (m_currentObject->hasScalarFields()) {
                ++lastIndex;
                if (m_currentObject->sfShown()) currentIndex = lastIndex;
            }
            SetComboBoxIndex(editor, currentIndex);
            break;
        }
        // Note: OBJECT_SELECTION_PROPERTIES case removed - selection
        // properties are now in standalone cvFindDataDockWidget
        default:
            QStyledItemDelegate::setEditorData(editor, index);
            break;
    }
}

void ccPropertiesTreeDelegate::updateItem(QStandardItem* item) {
    if (!m_currentObject || item->column() == 0 || !item->data().isValid())
        return;

    bool redraw = false;
    switch (item->data().toInt()) {
        case OBJECT_NAME: {
            m_currentObject->setName(item->text());
            emit ccObjectPropertiesChanged(m_currentObject);
        } break;
        case OBJECT_VISIBILITY: {
            bool objectWasDisplayed = m_currentObject->isDisplayed();
            if (m_currentObject->isA(CV_TYPES::LABEL_2D)) {
                cc2DLabel* label = ccHObjectCaster::To2DLabel(m_currentObject);
                if (label) {
                    label->setVisible(item->checkState() == Qt::Checked);
                    label->updateLabel();
                    break;
                }
            } else if (m_currentObject->isA(CV_TYPES::VIEWPORT_2D_LABEL)) {
                cc2DViewportLabel* label =
                        ccHObjectCaster::To2DViewportLabel(m_currentObject);
                if (label) {
                    label->setVisible(item->checkState() == Qt::Checked);
                    label->updateLabel();
                    break;
                }
            } else if (m_currentObject->isKindOf(CV_TYPES::FACET)) {
                ccPlanarEntityInterface* plane =
                        ccHObjectCaster::ToPlanarEntity(m_currentObject);
                assert(plane);
                if (plane) {
                    plane->showNormalVector(item->checkState() == Qt::Checked);
                    ecvDisplayTools::SetRedrawRecursive(false);
                    m_currentObject->setRedrawFlagRecursive(true);
                    ecvDisplayTools::RedrawDisplay();
                    break;
                }
            } else if (m_currentObject->isKindOf(CV_TYPES::SENSOR)) {
                ccSensor* sensor = ccHObjectCaster::ToSensor(m_currentObject);
                if (sensor) {
                    sensor->setVisible(item->checkState() == Qt::Checked);
                    CC_DRAW_CONTEXT context;
                    context.visible =
                            sensor->isVisible() && sensor->isEnabled();
                    sensor->hideShowDrawings(context);
                    // for bbox
                    context.viewID = sensor->getViewId();
                    if (sensor->isSelected() && sensor->isEnabled()) {
                        // Check if Axes Grid is visible - if so, hide
                        // BoundingBox
                        bool shouldShowBB = true;
                        if (ecvDisplayTools::TheInstance()) {
                            AxesGridProperties axesGridProps;
                            ecvDisplayTools::TheInstance()
                                    ->getDataAxesGridProperties(context.viewID,
                                                                axesGridProps);
                            if (axesGridProps.visible) {
                                shouldShowBB = false;
                            }
                        }
                        if (shouldShowBB) {
                            sensor->showBB(context);
                        } else {
                            sensor->hideBB(context);
                        }
                    } else {
                        sensor->hideBB(context);
                    }
                    ecvDisplayTools::UpdateScreen();
                    break;
                }
            } else if (m_currentObject->isKindOf(CV_TYPES::PRIMITIVE)) {
                ccGenericPrimitive* prim =
                        ccHObjectCaster::ToPrimitive(m_currentObject);
                if (prim) {
                    prim->setVisible(item->checkState() == Qt::Checked);
                    CC_DRAW_CONTEXT context;
                    context.visible = prim->isVisible() && prim->isEnabled();
                    prim->hideShowDrawings(context);
                    // for bbox
                    context.viewID = prim->getViewId();
                    if (prim->isSelected() && prim->isEnabled()) {
                        // Check if Axes Grid is visible - if so, hide
                        // BoundingBox
                        bool shouldShowBB = true;
                        if (ecvDisplayTools::TheInstance()) {
                            AxesGridProperties axesGridProps;
                            ecvDisplayTools::TheInstance()
                                    ->getDataAxesGridProperties(context.viewID,
                                                                axesGridProps);
                            if (axesGridProps.visible) {
                                shouldShowBB = false;
                            }
                        }
                        if (shouldShowBB) {
                            prim->showBB(context);
                        } else {
                            prim->hideBB(context);
                        }
                    } else {
                        prim->hideBB(context);
                    }
                    ecvDisplayTools::UpdateScreen();
                    break;
                }
            } else {
                m_currentObject->setVisible(item->checkState() == Qt::Checked);
            }

            m_currentObject->setForceRedrawRecursive(true);

            bool objectIsDisplayed = m_currentObject->isDisplayed();
            if (objectWasDisplayed != objectIsDisplayed) {
                if (m_currentObject->isGroup())
                    emit ccObjectAndChildrenAppearanceChanged(m_currentObject,
                                                              false);
                else
                    emit ccObjectAppearanceChanged(m_currentObject, false);
            }
        } break;
        case OBJECT_NORMALS_SHOWN: {
            m_currentObject->showNormals(item->checkState() == Qt::Checked);
            if (m_currentObject->isKindOf(CV_TYPES::POINT_CLOUD)) {
                ecvDisplayTools::SetRedrawRecursive(false);
                m_currentObject->setRedrawFlagRecursive(true);
            }
        }
            redraw = true;
            break;
        case OBJECT_MATERIALS: {
            ccGenericMesh* mesh =
                    ccHObjectCaster::ToGenericMesh(m_currentObject);
            assert(mesh);
            mesh->showMaterials(item->checkState() == Qt::Checked);
            ecvDisplayTools::SetRedrawRecursive(false);
            mesh->setRedrawFlagRecursive(true);
        }
            redraw = true;
            break;
        case OBJECT_SF_SHOW_SCALE: {
            ccPointCloud* cloud =
                    ccHObjectCaster::ToPointCloud(m_currentObject);
            assert(cloud);
            cloud->showSFColorsScale(item->checkState() == Qt::Checked);
            ecvDisplayTools::SetRedrawRecursive(false);
            cloud->setRedrawFlagRecursive(true);
            ecvDisplayTools::RedrawDisplay(true);
        }
            redraw = false;
            break;
        case OBJECT_COORDINATE_SYSTEM_DISP_AXES: {
            ccCoordinateSystem* cs =
                    ccHObjectCaster::ToCoordinateSystem(m_currentObject);
            if (cs) {
                cs->ShowAxisLines(item->checkState() == Qt::Checked);
                CC_DRAW_CONTEXT context;
                context.visible = cs->isVisible();
                cs->hideShowDrawings(context);
                ecvDisplayTools::UpdateScreen();
            }
        }
            redraw = false;
            break;
        case OBJECT_COORDINATE_SYSTEM_DISP_PLANES: {
            ccCoordinateSystem* cs =
                    ccHObjectCaster::ToCoordinateSystem(m_currentObject);
            if (cs) {
                cs->ShowAxisPlanes(item->checkState() == Qt::Checked);
                ecvDisplayTools::SetRedrawRecursive(false);
                cs->setRedrawFlagRecursive(true);
            }
        }
            redraw = true;
            break;
        case OBJECT_FACET_CONTOUR: {
            ccFacet* facet = ccHObjectCaster::ToFacet(m_currentObject);
            assert(facet);
            if (facet && facet->getContour()) {
                facet->getContour()->setVisible(item->checkState() ==
                                                Qt::Checked);
                ecvDisplayTools::SetRedrawRecursive(false);
            }
        }
            redraw = true;
            break;
        case OBJECT_FACET_MESH: {
            ccFacet* facet = ccHObjectCaster::ToFacet(m_currentObject);
            assert(facet);
            if (facet && facet->getPolygon()) {
                facet->getPolygon()->setVisible(item->checkState() ==
                                                Qt::Checked);
                ecvDisplayTools::SetRedrawRecursive(false);
            }
        }
            redraw = true;
            break;
        case OBJECT_PLANE_NORMAL_VECTOR: {
            ccPlanarEntityInterface* plane =
                    ccHObjectCaster::ToPlanarEntity(m_currentObject);
            assert(plane);
            if (plane) {
                plane->showNormalVector(item->checkState() == Qt::Checked);
                ecvDisplayTools::SetRedrawRecursive(false);
                m_currentObject->setRedrawFlagRecursive(true);
            }
        }
            redraw = true;
            break;
        case OBJECT_MESH_WIRE: {
            ccGenericMesh* mesh =
                    ccHObjectCaster::ToGenericMesh(m_currentObject);
            assert(mesh);
            mesh->showWired(item->checkState() == Qt::Checked);
            ecvDisplayTools::SetRedrawRecursive(false);

            // unchecked points frame mode
            if (mesh->isShownAsWire()) {
                QStandardItem* item =
                        ITEM(tr("Pointsframe"), Qt::ItemIsUserCheckable,
                             OBJECT_MESH_POINTS);
                item->setCheckable(true);
                item->setCheckState(Qt::Unchecked);
            }
        }
            redraw = true;
            break;

        case OBJECT_MESH_POINTS: {
            ccGenericMesh* mesh =
                    ccHObjectCaster::ToGenericMesh(m_currentObject);
            assert(mesh);
            mesh->showPoints(item->checkState() == Qt::Checked);
            ecvDisplayTools::SetRedrawRecursive(false);

            // unchecked wired frame mode
            if (mesh->isShownAsPoints()) {
                QStandardItem* item =
                        ITEM(tr("Wireframe"), Qt::ItemIsUserCheckable,
                             OBJECT_MESH_WIRE);
                item->setCheckable(true);
                item->setCheckState(Qt::Unchecked);
            }
        }
            redraw = true;
            break;
        case OBJECT_MESH_STIPPLING: {
            ccGenericMesh* mesh =
                    ccHObjectCaster::ToGenericMesh(m_currentObject);
            assert(mesh);
            mesh->enableStippling(item->checkState() == Qt::Checked);
            ecvDisplayTools::SetRedrawRecursive(false);
        }
            redraw = true;
            break;
        case OBJECT_LABEL_DISP_2D: {
            cc2DLabel* label = ccHObjectCaster::To2DLabel(m_currentObject);
            assert(label);
            label->setDisplayedIn2D(item->checkState() == Qt::Checked);
            CC_DRAW_CONTEXT context;
            ecvDisplayTools::GetContext(context);
            label->update2DLabelView(context);
        }
            redraw = false;
            break;
        case OBJECT_LABEL_POINT_LEGEND: {
            cc2DLabel* label = ccHObjectCaster::To2DLabel(m_currentObject);
            assert(label);
            label->displayPointLegend(item->checkState() == Qt::Checked);
            CC_DRAW_CONTEXT context;
            ecvDisplayTools::GetContext(context);
            label->update2DLabelView(context);
        }
            redraw = false;
            break;
        case OBJECT_NAME_IN_3D: {
            m_currentObject->showNameIn3D(item->checkState() == Qt::Checked);
            ecvDisplayTools::SetRedrawRecursive(false);
        }
            redraw = true;
            break;
        case OBJECT_SHOW_TRANS_BUFFER_PATH: {
            ccIndexedTransformationBuffer* buffer =
                    ccHObjectCaster::ToTransBuffer(m_currentObject);
            assert(buffer);
            buffer->showPathAsPolyline(item->checkState() == Qt::Checked);
        }
            redraw = true;
            break;
        case OBJECT_SHOW_TRANS_BUFFER_TRIHDERONS: {
            ccIndexedTransformationBuffer* buffer =
                    ccHObjectCaster::ToTransBuffer(m_currentObject);
            assert(buffer);
            buffer->showTriherdons(item->checkState() == Qt::Checked);
        }
            redraw = true;
            break;
        case OBJECT_SENSOR_DRAW_FRUSTUM: {
            ccCameraSensor* sensor =
                    ccHObjectCaster::ToCameraSensor(m_currentObject);
            sensor->drawFrustum(item->checkState() == Qt::Checked);
            ecvDisplayTools::SetRedrawRecursive(false);
            sensor->setRedrawFlagRecursive(true);
            ecvDisplayTools::RedrawDisplay();
        }
            redraw = false;
            break;
        case OBJECT_SENSOR_DRAW_FRUSTUM_PLANES: {
            ccCameraSensor* sensor =
                    ccHObjectCaster::ToCameraSensor(m_currentObject);
            sensor->drawFrustumPlanes(item->checkState() == Qt::Checked);
            ecvDisplayTools::SetRedrawRecursive(false);
            sensor->setRedrawFlagRecursive(true);
            ecvDisplayTools::RedrawDisplay();
        }
            redraw = false;
            break;
        // ParaView-style View Properties handlers
        case OBJECT_VIEW_CAMERA_ORIENTATION_WIDGET: {
            bool visible = (item->checkState() == Qt::Checked);
            ecvDisplayTools::ToggleCameraOrientationWidget(visible);
            CVLog::Print(
                    QString("[View Properties] Camera Orientation Widget: %1")
                            .arg(visible ? "ON" : "OFF"));
        }
            redraw = false;
            break;
        case OBJECT_VIEW_LIGHT_KIT_INTENSITY: {
            // Light intensity is handled by editor's signal connection
            // The slider/spinbox valueChanged signal triggers the update
            // No action needed here in updateItem
        }
            redraw = false;
            break;
        case OBJECT_VIEW_DATA_AXES_GRID_VISIBLE: {
            // ParaView-style: handled in setEditorData (persistent editor)
            // No action needed here in updateItem
        }
            redraw = false;
            break;
    }

    if (redraw) {
        updateDisplay();
    }
}

void ccPropertiesTreeDelegate::updateDisplay() {
    ccHObject* object = m_currentObject;
    if (!object) return;

    bool objectIsDisplayed = object->isDisplayed();
    if (!objectIsDisplayed) {
        // DGM: point clouds may be mesh vertices of meshes which may depend on
        // several of their parameters
        if (object->isKindOf(CV_TYPES::POINT_CLOUD)) {
            ccHObject* parent = object->getParent();
            if (parent && parent->isKindOf(CV_TYPES::MESH) &&
                parent->isDisplayed())  // specific case: vertices
            {
                object = parent;
                objectIsDisplayed = true;
            }
        }
        // Allows show name toggle on normally non-visible objects to update the
        // screen
        else if (object->isKindOf(CV_TYPES::HIERARCHY_OBJECT)) {
            objectIsDisplayed = true;
        }
    }

    if (objectIsDisplayed) {
        if (object->isGroup())
            emit ccObjectAndChildrenAppearanceChanged(m_currentObject);
        else
            emit ccObjectAppearanceChanged(m_currentObject);
    }
}

void ccPropertiesTreeDelegate::updateModel() {
    // simply re-fill model!
    fillModel(m_currentObject);
}

// Note: updateSelectionProperties has been removed. Selection properties
// updates are now handled directly by cvFindDataDockWidget.

QMap<QString, QString> ccPropertiesTreeDelegate::getCurrentMeshTexturePathMap()
        const {
    if (!m_currentObject) {
        return QMap<QString, QString>();
    }
    return m_meshTexturePathMaps.value(m_currentObject,
                                       QMap<QString, QString>());
}

void ccPropertiesTreeDelegate::clearMeshTexturePathMap(ccHObject* mesh) {
    if (mesh) {
        m_meshTexturePathMaps.remove(mesh);
    }
}

void ccPropertiesTreeDelegate::scalarFieldChanged(int pos) {
    if (!m_currentObject) return;

    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_currentObject);
    if (cloud && cloud->getCurrentDisplayedScalarFieldIndex() + 1 != pos) {
        cloud->setCurrentDisplayedScalarField(pos - 1);
        cloud->showSF(pos > 0);

        updateCurrentEntity(pos > 0);

        // we must also reset the properties display!
        updateModel();
    }
}

void ccPropertiesTreeDelegate::spawnTextureFileEditor() {
    if (!m_currentObject) return;

    ecvTextureFileSelector* selector =
            dynamic_cast<ecvTextureFileSelector*>(QObject::sender());
    if (!selector) return;

    // get current material
    ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(m_currentObject);
    assert(mesh);
    const ccMaterialSet* materialSet = mesh ? mesh->getMaterialSet() : nullptr;
    if (materialSet) {
        // persistent settings
        QString currentPath =
                ecvSettingManager::getValue(ecvPS::LoadFile(),
                                            ecvPS::TextureFilePath(),
                                            ecvFileUtils::defaultDocPath())
                        .toString();
        QString currentOpenDlgFilter =
                ecvSettingManager::getValue(ecvPS::LoadFile(),
                                            ecvPS::SelectedImageInputFilter(),
                                            "*.png")
                        .toString();

        QStringList fileFilters;
        // we grab the list of supported image file formats (reading)
        QList<QByteArray> formats = QImageReader::supportedImageFormats();
        if (formats.empty()) {
            fileFilters << "*.bmp"
                        << "*.png"
                        << "*.jpg"
                        << "Image file (*.*)";
        } else {
            // we convert this list into a proper "filters" string
            for (int i = 0; i < formats.size(); ++i) {
                QString filter = QString("*.%1").arg(formats[i].data());
                fileFilters.append(filter);
            }
            fileFilters.append("Image file (*.*)");
        }

        // dialog options
        QFileDialog::Options dialogOptions = QFileDialog::Options();
        if (!ecvOptions::Instance().useNativeDialogs) {
            dialogOptions |= QFileDialog::DontUseNativeDialog;
        }

        // file choosing dialog
        QString selectedFiles = QFileDialog::getOpenFileName(
                MainWindow::TheInstance(), tr("Open Texture file(s)"),
                currentPath, fileFilters.join(";;"), &currentOpenDlgFilter,
                dialogOptions);

        if (selectedFiles.isEmpty()) return;

        // persistent save last loading parameters
        currentPath = QFileInfo(selectedFiles).absolutePath();
        ecvSettingManager::setValue(ecvPS::LoadFile(), ecvPS::TextureFilePath(),
                                    currentPath);
        ecvSettingManager::setValue(ecvPS::LoadFile(),
                                    ecvPS::SelectedImageInputFilter(),
                                    currentOpenDlgFilter);

        if (QFileInfo(selectedFiles).exists() && mesh) {
            // Add to current mesh's texture path map
            QMap<QString, QString>& texturePathMap =
                    m_meshTexturePathMaps[m_currentObject];
            QString fileName = QFileInfo(selectedFiles).fileName();
            texturePathMap[fileName] = selectedFiles;
            selector->addItem(fileName, selectedFiles);
            selector->setSelectedTexturefile(selectedFiles);
        }

        updateModel();
    }
}

void ccPropertiesTreeDelegate::spawnColorRampEditor() {
    if (!m_currentObject) return;

    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_currentObject);
    assert(cloud);
    ccScalarField* sf =
            (cloud ? static_cast<ccScalarField*>(
                             cloud->getCurrentDisplayedScalarField())
                   : nullptr);
    if (sf) {
        ccColorScaleEditorDialog* editorDialog = new ccColorScaleEditorDialog(
                ccColorScalesManager::GetUniqueInstance(),
                MainWindow::TheInstance(), sf->getColorScale(),
                ecvDisplayTools::GetMainWindow());
        editorDialog->setAssociatedScalarField(sf);
        if (editorDialog->exec()) {
            if (editorDialog->getActiveScale()) {
                sf->setColorScale(editorDialog->getActiveScale());
                updateCurrentEntity();
            }

            // save current scale manager state to persistent settings
            ccColorScalesManager::GetUniqueInstance()->toPersistentSettings();

            updateModel();
        }
    }
}

void ccPropertiesTreeDelegate::textureFileChanged(int pos) {
    if (!m_currentObject) return;

    if (pos < 0) {
        assert(false);
        return;
    }

    ecvTextureFileSelector* selector =
            dynamic_cast<ecvTextureFileSelector*>(QObject::sender());
    if (!selector) return;

    QString textureFilepath = selector->getTexturefilePath(pos);
    if (textureFilepath.isEmpty()) {
        return;
    }

    if (!QFileInfo(textureFilepath).exists()) {
        CVLog::Error(
                tr("Internal error: texture file : %1 doesn't exist anymore!")
                        .arg(textureFilepath));
        return;
    }

    // get current material
    ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(m_currentObject);
    assert(mesh);
    const ccMaterialSet* materialSet = mesh ? mesh->getMaterialSet() : nullptr;
    if (materialSet && materialSet->findMaterialByName(
                               QFileInfo(textureFilepath).fileName()) < 0) {
        if (!mesh->updateTextures(CVTools::FromQString(textureFilepath))) {
            CVLog::Warning(
                    "Update Textures failed, please toggle shown material "
                    "first!");
        } else {
            ecvDisplayTools::UpdateScreen();
        }

        updateModel();
    }
}

void ccPropertiesTreeDelegate::colorScaleChanged(int pos) {
    if (!m_currentObject) return;

    if (pos < 0) {
        assert(false);
        return;
    }

    ccColorScaleSelector* selector =
            dynamic_cast<ccColorScaleSelector*>(QObject::sender());
    if (!selector) return;

    ccColorScale::Shared colorScale = selector->getScale(pos);
    if (!colorScale) {
        CVLog::Error(tr(
                "Internal error: color scale doesn't seem to exist anymore!"));
        return;
    }

    // get current SF
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_currentObject);
    assert(cloud);
    ccScalarField* sf = cloud ? static_cast<ccScalarField*>(
                                        cloud->getCurrentDisplayedScalarField())
                              : nullptr;
    if (sf && sf->getColorScale() != colorScale) {
        sf->setColorScale(colorScale);

        updateCurrentEntity();
        updateModel();
    }
}

void ccPropertiesTreeDelegate::colorRampStepsChanged(int pos) {
    if (!m_currentObject) return;

    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_currentObject);
    assert(cloud);
    ccScalarField* sf = static_cast<ccScalarField*>(
            cloud->getCurrentDisplayedScalarField());
    if (sf && sf->getColorRampSteps() != static_cast<unsigned>(pos)) {
        sf->setColorRampSteps(static_cast<unsigned>(pos));
        updateCurrentEntity();
    }
}

void ccPropertiesTreeDelegate::octreeDisplayModeChanged(int pos) {
    if (!m_currentObject) return;
    QComboBox* comboBox = dynamic_cast<QComboBox*>(QObject::sender());
    if (!comboBox) return;

    ccOctree* octree = ccHObjectCaster::ToOctree(m_currentObject);
    assert(octree);

    int mode = comboBox->itemData(pos, Qt::UserRole).toInt();
    if (octree->getDisplayMode() != mode) {
        octree->setDisplayMode(static_cast<ccOctree::DisplayMode>(mode));

        ecvDisplayTools::SetRedrawRecursive(false);
        ccOctreeProxy* octreeProxy =
                ccHObjectCaster::ToOctreeProxy(m_currentObject);
        octreeProxy->setRedrawFlagRecursive(true);

        updateDisplay();
    }
}

void ccPropertiesTreeDelegate::octreeDisplayedLevelChanged(int val) {
    if (!m_currentObject) return;

    ccOctree* octree = ccHObjectCaster::ToOctree(m_currentObject);
    assert(octree);

    if (octree->getDisplayedLevel() != val)  // to avoid infinite loops!
    {
        octree->setDisplayedLevel(val);

        updateCurrentEntity();

        // we must also reset the properties display!
        updateModel();
    }
}

void ccPropertiesTreeDelegate::circleResolutionChanged(int val) {
    if (!m_currentObject) {
        return;
    }

    ccCircle* circle = ccHObjectCaster::ToCircle(m_currentObject);
    assert(circle);
    if (!circle) return;

    if (circle->getResolution() != static_cast<unsigned int>(val)) {
        bool wasVisible = circle->isVisible();
        circle->setResolution(val);
        circle->setVisible(wasVisible);

        updateCurrentEntity();

        // record item role to force the scroll focus (see 'createEditor').
        m_lastFocusItemRole = OBJECT_CIRCLE_RESOLUTION;

        // we must also reset the properties display!
        updateModel();
    }
}

void ccPropertiesTreeDelegate::circleRadiusChanged(double val) {
    if (!m_currentObject) return;

    ccCircle* circle = ccHObjectCaster::ToCircle(m_currentObject);
    assert(circle);
    if (!circle) return;

    if (circle->getRadius() != val) {
        bool wasVisible = circle->isVisible();
        circle->setRadius(val);
        circle->setVisible(wasVisible);

        updateCurrentEntity();

        // record item role to force the scroll focus (see 'createEditor').
        m_lastFocusItemRole = OBJECT_CIRCLE_RADIUS;

        // we must also reset the properties display!
        updateModel();
    }
}

void ccPropertiesTreeDelegate::discRadiusChanged(double val) {
    if (!m_currentObject) return;

    ccDisc* disc = ccHObjectCaster::ToDisc(m_currentObject);
    assert(disc);
    if (!disc) return;

    PointCoordinateType radius = static_cast<PointCoordinateType>(val);
    if (disc->getRadius() != radius) {
        disc->setRadius(radius);

        updateCurrentEntity();

        // record item role to force the scroll focus (see 'createEditor').
        m_lastFocusItemRole = OBJECT_DISC_RADIUS;
        updateModel();
    }
}

void ccPropertiesTreeDelegate::primitivePrecisionChanged(int val) {
    if (!m_currentObject) return;

    ccGenericPrimitive* primitive =
            ccHObjectCaster::ToPrimitive(m_currentObject);
    assert(primitive);

    if (primitive->getDrawingPrecision() != static_cast<unsigned int>(val)) {
        bool wasVisible = primitive->isVisible();
        primitive->setDrawingPrecision(static_cast<unsigned>(val));
        primitive->setVisible(wasVisible);

        updateCurrentEntity();

        // we must also reset the properties display!
        updateModel();
    }
}

void ccPropertiesTreeDelegate::sphereRadiusChanged(double val) {
    if (!m_currentObject) return;

    ccSphere* sphere = ccHObjectCaster::ToSphere(m_currentObject);
    assert(sphere);

    PointCoordinateType radius = static_cast<PointCoordinateType>(val);
    if (sphere->getRadius() != radius) {
        bool wasVisible = sphere->isVisible();
        sphere->setRadius(radius);
        sphere->setVisible(wasVisible);

        updateCurrentEntity();

        // we must also reset the properties display!
        updateModel();
    }
}

void ccPropertiesTreeDelegate::coneHeightChanged(double val) {
    if (!m_currentObject) return;

    ccCone* cone = ccHObjectCaster::ToCone(m_currentObject);
    assert(cone);

    PointCoordinateType height = static_cast<PointCoordinateType>(val);
    if (cone->getHeight() != height) {
        bool wasVisible = cone->isVisible();
        cone->setHeight(height);
        cone->setVisible(wasVisible);

        updateCurrentEntity();

        // we must also reset the properties display!
        updateModel();
    }
}

void ccPropertiesTreeDelegate::coneBottomRadiusChanged(double val) {
    if (!m_currentObject) return;

    ccCone* cone = ccHObjectCaster::ToCone(m_currentObject);
    assert(cone);

    PointCoordinateType radius = static_cast<PointCoordinateType>(val);
    if (cone->getBottomRadius() != radius) {
        bool wasVisible = cone->isVisible();
        cone->setBottomRadius(radius);  // works for both the bottom and top
                                        // radii for cylinders!
        cone->setVisible(wasVisible);

        updateCurrentEntity();

        // we must also reset the properties display!
        updateModel();
    }
}

void ccPropertiesTreeDelegate::coneTopRadiusChanged(double val) {
    if (!m_currentObject) return;

    ccCone* cone = ccHObjectCaster::ToCone(m_currentObject);
    assert(cone);

    PointCoordinateType radius = static_cast<PointCoordinateType>(val);
    if (cone->getTopRadius() != radius) {
        bool wasVisible = cone->isVisible();
        cone->setTopRadius(radius);  // works for both the bottom and top radii
                                     // for cylinders!
        cone->setVisible(wasVisible);

        updateCurrentEntity();

        // we must also reset the properties display!
        updateModel();
    }
}

void ccPropertiesTreeDelegate::imageAlphaChanged(int val) {
    ccImage* image = ccHObjectCaster::ToImage(m_currentObject);

    float alpha = val / 255.0f;
    if (image && image->getAlpha() != alpha) {
        image->setAlpha(alpha);
        ecvDisplayTools::ChangeOpacity(
                alpha, CVTools::FromQString(image->getViewId()));
    }
}

void ccPropertiesTreeDelegate::opacityChanged(int val) {
    if (!m_currentObject) return;

    // Convert slider value [0, 100] to opacity [0.0, 1.0]
    float opacity = val / 100.0f;

    // Check if this is a folder with children
    if (m_currentObject->getChildrenNumber() > 0) {
        // For folders, apply opacity to all renderable children recursively
        std::function<void(ccHObject*, float)> applyOpacityRecursive =
                [&applyOpacityRecursive](ccHObject* obj, float op) {
                    if (!obj || !obj->isEnabled()) return;

                    // Check if this is a renderable object
                    bool isRenderable = (obj->isKindOf(CV_TYPES::POINT_CLOUD) ||
                                         obj->isKindOf(CV_TYPES::MESH) ||
                                         obj->isKindOf(CV_TYPES::PRIMITIVE) ||
                                         obj->isKindOf(CV_TYPES::POLY_LINE) ||
                                         obj->isKindOf(CV_TYPES::FACET));

                    if (isRenderable) {
                        // Check if opacity actually changed to avoid
                        // unnecessary updates
                        if (std::abs(obj->getOpacity() - op) >= 0.001f) {
                            obj->setOpacity(op);

                            // Determine entity type for proper property
                            // application
                            ENTITY_TYPE entityType =
                                    ENTITY_TYPE::ECV_POINT_CLOUD;
                            if (obj->isKindOf(CV_TYPES::POINT_CLOUD)) {
                                entityType = ENTITY_TYPE::ECV_POINT_CLOUD;
                            } else if (obj->isKindOf(CV_TYPES::MESH) ||
                                       obj->isKindOf(CV_TYPES::PRIMITIVE)) {
                                entityType = ENTITY_TYPE::ECV_MESH;
                            } else if (obj->isKindOf(CV_TYPES::POLY_LINE)) {
                                entityType = ENTITY_TYPE::ECV_LINES_3D;
                            } else if (obj->isKindOf(CV_TYPES::FACET)) {
                                entityType = ENTITY_TYPE::ECV_MESH;
                            }

                            // Create property parameter and apply opacity
                            // change
                            PROPERTY_PARAM param(obj, static_cast<double>(op));
                            param.entityType = entityType;
                            param.viewId = obj->getViewId();
                            param.viewport = 0;

                            // Apply the opacity change via display tools
                            ecvDisplayTools::ChangeEntityProperties(param,
                                                                    true);
                        }
                    }

                    // Recursively process children
                    for (unsigned i = 0; i < obj->getChildrenNumber(); ++i) {
                        applyOpacityRecursive(obj->getChild(i), op);
                    }
                };

        applyOpacityRecursive(m_currentObject, opacity);

        CVLog::PrintVerbose(
                QString("[ccPropertiesTreeDelegate::opacityChanged] "
                        "Set opacity to %1 for folder '%2' and all renderable "
                        "children")
                        .arg(opacity)
                        .arg(m_currentObject->getName()));
    } else {
        // Single object - original behavior
        // Check if opacity actually changed to avoid unnecessary updates
        if (std::abs(m_currentObject->getOpacity() - opacity) < 0.001f) {
            return;
        }

        // Store the new opacity in the object
        m_currentObject->setOpacity(opacity);

        // Determine entity type for proper property application
        ENTITY_TYPE entityType = ENTITY_TYPE::ECV_POINT_CLOUD;  // Default

        if (m_currentObject->isKindOf(CV_TYPES::POINT_CLOUD)) {
            entityType = ENTITY_TYPE::ECV_POINT_CLOUD;
        } else if (m_currentObject->isKindOf(CV_TYPES::MESH) ||
                   m_currentObject->isKindOf(CV_TYPES::PRIMITIVE)) {
            entityType = ENTITY_TYPE::ECV_MESH;
        } else if (m_currentObject->isKindOf(CV_TYPES::POLY_LINE)) {
            entityType = ENTITY_TYPE::ECV_LINES_3D;
        } else if (m_currentObject->isKindOf(CV_TYPES::FACET)) {
            entityType = ENTITY_TYPE::ECV_MESH;
        }

        // Create property parameter and apply opacity change
        PROPERTY_PARAM param(m_currentObject, static_cast<double>(opacity));
        param.entityType = entityType;
        param.viewId = m_currentObject->getViewId();
        param.viewport = 0;

        // Apply the opacity change via display tools
        ecvDisplayTools::ChangeEntityProperties(param, true);

        CVLog::PrintVerbose(
                QString("[ccPropertiesTreeDelegate::opacityChanged] "
                        "Set opacity to %1 for object '%2'")
                        .arg(opacity)
                        .arg(m_currentObject->getName()));
    }
}

void ccPropertiesTreeDelegate::applyImageViewport() {
    if (!m_currentObject) return;

    ccImage* image = ccHObjectCaster::ToImage(m_currentObject);
    assert(image);

    if (image->getAssociatedSensor() &&
        image->getAssociatedSensor()->applyViewport()) {
        CVLog::Print("[ApplyImageViewport] Viewport applied");
    }
}

void ccPropertiesTreeDelegate::applySensorViewport() {
    if (!m_currentObject) return;

    ccSensor* sensor = ccHObjectCaster::ToSensor(m_currentObject);
    assert(sensor);

    if (sensor->applyViewport()) {
        CVLog::Print(tr("[ApplySensorViewport] Viewport applied"));
    }
}

void ccPropertiesTreeDelegate::applyLabelViewport() {
    if (!m_currentObject) return;

    cc2DViewportObject* viewport =
            ccHObjectCaster::To2DViewportObject(m_currentObject);
    assert(viewport);

    ecvDisplayTools::SetViewportParameters(viewport->getParameters());
    ecvDisplayTools::UpdateScreen();
}

void ccPropertiesTreeDelegate::updateLabelViewport() {
    if (!m_currentObject) return;

    cc2DViewportObject* viewport =
            ccHObjectCaster::To2DViewportObject(m_currentObject);
    assert(viewport);

    viewport->setParameters(ecvDisplayTools::GetViewportParameters());
    CVLog::Print(tr("Viewport '%1' has been updated").arg(viewport->getName()));
}

void ccPropertiesTreeDelegate::sensorScaleChanged(double val) {
    if (!m_currentObject) return;

    ccSensor* sensor = ccHObjectCaster::ToSensor(m_currentObject);
    assert(sensor);

    if (sensor &&
        sensor->getGraphicScale() != static_cast<PointCoordinateType>(val)) {
        sensor->setGraphicScale(static_cast<PointCoordinateType>(val));
        updateCurrentEntity();
    }
}

void ccPropertiesTreeDelegate::coordinateSystemDisplayScaleChanged(double val) {
    if (!m_currentObject) {
        return;
    }

    ccCoordinateSystem* cs =
            ccHObjectCaster::ToCoordinateSystem(m_currentObject);
    assert(cs);

    if (cs && cs->getDisplayScale() != static_cast<PointCoordinateType>(val)) {
        cs->setDisplayScale(static_cast<PointCoordinateType>(val));
        updateDisplay();
    }
}

void ccPropertiesTreeDelegate::sensorUncertaintyChanged() {
    if (!m_currentObject) return;

    QLineEdit* lineEdit = qobject_cast<QLineEdit*>(QObject::sender());
    if (!lineEdit) {
        assert(false);
        return;
    }

    ccGBLSensor* sensor = ccHObjectCaster::ToGBLSensor(m_currentObject);
    assert(sensor);

    PointCoordinateType uncertainty =
            static_cast<PointCoordinateType>(lineEdit->text().toDouble());
    if (sensor && sensor->getUncertainty() != uncertainty) {
        sensor->setUncertainty(uncertainty);
    }
}

void ccPropertiesTreeDelegate::sensorIndexChanged(double val) {
    if (!m_currentObject) return;

    ccSensor* sensor = ccHObjectCaster::ToSensor(m_currentObject);
    assert(sensor);

    if (sensor && sensor->getActiveIndex() != val) {
        sensor->setActiveIndex(val);
        updateCurrentEntity();
    }
}

void ccPropertiesTreeDelegate::trihedronsScaleChanged(double val) {
    if (!m_currentObject) return;

    ccIndexedTransformationBuffer* buffer =
            ccHObjectCaster::ToTransBuffer(m_currentObject);
    assert(buffer);

    if (buffer && buffer->triherdonsDisplayScale() != static_cast<float>(val)) {
        buffer->setTriherdonsDisplayScale(static_cast<float>(val));
        if (buffer->triherdonsShown()) {
            updateCurrentEntity();
        }
    }
}

void ccPropertiesTreeDelegate::cloudPointSizeChanged(int size) {
    if (!m_currentObject) return;

    ccGenericPointCloud* cloud =
            ccHObjectCaster::ToGenericPointCloud(m_currentObject);
    assert(cloud);

    if (cloud && cloud->getPointSize() != size) {
        cloud->setPointSize(static_cast<unsigned>(size));
        updateCurrentEntity(false);
    }
}

void ccPropertiesTreeDelegate::polyineWidthChanged(int size) {
    if (!m_currentObject) return;

    ccPolyline* polyline = ccHObjectCaster::ToPolyline(m_currentObject);
    assert(polyline);

    if (polyline &&
        polyline->getWidth() != static_cast<PointCoordinateType>(size)) {
        polyline->setWidth(static_cast<PointCoordinateType>(size));
        updateCurrentEntity(false);
    }
}

void ccPropertiesTreeDelegate::coordinateSystemAxisWidthChanged(int size) {
    if (!m_currentObject) {
        return;
    }

    ccCoordinateSystem* cs =
            ccHObjectCaster::ToCoordinateSystem(m_currentObject);
    assert(cs);

    if (cs && cs->getAxisWidth() != static_cast<PointCoordinateType>(size)) {
        cs->setAxisWidth(static_cast<PointCoordinateType>(size));
        updateDisplay();
    }
}

void ccPropertiesTreeDelegate::objectDisplayChanged(
        const QString& newDisplayTitle) {
    if (!m_currentObject) return;

    QString actualDisplayTitle = Settings::APP_TITLE;

    if (actualDisplayTitle != newDisplayTitle) {
        // we first mark the "old displays" before removal,
        // to be sure that they will also be redrawn!
        // m_currentObject->prepareDisplayForRefresh_recursive();

        // ccGLWindow* win = MainWindow::GetGLWindow(newDisplayTitle);
        // m_currentObject->setDisplay_recursive(win);
        // if (win)
        //{
        //	m_currentObject->prepareDisplayForRefresh_recursive();
        //	win->zoomGlobal();
        // }

        // MainWindow::TheInstance()->refreshAll();
    }
}

void ccPropertiesTreeDelegate::colorSourceChanged(const QString& source) {
    if (!m_currentObject) return;

    bool appearanceChanged = false;

    ecvDisplayTools::SetRedrawRecursive(false);

    if (source == s_noneString) {
        appearanceChanged =
                m_currentObject->colorsShown() || m_currentObject->sfShown();
        m_currentObject->showColors(false);
        m_currentObject->showSF(false);
    } else if (source == s_rgbColor) {
        appearanceChanged =
                !m_currentObject->colorsShown() || m_currentObject->sfShown();
        m_currentObject->showColors(true);
        m_currentObject->showSF(false);
        if (m_currentObject->hasColors() &&
            !m_currentObject->isColorOverridden()) {
            m_currentObject->setRedrawFlagRecursive(true);
        }
    } else if (source == s_sfColor) {
        appearanceChanged =
                m_currentObject->colorsShown() || !m_currentObject->sfShown();
        m_currentObject->showColors(false);
        m_currentObject->showSF(true);
    } else {
        // assert(false);
        CVLog::Warning(QString("unsupported source type [%1]").arg(source));
    }

    if (appearanceChanged) {
        updateDisplay();
    } else {
        ecvDisplayTools::SetRedrawRecursive(true);
    }
}

void ccPropertiesTreeDelegate::updateCurrentEntity(bool redraw /* = true*/) {
    ecvDisplayTools::SetRedrawRecursive(false);
    m_currentObject->setRedrawFlagRecursive(redraw);
    updateDisplay();
}

// ParaView-style View Properties implementation

void ccPropertiesTreeDelegate::lightIntensityChanged(double intensity) {
    if (!ecvDisplayTools::TheInstance()) {
        return;
    }

    // Apply light intensity to backend
    ecvDisplayTools::TheInstance()->setLightIntensity(intensity);

    // Trigger screen update
    ecvDisplayTools::UpdateScreen();
}

void ccPropertiesTreeDelegate::dataAxesGridEditRequested() {
    // Get viewID from current object
    if (!m_currentObject) {
        return;
    }

    QString viewID = m_currentObject->getViewId();
    if (viewID.isEmpty()) {
        return;
    }

    // Check if we have a valid display tools instance
    if (!ecvDisplayTools::TheInstance()) {
        return;
    }

    // Create and show dialog with current properties
    ecvAxesGridDialog dialog(tr("Data Axes Grid Properties"), m_view);

    // Get current properties from backend (using struct-based interface)
    AxesGridProperties props;

    try {
        ecvDisplayTools::TheInstance()->getDataAxesGridProperties(viewID,
                                                                  props);
    } catch (const std::exception& e) {
        CVLog::Warning(
                QString("[Data Axes Grid] Exception getting properties: %1")
                        .arg(e.what()));
        props = AxesGridProperties();
    } catch (...) {
        CVLog::Warning("[Data Axes Grid] Unknown exception getting properties");
        props = AxesGridProperties();
    }

    // Set current values in dialog from backend
    try {
        // Clamp color values to valid range [0, 255]
        int r = std::max(0, std::min(255, static_cast<int>(props.color.x)));
        int g = std::max(0, std::min(255, static_cast<int>(props.color.y)));
        int b = std::max(0, std::min(255, static_cast<int>(props.color.z)));
        dialog.setColor(QColor::fromRgb(r, g, b));

        // Set all properties from backend
        dialog.setLineWidth(props.lineWidth);
        dialog.setOpacity(props.opacity);
        dialog.setShowLabels(props.showLabels);
        dialog.setShowGrid(props.showGrid);
        dialog.setXTitle(props.xTitle);
        dialog.setYTitle(props.yTitle);
        dialog.setZTitle(props.zTitle);
        dialog.setXAxisUseCustomLabels(props.xUseCustomLabels);
        dialog.setYAxisUseCustomLabels(props.yUseCustomLabels);
        dialog.setZAxisUseCustomLabels(props.zUseCustomLabels);
        dialog.setXAxisCustomLabels(props.xCustomLabels);
        dialog.setYAxisCustomLabels(props.yCustomLabels);
        dialog.setZAxisCustomLabels(props.zCustomLabels);
        dialog.setUseCustomBounds(props.useCustomBounds);

        // Set custom bounds values (FIX: These were missing!)
        dialog.setXMin(props.xMin);
        dialog.setXMax(props.xMax);
        dialog.setYMin(props.yMin);
        dialog.setYMax(props.yMax);
        dialog.setZMin(props.zMin);
        dialog.setZMax(props.zMax);
    } catch (const std::exception& e) {
        CVLog::Warning(QString("[Data Axes Grid] Exception setting dialog "
                               "properties: %1")
                               .arg(e.what()));
    } catch (...) {
        CVLog::Warning(
                "[Data Axes Grid] Unknown exception setting dialog properties");
    }

    // Lambda for applying properties (used by both Apply and OK buttons)
    auto applyProperties = [&]() {
        // Use the new struct-based interface (cleaner API)
        AxesGridProperties props;

        // Get all values from dialog
        QColor dialogColor = dialog.getGridColor();
        props.visible = true;  // Auto-enable when editing (ParaView-style)
        props.color = CCVector3(dialogColor.red(), dialogColor.green(),
                                dialogColor.blue());
        props.lineWidth = dialog.getLineWidth();
        // Keep spacing and subdivisions from backend (not exposed in dialog
        // yet) props.spacing and props.subdivisions already set from
        // getDataAxesGridProperties
        props.showLabels = dialog.getShowLabels();
        props.opacity = dialog.getOpacity();

        // Extended properties
        props.showGrid = dialog.getShowGrid();
        props.xTitle = dialog.getXTitle();
        props.yTitle = dialog.getYTitle();
        props.zTitle = dialog.getZTitle();
        props.xUseCustomLabels = dialog.getXAxisUseCustomLabels();
        props.yUseCustomLabels = dialog.getYAxisUseCustomLabels();
        props.zUseCustomLabels = dialog.getZAxisUseCustomLabels();
        props.useCustomBounds = dialog.getUseCustomBounds();

        // Custom labels (ParaView-style)
        props.xCustomLabels = dialog.getXAxisCustomLabels();
        props.yCustomLabels = dialog.getYAxisCustomLabels();
        props.zCustomLabels = dialog.getZAxisCustomLabels();

        // Custom bounds values (FIX: These were missing!)
        props.xMin = dialog.getXMin();
        props.xMax = dialog.getXMax();
        props.yMin = dialog.getYMin();
        props.yMax = dialog.getYMax();
        props.zMin = dialog.getZMin();
        props.zMax = dialog.getZMax();

        // Apply using clean struct-based interface
        ecvDisplayTools::TheInstance()->setDataAxesGridProperties(viewID,
                                                                  props);
        ecvDisplayTools::UpdateScreen();
    };

    // Connect Apply button for real-time preview (ParaView-style)
    connect(&dialog, &ecvAxesGridDialog::applyRequested, this, applyProperties);

    // Show non-modal dialog (allows moving and interacting with scene)
    dialog.show();

    // Wait for dialog to close
    int result = dialog.exec();

    // Apply on OK (final confirmation)
    if (result == QDialog::Accepted) {
        applyProperties();
    }
}

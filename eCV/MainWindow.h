﻿//##########################################################################
//#                                                                        #
//#                              EROWCLOUDVIEWER                           #
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

#ifndef ECV_MAIN_WINDOW_HEADER
#define ECV_MAIN_WINDOW_HEADER

// LOCAL
#include "ecvEntityAction.h"
#include "ecvMainAppInterface.h"
#include "ecvPickingListener.h"

// CV_CORE_LIB
#include <AutoSegmentationTools.h>
#include <CVTools.h>

// QT
#include <QAction>
#include <QColorDialog>
#include <QDebug>
#include <QDesktopServices>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QMouseEvent>
#include <QStatusBar>
#include <QString>
#include <QTextEdit>
#include <QTime>
#include <QToolBar>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

// system
#include <algorithm>
#include <map>
#include <vector>

const int CLOUDVIEWER_LANG_ENGLISH = 0;
const int CLOUDVIEWER_LANG_CHINESE = 1;

using std::map;
using std::string;
using std::vector;

class ccHObject;
class ccPickingHub;
class ccPluginUIManager;
class ccDBRoot;
class ecvFilterTool;
class ecvRecentFiles;
class ecvAnnotationsTool;
class ecvFilterWindowTool;
class ccTracePolylineTool;
class ccGraphicalSegmentationTool;
class ccGraphicalTransformationTool;
class ecvDeepSemanticSegmentationTool;

class ecvUpdateDlg;
class ccOverlayDialog;
class ccComparisonDlg;
class ecvFilterByLabelDlg;
class ccPointPropertiesDlg;
class ecvCameraParamEditDlg;
class ecvAnimationParamDlg;
class ccPointListPickingDlg;
class ecvPrimitiveFactoryDlg;
class ccPointPairRegistrationDlg;

class QMdiArea;
class QMdiSubWindow;
class QTreeWidgetItem;
class QUIWidget;

struct dbTreeSelectionInfo;

namespace Ui {
class MainViewerClass;
}

#ifdef BUILD_RECONSTRUCTION
namespace cloudViewer {
class ReconstructionWidget;
}
#endif

class QSimpleUpdater;

class MainWindow : public QMainWindow,
                   public ecvMainAppInterface,
                   public ccPickingListener {
    Q_OBJECT

protected:
    MainWindow();
    ~MainWindow() override;

public:  // static method
    //! Static shortcut to MainWindow::updateUI
    static void UpdateUI();

    //! Returns the unique instance of this object
    static MainWindow* TheInstance();

    //! Static shortcut to MainWindow::getActiveWindow
    static QWidget* GetActiveRenderWindow();

    //! Returns a given GL sub-window (determined by its title)
    /** \param title window title
     **/
    static QWidget* GetRenderWindow(const QString& title);

    //! Returns all GL sub-windows
    /** \param[in,out] glWindows vector to store all sub-windows
     **/
    static void GetRenderWindows(std::vector<QWidget*>& windows);

    //! Deletes current main window instance
    static void DestroyInstance();

    static void ChangeStyle(const QString& qssFile);

public slots:
    // Picking opeations
    void enablePickingOperation(QString message);
    void cancelPreviousPickingOperation(bool aborted);

public:
    void setUiManager(QUIWidget* uiManager);

    //! Saves position and state of all GUI elements
    void saveGUIElementsPos();

    void setAutoPickPivot(bool state);
    void setOrthoView();
    void setPerspectiveView();

    void updateViewModePopUpMenu();

    ccBBox getSelectedEntityBbox();

    void updateFullScreenMenu(bool state);

    void addToDBAuto(const QStringList& filenames, bool displayDialog = true);

    void addToDB(const QStringList& filenames,
                 QString fileFilter = QString(),
                 bool displayDialog = true);

    //! Sets up the UI (menus and toolbars) based on loaded plugins
    void initPlugins();

    //! Updates the 'Properties' view
    void updatePropertiesView();

    void doActionSaveViewportAsCamera();

    //! Inherited from ccPickingListener
    void onItemPicked(const PickedItem& pi) override;

    //! Returns real 'dbRoot' object
    inline ccDBRoot* db() { return m_ccRoot; }

    //! Returns the number of 3D views
    int getRenderWindowCount() const;

    //! Returns MDI area subwindow corresponding to a given 3D view
    QMdiSubWindow* getMDISubWindow(QWidget* win);
    QWidget* getActiveWindow() override;
    QWidget* getWindow(int index) const;
    void update3DViewsMenu();

public:
    //! Flag: first time the window is made visible
    bool m_FirstShow;
    static bool s_autoSaveGuiElementPos;

public:  // inherited from ecvMainAppInterface
    void spawnHistogramDialog(const std::vector<unsigned>& histoValues,
                              double minVal,
                              double maxVal,
                              QString title,
                              QString xAxisLabel) override;
    ccPickingHub* pickingHub() override { return m_pickingHub; }

    void toggleExclusiveFullScreen(bool state) override;
    void toggle3DView(bool state) override;
    void forceConsoleDisplay() override;
    ccHObject* dbRootObject() override;
    // void updateScreen() override;
    void refreshAll(bool only2D = false, bool forceRedraw = true) override;
    void refreshSelected(bool only2D = false, bool forceRedraw = true) override;
    void refreshObject(ccHObject* obj,
                       bool only2D = false,
                       bool forceRedraw = true) override;
    void refreshObjects(ccHObject::Container objs,
                        bool only2D = false,
                        bool forceRedraw = true) override;
    void resetSelectedBBox() override;
    void removeFromDB(ccHObject* obj, bool autoDelete = true) override;
    void setSelectedInDB(ccHObject* obj, bool selected) override;
    void putObjectBackIntoDBTree(ccHObject* obj,
                                 const ccHObjectContext& context) override;
    inline QMainWindow* getMainWindow() override { return this; }
    inline const ccHObject::Container& getSelectedEntities() const override {
        return m_selectedEntities;
    }
    ccHObjectContext removeObjectTemporarilyFromDBTree(ccHObject* obj) override;

    ccColorScalesManager* getColorScalesManager() override;

    void addToDB(ccHObject* obj,
                 bool updateZoom = false,
                 bool autoExpandDBTree = true,
                 bool checkDimensions = false,
                 bool autoRedraw = true) override;

    void registerOverlayDialog(ccOverlayDialog* dlg, Qt::Corner pos) override;
    void unregisterOverlayDialog(ccOverlayDialog* dlg) override;
    void updateOverlayDialogsPlacement() override;
    ccHObject* loadFile(QString filename, bool silent) override;

    // int getDevicePixelRatio() const override;
    void setView(CC_VIEW_ORIENTATION view) override;

    void dispToConsole(
            QString message,
            ConsoleMessageLevel level = STD_CONSOLE_MESSAGE) override;
    ccUniqueIDGenerator::Shared getUniqueIDGenerator() override;

    void addWidgetToQMdiArea(QWidget* widget) override;

    void increasePointSize() override;
    void decreasePointSize() override;
    void updateUI() override;
    void freezeUI(bool state) override;
    void zoomOnSelectedEntities() override;
    void zoomOnEntities(ccHObject* obj) override;
    void setGlobalZoom() override;

private:
    /***** Utils Methods ***/
    void connectActions();
    void initThemes();
    void initLanguages();
    void initApplicationUpdate();
    void initial();
    void initStatusBar();
    void initDBRoot();
    void initConsole();

    //! Adds the "Edit Plane" action to the given menu.
    /**
     * This is the only MainWindow UI action used externally (by ccDBRoot).
     **/
    void addEditPlaneAction(QMenu& menu) const;

    //! Makes the window including an entity zoom on it (helper)
    void zoomOn(ccHObject* object);

    //! Computes the orientation of an entity
    /** Either fit a plane or a 'facet' (2D polygon)
     **/
    void doComputePlaneOrientation(bool fitFacet);
    void doActionComputeMesh(CC_TRIANGULATION_TYPES type);
    //! Creates point clouds from multiple 'components'
    void createComponentsClouds(
            ccGenericPointCloud* cloud,
            cloudViewer::ReferenceCloudContainer& components,
            unsigned minPointPerComponent,
            bool randomColors,
            bool selectComponents,
            bool sortBysize = true);

public slots:
    void doActionPerspectiveProjection();
    void doActionOrthogonalProjection();

private slots:
    // status slots
    void onMousePosChanged(const QPoint& pos);
    // File menu slots
    void doActionOpenFile();
    void doActionSaveFile();
    void changeTheme();
    void changeLanguage();
    void doActionGlobalShiftSeetings();
    void doActionResetGUIElementsPos();
    void doShowPrimitiveFactory();

    void doCheckForUpdate();

    void doActionComputeNormals();
    void doActionInvertNormals();
    void doActionConvertNormalsToHSV();
    void doActionOrientNormalsFM();
    void doActionOrientNormalsMST();
    void doActionExportNormalToSF();
    void doActionConvertNormalsToDipDir();

    void doActionComputeOctree();
    void doActionResampleWithOctree();

    void doBoxAnnotation();
    void doSemanticAnnotation();
    void doAnnotations(int mode);

    //! Creates a cloud with a single point
    void createSinglePointCloud();
    //! Creates a cloud from the clipboard (ASCII) data
    void createPointCloudFromClipboard();

    // sand box research
    void doActionComputeKdTree();
    void doComputeBestFitBB();
    void doActionComputeDistanceMap();
    void doActionComputeDistToBestFitQuadric3D();
    void doAction4pcsRegister();
    void doSphericalNeighbourhoodExtractionTest();
    void doCylindricalNeighbourhoodExtractionTest();
    void doActionCreateCloudFromEntCenters();
    void doActionComputeBestICPRmsMatrix();
    void doActionFindBiggestInnerRectangle();

    //! Toggles the 'show Qt warnings in Console' option
    void doEnableQtWarnings(bool);

    // Edit method
    void doActionComputeMeshAA();
    void doActionComputeMeshLS();
    void doActionConvexHull();
    void doActionPoissonReconstruction();
    void doMeshTwoPolylines();

    void doActionMeshScanGrids();
    void doActionCreateGBLSensor();
    void doActionCreateCameraSensor();
    void doActionModifySensor();
    void doActionProjectUncertainty();
    void doActionCheckPointsInsideFrustum();
    void doActionComputeDistancesFromSensor();
    void doActionComputeScatteringAngles();
    void doActionSetViewFromSensor();
    void doActionShowDepthBuffer();
    void doActionExportDepthBuffer();
    void doActionComputePointsVisibility();

    void doActionShowWaveDialog();
    void doActionCompressFWFData();

    void doActionConvertTextureToColor();
    void doActionSamplePointsOnMesh();
    void doActionSamplePointsOnPolyline();
    void doConvertPolylinesToMesh();
    void doBSplineFittingFromCloud();
    void doActionSmoothMeshSF();
    void doActionEnhanceMeshSF();
    void doActionSubdivideMesh();
    void doActionFlipMeshTriangles();
    void doActionSmoothMeshLaplacian();
    void doActionFlagMeshVertices();
    void doActionMeasureMeshVolume();
    void doActionMeasureMeshSurface();

    void doActionCreatePlane();
    void doActionEditPlane();
    void doActionFlipPlane();
    void doActionComparePlanes();

    //! Clones currently selected entities
    void doActionClone();
    void doActionMerge();

    void activateTracePolylineMode();
    void deactivateTracePolylineMode(bool);

    void popMenuInConsole(const QPoint&);
    void clearConsoleItems();
    void clearConsole();

    // Display method
    //! toggles full screen
    void toggleFullScreen(bool state);
    //! Slot called when the exclusive full screen mode is toggled on a window
    void onExclusiveFullScreenToggled(bool);
    void showDisplayOptions();

    void toggleActiveWindowAutoPickRotCenter(bool state);
    void toggleRotationCenterVisibility(bool state);
    void doActionResetRotCenter();

    void doActionEditCamera();
    void toggleLockRotationAxis();
    void doActionAnimation();
    void doActionScreenShot();
    void doActionToggleOrientationMarker(bool state);

    // About menu slots
    void help();
    void showEvent(QShowEvent* event) override;
    void closeEvent(QCloseEvent* event) override;
    void moveEvent(QMoveEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

    void toggleVisualDebugTraces();

    void updateUIWithSelection();
    void doActionApplyTransformation();
    void doActionApplyScale();
    void activateTranslateRotateMode();
    void deactivateTranslateRotateMode(bool state);

    void updateMenus();
    void on3DViewActivated(QMdiSubWindow*);

    //! Handles new label
    void handleNewLabel(ccHObject*);

    // Point picking mechanism
    void activatePointPickingMode();
    void deactivatePointPickingMode(bool);

    // Point list picking mechanism
    void activatePointListPickingMode();
    void deactivatePointListPickingMode(bool);

    //! Removes all entities currently loaded in the DB tree
    void clearAll();

    // color menu
    void doActionSetUniqueColor();
    void doActionColorize();
    void doActionRGBToGreyScale();
    void doActionSetColor(bool colorize);
    void doActionSetColorGradient();
    void doActionInterpolateColors();
    void doActionChangeColorLevels();
    void doActionEnhanceRGBWithIntensities();
    void doActionColorFromScalars();

    // scalar field menu
    void showSelectedEntitiesHistogram();
    void doActionComputeStatParams();
    void doActionSFGradient();
    void doActionOpenColorScalesManager();
    void doActionSFGaussianFilter();
    void doActionSFBilateralFilter();
    void doActionFilterByLabel();
    void doActionFilterByValue();

    void doActionScalarFieldFromColor();
    void doActionSFConvertToRGB();
    void doActionSFConvertToRandomRGB();
    void doActionRenameSF();
    void doActionAddConstantSF();
    void doActionImportSFFromFile();
    void doActionAddIdField();
    void doActionExportCoordToSF();
    void doActionSetSFAsCoord();
    void doActionInterpolateScalarFields();
    void doActionScalarFieldArithmetic();

    // Current active scalar field
    void doActionToggleActiveSFColorScale();
    void doActionShowActiveSFPrevious();
    void doActionShowActiveSFNext();
    //! Active SF action fork
    /** - action=0 : toggle SF color scale
            - action=1 : activate previous SF
            - action=2 : activate next SF
            \param action action id
    **/
    void doApplyActiveSFAction(int action);

    void doRemoveDuplicatePoints();
    void doActionSubsample();
    void doActionEditGlobalShiftAndScale();

    // Tools -> Registration
    void doActionMatchScales();
    void doActionMatchBBCenters();
    void doActionRegister();
    void activateRegisterPointPairTool();
    void deactivateRegisterPointPairTool(bool state);

    inline void doActionMoveBBCenterToOrigin() {
        doActionFastRegistration(MoveBBCenterToOrigin);
    }
    inline void doActionMoveBBMinCornerToOrigin() {
        doActionFastRegistration(MoveBBMinCornerToOrigin);
    }
    inline void doActionMoveBBMaxCornerToOrigin() {
        doActionFastRegistration(MoveBBMaxCornerToOrigin);
    }

    // Tools -> Recognition
    void doSemanticSegmentation();
    void deactivateSemanticSegmentation(bool);

    // Tools -> Segmentation
    void doActionDBScanCluster();
    void doActionPlaneSegmentation();

    // Tools -> Segmentation
    void activateSegmentationMode();
    void deactivateSegmentationMode(bool);
    void activateFilterWindowMode();
    void deactivateFilterWindowMode(bool);

    void doActionFilterMode(int mode);
    void activateClippingMode();
    void activateSliceMode();
    void activateProbeMode();
    void activateDecimateMode();
    void activateIsoSurfaceMode();
    void activateThresholdMode();
    void activateSmoothMode();
    void activateGlyphMode();
    void activateStreamlineMode();

    void doActionLabelConnectedComponents();
    void doActionKMeans();
    void doActionFrontPropagation();
    void doActionExportPlaneInfo();
    void doActionExportCloudInfo();

    void doActionCloudCloudDist();
    void doActionCloudMeshDist();
    void doActionCloudPrimitiveDist();
    void deactivateComparisonMode(int result);
    void doActionComputeCPS();

    void doActionFitSphere();
    void doActionFitPlane();
    void doActionFitFacet();
    void doActionFitQuadric();

    void doActionSORFilter();
    void doActionFilterNoise();
    void doActionVoxelSampling();

    void doActionUnroll();
    void doComputeGeometricFeature();

private:
    //! Apply transformation to the selected entities
    void applyTransformation(const ccGLMatrixd& transMat);

    //! Enables menu entires based on the current selection
    void enableUIItems(dbTreeSelectionInfo& selInfo);

    /***** Slots of QMenuBar and QToolBar *****/
    void getFileFilltersAndHistory(QStringList& fileFilters,
                                   QString& currentOpenDlgFilter);

    //! Shortcut: asks the user to select one cloud
    /** \param defaultCloudEntity a cloud to select by default (optional)
        \param inviteMessage invite message (default is something like 'Please
    select an entity:') (optional) \return the selected cloud (or null if the
    user cancelled the operation)
    **/
    ccPointCloud* askUserToSelectACloud(ccHObject* defaultCloudEntity = nullptr,
                                        QString inviteMessage = QString());

    void toggleSelectedEntitiesProperty(
            ccEntityAction::TOGGLE_PROPERTY property);
    void clearSelectedEntitiesProperty(ccEntityAction::CLEAR_PROPERTY property);

    enum FastRegistrationMode {
        MoveBBCenterToOrigin,
        MoveBBMinCornerToOrigin,
        MoveBBMaxCornerToOrigin
    };

    void doActionFastRegistration(FastRegistrationMode mode);

private:
    Ui::MainViewerClass* m_ui;

    //! DB & DB Tree
    ccDBRoot* m_ccRoot;

    //! Currently selected entities;
    ccHObject::Container m_selectedEntities;

    //! UI frozen state (see freezeUI)
    bool m_uiFrozen;

    //! Recent files menu
    ecvRecentFiles* m_recentFiles;

    //! View mode pop-up menu button
    QToolButton* m_viewModePopupButton;

    //! Point picking hub
    ccPickingHub* m_pickingHub;

    /******************************/
    /***        MDI AREA        ***/
    /******************************/
    QMdiArea* m_mdiArea;

    //! CloudViewer MDI area overlay dialogs
    struct ccMDIDialogs {
        ccOverlayDialog* dialog;
        Qt::Corner position;

        //! Constructor with dialog and position
        ccMDIDialogs(ccOverlayDialog* dlg, Qt::Corner pos)
            : dialog(dlg), position(pos) {}
    };

    //! Repositions an MDI dialog at its right position
    void repositionOverlayDialog(ccMDIDialogs& mdiDlg);

    //! Registered MDI area 'overlay' dialogs
    std::vector<ccMDIDialogs> m_mdiDialogs;

    /*** dialogs ***/
    //! Application update dialog
    ecvUpdateDlg* m_updateDlg;
    //! Camera params dialog
    ecvCameraParamEditDlg* m_cpeDlg;
    //! Animation params dialog
    ecvAnimationParamDlg* m_animationDlg;
    //! Graphical segmentation dialog
    ccGraphicalSegmentationTool* m_gsTool;
    //! Polyline tracing tool
    ccTracePolylineTool* m_tplTool;
    //! Graphical transformation dialog
    ccGraphicalTransformationTool* m_transTool;
    //! Cloud comparison dialog
    ccComparisonDlg* m_compDlg;
    //! Point properties mode dialog
    ccPointPropertiesDlg* m_ppDlg;
    //! Point list picking
    ccPointListPickingDlg* m_plpDlg;
    //! Point-pair registration
    ccPointPairRegistrationDlg* m_pprDlg;
    //! Primitive factory dialog
    ecvPrimitiveFactoryDlg* m_pfDlg;

    //! filter tool dialog
    ecvFilterTool* m_filterTool;
    //! Annotation tool dialog
    ecvAnnotationsTool* m_annoTool;
    //! Filter Label Tool dialog
    ecvFilterByLabelDlg* m_filterLabelTool;
    //! Filter Window tool
    ecvFilterWindowTool* m_filterWindowTool;
    //! Deep Semantic Segmentation tool dialog
    ecvDeepSemanticSegmentationTool* m_dssTool;

    QVBoxLayout* m_layout;
    QUIWidget* m_uiManager;
    QLabel* m_mousePosLabel;
    QLabel* m_systemInfoLabel;

    // For full screen
    QWidget* m_currentFullWidget;
    //! Wether exclusive full screen is enabled or not
    bool m_exclusiveFullscreen;
    //! Former geometry (for exclusive full-screen display)
    QByteArray m_formerGeometry;

    enum VIEWMODE { PERSPECTIVE, ORTHOGONAL };
    VIEWMODE m_lastViewMode;

    /*** plugins ***/
    //! Manages plugins - menus, toolbars, and the about dialog
    ccPluginUIManager* m_pluginUIManager;

private:
#ifdef BUILD_RECONSTRUCTION
    void initReconstructions();
    void autoShowReconstructionToolBar(bool state);
    cloudViewer::ReconstructionWidget* m_rcw;
#endif

signals:
    //! Signal emitted when the exclusive full screen is toggled
    void exclusiveFullScreenToggled(bool exclusive);

    //! Signal emitted when the selected object is translated by the user
    void translation(const CCVector3d& t);

    //! Signal emitted when the selected object is rotated by the user
    /** \param rotMat rotation applied to current viewport (4x4 OpenGL matrix)
     **/
    void rotation(const ccGLMatrixd& rotMat);
};

#endif  // ECV_MAIN_WINDOW_HEADER

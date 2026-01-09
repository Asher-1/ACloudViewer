// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MainWindow.h"

// Local
#include "ecvLayoutManager.h"
#include "ecvShortcutDialog.h"

// Qt
#include <QGuiApplication>
#include <QScreen>
#include <QSettings>
#include <QThread>
#include <QTimer>

// Standard
#include <algorithm>

#include "ecvAnnotationsTool.h"
#include "ecvApplication.h"
#include "ecvConsole.h"
#include "ecvCropTool.h"
#include "ecvFilterTool.h"
#include "ecvGraphicalSegmentationTool.h"
#include "ecvGraphicalTransformationTool.h"
#include "ecvHistogramWindow.h"
#include "ecvInnerRect2DFinder.h"
#include "ecvLibAlgorithms.h"
#include "ecvMeasurementTool.h"
#include "ecvPersistentSettings.h"
#include "ecvRecentFiles.h"
#include "ecvRegistrationTools.h"
#include "ecvScaleDlg.h"
#include "ecvSettingManager.h"
#include "ecvTracePolylineTool.h"
#include "ecvTranslationManager.h"
#include "ecvUIManager.h"
#include "ecvUtils.h"

// CV_CORE_LIB
#include <CVMath.h>
#include <CVPointCloud.h>
#include <CloudSamplingTools.h>
#include <Delaunay2dMesh.h>
#include <Jacobi.h>
#include <MemoryInfo.h>
#include <MeshSamplingTools.h>
#include <NormalDistribution.h>
#include <ParallelSort.h>
#include <RegistrationTools.h>
#include <ScalarFieldTools.h>
#include <StatisticalTestingTools.h>
#include <WeibullDistribution.h>
#include <ecvVolumeCalcTool.h>

// for tests
#include <ChamferDistanceTransform.h>
#include <SaitoSquaredDistanceTransform.h>

// ECV_DB_LIB
#include <ecv2DLabel.h>
#include <ecv2DViewportObject.h>
#include <ecvCameraSensor.h>
#include <ecvCircle.h>
#include <ecvColorScalesManager.h>
#include <ecvCylinder.h>
#include <ecvDisc.h>
#include <ecvDisplayTools.h>
#include <ecvFacet.h>
#include <ecvFileUtils.h>
#include <ecvGBLSensor.h>
#include <ecvGenericPointCloud.h>
#include <ecvImage.h>
#include <ecvKdTree.h>
#include <ecvPlane.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvProgressDialog.h>
#include <ecvQuadric.h>
#include <ecvRenderingTools.h>
#include <ecvScalarField.h>
#include <ecvSphere.h>
#include <ecvSubMesh.h>

// ECV_IO_LIB
#include <AsciiFilter.h>
#include <BinFilter.h>
#include <DepthMapFileFilter.h>
#include <ecvGlobalShiftManager.h>
#include <ecvShiftAndScaleCloudDlg.h>

// common
#include <CommonSettings.h>
#include <ecvCommon.h>
#include <ecvCustomViewpointsToolbar.h>
#include <ecvOptions.h>
#include <ecvPickingHub.h>

// common dialogs
#include <ecvCameraParamEditDlg.h>
#include <ecvDisplayOptionsDlg.h>
#include <ecvPickOneElementDlg.h>

// Qt UI files
#include "ui_MainWindow.h"
#include "ui_distanceMapDlg.h"
#include "ui_globalShiftSettingsDlg.h"

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

// dialogs
#include "ecvAboutDialog.h"
#include "ecvAlignDlg.h"
#include "ecvAnimationParamDlg.h"
#include "ecvApplyTransformationDlg.h"
#include "ecvAskThreeDoubleValuesDlg.h"
#include "ecvCamSensorProjectionDlg.h"
#include "ecvColorFromScalarDlg.h"
#include "ecvColorScaleEditorDlg.h"
#include "ecvComparisonDlg.h"
#include "ecvEntitySelectionDlg.h"
#include "ecvFilterByLabelDlg.h"
#include "ecvFilterByValueDlg.h"
#include "ecvGBLSensorProjectionDlg.h"
#include "ecvGeomFeaturesDlg.h"
#include "ecvItemSelectionDlg.h"
#include "ecvLabelingDlg.h"
#include "ecvMatchScalesDlg.h"
#include "ecvNoiseFilterDlg.h"
#include "ecvOrderChoiceDlg.h"
#include "ecvPlaneEditDlg.h"
#include "ecvPointListPickingDlg.h"
#include "ecvPointPairRegistrationDlg.h"
#include "ecvPointPropertiesDlg.h"
#include "ecvPoissonReconDlg.h"
#include "ecvPrimitiveDistanceDlg.h"
#include "ecvPrimitiveFactoryDlg.h"
#include "ecvPtsSamplingDlg.h"
#include "ecvRegistrationDlg.h"
#include "ecvRenderToFileDlg.h"
#include "ecvSORFilterDlg.h"
#include "ecvSensorComputeDistancesDlg.h"
#include "ecvSensorComputeScatteringAnglesDlg.h"
#include "ecvShiftAndScaleCloudDlg.h"
#include "ecvSmoothPolylineDlg.h"
#include "ecvSubsamplingDlg.h"
#include "ecvUnrollDlg.h"
#include "ecvUpdateDlg.h"
#include "ecvWaveformDialog.h"

// other
#include "db_tree/ecvDBRoot.h"
#include "db_tree/ecvPropertiesTreeDelegate.h"
#include "pluginManager/ecvPluginUIManager.h"

// 3D mouse handler
#ifdef CC_3DXWARE_SUPPORT
#include "cc3DMouseManager.h"
#endif

// Gamepads
#ifdef CC_GAMEPAD_SUPPORT
#include "ccGamepadManager.h"
#endif

// Reconstruction
#ifdef BUILD_RECONSTRUCTION
#include "reconstruction/ReconstructionWidget.h"
#endif

// QPCL_ENGINE_LIB
#ifdef USE_PCL_BACKEND
#include <PclUtils/PCLDisplayTools.h>
#include <Tools/AnnotationTools/PclAnnotationTool.h>
#include <Tools/CameraTools/EditCameraTool.h>
#include <Tools/Common/CurveFitting.h>
#include <Tools/FilterTools/PclFiltersTool.h>
#include <Tools/MeasurementTools/PclMeasurementTools.h>
#include <Tools/SelectionTools/cvFindDataDockWidget.h>
#include <Tools/SelectionTools/cvSelectionData.h>
#include <Tools/SelectionTools/cvSelectionHighlighter.h>
#include <Tools/SelectionTools/cvSelectionStorage.h>
#include <Tools/SelectionTools/cvSelectionToolController.h>
#include <Tools/SelectionTools/cvViewSelectionManager.h>
#include <Tools/TransformTools/PclTransformTool.h>
#endif

// ECV_PYTHON_LIB
#ifdef USE_PYTHON_MODULE
#include <recognition/PythonInterface.h>

#include "ecvDeepSemanticSegmentationTool.h"
#endif

// SYSTEM
#ifdef CV_WINDOWS
#include <omp.h>
#endif

#ifdef USE_VLD
// VLD
#include <vld.h>
#endif

#ifdef USE_TBB
#include <tbb/tbb_stddef.h>
#endif

// global static pointer (as there should only be one instance of MainWindow!)
static MainWindow* s_instance = nullptr;

// default 'All files' file filter
static const QString s_allFilesFilter("All (*.*)");
// default file filter separator
static const QString s_fileFilterSeparator(";;");

static const float GLOBAL_OPACITY = 0.5;

enum PickingOperation {
    NO_PICKING_OPERATION,
    PICKING_ROTATION_CENTER,
    PICKING_LEVEL_POINTS,
};
static PickingOperation s_currentPickingOperation = NO_PICKING_OPERATION;
static std::vector<cc2DLabel*> s_levelLabels;
static ccPointCloud* s_levelMarkersCloud = nullptr;
static ccHObject* s_levelEntity = nullptr;

static QFileDialog::Options ECVFileDialogOptions() {
    // dialog options
    QFileDialog::Options dialogOptions = QFileDialog::Options();
    dialogOptions |= QFileDialog::DontResolveSymlinks;
    if (!ecvOptions::Instance().useNativeDialogs) {
        dialogOptions |= QFileDialog::DontUseNativeDialog;
    }
    return dialogOptions;
}

// Helper: check for a filename validity
static bool IsValidFileName(QString filename) {
#ifdef CV_WINDOWS
    QString sPattern(
            "^(?!^(PRN|AUX|CLOCK\\$|NUL|CON|COM\\d|LPT\\d|\\..*)(\\..+)?$)[^"
            "\\x00-\\x1f\\\\?*:\\"
            ";|/]+$");
#else
    QString sPattern(
            "^(([a-zA-Z]:|\\\\)\\\\)?(((\\.)|(\\.\\.)|([^\\\\/:\\*\\?"
            "\\|<>\\. ](([^\\\\/:\\*\\?"
            "\\|<>\\. ])|([^\\\\/:\\*\\?"
            "\\|<>]*[^\\\\/:\\*\\?"
            "\\|<>\\. ]))?))\\\\)*[^\\\\/:\\*\\?"
            "\\|<>\\. ](([^\\\\/:\\*\\?"
            "\\|<>\\. ])|([^\\\\/:\\*\\?"
            "\\|<>]*[^\\\\/:\\*\\?"
            "\\|<>\\. ]))?$");
#endif

    // Use QtCompat for Qt5/Qt6 compatibility
    QtCompatRegExp regex(sPattern);
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    QRegularExpressionMatch match = regex.match(filename);
    return match.hasMatch();
#else
    return regex.exactMatch(filename);
#endif
}

MainWindow::MainWindow()
    : m_FirstShow(true),
      m_ui(new Ui::MainViewerClass),
      m_ccRoot(nullptr),
      m_uiFrozen(false),
      m_recentFiles(new ecvRecentFiles(this)),
      m_viewModePopupButton(nullptr),
      m_pickingHub(nullptr),
      m_updateDlg(nullptr),
      m_cpeDlg(nullptr),
      m_animationDlg(nullptr),
      m_gsTool(nullptr),
      m_tplTool(nullptr),
      m_transTool(nullptr),
      m_compDlg(nullptr),
      m_ppDlg(nullptr),
      m_plpDlg(nullptr),
      m_pprDlg(nullptr),
      m_pfDlg(nullptr),
      m_filterTool(nullptr),
      m_annoTool(nullptr),
      m_filterLabelTool(nullptr),
      m_measurementTool(nullptr),
      m_dssTool(nullptr),
#ifdef USE_PCL_BACKEND
      m_selectionController(nullptr),
      m_findDataDock(nullptr),
#endif
      m_layout(nullptr),
      m_uiManager(nullptr),
      m_mousePosLabel(nullptr),
      m_systemInfoLabel(nullptr),
      m_memoryUsageWidget(nullptr),
      m_memoryUsageProgressBar(nullptr),
      m_memoryUsageLabel(nullptr),
      m_memoryUsageTimer(nullptr),
      m_currentFullWidget(nullptr),
      m_exclusiveFullscreen(false),
      m_lastViewMode(VIEWMODE::ORTHOGONAL),
      m_shortcutDlg(nullptr)
#ifdef BUILD_RECONSTRUCTION
      ,
      m_rcw(nullptr)
#endif
{
    m_ui->setupUi(this);

    setWindowTitle(QStringLiteral("ACloudViewer v") +
                   ecvApp->versionLongStr(false));

    m_pluginUIManager = new ccPluginUIManager(this, this);

    // Create layout manager (after m_pluginUIManager is created)
    m_layoutManager = new ecvLayoutManager(this, m_pluginUIManager);

    ccTranslationManager::get().populateMenu(m_ui->langAction,
                                             ecvApp->translationPath());

#ifdef Q_OS_MAC
    m_ui->actionAbout->setMenuRole(QAction::AboutRole);
    m_ui->actionAboutPlugins->setMenuRole(QAction::ApplicationSpecificRole);

    m_ui->actionFullScreen->setText(tr("Enter Full Screen"));
    m_ui->actionFullScreen->setShortcut(
            QKeySequence(Qt::CTRL + Qt::META + Qt::Key_F));
#endif

    // Initialization
    initial();

    // restore the state of the 'auto-restore' menu entry
    // (do that before connecting the actions)
    {
        QSettings settings;
        bool doNotAutoRestoreGeometry =
                settings.value(ecvPS::DoNotRestoreWindowGeometry(),
                               !m_ui->actionRestoreWindowOnStartup->isChecked())
                        .toBool();
        m_ui->actionRestoreWindowOnStartup->setChecked(
                !doNotAutoRestoreGeometry);
    }

    // connect actions
    connectActions();

    setupInputDevices();

    freezeUI(false);

    updateUI();

    // Register ViewToolBar as a left-side toolbar
    m_layoutManager->registerLeftSideToolBar(m_ui->ViewToolBar);

    // Register Console dock widget as a bottom dock widget
    m_layoutManager->registerBottomDockWidget(m_ui->consoleDock);

    // Create Find Data dock widget (ParaView-style selection properties panel)
    // This dock is independent of selection tool state and can be shown/hidden
    // by the user
#ifdef USE_PCL_BACKEND
    {
        m_findDataDock = new cvFindDataDockWidget(this);
        // Add dock to the right side (like ParaView)
        addDockWidget(Qt::RightDockWidgetArea, m_findDataDock);

        // Hide by default (user can show it via View menu)
        m_findDataDock->hide();

        // Connect extractedObjectReady to add extracted objects to scene
        connect(m_findDataDock, &cvFindDataDockWidget::extractedObjectReady,
                this, [this](ccHObject* obj) {
                    if (obj) {
                        CVLog::Print(QString("[Extract] Adding extracted "
                                             "object '%1' "
                                             "to database")
                                             .arg(obj->getName()));
                        addToDB(obj, true, true, false);
                    }
                });

        // Register with layout manager for proper default layout handling
        m_layoutManager->registerRightSideDockWidget(m_findDataDock);

        // Add Find Data dock's toggle action to Toolbars menu
        // Using QDockWidget's built-in toggleViewAction for standard behavior
        QAction* selectionPropsAction = m_findDataDock->toggleViewAction();
        selectionPropsAction->setText(tr("Find Data (Selection)"));
        m_ui->menuToolbars->addAction(selectionPropsAction);

        // Store action for external access (e.g., sync with selection tool
        // state)
        m_selectionPropsAction = selectionPropsAction;
    }

    // Initialize selection controller AFTER m_findDataDock is created
    // This ensures configure() can properly set up the dock widget
    initSelectionController();
#else
    CVLog::Warning(
            "[MainWindow] USE_PCL_BACKEND not defined - Find Data dock not "
            "created");
#endif

    // advanced widgets not handled by QDesigner
    {  // view mode pop-up menu
        m_viewModePopupButton = new QToolButton();
        QMenu* menu = new QMenu(m_viewModePopupButton);
        menu->addAction(m_ui->actionOrthogonalProjection);
        menu->addAction(m_ui->actionPerspectiveProjection);

        m_viewModePopupButton->setMenu(menu);
        m_viewModePopupButton->setPopupMode(QToolButton::InstantPopup);
        m_viewModePopupButton->setToolTip("Set current view mode");
        m_viewModePopupButton->setStatusTip(m_viewModePopupButton->toolTip());
        m_ui->ViewToolBar->insertWidget(m_ui->actionZoomToBox,
                                        m_viewModePopupButton);
        m_viewModePopupButton->setEnabled(false);
    }

    {  // custom viewports configuration
        QToolBar* customViewpointsToolbar =
                new ecvCustomViewpointsToolbar(this);
        customViewpointsToolbar->setObjectName("customViewpointsToolbar");
        customViewpointsToolbar->layout()->setSpacing(0);
        this->addToolBar(Qt::TopToolBarArea, customViewpointsToolbar);
    }

    {  // orthogonal projection mode (default)
        m_ui->actionOrthogonalProjection->trigger();
        ecvConsole::Print("Perspective off!");
    }

    {  // restore options
        QSettings settings;

        // auto pick center
        bool autoPickRotationCenter =
                settings.value(ecvPS::AutoPickRotationCenter(), false).toBool();
        if (autoPickRotationCenter) {
            m_ui->actionAutoPickPivot->toggle();
        }

        // show center
        bool autoShowCenterAxis =
                settings.value(ecvPS::AutoShowCenter(), true).toBool();
        m_ui->actionShowPivot->blockSignals(true);
        m_ui->actionShowPivot->setChecked(autoShowCenterAxis);
        m_ui->actionShowPivot->blockSignals(false);
        toggleRotationCenterVisibility(autoShowCenterAxis);
    }

    // Shortcut management
    {
        populateActionList();
        // Alphabetical sort
        std::sort(m_actions.begin(), m_actions.end(),
                  [](const QAction* a, const QAction* b) {
                      return a->text() < b->text();
                  });

        m_shortcutDlg = new ecvShortcutDialog(m_actions, this);
        m_shortcutDlg->restoreShortcutsFromQSettings();

        connect(m_ui->actionShortcutSettings, &QAction::triggered, this,
                &MainWindow::showShortcutDialog);
    }

    refreshAll();

#ifdef CV_WINDOWS
#ifdef QT_DEBUG
    // speed up debugging on windows
    doActionToggleOrientationMarker(false);
#else
    doActionToggleOrientationMarker(true);
#endif
#else
    doActionToggleOrientationMarker(true);
#endif

#ifdef USE_PYTHON_MODULE
// QString applicationPath = QCoreApplication::applicationDirPath();
// QString pyHome = applicationPath + "/python38";
// if (!PythonInterface::SetPythonHome(CVTools::FromQString(pyHome).c_str())) {
//     CVLog::Warning(QString("Setting python home failed! Invalid path: [%1].")
//                            .arg(pyHome));
// } else {
//     CVLog::Print(QString("Setting python home [%1]
//     successfully!").arg(pyHome));
// }
#else
    m_ui->actionSemanticSegmentation->setEnabled(false);
#endif

    // Apply unified icon size and style to all toolbars created in constructor
    // This handles UI toolbars, customViewpointsToolbar, and reconstruction
    // toolbars
    updateAllToolbarIconSizes();

#ifdef USE_TBB
    ecvConsole::Print(tr("[TBB] Using Intel's Threading Building Blocks %1.%2")
                              .arg(QString::number(TBB_VERSION_MAJOR),
                                   QString::number(TBB_VERSION_MINOR)));
#endif

    // print welcome message
    ecvConsole::Print(
            tr("[ACloudViewer Software start], Welcome to use ACloudViewer"));
}

MainWindow::~MainWindow() {
    destroyInputDevices();

    cancelPreviousPickingOperation(false);  // just in case

    // Reconstruction must before m_ccRoot
#ifdef BUILD_RECONSTRUCTION
    if (m_rcw) {
        m_rcw->release();
    }
    m_rcw = nullptr;
#endif

    assert(m_ccRoot && m_mdiArea);
    m_ccRoot->unloadAll();
    m_ccRoot->disconnect();
    m_mdiArea->disconnect();

    // we don't want any other dialog/function to use the following structures
    ccDBRoot* ccRoot = m_ccRoot;
    m_ccRoot = nullptr;

    m_updateDlg = nullptr;
    m_cpeDlg = nullptr;
    m_animationDlg = nullptr;
    m_mousePosLabel = nullptr;
    m_systemInfoLabel = nullptr;
    m_memoryUsageWidget = nullptr;
    m_memoryUsageProgressBar = nullptr;
    m_memoryUsageLabel = nullptr;
    m_memoryUsageTimer = nullptr;

    m_measurementTool = nullptr;
    m_gsTool = nullptr;
    m_transTool = nullptr;
    m_filterTool = nullptr;
    m_annoTool = nullptr;
    m_dssTool = nullptr;
    m_filterLabelTool = nullptr;

    // Selection tools are now managed by cvSelectionToolController
    // The controller is a singleton and handles its own cleanup

    m_compDlg = nullptr;
    m_ppDlg = nullptr;
    m_plpDlg = nullptr;
    m_pprDlg = nullptr;
    m_pfDlg = nullptr;

    // release all 'overlay' dialogs
    while (!m_mdiDialogs.empty()) {
        ccMDIDialogs mdiDialog = m_mdiDialogs.back();
        m_mdiDialogs.pop_back();

        mdiDialog.dialog->disconnect();
        mdiDialog.dialog->stop(false);
        mdiDialog.dialog->setParent(nullptr);
        delete mdiDialog.dialog;
    }

    // m_mdiDialogs.clear();
    std::vector<QWidget*> windows;
    GetRenderWindows(windows);
    for (QWidget* window : windows) {
        m_mdiArea->removeSubWindow(window);
    }

    ecvDisplayTools::SetSceneDB(nullptr);
    ecvDisplayTools::ReleaseInstance();

    if (ccRoot) {
        delete ccRoot;
        ccRoot = nullptr;
    }

    delete m_ui;
    m_ui = nullptr;

    // if we flush the console, it will try to display the console window while
    // we are destroying everything!
    ecvConsole::ReleaseInstance(false);
}

QMenu* MainWindow::createPopupMenu() {
    // Call the base class implementation to get the standard toolbar actions
    QMenu* menu = QMainWindow::createPopupMenu();

    // Add separator and custom actions
    if (menu && m_selectionPropsAction) {
        menu->addSeparator();
        menu->addAction(m_selectionPropsAction);
    }

    return menu;
}

// MainWindow Initialization
void MainWindow::initial() {
    // MDI Area
    {
        m_mdiArea = new QMdiArea(this);
        setCentralWidget(m_mdiArea);
        connect(m_mdiArea, &QMdiArea::subWindowActivated, this,
                &MainWindow::updateMenus);
        connect(m_mdiArea, &QMdiArea::subWindowActivated, this,
                &MainWindow::on3DViewActivated);
        m_mdiArea->installEventFilter(this);
    }

    bool stereoMode = QSurfaceFormat::defaultFormat().stereo();
    ecvDisplayTools::Init(new PCLDisplayTools(), this, stereoMode);

    // init themes
    initThemes();

    // init languages
    initLanguages();

    // init console
    initConsole();

    // init db root
    initDBRoot();

    // init status bar
    initStatusBar();

// Reconstruction
#ifdef BUILD_RECONSTRUCTION
    initReconstructions();
#endif

    QWidget* viewWidget = ecvDisplayTools::GetMainScreen();
    viewWidget->setMinimumSize(400, 300);
    m_mdiArea->addSubWindow(viewWidget);

    // Install event filter on the VTK render widget to capture ESC key
    // VTK render window doesn't pass key events to Qt by default
    viewWidget->installEventFilter(this);

    // picking hub
    {
        m_pickingHub = new ccPickingHub(this, this);
        connect(m_mdiArea, &QMdiArea::subWindowActivated, m_pickingHub,
                &ccPickingHub::onActiveWindowChanged);
    }

    if (m_pickingHub) {
        // we must notify the picking hub as well if the window is destroyed
        connect(this, &QObject::destroyed, m_pickingHub,
                &ccPickingHub::onActiveWindowDeleted);
    }

    viewWidget->showMaximized();
    viewWidget->update();
}

void MainWindow::updateAllToolbarIconSizes() {
    // Apply unified icon size and style to all existing toolbars
    // This ensures consistency across all toolbars
    if (!m_layoutManager) return;

    QScreen* screen = QGuiApplication::primaryScreen();
    int screenWidth = screen ? screen->geometry().width() : 1920;
    QList<QToolBar*> allToolbars = findChildren<QToolBar*>();

    for (QToolBar* toolbar : allToolbars) {
        if (toolbar && toolbar->parent() == this) {
            m_layoutManager->setToolbarIconSize(toolbar, screenWidth);
        }
    }
}

void MainWindow::initConsole() {
    // set Console
    ecvConsole::Init(m_ui->consoleWidget, this, this);
    m_ui->actionEnableQtWarnings->setChecked(ecvConsole::QtMessagesEnabled());
}

void MainWindow::connectActions() {
    assert(m_ccRoot);
    /***** Slots connection of QMenuBar and QToolBar *****/
    //"File" menu
    connect(m_ui->actionOpen, &QAction::triggered, this,
            &MainWindow::doActionOpenFile);
    connect(m_ui->actionSave, &QAction::triggered, this,
            &MainWindow::doActionSaveFile);
    connect(m_ui->actionSaveProject, &QAction::triggered, this,
            &MainWindow::doActionSaveProject);
    connect(m_ui->actionPrimitiveFactory, &QAction::triggered, this,
            &MainWindow::doShowPrimitiveFactory);
    connect(m_ui->actionClearAll, &QAction::triggered, this,
            &MainWindow::clearAll);
    connect(m_ui->actionExit, &QAction::triggered, this, &QWidget::close);

    //"Edit > Colors" menu
    connect(m_ui->actionSetUniqueColor, &QAction::triggered, this,
            &MainWindow::doActionSetUniqueColor);
    connect(m_ui->actionSetColorGradient, &QAction::triggered, this,
            &MainWindow::doActionSetColorGradient);
    connect(m_ui->actionChangeColorLevels, &QAction::triggered, this,
            &MainWindow::doActionChangeColorLevels);
    connect(m_ui->actionColorize, &QAction::triggered, this,
            &MainWindow::doActionColorize);
    connect(m_ui->actionRGBToGreyScale, &QAction::triggered, this,
            &MainWindow::doActionRGBToGreyScale);
    connect(m_ui->actionInterpolateColors, &QAction::triggered, this,
            &MainWindow::doActionInterpolateColors);
    connect(m_ui->actionEnhanceRGBWithIntensities, &QAction::triggered, this,
            &MainWindow::doActionEnhanceRGBWithIntensities);
    connect(m_ui->actionColorFromScalarField, &QAction::triggered, this,
            &MainWindow::doActionColorFromScalars);
    connect(m_ui->actionRGBGaussianFilter, &QAction::triggered, this,
            &MainWindow::doActionRGBGaussianFilter);
    connect(m_ui->actionRGBBilateralFilter, &QAction::triggered, this,
            &MainWindow::doActionRGBBilateralFilter);
    connect(m_ui->actionRGBMeanFilter, &QAction::triggered, this,
            &MainWindow::doActionRGBMeanFilter);
    connect(m_ui->actionRGBMedianFilter, &QAction::triggered, this,
            &MainWindow::doActionRGBMedianFilter);
    connect(m_ui->actionClearColor, &QAction::triggered, this, [=]() {
        clearSelectedEntitiesProperty(ccEntityAction::CLEAR_PROPERTY::COLORS);
    });

    //"Edit > Clean" menu
    connect(m_ui->actionSORFilter, &QAction::triggered, this,
            &MainWindow::doActionSORFilter);
    connect(m_ui->actionNoiseFilter, &QAction::triggered, this,
            &MainWindow::doActionFilterNoise);
    connect(m_ui->actionVoxelSampling, &QAction::triggered, this,
            &MainWindow::doActionVoxelSampling);

    //"Edit" menu
    connect(m_ui->actionSegment, &QAction::triggered, this,
            &MainWindow::activateSegmentationMode);
    connect(m_ui->actionRemoveDuplicatePoints, &QAction::triggered, this,
            &MainWindow::doRemoveDuplicatePoints);
    connect(m_ui->actionSubsample, &QAction::triggered, this,
            &MainWindow::doActionSubsample);
    connect(m_ui->actionEditGlobalShiftAndScale, &QAction::triggered, this,
            &MainWindow::doActionEditGlobalShiftAndScale);
    connect(m_ui->actionClone, &QAction::triggered, this,
            &MainWindow::doActionClone);
    connect(m_ui->actionMerge, &QAction::triggered, this,
            &MainWindow::doActionMerge);
    connect(m_ui->actionTracePolyline, &QAction::triggered, this,
            &MainWindow::activateTracePolylineMode);
    connect(m_ui->actionDelete, &QAction::triggered, m_ccRoot,
            &ccDBRoot::deleteSelectedEntities);
    connect(m_ui->actionApplyTransformation, &QAction::triggered, this,
            &MainWindow::doActionApplyTransformation);
    connect(m_ui->actionApplyScale, &QAction::triggered, this,
            &MainWindow::doActionApplyScale);
    connect(m_ui->actionTranslateRotate, &QAction::triggered, this,
            &MainWindow::activateTranslateRotateMode);

    // "Edit > Octree" menu
    connect(m_ui->actionComputeOctree, &QAction::triggered, this,
            &MainWindow::doActionComputeOctree);
    connect(m_ui->actionResampleWithOctree, &QAction::triggered, this,
            &MainWindow::doActionResampleWithOctree);

    //"Edit > Cloud" menu
    connect(m_ui->actionCreateSinglePointCloud, &QAction::triggered, this,
            &MainWindow::createSinglePointCloud);
    connect(m_ui->actionPasteCloudFromClipboard, &QAction::triggered, this,
            &MainWindow::createPointCloudFromClipboard);
    // the 'Paste from clipboard' tool depends on the clipboard state
    {
        const QClipboard* clipboard = QApplication::clipboard();
        assert(clipboard);
        m_ui->actionPasteCloudFromClipboard->setEnabled(
                clipboard->mimeData()->hasText());
        connect(clipboard, &QClipboard::dataChanged, [&]() {
            m_ui->actionPasteCloudFromClipboard->setEnabled(
                    clipboard->mimeData()->hasText());
        });
    }

    // "Edit > Normals" menu
    connect(m_ui->actionComputeNormals, &QAction::triggered, this,
            &MainWindow::doActionComputeNormals);
    connect(m_ui->actionInvertNormals, &QAction::triggered, this,
            &MainWindow::doActionInvertNormals);
    connect(m_ui->actionConvertNormalToHSV, &QAction::triggered, this,
            &MainWindow::doActionConvertNormalsToHSV);
    connect(m_ui->actionConvertNormalToDipDir, &QAction::triggered, this,
            &MainWindow::doActionConvertNormalsToDipDir);
    connect(m_ui->actionExportNormalToSF, &QAction::triggered, this,
            &MainWindow::doActionExportNormalToSF);
    connect(m_ui->actionOrientNormalsMST, &QAction::triggered, this,
            &MainWindow::doActionOrientNormalsMST);
    connect(m_ui->actionOrientNormalsFM, &QAction::triggered, this,
            &MainWindow::doActionOrientNormalsFM);
    connect(m_ui->actionClearNormals, &QAction::triggered, this, [=]() {
        clearSelectedEntitiesProperty(ccEntityAction::CLEAR_PROPERTY::NORMALS);
    });

    // "Edit > Mesh" menu
    connect(m_ui->actionComputeMeshAA, &QAction::triggered, this,
            &MainWindow::doActionComputeMeshAA);
    connect(m_ui->actionComputeMeshLS, &QAction::triggered, this,
            &MainWindow::doActionComputeMeshLS);
    connect(m_ui->actionConvexHull, &QAction::triggered, this,
            &MainWindow::doActionConvexHull);
    connect(m_ui->actionPoissonReconstruction, &QAction::triggered, this,
            &MainWindow::doActionPoissonReconstruction);
    connect(m_ui->actionMeshTwoPolylines, &QAction::triggered, this,
            &MainWindow::doMeshTwoPolylines);
    connect(m_ui->actionMeshScanGrids, &QAction::triggered, this,
            &MainWindow::doActionMeshScanGrids);
    connect(m_ui->actionConvertTextureToColor, &QAction::triggered, this,
            &MainWindow::doActionConvertTextureToColor);
    connect(m_ui->actionSamplePointsOnMesh, &QAction::triggered, this,
            &MainWindow::doActionSamplePointsOnMesh);
    connect(m_ui->actionSmoothMeshLaplacian, &QAction::triggered, this,
            &MainWindow::doActionSmoothMeshLaplacian);
    connect(m_ui->actionFlipMeshTriangles, &QAction::triggered, this,
            &MainWindow::doActionFlipMeshTriangles);
    connect(m_ui->actionSubdivideMesh, &QAction::triggered, this,
            &MainWindow::doActionSubdivideMesh);
    connect(m_ui->actionMeasureMeshSurface, &QAction::triggered, this,
            &MainWindow::doActionMeasureMeshSurface);
    connect(m_ui->actionMeasureMeshVolume, &QAction::triggered, this,
            &MainWindow::doActionMeasureMeshVolume);
    connect(m_ui->actionFlagMeshVertices, &QAction::triggered, this,
            &MainWindow::doActionFlagMeshVertices);
    // "Edit > Mesh > Scalar Field" menu
    connect(m_ui->actionSmoothMeshSF, &QAction::triggered, this,
            &MainWindow::doActionSmoothMeshSF);
    connect(m_ui->actionEnhanceMeshSF, &QAction::triggered, this,
            &MainWindow::doActionEnhanceMeshSF);
    // "Edit > Polyline" menu
    connect(m_ui->actionSamplePointsOnPolyline, &QAction::triggered, this,
            &MainWindow::doActionSamplePointsOnPolyline);
    connect(m_ui->actionSmoothPolyline, &QAction::triggered, this,
            &MainWindow::doActionSmoohPolyline);
    connect(m_ui->actionConvertPolylinesToMesh, &QAction::triggered, this,
            &MainWindow::doConvertPolylinesToMesh);
    connect(m_ui->actionBSplineFittingOnCloud, &QAction::triggered, this,
            &MainWindow::doBSplineFittingFromCloud);
    // "Edit > Plane" menu
    connect(m_ui->actionCreatePlane, &QAction::triggered, this,
            &MainWindow::doActionCreatePlane);
    connect(m_ui->actionEditPlane, &QAction::triggered, this,
            &MainWindow::doActionEditPlane);
    connect(m_ui->actionFlipPlane, &QAction::triggered, this,
            &MainWindow::doActionFlipPlane);
    connect(m_ui->actionComparePlanes, &QAction::triggered, this,
            &MainWindow::doActionComparePlanes);

    //"Edit > Circle" menu
    connect(m_ui->actionPromoteCircleToCylinder, &QAction::triggered, this,
            &MainWindow::doActionPromoteCircleToCylinder);

    //"Edit > Scalar fields" menu
    connect(m_ui->actionShowHistogram, &QAction::triggered, this,
            &MainWindow::showSelectedEntitiesHistogram);
    connect(m_ui->actionComputeStatParams, &QAction::triggered, this,
            &MainWindow::doActionComputeStatParams);
    connect(m_ui->actionSFGradient, &QAction::triggered, this,
            &MainWindow::doActionSFGradient);
    connect(m_ui->actionOpenColorScalesManager, &QAction::triggered, this,
            &MainWindow::doActionOpenColorScalesManager);
    connect(m_ui->actionGaussianFilter, &QAction::triggered, this,
            &MainWindow::doActionSFGaussianFilter);
    connect(m_ui->actionBilateralFilter, &QAction::triggered, this,
            &MainWindow::doActionSFBilateralFilter);
    connect(m_ui->actionFilterByLabel, &QAction::triggered, this,
            &MainWindow::doActionFilterByLabel);
    connect(m_ui->actionFilterByValue, &QAction::triggered, this,
            &MainWindow::doActionFilterByValue);
    connect(m_ui->actionScalarFieldFromColor, &QAction::triggered, this,
            &MainWindow::doActionScalarFieldFromColor);
    connect(m_ui->actionConvertToRGB, &QAction::triggered, this,
            &MainWindow::doActionSFConvertToRGB);
    connect(m_ui->actionConvertToRandomRGB, &QAction::triggered, this,
            &MainWindow::doActionSFConvertToRandomRGB);
    connect(m_ui->actionRenameSF, &QAction::triggered, this,
            &MainWindow::doActionRenameSF);
    connect(m_ui->actionAddConstantSF, &QAction::triggered, this,
            &MainWindow::doActionAddConstantSF);
    connect(m_ui->actionImportSFFromFile, &QAction::triggered, this,
            &MainWindow::doActionImportSFFromFile);
    connect(m_ui->actionAddIdField, &QAction::triggered, this,
            &MainWindow::doActionAddIdField);
    connect(m_ui->actionExportCoordToSF, &QAction::triggered, this,
            &MainWindow::doActionExportCoordToSF);
    connect(m_ui->actionSetSFAsCoord, &QAction::triggered, this,
            &MainWindow::doActionSetSFAsCoord);
    connect(m_ui->actionInterpolateSFs, &QAction::triggered, this,
            &MainWindow::doActionInterpolateScalarFields);
    connect(m_ui->actionScalarFieldArithmetic, &QAction::triggered, this,
            &MainWindow::doActionScalarFieldArithmetic);
    connect(m_ui->actionDeleteScalarField, &QAction::triggered, this, [=]() {
        clearSelectedEntitiesProperty(
                ccEntityAction::CLEAR_PROPERTY::CURRENT_SCALAR_FIELD);
    });
    connect(m_ui->actionDeleteAllSF, &QAction::triggered, this, [=]() {
        clearSelectedEntitiesProperty(
                ccEntityAction::CLEAR_PROPERTY::ALL_SCALAR_FIELDS);
    });

    //"Edit > Sensor > Ground-Based lidar" menu
    connect(m_ui->actionShowDepthBuffer, &QAction::triggered, this,
            &MainWindow::doActionShowDepthBuffer);
    connect(m_ui->actionExportDepthBuffer, &QAction::triggered, this,
            &MainWindow::doActionExportDepthBuffer);
    connect(m_ui->actionComputePointsVisibility, &QAction::triggered, this,
            &MainWindow::doActionComputePointsVisibility);

    //"Edit > Sensor" menu
    connect(m_ui->actionModifySensor, &QAction::triggered, this,
            &MainWindow::doActionModifySensor);
    connect(m_ui->actionCreateGBLSensor, &QAction::triggered, this,
            &MainWindow::doActionCreateGBLSensor);
    connect(m_ui->actionCreateCameraSensor, &QAction::triggered, this,
            &MainWindow::doActionCreateCameraSensor);
    connect(m_ui->actionProjectUncertainty, &QAction::triggered, this,
            &MainWindow::doActionProjectUncertainty);
    connect(m_ui->actionCheckPointsInsideFrustum, &QAction::triggered, this,
            &MainWindow::doActionCheckPointsInsideFrustum);
    connect(m_ui->actionComputeDistancesFromSensor, &QAction::triggered, this,
            &MainWindow::doActionComputeDistancesFromSensor);
    connect(m_ui->actionComputeScatteringAngles, &QAction::triggered, this,
            &MainWindow::doActionComputeScatteringAngles);
    connect(m_ui->actionViewFromSensor, &QAction::triggered, this,
            &MainWindow::doActionSetViewFromSensor);

    //"Edit > Waveform" menu
    connect(m_ui->actionShowWaveDialog, &QAction::triggered, this,
            &MainWindow::doActionShowWaveDialog);
    connect(m_ui->actionCompressFWFData, &QAction::triggered, this,
            &MainWindow::doActionCompressFWFData);

    //"Tools > Filter" menu
    connect(m_ui->actionClipFilter, &QAction::triggered, this,
            &MainWindow::activateClippingMode);
    connect(m_ui->actionSliceFilter, &QAction::triggered, this,
            &MainWindow::activateSliceMode);
    connect(m_ui->actionProbeFilter, &QAction::triggered, this,
            &MainWindow::activateProbeMode);
    connect(m_ui->actionDecimateFilter, &QAction::triggered, this,
            &MainWindow::activateDecimateMode);
    connect(m_ui->actionIsoSurfaceFilter, &QAction::triggered, this,
            &MainWindow::activateIsoSurfaceMode);
    connect(m_ui->actionThresholdFilter, &QAction::triggered, this,
            &MainWindow::activateThresholdMode);
    connect(m_ui->actionSmoothFilter, &QAction::triggered, this,
            &MainWindow::activateSmoothMode);
    connect(m_ui->actionStreamlineFilter, &QAction::triggered, this,
            &MainWindow::activateStreamlineMode);
    connect(m_ui->actionGlyphFilter, &QAction::triggered, this,
            &MainWindow::activateGlyphMode);

    // "Tools > Measurements" menu
    connect(m_ui->actionDistanceWidget, &QAction::triggered, this,
            &MainWindow::activateDistanceMode);
    connect(m_ui->actionProtractorWidget, &QAction::triggered, this,
            &MainWindow::activateProtractorMode);
    connect(m_ui->actionContourWidget, &QAction::triggered, this,
            &MainWindow::activateContourMode);

    // "Tools > Distances" menu
    connect(m_ui->actionCloudCloudDist, &QAction::triggered, this,
            &MainWindow::doActionCloudCloudDist);
    connect(m_ui->actionCloudMeshDist, &QAction::triggered, this,
            &MainWindow::doActionCloudMeshDist);
    connect(m_ui->actionCloudPrimitiveDist, &QAction::triggered, this,
            &MainWindow::doActionCloudPrimitiveDist);
    connect(m_ui->actionCPS, &QAction::triggered, this,
            &MainWindow::doActionComputeCPS);

    // "Tools > Annotations" menu
    connect(m_ui->actionBoxAnnotation, &QAction::triggered, this,
            &MainWindow::doBoxAnnotation);
    connect(m_ui->actionSemanticAnnotation, &QAction::triggered, this,
            &MainWindow::doSemanticAnnotation);

    //"Tools > Recognition" menu
    connect(m_ui->actionSemanticSegmentation, &QAction::triggered, this,
            &MainWindow::doSemanticSegmentation);

    //"Tools > Segmentation" menu
    connect(m_ui->actionDBScanCluster, &QAction::triggered, this,
            &MainWindow::doActionDBScanCluster);
    connect(m_ui->actionPlaneSegmentation, &QAction::triggered, this,
            &MainWindow::doActionPlaneSegmentation);

    //"Tools > Registration" menu
    connect(m_ui->actionMatchBBCenters, &QAction::triggered, this,
            &MainWindow::doActionMatchBBCenters);
    connect(m_ui->actionMatchScales, &QAction::triggered, this,
            &MainWindow::doActionMatchScales);
    connect(m_ui->actionRegister, &QAction::triggered, this,
            &MainWindow::doActionRegister);
    connect(m_ui->actionPointPairsAlign, &QAction::triggered, this,
            &MainWindow::activateRegisterPointPairTool);
    connect(m_ui->actionBBCenterToOrigin, &QAction::triggered, this,
            &MainWindow::doActionMoveBBCenterToOrigin);
    connect(m_ui->actionBBMinCornerToOrigin, &QAction::triggered, this,
            &MainWindow::doActionMoveBBMinCornerToOrigin);
    connect(m_ui->actionBBMaxCornerToOrigin, &QAction::triggered, this,
            &MainWindow::doActionMoveBBMaxCornerToOrigin);

    // "Tools > Fit" menu
    connect(m_ui->actionFitPlane, &QAction::triggered, this,
            &MainWindow::doActionFitPlane);
    connect(m_ui->actionFitSphere, &QAction::triggered, this,
            &MainWindow::doActionFitSphere);
    connect(m_ui->actionFitCircle, &QAction::triggered, this,
            &MainWindow::doActionFitCircle);
    connect(m_ui->actionFitFacet, &QAction::triggered, this,
            &MainWindow::doActionFitFacet);
    connect(m_ui->actionFitQuadric, &QAction::triggered, this,
            &MainWindow::doActionFitQuadric);

    // "Tools > Batch export" menu
    connect(m_ui->actionExportCloudInfo, &QAction::triggered, this,
            &MainWindow::doActionExportCloudInfo);
    connect(m_ui->actionExportPlaneInfo, &QAction::triggered, this,
            &MainWindow::doActionExportPlaneInfo);

    connect(m_ui->actionLabelConnectedComponents, &QAction::triggered, this,
            &MainWindow::doActionLabelConnectedComponents);
    connect(m_ui->actionComputeGeometricFeature, &QAction::triggered, this,
            &MainWindow::doComputeGeometricFeature);
    connect(m_ui->actionUnroll, &QAction::triggered, this,
            &MainWindow::doActionUnroll);
    connect(m_ui->actionPointListPicking, &QAction::triggered, this,
            &MainWindow::activatePointListPickingMode);
    connect(m_ui->actionPointPicking, &QAction::triggered, this,
            &MainWindow::activatePointPickingMode);

    //"Tools > Sand box (research)" menu
    connect(m_ui->actionComputeKdTree, &QAction::triggered, this,
            &MainWindow::doActionComputeKdTree);
    connect(m_ui->actionDistanceMap, &QAction::triggered, this,
            &MainWindow::doActionComputeDistanceMap);
    connect(m_ui->actionDistanceToBestFitQuadric3D, &QAction::triggered, this,
            &MainWindow::doActionComputeDistToBestFitQuadric3D);
    connect(m_ui->actionComputeBestFitBB, &QAction::triggered, this,
            &MainWindow::doComputeBestFitBB);
    connect(m_ui->actionAlign, &QAction::triggered, this,
            &MainWindow::doAction4pcsRegister);  // Aurelien BEY le 13/11/2008
    connect(m_ui->actionSNETest, &QAction::triggered, this,
            &MainWindow::doSphericalNeighbourhoodExtractionTest);
    connect(m_ui->actionCNETest, &QAction::triggered, this,
            &MainWindow::doCylindricalNeighbourhoodExtractionTest);
    connect(m_ui->actionFindBiggestInnerRectangle, &QAction::triggered, this,
            &MainWindow::doActionFindBiggestInnerRectangle);
    connect(m_ui->actionCreateCloudFromEntCenters, &QAction::triggered, this,
            &MainWindow::doActionCreateCloudFromEntCenters);
    connect(m_ui->actionComputeBestICPRmsMatrix, &QAction::triggered, this,
            &MainWindow::doActionComputeBestICPRmsMatrix);

    // Display (connect)
    connect(m_ui->actionFullScreen, &QAction::toggled, this,
            &MainWindow::toggleFullScreen);
    connect(m_ui->actionExclusiveFullScreen, &QAction::toggled, this,
            &MainWindow::toggleExclusiveFullScreen);
    connect(m_ui->action3DView, &QAction::toggled, this,
            &MainWindow::toggle3DView);
    connect(m_ui->actionResetGUIElementsPos, &QAction::triggered, this,
            &MainWindow::doActionResetGUIElementsPos);
    connect(m_ui->actionRestoreWindowOnStartup, &QAction::toggled, this,
            &MainWindow::doActionRestoreWindowOnStartup);
    connect(m_ui->actionSaveCustomLayout, &QAction::triggered, this,
            &MainWindow::doActionSaveCustomLayout);
    connect(m_ui->actionRestoreDefaultLayout, &QAction::triggered, this,
            &MainWindow::doActionRestoreDefaultLayout);
    connect(m_ui->actionRestoreCustomLayout, &QAction::triggered, this,
            &MainWindow::doActionRestoreCustomLayout);
    connect(m_ui->actionZoomAndCenter, &QAction::triggered, this,
            &MainWindow::zoomOnSelectedEntities);
    connect(m_ui->actionGlobalZoom, &QAction::triggered, this,
            &MainWindow::setGlobalZoom);

    // "Edit > Selection" menu - Initialize selection controller
    // NOTE: initSelectionController() is called AFTER m_findDataDock is created
    // (in MainWindow constructor), not here. See the note below.
    // The actual initialization is done after m_findDataDock is created.

    connect(m_ui->actionLockRotationAxis, &QAction::triggered, this,
            &MainWindow::toggleLockRotationAxis);
    connect(m_ui->actionSaveViewportAsObject, &QAction::triggered, this,
            &MainWindow::doActionSaveViewportAsCamera);
    //"Display > Active SF" menu
    connect(m_ui->actionToggleActiveSFColorScale, &QAction::triggered, this,
            &MainWindow::doActionToggleActiveSFColorScale);
    connect(m_ui->actionShowActiveSFPrevious, &QAction::triggered, this,
            &MainWindow::doActionShowActiveSFPrevious);
    connect(m_ui->actionShowActiveSFNext, &QAction::triggered, this,
            &MainWindow::doActionShowActiveSFNext);

    connect(m_ui->actionDisplayOptions, &QAction::triggered, this,
            &MainWindow::showDisplayOptions);
    connect(m_ui->actionSetViewTop, &QAction::triggered, this,
            [=]() { setView(CC_TOP_VIEW); });
    connect(m_ui->actionSetViewBottom, &QAction::triggered, this,
            [=]() { setView(CC_BOTTOM_VIEW); });
    connect(m_ui->actionSetViewFront, &QAction::triggered, this,
            [=]() { setView(CC_FRONT_VIEW); });
    connect(m_ui->actionSetViewBack, &QAction::triggered, this,
            [=]() { setView(CC_BACK_VIEW); });
    connect(m_ui->actionSetViewLeft, &QAction::triggered, this,
            [=]() { setView(CC_LEFT_VIEW); });
    connect(m_ui->actionSetViewRight, &QAction::triggered, this,
            [=]() { setView(CC_RIGHT_VIEW); });
    connect(m_ui->actionSetViewIso1, &QAction::triggered, this,
            [=]() { setView(CC_ISO_VIEW_1); });
    connect(m_ui->actionSetViewIso2, &QAction::triggered, this,
            [=]() { setView(CC_ISO_VIEW_2); });

    connect(m_ui->actionAutoPickPivot, &QAction::toggled, this,
            &MainWindow::toggleActiveWindowAutoPickRotCenter);
    connect(m_ui->actionShowPivot, &QAction::toggled, this,
            &MainWindow::toggleRotationCenterVisibility);
    connect(m_ui->actionResetPivot, &QAction::triggered, this,
            &MainWindow::doActionResetRotCenter);
    connect(m_ui->actionPerspectiveProjection, &QAction::triggered, this,
            &MainWindow::doActionPerspectiveProjection);
    connect(m_ui->actionOrthogonalProjection, &QAction::triggered, this,
            &MainWindow::doActionOrthogonalProjection);

    connect(m_ui->actionEditCamera, &QAction::triggered, this,
            &MainWindow::doActionEditCamera);
    connect(m_ui->actionAnimation, &QAction::triggered, this,
            &MainWindow::doActionAnimation);
    connect(m_ui->actionScreenShot, &QAction::triggered, this,
            &MainWindow::doActionScreenShot);
    connect(m_ui->actionToggleOrientationMarker, &QAction::triggered, this,
            &MainWindow::doActionToggleOrientationMarker);
    connect(m_ui->actionGlobalShiftSettings, &QAction::triggered, this,
            &MainWindow::doActionGlobalShiftSeetings);

    // About (connect)
    connect(m_ui->helpAction, &QAction::triggered, this, &MainWindow::help);
    connect(m_ui->actionAboutPlugins, &QAction::triggered, m_pluginUIManager,
            &ccPluginUIManager::showAboutDialog);
    connect(m_ui->actionEnableQtWarnings, &QAction::toggled, this,
            &MainWindow::doEnableQtWarnings);
    connect(m_ui->actionAbout, &QAction::triggered, this, [this]() {
        ecvAboutDialog* aboutDialog = new ecvAboutDialog(this);
        aboutDialog->exec();
    });

    // echo mode
    m_ui->consoleWidget->setProperty("contextMenuPolicy",
                                     Qt::CustomContextMenu);
    connect(m_ui->consoleWidget,
            &ecvCustomQListWidget::customContextMenuRequested, this,
            &MainWindow::popMenuInConsole);
    // DGM: we don't want to block the 'dropEvent' method of MainWindow!
    connect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::filesDropped,
            this, &MainWindow::addToDBAuto, Qt::QueuedConnection);

    // hidden
    connect(m_ui->actionEnableVisualDebugTraces, &QAction::triggered, this,
            &MainWindow::toggleVisualDebugTraces);

    connect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::newLabel, this,
            &MainWindow::handleNewLabel);
    connect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::autoPickPivot,
            this, &MainWindow::setAutoPickPivot);
    connect(ecvDisplayTools::TheInstance(),
            &ecvDisplayTools::exclusiveFullScreenToggled, this,
            &MainWindow::toggleExclusiveFullScreen);

    // Not yet implemented!
    connect(m_ui->actionKMeans, &QAction::triggered, this,
            &MainWindow::doActionKMeans);
    connect(m_ui->actionFrontPropagation, &QAction::triggered, this,
            &MainWindow::doActionFrontPropagation);

    // update
    initApplicationUpdate();

    // Set up dynamic menus
    m_ui->menuFile->insertMenu(m_ui->actionSave, m_recentFiles->menu());
}

void MainWindow::initApplicationUpdate() {
    // TODO: check update when application start!
    if (!m_updateDlg) {
        m_updateDlg = new ecvUpdateDlg(this);
        connect(m_ui->actionCheckForUpdates, &QAction::triggered, this,
                &MainWindow::doCheckForUpdate);
    }
}

void MainWindow::initThemes() {
    // Option (connect)
    m_ui->DfaultThemeAction->setData(QVariant(Themes::THEME_DEFAULT));
    m_ui->BlueThemeAction->setData(QVariant(Themes::THEME_BLUE));
    m_ui->LightBlueThemeAction->setData(QVariant(Themes::THEME_LIGHTBLUE));
    m_ui->DarkBlueThemeAction->setData(QVariant(Themes::THEME_DARKBLUE));
    m_ui->BlackThemeAction->setData(QVariant(Themes::THEME_BLACK));
    m_ui->LightBlackThemeAction->setData(QVariant(Themes::THEME_LIGHTBLACK));
    m_ui->FlatBlackThemeAction->setData(QVariant(Themes::THEME_FLATBLACK));
    m_ui->DarkBlackThemeAction->setData(QVariant(Themes::THEME_DarkBLACK));
    m_ui->GrayThemeAction->setData(QVariant(Themes::THEME_GRAY));
    m_ui->LightGrayThemeAction->setData(QVariant(Themes::THEME_LIGHTGRAY));
    m_ui->DarkGrayThemeAction->setData(QVariant(Themes::THEME_DarkGRAY));
    m_ui->FlatWhiteThemeAction->setData(QVariant(Themes::THEME_FLATWHITE));
    m_ui->PsBlackThemeAction->setData(QVariant(Themes::THEME_PSBLACK));
    m_ui->SilverThemeAction->setData(QVariant(Themes::THEME_SILVER));
    m_ui->BFThemeAction->setData(QVariant(Themes::THEME_BF));
    m_ui->TestThemeAction->setData(QVariant(Themes::THEME_TEST));
    m_ui->ParaviewThemeAction->setData(QVariant(Themes::THEME_PARAVIEW));

    connect(m_ui->DfaultThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->BlueThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->LightBlueThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->DarkBlueThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->BlackThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->LightBlackThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->FlatBlackThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->DarkBlackThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->GrayThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->LightGrayThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->DarkGrayThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->FlatWhiteThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->PsBlackThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->SilverThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->BFThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->TestThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
    connect(m_ui->ParaviewThemeAction, &QAction::triggered, this,
            &MainWindow::changeTheme);
}

void MainWindow::initLanguages() {
    m_ui->englishAction->setData(QVariant(CLOUDVIEWER_LANG_ENGLISH));
    m_ui->chineseAction->setData(QVariant(CLOUDVIEWER_LANG_CHINESE));
    connect(m_ui->englishAction, &QAction::triggered, this,
            &MainWindow::changeLanguage);
    connect(m_ui->chineseAction, &QAction::triggered, this,
            &MainWindow::changeLanguage);
}

void MainWindow::initStatusBar() {
    // set mouse position label
    {
        m_mousePosLabel = new QLabel(this);
        QFont ft;
        ft.setBold(true);
        m_mousePosLabel->setFont(ft);
        m_mousePosLabel->setMinimumSize(m_mousePosLabel->sizeHint());
        m_mousePosLabel->setAlignment(Qt::AlignHCenter);
        m_ui->statusBar->insertWidget(0, m_mousePosLabel, 1);
        connect(ecvDisplayTools::TheInstance(),
                &ecvDisplayTools::mousePosChanged, this,
                &MainWindow::onMousePosChanged);
    }

    // set memory usage display widget (ParaView-style)
    {
        m_memoryUsageWidget = new QWidget(this);
        // No layout needed - we'll use absolute positioning for overlay

        // Progress bar for memory usage (thicker, like ParaView)
        m_memoryUsageProgressBar = new QProgressBar(m_memoryUsageWidget);
        m_memoryUsageProgressBar->setMinimumWidth(
                240);  // Minimum width (doubled)
        m_memoryUsageProgressBar->setMaximumWidth(
                500);  // Maximum width (doubled)
        m_memoryUsageProgressBar->setFixedHeight(
                20);  // Thicker height (doubled from 10)
        m_memoryUsageProgressBar->setMinimum(0);
        m_memoryUsageProgressBar->setMaximum(100);
        m_memoryUsageProgressBar->setTextVisible(false);
        m_memoryUsageProgressBar->setSizePolicy(QSizePolicy::Fixed,
                                                QSizePolicy::Fixed);
        m_memoryUsageProgressBar->move(0, 0);  // Position at top-left of widget
        m_memoryUsageProgressBar->setStyleSheet(
                "QProgressBar {"
                "    border: 1px solid #999;"
                "    border-radius: 2px;"
                "    background-color: #e0e0e0;"
                "}"
                "QProgressBar::chunk {"
                "    background-color: #B8E6B8;"  // Light green, matches
                                                  // ParaView
                "    border-radius: 1px;"
                "}");

        // Label for memory usage text (overlay on progress bar)
        m_memoryUsageLabel = new QLabel(m_memoryUsageWidget);
        m_memoryUsageLabel->setMinimumWidth(240);  // Minimum width (doubled)
        m_memoryUsageLabel->setMaximumWidth(500);  // Maximum width (doubled)
        m_memoryUsageLabel->setFixedHeight(20);  // Same height as progress bar
        m_memoryUsageLabel->setAlignment(Qt::AlignCenter);  // Center text
        m_memoryUsageLabel->setSizePolicy(QSizePolicy::Fixed,
                                          QSizePolicy::Fixed);
        m_memoryUsageLabel->move(0, 0);  // Overlay on progress bar
        QFont labelFont = m_memoryUsageLabel->font();
        labelFont.setPointSize(labelFont.pointSize() - 1);
        m_memoryUsageLabel->setFont(labelFont);
        // Make label transparent so progress bar shows through
        m_memoryUsageLabel->setStyleSheet("background: transparent;");

        // Set widget size to match progress bar
        m_memoryUsageWidget->setFixedSize(
                240, 20);  // Will be updated by updateMemoryUsageWidgetSize
        m_memoryUsageWidget->setSizePolicy(QSizePolicy::Fixed,
                                           QSizePolicy::Fixed);
        m_ui->statusBar->addPermanentWidget(m_memoryUsageWidget, 0);

        // Create timer to update memory usage periodically
        m_memoryUsageTimer = new QTimer(this);
        connect(m_memoryUsageTimer, &QTimer::timeout, this,
                &MainWindow::updateMemoryUsage);
        m_memoryUsageTimer->start(5000);  // Update every 5 seconds

        // Initial update
        updateMemoryUsage();
        updateMemoryUsageWidgetSize();
    }

    statusBar()->showMessage(tr("Ready"));
}

void MainWindow::updateMemoryUsage() {
    if (!m_memoryUsageProgressBar || !m_memoryUsageLabel) {
        return;
    }

    // Get system memory information
    cloudViewer::system::MemoryInfo memInfo =
            cloudViewer::system::getMemoryInfo();

    if (memInfo.totalRam > 0) {
        // Calculate used memory (total - available)
        qint64 bytesUsed =
                static_cast<qint64>(memInfo.totalRam - memInfo.availableRam);
        qint64 bytesTotal = static_cast<qint64>(memInfo.totalRam);

        // Calculate percentage
        int percentage = static_cast<int>((bytesUsed * 100) / bytesTotal);
        m_memoryUsageProgressBar->setValue(percentage);

        // Format sizes
        QString usedStr = formatBytes(bytesUsed);
        QString totalStr = formatBytes(bytesTotal);

        // Get hostname
        QString hostname = QHostInfo::localHostName();

        // Update label text: "hostname: used/total percentage%"
        QString text = QString("%1: %2/%3 %4%")
                               .arg(hostname)
                               .arg(usedStr)
                               .arg(totalStr)
                               .arg(percentage);
        m_memoryUsageLabel->setText(text);
    }
}

void MainWindow::updateMemoryUsageWidgetSize() {
    if (!m_memoryUsageWidget || !m_memoryUsageProgressBar ||
        !m_memoryUsageLabel) {
        return;
    }

    // Get window width
    int windowWidth = width();

    // Calculate widget width based on window size (ParaView-style scaling)
    // Scale between 240px (min) and 500px (max) based on window width (doubled)
    const int minWidth = 240;
    const int maxWidth = 500;

    int widgetWidth;
    if (windowWidth <= 1280) {
        // Small windows: use minimum width
        widgetWidth = minWidth;
    } else if (windowWidth >= 2560) {
        // Large windows: use maximum width
        widgetWidth = maxWidth;
    } else {
        // Medium windows: linear interpolation
        double ratio = static_cast<double>(windowWidth - 1280) / (2560 - 1280);
        widgetWidth =
                static_cast<int>(minWidth + ratio * (maxWidth - minWidth));
    }

    // Update progress bar and label width and widget size
    m_memoryUsageProgressBar->setFixedWidth(widgetWidth);
    m_memoryUsageLabel->setFixedWidth(widgetWidth);
    m_memoryUsageWidget->setFixedSize(widgetWidth,
                                      20);  // Height is fixed at 20

    // Force update to reflect size changes
    m_memoryUsageWidget->updateGeometry();
    m_memoryUsageProgressBar->update();
    m_memoryUsageLabel->update();
}

QString MainWindow::formatBytes(qint64 bytes) {
    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;
    const qint64 TB = GB * 1024;

    if (bytes >= TB) {
        return QString("%1 TiB").arg(bytes / static_cast<double>(TB), 0, 'f',
                                     1);
    } else if (bytes >= GB) {
        return QString("%1 GiB").arg(bytes / static_cast<double>(GB), 0, 'f',
                                     1);
    } else if (bytes >= MB) {
        return QString("%1 MiB").arg(bytes / static_cast<double>(MB), 0, 'f',
                                     1);
    } else if (bytes >= KB) {
        return QString("%1 KiB").arg(bytes / static_cast<double>(KB), 0, 'f',
                                     1);
    } else {
        return QString("%1 B").arg(bytes);
    }
}

void MainWindow::initPlugins() {
    m_pluginUIManager->init();

    // Set up dynamic tool bars
    QToolBar* glPclToolbar = m_pluginUIManager->glPclToolbar();
    QToolBar* mainPluginToolbar = m_pluginUIManager->mainPluginToolbar();
    addToolBar(Qt::RightToolBarArea, glPclToolbar);
    addToolBar(Qt::RightToolBarArea, mainPluginToolbar);
    // Register plugin toolbars with layout manager
    m_layoutManager->registerRightSideToolBar(glPclToolbar);
    m_layoutManager->registerRightSideToolBar(mainPluginToolbar);

    // Combine all additional plugin toolbars into a single unified toolbar
    // But exclude Python plugins - they should be handled separately
    QList<QToolBar*> additionalToolbars =
            m_pluginUIManager->additionalPluginToolbars();

    // Separate Python plugin toolbars from other toolbars
    QList<QToolBar*> pythonPluginToolbars;
    QList<QToolBar*> otherPluginToolbars;

    for (QToolBar* toolbar : additionalToolbars) {
        if (ccPluginUIManager::isPythonPluginToolbar(toolbar)) {
            pythonPluginToolbars.append(toolbar);
        } else {
            otherPluginToolbars.append(toolbar);
        }
    }

    CVLog::PrintDebug(
            QString("[MainWindow] Found %1 additional plugin toolbars (%2 "
                    "Python, %3 others)")
                    .arg(additionalToolbars.size())
                    .arg(pythonPluginToolbars.size())
                    .arg(otherPluginToolbars.size()));

    // Handle Python plugin toolbars separately - add them individually
    for (QToolBar* pythonToolbar : pythonPluginToolbars) {
        CVLog::PrintDebug(
                QString("[MainWindow] Adding Python plugin toolbar '%1' "
                        "separately")
                        .arg(pythonToolbar->objectName()));
        addToolBar(Qt::TopToolBarArea, pythonToolbar);
        pythonToolbar->setVisible(true);
        pythonToolbar->show();
    }

    // Check if UnifiedPluginToolbar already exists (to avoid duplicate
    // creation)
    QToolBar* existingUnifiedToolbar =
            findChild<QToolBar*>("UnifiedPluginToolbar");
    if (existingUnifiedToolbar) {
        // Remove existing toolbar first to avoid duplicates
        CVLog::Print("[MainWindow] Removing existing UnifiedPluginToolbar");
        removeToolBar(existingUnifiedToolbar);
        existingUnifiedToolbar->deleteLater();
    }

    if (!otherPluginToolbars.isEmpty()) {
        QToolBar* unifiedPluginToolbar =
                new QToolBar(tr("MultipleActionsPlugins"), this);
        unifiedPluginToolbar->setObjectName("UnifiedPluginToolbar");

        // Collect all actions from additional toolbars, avoiding duplicates
        QSet<QAction*> addedActions;

        for (QToolBar* toolbar : otherPluginToolbars) {
            QList<QAction*> actions = toolbar->actions();
            CVLog::PrintDebug(
                    QString("[MainWindow] Processing toolbar '%1' with %2 "
                            "actions")
                            .arg(toolbar->objectName())
                            .arg(actions.size()));

            for (QAction* action : actions) {
                // Only add action if it's not already added to unified toolbar
                if (!addedActions.contains(action)) {
                    unifiedPluginToolbar->addAction(action);
                    addedActions.insert(action);
                }
            }

            // Add separator after each toolbar's actions (except after the last
            // toolbar)
            if (toolbar != otherPluginToolbars.last()) {
                unifiedPluginToolbar->addSeparator();
            }

            // IMPORTANT: Completely remove and hide the original toolbar
            // Set parent to nullptr to prevent it from being restored by
            // restoreState()
            removeToolBar(toolbar);
            toolbar->setParent(nullptr);
            toolbar->setVisible(false);
            toolbar->hide();
        }

        // Only add unified toolbar if it has actions
        if (!unifiedPluginToolbar->actions().isEmpty()) {
            // Add the unified toolbar to the top
            addToolBar(Qt::TopToolBarArea, unifiedPluginToolbar);
            unifiedPluginToolbar->setVisible(true);
            unifiedPluginToolbar->show();

            CVLog::PrintDebug(
                    QString("[MainWindow] Created UnifiedPluginToolbar "
                            "with %1 actions from %2 toolbars")
                            .arg(unifiedPluginToolbar->actions().size())
                            .arg(otherPluginToolbars.size()));
        } else {
            // No actions, delete the empty toolbar
            CVLog::Warning(
                    "[MainWindow] UnifiedPluginToolbar has no actions, "
                    "deleting");
            delete unifiedPluginToolbar;
        }
    }

    // Set up dynamic menus
    m_ui->menuBar->insertMenu(m_ui->menuDisplay->menuAction(),
                              m_pluginUIManager->pclAlgorithmMenu());
    m_ui->menuBar->insertMenu(m_ui->menuDisplay->menuAction(),
                              m_pluginUIManager->pluginMenu());

    m_ui->menuToolbars->addAction(
            m_pluginUIManager->actionShowMainPluginToolbar());
    m_ui->menuToolbars->addAction(
            m_pluginUIManager->actionShowPCLAlgorithmToolbar());

    // Apply unified icon size and style to all plugin toolbars
    // This includes glPclToolbar, mainPluginToolbar, and UnifiedPluginToolbar
    updateAllToolbarIconSizes();
}

void MainWindow::initDBRoot() {
    // db-tree
    {
        m_ccRoot =
                new ccDBRoot(m_ui->dbTreeView, m_ui->propertiesTreeView, this);
        connect(m_ccRoot, &ccDBRoot::selectionChanged, this,
                &MainWindow::updateUIWithSelection);
        connect(m_ccRoot, &ccDBRoot::dbIsEmpty, [&]() {
            updateUIWithSelection();
            updateMenus();
        });  // we don't call updateUI because there's no need to update the
             // properties dialog
        connect(m_ccRoot, &ccDBRoot::dbIsNotEmptyAnymore, [&]() {
            updateUIWithSelection();
            updateMenus();
        });  // we don't call updateUI because there's no need to update the
             // properties dialog
    }

    ecvDisplayTools::SetSceneDB(m_ccRoot->getRootEntity());
    m_ccRoot->updatePropertiesView();

    connect(ecvDisplayTools::TheInstance(),
            &ecvDisplayTools::entitySelectionChanged, this,
            [=](ccHObject* entity) { m_ccRoot->selectEntity(entity); });
    connect(ecvDisplayTools::TheInstance(),
            &ecvDisplayTools::entitiesSelectionChanged, this,
            [=](std::unordered_set<int> entities) {
                m_ccRoot->selectEntities(entities);
            });
}

#ifdef BUILD_RECONSTRUCTION
void MainWindow::initReconstructions() {
    // init reconstructions
    if (!m_rcw) {
        m_rcw = new cloudViewer::ReconstructionWidget(this);
    }

    // Set up dynamic tool bars
    QSettings settings;
    bool autoShowFlag =
            settings.value(ecvPS::AutoShowReconstructionToolBar(), true)
                    .toBool();
    QAction* showToolbarAction = new QAction(tr("Reconstruction"), this);
    connect(showToolbarAction, &QAction::toggled, this,
            &MainWindow::autoShowReconstructionToolBar);
    showToolbarAction->setCheckable(true);
    showToolbarAction->setEnabled(true);
    // Get screen width for icon size calculation
    QScreen* screen = QGuiApplication::primaryScreen();
    int screenWidth = screen ? screen->geometry().width() : 1920;

    for (QToolBar* toolbar : m_rcw->getReconstructionToolbars()) {
        addToolBar(Qt::TopToolBarArea, toolbar);
        connect(showToolbarAction, &QAction::toggled, toolbar,
                &QToolBar::setVisible);
    }
    m_ui->menuToolbars->addAction(showToolbarAction);
    showToolbarAction->setChecked(autoShowFlag);

    // Set up dynamic menus
    QMenu* rc_menu = new QMenu(tr("Reconstruction"), this);
    for (QMenu* menu : m_rcw->getReconstructionMenus()) {
        rc_menu->addMenu(menu);
    }
    m_ui->menuBar->insertMenu(m_ui->menuDisplay->menuAction(), rc_menu);

    // Set docker widget
    QDockWidget* logWidget = m_rcw->getLogWidget();
    this->addDockWidget(Qt::RightDockWidgetArea, logWidget);

    // Set reconstruction status bar
    m_ui->statusBar->insertPermanentWidget(1, m_rcw->getImageStatusBar(), 0);
    m_ui->statusBar->insertPermanentWidget(1, m_rcw->getTimerStatusBar(), 0);
}

void MainWindow::autoShowReconstructionToolBar(bool state) {
    QSettings settings;
    settings.setValue(ecvPS::AutoShowReconstructionToolBar(), state);
}
#endif

void MainWindow::toggleActiveWindowAutoPickRotCenter(bool state) {
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::SetAutoPickPivotAtCenter(state);

        // save the option
        {
            QSettings settings;
            settings.setValue(ecvPS::AutoPickRotationCenter(), state);
        }
    }
}

void MainWindow::doActionResetRotCenter() {
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::ResetCenterOfRotation();
    }
}

void MainWindow::toggleRotationCenterVisibility(bool state) {
    if (ecvDisplayTools::GetCurrentScreen()) {
        if (state) {
            ecvDisplayTools::SetPivotVisibility(
                    ecvDisplayTools::PIVOT_ALWAYS_SHOW);
        } else {
            ecvDisplayTools::SetPivotVisibility(ecvDisplayTools::PIVOT_HIDE);
        }

        // save the option
        {
            QSettings settings;
            settings.setValue(ecvPS::AutoShowCenter(), state);
        }
    }
}

void MainWindow::onMousePosChanged(const QPoint& pos) {
    if (m_mousePosLabel) {
        double x = pos.x();
        double y = pos.y();
        QString labelText = QString("Location | (%1, %2)")
                                    .arg(QString::number(x))
                                    .arg(QString::number(y));
        m_mousePosLabel->setText(labelText);
    }
}

void MainWindow::setAutoPickPivot(bool state) {
    m_ui->actionAutoPickPivot->blockSignals(true);
    m_ui->actionAutoPickPivot->setChecked(state);
    m_ui->actionAutoPickPivot->blockSignals(false);
    toggleActiveWindowAutoPickRotCenter(state);
}

void MainWindow::setOrthoView() {
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::SetPerspectiveState(false, true);

        // update pop-up menu 'top' icon
        if (m_viewModePopupButton)
            m_viewModePopupButton->setIcon(
                    m_ui->actionOrthogonalProjection->icon());
    }
}

void MainWindow::setPerspectiveView() {
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::SetPerspectiveState(true, true);

        // update pop-up menu 'top' icon
        if (m_viewModePopupButton)
            m_viewModePopupButton->setIcon(
                    m_ui->actionPerspectiveProjection->icon());
    }
}

int MainWindow::getRenderWindowCount() const {
    return m_mdiArea ? m_mdiArea->subWindowList().size() : 0;
}

QMdiSubWindow* MainWindow::getMDISubWindow(QWidget* win) {
    QList<QMdiSubWindow*> subWindowList = m_mdiArea->subWindowList();
    for (int i = 0; i < subWindowList.size(); ++i) {
        if (subWindowList[i]->widget() == win) return subWindowList[i];
    }

    // not found!
    return nullptr;
}

void MainWindow::update3DViewsMenu() {
    QList<QMdiSubWindow*> windows = m_mdiArea->subWindowList();
    if (!windows.isEmpty()) {
        // Dynamic Separator
        QAction* separator = new QAction(this);
        separator->setSeparator(true);

        int i = 0;

        for (QMdiSubWindow* window : windows) {
            QWidget* child = window->widget();

            QString text = QString("&%1 %2").arg(++i).arg(child->windowTitle());

            // connect(action, &QAction::triggered, this, [=]() {
            //	setActiveSubWindow(window);
            // });
        }
    }
}

void MainWindow::updateViewModePopUpMenu() {
    if (!m_viewModePopupButton) return;

    // update the view mode pop-up 'top' icon
    if (ecvDisplayTools::GetCurrentScreen()) {
        bool perspectiveEnabled = ecvDisplayTools::GetPerspectiveState();

        QAction* currentModeAction = nullptr;
        if (!perspectiveEnabled) {
            currentModeAction = m_ui->actionOrthogonalProjection;
        } else {
            currentModeAction = m_ui->actionPerspectiveProjection;
        }

        assert(currentModeAction);
        m_viewModePopupButton->setIcon(currentModeAction->icon());
        m_viewModePopupButton->setEnabled(true);
    } else {
        m_viewModePopupButton->setIcon(QIcon());
        m_viewModePopupButton->setEnabled(false);
    }
}

void MainWindow::addWidgetToQMdiArea(QWidget* viewWidget) {
    if (viewWidget && !MainWindow::GetRenderWindow(viewWidget->windowTitle())) {
        viewWidget->showMaximized();
        m_mdiArea->addSubWindow(viewWidget);
    } else {
        m_mdiArea->setActiveSubWindow(getMDISubWindow(viewWidget));
    }
}

QWidget* MainWindow::GetActiveRenderWindow() {
    return TheInstance()->getActiveWindow();
}

void MainWindow::GetRenderWindows(std::vector<QWidget*>& glWindows) {
    const QList<QMdiSubWindow*> windows =
            TheInstance()->m_mdiArea->subWindowList();

    if (windows.empty()) return;

    glWindows.clear();
    glWindows.reserve(windows.size());

    for (QMdiSubWindow* window : windows) {
        glWindows.push_back(window->widget());
    }
}

QWidget* MainWindow::GetRenderWindow(const QString& title) {
    const QList<QMdiSubWindow*> windows =
            TheInstance()->m_mdiArea->subWindowList();

    if (windows.empty()) return nullptr;

    for (QMdiSubWindow* window : windows) {
        QWidget* win = window->widget();
        if (win->windowTitle() == title) return win;
    }

    return nullptr;
}

QWidget* MainWindow::getWindow(int index) const {
    QList<QMdiSubWindow*> subWindowList = m_mdiArea->subWindowList();
    if (index >= 0 && index < subWindowList.size()) {
        QWidget* win = subWindowList[index]->widget();
        assert(win);
        return win;
    } else {
        assert(false);
        return nullptr;
    }
}

QWidget* MainWindow::getActiveWindow() {
    if (!m_mdiArea) {
        return nullptr;
    }

    QMdiSubWindow* activeSubWindow = m_mdiArea->activeSubWindow();
    if (activeSubWindow) {
        return activeSubWindow->widget();
    } else {
        QList<QMdiSubWindow*> subWindowList = m_mdiArea->subWindowList();
        if (!subWindowList.isEmpty()) {
            return subWindowList[0]->widget();
        }
    }

    return nullptr;
}

void MainWindow::ChangeStyle(const QString& qssFile) {
    QString fileName = qssFile;

    if (!fileName.isEmpty()) {
        QFile file(fileName);

        if (file.open(QFile::ReadOnly)) {
            QString str = file.readAll();
            static QString qssStyle;

            if (qssStyle == str) {
                return;
            }

            qssStyle = str;
            QString paletteColor = str.mid(20, 7);
            ecvApp->setPalette(QPalette(QColor(paletteColor)));
            ecvApp->setStyleSheet(qssStyle);
        }
    } else {
        // default
        ecvApp->setPalette(QPalette(QColor(240, 240, 240, 255)));
        ecvApp->setStyleSheet(QString());
    }
}

void MainWindow::setUiManager(QUIWidget* uiManager) {
    this->m_uiManager = uiManager;
}

void MainWindow::toggleVisualDebugTraces() {
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::ToggleDebugTrace();
        ecvDisplayTools::RedrawDisplay(true, false);
    }
}

void MainWindow::forceConsoleDisplay() {
    // if the console is hidden, we automatically display it!
    if (m_ui->consoleDock && m_ui->consoleDock->isHidden()) {
        m_ui->consoleDock->show();
        QApplication::processEvents();
    }
}

ccHObject* MainWindow::dbRootObject() {
    return (m_ccRoot ? m_ccRoot->getRootEntity() : nullptr);
}

void MainWindow::doEnableQtWarnings(bool state) {
    ecvConsole::EnableQtMessages(state);
}

/************** STATIC METHODS ******************/

MainWindow* MainWindow::TheInstance() {
    if (!s_instance) s_instance = new MainWindow();
    return s_instance;
}

void MainWindow::DestroyInstance() {
    delete s_instance;
    s_instance = nullptr;
}

void MainWindow::on3DViewActivated(QMdiSubWindow* mdiWin) {
    if (!mdiWin) {
        return;
    }

    QWidget* screen = mdiWin->widget();
    if (screen) {
        m_ui->actionExclusiveFullScreen->blockSignals(true);
        m_ui->actionExclusiveFullScreen->setChecked(
                ecvDisplayTools::ExclusiveFullScreen());
        m_ui->actionExclusiveFullScreen->blockSignals(false);
    }
    m_ui->actionExclusiveFullScreen->setEnabled(screen != nullptr);
}

void MainWindow::dispToConsole(
        QString message, ConsoleMessageLevel level /*=STD_CONSOLE_MESSAGE*/) {
    switch (level) {
        case STD_CONSOLE_MESSAGE:
            ecvConsole::Print(message);
            break;
        case WRN_CONSOLE_MESSAGE:
            ecvConsole::Warning(message);
            break;
        case ERR_CONSOLE_MESSAGE:
            ecvConsole::Error(message);
            break;
    }
}

ccUniqueIDGenerator::Shared MainWindow::getUniqueIDGenerator() {
    return ccObject::GetUniqueIDGenerator();
}

// Open point cloud
void MainWindow::getFileFilltersAndHistory(QStringList& fileFilters,
                                           QString& currentOpenDlgFilter) {
    currentOpenDlgFilter =
            ecvSettingManager::getValue(ecvPS::LoadFile(),
                                        ecvPS::SelectedInputFilter(),
                                        AsciiFilter::GetFileFilter())
                    .toString();

    // Add all available file I/O filters (with import capabilities)
    fileFilters.append(s_allFilesFilter);
    bool defaultFilterFound = false;
    {
        for (const FileIOFilter::Shared& filter : FileIOFilter::GetFilters()) {
            if (filter->importSupported()) {
                const QStringList fileFilterList = filter->getFileFilters(true);

                for (const QString& fileFilter : fileFilterList) {
                    fileFilters.append(fileFilter);
                    // is it the (last) default filter?
                    if (!defaultFilterFound &&
                        (currentOpenDlgFilter == fileFilter)) {
                        defaultFilterFound = true;
                    }
                }
            }
        }
    }

    // default filter is still valid?
    if (!defaultFilterFound) currentOpenDlgFilter = s_allFilesFilter;
}

void MainWindow::doActionOpenFile() {
    // persistent settings
    QString currentPath =
            ecvSettingManager::getValue(ecvPS::LoadFile(), ecvPS::CurrentPath(),
                                        ecvFileUtils::defaultDocPath())
                    .toString();

    QString currentOpenDlgFilter;
    QStringList fileFilters;
    getFileFilltersAndHistory(fileFilters, currentOpenDlgFilter);

    // file choosing dialog
    QStringList selectedFiles = QFileDialog::getOpenFileNames(
            this, tr("Open file(s)"), currentPath,
            fileFilters.join(s_fileFilterSeparator), &currentOpenDlgFilter,
            ECVFileDialogOptions());

    if (selectedFiles.isEmpty()) return;

    // persistent save last loading parameters
    currentPath = QFileInfo(selectedFiles[0]).absolutePath();
    ecvSettingManager::setValue(ecvPS::LoadFile(), ecvPS::CurrentPath(),
                                currentPath);
    ecvSettingManager::setValue(ecvPS::LoadFile(), ecvPS::SelectedInputFilter(),
                                currentOpenDlgFilter);

    // this way FileIOFilter will try to guess the file type automatically!
    if (currentOpenDlgFilter == s_allFilesFilter) currentOpenDlgFilter.clear();

    // load files
    addToDB(selectedFiles, currentOpenDlgFilter);
}

void MainWindow::addToDBAuto(const QStringList& filenames,
                             bool displayDialog /* = true*/) {
    addToDB(filenames, QString(), displayDialog);
}

void MainWindow::addToDB(const QStringList& filenames,
                         QString fileFilter /*=QString()*/,
                         bool displayDialog /* = true*/) {
    // to use the same 'global shift' for multiple files
    CCVector3d loadCoordinatesShift(0, 0, 0);
    bool loadCoordinatesTransEnabled = false;

    FileIOFilter::LoadParameters parameters;
    {
        parameters.alwaysDisplayLoadDialog = displayDialog;
        parameters.shiftHandlingMode =
                ecvGlobalShiftManager::DIALOG_IF_NECESSARY;
        parameters.coordinatesShift = &loadCoordinatesShift;
        parameters.coordinatesShiftEnabled = &loadCoordinatesTransEnabled;
        parameters.parentWidget = this;
    }

    // the same for 'addToDB' (if the first one is not supported, or if the
    // scale remains too big)
    CCVector3d addCoordinatesShift(0, 0, 0);

    const ecvOptions& options = ecvOptions::Instance();
    FileIOFilter::ResetSesionCounter();

    for (const QString& filename : filenames) {
        CC_FILE_ERROR result = CC_FERR_NO_ERROR;
        ccHObject* newGroup = FileIOFilter::LoadFromFile(filename, parameters,
                                                         result, fileFilter);

        if (newGroup) {
            if (!options.normalsDisplayedByDefault) {
                // disable the normals on all loaded clouds!
                ccHObject::Container clouds;
                newGroup->filterChildren(clouds, true, CV_TYPES::POINT_CLOUD);
                for (ccHObject* cloud : clouds) {
                    if (cloud) {
                        static_cast<ccGenericPointCloud*>(cloud)->showNormals(
                                false);
                    }
                }
            }

            addToDB(newGroup, true, true, false);

            m_recentFiles->addFilePath(filename);
        }

        if (result == CC_FERR_CANCELED_BY_USER) {
            // stop importing the file if the user has cancelled the current
            // process!
            break;
        }
    }

    statusBar()->showMessage(tr("%1 file(s) loaded").arg(filenames.size()),
                             2000);
}

void MainWindow::addToDB(ccHObject* obj,
                         bool updateZoom /*=false*/,
                         bool autoExpandDBTree /*=true*/,
                         bool checkDimensions /*=false*/,
                         bool autoRedraw /*=true*/) {
    // let's check that the new entity is not too big nor too far from scene
    // center!
    if (checkDimensions) {
        // get entity bounding box
        ccBBox bBox = obj->getBB_recursive();

        CCVector3 center = bBox.getCenter();
        PointCoordinateType diag = bBox.getDiagNorm();

        CCVector3d P = CCVector3d::fromArray(center.u);
        CCVector3d Pshift(0, 0, 0);
        double scale = 1.0;
        bool preserveCoordinateShift = true;
        // here we must test that coordinates are not too big whatever the case
        // because OpenGL really doesn't like big ones (even if we work with
        // GLdoubles :( ).
        if (ecvGlobalShiftManager::Handle(
                    P, diag, ecvGlobalShiftManager::DIALOG_IF_NECESSARY, false,
                    Pshift, &preserveCoordinateShift, &scale)) {
            bool needRescale = (scale != 1.0);
            bool needShift = (Pshift.norm2() > 0);

            if (needRescale || needShift) {
                ccGLMatrix mat;
                mat.toIdentity();
                mat.data()[0] = mat.data()[5] = mat.data()[10] =
                        static_cast<float>(scale);
                mat.setTranslation(Pshift);
                obj->applyGLTransformation_recursive(&mat);
                ecvConsole::Warning(
                        tr("Entity '%1' has been translated: (%2,%3,%4) and "
                           "rescaled of a factor %5 [original position will be "
                           "restored when saving]")
                                .arg(obj->getName())
                                .arg(Pshift.x, 0, 'f', 2)
                                .arg(Pshift.y, 0, 'f', 2)
                                .arg(Pshift.z, 0, 'f', 2)
                                .arg(scale, 0, 'f', 6));
            }

            // update 'global shift' and 'global scale' for ALL clouds
            // recursively
            if (preserveCoordinateShift) {
                // FIXME: why don't we do that all the time by the way?!
                ccHObject::Container children;
                children.push_back(obj);
                while (!children.empty()) {
                    ccHObject* child = children.back();
                    children.pop_back();

                    if (child->isKindOf(CV_TYPES::POINT_CLOUD)) {
                        ccGenericPointCloud* pc =
                                ccHObjectCaster::ToGenericPointCloud(child);
                        pc->setGlobalShift(pc->getGlobalShift() + Pshift);
                        pc->setGlobalScale(pc->getGlobalScale() * scale);
                    }

                    for (unsigned i = 0; i < child->getChildrenNumber(); ++i) {
                        children.push_back(child->getChild(i));
                    }
                }
            }
        }
    }

    // add object to DB root
    if (m_ccRoot) {
        // force a 'global zoom' if the DB was emtpy!
        if (!m_ccRoot->getRootEntity() ||
            m_ccRoot->getRootEntity()->getChildrenNumber() == 0) {
            updateZoom = true;
        }

        // avoid rendering other object this time
        ecvDisplayTools::SetRedrawRecursive(false);
        // redraw new added obj
        ecvDisplayTools::SetRedrawRecursive(obj, true);

        ccHObject::Container childs;
        obj->filterChildren(childs, true, CV_TYPES::IMAGE);
        if (!childs.empty()) {
            updateZoom = false;
            autoRedraw = true;
        }

        m_ccRoot->addElement(obj, autoExpandDBTree);
    } else {
        CVLog::Warning(
                tr("[MainWindow::addToDB] Internal error: no associated db?!"));
        assert(false);
    }

    // eventually we update the corresponding display
    if (updateZoom) {
        ecvDisplayTools::ZoomGlobal();  // automatically calls redrawDisplay()
    } else if (autoRedraw) {
        refreshObject(obj);
    }

#ifdef USE_PCL_BACKEND
    // ParaView-style: When new entities are added, refresh Data Producer combo
    // but do NOT disable selection tools (ParaView doesn't do this)
    // Reference: ParaView's pqSelectionManager::onSourceAdded() simply connects
    // to the new source's selectionChanged signal without disabling selection
    if (m_findDataDock) {
        // Use QTimer to defer the refresh to ensure VTK actors are fully
        // created
        QTimer::singleShot(100, this, [this]() {
            if (m_findDataDock) {
                m_findDataDock->refreshDataProducers();
                CVLog::PrintDebug(
                        "[MainWindow::addToDB] Data Producer combo refreshed "
                        "after "
                        "adding new entity");
            }
        });
    }

    // Invalidate cached selection data since scene content changed
    // This ensures stale polydata references are not used
    // Note: Do NOT disable selection tools - user may be in the middle of
    // selection
    if (m_selectionController) {
        m_selectionController->invalidateCache();
    }
#endif
}

void MainWindow::doActionEditCamera() {
    // current active MDI area
    QMdiSubWindow* qWin = m_mdiArea->activeSubWindow();
    if (!qWin) return;

#ifdef USE_PCL_BACKEND
    if (!m_cpeDlg) {
        m_cpeDlg = new ecvCameraParamEditDlg(qWin, m_pickingHub);
        EditCameraTool* tool =
                new EditCameraTool(ecvDisplayTools::GetVisualizer3D());
        m_cpeDlg->setCameraTool(tool);

        connect(m_mdiArea, &QMdiArea::subWindowActivated, m_cpeDlg,
                static_cast<void (ecvCameraParamEditDlg::*)(QMdiSubWindow*)>(
                        &ecvCameraParamEditDlg::linkWith));

        registerOverlayDialog(m_cpeDlg, Qt::BottomLeftCorner);
    }

    m_cpeDlg->linkWith(qWin);
    m_cpeDlg->start();
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif

    updateOverlayDialogsPlacement();
}

static unsigned s_viewportIndex = 0;
void MainWindow::doActionSaveViewportAsCamera() {
    QWidget* win = getActiveWindow();
    if (!win) return;

    cc2DViewportObject* viewportObject = new cc2DViewportObject(
            QString("Viewport #%1").arg(++s_viewportIndex));
    viewportObject->setParameters(ecvDisplayTools::GetViewportParameters());

    addToDB(viewportObject);
}

void MainWindow::toggleLockRotationAxis() {
    QMainWindow* win = ecvDisplayTools::GetMainWindow();
    if (win) {
        bool wasLocked = ecvDisplayTools::IsRotationAxisLocked();
        bool isLocked = !wasLocked;

        static CCVector3d s_lastAxis(0.0, 0.0, 1.0);
        if (isLocked) {
            ccAskThreeDoubleValuesDlg axisDlg(
                    "x", "y", "z", -1.0e12, 1.0e12, s_lastAxis.x, s_lastAxis.y,
                    s_lastAxis.z, 4, tr("Lock rotation axis"), this);
            if (axisDlg.buttonBox->button(QDialogButtonBox::Ok))
                axisDlg.buttonBox->button(QDialogButtonBox::Ok)->setFocus();
            if (!axisDlg.exec()) return;
            s_lastAxis.x = axisDlg.doubleSpinBox1->value();
            s_lastAxis.y = axisDlg.doubleSpinBox2->value();
            s_lastAxis.z = axisDlg.doubleSpinBox3->value();
        }
        ecvDisplayTools::LockRotationAxis(isLocked, s_lastAxis);

        m_ui->actionLockRotationAxis->blockSignals(true);
        m_ui->actionLockRotationAxis->setChecked(isLocked);
        m_ui->actionLockRotationAxis->blockSignals(false);

        if (isLocked) {
            ecvDisplayTools::DisplayNewMessage(
                    tr("[ROTATION LOCKED]"),
                    ecvDisplayTools::UPPER_CENTER_MESSAGE, false, 24 * 3600,
                    ecvDisplayTools::ROTAION_LOCK_MESSAGE);
        } else {
            ecvDisplayTools::DisplayNewMessage(
                    QString(), ecvDisplayTools::UPPER_CENTER_MESSAGE, false, 0,
                    ecvDisplayTools::ROTAION_LOCK_MESSAGE);
        }
        ecvDisplayTools::SetRedrawRecursive(false);
        ecvDisplayTools::RedrawDisplay(true, false);
    }
}

void MainWindow::doActionAnimation() {
    // current active MDI area
    QMdiSubWindow* qWin = m_mdiArea->activeSubWindow();
    if (!qWin) return;

    if (!m_animationDlg) {
        m_animationDlg = new ecvAnimationParamDlg(qWin, this, m_pickingHub);

        connect(m_mdiArea, &QMdiArea::subWindowActivated, m_animationDlg,
                static_cast<void (ecvAnimationParamDlg::*)(QMdiSubWindow*)>(
                        &ecvAnimationParamDlg::linkWith));

        registerOverlayDialog(m_animationDlg, Qt::BottomLeftCorner);
    }

    m_animationDlg->linkWith(qWin);
    m_animationDlg->start();
    updateOverlayDialogsPlacement();
}

void MainWindow::doActionScreenShot() {
    QWidget* win = getActiveWindow();
    if (!win) return;

    ccRenderToFileDlg rtfDlg(static_cast<unsigned>(win->width()),
                             static_cast<unsigned>(win->height()), this);

    if (rtfDlg.exec()) {
        QApplication::processEvents();
        ecvDisplayTools::RenderToFile(rtfDlg.getFilename(), rtfDlg.getZoom(),
                                      rtfDlg.dontScalePoints(),
                                      rtfDlg.renderOverlayItems());
    }
}

void MainWindow::doActionToggleOrientationMarker(bool state) {
    ecvDisplayTools::ToggleOrientationMarker(state);
}

void MainWindow::doActionSaveFile() {
    if (!haveSelection()) return;

    ccHObject clouds(tr("clouds"));
    ccHObject meshes(tr("meshes"));
    ccHObject polylines(tr("polylines"));
    ccHObject other(tr("other"));
    ccHObject otherSerializable(tr("serializable"));
    ccHObject::Container entitiesToDispatch;
    entitiesToDispatch.insert(entitiesToDispatch.begin(),
                              m_selectedEntities.begin(),
                              m_selectedEntities.end());
    ccHObject entitiesToSave;
    while (!entitiesToDispatch.empty()) {
        ccHObject* child = entitiesToDispatch.back();
        entitiesToDispatch.pop_back();

        if (child->isA(CV_TYPES::HIERARCHY_OBJECT)) {
            for (unsigned j = 0; j < child->getChildrenNumber(); ++j)
                entitiesToDispatch.push_back(child->getChild(j));
        } else {
            // we put the entity in the container corresponding to its type
            ccHObject* dest = nullptr;
            if (child->isA(CV_TYPES::POINT_CLOUD))
                dest = &clouds;
            else if (child->isKindOf(CV_TYPES::MESH))
                dest = &meshes;
            else if (child->isKindOf(CV_TYPES::POLY_LINE))
                dest = &polylines;
            else if (child->isSerializable())
                dest = &otherSerializable;
            else
                dest = &other;

            assert(dest);

            // we don't want double insertions if the user has clicked both the
            // father and child
            if (!dest->find(child->getUniqueID())) {
                dest->addChild(child, ccHObject::DP_NONE);
                entitiesToSave.addChild(child, ccHObject::DP_NONE);
            }
        }
    }

    bool hasCloud = (clouds.getChildrenNumber() != 0);
    bool hasMesh = (meshes.getChildrenNumber() != 0);
    bool hasPolylines = (polylines.getChildrenNumber() != 0);
    bool hasSerializable = (otherSerializable.getChildrenNumber() != 0);
    bool hasOther = (other.getChildrenNumber() != 0);

    int stdSaveTypes = static_cast<int>(hasCloud) + static_cast<int>(hasMesh) +
                       static_cast<int>(hasPolylines) +
                       static_cast<int>(hasSerializable);
    if (stdSaveTypes == 0) {
        ecvConsole::Error(tr("Can't save selected entity(ies) this way!"));
        return;
    }

    // we set up the right file filters, depending on the selected
    // entities type (cloud, mesh, etc.).
    QStringList fileFilters;
    {
        for (const FileIOFilter::Shared& filter : FileIOFilter::GetFilters()) {
            bool atLeastOneExclusive = false;

            // does this filter can export one or several clouds?
            bool canExportClouds = true;
            if (hasCloud) {
                bool isExclusive = true;
                bool multiple = false;
                canExportClouds =
                        (filter->canSave(CV_TYPES::POINT_CLOUD, multiple,
                                         isExclusive) &&
                         (multiple || clouds.getChildrenNumber() == 1));
                atLeastOneExclusive |= isExclusive;
            }

            // does this filter can export one or several meshes?
            bool canExportMeshes = true;
            if (hasMesh) {
                bool isExclusive = true;
                bool multiple = false;
                canExportMeshes =
                        (filter->canSave(CV_TYPES::MESH, multiple,
                                         isExclusive) &&
                         (multiple || meshes.getChildrenNumber() == 1));
                atLeastOneExclusive |= isExclusive;
            }

            // does this filter can export one or several polylines?
            bool canExportPolylines = true;
            if (hasPolylines) {
                bool isExclusive = true;
                bool multiple = false;
                canExportPolylines =
                        (filter->canSave(CV_TYPES::POLY_LINE, multiple,
                                         isExclusive) &&
                         (multiple || polylines.getChildrenNumber() == 1));
                atLeastOneExclusive |= isExclusive;
            }

            // does this filter can export one or several images?
            bool canExportImages = true;

            // does this filter can export one or several other serializable
            // entities?
            bool canExportSerializables = true;
            if (hasSerializable) {
                // check if all entities have the same type
                {
                    CV_CLASS_ENUM firstClassID =
                            otherSerializable.getChild(0)->getUniqueID();
                    for (unsigned j = 1;
                         j < otherSerializable.getChildrenNumber(); ++j) {
                        if (otherSerializable.getChild(j)->getUniqueID() !=
                            firstClassID) {
                            // we add a virtual second 'stdSaveType' so as to
                            // properly handle exlusivity
                            ++stdSaveTypes;
                            break;
                        }
                    }
                }

                for (unsigned j = 0; j < otherSerializable.getChildrenNumber();
                     ++j) {
                    ccHObject* child = otherSerializable.getChild(j);
                    bool isExclusive = true;
                    bool multiple = false;
                    canExportSerializables &=
                            (filter->canSave(child->getClassID(), multiple,
                                             isExclusive) &&
                             (multiple ||
                              otherSerializable.getChildrenNumber() == 1));
                    atLeastOneExclusive |= isExclusive;
                }
            }

            bool useThisFilter = canExportClouds && canExportMeshes &&
                                 canExportImages && canExportPolylines &&
                                 canExportSerializables &&
                                 (!atLeastOneExclusive || stdSaveTypes == 1);

            if (useThisFilter) {
                QStringList ff = filter->getFileFilters(false);
                for (int j = 0; j < ff.size(); ++j) fileFilters.append(ff[j]);
            }
        }
    }

    // persistent settings
    // default filter
    QString selectedFilter = fileFilters.first();
    if (hasCloud)
        selectedFilter =
                ecvSettingManager::getValue(ecvPS::SaveFile(),
                                            ecvPS::SelectedOutputFilterCloud(),
                                            selectedFilter)
                        .toString();
    else if (hasMesh)
        selectedFilter =
                ecvSettingManager::getValue(ecvPS::SaveFile(),
                                            ecvPS::SelectedOutputFilterMesh(),
                                            selectedFilter)
                        .toString();
    else if (hasPolylines)
        selectedFilter =
                ecvSettingManager::getValue(ecvPS::SaveFile(),
                                            ecvPS::SelectedOutputFilterPoly(),
                                            selectedFilter)
                        .toString();

    // default output path (+ filename)
    QString currentPath =
            ecvSettingManager::getValue(ecvPS::SaveFile(), ecvPS::CurrentPath(),
                                        ecvFileUtils::defaultDocPath())
                    .toString();
    QString fullPathName = currentPath;

    if (haveOneSelection()) {
        // hierarchy objects have generally as name: 'filename.ext (fullpath)'
        // so we must only take the first part! (otherwise this type of name
        // with a path inside perturbs the QFileDialog a lot ;))
        QString defaultFileName(m_selectedEntities.front()->getName());
        if (m_selectedEntities.front()->isA(CV_TYPES::HIERARCHY_OBJECT)) {
            QStringList parts =
                    defaultFileName.split(' ', QtCompat::SkipEmptyParts);
            if (!parts.empty()) {
                defaultFileName = parts[0];
            }
        }

        // we remove the extension
        defaultFileName = QFileInfo(defaultFileName).baseName();

        if (!IsValidFileName(defaultFileName)) {
            ecvConsole::Warning(
                    tr("[I/O] First entity's name would make an invalid "
                       "filename! Can't use it..."));
            defaultFileName = tr("project");
        }

        fullPathName += QString("/") + defaultFileName;
    }

    // ask the user for the output filename
    QString selectedFilename = QFileDialog::getSaveFileName(
            this, tr("Save file"), fullPathName,
            fileFilters.join(s_fileFilterSeparator), &selectedFilter,
            ECVFileDialogOptions());

    if (selectedFilename.isEmpty()) {
        // process cancelled by the user
        return;
    }

    // ignored items
    if (hasOther) {
        ecvConsole::Warning(
                tr("[I/O] The following selected entities won't be saved:"));
        for (unsigned i = 0; i < other.getChildrenNumber(); ++i) {
            ecvConsole::Warning(
                    tr("\t- %1s").arg(other.getChild(i)->getName()));
        }
    }

    CC_FILE_ERROR result = CC_FERR_NO_ERROR;
    FileIOFilter::SaveParameters parameters;
    {
        parameters.alwaysDisplaySaveDialog = true;
        parameters.parentWidget = this;
    }

    // specific case: BIN format
    if (selectedFilter == BinFilter::GetFileFilter()) {
        if (haveOneSelection()) {
            result = FileIOFilter::SaveToFile(m_selectedEntities.front(),
                                              selectedFilename, parameters,
                                              selectedFilter);
        } else {
            // we'll regroup all selected entities in a temporary group
            ccHObject tempContainer;
            ConvertToGroup(m_selectedEntities, tempContainer,
                           ccHObject::DP_NONE);
            if (tempContainer.getChildrenNumber()) {
                result = FileIOFilter::SaveToFile(&tempContainer,
                                                  selectedFilename, parameters,
                                                  selectedFilter);
            } else {
                ecvConsole::Warning(
                        tr("[I/O] None of the selected entities can be saved "
                           "this way..."));
                result = CC_FERR_NO_SAVE;
            }
        }

        // display the compatible version info for BIN files
        if (result == CC_FERR_NO_ERROR) {
            short fileVersion = BinFilter::GetLastSavedFileVersion();
            if (fileVersion != 0) {
                QString minVersion =
                        ecvApplication::GetMinVersionForFileVersion(
                                fileVersion);
                CVLog::Print(tr("This file can be loaded by ACloudViewer "
                                "version %1 and later")
                                     .arg(minVersion));
            }
        }
    } else if (entitiesToSave.getChildrenNumber() != 0) {
        result = FileIOFilter::SaveToFile(entitiesToSave.getChildrenNumber() > 1
                                                  ? &entitiesToSave
                                                  : entitiesToSave.getChild(0),
                                          selectedFilename, parameters,
                                          selectedFilter);

        if (result == CC_FERR_NO_ERROR && m_ccRoot) {
            m_ccRoot->unselectAllEntities();
        }
    }

    // update default filters
    if (hasCloud)
        ecvSettingManager::setValue(ecvPS::SaveFile(),
                                    ecvPS::SelectedOutputFilterCloud(),
                                    selectedFilter);
    if (hasMesh)
        ecvSettingManager::setValue(ecvPS::SaveFile(),
                                    ecvPS::SelectedOutputFilterMesh(),
                                    selectedFilter);
    if (hasPolylines)
        ecvSettingManager::setValue(ecvPS::SaveFile(),
                                    ecvPS::SelectedOutputFilterPoly(),
                                    selectedFilter);

    // we update current file path
    currentPath = QFileInfo(selectedFilename).absolutePath();
    ecvSettingManager::setValue(ecvPS::SaveFile(), ecvPS::CurrentPath(),
                                currentPath);
}

void MainWindow::doActionSaveProject() {
    if (!m_ccRoot || !m_ccRoot->getRootEntity()) {
        assert(false);
        return;
    }

    ccHObject* rootEntity = m_ccRoot->getRootEntity();
    if (rootEntity->getChildrenNumber() == 0) {
        return;
    }

    // default output path (+ filename)
    QSettings settings;
    settings.beginGroup(ecvPS::SaveFile());
    QString currentPath =
            settings.value(ecvPS::CurrentPath(), ecvFileUtils::defaultDocPath())
                    .toString();
    CVLog::PrintDebug(currentPath);
    QString fullPathName = currentPath;

    static QString s_previousProjectName{"project"};
    QString defaultFileName = s_previousProjectName;
    if (rootEntity->getChildrenNumber() == 1) {
        // If there's only on top entity, we can try to use its name as the
        // project name.
        ccHObject* topEntity = rootEntity->getChild(0);
        defaultFileName = topEntity->getName();
        if (topEntity->isA(CV_TYPES::HIERARCHY_OBJECT)) {
            // Hierarchy objects have generally as name: 'filename.ext
            // (fullpath)' so we must only take the first part! (otherwise this
            // type of name with a path inside disturbs the QFileDialog a lot
            // ;))
            QStringList parts =
                    defaultFileName.split(' ', QtCompat::SkipEmptyParts);
            if (!parts.empty()) {
                defaultFileName = parts[0];
            }
        }

        // we remove the extension
        defaultFileName = QFileInfo(defaultFileName).completeBaseName();

        if (!IsValidFileName(defaultFileName)) {
            CVLog::Warning(
                    tr("[I/O] Top entity's name would make an invalid "
                       "filename! Can't use it..."));
            defaultFileName = "project";
        }
    }
    fullPathName += QString("/") + defaultFileName;

    QString binFilter = BinFilter::GetFileFilter();

    // ask the user for the output filename
    QString selectedFilename = QFileDialog::getSaveFileName(
            this, tr("Save file"), fullPathName, binFilter, &binFilter,
            ECVFileDialogOptions());

    if (selectedFilename.isEmpty()) {
        // process cancelled by the user
        return;
    }

    FileIOFilter::SaveParameters parameters;
    {
        parameters.alwaysDisplaySaveDialog = true;
        parameters.parentWidget = this;
    }

    CC_FILE_ERROR result = FileIOFilter::SaveToFile(
            rootEntity->getChildrenNumber() == 1 ? rootEntity->getChild(0)
                                                 : rootEntity,
            selectedFilename, parameters, binFilter);

    // display the compatible version info for BIN files
    if (result == CC_FERR_NO_ERROR) {
        short fileVersion = BinFilter::GetLastSavedFileVersion();
        if (fileVersion != 0) {
            QString minVersion =
                    ecvApplication::GetMinVersionForFileVersion(fileVersion);
            CVLog::Print(tr("This file can be loaded by ACloudViewer version "
                            "%1 and later")
                                 .arg(minVersion));
        }
    }

    // we update the current 'save' path
    QFileInfo fi(selectedFilename);
    s_previousProjectName = fi.fileName();
    currentPath = fi.absolutePath();
    settings.setValue(ecvPS::CurrentPath(), currentPath);
    settings.endGroup();
}

void MainWindow::doActionApplyTransformation() {
    ccApplyTransformationDlg dlg(this);
    if (!dlg.exec()) return;

    ccGLMatrixd transMat = dlg.getTransformation();
    applyTransformation(transMat);
}

void MainWindow::applyTransformation(const ccGLMatrixd& mat) {
    // if the transformation is partly converted to global shift/scale
    bool updateGlobalShiftAndScale = false;
    double scaleChange = 1.0;
    CCVector3d shiftChange(0, 0, 0);
    ccGLMatrixd transMat = mat;

    // we must backup 'm_selectedEntities' as removeObjectTemporarilyFromDBTree
    // can modify it!
    ccHObject::Container selectedEntities = getSelectedEntities();

    bool firstCloud = true;

    for (ccHObject* entity : selectedEntities)  // warning, getSelectedEntites
                                                // may change during this loop!
    {
        // we don't test primitives (it's always ok while the 'vertices lock'
        // test would fail)
        if (!entity->isKindOf(CV_TYPES::PRIMITIVE)) {
            // specific test for locked vertices
            bool lockedVertices;
            ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(
                    entity, &lockedVertices);
            if (cloud) {
                if (lockedVertices) {
                    ecvUtils::DisplayLockedVerticesWarning(entity->getName(),
                                                           haveOneSelection());
                    continue;
                }

                if (firstCloud) {
                    // test if the translated cloud was already "too big"
                    //(in which case we won't bother the user about the fact
                    // that the transformed cloud will be too big...)
                    ccBBox localBBox = entity->getOwnBB();
                    CCVector3d Pl =
                            CCVector3d::fromArray(localBBox.minCorner().u);
                    double Dl = localBBox.getDiagNormd();

                    // the cloud was alright
                    if (!ecvGlobalShiftManager::NeedShift(Pl) &&
                        !ecvGlobalShiftManager::NeedRescale(Dl)) {
                        // test if the translated cloud is not "too big" (in
                        // local coordinate space)
                        ccBBox rotatedBox = entity->getOwnBB() * transMat;
                        double Dl2 = rotatedBox.getDiagNorm();
                        CCVector3d Pl2 =
                                CCVector3d::fromArray(rotatedBox.getCenter().u);

                        bool needShift = ecvGlobalShiftManager::NeedShift(Pl2);
                        bool needRescale =
                                ecvGlobalShiftManager::NeedRescale(Dl2);

                        if (needShift || needRescale) {
                            // existing shift information
                            CCVector3d globalShift = cloud->getGlobalShift();
                            double globalScale = cloud->getGlobalScale();

                            // we compute the transformation matrix in the
                            // global coordinate space
                            ccGLMatrixd globalTransMat = transMat;
                            globalTransMat.scale(1.0 / globalScale);
                            globalTransMat.setTranslation(
                                    globalTransMat.getTranslationAsVec3D() -
                                    globalShift);
                            // and we apply it to the cloud bounding-box
                            ccBBox rotatedBox =
                                    cloud->getOwnBB() * globalTransMat;
                            double Dg = rotatedBox.getDiagNorm();
                            CCVector3d Pg = CCVector3d::fromArray(
                                    rotatedBox.getCenter().u);

                            // ask the user the right values!
                            ecvShiftAndScaleCloudDlg sasDlg(Pl2, Dl2, Pg, Dg,
                                                            this);
                            sasDlg.showApplyAllButton(false);
                            sasDlg.showTitle(true);
                            sasDlg.setKeepGlobalPos(true);
                            sasDlg.showKeepGlobalPosCheckbox(
                                    false);  // we don't want the user to mess
                                             // with this!
                            sasDlg.showPreserveShiftOnSave(true);

                            // add "original" entry
                            int index = sasDlg.addShiftInfo(
                                    ecvGlobalShiftManager::ShiftInfo(
                                            tr("Original"), globalShift,
                                            globalScale));
                            // sasDlg.setCurrentProfile(index);
                            // add "suggested" entry
                            CCVector3d suggestedShift =
                                    ecvGlobalShiftManager::BestShift(Pg);
                            double suggestedScale =
                                    ecvGlobalShiftManager::BestScale(Dg);
                            index = sasDlg.addShiftInfo(
                                    ecvGlobalShiftManager::ShiftInfo(
                                            tr("Suggested"), suggestedShift,
                                            suggestedScale));
                            sasDlg.setCurrentProfile(index);
                            // add "last" entry (if available)
                            std::vector<ecvGlobalShiftManager::ShiftInfo>
                                    lastInfos;
                            if (ecvGlobalShiftManager::GetLast(lastInfos)) {
                                sasDlg.addShiftInfo(lastInfos);
                            }
                            // add entries from file (if any)
                            sasDlg.addFileInfo();

                            if (sasDlg.exec()) {
                                // get the relative modification to existing
                                // global shift/scale info
                                assert(cloud->getGlobalScale() != 0);
                                scaleChange = sasDlg.getScale() /
                                              cloud->getGlobalScale();
                                shiftChange = (sasDlg.getShift() -
                                               cloud->getGlobalShift());

                                updateGlobalShiftAndScale =
                                        (scaleChange != 1.0 ||
                                         shiftChange.norm2() != 0);

                                // update transformation matrix accordingly
                                if (updateGlobalShiftAndScale) {
                                    transMat.scale(scaleChange);
                                    transMat.setTranslation(
                                            transMat.getTranslationAsVec3D() +
                                            shiftChange * scaleChange);
                                }
                            } else if (sasDlg.cancelled()) {
                                CVLog::Warning(
                                        tr("[ApplyTransformation] Process "
                                           "cancelled by user"));
                                return;
                            }
                        }
                    }

                    firstCloud = false;
                }

                if (updateGlobalShiftAndScale) {
                    // apply translation as global shift
                    cloud->setGlobalShift(cloud->getGlobalShift() +
                                          shiftChange);
                    cloud->setGlobalScale(cloud->getGlobalScale() *
                                          scaleChange);
                    const CCVector3d& T = cloud->getGlobalShift();
                    double scale = cloud->getGlobalScale();
                    CVLog::Warning(
                            tr("[ApplyTransformation] Cloud '%1' global "
                               "shift/scale information has been updated: "
                               "shift = (%2,%3,%4) / scale = %5")
                                    .arg(cloud->getName())
                                    .arg(T.x)
                                    .arg(T.y)
                                    .arg(T.z)
                                    .arg(scale));
                }
            }
        }

        // we temporarily detach entity, as it may undergo
        //"severe" modifications (octree deletion, etc.) --> see
        // ccHObject::applyRigidTransformation
        ccHObjectContext objContext = removeObjectTemporarilyFromDBTree(entity);
        entity->setGLTransformation(ccGLMatrix(transMat.data()));
        // DGM FIXME: we only test the entity own bounding box (and we update
        // its shift & scale info) but we apply the transformation to all its
        // children?!
        entity->applyGLTransformation_recursive();
        // entity->prepareDisplayForRefresh_recursive();
        putObjectBackIntoDBTree(entity, objContext);
    }

    // reselect previously selected entities!
    if (m_ccRoot) m_ccRoot->selectEntities(selectedEntities);

    CVLog::Print(tr("[ApplyTransformation] Applied transformation matrix:"));
    CVLog::Print(transMat.toString(12, ' '));  // full precision
    CVLog::Print(
            tr("Hint: copy it (CTRL+C) and apply it - or its inverse - on any "
               "entity with the 'Edit > Apply transformation' tool"));

    refreshSelected();
}

typedef std::pair<ccHObject*, ccGenericPointCloud*> EntityCloudAssociation;
void MainWindow::doActionApplyScale() {
    ccScaleDlg dlg(this);
    if (!dlg.exec()) return;
    dlg.saveState();

    // save values for next time
    CCVector3d scales = dlg.getScales();
    bool keepInPlace = dlg.keepInPlace();
    bool rescaleGlobalShift = dlg.rescaleGlobalShift();

    // we must backup 'm_selectedEntities' as removeObjectTemporarilyFromDBTree
    // can modify it!
    ccHObject::Container selectedEntities = m_selectedEntities;

    // first check that all coordinates are kept 'small'
    std::vector<EntityCloudAssociation> candidates;
    {
        bool testBigCoordinates = true;
        // size_t processNum = 0;

        for (ccHObject* entity :
             selectedEntities)  // warning, getSelectedEntites may change during
                                // this loop!
        {
            bool lockedVertices;
            // try to get the underlying cloud (or the vertices set for a mesh)
            ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(
                    entity, &lockedVertices);
            // otherwise we can look if the selected entity is a polyline
            if (!cloud && entity->isA(CV_TYPES::POLY_LINE)) {
                cloud = dynamic_cast<ccGenericPointCloud*>(
                        static_cast<ccPolyline*>(entity)->getAssociatedCloud());
                if (!cloud || cloud->isAncestorOf(entity))
                    lockedVertices = true;
            }
            if (!cloud || !cloud->isKindOf(CV_TYPES::POINT_CLOUD)) {
                CVLog::Warning(
                        tr("[Apply scale] Entity '%1' can't be scaled this way")
                                .arg(entity->getName()));
                continue;
            }
            if (lockedVertices) {
                ecvUtils::DisplayLockedVerticesWarning(entity->getName(),
                                                       haveOneSelection());
                //++processNum;
                continue;
            }

            CCVector3 C(0, 0, 0);
            if (keepInPlace) C = cloud->getOwnBB().getCenter();

            // we must check that the resulting cloud coordinates are not too
            // big
            if (testBigCoordinates) {
                ccBBox bbox = cloud->getOwnBB();
                CCVector3 bbMin = bbox.minCorner();
                CCVector3 bbMax = bbox.maxCorner();

                double maxx = static_cast<double>(
                        std::max(std::abs(bbMin.x), std::abs(bbMax.x)));
                double maxy = static_cast<double>(
                        std::max(std::abs(bbMin.y), std::abs(bbMax.y)));
                double maxz = static_cast<double>(
                        std::max(std::abs(bbMin.z), std::abs(bbMax.z)));

                const double maxCoord =
                        ecvGlobalShiftManager::MaxCoordinateAbsValue();
                bool oldCoordsWereTooBig =
                        (maxx > maxCoord || maxy > maxCoord || maxz > maxCoord);

                if (!oldCoordsWereTooBig) {
                    maxx = static_cast<double>(std::max(
                            std::abs((bbMin.x - C.x) * scales.x + C.x),
                            std::abs((bbMax.x - C.x) * scales.x + C.x)));
                    maxy = static_cast<double>(std::max(
                            std::abs((bbMin.y - C.y) * scales.y + C.y),
                            std::abs((bbMax.y - C.y) * scales.y + C.y)));
                    maxz = static_cast<double>(std::max(
                            std::abs((bbMin.z - C.z) * scales.z + C.z),
                            std::abs((bbMax.z - C.z) * scales.z + C.z)));

                    bool newCoordsAreTooBig =
                            (maxx > maxCoord || maxy > maxCoord ||
                             maxz > maxCoord);

                    if (newCoordsAreTooBig) {
                        if (QMessageBox::question(
                                    this, tr("Big coordinates"),
                                    tr("Resutling coordinates will be too big "
                                       "(original precision may be lost!). "
                                       "Proceed anyway?"),
                                    QMessageBox::Yes,
                                    QMessageBox::No) == QMessageBox::Yes) {
                            // ok, we won't test anymore and proceed
                            testBigCoordinates = false;
                        } else {
                            // we stop the process
                            return;
                        }
                    }
                }
            }

            assert(cloud);
            ccHObject* parent = entity->getParent();
            if (parent && parent->isKindOf(CV_TYPES::MESH)) {
                candidates.emplace_back(parent, cloud);
            } else {
                candidates.emplace_back(entity, cloud);
            }
        }
    }

    if (candidates.empty()) {
        ecvConsole::Warning(
                tr("[Apply scale] No eligible entities (point clouds or "
                   "meshes) were selected!"));
        return;
    }

    // now do the real scaling work
    {
        for (auto& candidate : candidates) {
            ccHObject* ent = candidate.first;
            ccGenericPointCloud* cloud = candidate.second;

            CCVector3 C(0, 0, 0);
            if (keepInPlace) {
                C = cloud->getOwnBB().getCenter();
            }

            // we temporarily detach entity, as it may undergo
            //"severe" modifications (octree deletion, etc.) --> see
            // ccPointCloud::scale
            ccHObjectContext objContext =
                    removeObjectTemporarilyFromDBTree(cloud);

            cloud->scale(static_cast<PointCoordinateType>(scales.x),
                         static_cast<PointCoordinateType>(scales.y),
                         static_cast<PointCoordinateType>(scales.z), C);

            putObjectBackIntoDBTree(cloud, objContext);

            // don't forget the 'global shift'!
            // DGM: but not the global scale!
            if (rescaleGlobalShift) {
                const CCVector3d& shift = cloud->getGlobalShift();
                cloud->setGlobalShift(CCVector3d(shift.x * scales.x,
                                                 shift.y * scales.y,
                                                 shift.z * scales.z));
            }
        }
    }

    // reselect previously selected entities!
    if (m_ccRoot) m_ccRoot->selectEntities(selectedEntities);

    if (!keepInPlace) zoomOnSelectedEntities();

    resetSelectedBBox();
    ecvDisplayTools::SetRedrawRecursive(false);
    for (auto& candidate : candidates) {
        ccHObject* ent = candidate.first;
        if (ent && ent->isKindOf(CV_TYPES::MESH)) {
            ent->setRedrawFlagRecursive(true);
        }
        ccGenericPointCloud* cloud = candidate.second;
        if (cloud) {
            cloud->setRedrawFlagRecursive(true);
        }
    }
    refreshAll();
    updateUI();
}

void MainWindow::activateTranslateRotateMode() {
    if (!haveSelection()) return;

    if (!getActiveWindow()) return;

#ifdef USE_PCL_BACKEND
    PclTransformTool* pclTransTool =
            new PclTransformTool(ecvDisplayTools::GetVisualizer3D());
    if (!m_transTool) m_transTool = new ccGraphicalTransformationTool(this);
    if (m_transTool->getNumberOfValidEntities() != 0) {
        m_transTool->clear();
    }
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND

    if (!m_transTool->setTansformTool(pclTransTool) ||
        !m_transTool->linkWith(ecvDisplayTools::GetCurrentScreen())) {
        CVLog::Warning(
                "[MainWindow::activateTranslateRotateMode] Initialization "
                "failed!");
        return;
    }

    bool rejectedEntities = false;
    for (ccHObject* entity : getSelectedEntities()) {
        if (m_transTool->addEntity(entity)) {
            m_ccRoot->unselectEntity(entity);
        } else {
            rejectedEntities = true;
        }
    }

    if (m_transTool->getNumberOfValidEntities() == 0) {
        ecvConsole::Error(tr(
                "No entity eligible for manual transformation! (see console)"));
        return;
    } else if (rejectedEntities) {
        ecvConsole::Error(tr("Some entities were ingored! (see console)"));
    }

    // try to activate "moving mode" in current GL window
    if (m_transTool->start()) {
        connect(m_transTool, &ccOverlayDialog::processFinished, this,
                &MainWindow::deactivateTranslateRotateMode);
        registerOverlayDialog(m_transTool, Qt::TopRightCorner);
        freezeUI(true);
        updateOverlayDialogsPlacement();
    } else {
        ecvConsole::Error(tr("Unexpected error!"));  // indeed...
    }
}

void MainWindow::deactivateTranslateRotateMode(bool state) {
    if (m_transTool) {
        // reselect previously selected entities!
        if (state && m_ccRoot) {
            const ccHObject& transformedSet = m_transTool->getValidEntities();
            try {
                ccHObject::Container transformedEntities;
                transformedEntities.resize(transformedSet.getChildrenNumber());
                for (unsigned i = 0; i < transformedSet.getChildrenNumber();
                     ++i) {
                    transformedEntities[i] = transformedSet.getChild(i);
                }
                m_ccRoot->selectEntities(transformedEntities);
            } catch (const std::bad_alloc&) {
                // not enough memory (nothing to do)
            }
        }
        // m_transTool->close();
    }

    freezeUI(false);

    updateUI();
}

void MainWindow::clearAll() {
    if (!m_ccRoot) return;

    if (QMessageBox::question(
                this, tr("Close all"),
                tr("Are you sure you want to remove all loaded entities?"),
                QMessageBox::Yes, QMessageBox::No) != QMessageBox::Yes)
        return;

    m_ccRoot->unloadAll();
}

void MainWindow::enableAll() {
    for (QMdiSubWindow* window : m_mdiArea->subWindowList()) {
        window->setEnabled(true);
    }
}

void MainWindow::disableAll() {
    for (QMdiSubWindow* window : m_mdiArea->subWindowList()) {
        window->setEnabled(false);
    }
}

void MainWindow::updateUIWithSelection() {
    dbTreeSelectionInfo selInfo;

    m_selectedEntities.clear();

    if (m_ccRoot) {
        m_ccRoot->getSelectedEntities(m_selectedEntities, CV_TYPES::OBJECT,
                                      &selInfo);
    }

    enableUIItems(selInfo);
}

void MainWindow::enableUIItems(dbTreeSelectionInfo& selInfo) {
    bool dbIsEmpty = (!m_ccRoot || !m_ccRoot->getRootEntity() ||
                      m_ccRoot->getRootEntity()->getChildrenNumber() == 0);
    bool atLeastOneEntity = (selInfo.selCount > 0);
    bool atLeastOneCloud = (selInfo.cloudCount > 0);
    bool atLeastOneMesh = (selInfo.meshCount > 0);
    bool atLeastOneOctree = (selInfo.octreeCount > 0);
    bool atLeastOneNormal = (selInfo.normalsCount > 0);
    bool atLeastOneColor = (selInfo.colorCount > 0);
    bool atLeastOneSF = (selInfo.sfCount > 0);
    bool atLeastOneGrid = (selInfo.gridCound > 0);

    bool atLeastOneSensor = (selInfo.sensorCount > 0);
    bool atLeastOneGBLSensor = (selInfo.gblSensorCount > 0);
    bool atLeastOneCameraSensor = (selInfo.cameraSensorCount > 0);
    bool atLeastOnePolyline = (selInfo.polylineCount > 0);

    m_ui->actionTracePolyline->setEnabled(!dbIsEmpty);
    m_ui->actionZoomAndCenter->setEnabled(atLeastOneEntity);
    m_ui->actionSave->setEnabled(atLeastOneEntity);
    m_ui->actionSaveProject->setEnabled(!dbIsEmpty);
    m_ui->actionClone->setEnabled(atLeastOneEntity);
    m_ui->actionDelete->setEnabled(atLeastOneEntity);
    m_ui->actionImportSFFromFile->setEnabled(atLeastOneEntity);
    m_ui->actionExportCoordToSF->setEnabled(atLeastOneEntity);
    m_ui->actionSegment->setEnabled(atLeastOneEntity);
    m_ui->actionContourWidget->setEnabled(atLeastOneEntity);
    m_ui->actionDistanceWidget->setEnabled(atLeastOneEntity);
    m_ui->actionProtractorWidget->setEnabled(atLeastOneEntity);
    m_ui->actionTranslateRotate->setEnabled(atLeastOneEntity);
    m_ui->actionShowDepthBuffer->setEnabled(atLeastOneGBLSensor);
    m_ui->actionExportDepthBuffer->setEnabled(atLeastOneGBLSensor);
    m_ui->actionComputePointsVisibility->setEnabled(atLeastOneGBLSensor);
    m_ui->actionResampleWithOctree->setEnabled(atLeastOneCloud);
    m_ui->actionApplyScale->setEnabled(atLeastOneCloud || atLeastOneMesh ||
                                       atLeastOnePolyline);
    m_ui->actionApplyTransformation->setEnabled(atLeastOneEntity);
    m_ui->actionComputeOctree->setEnabled(atLeastOneCloud || atLeastOneMesh);
    m_ui->actionComputeNormals->setEnabled(atLeastOneCloud || atLeastOneMesh);
    m_ui->actionChangeColorLevels->setEnabled(atLeastOneCloud ||
                                              atLeastOneMesh);
    m_ui->actionEditGlobalShiftAndScale->setEnabled(
            atLeastOneCloud || atLeastOneMesh || atLeastOnePolyline);
    m_ui->actionSetUniqueColor->setEnabled(
            atLeastOneEntity /*atLeastOneCloud || atLeastOneMesh*/);  // DGM: we
                                                                      // can set
                                                                      // color
                                                                      // to a
                                                                      // group
                                                                      // now!
    m_ui->actionColorize->setEnabled(
            atLeastOneEntity /*atLeastOneCloud || atLeastOneMesh*/);  // DGM: we
                                                                      // can set
                                                                      // color
                                                                      // to a
                                                                      // group
                                                                      // now!
    // m_ui->actionDeleteScanGrid->setEnabled(atLeastOneGrid);

    m_ui->actionScalarFieldFromColor->setEnabled(atLeastOneEntity &&
                                                 atLeastOneColor);
    m_ui->actionComputeMeshAA->setEnabled(atLeastOneCloud);
    m_ui->actionComputeMeshLS->setEnabled(atLeastOneCloud);
    m_ui->actionConvexHull->setEnabled(atLeastOneCloud);
    m_ui->actionPoissonReconstruction->setEnabled(atLeastOneCloud);
    m_ui->actionMeshScanGrids->setEnabled(atLeastOneGrid);
    // actionComputeQuadric3D->setEnabled(atLeastOneCloud);
    m_ui->actionComputeBestFitBB->setEnabled(atLeastOneEntity);
    m_ui->actionComputeGeometricFeature->setEnabled(atLeastOneCloud);
    m_ui->actionRemoveDuplicatePoints->setEnabled(atLeastOneCloud);
    m_ui->actionFitPlane->setEnabled(atLeastOneEntity);
    m_ui->actionFitPlaneProxy->setEnabled(atLeastOneEntity);
    m_ui->actionFitSphere->setEnabled(atLeastOneCloud);
    m_ui->actionFitCircle->setEnabled(atLeastOneCloud);
    //    m_ui->actionLevel->setEnabled(atLeastOneEntity);
    m_ui->actionFitFacet->setEnabled(atLeastOneEntity);
    m_ui->actionFitQuadric->setEnabled(atLeastOneCloud);
    m_ui->actionSubsample->setEnabled(atLeastOneCloud);

    m_ui->actionSNETest->setEnabled(atLeastOneCloud);
    m_ui->actionExportCloudInfo->setEnabled(atLeastOneEntity);
    m_ui->actionExportPlaneInfo->setEnabled(atLeastOneEntity);

    m_ui->actionFilterByLabel->setEnabled(atLeastOneSF);
    m_ui->actionFilterByValue->setEnabled(atLeastOneSF);
    m_ui->actionConvertToRGB->setEnabled(atLeastOneSF);
    m_ui->actionConvertToRandomRGB->setEnabled(atLeastOneSF);
    m_ui->actionRenameSF->setEnabled(atLeastOneSF);
    m_ui->actionAddIdField->setEnabled(atLeastOneCloud);
    m_ui->actionComputeStatParams->setEnabled(atLeastOneSF);
    // m_ui->actionComputeStatParams2->setEnabled(atLeastOneSF);
    m_ui->actionShowHistogram->setEnabled(atLeastOneSF);
    m_ui->actionGaussianFilter->setEnabled(atLeastOneSF);
    m_ui->actionBilateralFilter->setEnabled(atLeastOneSF);
    m_ui->actionDeleteScalarField->setEnabled(atLeastOneSF);
    m_ui->actionDeleteAllSF->setEnabled(atLeastOneSF);
    // m_ui->actionMultiplySF->setEnabled(/*TODO: atLeastOneSF*/false);
    m_ui->actionSFGradient->setEnabled(atLeastOneSF);
    m_ui->actionSetSFAsCoord->setEnabled(atLeastOneSF && atLeastOneCloud);
    m_ui->actionInterpolateSFs->setEnabled(atLeastOneCloud || atLeastOneMesh);

    m_ui->actionSamplePointsOnMesh->setEnabled(atLeastOneMesh);
    m_ui->actionMeasureMeshSurface->setEnabled(atLeastOneMesh);
    m_ui->actionMeasureMeshVolume->setEnabled(atLeastOneMesh);
    m_ui->actionFlagMeshVertices->setEnabled(atLeastOneMesh);
    m_ui->actionSmoothMeshLaplacian->setEnabled(atLeastOneMesh);
    m_ui->actionConvertTextureToColor->setEnabled(atLeastOneMesh);
    m_ui->actionSubdivideMesh->setEnabled(atLeastOneMesh);
    m_ui->actionDistanceToBestFitQuadric3D->setEnabled(atLeastOneCloud);
    m_ui->actionDistanceMap->setEnabled(atLeastOneMesh || atLeastOneCloud);

    m_ui->menuMeshScalarField->setEnabled(atLeastOneSF && atLeastOneMesh);
    // actionSmoothMeshSF->setEnabled(atLeastOneSF && atLeastOneMesh);
    // actionEnhanceMeshSF->setEnabled(atLeastOneSF && atLeastOneMesh);

    m_ui->actionOrientNormalsMST->setEnabled(atLeastOneCloud &&
                                             atLeastOneNormal);
    m_ui->actionOrientNormalsFM->setEnabled(atLeastOneCloud &&
                                            atLeastOneNormal);
    m_ui->actionClearNormals->setEnabled(atLeastOneNormal);
    m_ui->actionInvertNormals->setEnabled(atLeastOneNormal);
    m_ui->actionConvertNormalToHSV->setEnabled(atLeastOneNormal);
    m_ui->actionConvertNormalToDipDir->setEnabled(atLeastOneNormal);
    m_ui->actionExportNormalToSF->setEnabled(atLeastOneNormal);
    m_ui->actionClearColor->setEnabled(atLeastOneColor);
    m_ui->actionRGBToGreyScale->setEnabled(atLeastOneColor);
    m_ui->actionEnhanceRGBWithIntensities->setEnabled(atLeastOneColor);
    m_ui->actionRGBGaussianFilter->setEnabled(atLeastOneColor);
    m_ui->actionRGBBilateralFilter->setEnabled(atLeastOneColor);
    m_ui->actionRGBMeanFilter->setEnabled(atLeastOneColor);
    m_ui->actionRGBMedianFilter->setEnabled(atLeastOneColor);
    m_ui->actionColorFromScalarField->setEnabled(atLeastOneSF);

    // == 1
    bool exactlyOneEntity = (selInfo.selCount == 1);
    bool exactlyOneGroup = (selInfo.groupCount == 1);
    bool exactlyOneCloud = (selInfo.cloudCount == 1);
    bool exactlyOneMesh = (selInfo.meshCount == 1);
    bool exactlyOneSF = (selInfo.sfCount == 1);
    bool exactlyOneSensor = (selInfo.sensorCount == 1);
    bool exactlyOneCameraSensor = (selInfo.cameraSensorCount == 1);

    m_ui->actionSliceFilter->setEnabled(exactlyOneMesh);
    m_ui->actionDecimateFilter->setEnabled(exactlyOneMesh);
    m_ui->actionIsoSurfaceFilter->setEnabled(exactlyOneMesh);
    m_ui->actionSmoothFilter->setEnabled(exactlyOneMesh);

    m_ui->actionConvertPolylinesToMesh->setEnabled(atLeastOnePolyline ||
                                                   exactlyOneGroup);
    m_ui->actionSamplePointsOnPolyline->setEnabled(atLeastOnePolyline);
    m_ui->actionSmoothPolyline->setEnabled(atLeastOnePolyline);
    m_ui->actionMeshTwoPolylines->setEnabled(selInfo.selCount == 2 &&
                                             selInfo.polylineCount == 2);
    m_ui->actionModifySensor->setEnabled(exactlyOneSensor);
    m_ui->actionComputeDistancesFromSensor->setEnabled(atLeastOneCameraSensor ||
                                                       atLeastOneGBLSensor);
    m_ui->actionComputeScatteringAngles->setEnabled(exactlyOneSensor);
    m_ui->actionViewFromSensor->setEnabled(exactlyOneSensor);
    m_ui->actionCreateGBLSensor->setEnabled(atLeastOneCloud);
    m_ui->actionCreateCameraSensor->setEnabled(selInfo.selCount <=
                                               1);  // free now
    m_ui->actionProjectUncertainty->setEnabled(exactlyOneCameraSensor);
    m_ui->actionCheckPointsInsideFrustum->setEnabled(exactlyOneCameraSensor);
    m_ui->actionLabelConnectedComponents->setEnabled(atLeastOneCloud);
    m_ui->actionSORFilter->setEnabled(atLeastOneCloud);
    m_ui->actionNoiseFilter->setEnabled(atLeastOneCloud);
    m_ui->actionVoxelSampling->setEnabled(atLeastOneCloud);
    m_ui->actionUnroll->setEnabled(exactlyOneEntity);
    //    m_ui->actionStatisticalTest->setEnabled(exactlyOneEntity &&
    //    exactlyOneSF);
    m_ui->actionAddConstantSF->setEnabled(exactlyOneCloud || exactlyOneMesh);
    //    m_ui->actionEditGlobalScale->setEnabled(exactlyOneCloud ||
    //    exactlyOneMesh);
    m_ui->actionComputeKdTree->setEnabled(exactlyOneCloud || exactlyOneMesh);
    m_ui->actionShowWaveDialog->setEnabled(exactlyOneCloud);
    m_ui->actionCompressFWFData->setEnabled(atLeastOneCloud);

    m_ui->actionKMeans->setEnabled(
            /*TODO: exactlyOneEntity && exactlyOneSF*/ false);
    m_ui->actionFrontPropagation->setEnabled(
            /*TODO: exactlyOneEntity && exactlyOneSF*/ false);

    // actionCreatePlane->setEnabled(true);
    m_ui->actionEditPlane->setEnabled(selInfo.planeCount == 1);
    m_ui->actionFlipPlane->setEnabled(selInfo.planeCount != 0);
    m_ui->actionComparePlanes->setEnabled(selInfo.planeCount == 2);

    m_ui->actionPromoteCircleToCylinder->setEnabled((selInfo.selCount == 1) &&
                                                    (selInfo.circleCount == 1));

    m_ui->actionFindBiggestInnerRectangle->setEnabled(exactlyOneCloud);

    //	m_ui->menuActiveScalarField->setEnabled((exactlyOneCloud ||
    // exactlyOneMesh) && selInfo.sfCount > 0);
    m_ui->actionClipFilter->setEnabled(atLeastOneCloud || atLeastOneMesh ||
                                       (selInfo.groupCount != 0));
    m_ui->actionProbeFilter->setEnabled(atLeastOneCloud || atLeastOneMesh ||
                                        (selInfo.groupCount != 0));
    m_ui->actionGlyphFilter->setEnabled(atLeastOneCloud || atLeastOneMesh ||
                                        (selInfo.groupCount != 0));
    m_ui->actionStreamlineFilter->setEnabled(
            atLeastOneCloud || atLeastOneMesh || (selInfo.groupCount != 0));
    m_ui->actionThresholdFilter->setEnabled(atLeastOneCloud || atLeastOneMesh ||
                                            (selInfo.groupCount != 0));

#ifdef USE_PYTHON_MODULE
    m_ui->actionSemanticSegmentation->setEnabled(atLeastOneCloud);
#endif  // USE_PYTHON_MODULE

    m_ui->actionDBScanCluster->setEnabled(atLeastOneCloud);
    m_ui->actionPlaneSegmentation->setEnabled(atLeastOneCloud);
    //	m_ui->actionExtractSections->setEnabled(atLeastOneCloud);
    // m_ui->actionRasterize->setEnabled(exactlyOneCloud);
    m_ui->actionBoxAnnotation->setEnabled(exactlyOneCloud);
    m_ui->actionSemanticAnnotation->setEnabled(exactlyOneCloud);

    m_ui->actionCompute2HalfDimVolume->setEnabled(
            selInfo.cloudCount == selInfo.selCount && selInfo.cloudCount >= 1 &&
            selInfo.cloudCount <= 2);  // one or two clouds!
    m_ui->actionPointListPicking->setEnabled(exactlyOneEntity);

    // == 2
    bool exactlyTwoEntities = (selInfo.selCount == 2);
    bool exactlyTwoClouds = (selInfo.cloudCount == 2);
    // bool exactlyTwoSF = (selInfo.sfCount == 2);

    m_ui->actionRegister->setEnabled(exactlyTwoEntities);
    m_ui->actionInterpolateColors->setEnabled(exactlyTwoEntities &&
                                              atLeastOneColor);
    m_ui->actionPointPairsAlign->setEnabled(atLeastOneEntity);
    m_ui->actionBBCenterToOrigin->setEnabled(atLeastOneEntity);
    m_ui->actionBBMinCornerToOrigin->setEnabled(atLeastOneEntity);
    m_ui->actionBBMaxCornerToOrigin->setEnabled(atLeastOneEntity);
    m_ui->actionAlign->setEnabled(
            exactlyTwoEntities);  // Aurelien BEY le 13/11/2008
    m_ui->actionCloudCloudDist->setEnabled(exactlyTwoClouds);
    m_ui->actionCloudMeshDist->setEnabled(exactlyTwoEntities && atLeastOneMesh);
    m_ui->actionCloudPrimitiveDist->setEnabled(
            atLeastOneCloud && (atLeastOneMesh || atLeastOnePolyline));
    m_ui->actionCPS->setEnabled(exactlyTwoClouds);
    m_ui->actionScalarFieldArithmetic->setEnabled(exactlyOneEntity &&
                                                  atLeastOneSF);

    //>1
    bool atLeastTwoEntities = (selInfo.selCount > 1);

    m_ui->actionMerge->setEnabled(atLeastTwoEntities);
    m_ui->actionMatchBBCenters->setEnabled(atLeastTwoEntities);
    m_ui->actionMatchScales->setEnabled(atLeastTwoEntities);

    // standard plugins
    m_pluginUIManager->handleSelectionChanged();
}

void MainWindow::moveEvent(QMoveEvent* event) {
    QMainWindow::moveEvent(event);

    updateOverlayDialogsPlacement();
}

void MainWindow::resizeEvent(QResizeEvent* event) {
    QMainWindow::resizeEvent(event);

    updateOverlayDialogsPlacement();
    updateMemoryUsageWidgetSize();
}

bool MainWindow::eventFilter(QObject* obj, QEvent* event) {
    switch (event->type()) {
        case QEvent::Resize:
        case QEvent::Move:
            updateOverlayDialogsPlacement();
            break;
        case QEvent::KeyPress: {
            // Handle ESC key globally to exit selection tools
            // This is needed because VTK render window captures key events
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
            if (keyEvent->key() == Qt::Key_Escape) {
                CVLog::PrintDebug(
                        "[MainWindow::eventFilter] ESC key detected, calling "
                        "handleEscapeKey");
                // Handle ESC key the same way as keyPressEvent
                handleEscapeKey();
                return true;  // Event handled
            }
            break;
        }
        default:
            // nothing to do
            break;
    }

    // standard event processing
    return QObject::eventFilter(obj, event);
}

void MainWindow::handleEscapeKey() {
    // First, stop any active measurement tool and uncheck its button
    if (m_measurementTool && m_measurementTool->started()) {
        m_measurementTool->stop(false);

        // Uncheck measurement tool actions if they are checkable
        if (m_ui->actionDistanceWidget &&
            m_ui->actionDistanceWidget->isCheckable()) {
            m_ui->actionDistanceWidget->setChecked(false);
        }
        if (m_ui->actionProtractorWidget &&
            m_ui->actionProtractorWidget->isCheckable()) {
            m_ui->actionProtractorWidget->setChecked(false);
        }
        if (m_ui->actionContourWidget &&
            m_ui->actionContourWidget->isCheckable()) {
            m_ui->actionContourWidget->setChecked(false);
        }
    }

    // Second, disable all active selection tools (SelectionTools
    // module) This ensures ESC exits selection modes like Rectangle
    // Select, Polygon Select, etc.
    CVLog::PrintDebug("[MainWindow] Disabling all selection tools");
    // Disable all selection tools via controller
    // The controller handles unchecking all actions
#ifdef USE_PCL_BACKEND
    if (m_selectionController) {
        m_selectionController->handleEscapeKey();
    }
#endif

    // Then handle picking and fullscreen
    cancelPreviousPickingOperation(true);

    // Handle exclusive fullscreen mode (when a sub-widget is fullscreen, not
    // MainWindow itself)
    if (m_exclusiveFullscreen) {
        toggleExclusiveFullScreen(false);
    }
    // Handle normal fullscreen mode (when MainWindow itself is fullscreen)
    else if (this->isFullScreen()) {
        this->showNormal();
    }
}

void MainWindow::keyPressEvent(QKeyEvent* event) {
    switch (event->key()) {
        case Qt::Key_Escape:
            CVLog::Print(
                    "[MainWindow::keyPressEvent] ESC key received, calling "
                    "handleEscapeKey");
            handleEscapeKey();
            break;
        default:
            QMainWindow::keyPressEvent(event);
    }
}

ccBBox MainWindow::getSelectedEntityBbox() {
    ccHObject tempGroup(QString("TempGroup"));
    size_t selNum = m_selectedEntities.size();
    for (size_t i = 0; i < selNum; ++i) {
        ccHObject* entity = m_selectedEntities[i];
        tempGroup.addChild(entity, ccHObject::DP_NONE);
    }

    ccBBox box;
    if (tempGroup.getChildrenNumber() != 0) {
        box = tempGroup.getDisplayBB_recursive(false);
    }
    return box;
}

void MainWindow::addEditPlaneAction(QMenu& menu) const {
    menu.addAction(m_ui->actionEditPlane);
}

void MainWindow::zoomOn(ccHObject* object) {
    if (ecvDisplayTools::GetCurrentScreen()) {
        ccBBox box = object->getDisplayBB_recursive(false);
        ecvDisplayTools::UpdateConstellationCenterAndZoom(&box);
    }
}

void MainWindow::setView(CC_VIEW_ORIENTATION view) {
    ccBBox* bbox = nullptr;
    ccBBox box = getSelectedEntityBbox();
    if (box.isValid()) {
        bbox = &box;
    }
    ecvDisplayTools::SetView(view, bbox);
}

void MainWindow::updateMenus() {
    QWidget* active3DView = getActiveWindow();
    bool hasMdiChild = (active3DView != nullptr);
    int mdiChildCount = getRenderWindowCount();
    bool hasLoadedEntities =
            (m_ccRoot && m_ccRoot->getRootEntity() &&
             m_ccRoot->getRootEntity()->getChildrenNumber() != 0);
    bool hasSelectedEntities =
            (m_ccRoot && m_ccRoot->countSelectedEntities() > 0);

    // General Menu
    m_ui->menuEdit->setEnabled(true /*hasSelectedEntities*/);
    m_ui->menuTools->setEnabled(true /*hasSelectedEntities*/);

    ////View Menu
    m_ui->ViewToolBar->setEnabled(hasMdiChild);

    ////oher actions
    m_ui->actionSegment->setEnabled(hasMdiChild && hasSelectedEntities);
    m_ui->actionContourWidget->setEnabled(hasMdiChild && hasSelectedEntities);
    m_ui->actionDistanceWidget->setEnabled(hasMdiChild && hasSelectedEntities);
    m_ui->actionProtractorWidget->setEnabled(hasMdiChild &&
                                             hasSelectedEntities);
    m_ui->actionTranslateRotate->setEnabled(hasMdiChild && hasSelectedEntities);
    m_ui->actionPointPicking->setEnabled(hasMdiChild && hasLoadedEntities);
    m_ui->actionPointListPicking->setEnabled(hasLoadedEntities);
    // m_ui->actionTestFrameRate->setEnabled(hasMdiChild);
    // m_ui->actionRenderToFile->setEnabled(hasMdiChild);
    // m_ui->actionToggleCenteredPerspective->setEnabled(hasMdiChild);
    // m_ui->actionToggleViewerBasedPerspective->setEnabled(hasMdiChild);

    // plugins
    m_pluginUIManager->updateMenus();
}

void MainWindow::putObjectBackIntoDBTree(ccHObject* obj,
                                         const ccHObjectContext& context) {
    assert(obj);
    if (!obj || !m_ccRoot) return;

    if (context.parent) {
        context.parent->addChild(obj, context.parentFlags);
        obj->addDependency(context.parent, context.childFlags);
    }

    // DGM: we must call 'notifyGeometryUpdate' as any call to this method
    // while the object was temporarily 'cut' from the DB tree were
    // ineffective!
    obj->notifyGeometryUpdate();

    m_ccRoot->addElement(obj, false);
}

MainWindow::ccHObjectContext MainWindow::removeObjectTemporarilyFromDBTree(
        ccHObject* obj) {
    ccHObjectContext context;

    assert(obj);
    if (!m_ccRoot || !obj) return context;

    // mandatory (to call putObjectBackIntoDBTree)
    context.parent = obj->getParent();

    // remove the object's dependency to its father (in case it undergoes
    // "severe" modifications)
    if (context.parent) {
        context.parentFlags = context.parent->getDependencyFlagsWith(obj);
        context.childFlags = obj->getDependencyFlagsWith(context.parent);

        context.parent->removeDependencyWith(obj);
        obj->removeDependencyWith(context.parent);
    }

    m_ccRoot->removeElement(obj);

    return context;
}

ccColorScalesManager* MainWindow::getColorScalesManager() {
    return ccColorScalesManager::GetUniqueInstance();
}

bool MainWindow::s_autoSaveGuiElementPos = true;
void MainWindow::doActionResetGUIElementsPos() {
    // show the user it will be maximized
    showMaximized();
    if (this->m_uiManager) {
        this->m_uiManager->showMaximized();
    }

    QSettings settings;
    settings.remove(ecvPS::MainWinGeom());
    settings.remove(ecvPS::MainWinState());

    QMessageBox::information(this, tr("Restart"),
                             tr("To finish the process, you'll have to close "
                                "and restart ACloudViewer"));

    // to avoid saving them right away!
    s_autoSaveGuiElementPos = false;
}

void MainWindow::doActionSaveCustomLayout() {
    if (m_layoutManager) {
        m_layoutManager->saveCustomLayout();
        QMessageBox::information(
                this, tr("Save Custom Layout"),
                tr("Current layout has been saved as custom layout. You can "
                   "restore it later using the 'Restore Custom Layout' "
                   "action."));
    } else {
        CVLog::Error("[MainWindow] Layout manager is not initialized!");
    }
}

void MainWindow::doActionRestoreDefaultLayout() {
    if (m_layoutManager) {
        m_layoutManager->restoreDefaultLayout();
    } else {
        CVLog::Error("[MainWindow] Layout manager is not initialized!");
    }
}

void MainWindow::doActionRestoreCustomLayout() {
    if (m_layoutManager) {
        if (!m_layoutManager->restoreCustomLayout()) {
            QMessageBox::warning(this, tr("Restore Custom Layout"),
                                 tr("No saved custom layout found. Please save "
                                    "current layout first."));
        }
    } else {
        CVLog::Error("[MainWindow] Layout manager is not initialized!");
    }
}

void MainWindow::doActionRestoreWindowOnStartup(bool state) {
    QSettings settings;
    settings.setValue(ecvPS::DoNotRestoreWindowGeometry(), !state);
}

void MainWindow::toggleFullScreen(bool state) {
    if (m_uiManager != nullptr) {
        m_uiManager->toggleFullScreen(state);
    } else {
        state ? showFullScreen() : showNormal();
    }

#ifdef Q_OS_MAC
    if (state) {
        m_ui->actionFullScreen->setText(tr("Exit Full Screen"));
    } else {
        m_ui->actionFullScreen->setText(tr("Enter Full Screen"));
    }
#endif

    m_ui->actionFullScreen->setChecked(state);
}

void MainWindow::toggleExclusiveFullScreen(bool state) {
    if (state) {
        // we are currently in normal screen mode
        if (!m_exclusiveFullscreen) {
            m_currentFullWidget = getActiveWindow();
            if (m_currentFullWidget) {
                m_formerGeometry = m_currentFullWidget->saveGeometry();
                m_currentFullWidget->setWindowFlags(Qt::Dialog);
                // Install event filter to capture ESC key in fullscreen mode
                m_currentFullWidget->installEventFilter(this);
            }

            m_exclusiveFullscreen = true;
            if (m_currentFullWidget) {
                m_currentFullWidget->showFullScreen();
                // Ensure the widget has keyboard focus to receive ESC key
                m_currentFullWidget->setFocus();
            } else {
                showFullScreen();
            }

            onExclusiveFullScreenToggled(state);
            ecvDisplayTools::DisplayNewMessage(
                    "Press F11 or ESC to disable full-screen mode",
                    ecvDisplayTools::UPPER_CENTER_MESSAGE, false, 30,
                    ecvDisplayTools::FULL_SCREEN_MESSAGE);
        }
    } else {
        // if we are currently in full-screen mode
        if (m_exclusiveFullscreen) {
            if (m_currentFullWidget) {
                m_currentFullWidget->setWindowFlags(Qt::SubWindow);
            }

            m_exclusiveFullscreen = false;
            onExclusiveFullScreenToggled(state);
            ecvDisplayTools::DisplayNewMessage(
                    QString(), ecvDisplayTools::UPPER_CENTER_MESSAGE, false, 0,
                    ecvDisplayTools::FULL_SCREEN_MESSAGE);  // remove any
                                                            // message

            if (m_currentFullWidget) {
                m_currentFullWidget->showNormal();
                if (!m_formerGeometry.isNull()) {
                    m_currentFullWidget->restoreGeometry(m_formerGeometry);
                    m_formerGeometry.clear();
                }
            } else {
                showNormal();
            }
        }
    }

    QCoreApplication::processEvents();
    if (m_currentFullWidget) {
        m_currentFullWidget->setFocus();
    }

    ecvDisplayTools::SetRedrawRecursive(false);
    ecvDisplayTools::RedrawDisplay(true, false);
}

void MainWindow::toggle3DView(bool state) {
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::Toggle2Dviewer(!state);
    }
}

void MainWindow::onExclusiveFullScreenToggled(bool state) {
    // we simply update the full-screen action method icon (whatever the window)
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::SetExclusiveFullScreenFlage(state);
        m_ui->actionExclusiveFullScreen->blockSignals(true);
        m_ui->actionExclusiveFullScreen->setChecked(state);
        m_ui->actionExclusiveFullScreen->blockSignals(false);
    }
}

void MainWindow::updateFullScreenMenu(bool state) {
    m_ui->actionFullScreen->setChecked(state);
}

void MainWindow::handleNewLabel(ccHObject* entity) {
    if (entity) {
        addToDB(entity);
    } else {
        assert(false);
    }
}

void MainWindow::activatePointListPickingMode() {
    // there should be only one point cloud in current selection!
    if (!haveOneSelection()) {
        ecvConsole::Error(tr("Select one and only one entity!"));
        return;
    }

    ccPointCloud* pc = ccHObjectCaster::ToPointCloud(m_selectedEntities[0]);
    if (!pc) {
        ecvConsole::Error(tr("Wrong type of entity"));
        return;
    }

    if (!pc->isVisible() || !pc->isEnabled()) {
        ecvConsole::Error(tr("Points must be visible!"));
        return;
    }

    if (!m_plpDlg) {
        m_plpDlg = new ccPointListPickingDlg(m_pickingHub, this);
        connect(m_plpDlg, &ccOverlayDialog::processFinished, this,
                &MainWindow::deactivatePointListPickingMode);

        registerOverlayDialog(m_plpDlg, Qt::TopRightCorner);
    }

    // DGM: we must update marker size spin box value (as it may have changed by
    // the user with the "display dialog")
    m_plpDlg->markerSizeSpinBox->setValue(
            ecvDisplayTools::GetDisplayParameters().labelMarkerSize);

    m_plpDlg->linkWith(ecvDisplayTools::GetCurrentScreen());
    m_plpDlg->linkWithCloud(pc);

    freezeUI(true);

    if (!m_plpDlg->start())
        deactivatePointListPickingMode(false);
    else
        updateOverlayDialogsPlacement();
}

void MainWindow::deactivatePointListPickingMode(bool state) {
    Q_UNUSED(state);
    if (m_plpDlg) {
        m_plpDlg->linkWithCloud(nullptr);
    }

    freezeUI(false);

    updateUI();
}

void MainWindow::activatePointPickingMode() {
    if (m_ccRoot) {
        m_ccRoot->unselectAllEntities();  // we don't want any entity selected
                                          // (especially existing labels!)
    }

    if (!m_ppDlg) {
        m_ppDlg = new ccPointPropertiesDlg(m_pickingHub, this);
        connect(m_ppDlg, &ccOverlayDialog::processFinished, this,
                &MainWindow::deactivatePointPickingMode);
        connect(m_ppDlg, &ccPointPropertiesDlg::newLabel, this,
                &MainWindow::handleNewLabel);

        registerOverlayDialog(m_ppDlg, Qt::TopRightCorner);
    }

    m_ppDlg->linkWith(ecvDisplayTools::GetCurrentScreen());

    freezeUI(true);

    if (!m_ppDlg->start())
        deactivatePointPickingMode(false);
    else
        updateOverlayDialogsPlacement();
}

void MainWindow::deactivatePointPickingMode(bool state) {
    Q_UNUSED(state);
    freezeUI(false);
    updateUI();
}

void MainWindow::activateTracePolylineMode() {
    if (!m_tplTool) {
        m_tplTool = new ccTracePolylineTool(m_pickingHub, this);
        connect(m_tplTool, &ccOverlayDialog::processFinished, this,
                &MainWindow::deactivateTracePolylineMode);
        registerOverlayDialog(m_tplTool, Qt::TopRightCorner);
    }

    m_tplTool->linkWith(ecvDisplayTools::GetCurrentScreen());

    freezeUI(true);
    m_ui->ViewToolBar->setDisabled(false);

    if (!m_tplTool->start())
        deactivateTracePolylineMode(false);
    else
        updateOverlayDialogsPlacement();
}

void MainWindow::deactivateTracePolylineMode(bool) {
    freezeUI(false);
    updateUI();
}

void MainWindow::registerOverlayDialog(ccOverlayDialog* dlg, Qt::Corner pos) {
    // check for existence
    for (ccMDIDialogs& mdi : m_mdiDialogs) {
        if (mdi.dialog == dlg) {
            // we only update its position in this case
            mdi.position = pos;
            repositionOverlayDialog(mdi);
            return;
        }
    }

    // otherwise we add it to DB
    m_mdiDialogs.push_back(ccMDIDialogs(dlg, pos));

    // automatically update the dialog placement when its shown
    connect(dlg, &ccOverlayDialog::shown, this, [=]() {
        // check for existence
        for (ccMDIDialogs& mdi : m_mdiDialogs) {
            if (mdi.dialog == dlg) {
                repositionOverlayDialog(mdi);
                break;
            }
        }
    });

    repositionOverlayDialog(m_mdiDialogs.back());
}

void MainWindow::unregisterOverlayDialog(ccOverlayDialog* dialog) {
    for (std::vector<ccMDIDialogs>::iterator it = m_mdiDialogs.begin();
         it != m_mdiDialogs.end(); ++it) {
        if (it->dialog == dialog) {
            m_mdiDialogs.erase(it);
            break;
        }
    }
}

void MainWindow::updateOverlayDialogsPlacement() {
    for (ccMDIDialogs& mdiDlg : m_mdiDialogs) {
        repositionOverlayDialog(mdiDlg);
    }
}

ccHObject* MainWindow::loadFile(QString filename, bool silent) {
    FileIOFilter::LoadParameters parameters;
    {
        parameters.alwaysDisplayLoadDialog = silent ? false : true;
        parameters.shiftHandlingMode =
                ecvGlobalShiftManager::NO_DIALOG_AUTO_SHIFT;
        parameters.parentWidget = silent ? nullptr : this;
    }

    CC_FILE_ERROR result = CC_FERR_NO_ERROR;
    ccHObject* newGroup =
            FileIOFilter::LoadFromFile(filename, parameters, result);

    return newGroup;
}

void MainWindow::repositionOverlayDialog(ccMDIDialogs& mdiDlg) {
    if (!mdiDlg.dialog || !mdiDlg.dialog->isVisible() || !m_mdiArea) return;

    int dx = 0;
    int dy = 0;
    static const int margin = 5;
    // QRect screenRect = ecvDisplayTools::GetScreenRect();
    switch (mdiDlg.position) {
        case Qt::TopLeftCorner:
            dx = margin;
            dy = margin;
            break;
        case Qt::TopRightCorner:
            dx = std::max(margin,
                          m_mdiArea->width() - mdiDlg.dialog->width() - margin);
            dy = margin;
            break;
        case Qt::BottomLeftCorner:
            dx = margin;
            dy = std::max(margin, m_mdiArea->height() -
                                          mdiDlg.dialog->height() - margin);
            break;
        case Qt::BottomRightCorner:
            dx = std::max(margin,
                          m_mdiArea->width() - mdiDlg.dialog->width() - margin);
            dy = std::max(margin, m_mdiArea->height() -
                                          mdiDlg.dialog->height() - margin);
            break;
    }

    // show();
    mdiDlg.dialog->move(m_mdiArea->mapToGlobal(QPoint(dx, dy)));
    mdiDlg.dialog->raise();
}

// helper for doActionMerge
void AddToRemoveList(ccHObject* toRemove,
                     ccHObject::Container& toBeRemovedList) {
    // is a parent or sibling already in the "toBeRemoved" list?
    std::size_t j = 0;
    std::size_t count = toBeRemovedList.size();
    while (j < count) {
        if (toBeRemovedList[j]->isAncestorOf(toRemove)) {
            toRemove = nullptr;
            break;
        } else if (toRemove->isAncestorOf(toBeRemovedList[j])) {
            toBeRemovedList[j] = toBeRemovedList.back();
            toBeRemovedList.pop_back();
            count--;
            j++;
        } else {
            // forward
            j++;
        }
    }

    if (toRemove) toBeRemovedList.push_back(toRemove);
}

void MainWindow::doActionMerge() {
    // let's look for clouds or meshes (warning: we don't mix them)
    std::vector<ccPointCloud*> clouds;
    std::vector<ccMesh*> meshes;

    try {
        for (ccHObject* entity : getSelectedEntities()) {
            if (!entity) continue;

            if (entity->isA(CV_TYPES::POINT_CLOUD)) {
                ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(entity);
                clouds.push_back(cloud);
            } else if (entity->isKindOf(CV_TYPES::MESH)) {
                ccMesh* mesh = ccHObjectCaster::ToMesh(entity);
                // this is a purely theoretical test for now!
                if (mesh && mesh->getAssociatedCloud() &&
                    mesh->getAssociatedCloud()->isA(CV_TYPES::POINT_CLOUD)) {
                    meshes.push_back(mesh);
                } else {
                    ecvConsole::Warning(
                            tr("Only meshes with standard vertices are handled "
                               "for now! Can't merge entity '%1'...")
                                    .arg(entity->getName()));
                }
            } else {
                ecvConsole::Warning(tr("Entity '%1' is neither a cloud nor a "
                                       "mesh, can't merge it!")
                                            .arg(entity->getName()));
            }
        }
    } catch (const std::bad_alloc&) {
        CVLog::Error(tr("Not enough memory!"));
        return;
    }

    if (clouds.empty() && meshes.empty()) {
        CVLog::Error(tr("Select only clouds or meshes!"));
        return;
    }
    if (!clouds.empty() && !meshes.empty()) {
        CVLog::Error(tr("Can't mix point clouds and meshes!"));
    }

    // merge clouds?
    if (!clouds.empty()) {
        // we deselect all selected entities (as most of them are going to
        // disappear)
        if (m_ccRoot) {
            m_ccRoot->unselectAllEntities();
            assert(!haveSelection());
            // m_selectedEntities.clear();
        }

        // we will remove the useless clouds/meshes later
        ccHObject::Container toBeRemoved;

        ccPointCloud* firstCloud = nullptr;
        ccHObjectContext firstCloudContext;

        // whether to generate the 'original cloud index' scalar field or not
        cloudViewer::ScalarField* ocIndexSF = nullptr;
        size_t cloudIndex = 0;

        for (size_t i = 0; i < clouds.size(); ++i) {
            ccPointCloud* pc = clouds[i];
            if (!firstCloud) {
                // we don't delete the first cloud (we'll merge the other one
                // 'inside' it
                firstCloud = pc;
                // we still have to temporarily detach the first cloud, as it
                // may undergo "severe" modifications (octree deletion, etc.)
                //--> see ccPointCloud::operator +=
                firstCloudContext =
                        removeObjectTemporarilyFromDBTree(firstCloud);

                if (QMessageBox::question(
                            this, tr("Original cloud index"),
                            tr("Do you want to generate a scalar field with "
                               "the original cloud index?")) ==
                    QMessageBox::Yes) {
                    int sfIdx = pc->getScalarFieldIndexByName(
                            CC_ORIGINAL_CLOUD_INDEX_SF_NAME);
                    if (sfIdx < 0) {
                        sfIdx = pc->addScalarField(
                                CC_ORIGINAL_CLOUD_INDEX_SF_NAME);
                    }
                    if (sfIdx < 0) {
                        ecvConsole::Error(
                                tr("Couldn't allocate a new scalar field for "
                                   "storing the original cloud index! Try to "
                                   "free some memory ..."));
                        return;
                    } else {
                        ocIndexSF = pc->getScalarField(sfIdx);
                        ocIndexSF->fill(0);
                        firstCloud->setCurrentDisplayedScalarField(sfIdx);
                    }
                }
            } else {
                unsigned countBefore = firstCloud->size();
                unsigned countAdded = pc->size();
                *firstCloud += pc;

                // success?
                if (firstCloud->size() == countBefore + countAdded) {
                    // firstCloud->prepareDisplayForRefresh_recursive();

                    ccHObject* toRemove = nullptr;
                    // if the entity to remove is a group with a unique child,
                    // we can remove it as well
                    ccHObject* parent = pc->getParent();
                    if (parent && parent->isA(CV_TYPES::HIERARCHY_OBJECT) &&
                        parent->getChildrenNumber() == 1 &&
                        parent != firstCloudContext.parent)
                        toRemove = parent;
                    else
                        toRemove = pc;

                    AddToRemoveList(toRemove, toBeRemoved);

                    if (ocIndexSF) {
                        ScalarType index =
                                static_cast<ScalarType>(++cloudIndex);
                        for (unsigned i = 0; i < countAdded; ++i) {
                            ocIndexSF->setValue(countBefore + i, index);
                        }
                    }
                } else {
                    ecvConsole::Error(
                            tr("Fusion failed! (not enough memory?)"));
                    break;
                }
                pc = nullptr;
            }
        }

        if (ocIndexSF) {
            ocIndexSF->computeMinAndMax();
            firstCloud->showSF(true);
        }

        // something to remove?
        while (!toBeRemoved.empty()) {
            if (toBeRemoved.back() && m_ccRoot) {
                m_ccRoot->removeElement(toBeRemoved.back());
            }
            toBeRemoved.pop_back();
        }

        // put back first cloud in DB
        if (firstCloud) {
            putObjectBackIntoDBTree(firstCloud, firstCloudContext);
            if (m_ccRoot) m_ccRoot->selectEntity(firstCloud);
        }
    }
    // merge meshes?
    else if (!meshes.empty()) {
        bool createSubMeshes = true;
        // createSubMeshes = (QMessageBox::question(this, "Create sub-meshes",
        // "Do you want to create sub-mesh entities corresponding to each source
        // mesh? (requires more memory)", QMessageBox::Yes, QMessageBox::No) ==
        // QMessageBox::Yes);

        // meshes are merged
        ccPointCloud* baseVertices = new ccPointCloud("vertices");
        ccMesh* baseMesh = new ccMesh(baseVertices);
        baseMesh->setName("Merged mesh");
        baseMesh->addChild(baseVertices);
        baseVertices->setEnabled(false);

        for (ccMesh* mesh : meshes) {
            // if (mesh->isA(CV_TYPES::PRIMITIVE))
            //{
            //	mesh = mesh->ccMesh::cloneMesh(); //we want a clone of the mesh
            // part, not the primitive!
            // }

            if (!baseMesh->merge(mesh, createSubMeshes)) {
                ecvConsole::Error(tr("Fusion failed! (not enough memory?)"));
                break;
            }
        }

        baseMesh->setVisible(true);
        addToDB(baseMesh);

        if (m_ccRoot) m_ccRoot->selectEntity(baseMesh);
    }

    updateUI();
}

void MainWindow::refreshAll(bool only2D /* = false*/,
                            bool forceRedraw /* = true*/) {
    ecvDisplayTools::RedrawDisplay(only2D, forceRedraw);
}

void MainWindow::refreshSelected(bool only2D /* = false*/,
                                 bool forceRedraw /* = true*/) {
    ecvDisplayTools::SetRedrawRecursive(false);
    for (ccHObject* entity : getSelectedEntities()) {
        if (entity) {
            entity->setRedrawFlagRecursive(true);
        }
    }
    ecvDisplayTools::RedrawDisplay(only2D, forceRedraw);
}

void MainWindow::refreshObject(ccHObject* obj, bool only2D, bool forceRedraw) {
    if (!obj) {
        return;
    }

    ecvDisplayTools::SetRedrawRecursive(false);
    obj->setRedrawFlagRecursive(true);
    ecvDisplayTools::RedrawDisplay(only2D, forceRedraw);
}

void MainWindow::refreshObjects(ccHObject::Container objs,
                                bool only2D,
                                bool forceRedraw) {
    ecvDisplayTools::SetRedrawRecursive(false);
    for (ccHObject* entity : objs) {
        if (entity) {
            entity->setRedrawFlagRecursive(true);
        }
    }
    ecvDisplayTools::RedrawDisplay(only2D, forceRedraw);
}

void MainWindow::resetSelectedBBox() {
    for (ccHObject* entity : getSelectedEntities()) {
        if (entity) {
            ecvDisplayTools::RemoveBB(entity->getViewId());
        }
    }
}

void MainWindow::toggleActiveWindowCenteredPerspective() {
    QWidget* win = getActiveWindow();
    if (win) {
        const ecvViewportParameters& params =
                ecvDisplayTools::GetViewportParameters();
        // we need to check this only if we are already in object-centered
        // perspective mode
        if (params.perspectiveView && params.objectCenteredView) {
            return;
        }
        doActionPerspectiveProjection();
        refreshAll(true, false);
        updateViewModePopUpMenu();
    }
}

void MainWindow::toggleActiveWindowViewerBasedPerspective() {
    QWidget* win = getActiveWindow();
    if (win) {
        const ecvViewportParameters& params =
                ecvDisplayTools::GetViewportParameters();
        // we need to check this only if we are already in viewer-based
        // perspective mode
        if (params.perspectiveView && !params.objectCenteredView) {
            return;
        }
        doActionOrthogonalProjection();
        refreshAll(true, false);
        updateViewModePopUpMenu();
        // updatePivotVisibilityPopUpMenu(win);
    }
}

void MainWindow::createSinglePointCloud() {
    // ask the user to input the point coordinates
    static CCVector3d s_lastPoint(0, 0, 0);
    static size_t s_lastPointIndex = 0;
    ccAskThreeDoubleValuesDlg axisDlg(
            "x", "y", "z", -1.0e12, 1.0e12, s_lastPoint.x, s_lastPoint.y,
            s_lastPoint.z, 4, tr("Point coordinates"), this);
    if (axisDlg.buttonBox->button(QDialogButtonBox::Ok))
        axisDlg.buttonBox->button(QDialogButtonBox::Ok)->setFocus();
    if (!axisDlg.exec()) return;
    s_lastPoint.x = axisDlg.doubleSpinBox1->value();
    s_lastPoint.y = axisDlg.doubleSpinBox2->value();
    s_lastPoint.z = axisDlg.doubleSpinBox3->value();

    // create the cloud
    ccPointCloud* cloud = new ccPointCloud();
    if (!cloud->reserve(1)) {
        delete cloud;
        CVLog::Error(tr("Not enough memory!"));
        return;
    }
    cloud->setName(tr("Point #%1").arg(++s_lastPointIndex));
    cloud->addPoint(CCVector3::fromArray(s_lastPoint.u));
    cloud->setPointSize(5);

    // add it to the DB tree
    addToDB(cloud, true, true, true, true);

    // select it
    m_ccRoot->unselectAllEntities();
    setSelectedInDB(cloud, true);
}

void MainWindow::createPointCloudFromClipboard() {
    const QClipboard* clipboard = QApplication::clipboard();
    assert(clipboard);
    const QMimeData* mimeData = clipboard->mimeData();
    if (!mimeData) {
        CVLog::Warning(tr("Clipboard is empty"));
        return;
    }

    if (!mimeData->hasText()) {
        CVLog::Error("ASCII/text data expected");
        return;
    }

    // try to convert the data to a point cloud
    FileIOFilter::LoadParameters parameters;
    {
        parameters.alwaysDisplayLoadDialog = true;
        parameters.shiftHandlingMode =
                ecvGlobalShiftManager::DIALOG_IF_NECESSARY;
        parameters.parentWidget = this;
    }

    ccHObject container;
    QByteArray data = mimeData->data("text/plain");
    CC_FILE_ERROR result = AsciiFilter().loadAsciiData(data, tr("Clipboard"),
                                                       container, parameters);
    if (result != CC_FERR_NO_ERROR) {
        FileIOFilter::DisplayErrorMessage(result, tr("loading"),
                                          tr("from the clipboard"));
        return;
    }

    // we only expect clouds
    ccHObject::Container clouds;
    if (container.filterChildren(clouds, true, CV_TYPES::POINT_CLOUD) == 0) {
        assert(false);
        CVLog::Error(tr("No cloud loaded"));
        return;
    }

    // detach the clouds from the loading container
    for (ccHObject* cloud : clouds) {
        if (cloud) {
            container.removeDependencyWith(cloud);
        }
    }
    container.removeAllChildren();

    // retrieve or create the group to store the 'clipboard' clouds
    ccHObject* clipboardGroup = nullptr;
    {
        static unsigned s_clipboardGroupID = 0;

        if (s_clipboardGroupID != 0) {
            clipboardGroup = dbRootObject()->find(s_clipboardGroupID);
            if (nullptr == clipboardGroup) {
                // can't find the previous group
                s_clipboardGroupID = 0;
            }
        }

        if (s_clipboardGroupID == 0) {
            clipboardGroup = new ccHObject(tr("Clipboard"));
            s_clipboardGroupID = clipboardGroup->getUniqueID();
            addToDB(clipboardGroup, false, false, false, false);
        }
    }
    assert(clipboardGroup);

    bool normalsDisplayedByDefault =
            ecvOptions::Instance().normalsDisplayedByDefault;
    for (ccHObject* cloud : clouds) {
        if (cloud) {
            clipboardGroup->addChild(cloud);
            cloud->setName(
                    tr("Cloud #%1").arg(clipboardGroup->getChildrenNumber()));

            if (!normalsDisplayedByDefault) {
                // disable the normals on all loaded clouds!
                static_cast<ccGenericPointCloud*>(cloud)->showNormals(false);
            }
        }
    }

    // eventually, we can add the clouds to the DB tree
    for (size_t i = 0; i < clouds.size(); ++i) {
        ccHObject* cloud = clouds[i];
        if (cloud) {
            bool lastCloud = (i + 1 == clouds.size());
            addToDB(cloud, lastCloud, lastCloud, true, lastCloud);
        }
    }

    QMainWindow::statusBar()->showMessage(
            tr("%1 cloud(s) loaded from the clipboard").arg(clouds.size()),
            2000);
}

void MainWindow::removeFromDB(ccHObject* obj, bool autoDelete) {
    if (!obj) return;

    obj->removeFromRenderScreen(true);

    // remove dependency to avoid deleting the object when removing it from DB
    // tree
    if (!autoDelete && obj->getParent())
        obj->getParent()->removeDependencyWith(obj);

    if (m_ccRoot) m_ccRoot->removeElement(obj);
}

void MainWindow::setSelectedInDB(ccHObject* obj, bool selected) {
    if (obj && m_ccRoot) {
        if (selected)
            m_ccRoot->selectEntity(obj);
        else
            m_ccRoot->unselectEntity(obj);
    }
}

void MainWindow::freezeUI(bool state) {
    // freeze standard toolbar
    m_ui->menuBar->setDisabled(state);
    m_ui->DockableDBTree->setDisabled(state);
    m_ui->mainToolBar->setDisabled(state);
    m_ui->SFToolBar->setDisabled(state);
    m_ui->FilterToolBar->setDisabled(state);
    m_ui->AnnotationToolBar->setDisabled(state);

    // freeze plugin toolbars
    m_pluginUIManager->mainPluginToolbar()->setDisabled(state);
    for (QToolBar* toolbar : m_pluginUIManager->additionalPluginToolbars()) {
        toolbar->setDisabled(state);
    }

    if (!state) {
        updateMenus();
    }

    m_uiFrozen = state;
}

void MainWindow::zoomOnSelectedEntities() {
    ccBBox bbox = getSelectedEntityBbox();
    if (bbox.isValid()) {
        ecvDisplayTools::UpdateConstellationCenterAndZoom(&bbox, false);
    } else {
        CVLog::Warning(tr("Selected entities have no valid bounding-box!"));
    }
}

void MainWindow::zoomOnEntities(ccHObject* obj) {
    if (obj) {
        ccBBox bbox = obj->getDisplayBB_recursive(false);
        if (bbox.isValid()) {
            ecvDisplayTools::UpdateConstellationCenterAndZoom(&bbox, false);
        } else {
            CVLog::Warning(tr("entity [%1] has no valid bounding-box!")
                                   .arg(obj->getName()));
        }
    }
}

void MainWindow::setGlobalZoom() {
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::SetRedrawRecursive(false);
        ecvDisplayTools::ZoomGlobal();
    }
}

//=============================================================================
// SELECTION TOOLS - Using centralized cvSelectionToolController
// (ParaView-style)
//=============================================================================

#if defined(USE_PCL_BACKEND)
void MainWindow::initSelectionController() {
    // Get the singleton controller
    m_selectionController = cvSelectionToolController::instance();
    m_selectionController->initialize(this);

    // Set visualizer
    ecvGenericVisualizer3D* viewer = ecvDisplayTools::GetVisualizer3D();
    if (viewer) {
        m_selectionController->setVisualizer(viewer);
    }

    // Setup all actions using the SelectionActions struct
    cvSelectionToolController::SelectionActions actions;
    actions.selectSurfaceCells = m_ui->actionSelectSurfaceCells;
    actions.selectSurfacePoints = m_ui->actionSelectSurfacePoints;
    actions.selectFrustumCells = m_ui->actionSelectFrustumCells;
    actions.selectFrustumPoints = m_ui->actionSelectFrustumPoints;
    actions.selectPolygonCells = m_ui->actionSelectPolygonCells;
    actions.selectPolygonPoints = m_ui->actionSelectPolygonPoints;
    actions.selectBlocks = m_ui->actionSelectBlocks;
    actions.selectFrustumBlocks = m_ui->actionSelectFrustumBlocks;
    actions.interactiveSelectCells = m_ui->actionInteractiveSelectCells;
    actions.interactiveSelectPoints = m_ui->actionInteractiveSelectPoints;
    actions.hoverCells = m_ui->actionHoverCells;
    actions.hoverPoints = m_ui->actionHoverPoints;
    actions.addSelection = m_ui->actionAddSelection;
    actions.subtractSelection = m_ui->actionSubtractSelection;
    actions.toggleSelection = m_ui->actionToggleSelection;
    actions.growSelection = m_ui->actionGrowSelection;
    actions.shrinkSelection = m_ui->actionShrinkSelection;
    actions.clearSelection = m_ui->actionClearSelection;
    actions.zoomToBox = m_ui->actionZoomToBox;

    m_selectionController->setupActions(actions);

    // Connect controller signals to MainWindow slots
    connect(m_selectionController,
            &cvSelectionToolController::selectionFinished, this,
            &MainWindow::onSelectionFinished);

    // Note: Selection tool state is now decoupled from Find Data dock
    // visibility (per ParaView design). The dock can be shown/hidden
    // independently by the user through the View menu. Selection tools work
    // regardless of dock visibility.
    connect(m_selectionController,
            &cvSelectionToolController::selectionToolStateChanged, this,
            [this](bool active) {
                Q_UNUSED(active);
                // Dock visibility is now user-controlled, not tied to tool
                // state The user can show/hide the Find Data dock independently
            });

    connect(m_selectionController,
            &cvSelectionToolController::selectionPropertiesUpdateRequested,
            this, [this](const cvSelectionData& data) {
                // Update the Find Data dock with new selection data
                if (m_findDataDock) {
                    m_findDataDock->updateSelection(data);
                }
            });

    // CRITICAL FIX: Connect properties delegate's clear request to selection
    // manager This prevents crashes from dangling pointers when objects are
    // deleted
    if (m_ccRoot && m_ccRoot->getPropertiesDelegate()) {
        connect(m_ccRoot->getPropertiesDelegate(),
                &ccPropertiesTreeDelegate::requestClearSelection, this,
                [this]() {
                    CVLog::Print(
                            "[MainWindow] Clearing selection data due to "
                            "object changes");
                    auto* manager = getSelectionManager();
                    if (manager) {
                        manager->clearCurrentSelection();
                    }
                    // Also clear highlights
                    if (m_selectionController &&
                        m_selectionController->highlighter()) {
                        m_selectionController->highlighter()->clearHighlights();
                    }
                });
    }

    // Connect zoom to box signal for notification (zoom is handled by
    // cvZoomToBoxTool)
    connect(m_selectionController,
            &cvSelectionToolController::zoomToBoxRequested, this,
            [this](int xmin, int ymin, int xmax, int ymax) {
                CVLog::PrintDebug(
                        QString("[MainWindow] Zoom to box completed: [%1, "
                                "%2, %3, %4]")
                                .arg(xmin)
                                .arg(ymin)
                                .arg(xmax)
                                .arg(ymax));
                // Zoom is already performed by cvZoomToBoxTool using VTK
                // This signal is for notification/logging purposes
                ecvDisplayTools::UpdateScreen();
            });

    // Set the properties delegate for the controller
    if (m_ccRoot && m_ccRoot->getPropertiesDelegate()) {
        m_selectionController->setPropertiesDelegate(
                m_ccRoot->getPropertiesDelegate());
    }

    if (m_findDataDock) {
        ecvGenericVisualizer3D* visualizer = ecvDisplayTools::GetVisualizer3D();
        cvSelectionHighlighter* highlighter =
                m_selectionController->highlighter();
        cvViewSelectionManager* manager = getSelectionManager();

        CVLog::PrintDebug(
                QString("[MainWindow::initSelectionController] Calling "
                        "configure: "
                        "highlighter=%1, manager=%2, visualizer=%3")
                        .arg(highlighter != nullptr)
                        .arg(manager != nullptr)
                        .arg(visualizer != nullptr));

        m_findDataDock->configure(highlighter, manager, visualizer);
    } else {
        CVLog::Warning(
                "[MainWindow::initSelectionController] m_findDataDock is "
                "nullptr!");
    }
}

void MainWindow::disableAllSelectionTools(void* except) {
    // Delegate to the controller - it handles all tool management
    if (m_selectionController) {
        // Pass nullptr to disable all tools
        m_selectionController->disableAllTools(nullptr);
    }
}

cvViewSelectionManager* MainWindow::getSelectionManager() const {
    if (m_selectionController) {
        return m_selectionController->manager();
    }
    return nullptr;
}

void MainWindow::onSelectionFinished(const cvSelectionData& selectionData) {
    // CRITICAL FIX: Don't call setCurrentSelection here!
    // The tool has already set it via manager->setCurrentSelection()
    // Calling it again causes infinite recursion:
    //   setCurrentSelection → selectionChanged → selectionFinished → here →
    //   setCurrentSelection...

    // Get manager from controller
    cvViewSelectionManager* manager = getSelectionManager();
    if (!manager) {
        return;
    }

    // NOTE: Selection is already stored by the tool/controller
    // We just need to update the UI based on current selection state
    bool hasSelection = !selectionData.isEmpty();

    // Enable/disable manipulation actions based on selection state
    m_ui->actionGrowSelection->setEnabled(hasSelection);
    m_ui->actionShrinkSelection->setEnabled(hasSelection);
    m_ui->actionClearSelection->setEnabled(hasSelection);

    // Update Find Data dock widget with selection data
    // (Selection properties are now in standalone cvFindDataDockWidget)
    if (m_findDataDock) {
        m_findDataDock->updateSelection(selectionData);
    }

    // NOTE: Highlighting is already done in
    // cvRenderViewSelectionReaction::finalizeSelection() Do NOT call
    // highlighter->highlightSelection() here again as it causes double
    // highlighting and potential crashes due to actor management issues. We
    // only need to clear highlights when selection is empty.
    if (!hasSelection) {
        cvSelectionHighlighter* highlighter =
                m_selectionController ? m_selectionController->highlighter()
                                      : nullptr;
        if (highlighter) {
            highlighter->clearHighlights();
        }
    }

    ecvDisplayTools::UpdateScreen();

    CVLog::PrintDebug(QString("[MainWindow] Selection UI updated: %1 elements")
                              .arg(selectionData.count()));
}

void MainWindow::onSelectionToolActivated(QAction* action) {
    bool isSelectionTool = (action && action->isChecked());

    CVLog::PrintDebug(
            QString("[MainWindow] Selection tool %1: %2")
                    .arg(action ? action->text() : "unknown")
                    .arg(isSelectionTool ? "activated" : "deactivated"));

    // Set visualizer for other property editors if needed
    if (m_ccRoot && m_ccRoot->getPropertiesDelegate()) {
        if (isSelectionTool) {
            ecvGenericVisualizer3D* viewer = ecvDisplayTools::GetVisualizer3D();
            if (viewer) {
                m_ccRoot->getPropertiesDelegate()->setVisualizer(viewer);
            }
        }
    }
}

void MainWindow::onSelectionRestored(const cvSelectionData& selection) {
    cvViewSelectionManager* manager = getSelectionManager();
    if (manager) {
        manager->setCurrentSelection(selection);
        CVLog::PrintDebug(QString("[MainWindow] Selection restored: %1 %2")
                                  .arg(selection.count())
                                  .arg(selection.fieldTypeString()));
        if (m_ccRoot) {
            m_ccRoot->updatePropertiesView();
        }
    }
}

void MainWindow::onSelectionHistoryChanged() {
    if (m_selectionController) {
        cvSelectionHistory* history = m_selectionController->history();
        if (history) {
            CVLog::PrintDebug(
                    QString("[MainWindow] Selection history changed - "
                            "Can undo: %1, Can redo: %2")
                            .arg(history->canUndo())
                            .arg(history->canRedo()));
        }
    }
}

void MainWindow::onBookmarksChanged() {
    CVLog::PrintDebug("[MainWindow] Selection bookmarks changed");
}

void MainWindow::undoSelection() {
    if (m_selectionController) {
        m_selectionController->undoSelection();
    }
}

void MainWindow::redoSelection() {
    if (m_selectionController) {
        m_selectionController->redoSelection();
    }
}

#endif

//=============================================================================
// SELECTION TOOLS - Using centralized cvSelectionToolController
// (ParaView-style)
//=============================================================================

void MainWindow::increasePointSize() {
    ecvDisplayTools::SetPointSize(
            ecvDisplayTools::GetViewportParameters().defaultPointSize + 1);
    refreshAll();
}

void MainWindow::decreasePointSize() {
    ecvDisplayTools::SetPointSize(
            ecvDisplayTools::GetViewportParameters().defaultPointSize - 1);
    refreshAll();
}

void MainWindow::setupInputDevices() {
#ifdef CC_3DXWARE_SUPPORT
    m_3DMouseManager = new cc3DMouseManager(this, this);
    m_ui->menuFile->insertMenu(m_UI->actionCloseAll, m_3DMouseManager->menu());
#endif

#ifdef CC_GAMEPAD_SUPPORT
    m_gamepadManager = new ccGamepadManager(this, this);
    m_ui->menuFile->insertMenu(m_ui->actionClearAll, m_gamepadManager->menu());
#endif

#if defined(CC_3DXWARE_SUPPORT) || defined(CC_GAMEPAD_SUPPORT)
    m_ui->menuFile->insertSeparator(m_ui->actionClearAll);
#endif
}

void MainWindow::destroyInputDevices() {
#ifdef CC_GAMEPAD_SUPPORT
    delete m_gamepadManager;
    m_gamepadManager = nullptr;
#endif

#ifdef CC_3DXWARE_SUPPORT
    delete m_3DMouseManager;
    m_3DMouseManager = nullptr;
#endif
}

void MainWindow::showDisplayOptions() {
    ccDisplayOptionsDlg displayOptionsDlg(this);
    connect(&displayOptionsDlg, &ccDisplayOptionsDlg::aspectHasChanged, this,
            [=]() { refreshAll(); });

    displayOptionsDlg.exec();

    disconnect(&displayOptionsDlg);
}

void MainWindow::doActionPerspectiveProjection() {
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::SetPerspectiveProjection();
    }

    setPerspectiveView();

    updateViewModePopUpMenu();
}

void MainWindow::doActionOrthogonalProjection() {
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::SetOrthoProjection();
    }

    setOrthoView();

    updateViewModePopUpMenu();
}

void MainWindow::updateUI() {
    updateUIWithSelection();
    updateMenus();
    updatePropertiesView();
}

void MainWindow::updatePropertiesView() {
    if (m_ccRoot) {
        m_ccRoot->updatePropertiesView();
    }
}

void MainWindow::enablePickingOperation(QString message) {
    assert(m_pickingHub);
    if (!m_pickingHub->addListener(this)) {
        CVLog::Error(
                tr("Can't start the picking mechanism (another tool is already "
                   "using it)"));
        return;
    }

    // specific case: we prevent the 'point-pair based alignment' tool to
    // process the picked point! if (m_pprDlg) 	m_pprDlg->pause(true);

    ecvDisplayTools::DisplayNewMessage(
            message, ecvDisplayTools::LOWER_LEFT_MESSAGE, true, 24 * 3600);
    ecvDisplayTools::RedrawDisplay(true, true);

    freezeUI(true);
}

void MainWindow::cancelPreviousPickingOperation(bool aborted) {
    switch (s_currentPickingOperation) {
        case PICKING_ROTATION_CENTER:
            // nothing to do
            break;
        case PICKING_LEVEL_POINTS:
            if (s_levelMarkersCloud) {
                ecvDisplayTools::RemoveFromOwnDB(s_levelMarkersCloud);
                delete s_levelMarkersCloud;
                s_levelMarkersCloud = nullptr;
            }
            break;
        default:
            // assert(false);
            break;
    }

    if (aborted) {
        ecvDisplayTools::DisplayNewMessage(
                QString(),
                ecvDisplayTools::LOWER_LEFT_MESSAGE);  // clear previous
                                                       // messages
        ecvDisplayTools::DisplayNewMessage("Picking operation aborted",
                                           ecvDisplayTools::LOWER_LEFT_MESSAGE);
    }
    ecvDisplayTools::SetRedrawRecursive(false);
    ecvDisplayTools::RedrawDisplay(true, false);

    // specific case: we allow the 'point-pair based alignment' tool to process
    // the picked point!
    if (m_pprDlg) m_pprDlg->pause(false);

    freezeUI(false);

    m_pickingHub->removeListener(this);
    s_currentPickingOperation = NO_PICKING_OPERATION;
}

void MainWindow::onItemPicked(const PickedItem& pi) {
    if (!m_pickingHub) {
        return;
    }

    if (!pi.entity) {
        return;
    } else {
        pi.entity->setRedrawFlagRecursive(false);
    }

    CCVector3 pickedPoint = pi.P3D;
    switch (s_currentPickingOperation) {
        case PICKING_LEVEL_POINTS: {
            if (!s_levelMarkersCloud) {
                assert(false);
                cancelPreviousPickingOperation(true);
            }

            for (unsigned i = 0; i < s_levelMarkersCloud->size(); ++i) {
                const CCVector3* P = s_levelMarkersCloud->getPoint(i);
                if ((pickedPoint - *P).norm() < 1.0e-6) {
                    CVLog::Warning(
                            tr("[Level] Point is too close from the others!"));
                    return;
                }
            }

            // add the corresponding marker
            s_levelMarkersCloud->addPoint(pickedPoint);
            unsigned markerCount = s_levelMarkersCloud->size();
            cc2DLabel* label = new cc2DLabel();
            label->addPickedPoint(s_levelMarkersCloud, markerCount - 1);
            label->setName(tr("P#%1").arg(markerCount));
            label->setDisplayedIn2D(false);
            label->setVisible(true);
            s_levelMarkersCloud->addChild(label);
            refreshSelected(s_levelMarkersCloud);

            if (markerCount == 3) {
                // we have enough points!
                const CCVector3* A = s_levelMarkersCloud->getPoint(0);
                const CCVector3* B = s_levelMarkersCloud->getPoint(1);
                const CCVector3* C = s_levelMarkersCloud->getPoint(2);
                CCVector3 X = *B - *A;
                CCVector3 Y = *C - *A;
                CCVector3 Z = X.cross(Y);
                // we choose 'Z' so that it points 'upward' relatively to the
                // camera (assuming the user will be looking from the top)
                CCVector3d viewDir = ecvDisplayTools::GetCurrentViewDir();
                if (CCVector3d::fromArray(Z.u).dot(viewDir) > 0) {
                    Z = -Z;
                }
                Y = Z.cross(X);
                X.normalize();
                Y.normalize();
                Z.normalize();

                ccGLMatrixd trans;
                double* mat = trans.data();
                mat[0] = X.x;
                mat[4] = X.y;
                mat[8] = X.z;
                mat[12] = 0;
                mat[1] = Y.x;
                mat[5] = Y.y;
                mat[9] = Y.z;
                mat[13] = 0;
                mat[2] = Z.x;
                mat[6] = Z.y;
                mat[10] = Z.z;
                mat[14] = 0;
                mat[3] = 0;
                mat[7] = 0;
                mat[11] = 0;
                mat[15] = 1;

                CCVector3d T = -CCVector3d::fromArray(A->u);
                trans.apply(T);
                T += CCVector3d::fromArray(A->u);
                trans.setTranslation(T);

                assert(haveOneSelection() &&
                       m_selectedEntities.front() == s_levelEntity);
                applyTransformation(trans);

                // clear message
                ecvDisplayTools::DisplayNewMessage(
                        QString(), ecvDisplayTools::LOWER_LEFT_MESSAGE,
                        false);  // clear previous message
                ecvDisplayTools::SetView(CC_TOP_VIEW);
            } else {
                // we need more points!
                return;
            }
        }
            // we use the next 'case' entry (PICKING_ROTATION_CENTER) to
            // redefine the rotation center as well!
            assert(s_levelMarkersCloud && s_levelMarkersCloud->size() != 0);
            pickedPoint = *s_levelMarkersCloud->getPoint(0);
            // break;

        case PICKING_ROTATION_CENTER: {
            CCVector3d newPivot = CCVector3d::fromArray(pickedPoint.u);
            // specific case: transformation tool is enabled
            if (m_transTool && m_transTool->started()) {
                m_transTool->setRotationCenter(newPivot);
                const unsigned& precision =
                        ecvDisplayTools::GetDisplayParameters()
                                .displayedNumPrecision;
                ecvDisplayTools::DisplayNewMessage(
                        QString(), ecvDisplayTools::LOWER_LEFT_MESSAGE,
                        false);  // clear previous message
                ecvDisplayTools::DisplayNewMessage(
                        tr("Point (%1 ; %2 ; %3) set as rotation center for "
                           "interactive transformation")
                                .arg(pickedPoint.x, 0, 'f', precision)
                                .arg(pickedPoint.y, 0, 'f', precision)
                                .arg(pickedPoint.z, 0, 'f', precision),
                        ecvDisplayTools::LOWER_LEFT_MESSAGE, true);
            } else {
                const ecvViewportParameters& params =
                        ecvDisplayTools::GetViewportParameters();
                if (!params.perspectiveView || params.objectCenteredView) {
                    // apply current GL transformation (if any)
                    pi.entity->getGLTransformation().apply(newPivot);
                    ecvDisplayTools::SetPivotPoint(newPivot, true, true);
                }
            }
            // s_pickingWindow->redraw(); //already called by
            // 'cancelPreviousPickingOperation' (see below)
        } break;

        default:
            assert(false);
            break;
    }

    cancelPreviousPickingOperation(false);
}

void MainWindow::UpdateUI() { TheInstance()->updateUI(); }

void MainWindow::showEvent(QShowEvent* event) {
    QMainWindow::showEvent(event);
    // Update memory usage widget size when window is shown
    updateMemoryUsageWidgetSize();

    if (!m_FirstShow) {
        return;
    }

    // Use layout manager to restore or setup layout
    if (m_layoutManager) {
        m_layoutManager->restoreGUILayout(false);
        // After restoring layout, ensure all toolbar icon sizes are updated
        // This is critical because restoreGUILayout may restore saved icon
        // sizes
        updateAllToolbarIconSizes();
    } else {
        CVLog::Error("[MainWindow] Layout manager is not initialized!");
    }

    m_FirstShow = false;

    if (isFullScreen()) {
        m_ui->actionFullScreen->setChecked(true);
    }
}

// exit event
void MainWindow::closeEvent(QCloseEvent* event) {
    if ((m_ccRoot && m_ccRoot->getRootEntity()->getChildrenNumber() == 0) ||
        QMessageBox::question(
                this, tr("Quit"), tr("Are you sure you want to quit?"),
                QMessageBox::Ok, QMessageBox::Cancel) != QMessageBox::Cancel) {
        event->accept();
    } else {
        event->ignore();
    }

    if (s_autoSaveGuiElementPos) {
        saveGUIElementsPos();
    }
}

void MainWindow::saveGUIElementsPos() {
    // Use layout manager to save layout
    if (m_layoutManager) {
        m_layoutManager->saveGUILayout();
    } else {
        CVLog::Error("[MainWindow] Layout manager is not initialized!");
    }
}

void MainWindow::doShowPrimitiveFactory() {
    if (!m_pfDlg) m_pfDlg = new ecvPrimitiveFactoryDlg(this);

    m_pfDlg->setModal(false);
    m_pfDlg->setWindowModality(Qt::NonModal);
    m_pfDlg->show();
}

void MainWindow::doCheckForUpdate() {
    if (m_updateDlg) {
        m_updateDlg->setModal(false);
        m_updateDlg->setWindowModality(Qt::NonModal);
        m_updateDlg->show();
    }
}

void MainWindow::doActionComputeNormals() {
    if (!ccEntityAction::computeNormals(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionInvertNormals() {
    if (!ccEntityAction::invertNormals(m_selectedEntities)) return;

    refreshSelected();
}

void MainWindow::doActionConvertNormalsToDipDir() {
    if (!ccEntityAction::convertNormalsTo(
                m_selectedEntities,
                ccEntityAction::NORMAL_CONVERSION_DEST::DIP_DIR_SFS)) {
        return;
    }

    refreshSelected();
    updateUI();
}

void MainWindow::doActionExportNormalToSF() {
    if (!ccEntityAction::exportNormalToSF(m_selectedEntities, this)) {
        return;
    }

    refreshSelected();
    updateUI();
}

void MainWindow::doActionConvertNormalsToHSV() {
    if (!ccEntityAction::convertNormalsTo(
                m_selectedEntities,
                ccEntityAction::NORMAL_CONVERSION_DEST::HSV_COLORS)) {
        return;
    }

    refreshSelected();
    updateUI();
}

void MainWindow::doActionOrientNormalsMST() {
    if (!ccEntityAction::orientNormalsMST(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionOrientNormalsFM() {
    if (!ccEntityAction::orientNormalsFM(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

static double s_kdTreeMaxErrorPerCell = 0.1;
void MainWindow::doActionComputeKdTree() {
    ccGenericPointCloud* cloud = nullptr;

    if (haveOneSelection()) {
        ccHObject* ent = m_selectedEntities.back();
        bool lockedVertices;
        cloud = ccHObjectCaster::ToGenericPointCloud(ent, &lockedVertices);
        if (lockedVertices) {
            ecvUtils::DisplayLockedVerticesWarning(ent->getName(), true);
            return;
        }
    }

    if (!cloud) {
        CVLog::Error(tr("Selected one and only one point cloud or mesh!"));
        return;
    }

    bool ok;
    s_kdTreeMaxErrorPerCell = QInputDialog::getDouble(
            this, tr("Compute Kd-tree"), tr("Max error per leaf cell:"),
            s_kdTreeMaxErrorPerCell, 1.0e-6, 1.0e6, 6, &ok);
    if (!ok) return;

    ecvProgressDialog pDlg(true, this);

    // computation
    QElapsedTimer eTimer;
    eTimer.start();
    ccKdTree* kdtree = new ccKdTree(cloud);

    if (kdtree->build(
                s_kdTreeMaxErrorPerCell,
                cloudViewer::DistanceComputationTools::MAX_DIST_95_PERCENT, 4,
                1000, &pDlg)) {
        qint64 elapsedTime_ms = eTimer.elapsed();

        ecvConsole::Print("[doActionComputeKdTree] Timing: %2.3f s",
                          static_cast<double>(elapsedTime_ms) / 1.0e3);
        cloud->setEnabled(true);  // for mesh vertices!
        cloud->addChild(kdtree);
        kdtree->setVisible(true);

#ifdef QT_DEBUG
        kdtree->convertCellIndexToSF();
#else
        kdtree->convertCellIndexToRandomColor();
#endif

        addToDB(kdtree);
        // update added point cloud
        refreshObject(cloud);
        updateUI();
    } else {
        CVLog::Error(tr("An error occurred!"));
        delete kdtree;
        kdtree = nullptr;
    }
}

void MainWindow::doActionComputeOctree() {
    if (!ccEntityAction::computeOctree(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionResampleWithOctree() {
    bool ok;
    int pointCount = QInputDialog::getInt(this, tr("Resample with octree"),
                                          tr("Points (approx.)"), 1000000, 1,
                                          INT_MAX, 100000, &ok);
    if (!ok) return;

    ecvProgressDialog pDlg(false, this);
    pDlg.setAutoClose(false);

    assert(pointCount > 0);
    unsigned aimedPoints = static_cast<unsigned>(pointCount);

    bool errors = false;

    for (ccHObject* entity : getSelectedEntities()) {
        ccPointCloud* cloud = nullptr;

        /*if (ent->isKindOf(CV_TYPES::MESH)) //TODO
                cloud =
        ccHObjectCaster::ToGenericMesh(ent)->getAssociatedCloud(); else */
        if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
            cloud = static_cast<ccPointCloud*>(entity);
        }

        if (cloud) {
            ccOctree::Shared octree = cloud->getOctree();
            if (!octree) {
                octree = cloud->computeOctree(&pDlg);
                if (!octree) {
                    ecvConsole::Error(
                            tr("Could not compute octree for cloud '%1'")
                                    .arg(cloud->getName()));
                    continue;
                }
            }

            cloud->setEnabled(false);
            QElapsedTimer eTimer;
            eTimer.start();
            cloudViewer::GenericIndexedCloud* result =
                    cloudViewer::CloudSamplingTools::resampleCloudWithOctree(
                            cloud, aimedPoints,
                            cloudViewer::CloudSamplingTools::
                                    CELL_GRAVITY_CENTER,
                            &pDlg, octree.data());

            if (result) {
                ecvConsole::Print("[ResampleWithOctree] Timing: %3.2f s.",
                                  eTimer.elapsed() / 1.0e3);
                ccPointCloud* newCloud = ccPointCloud::From(result, cloud);

                delete result;
                result = nullptr;

                if (newCloud) {
                    addToDB(newCloud);
                } else {
                    errors = true;
                }
            }
        }
    }

    if (errors) {
        CVLog::Error(
                tr("[ResampleWithOctree] Errors occurred during the process! "
                   "Result may be incomplete!"));
    }
}

void MainWindow::doActionComputeMeshAA() {
    doActionComputeMesh(cloudViewer::DELAUNAY_2D_AXIS_ALIGNED);
}

void MainWindow::doActionComputeMeshLS() {
    doActionComputeMesh(cloudViewer::DELAUNAY_2D_BEST_LS_PLANE);
}

void MainWindow::doActionConvexHull() {
    if (!haveSelection()) {
        return;
    }

    ccHObject::Container clouds;
    for (auto ent : getSelectedEntities()) {
        if (!ent->isKindOf(CV_TYPES::POINT_CLOUD)) {
            CVLog::Warning("only point cloud is supported!");
            continue;
        }
        clouds.push_back(ent);
    }

    ccHObject::Container meshes;
    if (ccEntityAction::ConvexHull(clouds, meshes, this)) {
        for (size_t i = 0; i < meshes.size(); ++i) {
            addToDB(meshes[i]);
        }
    } else {
        ecvConsole::Error(tr("Error(s) occurred! See the Console messages"));
        return;
    }

    updateUI();
}

void MainWindow::doActionPoissonReconstruction() {
    if (!haveSelection()) {
        return;
    }

    // select candidates
    ecvPoissonReconDlg prpDlg(this);
    {
        for (ccHObject* entity : getSelectedEntities()) {
            if (!prpDlg.addEntity(entity)) {
            }
        }
    }

    if (prpDlg.start()) {
        ccHObject::Container& meshes = prpDlg.getReconstructions();
        for (size_t i = 0; i < meshes.size(); ++i) {
            addToDB(meshes[i]);
        }
    } else {
        ecvConsole::Error(tr("Error(s) occurred! See the Console messages"));
    }

    updateUI();
}

void MainWindow::doActionComputeMesh(cloudViewer::TRIANGULATION_TYPES type) {
    // ask the user for the max edge length
    static double s_meshMaxEdgeLength = 0.0;
    {
        bool ok = true;
        double maxEdgeLength = QInputDialog::getDouble(
                this, tr("Triangulate"), tr("Max edge length (0 = no limit)"),
                s_meshMaxEdgeLength, 0, 1.0e9, 8, &ok);
        if (!ok) return;
        s_meshMaxEdgeLength = maxEdgeLength;
    }

    // select candidates
    ccHObject::Container clouds;
    bool hadNormals = false;
    {
        for (ccHObject* entity : getSelectedEntities()) {
            if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
                clouds.push_back(entity);
                if (entity->isA(CV_TYPES::POINT_CLOUD)) {
                    hadNormals |=
                            static_cast<ccPointCloud*>(entity)->hasNormals();
                }
            }
        }
    }

    // if the cloud(s) already had normals, ask the use if wants to update them
    // or keep them as is (can look strange!)
    bool updateNormals = false;
    if (hadNormals) {
        updateNormals =
                (QMessageBox::question(
                         this, tr("Keep old normals?"),
                         tr("Cloud(s) already have normals. Do you want to "
                            "update them (yes) or keep the old ones (no)?"),
                         QMessageBox::Yes,
                         QMessageBox::No) == QMessageBox::Yes);
    }

    ecvProgressDialog pDlg(false, this);
    pDlg.setAutoClose(false);
    pDlg.setWindowTitle(tr("Triangulation"));
    pDlg.setInfo(tr("Triangulation in progress..."));
    pDlg.setRange(0, 0);
    pDlg.show();
    QApplication::processEvents();

    bool errors = false;
    for (size_t i = 0; i < clouds.size(); ++i) {
        ccHObject* ent = clouds[i];
        assert(ent->isKindOf(CV_TYPES::POINT_CLOUD));

        // compute mesh
        ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(ent);
        ccMesh* mesh = ccMesh::Triangulate(
                cloud, type, updateNormals,
                static_cast<PointCoordinateType>(s_meshMaxEdgeLength),
                2  // XY plane by default
        );
        if (mesh) {
            cloud->setVisible(false);  // can't disable the cloud as the
                                       // resulting mesh will be its child!
            cloud->addChild(mesh);
            addToDB(mesh);
            if (i == 0) {
                m_ccRoot->selectEntity(mesh);  // auto-select first element
            }
        } else {
            errors = true;
        }
    }

    if (errors) {
        ecvConsole::Error(tr("Error(s) occurred! See the Console messages"));
    }

    updateUI();
}

void MainWindow::doMeshTwoPolylines() {
    if (m_selectedEntities.size() != 2) return;

    ccPolyline* p1 = ccHObjectCaster::ToPolyline(m_selectedEntities[0]);
    ccPolyline* p2 = ccHObjectCaster::ToPolyline(m_selectedEntities[1]);
    if (!p1 || !p2) {
        ecvConsole::Error(tr("Select 2 and only 2 polylines"));
        return;
    }

    // Ask the user how the 2D projection should be computed
    bool useViewingDir = false;
    CCVector3 viewingDir(0, 0, 0);
    if (ecvDisplayTools::GetCurrentScreen()) {
        useViewingDir =
                (QMessageBox::question(this, tr("Projection method"),
                                       tr("Use best fit plane (yes) or the "
                                          "current viewing direction (no)"),
                                       QMessageBox::Yes,
                                       QMessageBox::No) == QMessageBox::No);
        if (useViewingDir) {
            viewingDir = -CCVector3::fromArray(
                    ecvDisplayTools::GetCurrentViewDir().u);
        }
    }

    ccMesh* mesh = ccMesh::TriangulateTwoPolylines(
            p1, p2, useViewingDir ? &viewingDir : 0);
    if (mesh) {
        addToDB(mesh);
        if (mesh->computePerVertexNormals()) {
            mesh->showNormals(true);
        } else {
            CVLog::Warning(
                    tr("[Mesh two polylines] Failed to compute normals!"));
        }
    } else {
        CVLog::Error(tr("Failed to create mesh (see Console)"));
        forceConsoleDisplay();
    }
}

void MainWindow::doActionMeshScanGrids() {
    // ask the user for the min angle (inside triangles)
    static double s_meshMinTriangleAngle_deg = 1.0;
    {
        bool ok = true;
        double minAngle_deg = QInputDialog::getDouble(
                this, tr("Triangulate"), tr("Min triangle angle (in degrees)"),
                s_meshMinTriangleAngle_deg, 0, 90.0, 3, &ok);
        if (!ok) return;
        s_meshMinTriangleAngle_deg = minAngle_deg;
    }

    // look for clouds with scan grids
    for (ccHObject* entity : getSelectedEntities()) {
        if (!entity || !entity->isA(CV_TYPES::POINT_CLOUD)) {
            continue;
        }

        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(entity);
        assert(cloud);

        for (size_t i = 0; i < cloud->gridCount(); ++i) {
            ccPointCloud::Grid::Shared grid = cloud->grid(i);
            if (!grid) {
                assert(false);
                continue;
            }

            ccMesh* gridMesh =
                    cloud->triangulateGrid(*grid, s_meshMinTriangleAngle_deg);
            if (gridMesh) {
                cloud->addChild(gridMesh);
                cloud->setVisible(false);  // hide the cloud
                addToDB(gridMesh, false, true, false, false);
            }
        }
    }

    updateUI();
}

void MainWindow::doActionComputeDistancesFromSensor() {
    // we support more than just one sensor in selection
    if (!haveSelection()) {
        ecvConsole::Error("Select at least a sensor.");
        return;
    }

    // start dialog
    ccSensorComputeDistancesDlg cdDlg(this);
    if (!cdDlg.exec()) return;

    for (ccHObject* entity : getSelectedEntities()) {
        ccSensor* sensor = ccHObjectCaster::ToSensor(entity);
        assert(sensor);
        if (!sensor) continue;  // skip this entity

        // get associated cloud
        ccHObject* defaultCloud =
                sensor->getParent() &&
                                sensor->getParent()->isA(CV_TYPES::POINT_CLOUD)
                        ? sensor->getParent()
                        : 0;
        ccPointCloud* cloud = askUserToSelectACloud(
                defaultCloud,
                "Select a cloud on which to project the uncertainty:");
        if (!cloud) {
            return;
        }

        // sensor center
        CCVector3 sensorCenter;
        if (!sensor->getActiveAbsoluteCenter(sensorCenter)) return;

        // squared required?
        bool squared = cdDlg.computeSquaredDistances();

        // set up a new scalar field
        const char* defaultRangesSFname =
                squared ? CC_DEFAULT_SQUARED_RANGES_SF_NAME
                        : CC_DEFAULT_RANGES_SF_NAME;
        int sfIdx = cloud->getScalarFieldIndexByName(defaultRangesSFname);
        if (sfIdx < 0) {
            sfIdx = cloud->addScalarField(defaultRangesSFname);
            if (sfIdx < 0) {
                ecvConsole::Error("Not enough memory!");
                return;
            }
        }
        cloudViewer::ScalarField* distances = cloud->getScalarField(sfIdx);

        for (unsigned i = 0; i < cloud->size(); ++i) {
            const CCVector3* P = cloud->getPoint(i);
            ScalarType s = static_cast<ScalarType>(
                    squared ? (*P - sensorCenter).norm2()
                            : (*P - sensorCenter).norm());
            distances->setValue(i, s);
        }

        distances->computeMinAndMax();
        cloud->setCurrentDisplayedScalarField(sfIdx);
        cloud->showSF(true);
    }

    refreshSelected();
    updateUI();
}

void MainWindow::doActionComputeScatteringAngles() {
    // there should be only one sensor in current selection!
    if (!haveOneSelection() ||
        !m_selectedEntities[0]->isKindOf(CV_TYPES::GBL_SENSOR)) {
        ecvConsole::Error("Select one and only one GBL sensor!");
        return;
    }

    ccSensor* sensor = ccHObjectCaster::ToSensor(m_selectedEntities[0]);
    assert(sensor);

    // sensor center
    CCVector3 sensorCenter;
    if (!sensor->getActiveAbsoluteCenter(sensorCenter)) return;

    // get associated cloud
    ccHObject* defaultCloud =
            sensor->getParent() &&
                            sensor->getParent()->isA(CV_TYPES::POINT_CLOUD)
                    ? sensor->getParent()
                    : nullptr;
    ccPointCloud* cloud = askUserToSelectACloud(
            defaultCloud,
            "Select a cloud on which to project the uncertainty:");
    if (!cloud) {
        return;
    }
    if (!cloud->hasNormals()) {
        ecvConsole::Error("The cloud must have normals!");
        return;
    }

    ccSensorComputeScatteringAnglesDlg cdDlg(this);
    if (!cdDlg.exec()) return;

    bool toDegreeFlag = cdDlg.anglesInDegrees();

    // prepare a new scalar field
    const char* defaultScatAnglesSFname =
            toDegreeFlag ? CC_DEFAULT_DEG_SCATTERING_ANGLES_SF_NAME
                         : CC_DEFAULT_RAD_SCATTERING_ANGLES_SF_NAME;
    int sfIdx = cloud->getScalarFieldIndexByName(defaultScatAnglesSFname);
    if (sfIdx < 0) {
        sfIdx = cloud->addScalarField(defaultScatAnglesSFname);
        if (sfIdx < 0) {
            ecvConsole::Error("Not enough memory!");
            return;
        }
    }
    cloudViewer::ScalarField* angles = cloud->getScalarField(sfIdx);

    // perform computations
    for (unsigned i = 0; i < cloud->size(); ++i) {
        // the point position
        const CCVector3* P = cloud->getPoint(i);

        // build the ray
        CCVector3 ray = *P - sensorCenter;
        ray.normalize();

        // get the current normal
        CCVector3 normal(cloud->getPointNormal(i));
        // normal.normalize(); //should already be the case!

        // compute the angle
        PointCoordinateType cosTheta = ray.dot(normal);
        ScalarType theta = std::acos(std::min(std::abs(cosTheta), 1.0f));

        if (toDegreeFlag) theta = cloudViewer::RadiansToDegrees(theta);

        angles->setValue(i, theta);
    }

    angles->computeMinAndMax();
    cloud->setCurrentDisplayedScalarField(sfIdx);
    cloud->showSF(true);

    refreshObject(cloud);
    updateUI();
}

void MainWindow::doActionSetViewFromSensor() {
    // there should be only one sensor in current selection!
    if (!haveOneSelection() ||
        !m_selectedEntities[0]->isKindOf(CV_TYPES::SENSOR)) {
        ecvConsole::Error("Select one and only one sensor!");
        return;
    }

    ccSensor* sensor = ccHObjectCaster::ToSensor(m_selectedEntities[0]);
    assert(sensor);

    if (sensor->applyViewport()) {
        ecvConsole::Print("[doActionSetViewFromSensor] Viewport applied");
    }
}

void MainWindow::doActionCreateGBLSensor() {
    ccGBLSensorProjectionDlg spDlg(this);
    if (!spDlg.exec()) return;

    // We create the corresponding sensor for each input cloud (in a perfect
    // world, there should be only one ;)
    for (ccHObject* entity : getSelectedEntities()) {
        if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
            ccGenericPointCloud* cloud =
                    ccHObjectCaster::ToGenericPointCloud(entity);

            // we create a new sensor
            ccGBLSensor* sensor = new ccGBLSensor();

            // we init its parameters with the dialog
            spDlg.updateGBLSensor(sensor);

            // we compute projection
            if (sensor->computeAutoParameters(cloud)) {
                cloud->addChild(sensor);

                // we try to guess the sensor relative size (dirty)
                ccBBox bb = cloud->getOwnBB();
                double diag = bb.getDiagNorm();
                if (diag < 1.0)
                    sensor->setGraphicScale(
                            static_cast<PointCoordinateType>(1.0e-3));
                else if (diag > 10000.0)
                    sensor->setGraphicScale(
                            static_cast<PointCoordinateType>(1.0e3));

                // we display depth buffer
                int errorCode;
                if (sensor->computeDepthBuffer(cloud, errorCode)) {
                    ccRenderingTools::ShowDepthBuffer(sensor, this);
                } else {
                    ecvConsole::Error(ccGBLSensor::GetErrorString(errorCode));
                }

                ////DGM: test
                //{
                //	//add positions
                //	const unsigned count = 1000;
                //	const PointCoordinateType R = 100;
                //	const PointCoordinateType dh = 100;
                //	for (unsigned i=0; i<1000; ++i)
                //	{
                //		float angle = (float)i/(float)count * 6 * M_PI;
                //		float X = R * cos(angle);
                //		float Y = R * sin(angle);
                //		float Z = (float)i/(float)count * dh;

                //		ccIndexedTransformation trans;
                //		trans.initFromParameters(-angle,CCVector3(0,0,1),CCVector3(X,Y,Z));
                //		sensor->addPosition(trans,i);
                //	}
                //}

                // set position
                // ccIndexedTransformation trans;
                // sensor->addPosition(trans,0);

                QWidget* win = static_cast<QWidget*>(getActiveWindow());
                if (win) {
                    // sensor->setDisplay_recursive(win);
                    sensor->setVisible(true);
                    ccBBox box = cloud->getOwnBB();
                    ecvDisplayTools::UpdateConstellationCenterAndZoom(&box);
                }

                addToDB(sensor);
            } else {
                CVLog::Error("Failed to create sensor");
                delete sensor;
                sensor = nullptr;
            }
        }
    }

    updateUI();
}

void MainWindow::doActionCreateCameraSensor() {
    // we create the camera sensor
    ccCameraSensor* sensor = new ccCameraSensor();

    ccHObject* ent = nullptr;
    if (haveSelection()) {
        assert(haveOneSelection());
        ent = m_selectedEntities.front();
    }

    // we try to guess the sensor relative size (dirty)
    if (ent && ent->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(ent);
        ccBBox bb = cloud->getOwnBB();
        double diag = bb.getDiagNorm();
        if (diag < 1.0)
            sensor->setGraphicScale(static_cast<PointCoordinateType>(1.0e-3));
        else if (diag > 10000.0)
            sensor->setGraphicScale(static_cast<PointCoordinateType>(1.0e3));

        // set position
        ccIndexedTransformation trans;
        sensor->addPosition(trans, 0);
    }

    ccCamSensorProjectionDlg spDlg(this);
    // spDlg.initWithCamSensor(sensor); //DGM: we'd better leave the default
    // parameters of the dialog!
    if (!spDlg.exec()) {
        delete sensor;
        return;
    }
    spDlg.updateCamSensor(sensor);

    QWidget* win = nullptr;
    if (ent) {
        ent->addChild(sensor);
        win = static_cast<QWidget*>(ecvDisplayTools::GetCurrentScreen());
    } else {
        win = getActiveWindow();
    }

    if (win) {
        // sensor->setDisplay(win);
        sensor->setVisible(true);
        if (ent) {
            ccBBox box = ent->getOwnBB();
            ecvDisplayTools::UpdateConstellationCenterAndZoom(&box);
        }
    }

    addToDB(sensor);

    updateUI();
}

void MainWindow::doActionModifySensor() {
    // there should be only one point cloud with sensor in current selection!
    if (!haveOneSelection() ||
        !m_selectedEntities[0]->isKindOf(CV_TYPES::SENSOR)) {
        ecvConsole::Error("Select one and only one sensor!");
        return;
    }

    ccSensor* sensor = static_cast<ccSensor*>(m_selectedEntities[0]);

    // Ground based laser sensors
    if (sensor->isA(CV_TYPES::GBL_SENSOR)) {
        ccGBLSensor* gbl = static_cast<ccGBLSensor*>(sensor);

        ccGBLSensorProjectionDlg spDlg(this);
        spDlg.initWithGBLSensor(gbl);

        if (!spDlg.exec()) return;

        // we update its parameters
        spDlg.updateGBLSensor(gbl);

        // we re-project the associated cloud (if any)
        if (gbl->getParent() &&
            gbl->getParent()->isKindOf(CV_TYPES::POINT_CLOUD)) {
            ccGenericPointCloud* cloud =
                    ccHObjectCaster::ToGenericPointCloud(gbl->getParent());

            int errorCode;
            if (gbl->computeDepthBuffer(cloud, errorCode)) {
                // we display depth buffer
                ccRenderingTools::ShowDepthBuffer(gbl, this);
            } else {
                ecvConsole::Error(ccGBLSensor::GetErrorString(errorCode));
            }
        } else {
            // ecvConsole::Warning(QString("Internal error: sensor ('%1') parent
            // is not a point cloud!").arg(sensor->getName()));
        }
    }
    // Camera sensors
    else if (sensor->isA(CV_TYPES::CAMERA_SENSOR)) {
        ccCameraSensor* cam = static_cast<ccCameraSensor*>(sensor);

        ccCamSensorProjectionDlg spDlg(this);
        spDlg.initWithCamSensor(cam);

        if (!spDlg.exec()) return;

        // we update its parameters
        spDlg.updateCamSensor(cam);
    } else {
        ecvConsole::Error("Can't modify this kind of sensor!");
        return;
    }

    if (sensor->isVisible() && sensor->isEnabled()) {
        refreshObject(sensor);
    }

    updateUI();
}

void MainWindow::doActionProjectUncertainty() {
    // there should only be one sensor in the current selection!
    if (!haveOneSelection() ||
        !m_selectedEntities[0]->isKindOf(CV_TYPES::CAMERA_SENSOR)) {
        ecvConsole::Error(
                "Select one and only one camera (projective) sensor!");
        return;
    }

    ccCameraSensor* sensor =
            ccHObjectCaster::ToCameraSensor(m_selectedEntities[0]);
    if (!sensor) {
        assert(false);
        return;
    }

    const ccCameraSensor::LensDistortionParameters::Shared& distParams =
            sensor->getDistortionParameters();
    if (!distParams ||
        distParams->getModel() != ccCameraSensor::BROWN_DISTORTION) {
        CVLog::Error(
                "Sensor has no associated uncertainty model! (Brown, etc.)");
        return;
    }

    // we need a cloud to project the uncertainty on!
    ccHObject* defaultCloud =
            sensor->getParent() &&
                            sensor->getParent()->isA(CV_TYPES::POINT_CLOUD)
                    ? sensor->getParent()
                    : nullptr;
    ccPointCloud* pointCloud = askUserToSelectACloud(
            defaultCloud,
            "Select a cloud on which to project the uncertainty:");
    if (!pointCloud) {
        return;
    }

    cloudViewer::ReferenceCloud points(pointCloud);
    if (!points.reserve(pointCloud->size())) {
        ecvConsole::Error("Not enough memory!");
        return;
    }
    points.addPointIndex(0, pointCloud->size());

    // compute uncertainty
    std::vector<Vector3Tpl<ScalarType>> accuracy;
    if (!sensor->computeUncertainty(&points, accuracy /*, false*/)) {
        ecvConsole::Error("Not enough memory!");
        return;
    }

    /////////////
    // SIGMA D //
    /////////////
    const char dimChar[3] = {'x', 'y', 'z'};
    for (unsigned d = 0; d < 3; ++d) {
        // add scalar field
        QString sfName = QString("[%1] Uncertainty (%2)")
                                 .arg(sensor->getName())
                                 .arg(dimChar[d]);
        int sfIdx = pointCloud->getScalarFieldIndexByName(qPrintable(sfName));
        if (sfIdx < 0) sfIdx = pointCloud->addScalarField(qPrintable(sfName));
        if (sfIdx < 0) {
            CVLog::Error("An error occurred! (see console)");
            return;
        }

        // fill scalar field
        cloudViewer::ScalarField* sf = pointCloud->getScalarField(sfIdx);
        assert(sf);
        if (sf) {
            unsigned count = static_cast<unsigned>(accuracy.size());
            assert(count == pointCloud->size());
            for (unsigned i = 0; i < count; i++)
                sf->setValue(i, accuracy[i].u[d]);
            sf->computeMinAndMax();
        }
    }

    /////////////////
    // SIGMA TOTAL //
    /////////////////

    // add scalar field
    {
        QString sfName =
                QString("[%1] Uncertainty (3D)").arg(sensor->getName());
        int sfIdx = pointCloud->getScalarFieldIndexByName(qPrintable(sfName));
        if (sfIdx < 0) sfIdx = pointCloud->addScalarField(qPrintable(sfName));
        if (sfIdx < 0) {
            CVLog::Error("An error occurred! (see console)");
            return;
        }

        // fill scalar field
        cloudViewer::ScalarField* sf = pointCloud->getScalarField(sfIdx);
        assert(sf);
        if (sf) {
            unsigned count = static_cast<unsigned>(accuracy.size());
            assert(count == pointCloud->size());
            for (unsigned i = 0; i < count; i++)
                sf->setValue(i, accuracy[i].norm());
            sf->computeMinAndMax();
        }

        pointCloud->showSF(true);
        pointCloud->setCurrentDisplayedScalarField(sfIdx);
    }

    refreshObject(pointCloud);
}

void MainWindow::doActionCheckPointsInsideFrustum() {
    // there should be only one camera sensor in the current selection!
    if (!haveOneSelection() ||
        !m_selectedEntities[0]->isKindOf(CV_TYPES::CAMERA_SENSOR)) {
        ecvConsole::Error("Select one and only one camera sensor!");
        return;
    }

    ccCameraSensor* sensor =
            ccHObjectCaster::ToCameraSensor(m_selectedEntities[0]);
    if (!sensor) return;

    // we need a cloud to filter!
    ccHObject* defaultCloud =
            sensor->getParent() &&
                            sensor->getParent()->isA(CV_TYPES::POINT_CLOUD)
                    ? sensor->getParent()
                    : 0;
    ccPointCloud* pointCloud =
            askUserToSelectACloud(defaultCloud, "Select a cloud to filter:");
    if (!pointCloud) {
        return;
    }

    // comupte/get the point cloud's octree
    ccOctree::Shared octree = pointCloud->getOctree();
    if (!octree) {
        octree = pointCloud->computeOctree();
        if (!octree) {
            ecvConsole::Error("Failed to compute the octree!");
            return;
        }
    }
    assert(octree);

    // filter octree then project the points
    std::vector<unsigned> inCameraFrustum;
    if (!octree->intersectWithFrustum(sensor, inCameraFrustum)) {
        ecvConsole::Error("Failed to intersect sensor frustum with octree!");
    } else {
        // scalar field
        const char sfName[] = "Frustum visibility";
        int sfIdx = pointCloud->getScalarFieldIndexByName(sfName);

        if (inCameraFrustum.empty()) {
            ecvConsole::Error("No point fell inside the frustum!");
            if (sfIdx >= 0) pointCloud->deleteScalarField(sfIdx);
        } else {
            if (sfIdx < 0) sfIdx = pointCloud->addScalarField(sfName);
            if (sfIdx < 0) {
                CVLog::Error(
                        "Failed to allocate memory for output scalar field!");
                return;
            }

            cloudViewer::ScalarField* sf = pointCloud->getScalarField(sfIdx);
            assert(sf);
            if (sf) {
                sf->fill(0);

                const ScalarType c_insideValue = static_cast<ScalarType>(1);

                for (unsigned index : inCameraFrustum) {
                    sf->setValue(index, c_insideValue);
                }

                sf->computeMinAndMax();
                pointCloud->setCurrentDisplayedScalarField(sfIdx);
                pointCloud->showSF(true);
                refreshObject(pointCloud);
            }
        }
    }

    updateUI();
}

void MainWindow::doActionShowDepthBuffer() {
    if (!haveSelection()) return;

    for (ccHObject* entity : getSelectedEntities()) {
        if (entity->isKindOf(CV_TYPES::GBL_SENSOR)) {
            ccGBLSensor* sensor =
                    static_cast<ccGBLSensor*>(m_selectedEntities[0]);
            if (sensor->getDepthBuffer().zBuff.empty()) {
                // look for depending cloud
                ccGenericPointCloud* cloud =
                        ccHObjectCaster::ToGenericPointCloud(
                                entity->getParent());
                if (cloud) {
                    // force depth buffer computation
                    int errorCode;
                    if (!sensor->computeDepthBuffer(cloud, errorCode)) {
                        ecvConsole::Error(
                                ccGBLSensor::GetErrorString(errorCode));
                    }
                } else {
                    ecvConsole::Error(QString("Internal error: sensor ('%1') "
                                              "parent is not a point cloud!")
                                              .arg(sensor->getName()));
                    return;
                }
            }

            ccRenderingTools::ShowDepthBuffer(sensor, this);
        }
    }
}

void MainWindow::doActionExportDepthBuffer() {
    if (!haveSelection()) return;

    // persistent settings
    QSettings settings;
    settings.beginGroup(ecvPS::SaveFile());
    QString currentPath =
            settings.value(ecvPS::CurrentPath(), ecvFileUtils::defaultDocPath())
                    .toString();

    QString filename = QFileDialog::getSaveFileName(
            this, "Select output file", currentPath,
            DepthMapFileFilter::GetFileFilter(), nullptr,
            ECVFileDialogOptions());
    if (filename.isEmpty()) {
        // process cancelled by user
        return;
    }

    // save last saving location
    settings.setValue(ecvPS::CurrentPath(), QFileInfo(filename).absolutePath());
    settings.endGroup();

    ccHObject* toSave = nullptr;
    bool multEntities = false;
    if (haveOneSelection()) {
        toSave = m_selectedEntities.front();
    } else {
        toSave = new ccHObject("Temp Group");

        for (ccHObject* entity : getSelectedEntities()) {
            toSave->addChild(entity, ccHObject::DP_NONE);
        }
        multEntities = true;
    }

    DepthMapFileFilter::SaveParameters parameters;
    { parameters.alwaysDisplaySaveDialog = true; }
    CC_FILE_ERROR result =
            DepthMapFileFilter().saveToFile(toSave, filename, parameters);

    if (result != CC_FERR_NO_ERROR) {
        FileIOFilter::DisplayErrorMessage(result, "saving", filename);
    } else {
        CVLog::Print(
                QString("[I/O] File '%1' saved successfully").arg(filename));
    }

    if (multEntities) {
        delete toSave;
        toSave = nullptr;
    }
}

void MainWindow::doActionComputePointsVisibility() {
    // there should be only one camera sensor in the current selection!
    if (!haveOneSelection() ||
        !m_selectedEntities[0]->isKindOf(CV_TYPES::GBL_SENSOR)) {
        ecvConsole::Error("Select one and only one GBL/TLS sensor!");
        return;
    }

    ccGBLSensor* sensor = ccHObjectCaster::ToGBLSensor(m_selectedEntities[0]);
    if (!sensor) return;

    // we need a cloud to filter!
    ccHObject* defaultCloud =
            sensor->getParent() &&
                            sensor->getParent()->isA(CV_TYPES::POINT_CLOUD)
                    ? sensor->getParent()
                    : 0;
    ccPointCloud* pointCloud =
            askUserToSelectACloud(defaultCloud, "Select a cloud to filter:");
    if (!pointCloud) {
        return;
    }

    if (sensor->getDepthBuffer().zBuff.empty()) {
        if (defaultCloud) {
            // the sensor has no depth buffer, we'll ask the user if he wants to
            // compute it first
            if (QMessageBox::warning(this, "Depth buffer.",
                                     "Sensor has no depth buffer: do you want "
                                     "to compute it now?",
                                     QMessageBox::Yes | QMessageBox::No,
                                     QMessageBox::Yes) == QMessageBox::No) {
                // we can stop then...
                return;
            }

            int errorCode;
            if (sensor->computeDepthBuffer(
                        static_cast<ccPointCloud*>(defaultCloud), errorCode)) {
                ccRenderingTools::ShowDepthBuffer(sensor, this);
            } else {
                ecvConsole::Error(ccGBLSensor::GetErrorString(errorCode));
                return;
            }
        } else {
            ecvConsole::Error(
                    "Sensor has no depth buffer (and no associated cloud?)");
            return;
        }
    }

    // scalar field
    const char sfName[] = "Sensor visibility";
    int sfIdx = pointCloud->getScalarFieldIndexByName(sfName);
    if (sfIdx < 0) sfIdx = pointCloud->addScalarField(sfName);
    if (sfIdx < 0) {
        CVLog::Error("Failed to allocate memory for output scalar field!");
        return;
    }

    cloudViewer::ScalarField* sf = pointCloud->getScalarField(sfIdx);
    assert(sf);
    if (sf) {
        sf->fill(0);

        // progress bar
        ecvProgressDialog pdlg(true);
        cloudViewer::NormalizedProgress nprogress(&pdlg, pointCloud->size());
        pdlg.setMethodTitle(tr("Compute visibility"));
        pdlg.setInfo(tr("Points: %L1").arg(pointCloud->size()));
        pdlg.start();
        QApplication::processEvents();

        for (unsigned i = 0; i < pointCloud->size(); i++) {
            const CCVector3* P = pointCloud->getPoint(i);
            unsigned char visibility = sensor->checkVisibility(*P);
            ScalarType visValue = static_cast<ScalarType>(visibility);

            sf->setValue(i, visValue);

            if (!nprogress.oneStep()) {
                // cancelled by user
                pointCloud->deleteScalarField(sfIdx);
                sf = nullptr;
                break;
            }
        }

        if (sf) {
            sf->computeMinAndMax();
            pointCloud->setCurrentDisplayedScalarField(sfIdx);
            pointCloud->showSF(true);

            ecvConsole::Print(QString("Visibility computed for cloud '%1'")
                                      .arg(pointCloud->getName()));
            ecvConsole::Print(QString("\tVisible = %1").arg(POINT_VISIBLE));
            ecvConsole::Print(QString("\tHidden = %1").arg(POINT_HIDDEN));
            ecvConsole::Print(
                    QString("\tOut of range = %1").arg(POINT_OUT_OF_RANGE));
            ecvConsole::Print(
                    QString("\tOut of fov = %1").arg(POINT_OUT_OF_FOV));
        }
        refreshObject(pointCloud);
    }

    updateUI();
}

void MainWindow::doActionCompressFWFData() {
    for (ccHObject* entity : getSelectedEntities()) {
        if (!entity || !entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
            continue;
        }

        ccPointCloud* cloud = static_cast<ccPointCloud*>(entity);
        cloud->compressFWFData();
    }
}

void MainWindow::doActionShowWaveDialog() {
    if (!haveSelection()) return;

    ccHObject* entity = haveOneSelection() ? m_selectedEntities[0] : nullptr;
    if (!entity || !entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ecvConsole::Error("Select one point cloud!");
        return;
    }

    ccPointCloud* cloud = static_cast<ccPointCloud*>(entity);
    if (!cloud->hasFWF()) {
        ecvConsole::Error("Cloud has no associated waveform information");
        return;
    }

    ccWaveDialog* wDlg = new ccWaveDialog(cloud, m_pickingHub, this);
    wDlg->setAttribute(Qt::WA_DeleteOnClose);
    wDlg->setModal(false);
    wDlg->show();
}

void MainWindow::doActionConvertTextureToColor() {
    if (!ccEntityAction::convertTextureToColor(m_selectedEntities, this))
        return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionSamplePointsOnMesh() {
    static unsigned s_ptsSamplingCount = 1000000;
    static double s_ptsSamplingDensity = 10.0;
    static bool s_ptsSampleNormals = true;
    static bool s_useDensity = false;

    ccPtsSamplingDlg dlg(this);
    // restore last parameters
    dlg.setPointsNumber(s_ptsSamplingCount);
    dlg.setDensityValue(s_ptsSamplingDensity);
    dlg.setGenerateNormals(s_ptsSampleNormals);
    dlg.setUseDensity(s_useDensity);
    if (!dlg.exec()) return;

    ecvProgressDialog pDlg(false, this);
    pDlg.setAutoClose(false);

    bool withNormals = dlg.generateNormals();
    bool withRGB = dlg.interpolateRGB();
    bool withTexture = dlg.interpolateTexture();
    s_useDensity = dlg.useDensity();
    assert(dlg.getPointsNumber() >= 0);
    s_ptsSamplingCount = static_cast<unsigned>(dlg.getPointsNumber());
    s_ptsSamplingDensity = dlg.getDensityValue();
    s_ptsSampleNormals = withNormals;

    bool errors = false;

    for (ccHObject* entity : getSelectedEntities()) {
        if (!entity->isKindOf(CV_TYPES::MESH)) continue;

        ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(entity);
        assert(mesh);

        ccPointCloud* cloud = mesh->samplePoints(
                s_useDensity,
                s_useDensity ? s_ptsSamplingDensity : s_ptsSamplingCount,
                withNormals, withRGB, withTexture, &pDlg);

        if (cloud) {
            cloud->showNormals(false);
            addToDB(cloud);
        } else {
            errors = true;
        }
    }

    if (errors)
        CVLog::Error(
                tr("[doActionSamplePointsOnMesh] Errors occurred during the "
                   "process! Result may be incomplete!"));
}

void MainWindow::doActionSamplePointsOnPolyline() {
    static unsigned s_ptsSamplingCount = 1000;
    static double s_ptsSamplingDensity = 10.0;
    static bool s_useDensity = false;

    ccPtsSamplingDlg dlg(this);
    dlg.setWindowTitle(tr("Points Sampling on polyline"));
    // restore last parameters
    dlg.setPointsNumber(s_ptsSamplingCount);
    dlg.setDensityValue(s_ptsSamplingDensity);
    dlg.setUseDensity(s_useDensity);
    dlg.optionsFrame->setVisible(false);
    if (!dlg.exec()) return;

    assert(dlg.getPointsNumber() >= 0);
    s_ptsSamplingCount = static_cast<unsigned>(dlg.getPointsNumber());
    s_ptsSamplingDensity = dlg.getDensityValue();
    s_useDensity = dlg.useDensity();

    bool errors = false;

    for (ccHObject* entity : getSelectedEntities()) {
        if (!entity->isKindOf(CV_TYPES::POLY_LINE)) continue;

        ccPolyline* poly = ccHObjectCaster::ToPolyline(entity);
        assert(poly);

        ccPointCloud* cloud = poly->samplePoints(
                s_useDensity,
                s_useDensity ? s_ptsSamplingDensity : s_ptsSamplingCount, true);

        if (cloud) {
            addToDB(cloud);
        } else {
            errors = true;
        }
    }

    if (errors) {
        CVLog::Error(
                tr("[doActionSamplePointsOnPolyline] Errors occurred during "
                   "the process! Result may be incomplete!"));
    }
}

void MainWindow::doActionSmoohPolyline() {
    static int s_iterationCount = 5;
    static double s_ratio = 0.25;

    ccSmoothPolylineDialog dlg(this);
    // restore last parameters
    dlg.setIerationCount(s_iterationCount);
    dlg.setRatio(s_ratio);
    if (!dlg.exec()) return;

    s_iterationCount = dlg.getIerationCount();
    s_ratio = dlg.getRatio();

    bool errors = false;

    ccHObject::Container selectedEntities = getSelectedEntities();
    m_ccRoot->unselectAllEntities();

    for (ccHObject* entity : selectedEntities) {
        if (!entity->isKindOf(CV_TYPES::POLY_LINE)) continue;

        ccPolyline* poly = ccHObjectCaster::ToPolyline(entity);
        assert(poly);

        ccPolyline* smoothPoly = poly->smoothChaikin(
                s_ratio, static_cast<unsigned>(s_iterationCount));
        if (smoothPoly) {
            if (poly->getParent()) {
                poly->getParent()->addChild(smoothPoly);
            }
            poly->setEnabled(false);
            addToDB(smoothPoly);

            m_ccRoot->selectEntity(smoothPoly, true);
        } else {
            errors = true;
        }
    }

    if (errors) {
        CVLog::Error(
                tr("[DoActionSmoohPolyline] Errors occurred during the "
                   "process! Result may be incomplete!"));
    }

    refreshAll();
}

void MainWindow::doConvertPolylinesToMesh() {
    if (!haveSelection()) return;

    std::vector<ccPolyline*> polylines;
    try {
        if (haveOneSelection() &&
            m_selectedEntities.back()->isA(CV_TYPES::HIERARCHY_OBJECT)) {
            ccHObject* obj = m_selectedEntities.back();
            for (unsigned i = 0; i < obj->getChildrenNumber(); ++i) {
                if (obj->getChild(i)->isA(CV_TYPES::POLY_LINE))
                    polylines.push_back(
                            static_cast<ccPolyline*>(obj->getChild(i)));
            }
        } else {
            for (ccHObject* entity : getSelectedEntities()) {
                if (entity->isA(CV_TYPES::POLY_LINE)) {
                    polylines.push_back(static_cast<ccPolyline*>(entity));
                }
            }
        }
    } catch (const std::bad_alloc&) {
        ecvConsole::Error(tr("Not enough memory!"));
        return;
    }

    if (polylines.empty()) {
        ecvConsole::Error(
                tr("Select a group of polylines or multiple polylines (contour "
                   "plot)!"));
        return;
    }

    ccPickOneElementDlg poeDlg(tr("Projection dimension"),
                               tr("Contour plot to mesh"), this);
    poeDlg.addElement("X");
    poeDlg.addElement("Y");
    poeDlg.addElement("Z");
    poeDlg.setDefaultIndex(2);
    if (!poeDlg.exec()) return;

    int dim = poeDlg.getSelectedIndex();
    assert(dim >= 0 && dim < 3);

    const unsigned char Z = static_cast<unsigned char>(dim);
    const unsigned char X = Z == 2 ? 0 : Z + 1;
    const unsigned char Y = X == 2 ? 0 : X + 1;

    // number of segments
    unsigned segmentCount = 0;
    unsigned vertexCount = 0;
    {
        for (ccPolyline* poly : polylines) {
            if (poly) {
                // count the total number of vertices and segments
                vertexCount += poly->size();
                segmentCount += poly->segmentCount();
            }
        }
    }

    if (segmentCount < 2) {
        // not enough points/segments
        CVLog::Error(tr("Not enough segments!"));
        return;
    }

    // we assume we link with CGAL now (if not the call to
    // Delaunay2dMesh::buildMesh will fail anyway)
    std::vector<CCVector2> points2D;
    std::vector<int> segments2D;
    try {
        points2D.reserve(vertexCount);
        segments2D.reserve(segmentCount * 2);
    } catch (const std::bad_alloc&) {
        // not enough memory
        CVLog::Error(tr("Not enough memory"));
        return;
    }

    // fill arrays
    {
        for (ccPolyline* poly : polylines) {
            if (poly == nullptr) continue;

            unsigned vertCount = poly->size();
            int vertIndex0 = static_cast<int>(points2D.size());
            bool closed = poly->isClosed();
            for (unsigned v = 0; v < vertCount; ++v) {
                const CCVector3* P = poly->getPoint(v);
                int vertIndex = static_cast<int>(points2D.size());
                points2D.push_back(CCVector2(P->u[X], P->u[Y]));

                if (v + 1 < vertCount) {
                    segments2D.push_back(vertIndex);
                    segments2D.push_back(vertIndex + 1);
                } else if (closed) {
                    segments2D.push_back(vertIndex);
                    segments2D.push_back(vertIndex0);
                }
            }
        }
        assert(points2D.size() == vertexCount);
        assert(segments2D.size() == segmentCount * 2);
    }

    cloudViewer::Delaunay2dMesh* delaunayMesh = new cloudViewer::Delaunay2dMesh;
    std::string errorStr;
    if (!delaunayMesh->buildMesh(points2D, segments2D, errorStr)) {
        CVLog::Error(tr("Third party library error: %1")
                             .arg(QString::fromStdString(errorStr)));
        delete delaunayMesh;
        return;
    }

    ccPointCloud* vertices = new ccPointCloud(tr("vertices"));
    if (!vertices->reserve(vertexCount)) {
        // not enough memory
        CVLog::Error(tr("Not enough memory"));
        delete vertices;
        delete delaunayMesh;
        return;
    }

    // fill vertices cloud
    {
        for (ccPolyline* poly : polylines) {
            unsigned vertCount = poly->size();
            for (unsigned v = 0; v < vertCount; ++v) {
                const CCVector3* P = poly->getPoint(v);
                vertices->addPoint(*P);
            }
        }
        delaunayMesh->linkMeshWith(vertices, false);
    }

#ifdef QT_DEBUG
    // Test delaunay output
    {
        unsigned vertCount = vertices->size();
        for (unsigned i = 0; i < delaunayMesh->size(); ++i) {
            const cloudViewer::VerticesIndexes* tsi =
                    delaunayMesh->getTriangleVertIndexes(i);
            assert(tsi->i1 < vertCount && tsi->i2 < vertCount &&
                   tsi->i3 < vertCount);
        }
    }
#endif

    ccMesh* mesh = new ccMesh(delaunayMesh, vertices);
    if (mesh->size() != delaunayMesh->size()) {
        // not enough memory (error will be issued later)
        delete mesh;
        mesh = nullptr;
    }

    // don't need this anymore
    delete delaunayMesh;
    delaunayMesh = nullptr;

    if (mesh) {
        mesh->addChild(vertices);
        mesh->setVisible(true);
        vertices->setEnabled(false);

        if (mesh->computePerVertexNormals()) {
            mesh->showNormals(true);
        } else {
            CVLog::Warning(
                    tr("[Contour plot to mesh] Failed to compute normals!"));
        }

        // global shift & scale (we copy it from the first polyline by default)
        vertices->setGlobalShift(polylines.front()->getGlobalShift());
        vertices->setGlobalScale(polylines.front()->getGlobalScale());

        addToDB(mesh);
    } else {
        CVLog::Error(tr("Not enough memory!"));
        delete vertices;
        vertices = nullptr;
    }
}

void MainWindow::doBSplineFittingFromCloud() {
#ifdef USE_PCL_BACKEND
    // find candidates
    std::vector<ccPointCloud*> clouds;
    {
        for (ccHObject* entity : getSelectedEntities()) {
            if (entity->isA(CV_TYPES::POINT_CLOUD)) {
                ccPointCloud* cloud = static_cast<ccPointCloud*>(entity);
                clouds.push_back(cloud);
            }
        }
    }

    if (clouds.empty()) {
        ecvConsole::Error(tr("Select at least one point cloud!"));
        return;
    }

    ccPolyline* polyLine =
            CurveFittingTool::CurveFitting::BsplineFitting(*clouds[0]);
    if (polyLine) {
        addToDB(polyLine);
    }

    if (polyLine && m_ccRoot) {
        m_ccRoot->selectEntity(polyLine);
    }

    updateUI();

#else
    CVLog::Warning(
            "[doBSplineFittingFromCloud] please use pcl as backend and then "
            "try again!");
    return;
#endif
}

void MainWindow::doActionSmoothMeshSF() {
    if (!ccEntityAction::processMeshSF(m_selectedEntities,
                                       ccMesh::SMOOTH_MESH_SF, this))
        return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionEnhanceMeshSF() {
    if (!ccEntityAction::processMeshSF(m_selectedEntities,
                                       ccMesh::ENHANCE_MESH_SF, this))
        return;

    refreshSelected();
    updateUI();
}

static double s_subdivideMaxArea = 1.0;
void MainWindow::doActionSubdivideMesh() {
    bool ok;
    s_subdivideMaxArea = QInputDialog::getDouble(
            this, tr("Subdivide mesh"), tr("Max area per triangle:"),
            s_subdivideMaxArea, 1e-6, 1e6, 8, &ok);
    if (!ok) return;

    for (ccHObject* entity : getSelectedEntities()) {
        if (entity->isKindOf(CV_TYPES::MESH)) {
            // single mesh?
            if (entity->isA(CV_TYPES::MESH)) {
                ccMesh* mesh = static_cast<ccMesh*>(entity);
                // avoid rendering this object this time
                mesh->setEnabled(false);

                ccMesh* subdividedMesh = nullptr;
                try {
                    subdividedMesh =
                            mesh->subdivide(static_cast<PointCoordinateType>(
                                    s_subdivideMaxArea));
                } catch (...) {
                    CVLog::Error(
                            tr("[Subdivide] An error occurred while trying to "
                               "subdivide mesh '%1' (not enough memory?)")
                                    .arg(mesh->getName()));
                }

                if (subdividedMesh) {
                    subdividedMesh->setName(tr("%1.subdivided(S<%2)")
                                                    .arg(mesh->getName())
                                                    .arg(s_subdivideMaxArea));
                    refreshObject(mesh, true, true);
                    // avoid rendering this object this time
                    mesh->setEnabled(false);
                    addToDB(subdividedMesh);
                } else {
                    ecvConsole::Warning(tr("[Subdivide] Failed to subdivide "
                                           "mesh '%1' (not enough memory?)")
                                                .arg(mesh->getName()));
                }
            } else {
                CVLog::Warning(tr("[Subdivide] Works only on real meshes!"));
            }
        }
    }

    updateUI();
}

void MainWindow::doActionFlipMeshTriangles() {
    bool warningIssued = false;
    ecvDisplayTools::SetRedrawRecursive(false);
    for (ccHObject* entity : getSelectedEntities()) {
        if (entity->isKindOf(CV_TYPES::MESH)) {
            // single mesh?
            if (entity->isA(CV_TYPES::MESH)) {
                ccMesh* mesh = static_cast<ccMesh*>(entity);
                mesh->flipTriangles();
                mesh->setRedrawFlagRecursive(true);
            } else if (!warningIssued) {
                CVLog::Warning("[Flip triangles] Works only on real meshes!");
                warningIssued = true;
            }
        }
    }

    refreshAll();
}

void MainWindow::doActionSmoothMeshLaplacian() {
    static unsigned s_laplacianSmooth_nbIter = 20;
    static double s_laplacianSmooth_factor = 0.2;

    bool ok;
    s_laplacianSmooth_nbIter =
            QInputDialog::getInt(this, tr("Smooth mesh"), tr("Iterations:"),
                                 s_laplacianSmooth_nbIter, 1, 1000, 1, &ok);
    if (!ok) return;
    s_laplacianSmooth_factor = QInputDialog::getDouble(
            this, tr("Smooth mesh"), tr("Smoothing factor:"),
            s_laplacianSmooth_factor, 0, 100, 3, &ok);
    if (!ok) return;

    ecvProgressDialog pDlg(true, this);
    pDlg.setAutoClose(false);

    ecvDisplayTools::SetRedrawRecursive(false);
    for (ccHObject* entity : getSelectedEntities()) {
        if (entity->isA(CV_TYPES::MESH) ||
            entity->isA(CV_TYPES::PRIMITIVE))  // FIXME: can we really do this
                                               // with primitives?
        {
            ccMesh* mesh = ccHObjectCaster::ToMesh(entity);

            if (mesh->laplacianSmooth(s_laplacianSmooth_nbIter,
                                      static_cast<PointCoordinateType>(
                                              s_laplacianSmooth_factor),
                                      &pDlg)) {
                mesh->setRedrawFlagRecursive(true);
            } else {
                ecvConsole::Warning(
                        tr("Failed to apply Laplacian smoothing to mesh '%1'")
                                .arg(mesh->getName()));
            }
        }
    }

    refreshAll();
    updateUI();
}

void MainWindow::doActionFlagMeshVertices() {
    bool errors = false;
    bool success = false;

    ecvDisplayTools::SetRedrawRecursive(false);
    for (ccHObject* entity : getSelectedEntities()) {
        if (entity->isKindOf(CV_TYPES::MESH)) {
            ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(entity);
            ccPointCloud* vertices = ccHObjectCaster::ToPointCloud(
                    mesh ? mesh->getAssociatedCloud() : 0);
            if (mesh && vertices) {
                // prepare a new scalar field
                int sfIdx = vertices->getScalarFieldIndexByName(
                        CC_DEFAULT_MESH_VERT_FLAGS_SF_NAME);
                if (sfIdx < 0) {
                    sfIdx = vertices->addScalarField(
                            CC_DEFAULT_MESH_VERT_FLAGS_SF_NAME);
                    if (sfIdx < 0) {
                        ecvConsole::Warning(tr("Not enough memory to flag the "
                                               "vertices of mesh '%1'!")
                                                    .arg(mesh->getName()));
                        errors = true;
                        continue;
                    }
                }
                cloudViewer::ScalarField* flags =
                        vertices->getScalarField(sfIdx);

                cloudViewer::MeshSamplingTools::EdgeConnectivityStats stats;
                if (cloudViewer::MeshSamplingTools::flagMeshVerticesByType(
                            mesh, flags, &stats)) {
                    vertices->setCurrentDisplayedScalarField(sfIdx);
                    ccScalarField* sf =
                            vertices->getCurrentDisplayedScalarField();
                    if (sf) {
                        sf->setColorScale(ccColorScalesManager::GetDefaultScale(
                                ccColorScalesManager::VERTEX_QUALITY));
                        // sf->setColorRampSteps(3); //ugly :(
                    }
                    vertices->showSF(true);
                    mesh->showSF(true);
                    mesh->setRedrawFlagRecursive(true);
                    success = true;

                    // display stats in the Console as well
                    ecvConsole::Print(tr("[Mesh Quality] Mesh '%1' edges: %2 "
                                         "total (normal: %3 / on hole borders: "
                                         "%4 / non-manifold: %5)")
                                              .arg(entity->getName())
                                              .arg(stats.edgesCount)
                                              .arg(stats.edgesSharedByTwo)
                                              .arg(stats.edgesNotShared)
                                              .arg(stats.edgesSharedByMore));
                } else {
                    vertices->deleteScalarField(sfIdx);
                    sfIdx = -1;
                    ecvConsole::Warning(tr("Not enough memory to flag the "
                                           "vertices of mesh '%1'!")
                                                .arg(mesh->getName()));
                    errors = true;
                }
            } else {
                assert(false);
            }
        }
    }

    refreshAll();
    updateUI();

    if (success) {
        // display reminder
        forceConsoleDisplay();
        ecvConsole::Print(
                tr("[Mesh Quality] SF flags: %1 (NORMAL) / %2 (BORDER) / (%3) "
                   "NON-MANIFOLD")
                        .arg(cloudViewer::MeshSamplingTools::VERTEX_NORMAL)
                        .arg(cloudViewer::MeshSamplingTools::VERTEX_BORDER)
                        .arg(cloudViewer::MeshSamplingTools::
                                     VERTEX_NON_MANIFOLD));
    }

    if (errors) {
        ecvConsole::Error(tr("Error(s) occurred! Check the console..."));
    }
}

void MainWindow::doActionMeasureMeshVolume() {
    for (ccHObject* entity : getSelectedEntities()) {
        if (entity->isKindOf(CV_TYPES::MESH)) {
            ccMesh* mesh = ccHObjectCaster::ToMesh(entity);
            if (mesh) {
                // we compute the mesh volume
                double V =
                        cloudViewer::MeshSamplingTools::computeMeshVolume(mesh);
                // we force the console to display itself
                forceConsoleDisplay();
                ecvConsole::Print(
                        tr("[Mesh Volume] Mesh '%1': V=%2 (cube units)")
                                .arg(entity->getName())
                                .arg(V));

                // check that the mesh is closed
                cloudViewer::MeshSamplingTools::EdgeConnectivityStats stats;
                if (cloudViewer::MeshSamplingTools::
                            computeMeshEdgesConnectivity(mesh, stats)) {
                    if (stats.edgesNotShared != 0) {
                        ecvConsole::Warning(
                                tr("[Mesh Volume] The above volume might be "
                                   "invalid (mesh has holes)"));
                    } else if (stats.edgesSharedByMore != 0) {
                        ecvConsole::Warning(
                                tr("[Mesh Volume] The above volume might be "
                                   "invalid (mesh has non-manifold edges)"));
                    }
                } else {
                    ecvConsole::Warning(
                            tr("[Mesh Volume] The above volume might be "
                               "invalid (not enough memory to check if the "
                               "mesh is closed)"));
                }
            } else {
                assert(false);
            }
        }
    }
}

void MainWindow::doActionMeasureMeshSurface() {
    for (ccHObject* entity : getSelectedEntities()) {
        if (entity->isKindOf(CV_TYPES::MESH)) {
            ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(entity);
            if (mesh) {
                double S =
                        cloudViewer::MeshSamplingTools::computeMeshArea(mesh);
                // we force the console to display itself
                forceConsoleDisplay();
                ecvConsole::Print(
                        tr("[Mesh Surface] Mesh '%1': S=%2 (square units)")
                                .arg(entity->getName())
                                .arg(S));
                if (mesh->size()) {
                    ecvConsole::Print(tr("[Mesh Surface] Average triangle "
                                         "surface: %1 (square units)")
                                              .arg(S / double(mesh->size())));
                }
            } else {
                assert(false);
            }
        }
    }
}

void MainWindow::doActionCreatePlane() {
    ccPlaneEditDlg* peDlg = new ccPlaneEditDlg(m_pickingHub, this);
    peDlg->show();
}

void MainWindow::doActionEditPlane() {
    if (!haveSelection()) {
        assert(false);
        return;
    }

    ccPlane* plane = ccHObjectCaster::ToPlane(m_selectedEntities.front());
    if (!plane) {
        assert(false);
        return;
    }

    ccPlaneEditDlg* peDlg = new ccPlaneEditDlg(m_pickingHub, this);
    peDlg->initWithPlane(plane);
    peDlg->show();
}

void MainWindow::doActionFlipPlane() {
    if (!haveSelection()) {
        assert(false);
        return;
    }

    ecvDisplayTools::SetRedrawRecursive(false);
    for (ccHObject* entity : m_selectedEntities) {
        ccPlane* plane = ccHObjectCaster::ToPlane(entity);
        if (plane) {
            plane->flip();
            plane->setRedrawFlagRecursive(true);
        }
    }

    refreshAll();
    updatePropertiesView();
}

void MainWindow::doActionPromoteCircleToCylinder() {
    if (!haveOneSelection()) {
        assert(false);
        return;
    }

    ccCircle* circle = ccHObjectCaster::ToCircle(m_selectedEntities.front());
    if (!circle) {
        assert(false);
        return;
    }

    static double CylinderHeight = 0.0;
    if (CylinderHeight == 0.0) {
        CylinderHeight = 2 * circle->getRadius();
    }
    bool ok = false;
    double value = QInputDialog::getDouble(
            this, tr("Cylinder height"), tr("Height"), CylinderHeight, 0.0,
            std::numeric_limits<double>::max(), 6, &ok);
    if (!ok) {
        return;
    }

    CylinderHeight = value;

    ccCylinder* cylinder = new ccCylinder(
            static_cast<PointCoordinateType>(circle->getRadius()),
            static_cast<PointCoordinateType>(CylinderHeight),
            &circle->getGLTransformationHistory(),
            tr("Cylinder from ") + circle->getName());

    circle->setEnabled(false);
    if (circle->getParent()) {
        circle->getParent()->addChild(cylinder);
    }

    addToDB(cylinder, true, true);
    setSelectedInDB(circle, false);
    setSelectedInDB(cylinder, true);
}

void MainWindow::doActionComparePlanes() {
    if (m_selectedEntities.size() != 2) {
        ecvConsole::Error("Select 2 planes!");
        return;
    }

    if (!m_selectedEntities[0]->isKindOf(CV_TYPES::PLANE) ||
        !m_selectedEntities[1]->isKindOf(CV_TYPES::PLANE)) {
        ecvConsole::Error("Select 2 planes!");
        return;
    }

    ccPlane* p1 = ccHObjectCaster::ToPlane(m_selectedEntities[0]);
    ccPlane* p2 = ccHObjectCaster::ToPlane(m_selectedEntities[1]);

    QStringList info;
    info << QString("Plane 1: %1").arg(p1->getName());
    CVLog::Print(QString("[Compare] ") + info.last());

    info << QString("Plane 2: %1").arg(p2->getName());
    CVLog::Print(QString("[Compare] ") + info.last());

    CCVector3 N1;
    CCVector3 N2;
    PointCoordinateType d1;
    PointCoordinateType d2;
    p1->getEquation(N1, d1);
    p2->getEquation(N2, d2);

    double angle_rad = N1.angle_rad(N2);
    info << QString("Angle P1/P2: %1 deg.")
                    .arg(cloudViewer::RadiansToDegrees(angle_rad));
    CVLog::Print(QString("[Compare] ") + info.last());

    PointCoordinateType planeEq1[4] = {N1.x, N1.y, N1.z, d1};
    PointCoordinateType planeEq2[4] = {N2.x, N2.y, N2.z, d2};
    CCVector3 C1 = p1->getCenter();
    ScalarType distCenter1ToPlane2 =
            cloudViewer::DistanceComputationTools::computePoint2PlaneDistance(
                    &C1, planeEq2);
    info << QString("Distance Center(P1)/P2: %1").arg(distCenter1ToPlane2);
    CVLog::Print(QString("[Compare] ") + info.last());

    CCVector3 C2 = p2->getCenter();
    ScalarType distCenter2ToPlane1 =
            cloudViewer::DistanceComputationTools::computePoint2PlaneDistance(
                    &C2, planeEq1);
    info << QString("Distance Center(P2)/P1: %1").arg(distCenter2ToPlane1);
    CVLog::Print(QString("[Compare] ") + info.last());

    // pop-up summary
    QMessageBox::information(this, "Plane comparison", info.join("\n"));
    forceConsoleDisplay();
}

// help
void MainWindow::help() {
    QDesktopServices::openUrl(
            QUrl(QStringLiteral("https://asher-1.github.io/docs")));
    ecvConsole::Print(
            tr("[ACloudViewer help] https://asher-1.github.io/docs!"));
}

// Change theme: Windows/Darcula
void MainWindow::changeTheme() {
    QAction* action = qobject_cast<QAction*>(sender());
    QVariant v = action->data();

    QString qssfile = (QString)v.value<QString>();
    ChangeStyle(qssfile);

    // persistent saving
    ecvSettingManager::setValue(ecvPS::ThemeSettings(), ecvPS::CurrentTheme(),
                                qssfile);
}

// Change language: English/Chinese
void MainWindow::changeLanguage() {
    QAction* action = qobject_cast<QAction*>(sender());
    QVariant v = action->data();
    int language = (int)v.value<int>();

    switch (language) {
        case CLOUDVIEWER_LANG_ENGLISH: {
            ecvConsole::Print(
                    tr("[changeLanguage] Change to English language"));
            break;
        }
        case CLOUDVIEWER_LANG_CHINESE: {
            ecvConsole::Warning(
                    tr("[changeLanguage] Doesn't support Chinese temporarily"));
            break;
        }
    }
}

void MainWindow::doActionGlobalShiftSeetings() {
    QDialog dialog(this);
    Ui_GlobalShiftSettingsDialog ui;
    ui.setupUi(&dialog);

    ui.maxAbsCoordSpinBox->setValue(static_cast<int>(
            log10(ecvGlobalShiftManager::MaxCoordinateAbsValue())));
    ui.maxAbsDiagSpinBox->setValue(static_cast<int>(
            log10(ecvGlobalShiftManager::MaxBoundgBoxDiagonal())));

    if (!dialog.exec()) {
        return;
    }

    double maxAbsCoord =
            pow(10.0, static_cast<double>(ui.maxAbsCoordSpinBox->value()));
    double maxAbsDiag =
            pow(10.0, static_cast<double>(ui.maxAbsDiagSpinBox->value()));

    ecvGlobalShiftManager::SetMaxCoordinateAbsValue(maxAbsCoord);
    ecvGlobalShiftManager::SetMaxBoundgBoxDiagonal(maxAbsDiag);

    CVLog::Print(tr("[Global Shift] Max abs. coord = %1 / max abs. diag = %2")
                         .arg(ecvGlobalShiftManager::MaxCoordinateAbsValue(), 0,
                              'e', 0)
                         .arg(ecvGlobalShiftManager::MaxBoundgBoxDiagonal(), 0,
                              'e', 0));

    // save to persistent settings
    {
        ecvSettingManager::setValue(ecvPS::GlobalShift(), ecvPS::MaxAbsCoord(),
                                    maxAbsCoord);
        ecvSettingManager::setValue(ecvPS::GlobalShift(), ecvPS::MaxAbsDiag(),
                                    maxAbsDiag);
    }
}

ccPointCloud* MainWindow::askUserToSelectACloud(ccHObject* defaultCloudEntity,
                                                QString inviteMessage) {
    ccHObject::Container clouds;
    m_ccRoot->getRootEntity()->filterChildren(clouds, true,
                                              CV_TYPES::POINT_CLOUD, true);
    if (clouds.empty()) {
        ecvConsole::Error(tr("No cloud in database!"));
        return 0;
    }
    // default selected index
    int selectedIndex = 0;
    if (defaultCloudEntity) {
        for (size_t i = 1; i < clouds.size(); ++i) {
            if (clouds[i] == defaultCloudEntity) {
                selectedIndex = static_cast<int>(i);
                break;
            }
        }
    }
    // ask the user to choose a cloud
    {
        selectedIndex = ccItemSelectionDlg::SelectEntity(clouds, selectedIndex,
                                                         this, inviteMessage);
        if (selectedIndex < 0) return 0;
    }

    assert(selectedIndex >= 0 &&
           static_cast<size_t>(selectedIndex) < clouds.size());
    return ccHObjectCaster::ToPointCloud(clouds[selectedIndex]);
}

void MainWindow::toggleSelectedEntitiesProperty(
        ccEntityAction::TOGGLE_PROPERTY property) {
    if (!ccEntityAction::toggleProperty(m_selectedEntities, property)) {
        return;
    }

    refreshSelected();
    updateUI();
}

void MainWindow::clearSelectedEntitiesProperty(
        ccEntityAction::CLEAR_PROPERTY property) {
    ecvDisplayTools::SetRedrawRecursive(false);
    if (!ccEntityAction::clearProperty(m_selectedEntities, property, this)) {
        return;
    }

    refreshSelected();
    updateUI();
}

void MainWindow::doActionFastRegistration(FastRegistrationMode mode) {
    // we need at least 1 entity
    if (m_selectedEntities.empty()) return;

    // we must backup 'm_selectedEntities' as removeObjectTemporarilyFromDBTree
    // can modify it!
    ccHObject::Container selectedEntities = m_selectedEntities;

    for (ccHObject* entity : selectedEntities) {
        ccBBox box = entity->getBB_recursive();

        CCVector3 T;  // translation

        switch (mode) {
            case MoveBBCenterToOrigin:
                T = -box.getCenter();
                break;
            case MoveBBMinCornerToOrigin:
                T = -box.minCorner();
                break;
            case MoveBBMaxCornerToOrigin:
                T = -box.maxCorner();
                break;
            default:
                assert(false);
                return;
        }

        // transformation (used only for translation)
        ccGLMatrix glTrans;
        glTrans.setTranslation(T);

        forceConsoleDisplay();
        ecvConsole::Print(QString("[Synchronize] Transformation matrix (%1):")
                                  .arg(entity->getName()));
        ecvConsole::Print(glTrans.toString(12, ' '));  // full precision
        ecvConsole::Print(
                "Hint: copy it (CTRL+C) and apply it - or its inverse - on any "
                "entity with the 'Edit > Apply transformation' tool");

        // we temporarily detach entity, as it may undergo
        //"severe" modifications (octree deletion, etc.) --> see
        // ccHObject::applyGLTransformation
        ccHObjectContext objContext = removeObjectTemporarilyFromDBTree(entity);
        entity->applyGLTransformation_recursive(&glTrans);
        putObjectBackIntoDBTree(entity, objContext);
    }

    // reselect previously selected entities!
    if (m_ccRoot) m_ccRoot->selectEntities(selectedEntities);

    refreshSelected();
    zoomOnSelectedEntities();

    updateUI();
}

void MainWindow::doActionColorize() { doActionSetColor(true); }

void MainWindow::doActionSetUniqueColor() { doActionSetColor(false); }

void MainWindow::doActionSetColor(bool colorize) {
    ecvDisplayTools::SetRedrawRecursive(false);
    if (!ccEntityAction::setColor(m_selectedEntities, colorize, this)) return;

    updateUI();
}

void MainWindow::doActionRGBToGreyScale() {
    if (!ccEntityAction::rgbToGreyScale(m_selectedEntities)) return;

    refreshSelected();
}

void MainWindow::doActionSetColorGradient() {
    if (!ccEntityAction::setColorGradient(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionChangeColorLevels() {
    ccEntityAction::changeColorLevels(m_selectedEntities, this);
    refreshSelected();
}

void MainWindow::doActionInterpolateColors() {
    if (!ccEntityAction::interpolateColors(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionEnhanceRGBWithIntensities() {
    if (!ccEntityAction::enhanceRGBWithIntensities(m_selectedEntities, this))
        return;

    refreshSelected();
}

void MainWindow::doActionColorFromScalars() {
    for (ccHObject* entity : getSelectedEntities()) {
        // for "real" point clouds only
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(entity);
        if (cloud) {
            // create color from scalar dialogue
            ccColorFromScalarDlg* cfsDlg =
                    new ccColorFromScalarDlg(this, cloud);
            cfsDlg->setAttribute(Qt::WA_DeleteOnClose, true);
            cfsDlg->show();
        }
    }
}

void MainWindow::doActionSORFilter() {
    ecvSORFilterDlg sorDlg(this);

    // set semi-persistent/dynamic parameters
    static int s_sorFilterKnn = 6;
    static double s_sorFilterNSigma = 1.0;
    sorDlg.knnSpinBox->setValue(s_sorFilterKnn);
    sorDlg.nSigmaDoubleSpinBox->setValue(s_sorFilterNSigma);
    if (!sorDlg.exec()) return;

    // update semi-persistent/dynamic parameters
    s_sorFilterKnn = sorDlg.knnSpinBox->value();
    s_sorFilterNSigma = sorDlg.nSigmaDoubleSpinBox->value();

    ecvProgressDialog pDlg(true, this);
    pDlg.setAutoClose(false);

    bool firstCloud = true;

    ccHObject::Container selectedEntities =
            getSelectedEntities();  // we have to use a local copy:
                                    // 'selectEntity' will change the set of
                                    // currently selected entities!

    for (ccHObject* entity : selectedEntities) {
        // specific test for locked vertices
        bool lockedVertices;
        ccPointCloud* cloud =
                ccHObjectCaster::ToPointCloud(entity, &lockedVertices);
        if (cloud && lockedVertices) {
            ecvUtils::DisplayLockedVerticesWarning(entity->getName(),
                                                   haveOneSelection());
            continue;
        }

        // computation
        cloudViewer::ReferenceCloud* selection =
                cloudViewer::CloudSamplingTools::sorFilter(
                        cloud, s_sorFilterKnn, s_sorFilterNSigma, 0, &pDlg);

        if (selection && cloud) {
            if (selection->size() == cloud->size()) {
                CVLog::Warning(QString("[doActionSORFilter] No points were "
                                       "removed from cloud '%1'")
                                       .arg(cloud->getName()));
            } else {
                ccPointCloud* cleanCloud = cloud->partialClone(selection);
                if (cleanCloud) {
                    cleanCloud->setName(cloud->getName() + QString(".clean"));
                    // cleanCloud->setDisplay(cloud->getDisplay());
                    if (cloud->getParent())
                        cloud->getParent()->addChild(cleanCloud);
                    addToDB(cleanCloud);

                    cloud->setEnabled(false);
                    if (firstCloud) {
                        ecvConsole::Warning(
                                "Previously selected entities (sources) have "
                                "been hidden!");
                        firstCloud = false;
                        m_ccRoot->selectEntity(cleanCloud, true);
                    }
                } else {
                    ecvConsole::Warning(
                            QString("[doActionSORFilter] Not enough memory to "
                                    "create a clean version of cloud '%1'!")
                                    .arg(cloud->getName()));
                }
            }

            delete selection;
            selection = nullptr;
        } else {
            // no points fall inside selection!
            if (cloud != nullptr) {
                ecvConsole::Warning(
                        QString("[doActionSORFilter] Failed to apply the noise "
                                "filter to cloud '%1'! (not enough memory?)")
                                .arg(cloud->getName()));
            } else {
                ecvConsole::Warning(
                        "[doActionSORFilter] Trying to apply the noise filter "
                        "to null cloud");
            }
        }
    }

    updateUI();
}

void MainWindow::doActionFilterNoise() {
    PointCoordinateType kernelRadius =
            ccLibAlgorithms::GetDefaultCloudKernelSize(m_selectedEntities);

    ecvNoiseFilterDlg noiseDlg(this);

    // set semi-persistent/dynamic parameters
    static bool s_noiseFilterUseKnn = false;
    static int s_noiseFilterKnn = 6;
    static bool s_noiseFilterUseAbsError = false;
    static double s_noiseFilterAbsError = 1.0;
    static double s_noiseFilterNSigma = 1.0;
    static bool s_noiseFilterRemoveIsolatedPoints = false;
    noiseDlg.radiusDoubleSpinBox->setValue(kernelRadius);
    noiseDlg.knnSpinBox->setValue(s_noiseFilterKnn);
    noiseDlg.nSigmaDoubleSpinBox->setValue(s_noiseFilterNSigma);
    noiseDlg.absErrorDoubleSpinBox->setValue(s_noiseFilterAbsError);
    noiseDlg.removeIsolatedPointsCheckBox->setChecked(
            s_noiseFilterRemoveIsolatedPoints);
    if (s_noiseFilterUseAbsError)
        noiseDlg.absErrorRadioButton->setChecked(true);
    else
        noiseDlg.relativeRadioButton->setChecked(true);
    if (s_noiseFilterUseKnn)
        noiseDlg.knnRadioButton->setChecked(true);
    else
        noiseDlg.radiusRadioButton->setChecked(true);

    if (!noiseDlg.exec()) return;

    // update semi-persistent/dynamic parameters
    kernelRadius = static_cast<PointCoordinateType>(
            noiseDlg.radiusDoubleSpinBox->value());
    s_noiseFilterUseKnn = noiseDlg.knnRadioButton->isChecked();
    s_noiseFilterKnn = noiseDlg.knnSpinBox->value();
    s_noiseFilterUseAbsError = noiseDlg.absErrorRadioButton->isChecked();
    s_noiseFilterNSigma = noiseDlg.nSigmaDoubleSpinBox->value();
    s_noiseFilterAbsError = noiseDlg.absErrorDoubleSpinBox->value();
    s_noiseFilterRemoveIsolatedPoints =
            noiseDlg.removeIsolatedPointsCheckBox->isChecked();

    ecvProgressDialog pDlg(true, this);
    pDlg.setAutoClose(false);

    bool firstCloud = true;

    ccHObject::Container selectedEntities =
            getSelectedEntities();  // we have to use a local copy: and
                                    // 'selectEntity' will change the set of
                                    // currently selected entities!

    for (ccHObject* entity : selectedEntities) {
        // specific test for locked vertices
        bool lockedVertices;
        ccPointCloud* cloud =
                ccHObjectCaster::ToPointCloud(entity, &lockedVertices);
        if (cloud && lockedVertices) {
            ecvUtils::DisplayLockedVerticesWarning(entity->getName(),
                                                   haveOneSelection());
            continue;
        }

        // computation
        cloudViewer::ReferenceCloud* selection =
                cloudViewer::CloudSamplingTools::noiseFilter(
                        cloud, kernelRadius, s_noiseFilterNSigma,
                        s_noiseFilterRemoveIsolatedPoints, s_noiseFilterUseKnn,
                        s_noiseFilterKnn, s_noiseFilterUseAbsError,
                        s_noiseFilterAbsError, 0, &pDlg);

        if (selection && cloud) {
            if (selection->size() == cloud->size()) {
                CVLog::Warning(QString("[doActionFilterNoise] No points were "
                                       "removed from cloud '%1'")
                                       .arg(cloud->getName()));
            } else {
                ccPointCloud* cleanCloud = cloud->partialClone(selection);
                if (cleanCloud) {
                    cleanCloud->setName(cloud->getName() + QString(".clean"));
                    // cleanCloud->setDisplay(cloud->getDisplay());
                    if (cloud->getParent())
                        cloud->getParent()->addChild(cleanCloud);
                    addToDB(cleanCloud);

                    cloud->setEnabled(false);
                    if (firstCloud) {
                        ecvConsole::Warning(
                                "Previously selected entities (sources) have "
                                "been hidden!");
                        firstCloud = false;
                        m_ccRoot->selectEntity(cleanCloud, true);
                    }
                } else {
                    ecvConsole::Warning(
                            QString("[doActionFilterNoise] Not enough memory "
                                    "to create a clean version of cloud '%1'!")
                                    .arg(cloud->getName()));
                }
            }

            delete selection;
            selection = nullptr;
        } else {
            // no points fall inside selection!
            if (cloud != nullptr) {
                ecvConsole::Warning(QString("[doActionFilterNoise] Failed to "
                                            "apply the noise filter to cloud "
                                            "'%1'! (not enough memory?)")
                                            .arg(cloud->getName()));
            } else {
                ecvConsole::Warning(
                        "[doActionFilterNoise] Trying to apply the noise "
                        "filter to null cloud");
            }
        }
    }

    updateUI();
}

void MainWindow::doActionVoxelSampling() {
    if (!haveSelection()) {
        return;
    }

    ccHObject::Container selectedClouds;
    for (auto ent : getSelectedEntities()) {
        if (!ent->isKindOf(CV_TYPES::POINT_CLOUD)) {
            ecvConsole::Warning("only point cloud is supported!");
            continue;
        }

        bool lockedVertices = false;
        ccPointCloud* pc = ccHObjectCaster::ToPointCloud(ent, &lockedVertices);
        if (!pc || lockedVertices) {
            continue;
        }

        selectedClouds.push_back(ent);
    }

    ccHObject::Container outClouds;
    if (!ccEntityAction::VoxelSampling(selectedClouds, outClouds, this)) {
        ecvConsole::Error(
                "[MainWindow::doActionVoxelSampling] voxel sampling failed!");
        return;
    }

    assert(outClouds.size() == selectedClouds.size());

    bool firstCloud = true;
    for (size_t i = 0; i < outClouds.size(); ++i) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(selectedClouds[i]);
        ccPointCloud* cleanCloud = ccHObjectCaster::ToPointCloud(outClouds[i]);
        if (cleanCloud) {
            cleanCloud->setName(cloud->getName() + QString(".clean"));
            if (cloud->getParent()) cloud->getParent()->addChild(cleanCloud);
            addToDB(cleanCloud);

            CVLog::Print(QString("%1 down sampled from %2 points to %3 points.")
                                 .arg(cloud->getName())
                                 .arg(cloud->size())
                                 .arg(cleanCloud->size()));

            cloud->setEnabled(false);
            if (firstCloud) {
                ecvConsole::Warning(
                        "Previously selected entities (sources) have been "
                        "hidden!");
                firstCloud = false;
                m_ccRoot->selectEntity(cleanCloud, true);
            }
        } else {
            ecvConsole::Warning(
                    QString("[doActionSORFilter] Not enough memory to create a "
                            "clean version of cloud '%1'!")
                            .arg(cloud->getName()));
        }
    }

    updateUI();
}

void MainWindow::doActionClone() {
    ccHObject* lastClone = nullptr;

    for (ccHObject* entity : getSelectedEntities()) {
        ccHObject* clone = nullptr;

        if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
            clone = ccHObjectCaster::ToGenericPointCloud(entity)->clone();
            if (!clone) {
                ecvConsole::Error(tr("An error occurred while cloning cloud %1")
                                          .arg(entity->getName()));
            }
        } else if (entity->isKindOf(CV_TYPES::PRIMITIVE)) {
            clone = static_cast<ccGenericPrimitive*>(entity)->clone();
            if (!clone) {
                ecvConsole::Error(
                        tr("An error occurred while cloning primitive %1")
                                .arg(entity->getName()));
            }
        } else if (entity->isA(CV_TYPES::MESH)) {
            clone = ccHObjectCaster::ToMesh(entity)->cloneMesh();
            if (!clone) {
                ecvConsole::Error(tr("An error occurred while cloning mesh %1")
                                          .arg(entity->getName()));
            }
        } else if (entity->isA(CV_TYPES::POLY_LINE)) {
            ccPolyline* poly = ccHObjectCaster::ToPolyline(entity);
            clone = (poly ? new ccPolyline(*poly) : 0);
            if (!clone) {
                ecvConsole::Error(
                        tr("An error occurred while cloning polyline %1")
                                .arg(entity->getName()));
            }
        } else if (entity->isA(CV_TYPES::CIRCLE)) {
            clone = ccHObjectCaster::ToCircle(entity)->clone();
            if (!clone) {
                ecvConsole::Error(
                        tr("An error occurred while cloning circle %1")
                                .arg(entity->getName()));
            }
        } else if (entity->isA(CV_TYPES::DISC)) {
            ccDisc* disc = ccHObjectCaster::ToDisc(entity);
            clone = (disc ? disc->clone() : 0);
            if (!clone) {
                ecvConsole::Error(tr("An error occurred while cloning disc %1")
                                          .arg(entity->getName()));
            }
        } else if (entity->isA(CV_TYPES::FACET)) {
            ccFacet* facet = ccHObjectCaster::ToFacet(entity);
            clone = (facet ? facet->clone() : 0);
            if (!clone) {
                ecvConsole::Error(tr("An error occurred while cloning facet %1")
                                          .arg(entity->getName()));
            }
        } else {
            CVLog::Warning(
                    tr("Entity '%1' can't be cloned (type not supported yet!)")
                            .arg(entity->getName()));
        }

        if (clone) {
            // copy GL transformation history
            clone->setGLTransformationHistory(
                    entity->getGLTransformationHistory());
            addToDB(clone);
            lastClone = clone;
        }
    }

    if (lastClone && m_ccRoot) {
        m_ccRoot->selectEntity(lastClone);
    }

    updateUI();
}

// consoleTable right click envent
void MainWindow::popMenuInConsole(const QPoint& pos) {
    QAction clearItemsAction(tr("Clear selected items"), this);
    QAction clearConsoleAction(tr("Clear console"), this);
    connect(&clearItemsAction, &QAction::triggered, this,
            &MainWindow::clearConsoleItems);
    connect(&clearConsoleAction, &QAction::triggered, this,
            &MainWindow::clearConsole);
    QMenu menu(m_ui->consoleWidget);
    menu.addAction(&clearItemsAction);
    menu.addAction(&clearConsoleAction);
    menu.exec(QCursor::pos());  // show in mouse position
}

// Clear consoleTable
void MainWindow::clearConsole() { m_ui->consoleWidget->clear(); }

// Remove selected items in consoleTable
void MainWindow::clearConsoleItems() {
    /*get selected items*/
    QList<QListWidgetItem*> items = m_ui->consoleWidget->selectedItems();

    if (items.count() > 0) {
        if (QMessageBox::Yes ==
            QMessageBox::question(this, QStringLiteral("Remove Item"),
                                  QStringLiteral("Remove %1 log information(s)")
                                          .arg(QString::number(items.count())),
                                  QMessageBox::Yes | QMessageBox::No,
                                  QMessageBox::Yes)) {
            foreach (QListWidgetItem* var, items) {
                m_ui->consoleWidget->removeItemWidget(var);
                items.removeOne(var);
                delete var;
            }
        }
    }
}

// sand box research
void MainWindow::doComputeBestFitBB() {
    if (QMessageBox::warning(
                this, tr("This method is for test purpose only"),
                tr("Cloud(s) are going to be rotated while still displayed in "
                   "their previous position! Proceed?"),
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No) != QMessageBox::Yes) {
        return;
    }

    // avoid rendering obj that not been selected again
    ecvDisplayTools::SetRedrawRecursive(false);

    // backup selected entities as removeObjectTemporarilyFromDBTree can modify
    // them
    ccHObject::Container selectedEntities = getSelectedEntities();
    for (ccHObject* entity : selectedEntities)  // warning, getSelectedEntites
                                                // may change during this loop!
    {
        ccGenericPointCloud* cloud =
                ccHObjectCaster::ToGenericPointCloud(entity);

        if (cloud && cloud->isA(CV_TYPES::POINT_CLOUD))  // TODO
        {
            cloudViewer::Neighbourhood Yk(cloud);

            cloudViewer::SquareMatrixd covMat = Yk.computeCovarianceMatrix();
            if (covMat.isValid()) {
                cloudViewer::SquareMatrixd eigVectors;
                std::vector<double> eigValues;
                if (Jacobi<double>::ComputeEigenValuesAndVectors(
                            covMat, eigVectors, eigValues, true)) {
                    Jacobi<double>::SortEigenValuesAndVectors(eigVectors,
                                                              eigValues);

                    ccGLMatrix trans;
                    GLfloat* rotMat = trans.data();
                    for (unsigned j = 0; j < 3; ++j) {
                        double u[3];
                        Jacobi<double>::GetEigenVector(eigVectors, j, u);
                        CCVector3 v(static_cast<PointCoordinateType>(u[0]),
                                    static_cast<PointCoordinateType>(u[1]),
                                    static_cast<PointCoordinateType>(u[2]));
                        v.normalize();
                        rotMat[j * 4] = static_cast<float>(v.x);
                        rotMat[j * 4 + 1] = static_cast<float>(v.y);
                        rotMat[j * 4 + 2] = static_cast<float>(v.z);
                    }

                    const CCVector3* G = Yk.getGravityCenter();
                    assert(G);
                    trans.shiftRotationCenter(*G);

                    cloud->setGLTransformation(trans);
                    trans.invert();

                    // we temporarily detach entity, as it may undergo
                    //"severe" modifications (octree deletion, etc.) --> see
                    // ccPointCloud::applyRigidTransformation
                    ccHObjectContext objContext =
                            removeObjectTemporarilyFromDBTree(cloud);
                    static_cast<ccPointCloud*>(cloud)->applyRigidTransformation(
                            trans);
                    putObjectBackIntoDBTree(cloud, objContext);
                    entity->setRedrawFlagRecursive(true);
                    CC_DRAW_CONTEXT context;
                    context.removeViewID =
                            QString::number(entity->getUniqueID());
                    ecvDisplayTools::RemoveBB(context);
                }
            }
        }
    }

    refreshAll();
}

void MainWindow::doActionComputeDistanceMap() {
    static unsigned steps = 128;
    static double margin = 0.0;
    static bool filterRange = false;
    static double range[2] = {0.0, 1.0};

    // show dialog
    {
        QDialog dialog(this);
        Ui_DistanceMapDialog ui;
        ui.setupUi(&dialog);

        ui.stepsSpinBox->setValue(static_cast<int>(steps));
        ui.marginDoubleSpinBox->setValue(margin);
        ui.rangeCheckBox->setChecked(filterRange);
        ui.minDistDoubleSpinBox->setValue(range[0]);
        ui.maxDistDoubleSpinBox->setValue(range[1]);

        if (!dialog.exec()) {
            return;
        }

        steps = static_cast<unsigned>(ui.stepsSpinBox->value());
        margin = ui.marginDoubleSpinBox->value();
        filterRange = ui.rangeCheckBox->isChecked();
        range[0] = ui.minDistDoubleSpinBox->value();
        range[1] = ui.maxDistDoubleSpinBox->value();
    }

    ecvProgressDialog pDlg(true, this);
    pDlg.setAutoClose(false);

    for (ccHObject* entity : getSelectedEntities()) {
        if (!entity->isKindOf(CV_TYPES::MESH) &&
            !entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
            // non handled entity type
            continue;
        }

        // cloudViewer::ChamferDistanceTransform cdt;
        cloudViewer::SaitoSquaredDistanceTransform cdt;
        if (!cdt.initGrid(Tuple3ui(steps, steps, steps))) {
            // not enough memory
            CVLog::Error(tr("Not enough memory!"));
            return;
        }

        ccBBox box = entity->getOwnBB();
        PointCoordinateType largestDim =
                box.getMaxBoxDim() + static_cast<PointCoordinateType>(margin);
        PointCoordinateType cellDim = largestDim / steps;
        CCVector3 minCorner =
                box.getCenter() - CCVector3(1, 1, 1) * (largestDim / 2);

        bool result = false;
        if (entity->isKindOf(CV_TYPES::MESH)) {
            ccMesh* mesh = static_cast<ccMesh*>(entity);
            result = cdt.initDT(mesh, cellDim, minCorner, &pDlg);
        } else {
            ccGenericPointCloud* cloud =
                    static_cast<ccGenericPointCloud*>(entity);
            result = cdt.initDT(cloud, cellDim, minCorner, &pDlg);
        }

        if (!result) {
            CVLog::Error(tr("Not enough memory!"));
            return;
        }

        // cdt.propagateDistance(CHAMFER_345, &pDlg);
        cdt.propagateDistance(&pDlg);

        // convert the grid to a cloud
        ccPointCloud* gridCloud = new ccPointCloud(
                entity->getName() + tr(".distance_grid(%1)").arg(steps));
        {
            unsigned pointCount = steps * steps * steps;
            if (!gridCloud->reserve(pointCount)) {
                CVLog::Error(tr("Not enough memory!"));
                delete gridCloud;
                return;
            }

            ccScalarField* sf = new ccScalarField("DT values");
            if (!sf->reserveSafe(pointCount)) {
                CVLog::Error(tr("Not enough memory!"));
                delete gridCloud;
                sf->release();
                return;
            }

            for (unsigned i = 0; i < steps; ++i) {
                for (unsigned j = 0; j < steps; ++j) {
                    for (unsigned k = 0; k < steps; ++k) {
                        ScalarType d = std::sqrt(static_cast<ScalarType>(
                                               cdt.getValue(i, j, k))) *
                                       cellDim;

                        if (!filterRange || (d >= range[0] && d <= range[1])) {
                            gridCloud->addPoint(minCorner + CCVector3(i + 0.5,
                                                                      j + 0.5,
                                                                      k + 0.5) *
                                                                    cellDim);
                            sf->addElement(d);
                        }
                    }
                }
            }

            sf->computeMinAndMax();
            int sfIdx = gridCloud->addScalarField(sf);

            if (gridCloud->size() == 0) {
                CVLog::Warning(tr("[DistanceMap] Cloud '%1': no point falls "
                                  "inside the specified range")
                                       .arg(entity->getName()));
                delete gridCloud;
                gridCloud = nullptr;
            } else {
                gridCloud->setCurrentDisplayedScalarField(sfIdx);
                gridCloud->showSF(true);
                // gridCloud->setDisplay(entity->getDisplay());
                gridCloud->shrinkToFit();
                // entity->prepareDisplayForRefresh();
                addToDB(gridCloud);
            }
        }
    }
}

void MainWindow::doActionComputeDistToBestFitQuadric3D() {
    bool ok = true;
    int steps = QInputDialog::getInt(
            this, tr("Distance to best fit quadric (3D)"),
            tr("Steps (per dim.)"), 50, 10, 10000, 10, &ok);
    if (!ok) return;

    for (ccHObject* entity : getSelectedEntities()) {
        if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
            ccGenericPointCloud* cloud =
                    ccHObjectCaster::ToGenericPointCloud(entity);
            cloudViewer::Neighbourhood Yk(cloud);

            double Q[10];
            if (Yk.compute3DQuadric(Q)) {
                const double& a = Q[0];
                const double& b = Q[1];
                const double& c = Q[2];
                const double& e = Q[3];
                const double& f = Q[4];
                const double& g = Q[5];
                const double& l = Q[6];
                const double& m = Q[7];
                const double& n = Q[8];
                const double& d = Q[9];

                // gravity center
                const CCVector3* G = Yk.getGravityCenter();
                if (!G) {
                    ecvConsole::Warning(tr("Failed to get the center of "
                                           "gravity of cloud '%1'!")
                                                .arg(cloud->getName()));
                    continue;
                }

                const ccBBox bbox = cloud->getOwnBB();
                PointCoordinateType maxDim = bbox.getMaxBoxDim();
                CCVector3 C = bbox.getCenter();

                // Sample points on a cube and compute for each of them the
                // distance to the Quadric
                ccPointCloud* newCloud = new ccPointCloud();
                if (!newCloud->reserve(steps * steps * steps)) {
                    ecvConsole::Error(tr("Not enough memory!"));
                }

                const char defaultSFName[] = "Dist. to 3D quadric";
                int sfIdx = newCloud->getScalarFieldIndexByName(defaultSFName);
                if (sfIdx < 0) sfIdx = newCloud->addScalarField(defaultSFName);
                if (sfIdx < 0) {
                    ecvConsole::Error(
                            tr("Couldn't allocate a new scalar field for "
                               "computing distances! Try to free some memory "
                               "..."));
                    delete newCloud;
                    continue;
                }

                ccScalarField* sf = static_cast<ccScalarField*>(
                        newCloud->getScalarField(sfIdx));
                assert(sf);

                // FILE* fp = fopen("doActionComputeQuadric3D_trace.txt","wt");
                for (int x = 0; x < steps; ++x) {
                    CCVector3 P;
                    P.x = C.x +
                          maxDim * (static_cast<PointCoordinateType>(x) /
                                            static_cast<PointCoordinateType>(
                                                    steps - 1) -
                                    PC_ONE / 2);
                    for (int y = 0; y < steps; ++y) {
                        P.y = C.y +
                              maxDim *
                                      (static_cast<PointCoordinateType>(y) /
                                               static_cast<PointCoordinateType>(
                                                       steps - 1) -
                                       PC_ONE / 2);
                        for (int z = 0; z < steps; ++z) {
                            P.z = C.z +
                                  maxDim *
                                          (static_cast<PointCoordinateType>(z) /
                                                   static_cast<
                                                           PointCoordinateType>(
                                                           steps - 1) -
                                           PC_ONE / 2);
                            newCloud->addPoint(P);

                            // compute distance to quadric
                            CCVector3 Pc = P - *G;
                            ScalarType dist = static_cast<ScalarType>(
                                    a * Pc.x * Pc.x + b * Pc.y * Pc.y +
                                    c * Pc.z * Pc.z + e * Pc.x * Pc.y +
                                    f * Pc.y * Pc.z + g * Pc.x * Pc.z +
                                    l * Pc.x + m * Pc.y + n * Pc.z + d);

                            sf->addElement(dist);
                            // fprintf(fp,"%f %f %f %f\n",Pc.x,Pc.y,Pc.z,dist);
                        }
                    }
                }
                // fclose(fp);

                if (sf) {
                    sf->computeMinAndMax();
                    newCloud->setCurrentDisplayedScalarField(sfIdx);
                    newCloud->showSF(true);
                }
                newCloud->setName(tr("Distance map to 3D quadric"));

                addToDB(newCloud);
            } else {
                ecvConsole::Warning(
                        tr("Failed to compute 3D quadric on cloud '%1'")
                                .arg(cloud->getName()));
            }
        }
    }
}

// Aurelien BEY le 13/11/2008
void MainWindow::doAction4pcsRegister() {
    if (QMessageBox::warning(
                this, tr("Work in progress"),
                tr("This method is still under development: are you sure you "
                   "want to use it? (a crash may likely happen)"),
                QMessageBox::Yes, QMessageBox::No) == QMessageBox::No)
        return;

    if (m_selectedEntities.size() != 2) {
        ecvConsole::Error(tr("Select 2 point clouds!"));
        return;
    }

    if (!m_selectedEntities[0]->isKindOf(CV_TYPES::POINT_CLOUD) ||
        !m_selectedEntities[1]->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ecvConsole::Error(tr("Select 2 point clouds!"));
        return;
    }

    ccGenericPointCloud* model =
            ccHObjectCaster::ToGenericPointCloud(m_selectedEntities[0]);
    ccGenericPointCloud* data =
            ccHObjectCaster::ToGenericPointCloud(m_selectedEntities[1]);

    ccAlignDlg aDlg(model, data);
    if (!aDlg.exec()) return;

    // model = aDlg.getModelObject();
    data = aDlg.getDataObject();

    // Take the correct number of points among the clouds
    cloudViewer::ReferenceCloud* subModel = aDlg.getSampledModel();
    cloudViewer::ReferenceCloud* subData = aDlg.getSampledData();

    unsigned nbMaxCandidates = aDlg.isNumberOfCandidatesLimited()
                                       ? aDlg.getMaxNumberOfCandidates()
                                       : 0;

    ecvProgressDialog pDlg(true, this);

    cloudViewer::PointProjectionTools::Transformation transform;
    if (cloudViewer::FPCSRegistrationTools::RegisterClouds(
                subModel, subData, transform,
                static_cast<ScalarType>(aDlg.getDelta()),
                static_cast<ScalarType>(aDlg.getDelta() / 2),
                static_cast<PointCoordinateType>(aDlg.getOverlap()),
                aDlg.getNbTries(), 5000, &pDlg, nbMaxCandidates)) {
        // output resulting transformation matrix
        {
            ccGLMatrix transMat =
                    FromCCLibMatrix<double, float>(transform.R, transform.T);
            forceConsoleDisplay();
            ecvConsole::Print(tr("[Align] Resulting matrix:"));
            ecvConsole::Print(transMat.toString(12, ' '));  // full precision
            ecvConsole::Print(tr(
                    "Hint: copy it (CTRL+C) and apply it - or its inverse - on "
                    "any entity with the 'Edit > Apply transformation' tool"));
        }

        ccPointCloud* newDataCloud =
                data->isA(CV_TYPES::POINT_CLOUD)
                        ? static_cast<ccPointCloud*>(data)->cloneThis()
                        : ccPointCloud::From(data, data);

        if (data->getParent()) data->getParent()->addChild(newDataCloud);
        newDataCloud->setName(data->getName() + tr(".registered"));
        transform.apply(*newDataCloud);
        newDataCloud->invalidateBoundingBox();  // invalidate bb
        addToDB(newDataCloud);
        zoomOn(newDataCloud);

        data->setEnabled(false);
    } else {
        ecvConsole::Warning(tr("[Align] Registration failed!"));
    }

    if (subModel) delete subModel;
    if (subData) delete subData;

    updateUI();
}

// semi-persistent parameters
static ccLibAlgorithms::ScaleMatchingAlgorithm s_msAlgorithm =
        ccLibAlgorithms::PCA_MAX_DIM;
static double s_msRmsDiff = 1.0e-5;
static int s_msFinalOverlap = 100;

void MainWindow::doActionMatchScales() {
    // we need at least 2 entities
    if (m_selectedEntities.size() < 2) return;

    // we must select the point clouds and meshes
    ccHObject::Container selectedEntities;
    try {
        for (ccHObject* entity : getSelectedEntities()) {
            if (entity->isKindOf(CV_TYPES::POINT_CLOUD) ||
                entity->isKindOf(CV_TYPES::MESH)) {
                selectedEntities.push_back(entity);
            }
        }
    } catch (const std::bad_alloc&) {
        ecvConsole::Error(tr("Not enough memory!"));
        return;
    }

    ccMatchScalesDlg msDlg(selectedEntities, 0, this);
    msDlg.setSelectedAlgorithm(s_msAlgorithm);
    msDlg.rmsDifferenceLineEdit->setText(QString::number(s_msRmsDiff, 'e', 1));
    msDlg.overlapSpinBox->setValue(s_msFinalOverlap);

    if (!msDlg.exec()) return;

    // save semi-persistent parameters
    s_msAlgorithm = msDlg.getSelectedAlgorithm();
    if (s_msAlgorithm == ccLibAlgorithms::ICP_SCALE) {
        s_msRmsDiff = msDlg.rmsDifferenceLineEdit->text().toDouble();
        s_msFinalOverlap = msDlg.overlapSpinBox->value();
    }

    ccLibAlgorithms::ApplyScaleMatchingAlgorithm(
            s_msAlgorithm, selectedEntities, s_msRmsDiff, s_msFinalOverlap,
            msDlg.getSelectedIndex(), this);

    // reselect previously selected entities!
    if (m_ccRoot) m_ccRoot->selectEntities(selectedEntities);

    updateUI();
}

void MainWindow::doActionMatchBBCenters() {
    // we need at least 2 entities
    if (m_selectedEntities.size() < 2) return;

    // we must backup 'm_selectedEntities' as removeObjectTemporarilyFromDBTree
    // can modify it!
    ccHObject::Container selectedEntities = m_selectedEntities;

    // by default, we take the first entity as reference
    // TODO: maybe the user would like to select the reference himself ;)
    ccHObject* refEnt = selectedEntities[0];
    CCVector3 refCenter = refEnt->getBB_recursive().getCenter();

    for (ccHObject* entity : selectedEntities)  // warning, getSelectedEntites
                                                // may change during this loop!
    {
        CCVector3 center = entity->getBB_recursive().getCenter();

        CCVector3 T = refCenter - center;

        // transformation (used only for translation)
        ccGLMatrix glTrans;
        glTrans += T;

        forceConsoleDisplay();
        ecvConsole::Print(tr("[Synchronize] Transformation matrix (%1 --> %2):")
                                  .arg(entity->getName(),
                                       selectedEntities[0]->getName()));
        ecvConsole::Print(glTrans.toString(12, ' '));  // full precision
        ecvConsole::Print(
                tr("Hint: copy it (CTRL+C) and apply it - or its inverse - on "
                   "any entity with the 'Edit > Apply transformation' tool"));

        // we temporarily detach entity, as it may undergo
        //"severe" modifications (octree deletion, etc.) --> see
        // ccHObject::applyGLTransformation
        ccHObjectContext objContext = removeObjectTemporarilyFromDBTree(entity);
        entity->applyGLTransformation_recursive(&glTrans);
        putObjectBackIntoDBTree(entity, objContext);
    }

    // reselect previously selected entities!
    if (m_ccRoot) m_ccRoot->selectEntities(selectedEntities);

    zoomOnSelectedEntities();
    refreshSelected();
    updateUI();
}

void MainWindow::doActionRegister() {
    if (m_selectedEntities.size() != 2 ||
        (!m_selectedEntities[0]->isKindOf(CV_TYPES::POINT_CLOUD) &&
         !m_selectedEntities[0]->isKindOf(CV_TYPES::MESH)) ||
        (!m_selectedEntities[1]->isKindOf(CV_TYPES::POINT_CLOUD) &&
         !m_selectedEntities[1]->isKindOf(CV_TYPES::MESH))) {
        ecvConsole::Error(tr("Select 2 point clouds or meshes!"));
        return;
    }

    ccHObject* data = static_cast<ccHObject*>(m_selectedEntities[0]);
    ccHObject* model = static_cast<ccHObject*>(m_selectedEntities[1]);
    if (data->isKindOf(CV_TYPES::MESH) &&
        model->isKindOf(CV_TYPES::POINT_CLOUD)) {
        // by default, prefer the mesh as the reference
        std::swap(data, model);
    }

    ccRegistrationDlg rDlg(data, model, this);
    if (!rDlg.exec()) return;

    // model and data order may have changed!
    model = rDlg.getModelEntity();
    data = rDlg.getDataEntity();

    double minRMSDecrease = rDlg.getMinRMSDecrease();
    if (std::isnan(minRMSDecrease)) {
        CVLog::Error(tr("Invalid minimum RMS decrease value"));
        return;
    }
    if (minRMSDecrease < ccRegistrationDlg::GetAbsoluteMinRMSDecrease()) {
        minRMSDecrease = ccRegistrationDlg::GetAbsoluteMinRMSDecrease();
        CVLog::Error(tr("Minimum RMS decrease value is too small.\n%1 will be "
                        "used instead (numerical accuracy limit).")
                             .arg(minRMSDecrease, 0, 'E', 1));
        rDlg.setMinRMSDecrease(minRMSDecrease);
    }

    cloudViewer::ICPRegistrationTools::Parameters parameters;
    {
        parameters.convType = rDlg.getConvergenceMethod();
        parameters.minRMSDecrease = minRMSDecrease;
        parameters.nbMaxIterations = rDlg.getMaxIterationCount();
        parameters.adjustScale = rDlg.adjustScale();
        parameters.filterOutFarthestPoints = rDlg.removeFarthestPoints();
        parameters.samplingLimit = rDlg.randomSamplingLimit();
        parameters.finalOverlapRatio = rDlg.getFinalOverlap() / 100.0;
        parameters.transformationFilters = rDlg.getTransformationFilters();
        parameters.maxThreadCount = rDlg.getMaxThreadCount();
        parameters.useC2MSignedDistances = rDlg.useC2MSignedDistances();
        parameters.normalsMatching = rDlg.normalsMatchingOption();
    }
    bool useDataSFAsWeights = rDlg.useDataSFAsWeights();
    bool useModelSFAsWeights = rDlg.useModelSFAsWeights();

    // semi-persistent storage (for next call)
    rDlg.saveParameters();

    ccGLMatrix transMat;
    double finalError = 0.0;
    double finalScale = 1.0;
    unsigned finalPointCount = 0;

    if (ccRegistrationTools::ICP(
                data, model, transMat, finalScale, finalError, finalPointCount,
                parameters, useDataSFAsWeights, useModelSFAsWeights, this)) {
        QString rmsString = tr("Final RMS*: %1 (computed on %2 points)")
                                    .arg(finalError)
                                    .arg(finalPointCount);
        QString rmsDisclaimerString =
                tr("(* RMS is potentially weighted, depending on the selected "
                   "options)");
        CVLog::Print(QString("[Register] ") + rmsString);
        CVLog::Print(QString("[Register] ") + rmsDisclaimerString);

        QStringList summary;
        summary << rmsString;
        summary << rmsDisclaimerString;
        summary << "----------------";

        // transformation matrix
        {
            summary << "Transformation matrix";
            summary << transMat.toString(
                    3, '\t');  // low precision, just for display
            summary << "----------------";

            CVLog::Print(tr("[Register] Applied transformation matrix:"));
            CVLog::Print(transMat.toString(12, ' '));  // full precision
            CVLog::Print(tr(
                    "Hint: copy it (CTRL+C) and apply it - or its inverse - on "
                    "any entity with the 'Edit > Apply transformation' tool"));
        }

        if (parameters.adjustScale) {
            QString scaleString =
                    tr("Scale: %1 (already integrated in above matrix!)")
                            .arg(finalScale);
            CVLog::Warning(tr("[Register] ") + scaleString);
            summary << scaleString;
        } else {
            CVLog::Print(tr("[Register] Scale: fixed (1.0)"));
            summary << tr("Scale: fixed (1.0)");
        }

        // overlap
        summary << "----------------";
        QString overlapString =
                tr("Theoretical overlap: %1%")
                        .arg(static_cast<int>(parameters.finalOverlapRatio *
                                              100));
        CVLog::Print(tr("[Register] %1").arg(overlapString));
        summary << overlapString;

        summary << "----------------";
        summary << tr("This report has been output to Console (F8)");

        // cloud to move
        ccGenericPointCloud* pc = nullptr;

        if (data->isKindOf(CV_TYPES::POINT_CLOUD)) {
            pc = ccHObjectCaster::ToGenericPointCloud(data);
        } else if (data->isKindOf(CV_TYPES::MESH)) {
            ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(data);
            pc = mesh->getAssociatedCloud();

            // warning: point cloud is locked!
            if (pc->isLocked()) {
                pc = nullptr;
                // we ask the user about cloning the 'data' mesh
                QMessageBox::StandardButton result = QMessageBox::question(
                        this, tr("Registration"),
                        tr("Data mesh vertices are locked (they may be shared "
                           "with other meshes): Do you wish to clone this mesh "
                           "to apply transformation?"),
                        QMessageBox::Ok | QMessageBox::Cancel, QMessageBox::Ok);

                // continue process?
                if (result == QMessageBox::Ok) {
                    ccGenericMesh* newMesh = nullptr;
                    if (mesh->isA(CV_TYPES::MESH))
                        newMesh = static_cast<ccMesh*>(mesh)->cloneMesh();
                    else {
                        // FIXME TODO
                        CVLog::Error(tr("Doesn't work on sub-meshes yet!"));
                    }

                    if (newMesh) {
                        // newMesh->setDisplay(data->getDisplay());
                        addToDB(newMesh);
                        data = newMesh;
                        pc = newMesh->getAssociatedCloud();
                    } else {
                        CVLog::Error(
                                tr("Failed to clone 'data' mesh! (not enough "
                                   "memory?)"));
                    }
                }
            }
        }

        // if we managed to get a point cloud to move!
        if (pc) {
            // we temporarily detach cloud, as it may undergo
            //"severe" modifications (octree deletion, etc.) --> see
            // ccPointCloud::applyRigidTransformation
            ccHObjectContext objContext = removeObjectTemporarilyFromDBTree(pc);
            pc->applyRigidTransformation(transMat);
            putObjectBackIntoDBTree(pc, objContext);

            // don't forget to update mesh bounding box also!
            if (data->isKindOf(CV_TYPES::MESH))
                ccHObjectCaster::ToGenericMesh(data)->refreshBB();

            // don't forget global shift
            ccGenericPointCloud* refPc =
                    ccHObjectCaster::ToGenericPointCloud(model);
            if (refPc) {
                if (refPc->isShifted()) {
                    const CCVector3d& Pshift = refPc->getGlobalShift();
                    const double& scale = refPc->getGlobalScale();
                    pc->setGlobalShift(Pshift);
                    pc->setGlobalScale(scale);
                    CVLog::Warning(tr("[ICP] Aligned entity global shift has "
                                      "been updated to match the reference: "
                                      "(%1,%2,%3) [x%4]")
                                           .arg(Pshift.x)
                                           .arg(Pshift.y)
                                           .arg(Pshift.z)
                                           .arg(scale));
                } else if (pc->isShifted())  // we'll ask the user first before
                                             // dropping the shift information
                                             // on the aligned cloud
                {
                    if (QMessageBox::question(
                                this, tr("Drop shift information?"),
                                tr("Aligned entity is shifted but reference "
                                   "cloud is not: drop global shift "
                                   "information?"),
                                QMessageBox::Yes,
                                QMessageBox::No) == QMessageBox::Yes) {
                        pc->setGlobalShift(0, 0, 0);
                        pc->setGlobalScale(1.0);
                        CVLog::Warning(
                                tr("[ICP] Aligned entity global shift has been "
                                   "reset to match the reference!"));
                    }
                }
            }

            // data->prepareDisplayForRefresh_recursive();
            data->setName(data->getName() + tr(".registered"));
            // avoid rendering other object this time
            ecvDisplayTools::SetRedrawRecursive(false);
            zoomOn(data);
        }

        // pop-up summary
        QMessageBox::information(this, tr("Register info"), summary.join("\n"));
        forceConsoleDisplay();
    }

    updateUI();
}

void MainWindow::activateRegisterPointPairTool() {
    if (!haveSelection()) {
        ecvConsole::Error(
                tr("Select one or two entities (point cloud or mesh)!"));
        return;
    }

    ccHObject::Container alignedEntities;
    ccHObject::Container refEntities;
    try {
        ccHObject::Container entities;
        entities.reserve(m_selectedEntities.size());

        for (ccHObject* entity : m_selectedEntities) {
            // for now, we only handle clouds or meshes
            if (entity->isKindOf(CV_TYPES::POINT_CLOUD) ||
                entity->isKindOf(CV_TYPES::MESH)) {
                entities.push_back(entity);
            }
        }

        if (entities.empty()) {
            ecvConsole::Error(
                    "Select at least one entity (point cloud or mesh)!");
            return;
        } else if (entities.size() == 1) {
            alignedEntities = entities;
        } else {
            std::vector<int> indexes;
            if (!ecvEntitySelectionDialog::SelectEntities(
                        entities, indexes, this,
                        tr("Select aligned entities"))) {
                // process cancelled by the user
                return;
            }

            // add the selected indexes as 'aligned' entities
            alignedEntities.reserve(indexes.size());
            for (size_t i = 0; i < indexes.size(); ++i) {
                alignedEntities.push_back(entities[indexes[i]]);
            }

            // add the others as 'reference' entities
            assert(indexes.size() <= entities.size());
            refEntities.reserve(entities.size() - indexes.size());
            for (size_t i = 0; i < entities.size(); ++i) {
                if (std::find(indexes.begin(), indexes.end(), i) ==
                    indexes.end()) {
                    refEntities.push_back(entities[i]);
                }
            }
        }
    } catch (const std::bad_alloc&) {
        CVLog::Error(tr("Not enough memory"));
        return;
    }

    if (alignedEntities.empty()) {
        CVLog::Error(tr("No aligned entity selected"));
        return;
    }

    // deselect all entities
    if (m_ccRoot) {
        m_ccRoot->unselectAllEntities();
    }

    if (!m_pprDlg) {
        m_pprDlg = new ccPointPairRegistrationDlg(m_pickingHub, this, this);
        connect(m_pprDlg, &ccOverlayDialog::processFinished, this,
                &MainWindow::deactivateRegisterPointPairTool);
        registerOverlayDialog(m_pprDlg, Qt::TopRightCorner);
    }

    if (!getActiveWindow()) {
        CVLog::Error(tr(
                "[PointPairRegistration] Failed to create dedicated 3D view!"));
        return;
    }

    if (!m_pprDlg->init(ecvDisplayTools::GetCurrentScreen(), alignedEntities,
                        &refEntities))
        deactivateRegisterPointPairTool(false);

    freezeUI(true);

    if (!m_pprDlg->start()) {
        // reselect previously selected entities!
        if (m_ccRoot) {
            m_ccRoot->selectEntities(alignedEntities);
            m_ccRoot->selectEntities(refEntities);
        }
        if (ecvDisplayTools::GetCurrentScreen()) {
            ecvDisplayTools::SetRedrawRecursive(false);
            zoomOnSelectedEntities();
            ecvDisplayTools::RedrawDisplay();
        }
        deactivateRegisterPointPairTool(false);
    } else
        updateOverlayDialogsPlacement();
}

void MainWindow::deactivateRegisterPointPairTool(bool state) {
    if (m_pprDlg) m_pprDlg->clear();

    QList<QMdiSubWindow*> subWindowList = m_mdiArea->subWindowList();
    if (!subWindowList.isEmpty()) subWindowList.first()->showMaximized();

    freezeUI(false);

    updateUI();
}

void MainWindow::doSphericalNeighbourhoodExtractionTest() {
    size_t selNum = m_selectedEntities.size();
    if (selNum < 1) return;

    // spherical neighborhood extraction radius
    PointCoordinateType sphereRadius =
            ccLibAlgorithms::GetDefaultCloudKernelSize(m_selectedEntities);
    if (sphereRadius < 0) {
        ecvConsole::Error(tr("Invalid kernel size!"));
        return;
    }

    bool ok;
    double val = QInputDialog::getDouble(this, tr("SNE test"), tr("Radius:"),
                                         static_cast<double>(sphereRadius),
                                         DBL_MIN, 1.0e9, 8, &ok);
    if (!ok) return;
    sphereRadius = static_cast<PointCoordinateType>(val);

    QString sfName = tr("Spherical extraction test (%1)").arg(sphereRadius);

    ecvProgressDialog pDlg(true, this);
    pDlg.setAutoClose(false);
    ecvDisplayTools::SetRedrawRecursive(false);
    for (size_t i = 0; i < selNum; ++i) {
        // we only process clouds
        if (!m_selectedEntities[i]->isA(CV_TYPES::POINT_CLOUD)) {
            continue;
        }
        ccPointCloud* cloud =
                ccHObjectCaster::ToPointCloud(m_selectedEntities[i]);

        int sfIdx = cloud->getScalarFieldIndexByName(qPrintable(sfName));
        if (sfIdx < 0) sfIdx = cloud->addScalarField(qPrintable(sfName));
        if (sfIdx < 0) {
            ecvConsole::Error(tr("Failed to create scalar field on cloud '%1' "
                                 "(not enough memory?)")
                                      .arg(cloud->getName()));
            return;
        }

        ccOctree::Shared octree = cloud->getOctree();
        if (!octree) {
            pDlg.reset();
            pDlg.show();
            octree = cloud->computeOctree(&pDlg);
            if (!octree) {
                ecvConsole::Error(tr("Couldn't compute octree for cloud '%1'!")
                                          .arg(cloud->getName()));
                return;
            }
        }

        cloudViewer::ScalarField* sf = cloud->getScalarField(sfIdx);
        sf->fill(NAN_VALUE);
        cloud->setCurrentScalarField(sfIdx);

        QElapsedTimer eTimer;
        eTimer.start();

        size_t extractedPoints = 0;
        unsigned char level =
                octree->findBestLevelForAGivenNeighbourhoodSizeExtraction(
                        sphereRadius);
        std::random_device rd;   // non-deterministic generator
        std::mt19937 gen(rd());  // to seed mersenne twister.
        std::uniform_int_distribution<unsigned> dist(0, cloud->size() - 1);

        const unsigned samples = 1000;
        for (unsigned j = 0; j < samples; ++j) {
            unsigned randIndex = dist(gen);
            cloudViewer::DgmOctree::NeighboursSet neighbours;
            octree->getPointsInSphericalNeighbourhood(
                    *cloud->getPoint(randIndex), sphereRadius, neighbours,
                    level);
            size_t neihgboursCount = neighbours.size();
            extractedPoints += neihgboursCount;
            for (size_t k = 0; k < neihgboursCount; ++k)
                cloud->setPointScalarValue(neighbours[k].pointIndex,
                                           static_cast<ScalarType>(sqrt(
                                                   neighbours[k].squareDistd)));
        }
        ecvConsole::Print(
                tr("[SNE_TEST] Mean extraction time = %1 ms (radius = %2, "
                   "mean(neighbours) = %3)")
                        .arg(eTimer.elapsed())
                        .arg(sphereRadius)
                        .arg(extractedPoints / static_cast<double>(samples)));

        sf->computeMinAndMax();
        cloud->setCurrentDisplayedScalarField(sfIdx);
        cloud->showSF(true);
        cloud->setRedrawFlagRecursive(true);
    }

    refreshAll();
    updateUI();
}

void MainWindow::doCylindricalNeighbourhoodExtractionTest() {
    bool ok;
    double radius = QInputDialog::getDouble(this, tr("CNE Test"), tr("radius"),
                                            0.02, 1.0e-6, 1.0e6, 6, &ok);
    if (!ok) return;

    double height = QInputDialog::getDouble(this, tr("CNE Test"), tr("height"),
                                            0.05, 1.0e-6, 1.0e6, 6, &ok);
    if (!ok) return;

    ccPointCloud* cloud = new ccPointCloud(tr("cube"));
    const unsigned ptsCount = 1000000;
    if (!cloud->reserve(ptsCount)) {
        ecvConsole::Error(tr("Not enough memory!"));
        delete cloud;
        return;
    }

    // fill a unit cube with random points
    {
        std::random_device rd;   // non-deterministic generator
        std::mt19937 gen(rd());  // to seed mersenne twister.
        std::uniform_real_distribution<double> dist(0, 1);

        for (unsigned i = 0; i < ptsCount; ++i) {
            CCVector3 P(dist(gen), dist(gen), dist(gen));

            cloud->addPoint(P);
        }
    }

    // get/Add scalar field
    static const char DEFAULT_CNE_TEST_TEMP_SF_NAME[] = "CNE test";
    int sfIdx = cloud->getScalarFieldIndexByName(DEFAULT_CNE_TEST_TEMP_SF_NAME);
    if (sfIdx < 0) sfIdx = cloud->addScalarField(DEFAULT_CNE_TEST_TEMP_SF_NAME);
    if (sfIdx < 0) {
        ecvConsole::Error(tr("Not enough memory!"));
        delete cloud;
        return;
    }
    cloud->setCurrentScalarField(sfIdx);

    // reset scalar field
    cloud->getScalarField(sfIdx)->fill(NAN_VALUE);

    ecvProgressDialog pDlg(true, this);
    ccOctree::Shared octree = cloud->computeOctree(&pDlg);
    if (octree) {
        QElapsedTimer subTimer;
        subTimer.start();
        unsigned long long extractedPoints = 0;
        unsigned char level =
                octree->findBestLevelForAGivenNeighbourhoodSizeExtraction(
                        static_cast<PointCoordinateType>(
                                2.5 * radius));  // 2.5 = empirical
        const unsigned samples = 1000;
        std::random_device rd;   // non-deterministic generator
        std::mt19937 gen(rd());  // to seed mersenne twister.
        std::uniform_real_distribution<PointCoordinateType> distAngle(
                0, static_cast<PointCoordinateType>(2 * M_PI));
        std::uniform_int_distribution<unsigned> distIndex(0, ptsCount - 1);

        for (unsigned j = 0; j < samples; ++j) {
            // generate random normal vector
            CCVector3 dir(0, 0, 1);
            {
                ccGLMatrix rot;
                rot.initFromParameters(distAngle(gen), distAngle(gen),
                                       distAngle(gen), CCVector3(0, 0, 0));
                rot.applyRotation(dir);
            }
            unsigned randIndex = distIndex(gen);

            cloudViewer::DgmOctree::CylindricalNeighbourhood cn;
            cn.center = *cloud->getPoint(randIndex);
            cn.dir = dir;
            cn.level = level;
            cn.radius = static_cast<PointCoordinateType>(radius);
            cn.maxHalfLength = static_cast<PointCoordinateType>(height / 2);

            octree->getPointsInCylindricalNeighbourhood(cn);
            // octree->getPointsInSphericalNeighbourhood(*cloud->getPoint(randIndex),radius,neighbours,level);
            size_t neihgboursCount = cn.neighbours.size();
            extractedPoints += static_cast<unsigned long long>(neihgboursCount);
            for (size_t k = 0; k < neihgboursCount; ++k) {
                cloud->setPointScalarValue(
                        cn.neighbours[k].pointIndex,
                        static_cast<ScalarType>(
                                sqrt(cn.neighbours[k].squareDistd)));
            }
        }
        ecvConsole::Print(
                tr("[CNE_TEST] Mean extraction time = %1 ms (radius = %2, "
                   "height = %3, mean(neighbours) = %4)")
                        .arg(subTimer.elapsed())
                        .arg(radius)
                        .arg(height)
                        .arg(static_cast<double>(extractedPoints) / samples));
    } else {
        ecvConsole::Error(tr("Failed to compute octree!"));
    }

    ccScalarField* sf =
            static_cast<ccScalarField*>(cloud->getScalarField(sfIdx));
    sf->computeMinAndMax();
    sf->showNaNValuesInGrey(false);
    cloud->setCurrentDisplayedScalarField(sfIdx);
    cloud->showSF(true);

    addToDB(cloud);

    updateUI();
}

void MainWindow::doActionCreateCloudFromEntCenters() {
    size_t selNum = getSelectedEntities().size();

    ccPointCloud* centers = new ccPointCloud(tr("centers"));
    if (!centers->reserve(static_cast<unsigned>(selNum))) {
        CVLog::Error(tr("Not enough memory!"));
        delete centers;
        centers = nullptr;
        return;
    }

    // look for clouds
    {
        for (ccHObject* entity : getSelectedEntities()) {
            ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(entity);

            if (cloud == nullptr) {
                continue;
            }

            centers->addPoint(cloud->getOwnBB().getCenter());

            // we display the cloud in the same window as the first (selected)
            // cloud we encounter if (!centers->getDisplay())
            //{
            //	centers->setDisplay(cloud->getDisplay());
            // }
        }
    }

    if (centers->size() == 0) {
        CVLog::Error(tr("No cloud in selection?!"));
        delete centers;
        centers = nullptr;
    } else {
        centers->resize(centers->size());
        centers->setPointSize(10);
        centers->setVisible(true);
        addToDB(centers);
    }
}

void MainWindow::doActionComputeBestICPRmsMatrix() {
    // look for clouds
    std::vector<ccPointCloud*> clouds;
    try {
        for (ccHObject* entity : getSelectedEntities()) {
            ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(entity);
            if (cloud) {
                clouds.push_back(cloud);
            }
        }
    } catch (const std::bad_alloc&) {
        CVLog::Error(tr("Not enough memory!"));
        return;
    }

    size_t cloudCount = clouds.size();
    if (cloudCount < 2) {
        CVLog::Error(tr("Need at least two clouds!"));
        return;
    }

    // init matrices
    std::vector<double> rmsMatrix;
    std::vector<ccGLMatrix> matrices;
    std::vector<std::pair<double, double>> matrixAngles;
    try {
        rmsMatrix.resize(cloudCount * cloudCount, 0);

        // init all possible transformations
        static const double angularStep_deg = 45.0;
        unsigned phiSteps = static_cast<unsigned>(360.0 / angularStep_deg);
        assert(cloudViewer::LessThanEpsilon(
                std::abs(360.0 - phiSteps * angularStep_deg)));
        unsigned thetaSteps = static_cast<unsigned>(180.0 / angularStep_deg);
        assert(cloudViewer::LessThanEpsilon(
                std::abs(180.0 - thetaSteps * angularStep_deg)));
        unsigned rotCount = phiSteps * (thetaSteps - 1) + 2;
        matrices.reserve(rotCount);
        matrixAngles.reserve(rotCount);

        for (unsigned j = 0; j <= thetaSteps; ++j) {
            // we want to cover the full [0-180] interval! ([-90;90] in fact)
            double theta_deg = j * angularStep_deg - 90.0;
            for (unsigned i = 0; i < phiSteps; ++i) {
                double phi_deg = i * angularStep_deg;
                ccGLMatrix trans;
                trans.initFromParameters(
                        static_cast<float>(
                                cloudViewer::DegreesToRadians(phi_deg)),
                        static_cast<float>(
                                cloudViewer::DegreesToRadians(theta_deg)),
                        0, CCVector3(0, 0, 0));
                matrices.push_back(trans);
                matrixAngles.push_back(
                        std::pair<double, double>(phi_deg, theta_deg));

                // for poles, no need to rotate!
                if (j == 0 || j == thetaSteps) break;
            }
        }
    } catch (const std::bad_alloc&) {
        CVLog::Error(tr("Not enough memory!"));
        return;
    }

    // let's start!
    {
        ecvProgressDialog pDlg(true, this);
        pDlg.setMethodTitle(tr("Testing all possible positions"));
        pDlg.setInfo(tr("%1 clouds and %2 positions")
                             .arg(cloudCount)
                             .arg(matrices.size()));
        cloudViewer::NormalizedProgress nProgress(
                &pDlg,
                static_cast<unsigned>(((cloudCount * (cloudCount - 1)) / 2) *
                                      matrices.size()));
        pDlg.start();
        QApplication::processEvents();

        // #define TEST_GENERATION
#ifdef TEST_GENERATION
        ccPointCloud* testSphere = new ccPointCloud();
        testSphere->reserve(matrices.size());
#endif

        for (size_t i = 0; i < cloudCount - 1; ++i) {
            ccPointCloud* A = clouds[i];
            A->computeOctree();

            for (size_t j = i + 1; j < cloudCount; ++j) {
                ccGLMatrix transBToZero;
                transBToZero.toIdentity();
                transBToZero.setTranslation(-clouds[j]->getOwnBB().getCenter());

                ccGLMatrix transFromZeroToA;
                transFromZeroToA.toIdentity();
                transFromZeroToA.setTranslation(A->getOwnBB().getCenter());

#ifndef TEST_GENERATION
                double minRMS = -1.0;
                int bestMatrixIndex = -1;
                ccPointCloud* bestB = nullptr;
#endif
                for (size_t k = 0; k < matrices.size(); ++k) {
                    ccPointCloud* B = clouds[j]->cloneThis();
                    if (!B) {
                        CVLog::Error(tr("Not enough memory!"));
                        return;
                    }

                    ccGLMatrix BtoA =
                            transFromZeroToA * matrices[k] * transBToZero;
                    B->applyRigidTransformation(BtoA);

#ifndef TEST_GENERATION
                    double finalRMS = 0.0;
                    unsigned finalPointCount = 0;
                    cloudViewer::ICPRegistrationTools::RESULT_TYPE result;
                    cloudViewer::ICPRegistrationTools::ScaledTransformation
                            registerTrans;
                    cloudViewer::ICPRegistrationTools::Parameters params;
                    {
                        params.convType = cloudViewer::ICPRegistrationTools::
                                MAX_ERROR_CONVERGENCE;
                        params.minRMSDecrease = 1.0e-6;
                    }

                    result = cloudViewer::ICPRegistrationTools::Register(
                            A, 0, B, params, registerTrans, finalRMS,
                            finalPointCount);

                    if (result >=
                        cloudViewer::ICPRegistrationTools::ICP_ERROR) {
                        delete B;
                        if (bestB) delete bestB;
                        CVLog::Error(
                                tr("An error occurred while performing ICP!"));
                        return;
                    }

                    if (minRMS < 0 || finalRMS < minRMS) {
                        minRMS = finalRMS;
                        bestMatrixIndex = static_cast<int>(k);
                        std::swap(bestB, B);
                    }

                    if (B) {
                        delete B;
                        B = nullptr;
                    }
#else
                    addToDB(B);

                    // Test sphere
                    CCVector3 Y(0, 1, 0);
                    matrices[k].apply(Y);
                    testSphere->addPoint(Y);
#endif

                    if (!nProgress.oneStep()) {
                        // process cancelled by user
                        return;
                    }
                }

#ifndef TEST_GENERATION
                if (bestMatrixIndex >= 0) {
                    assert(bestB);
                    ccHObject* group =
                            new ccHObject(tr("Best case #%1 / #%2 - RMS = %3")
                                                  .arg(i + 1)
                                                  .arg(j + 1)
                                                  .arg(minRMS));
                    group->addChild(bestB);
                    // group->setDisplay_recursive(A->getDisplay());
                    addToDB(group);
                    CVLog::Print(
                            tr("[doActionComputeBestICPRmsMatrix] Comparison "
                               "#%1 / #%2: min RMS = %3 (phi = %4 / theta = %5 "
                               "deg.)")
                                    .arg(i + 1)
                                    .arg(j + 1)
                                    .arg(minRMS)
                                    .arg(matrixAngles[bestMatrixIndex].first)
                                    .arg(matrixAngles[bestMatrixIndex].second));
                } else {
                    assert(!bestB);
                    CVLog::Warning(tr("[doActionComputeBestICPRmsMatrix] "
                                      "Comparison #%1 / #%2: INVALID")
                                           .arg(i + 1)
                                           .arg(j + 1));
                }

                rmsMatrix[i * cloudCount + j] = minRMS;
#else
                addToDB(testSphere);
                i = cloudCount;
                break;
#endif
            }
        }
    }

    // export result as a CSV file
#ifdef TEST_GENERATION
    if (false)
#endif
    {
        // persistent settings
        QString currentPath = ecvSettingManager::getValue(
                                      ecvPS::SaveFile(), ecvPS::CurrentPath(),
                                      ecvFileUtils::defaultDocPath())
                                      .toString();

        QString outputFilename = QFileDialog::getSaveFileName(
                this, tr("Select output file"), currentPath, "*.csv", nullptr,
                ECVFileDialogOptions());

        if (outputFilename.isEmpty()) return;

        QFile fp(outputFilename);
        if (fp.open(QFile::Text | QFile::WriteOnly)) {
            QTextStream stream(&fp);
            // header
            {
                stream << "RMS";
                for (ccPointCloud* cloud : clouds) {
                    stream << ";";
                    stream << cloud->getName();
                }
                stream << QtCompat::endl;
            }

            // rows
            for (size_t j = 0; j < cloudCount; ++j) {
                stream << clouds[j]->getName();
                stream << ";";
                for (size_t i = 0; i < cloudCount; ++i) {
                    stream << rmsMatrix[j * cloudCount + i];
                    stream << ";";
                }
                stream << QtCompat::endl;
            }

            CVLog::Print(tr("[doActionComputeBestICPRmsMatrix] Job done"));
        } else {
            CVLog::Error(tr("Failed to save output file?!"));
        }
    }
}

static int s_innerRectDim = 2;
void MainWindow::doActionFindBiggestInnerRectangle() {
    if (!haveSelection()) return;

    ccHObject* entity = haveOneSelection() ? m_selectedEntities[0] : nullptr;
    if (!entity || !entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ecvConsole::Error(tr("Select one point cloud!"));
        return;
    }

    bool ok;
    int dim = QInputDialog::getInt(this, tr("Dimension"),
                                   tr("Orthogonal dim (X=0 / Y=1 / Z=2)"),
                                   s_innerRectDim, 0, 2, 1, &ok);
    if (!ok) return;
    s_innerRectDim = dim;

    ccGenericPointCloud* cloud = static_cast<ccGenericPointCloud*>(entity);
    ccBox* box = ccInnerRect2DFinder().process(cloud,
                                               static_cast<unsigned char>(dim));

    if (box) {
        cloud->addChild(box);
        box->setVisible(true);
        addToDB(box);
    }

    updateUI();
}

// Edit scalar field
void MainWindow::spawnHistogramDialog(const std::vector<unsigned>& histoValues,
                                      double minVal,
                                      double maxVal,
                                      QString title,
                                      QString xAxisLabel) {
    ccHistogramWindowDlg* hDlg = new ccHistogramWindowDlg(this);
    hDlg->setAttribute(Qt::WA_DeleteOnClose, true);
    hDlg->setWindowTitle("Histogram");

    ccHistogramWindow* histogram = hDlg->window();
    {
        histogram->setTitle(title);
        histogram->fromBinArray(histoValues, minVal, maxVal);
        histogram->setAxisLabels(xAxisLabel, "Count");
        histogram->refresh();
    }

    hDlg->show();
}

void MainWindow::showSelectedEntitiesHistogram() {
    for (ccHObject* entity : getSelectedEntities()) {
        // for "real" point clouds only
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(entity);
        if (cloud) {
            // we display the histogram of the current scalar field
            ccScalarField* sf = static_cast<ccScalarField*>(
                    cloud->getCurrentDisplayedScalarField());
            if (sf) {
                ccHistogramWindowDlg* hDlg = new ccHistogramWindowDlg(this);
                hDlg->setAttribute(Qt::WA_DeleteOnClose, true);
                hDlg->setWindowTitle(
                        tr("Histogram [%1]").arg(cloud->getName()));

                ccHistogramWindow* histogram = hDlg->window();
                {
                    unsigned numberOfPoints = cloud->size();
                    unsigned numberOfClasses = static_cast<unsigned>(
                            sqrt(static_cast<double>(numberOfPoints)));
                    // we take the 'nearest' multiple of 4
                    numberOfClasses &= (~3);
                    numberOfClasses = std::max<unsigned>(4, numberOfClasses);
                    numberOfClasses = std::min<unsigned>(256, numberOfClasses);

                    histogram->setTitle(tr("%1 (%2 values) ")
                                                .arg(sf->getName())
                                                .arg(numberOfPoints));
                    bool showNaNValuesInGrey = sf->areNaNValuesShownInGrey();
                    histogram->fromSF(sf, numberOfClasses, true,
                                      showNaNValuesInGrey);
                    histogram->setAxisLabels(sf->getName(), tr("Count"));
                    histogram->refresh();
                }
                hDlg->show();
            }
        }
    }
}

void MainWindow::doActionComputeStatParams() {
    ccEntityAction::computeStatParams(m_selectedEntities, this);
}

void MainWindow::doActionSFGradient() {
    if (!ccLibAlgorithms::ApplyCCLibAlgorithm(
                ccLibAlgorithms::CCLIB_ALGO_SF_GRADIENT, m_selectedEntities,
                this))
        return;
    refreshSelected();
    updateUI();
}

void MainWindow::doActionOpenColorScalesManager() {
    ccColorScaleEditorDialog cseDlg(ccColorScalesManager::GetUniqueInstance(),
                                    this, ccColorScale::Shared(0), this);

    if (cseDlg.exec()) {
        // save current scale manager state to persistent settings
        ccColorScalesManager::GetUniqueInstance()->toPersistentSettings();
    }

    updateUI();
}

void MainWindow::doActionRGBGaussianFilter() {
    ccPointCloud::RgbFilterOptions filterParams;
    filterParams.filterType = ccPointCloud::RGB_FILTER_TYPES::GAUSSIAN;
    if (!ccEntityAction::rgbGaussianFilter(m_selectedEntities, filterParams,
                                           this))
        return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionRGBBilateralFilter() {
    ccPointCloud::RgbFilterOptions filterParams;
    filterParams.filterType = ccPointCloud::RGB_FILTER_TYPES::BILATERAL;
    if (!ccEntityAction::rgbGaussianFilter(m_selectedEntities, filterParams,
                                           this))
        return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionRGBMeanFilter() {
    ccPointCloud::RgbFilterOptions filterParams;
    filterParams.filterType = ccPointCloud::RGB_FILTER_TYPES::MEAN;
    if (!ccEntityAction::rgbGaussianFilter(m_selectedEntities, filterParams,
                                           this))
        return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionRGBMedianFilter() {
    ccPointCloud::RgbFilterOptions filterParams;
    filterParams.filterType = ccPointCloud::RGB_FILTER_TYPES::MEDIAN;
    if (!ccEntityAction::rgbGaussianFilter(m_selectedEntities, filterParams,
                                           this))
        return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionSFGaussianFilter() {
    ccPointCloud::RgbFilterOptions filterParams;
    filterParams.filterType = ccPointCloud::RGB_FILTER_TYPES::GAUSSIAN;
    if (!ccEntityAction::sfGaussianFilter(m_selectedEntities, filterParams,
                                          this))
        return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionSFBilateralFilter() {
    ccPointCloud::RgbFilterOptions filterParams;
    filterParams.filterType = ccPointCloud::RGB_FILTER_TYPES::BILATERAL;
    if (!ccEntityAction::sfGaussianFilter(m_selectedEntities, filterParams,
                                          this))
        return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionFilterByLabel() {
    if (!haveOneSelection()) {
        ecvConsole::Warning(tr("Select one and only one entity!"));
        return;
    }

    ccHObject* entity = m_selectedEntities[0];
    ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(entity);
    if (!cloud || !cloud->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ecvConsole::Warning(tr("only cloud is supported!"));
        return;
    }

    if (!m_filterLabelTool) {
        m_filterLabelTool = new ecvFilterByLabelDlg(this);
        connect(m_filterLabelTool, &ccOverlayDialog::processFinished, this,
                [=]() {
                    ecvDisplayTools::UpdateScreen();
                    freezeUI(false);
                    updateUI();
                });
        registerOverlayDialog(m_filterLabelTool, Qt::TopRightCorner);
    }

    if (!m_filterLabelTool->linkWith(ecvDisplayTools::GetCurrentScreen())) {
        CVLog::Warning(
                "[MainWindow::doSemanticSegmentation] Initialization failed!");
        return;
    }

    if (!m_filterLabelTool->setInputEntity(entity)) {
        return;
    }

    if (m_filterLabelTool->start()) {
        freezeUI(true);
        updateOverlayDialogsPlacement();
    } else {
        freezeUI(false);
        updateUI();
        ecvConsole::Error(tr("Unexpected error!"));  // indeed...
    }
}

void MainWindow::doActionFilterByValue() {
    typedef std::pair<ccHObject*, ccPointCloud*> EntityAndVerticesType;
    std::vector<EntityAndVerticesType> toFilter;

    for (ccHObject* entity : getSelectedEntities()) {
        ccGenericPointCloud* cloud =
                ccHObjectCaster::ToGenericPointCloud(entity);
        if (cloud && cloud->isA(CV_TYPES::POINT_CLOUD)) {
            ccPointCloud* pc = static_cast<ccPointCloud*>(cloud);
            cloudViewer::ScalarField* sf = pc->getCurrentDisplayedScalarField();
            if (sf) {
                toFilter.emplace_back(entity, pc);
            } else {
                ecvConsole::Warning(
                        tr("Entity [%1] has no active scalar field !")
                                .arg(entity->getName()));
            }
        }
    }

    if (toFilter.empty()) return;

    double minVald = 0.0;
    double maxVald = 1.0;

    // compute min and max "displayed" scalar values of currently selected
    // entities (the ones with an active scalar field only!)
    {
        for (size_t i = 0; i < toFilter.size(); ++i) {
            ccScalarField* sf =
                    toFilter[i].second->getCurrentDisplayedScalarField();
            assert(sf);

            if (i == 0) {
                minVald = static_cast<double>(sf->displayRange().start());
                maxVald = static_cast<double>(sf->displayRange().stop());
            } else {
                if (minVald > static_cast<double>(sf->displayRange().start()))
                    minVald = static_cast<double>(sf->displayRange().start());
                if (maxVald < static_cast<double>(sf->displayRange().stop()))
                    maxVald = static_cast<double>(sf->displayRange().stop());
            }
        }
    }

    ccFilterByValueDlg dlg(minVald, maxVald, -1.0e9, 1.0e9, this);
    if (!dlg.exec()) return;

    ccFilterByValueDlg::Mode mode = dlg.mode();
    assert(mode != ccFilterByValueDlg::CANCEL);

    ScalarType minVal = static_cast<ScalarType>(dlg.minDoubleSpinBox->value());
    ScalarType maxVal = static_cast<ScalarType>(dlg.maxDoubleSpinBox->value());

    ccHObject::Container results;
    {
        for (auto& item : toFilter) {
            ccHObject* ent = item.first;
            ccPointCloud* pc = item.second;
            // we set as output (OUT) the currently displayed scalar field
            int outSfIdx = pc->getCurrentDisplayedScalarFieldIndex();
            assert(outSfIdx >= 0);
            pc->setCurrentOutScalarField(outSfIdx);
            // pc->setCurrentScalarField(outSfIdx);

            ccHObject* resultInside = nullptr;
            ccHObject* resultOutside = nullptr;
            if (ent->isKindOf(CV_TYPES::MESH)) {
                pc->hidePointsByScalarValue(minVal, maxVal);
                if (ent->isA(CV_TYPES::MESH)/*|| ent->isKindOf(CV_TYPES::PRIMITIVE)*/) //TODO
                    resultInside = ccHObjectCaster::ToMesh(ent)
                                           ->createNewMeshFromSelection(false);
                else if (ent->isA(CV_TYPES::SUB_MESH))
                    resultInside =
                            ccHObjectCaster::ToSubMesh(ent)
                                    ->createNewSubMeshFromSelection(false);

                if (mode == ccFilterByValueDlg::SPLIT) {
                    pc->invertVisibilityArray();
                    if (ent->isA(CV_TYPES::MESH)/*|| ent->isKindOf(CV_TYPES::PRIMITIVE)*/) //TODO
                        resultOutside =
                                ccHObjectCaster::ToMesh(ent)
                                        ->createNewMeshFromSelection(false);
                    else if (ent->isA(CV_TYPES::SUB_MESH))
                        resultOutside =
                                ccHObjectCaster::ToSubMesh(ent)
                                        ->createNewSubMeshFromSelection(false);
                }

                pc->unallocateVisibilityArray();
            } else if (ent->isKindOf(CV_TYPES::POINT_CLOUD)) {
                // pc->hidePointsByScalarValue(minVal,maxVal);
                // result =
                // ccHObjectCaster::ToGenericPointCloud(ent)->hidePointsByScalarValue(false);
                // pc->unallocateVisibilityArray();

                // shortcut, as we know here that the point cloud is a
                // "ccPointCloud"
                resultInside =
                        pc->filterPointsByScalarValue(minVal, maxVal, false);

                if (mode == ccFilterByValueDlg::SPLIT) {
                    resultOutside =
                            pc->filterPointsByScalarValue(minVal, maxVal, true);
                }
            }

            if (resultInside) {
                ent->setEnabled(false);
                // resultInside->setDisplay(ent->getDisplay());
                // resultInside->prepareDisplayForRefresh();
                addToDB(resultInside);

                results.push_back(resultInside);
            }
            if (resultOutside) {
                ent->setEnabled(false);
                // resultOutside->setDisplay(ent->getDisplay());
                // resultOutside->prepareDisplayForRefresh();
                resultOutside->setName(resultOutside->getName() + ".outside");
                addToDB(resultOutside);

                results.push_back(resultOutside);
            }
        }
    }

    if (!results.empty()) {
        ecvConsole::Warning(
                tr("Previously selected entities (sources) have been hidden!"));
        if (m_ccRoot) {
            m_ccRoot->selectEntities(results);
        }
    }

    // refreshAll();
}

void MainWindow::doActionScalarFieldFromColor() {
    if (!ccEntityAction::sfFromColor(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionSFConvertToRGB() {
    if (!ccEntityAction::sfConvertToRGB(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionSFConvertToRandomRGB() {
    if (!ccEntityAction::sfConvertToRandomRGB(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionToggleActiveSFColorScale() {
    doApplyActiveSFAction(0);
}

void MainWindow::doActionShowActiveSFPrevious() { doApplyActiveSFAction(1); }

void MainWindow::doActionShowActiveSFNext() { doApplyActiveSFAction(2); }

void MainWindow::doApplyActiveSFAction(int action) {
    if (!haveOneSelection()) {
        if (haveSelection()) {
            ecvConsole::Error(tr("Select only one cloud or one mesh!"));
        }
        return;
    }
    ccHObject* ent = m_selectedEntities[0];

    bool lockedVertices;
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(ent, &lockedVertices);

    // for "real" point clouds only
    if (!cloud) return;
    if (lockedVertices && !ent->isAncestorOf(cloud)) {
        // see ccPropertiesTreeDelegate::fillWithMesh
        ecvUtils::DisplayLockedVerticesWarning(ent->getName(), true);
        return;
    }

    assert(cloud);
    int sfIdx = cloud->getCurrentDisplayedScalarFieldIndex();
    switch (action) {
        case 0:  // Toggle SF color scale
            if (sfIdx >= 0) {
                cloud->showSFColorsScale(!cloud->sfColorScaleShown());
            } else {
                ecvConsole::Warning(tr("No active scalar field on entity '%1'")
                                            .arg(ent->getName()));
            }
            break;
        case 1:  // Activate previous SF
            if (sfIdx >= 0) {
                cloud->setCurrentDisplayedScalarField(sfIdx - 1);
            }
            break;
        case 2:  // Activate next SF
            if (sfIdx + 1 <
                static_cast<int>(cloud->getNumberOfScalarFields())) {
                cloud->setCurrentDisplayedScalarField(sfIdx + 1);
            }
            break;
    }

    refreshSelected();
    updateUI();
}

void MainWindow::doActionRenameSF() {
    if (!ccEntityAction::sfRename(m_selectedEntities, this)) return;

    updateUI();
}

static double s_constantSFValue = 0.0;
void MainWindow::doActionAddConstantSF() {
    if (!haveOneSelection()) {
        if (haveSelection())
            ecvConsole::Error(tr("Select only one cloud or one mesh!"));
        return;
    }

    ccHObject* ent = m_selectedEntities[0];

    bool lockedVertices;
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(ent, &lockedVertices);

    // for "real" point clouds only
    if (!cloud) return;
    if (lockedVertices && !ent->isAncestorOf(cloud)) {
        ecvUtils::DisplayLockedVerticesWarning(ent->getName(), true);
        return;
    }

    QString defaultName = tr("Constant");
    unsigned trys = 1;
    while (cloud->getScalarFieldIndexByName(qPrintable(defaultName)) >= 0 ||
           trys > 99) {
        defaultName = tr("Constant #%1").arg(++trys);
    }

    // ask for a name
    bool ok;
    QString sfName = QInputDialog::getText(this, tr("New SF name"),
                                           tr("SF name (must be unique)"),
                                           QLineEdit::Normal, defaultName, &ok);
    if (!ok) return;
    if (sfName.isNull()) {
        CVLog::Error(tr("Invalid name"));
        return;
    }
    if (cloud->getScalarFieldIndexByName(qPrintable(sfName)) >= 0) {
        CVLog::Error(tr("Name already exists!"));
        return;
    }

    ScalarType sfValue = static_cast<ScalarType>(
            QInputDialog::getDouble(this, tr("Add constant value"), tr("value"),
                                    s_constantSFValue, -1.0e9, 1.0e9, 8, &ok));
    if (!ok) return;

    int sfIdx = cloud->getScalarFieldIndexByName(qPrintable(sfName));
    if (sfIdx < 0) sfIdx = cloud->addScalarField(qPrintable(sfName));
    if (sfIdx < 0) {
        CVLog::Error(tr("An error occurred! (see console)"));
        return;
    }

    cloudViewer::ScalarField* sf = cloud->getScalarField(sfIdx);
    assert(sf);
    if (sf) {
        ecvDisplayTools::SetRedrawRecursive(false);
        sf->fill(sfValue);
        sf->computeMinAndMax();
        cloud->setCurrentDisplayedScalarField(sfIdx);
        cloud->showSF(true);
        updateUI();
    }

    CVLog::Print(tr("New scalar field added to %1 (constant value: %2)")
                         .arg(cloud->getName())
                         .arg(sfValue));
}

void MainWindow::doActionImportSFFromFile() {
    if (!haveOneSelection()) {
        if (haveSelection())
            ecvConsole::Error(tr("Select only one cloud or one mesh!"));
        return;
    }

    bool lockedVertices;
    ccHObject* ent = m_selectedEntities[0];
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(ent, &lockedVertices);

    // for "real" point clouds only
    if (!cloud) return;
    if (lockedVertices && !ent->isAncestorOf(cloud)) {
        ecvUtils::DisplayLockedVerticesWarning(ent->getName(), true);
        return;
    }

    QString currentPath =
            ecvSettingManager::getValue(ecvPS::LoadFile(), ecvPS::CurrentPath(),
                                        ecvFileUtils::defaultDocPath())
                    .toString();
    QString filters = "*.labels";
    QString selectedFilter = filters;
    QString selectedFilename =
            QFileDialog::getOpenFileName(this, tr("import sf value from file"),
                                         currentPath, filters, &selectedFilter);

    if (selectedFilename.isEmpty()) {
        // process cancelled by the user
        return;
    }

    // we update current file path
    currentPath = QFileInfo(selectedFilename).absolutePath();
    ecvSettingManager::setValue(ecvPS::LoadFile(), ecvPS::CurrentPath(),
                                currentPath);
    std::string filename = CVTools::FromQString(selectedFilename);

    std::string scalarName =
            CVTools::FromQString(QFileInfo(selectedFilename).baseName());

    std::vector<size_t> scalars;
    if (!CVTools::QMappingReader(filename, scalars)) {
        return;
    }

    if (scalars.size() != cloud->size()) {
        CVLog::Warning("scalar files are probably corrupted and drop it!");
        return;
    }

    std::vector<std::vector<ScalarType>> scalarsVector;
    std::vector<std::vector<size_t>> tempScalarsvector;
    tempScalarsvector.push_back(scalars);
    ccEntityAction::ConvertToScalarType<size_t>(tempScalarsvector,
                                                scalarsVector);
    if (!ccEntityAction::importToSF(m_selectedEntities, scalarsVector,
                                    scalarName.c_str())) {
        CVLog::Error(
                "[MainWindow::doActionImportSFFromFile] import sf failed!");
    } else {
        CVLog::Print(tr("[MainWindow::doActionImportSFFromFile] "
                        "Import sf from file %1 successfully!")
                             .arg(filename.c_str()));
        updateUI();
    }
}

void MainWindow::doActionAddIdField() {
    if (!ccEntityAction::sfAddIdField(m_selectedEntities)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionExportCoordToSF() {
    if (!ccEntityAction::exportCoordToSF(m_selectedEntities, this)) {
        return;
    }

    refreshSelected();
    updateUI();
}

void MainWindow::doActionSetSFAsCoord() {
    if (!ccEntityAction::sfSetAsCoord(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionInterpolateScalarFields() {
    if (!ccEntityAction::interpolateSFs(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doActionScalarFieldArithmetic() {
    if (!ccEntityAction::sfArithmetic(m_selectedEntities, this)) return;

    refreshSelected();
    updateUI();
}

void MainWindow::doRemoveDuplicatePoints() {
    if (!haveSelection()) return;

    bool first = true;

    // persistent setting(s)
    double minDistanceBetweenPoints =
            ecvSettingManager::getValue(ecvPS::DuplicatePointsGroup(),
                                        ecvPS::DuplicatePointsMinDist(),
                                        1.0e-12)
                    .toDouble();

    bool ok;
    minDistanceBetweenPoints = QInputDialog::getDouble(
            this, tr("Remove duplicate points"),
            tr("Min distance between points:"), minDistanceBetweenPoints, 0,
            1.0e8, 12, &ok);
    if (!ok) return;

    // save parameter
    ecvSettingManager::setValue(ecvPS::DuplicatePointsGroup(),
                                ecvPS::DuplicatePointsMinDist(),
                                minDistanceBetweenPoints);

    static const char DEFAULT_DUPLICATE_TEMP_SF_NAME[] = "DuplicateFlags";

    ecvProgressDialog pDlg(true, this);
    pDlg.setAutoClose(false);

    ccHObject::Container selectedEntities =
            getSelectedEntities();  // we have to use a local copy:
                                    // 'unselectAllEntities' and 'selectEntity'
                                    // will change the set of currently selected
                                    // entities!

    for (ccHObject* entity : selectedEntities) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(entity);
        if (cloud) {
            // create temporary SF for 'duplicate flags'
            int sfIdx = cloud->getScalarFieldIndexByName(
                    DEFAULT_DUPLICATE_TEMP_SF_NAME);
            if (sfIdx < 0)
                sfIdx = cloud->addScalarField(DEFAULT_DUPLICATE_TEMP_SF_NAME);
            if (sfIdx >= 0)
                cloud->setCurrentScalarField(sfIdx);
            else {
                ecvConsole::Error(
                        tr("Couldn't create temporary scalar field! Not enough "
                           "memory?"));
                break;
            }

            ccOctree::Shared octree = cloud->getOctree();

            cloudViewer::GeometricalAnalysisTools::ErrorCode result =
                    cloudViewer::GeometricalAnalysisTools::FlagDuplicatePoints(
                            cloud, minDistanceBetweenPoints, &pDlg,
                            octree.data());

            if (result == cloudViewer::GeometricalAnalysisTools::NoError) {
                // count the number of duplicate points!
                cloudViewer::ScalarField* flagSF = cloud->getScalarField(sfIdx);
                unsigned duplicateCount = 0;
                assert(flagSF);
                if (flagSF) {
                    for (unsigned j = 0; j < flagSF->currentSize(); ++j) {
                        if (flagSF->getValue(j) != 0) {
                            ++duplicateCount;
                        }
                    }
                }

                if (duplicateCount == 0) {
                    ecvConsole::Print(tr("Cloud '%1' has no duplicate points")
                                              .arg(cloud->getName()));
                } else {
                    ecvConsole::Warning(
                            tr("Cloud '%1' has %2 duplicate point(s)")
                                    .arg(cloud->getName())
                                    .arg(duplicateCount));

                    ccPointCloud* filteredCloud =
                            cloud->filterPointsByScalarValue(0, 0);
                    if (filteredCloud) {
                        int sfIdx2 = filteredCloud->getScalarFieldIndexByName(
                                DEFAULT_DUPLICATE_TEMP_SF_NAME);
                        assert(sfIdx2 >= 0);
                        filteredCloud->deleteScalarField(sfIdx2);
                        filteredCloud->setName(
                                tr("%1.clean").arg(cloud->getName()));
                        // filteredCloud->setDisplay(cloud->getDisplay());
                        // filteredCloud->prepareDisplayForRefresh();
                        addToDB(filteredCloud);
                        if (first) {
                            m_ccRoot->unselectAllEntities();
                            first = false;
                        }
                        cloud->setEnabled(false);
                        m_ccRoot->selectEntity(filteredCloud, true);
                    }
                }
            } else {
                ecvConsole::Error(
                        tr("An error occurred! (Not enough memory?)"));
            }

            cloud->deleteScalarField(sfIdx);
        }
    }

    if (!first)
        ecvConsole::Warning(
                tr("Previously selected entities (sources) have been hidden!"));
}

void MainWindow::doActionSubsample() {
    // find candidates
    std::vector<ccPointCloud*> clouds;
    unsigned maxPointCount = 0;
    double maxCloudRadius = 0;
    ScalarType sfMin = NAN_VALUE;
    ScalarType sfMax = NAN_VALUE;
    {
        for (ccHObject* entity : getSelectedEntities()) {
            if (entity->isA(CV_TYPES::POINT_CLOUD)) {
                ccPointCloud* cloud = static_cast<ccPointCloud*>(entity);
                clouds.push_back(cloud);

                maxPointCount =
                        std::max<unsigned>(maxPointCount, cloud->size());
                maxCloudRadius = std::max<double>(
                        maxCloudRadius, cloud->getOwnBB().getDiagNorm());

                // we also look for the min and max sf values
                ccScalarField* sf = cloud->getCurrentDisplayedScalarField();
                if (sf) {
                    if (!ccScalarField::ValidValue(sfMin) ||
                        sfMin > sf->getMin())
                        sfMin = sf->getMin();
                    if (!ccScalarField::ValidValue(sfMax) ||
                        sfMax < sf->getMax())
                        sfMax = sf->getMax();
                }
            }
        }
    }

    if (clouds.empty()) {
        ecvConsole::Error(tr("Select at least one point cloud!"));
        return;
    }

    // Display dialog
    ccSubsamplingDlg sDlg(maxPointCount, maxCloudRadius, this);
    bool hasValidSF = ccScalarField::ValidValue(sfMin) &&
                      ccScalarField::ValidValue(sfMax);
    if (hasValidSF) sDlg.enableSFModulation(sfMin, sfMax);
    if (!sDlg.exec()) return;

    // process clouds
    ccHObject::Container resultingClouds;
    {
        ecvProgressDialog pDlg(false, this);
        pDlg.setAutoClose(false);

        pDlg.setMethodTitle(tr("Subsampling"));

        bool errors = false;

        QElapsedTimer eTimer;
        eTimer.start();

        for (size_t i = 0; i < clouds.size(); ++i) {
            ccPointCloud* cloud = clouds[i];
            cloudViewer::ReferenceCloud* sampledCloud =
                    sDlg.getSampledCloud(cloud, &pDlg);
            if (!sampledCloud) {
                ecvConsole::Warning(
                        tr("[Subsampling] Failed to subsample cloud '%1'!")
                                .arg(cloud->getName()));
                errors = true;
                continue;
            }

            int warnings = 0;
            ccPointCloud* newPointCloud =
                    cloud->partialClone(sampledCloud, &warnings);

            delete sampledCloud;
            sampledCloud = 0;

            if (newPointCloud) {
                newPointCloud->setName(cloud->getName() +
                                       QString(".subsampled"));
                newPointCloud->setGlobalShift(cloud->getGlobalShift());
                newPointCloud->setGlobalScale(cloud->getGlobalScale());
                if (cloud->getParent())
                    cloud->getParent()->addChild(newPointCloud);
                cloud->setEnabled(false);
                addToDB(newPointCloud);

                resultingClouds.push_back(newPointCloud);

                if (warnings) {
                    CVLog::Warning(
                            tr("[Subsampling] Not enough memory: colors, "
                               "normals or scalar fields may be missing!"));
                    errors = true;
                }
            } else {
                CVLog::Error(tr("Not enough memory!"));
                break;
            }
        }

        CVLog::Print(tr("[Subsampling] Timing: %1 s.")
                             .arg(eTimer.elapsed() / 1000.0, 7));

        if (errors) {
            CVLog::Error(tr("Errors occurred (see console)"));
        }
    }

    if (m_ccRoot) m_ccRoot->selectEntities(resultingClouds);

    updateUI();
}

void MainWindow::doActionEditGlobalShiftAndScale() {
    // get the global shift/scale info and bounding box of all selected clouds
    std::vector<std::pair<ccShiftedObject*, ccHObject*>> shiftedEntities;
    CCVector3d Pl(0, 0, 0);
    double Dl = 1.0;
    CCVector3d Pg(0, 0, 0);
    double Dg = 1.0;
    // shift and scale (if unique)
    CCVector3d shift(0, 0, 0);
    double scale = 1.0;
    {
        bool uniqueShift = true;
        bool uniqueScale = true;
        ccBBox localBB;
        // sadly we don't have a double-typed bounding box class yet ;)
        CCVector3d globalBBmin(0, 0, 0), globalBBmax(0, 0, 0);

        for (ccHObject* entity : getSelectedEntities()) {
            bool lockedVertices;
            ccShiftedObject* shifted =
                    ccHObjectCaster::ToShifted(entity, &lockedVertices);
            if (!shifted) {
                continue;
            }
            // for (unlocked) entities only
            if (lockedVertices) {
                // get the vertices
                assert(entity->isKindOf(CV_TYPES::MESH));
                ccGenericPointCloud* vertices =
                        static_cast<ccGenericMesh*>(entity)
                                ->getAssociatedCloud();
                if (!vertices || !entity->isAncestorOf(vertices)) {
                    ecvUtils::DisplayLockedVerticesWarning(entity->getName(),
                                                           haveOneSelection());
                    continue;
                }
                entity = vertices;
            }

            CCVector3 Al = entity->getOwnBB().minCorner();
            CCVector3 Bl = entity->getOwnBB().maxCorner();
            CCVector3d Ag = shifted->toGlobal3d<PointCoordinateType>(Al);
            CCVector3d Bg = shifted->toGlobal3d<PointCoordinateType>(Bl);

            // update local BB
            localBB.add(Al);
            localBB.add(Bl);

            // update global BB
            if (shiftedEntities.empty()) {
                globalBBmin = Ag;
                globalBBmax = Bg;
                shift = shifted->getGlobalShift();
                uniqueScale = shifted->getGlobalScale();
            } else {
                globalBBmin = CCVector3d(std::min(globalBBmin.x, Ag.x),
                                         std::min(globalBBmin.y, Ag.y),
                                         std::min(globalBBmin.z, Ag.z));
                globalBBmax = CCVector3d(std::max(globalBBmax.x, Bg.x),
                                         std::max(globalBBmax.y, Bg.y),
                                         std::max(globalBBmax.z, Bg.z));

                if (uniqueShift) {
                    uniqueShift = cloudViewer::LessThanEpsilon(
                            (shifted->getGlobalShift() - shift).norm());
                }
                if (uniqueScale) {
                    uniqueScale = cloudViewer::LessThanEpsilon(
                            std::abs(shifted->getGlobalScale() - scale));
                }
            }

            shiftedEntities.emplace_back(shifted, entity);
        }

        Pg = globalBBmin;
        Dg = (globalBBmax - globalBBmin).norm();

        Pl = CCVector3d::fromArray(localBB.minCorner().u);
        Dl = (localBB.maxCorner() - localBB.minCorner()).normd();

        if (!uniqueShift) shift = Pl - Pg;
        if (!uniqueScale) scale = Dg / Dl;
    }

    if (shiftedEntities.empty()) {
        return;
    }

    ecvShiftAndScaleCloudDlg sasDlg(Pl, Dl, Pg, Dg, this);
    sasDlg.showApplyAllButton(shiftedEntities.size() > 1);
    sasDlg.showApplyButton(shiftedEntities.size() == 1);
    sasDlg.showNoButton(false);
    sasDlg.setShiftFieldsPrecision(6);
    // add "original" entry
    int index = sasDlg.addShiftInfo(
            ecvGlobalShiftManager::ShiftInfo(tr("Original"), shift, scale));
    sasDlg.setCurrentProfile(index);
    // add "last" entry (if available)
    std::vector<ecvGlobalShiftManager::ShiftInfo> lastInfos;
    if (ecvGlobalShiftManager::GetLast(lastInfos)) {
        sasDlg.addShiftInfo(lastInfos);
    }
    // add entries from file (if any)
    sasDlg.addFileInfo();

    if (!sasDlg.exec()) return;

    shift = sasDlg.getShift();
    scale = sasDlg.getScale();
    bool preserveGlobalPos = sasDlg.keepGlobalPos();

    CVLog::Print(tr("[Global Shift/Scale] New shift: (%1, %2, %3)")
                         .arg(shift.x)
                         .arg(shift.y)
                         .arg(shift.z));
    CVLog::Print(tr("[Global Shift/Scale] New scale: %1").arg(scale));

    // apply new shift
    {
        for (auto& entity : shiftedEntities) {
            ccShiftedObject* shifted = entity.first;
            ccHObject* ent = entity.second;
            if (preserveGlobalPos) {
                // to preserve the global position of the cloud, we may have to
                // translate and/or rescale the cloud
                CCVector3d Ql =
                        CCVector3d::fromArray(ent->getOwnBB().minCorner().u);
                CCVector3d Qg = shifted->toGlobal3d(Ql);
                CCVector3d Ql2 = Qg * scale + shift;
                CCVector3d T = Ql2 - Ql;

                assert(shifted->getGlobalScale() > 0);
                double scaleCoef = scale / shifted->getGlobalScale();

                if (cloudViewer::GreaterThanEpsilon(T.norm()) ||
                    cloudViewer::GreaterThanEpsilon(
                            std::abs(scaleCoef - 1.0))) {
                    ccGLMatrix transMat;
                    transMat.toIdentity();
                    transMat.scale(static_cast<float>(scaleCoef));
                    transMat.setTranslation(T);

                    // DGM FIXME: we only test the entity own bounding box (and
                    // we update its shift & scale info) but we apply the
                    // transformation to all its children?!
                    ent->applyGLTransformation_recursive(&transMat);
                    // ent->prepareDisplayForRefresh_recursive();

                    CVLog::Warning(
                            tr("[Global Shift/Scale] To preserve its original "
                               "position, the entity '%1' has been translated "
                               "of (%2,%3,%4) and rescaled of a factor %5")
                                    .arg(ent->getName())
                                    .arg(T.x)
                                    .arg(T.y)
                                    .arg(T.z)
                                    .arg(scaleCoef));
                }
            }
            shifted->setGlobalShift(shift);
            shifted->setGlobalScale(scale);
        }
    }

    refreshSelected();
    updateUI();
}

// Tools measurement menu methods
void MainWindow::activateDistanceMode() {
#ifdef USE_PCL_BACKEND
    doActionMeasurementMode(
            ecvGenericMeasurementTools::MeasurementType::DISTANCE_WIDGET);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::activateProtractorMode() {
#ifdef USE_PCL_BACKEND
    doActionMeasurementMode(
            ecvGenericMeasurementTools::MeasurementType::PROTRACTOR_WIDGET);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::activateContourMode() {
#ifdef USE_PCL_BACKEND
    doActionMeasurementMode(
            ecvGenericMeasurementTools::MeasurementType::CONTOUR_WIDGET);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::doActionMeasurementMode(int mode) {
    if (!haveOneSelection()) return;

    // we have to use a local copy: 'unselectEntity' will change the set of
    // currently selected entities!
    ccHObject::Container selectedEntities = getSelectedEntities();

    if (!m_measurementTool) {
        m_measurementTool = new ecvMeasurementTool(this);
        connect(m_measurementTool, &ccOverlayDialog::processFinished, this,
                [=]() {
                    ccHObject::Container outs = m_measurementTool->getOutputs();
                    for (ccHObject* entity : outs) {
                        entity->setEnabled(true);
                    }

                    if (!outs.empty()) {
                        // hide origin entities.
                        for (ccHObject* entity : selectedEntities) {
                            entity->setEnabled(false);
                        }

                        m_ccRoot->selectEntities(outs);
                        refreshSelected();
                    }

                    freezeUI(false);
                    updateUI();
                });
        registerOverlayDialog(m_measurementTool, Qt::TopRightCorner);
    }

#ifdef USE_PCL_BACKEND
    ecvGenericVisualizer3D* viewer = ecvDisplayTools::GetVisualizer3D();
    if (!viewer) {
        CVLog::Error("[MainWindow] No visualizer available!");
        return;
    }

    ecvGenericMeasurementTools* measurementTool = new PclMeasurementTools(
            viewer, ecvGenericMeasurementTools::MeasurementType(mode));

    // Add the new tool instance to the measurement tool dialog
    m_measurementTool->setMeasurementTool(measurementTool);
    m_measurementTool->linkWith(ecvDisplayTools::GetCurrentScreen());

    for (ccHObject* entity : selectedEntities) {
        if (m_measurementTool->addAssociatedEntity(entity)) {
            // automatically deselect the entity (to avoid seeing its bounding
            // box ;)
            m_ccRoot->unselectEntity(entity);
        }
    }

    if (m_measurementTool->getNumberOfAssociatedEntity() == 0) {
        CVLog::Warning("[MainWindow] No valid entities for measurement!");
        return;
    }

    if (m_measurementTool->start()) {
        updateOverlayDialogsPlacement();
        ecvDisplayTools::UpdateScreen();
    } else {
        freezeUI(false);
        updateUI();
        ecvConsole::Error(tr("Unexpected error!"));
    }
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::activateClippingMode() {
#ifdef USE_PCL_BACKEND
    doActionFilterMode(ecvGenericFiltersTool::FilterType::CLIP_FILTER);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::activateSliceMode() {
#ifdef USE_PCL_BACKEND
    doActionFilterMode(ecvGenericFiltersTool::FilterType::SLICE_FILTER);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::activateProbeMode() {
#ifdef USE_PCL_BACKEND
    doActionFilterMode(ecvGenericFiltersTool::FilterType::PROBE_FILTER);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::activateDecimateMode() {
#ifdef USE_PCL_BACKEND
    doActionFilterMode(ecvGenericFiltersTool::FilterType::DECIMATE_FILTER);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::activateIsoSurfaceMode() {
#ifdef USE_PCL_BACKEND
    doActionFilterMode(ecvGenericFiltersTool::FilterType::ISOSURFACE_FILTER);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::activateThresholdMode() {
#ifdef USE_PCL_BACKEND
    doActionFilterMode(ecvGenericFiltersTool::FilterType::THRESHOLD_FILTER);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::activateSmoothMode() {
#ifdef USE_PCL_BACKEND
    doActionFilterMode(ecvGenericFiltersTool::FilterType::SMOOTH_FILTER);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::activateGlyphMode() {
#ifdef USE_PCL_BACKEND
    doActionFilterMode(ecvGenericFiltersTool::FilterType::GLYPH_FILTER);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::activateStreamlineMode() {
#ifdef USE_PCL_BACKEND
    doActionFilterMode(ecvGenericFiltersTool::FilterType::STREAMLINE_FILTER);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::doActionFilterMode(int mode) {
    if (!haveOneSelection()) return;

#ifdef USE_PCL_BACKEND
    ecvGenericFiltersTool* filter =
            new PclFiltersTool(ecvDisplayTools::GetVisualizer3D(),
                               ecvGenericFiltersTool::FilterType(mode));
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND

    // we have to use a local copy: 'unselectEntity' will change the set of
    // currently selected entities!
    ccHObject::Container selectedEntities = getSelectedEntities();
    if (!m_filterTool) {
        m_filterTool = new ecvFilterTool(this);
        connect(m_filterTool, &ccOverlayDialog::processFinished, this, [=]() {
            ccHObject::Container outs = m_filterTool->getOutputs();
            for (ccHObject* entity : outs) {
                entity->setEnabled(true);
            }

            if (!outs.empty()) {
                // hide origin entities.
                for (ccHObject* entity : selectedEntities) {
                    entity->setEnabled(false);
                }

                m_ccRoot->selectEntities(outs);
                refreshSelected();
            }

            freezeUI(false);
            updateUI();
        });
    }

    m_filterTool->setFilter(filter);
    m_filterTool->linkWith(ecvDisplayTools::GetCurrentScreen());

    for (ccHObject* entity : selectedEntities) {
        if (m_filterTool->addAssociatedEntity(entity)) {
            // automatically deselect the entity (to avoid seeing its bounding
            // box ;)
            m_ccRoot->unselectEntity(entity);
        }
    }

    if (m_filterTool->getNumberOfAssociatedEntity() == 0) {
        m_filterTool->close();
        return;
    }

    freezeUI(true);
    m_ui->ViewToolBar->setDisabled(false);

    if (m_filterTool->start()) {
        registerOverlayDialog(m_filterTool, Qt::TopRightCorner);
        freezeUI(true);
        updateOverlayDialogsPlacement();
    } else {
        ecvConsole::Error(tr("Unexpected error!"));  // indeed...
    }
}

void MainWindow::doBoxAnnotation() {
#ifdef USE_PCL_BACKEND
    doAnnotations(ecvGenericAnnotationTool::AnnotationMode::BOUNDINGBOX);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::doSemanticAnnotation() {
#ifdef USE_PCL_BACKEND
    doAnnotations(ecvGenericAnnotationTool::AnnotationMode::SEMANTICS);
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND
}

void MainWindow::doAnnotations(int mode) {
    if (!haveOneSelection()) {
        if (haveSelection()) ecvConsole::Error(tr("Select only one cloud!"));
        return;
    }

    ccHObject* ent = m_selectedEntities[0];
    if (!ent->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ecvConsole::Error(tr("only point cloud is supported!"));
        return;
    }

#ifdef USE_PCL_BACKEND
    PclAnnotationTool* annoTools = new PclAnnotationTool(
            ecvDisplayTools::GetVisualizer3D(),
            ecvGenericAnnotationTool::AnnotationMode(mode));
    if (!m_annoTool) {
        m_annoTool = new ecvAnnotationsTool(this);
        connect(m_annoTool, &ccOverlayDialog::processFinished, this, [=]() {
            ecvDisplayTools::UpdateScreen();
            freezeUI(false);
            updateUI();
        });
    }
#else
    CVLog::Warning(
            "[MainWindow] please use pcl as backend and then try again!");
    return;
#endif  // USE_PCL_BACKEND

    if (!m_annoTool->setAnnotationsTool(annoTools) ||
        !m_annoTool->linkWith(ecvDisplayTools::GetCurrentScreen())) {
        CVLog::Warning("[MainWindow::doAnnotations] Initialization failed!");
        return;
    }

    if (m_annoTool->addAssociatedEntity(ent)) {
        // automatically deselect the entity (to avoid seeing its bounding box
        // ;)
        m_ccRoot->unselectEntity(ent);
    }

    if (m_annoTool->getNumberOfAssociatedEntity() == 0) {
        m_annoTool->close();
        return;
    }

    freezeUI(true);
    m_ui->ViewToolBar->setDisabled(false);

    if (m_annoTool->start()) {
        registerOverlayDialog(m_annoTool, Qt::TopRightCorner);
        freezeUI(true);
        updateOverlayDialogsPlacement();
    } else {
        ecvConsole::Error(tr("Unexpected error!"));  // indeed...
    }
}

void MainWindow::doSemanticSegmentation() {
#ifdef USE_PYTHON_MODULE
    if (!haveSelection()) return;

    if (!m_dssTool) {
        m_dssTool = new ecvDeepSemanticSegmentationTool(this);
        connect(m_dssTool, &ccOverlayDialog::processFinished, this,
                &MainWindow::deactivateSemanticSegmentation);
        registerOverlayDialog(m_dssTool, Qt::TopRightCorner);
    }

    if (!m_dssTool->linkWith(ecvDisplayTools::GetCurrentScreen())) {
        CVLog::Warning(
                "[MainWindow::doSemanticSegmentation] Initialization failed!");
        return;
    }

    for (ccHObject* ent : getSelectedEntities()) {
        if (m_dssTool->addEntity(ent)) {
            // automatically deselect the entity (to avoid seeing its bounding
            // box ;)
            m_ccRoot->unselectEntity(ent);
        }
    }

    if (m_dssTool->getNumberOfValidEntities() == 0) {
        m_dssTool->close();
        ecvConsole::Warning(tr("no valid point cloud is selected!"));
        return;
    }

    if (m_dssTool->start()) {
        freezeUI(true);
        updateOverlayDialogsPlacement();
    } else {
        ecvConsole::Error(tr("Unexpected error!"));  // indeed...
    }

#else
    CVLog::Warning("python interface library has not been compiled!");
    return;
#endif  // USE_PYTHON_MODULE
}

void MainWindow::deactivateSemanticSegmentation(bool state) {
#ifdef USE_PYTHON_MODULE
    if (m_dssTool && state) {
        ccHObject::Container result;
        m_dssTool->getSegmentations(result);
        ccHObject::Container segmentedEntities;
        if (!result.empty()) {
            for (ccHObject* obj : result) {
                addToDB(obj);
                for (unsigned i = 0; i < obj->getChildrenNumber(); ++i) {
                    segmentedEntities.push_back(obj->getChild(i));
                }
            }
            m_ccRoot->selectEntities(segmentedEntities);
        } else {
            CVLog::Print(
                    tr("segmentation info has been exported to sf field!"));
        }
    }

    freezeUI(false);

    updateUI();
#endif  // USE_PYTHON_MODULE
}

void MainWindow::doActionDBScanCluster() {
    if (!haveSelection()) {
        return;
    }

    ccHObject::Container clouds;
    for (auto ent : getSelectedEntities()) {
        if (!ent->isKindOf(CV_TYPES::POINT_CLOUD)) {
            CVLog::Warning("only point cloud is supported!");
            continue;
        }
        clouds.push_back(ent);
    }

    if (!ccEntityAction::DBScanCluster(clouds, this)) {
        ecvConsole::Error(tr("Error(s) occurred! See the Console messages"));
        return;
    }

    refreshSelected();
    updateUI();
}

void MainWindow::doActionPlaneSegmentation() {
    if (!haveSelection()) {
        return;
    }

    ccHObject::Container clouds;
    for (auto ent : getSelectedEntities()) {
        if (!ent->isKindOf(CV_TYPES::POINT_CLOUD)) {
            CVLog::Warning("only point cloud is supported!");
            continue;
        }
        clouds.push_back(ent);
    }

    ccHObject::Container entities;
    if (ccEntityAction::RansacSegmentation(clouds, entities, this)) {
        for (size_t i = 0; i < entities.size(); ++i) {
            addToDB(entities[i]);
        }
    } else {
        ecvConsole::Error(tr("Error(s) occurred! See the Console messages"));
        return;
    }

    updateUI();
}

void MainWindow::activateSegmentationMode() {
    if (!haveSelection()) return;

    if (!m_gsTool) {
        m_gsTool = new ccGraphicalSegmentationTool(this);
        connect(m_gsTool, &ccOverlayDialog::processFinished, this,
                &MainWindow::deactivateSegmentationMode);
        registerOverlayDialog(m_gsTool, Qt::TopRightCorner);
    }

    m_gsTool->linkWith(ecvDisplayTools::GetCurrentScreen());

    for (ccHObject* entity : getSelectedEntities()) {
        entity->setSelected_recursive(false);
        m_gsTool->addEntity(entity);
    }

    if (m_gsTool->getNumberOfValidEntities() == 0) {
        ecvConsole::Error(tr("No segmentable entity in active window!"));
        return;
    }

    freezeUI(true);
    m_ui->ViewToolBar->setDisabled(false);

    if (!m_gsTool->start()) {
        deactivateSegmentationMode(false);
    } else {
        updateOverlayDialogsPlacement();
        bool perspectiveEnabled = ecvDisplayTools::GetPerspectiveState();
        if (!perspectiveEnabled)  // segmentation must work in perspective mode
        {
            doActionPerspectiveProjection();
            m_lastViewMode = VIEWMODE::ORTHOGONAL;
        } else {
            m_lastViewMode = VIEWMODE::PERSPECTIVE;
        }
    }
}

void MainWindow::deactivateSegmentationMode(bool state) {
    bool deleteHiddenParts = false;

    // shall we apply segmentation?
    if (state) {
        ccHObject* firstResult = nullptr;

        deleteHiddenParts = m_gsTool->deleteHiddenParts();

        // aditional vertices of which visibility array should be manually reset
        std::unordered_set<ccGenericPointCloud*> verticesToReset;

        QSet<ccHObject*>& segmentedEntities = m_gsTool->entities();
        for (QSet<ccHObject*>::iterator p = segmentedEntities.begin();
             p != segmentedEntities.end();) {
            ccHObject* entity = (*p);

            if (entity->isKindOf(CV_TYPES::POINT_CLOUD) ||
                entity->isKindOf(CV_TYPES::MESH)) {
                // first, do the things that must absolutely be done BEFORE
                // removing the entity from DB (even temporarily) bool
                // lockedVertices;
                ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(
                        entity /*,&lockedVertices*/);
                assert(cloud);
                if (cloud) {
                    // specific case: labels (do this before temporarily
                    // removing 'entity' from DB!)
                    ccHObject::Container labels;
                    if (m_ccRoot) {
                        m_ccRoot->getRootEntity()->filterChildren(
                                labels, true, CV_TYPES::LABEL_2D);
                    }
                    for (ccHObject::Container::iterator it = labels.begin();
                         it != labels.end(); ++it) {
                        // Warning: cc2DViewportLabel is also a kind of
                        // 'CV_TYPES::LABEL_2D'!
                        if ((*it)->isA(CV_TYPES::LABEL_2D)) {
                            // we must search for all dependent labels and
                            // remove them!!!
                            // TODO: couldn't we be more clever and update the
                            // label instead?
                            cc2DLabel* label = static_cast<cc2DLabel*>(*it);
                            bool removeLabel = false;
                            for (unsigned i = 0; i < label->size(); ++i) {
                                if (label->getPickedPoint(i).cloud == entity) {
                                    removeLabel = true;
                                    break;
                                }
                            }

                            if (removeLabel && label->getParent()) {
                                CVLog::Warning(
                                        tr("[Segmentation] Label %1 depends on "
                                           "cloud %2 and will be removed")
                                                .arg(label->getName(),
                                                     cloud->getName()));
                                ccHObject* labelParent = label->getParent();
                                ccHObjectContext objContext =
                                        removeObjectTemporarilyFromDBTree(
                                                labelParent);
                                labelParent->removeChild(label);
                                label = nullptr;
                                putObjectBackIntoDBTree(labelParent,
                                                        objContext);
                            }
                        }
                    }  // for each label
                }  // if (cloud)

                // we temporarily detach the entity, as it may undergo
                //"severe" modifications (octree deletion, etc.) --> see
                // ccPointCloud::createNewCloudFromVisibilitySelection
                ccHObjectContext objContext =
                        removeObjectTemporarilyFromDBTree(entity);

                // apply segmentation
                ccHObject* segmentationResult = nullptr;
                bool deleteOriginalEntity = deleteHiddenParts;
                if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
                    ccGenericPointCloud* genCloud =
                            ccHObjectCaster::ToGenericPointCloud(entity);
                    ccGenericPointCloud* segmentedCloud =
                            genCloud->createNewCloudFromVisibilitySelection(
                                    !deleteHiddenParts);
                    if (segmentedCloud && segmentedCloud->size() == 0) {
                        delete segmentationResult;
                        segmentationResult = nullptr;
                    } else {
                        segmentationResult = segmentedCloud;
                    }

                    deleteOriginalEntity |= (genCloud->size() == 0);
                }
				else if (entity->isKindOf(CV_TYPES::MESH)/*|| entity->isA(CV_TYPES::PRIMITIVE)*/) //TODO
				{
                    if (entity->isA(CV_TYPES::MESH)) {
                        segmentationResult =
                                ccHObjectCaster::ToMesh(entity)
                                        ->createNewMeshFromSelection(
                                                !deleteHiddenParts);
                    } else if (entity->isA(CV_TYPES::SUB_MESH)) {
                        segmentationResult =
                                ccHObjectCaster::ToSubMesh(entity)
                                        ->createNewSubMeshFromSelection(
                                                !deleteHiddenParts);
                    }

                    deleteOriginalEntity |=
                            (ccHObjectCaster::ToGenericMesh(entity)->size() ==
                             0);
                }

                if (segmentationResult) {
                    assert(cloud);

                    // we must take care of the remaining part
                    if (!deleteHiddenParts) {
                        // no need to put back the entity in DB if we delete it
                        // afterwards!
                        if (!deleteOriginalEntity) {
                            entity->setName(entity->getName() +
                                            QString(".remaining"));
                            putObjectBackIntoDBTree(entity, objContext);
                        }
                    } else {
                        // keep original name(s)
                        segmentationResult->setName(entity->getName());
                        if (entity->isKindOf(CV_TYPES::MESH) &&
                            segmentationResult->isKindOf(CV_TYPES::MESH)) {
                            ccGenericMesh* meshEntity =
                                    ccHObjectCaster::ToGenericMesh(entity);
                            ccHObjectCaster::ToGenericMesh(segmentationResult)
                                    ->getAssociatedCloud()
                                    ->setName(meshEntity->getAssociatedCloud()
                                                      ->getName());

                            // specific case: if the sub mesh is deleted
                            // afterwards (see below) then its associated
                            // vertices won't be 'reset' by the segmentation
                            // tool!
                            if (deleteHiddenParts &&
                                meshEntity->isA(CV_TYPES::SUB_MESH)) {
                                verticesToReset.insert(
                                        meshEntity->getAssociatedCloud());
                            }
                        }
                        assert(deleteOriginalEntity);
                        // deleteOriginalEntity = true;
                    }

                    if (segmentationResult->isA(CV_TYPES::SUB_MESH)) {
                        // for sub-meshes, we have no choice but to use its
                        // parent mesh!
                        objContext.parent =
                                static_cast<ccSubMesh*>(segmentationResult)
                                        ->getAssociatedMesh();
                    } else {
                        // otherwise we look for first non-mesh or non-cloud
                        // parent
                        while (objContext.parent &&
                               (objContext.parent->isKindOf(CV_TYPES::MESH) ||
                                objContext.parent->isKindOf(
                                        CV_TYPES::POINT_CLOUD))) {
                            objContext.parent = objContext.parent->getParent();
                        }
                    }

                    if (objContext.parent) {
                        objContext.parent->addChild(
                                segmentationResult);  // FiXME:
                                                      // objContext.parentFlags?
                    }

                    addToDB(segmentationResult);

                    if (!firstResult) {
                        firstResult = segmentationResult;
                    }
                } else if (!deleteOriginalEntity) {
                    // ecvConsole::Error("An error occurred! (not enough
                    // memory?)");
                    putObjectBackIntoDBTree(entity, objContext);
                }

                if (deleteOriginalEntity) {
                    p = segmentedEntities.erase(p);

                    delete entity;
                    entity = nullptr;
                } else {
                    ++p;
                }
            }
        }

        // specific actions
        {
            for (ccGenericPointCloud* cloud : verticesToReset) {
                cloud->resetVisibilityArray();
            }
        }

        if (firstResult && m_ccRoot) {
            m_ccRoot->selectEntity(firstResult);
        }
    }

    if (m_gsTool) {
        if (m_lastViewMode == VIEWMODE::ORTHOGONAL) {
            doActionOrthogonalProjection();
        } else if (m_lastViewMode == VIEWMODE::PERSPECTIVE) {
            doActionPerspectiveProjection();
        }

        ecvDisplayTools::DisplayNewMessage(
                QString(),
                ecvDisplayTools::UPPER_CENTER_MESSAGE);  // clear the area
        ecvDisplayTools::SetRedrawRecursive(false);
        m_gsTool->removeAllEntities();
        if (ecvDisplayTools::GetCurrentScreen()) {
            refreshAll();
        }
        ecvDisplayTools::SetInteractionMode(
                ecvDisplayTools::TRANSFORM_CAMERA());
    }

    freezeUI(false);

    updateUI();
}

struct ComponentIndexAndSize {
    unsigned index;
    unsigned size;

    ComponentIndexAndSize(unsigned i, unsigned s) : index(i), size(s) {}

    static bool DescendingCompOperator(const ComponentIndexAndSize& a,
                                       const ComponentIndexAndSize& b) {
        return a.size > b.size;
    }
};

void MainWindow::createComponentsClouds(
        ccGenericPointCloud* cloud,
        cloudViewer::ReferenceCloudContainer& components,
        unsigned minPointsPerComponent,
        bool randomColors,
        bool selectComponents,
        bool sortBysize /*=true*/) {
    if (!cloud || components.empty()) return;

    std::vector<ComponentIndexAndSize> sortedIndexes;
    std::vector<ComponentIndexAndSize>* _sortedIndexes = nullptr;
    if (sortBysize) {
        try {
            sortedIndexes.reserve(components.size());
        } catch (const std::bad_alloc&) {
            CVLog::Warning(
                    tr("[CreateComponentsClouds] Not enough memory to sort "
                       "components by size!"));
            sortBysize = false;
        }

        if (sortBysize)  // still ok?
        {
            unsigned compCount = static_cast<unsigned>(components.size());
            for (unsigned i = 0; i < compCount; ++i) {
                sortedIndexes.emplace_back(i, components[i]->size());
            }

            ParallelSort(sortedIndexes.begin(), sortedIndexes.end(),
                         ComponentIndexAndSize::DescendingCompOperator);

            _sortedIndexes = &sortedIndexes;
        }
    }

    // we create "real" point clouds for all input components
    {
        ccPointCloud* pc = cloud->isA(CV_TYPES::POINT_CLOUD)
                                   ? static_cast<ccPointCloud*>(cloud)
                                   : 0;

        // we create a new group to store all CCs
        ccHObject* ccGroup =
                new ccHObject(cloud->getName() + QString(" [CCs]"));

        // for each component
        for (size_t i = 0; i < components.size(); ++i) {
            cloudViewer::ReferenceCloud* compIndexes =
                    _sortedIndexes ? components[_sortedIndexes->at(i).index]
                                   : components[i];

            // if it has enough points
            if (compIndexes->size() >= minPointsPerComponent) {
                // we create a new entity
                ccPointCloud* compCloud =
                        (pc ? pc->partialClone(compIndexes)
                            : ccPointCloud::From(compIndexes));
                if (compCloud) {
                    // shall we colorize it with random color?
                    if (randomColors) {
                        ecvColor::Rgb col = ecvColor::Generator::Random();
                        compCloud->setRGBColor(col);
                        compCloud->showColors(true);
                        compCloud->showSF(false);
                    }

                    //'shift on load' information
                    if (pc) {
                        compCloud->setGlobalShift(pc->getGlobalShift());
                        compCloud->setGlobalScale(pc->getGlobalScale());
                    }
                    compCloud->setVisible(true);
                    compCloud->setName(
                            QString("CC#%1").arg(ccGroup->getChildrenNumber()));

                    // we add new CC to group
                    ccGroup->addChild(compCloud);

                    if (selectComponents && m_ccRoot)
                        m_ccRoot->selectEntity(compCloud, true);
                } else {
                    ecvConsole::Warning(
                            tr("[createComponentsClouds] Failed to create "
                               "component #%1! (not enough memory)")
                                    .arg(ccGroup->getChildrenNumber() + 1));
                }
            }

            delete compIndexes;
            compIndexes = nullptr;
        }

        components.clear();

        if (ccGroup->getChildrenNumber() == 0) {
            ecvConsole::Error(
                    "No component was created! Check the minimum size...");
            delete ccGroup;
            ccGroup = nullptr;
        } else {
            addToDB(ccGroup);
            ecvConsole::Print(tr("[createComponentsClouds] %1 component(s) "
                                 "were created from cloud '%2'")
                                      .arg(ccGroup->getChildrenNumber())
                                      .arg(cloud->getName()));
        }

        // auto-hide original cloud
        if (ccGroup) {
            cloud->setEnabled(false);
            ecvConsole::Warning(
                    tr("[createComponentsClouds] Original cloud has been "
                       "automatically hidden"));
        }
    }
}

void MainWindow::doActionLabelConnectedComponents() {
    // keep only the point clouds!
    std::vector<ccGenericPointCloud*> clouds;
    {
        for (ccHObject* entity : getSelectedEntities()) {
            if (entity->isKindOf(CV_TYPES::POINT_CLOUD))
                clouds.push_back(ccHObjectCaster::ToGenericPointCloud(entity));
        }
    }

    size_t count = clouds.size();
    if (count == 0) return;

    ccLabelingDlg dlg(this);
    if (count == 1) dlg.octreeLevelSpinBox->setCloud(clouds.front());
    if (!dlg.exec()) return;

    int octreeLevel = dlg.getOctreeLevel();
    unsigned minComponentSize =
            static_cast<unsigned>(std::max(0, dlg.getMinPointsNb()));
    bool randColors = dlg.randomColors();

    ecvProgressDialog pDlg(false, this);
    pDlg.setAutoClose(false);

    // we unselect all entities as we are going to automatically select the
    // created components (otherwise the user won't perceive the change!)
    if (m_ccRoot) {
        m_ccRoot->unselectAllEntities();
    }

    for (ccGenericPointCloud* cloud : clouds) {
        if (cloud && cloud->isA(CV_TYPES::POINT_CLOUD)) {
            ccPointCloud* pc = static_cast<ccPointCloud*>(cloud);

            ccOctree::Shared theOctree = cloud->getOctree();
            if (!theOctree) {
                ecvProgressDialog pOctreeDlg(true, this);
                theOctree = cloud->computeOctree(&pOctreeDlg);
                if (!theOctree) {
                    ecvConsole::Error(
                            tr("Couldn't compute octree for cloud '%s'!")
                                    .arg(cloud->getName()));
                    break;
                }
            }

            // we create/activate CCs label scalar field
            int sfIdx = pc->getScalarFieldIndexByName(
                    CC_CONNECTED_COMPONENTS_DEFAULT_LABEL_NAME);
            if (sfIdx < 0) {
                sfIdx = pc->addScalarField(
                        CC_CONNECTED_COMPONENTS_DEFAULT_LABEL_NAME);
            }
            if (sfIdx < 0) {
                ecvConsole::Error(
                        tr("Couldn't allocate a new scalar field for computing "
                           "ECV labels! Try to free some memory ..."));
                break;
            }
            pc->setCurrentScalarField(sfIdx);

            // we try to label all CCs
            cloudViewer::ReferenceCloudContainer components;
            int componentCount = cloudViewer::AutoSegmentationTools::
                    labelConnectedComponents(
                            cloud, static_cast<unsigned char>(octreeLevel),
                            false, &pDlg, theOctree.data());

            if (componentCount >= 0) {
                // if successful, we extract each ECV (stored in "components")

                // safety test
                int realComponentCount = 0;
                {
                    for (size_t i = 0; i < components.size(); ++i) {
                        if (components[i]->size() >= minComponentSize) {
                            ++realComponentCount;
                        }
                    }
                }

                if (realComponentCount > 500) {
                    // too many components
                    if (QMessageBox::warning(
                                this, tr("Many components"),
                                tr("Do you really expect up to %1 "
                                   "components?\n(this may take a lot of time "
                                   "to process and display)")
                                        .arg(realComponentCount),
                                QMessageBox::Yes,
                                QMessageBox::No) == QMessageBox::No) {
                        // cancel
                        pc->deleteScalarField(sfIdx);
                        if (pc->getNumberOfScalarFields() != 0) {
                            pc->setCurrentDisplayedScalarField(
                                    static_cast<int>(
                                            pc->getNumberOfScalarFields()) -
                                    1);
                        } else {
                            pc->showSF(false);
                        }
                        continue;
                    }
                }

                pc->getCurrentInScalarField()->computeMinAndMax();
                if (!cloudViewer::AutoSegmentationTools::
                            extractConnectedComponents(cloud, components)) {
                    ecvConsole::Warning(tr("[doActionLabelConnectedComponents] "
                                           "Something went wrong while "
                                           "extracting CCs from cloud %1...")
                                                .arg(cloud->getName()));
                }
            } else {
                ecvConsole::Warning(
                        tr("[doActionLabelConnectedComponents] Something went "
                           "wrong while extracting CCs from cloud %1...")
                                .arg(cloud->getName()));
            }

            // we delete the CCs label scalar field (we don't need it anymore)
            pc->deleteScalarField(sfIdx);
            sfIdx = -1;

            // we create "real" point clouds for all CCs
            if (!components.empty()) {
                createComponentsClouds(cloud, components, minComponentSize,
                                       randColors, true);
            }
        }
    }

    updateUI();
}

void MainWindow::doActionKMeans()  // TODO
{
    ecvConsole::Error(tr("Not yet implemented! Sorry ..."));
}

void MainWindow::doActionFrontPropagation()  // TODO
{
    ecvConsole::Error(tr("Not yet implemented! Sorry ..."));
}

void MainWindow::doActionCloudCloudDist() {
    if (getSelectedEntities().size() != 2) {
        ecvConsole::Error("Select 2 point clouds!");
        return;
    }

    if (!m_selectedEntities[0]->isKindOf(CV_TYPES::POINT_CLOUD) ||
        !m_selectedEntities[1]->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ecvConsole::Error(tr("Select 2 point clouds!"));
        return;
    }

    ccOrderChoiceDlg dlg(m_selectedEntities[0], tr("Compared"),
                         m_selectedEntities[1], tr("Reference"), this);
    if (!dlg.exec()) return;

    ccGenericPointCloud* compCloud =
            ccHObjectCaster::ToGenericPointCloud(dlg.getFirstEntity());
    ccGenericPointCloud* refCloud =
            ccHObjectCaster::ToGenericPointCloud(dlg.getSecondEntity());

    // assert(!m_compDlg);
    if (m_compDlg) delete m_compDlg;
    m_compDlg = new ccComparisonDlg(compCloud, refCloud,
                                    ccComparisonDlg::CLOUDCLOUD_DIST, this);
    connect(m_compDlg, &QDialog::finished, this,
            &MainWindow::deactivateComparisonMode);
    m_compDlg->show();
    // cDlg.setModal(false);
    // cDlg.exec();
    freezeUI(true);
}

void MainWindow::doActionCloudMeshDist() {
    if (getSelectedEntities().size() != 2) {
        ecvConsole::Error(tr("Select 2 entities!"));
        return;
    }

    bool isMesh[2] = {false, false};
    unsigned meshNum = 0;
    unsigned cloudNum = 0;
    for (unsigned i = 0; i < 2; ++i) {
        if (m_selectedEntities[i]->isKindOf(CV_TYPES::MESH)) {
            ++meshNum;
            isMesh[i] = true;
        } else if (m_selectedEntities[i]->isKindOf(CV_TYPES::POINT_CLOUD)) {
            ++cloudNum;
        }
    }

    if (meshNum == 0) {
        ecvConsole::Error(tr("Select at least one mesh!"));
        return;
    } else if (meshNum + cloudNum < 2) {
        ecvConsole::Error(tr("Select one mesh and one cloud or two meshes!"));
        return;
    }

    ccHObject* compEnt = nullptr;
    ccGenericMesh* refMesh = nullptr;

    if (meshNum == 1) {
        compEnt = m_selectedEntities[isMesh[0] ? 1 : 0];
        refMesh = ccHObjectCaster::ToGenericMesh(
                m_selectedEntities[isMesh[0] ? 0 : 1]);
    } else {
        ccOrderChoiceDlg dlg(m_selectedEntities[0], tr("Compared"),
                             m_selectedEntities[1], tr("Reference"), this);
        if (!dlg.exec()) return;

        compEnt = dlg.getFirstEntity();
        refMesh = ccHObjectCaster::ToGenericMesh(dlg.getSecondEntity());
    }

    // assert(!m_compDlg);
    if (m_compDlg) delete m_compDlg;
    m_compDlg = new ccComparisonDlg(compEnt, refMesh,
                                    ccComparisonDlg::CLOUDMESH_DIST, this);
    connect(m_compDlg, &QDialog::finished, this,
            &MainWindow::deactivateComparisonMode);
    m_compDlg->show();

    freezeUI(true);
}

void MainWindow::doActionCloudPrimitiveDist() {
    bool foundPrimitive = false;
    ccHObject::Container clouds;
    ccHObject* refEntity = nullptr;
    CV_CLASS_ENUM entityType = CV_TYPES::OBJECT;
    const char* errString =
            "[Compute Primitive Distances] Cloud to %s failed, error code = "
            "%i!";

    for (unsigned i = 0; i < getSelectedEntities().size(); ++i) {
        if (m_selectedEntities[i]->isKindOf(CV_TYPES::PRIMITIVE) ||
            m_selectedEntities[i]->isA(CV_TYPES::POLY_LINE)) {
            if (m_selectedEntities[i]->isA(CV_TYPES::PLANE) ||
                m_selectedEntities[i]->isA(CV_TYPES::SPHERE) ||
                m_selectedEntities[i]->isA(CV_TYPES::CYLINDER) ||
                m_selectedEntities[i]->isA(CV_TYPES::CONE) ||
                m_selectedEntities[i]->isA(CV_TYPES::BOX) ||
                m_selectedEntities[i]->isA(CV_TYPES::DISC) ||
                m_selectedEntities[i]->isA(CV_TYPES::POLY_LINE)) {
                if (foundPrimitive) {
                    ecvConsole::Error(
                            "[Compute Primitive Distances] Select only a "
                            "single Plane/Box/Sphere/Cylinder/Cone/Polyline "
                            "Primitive");
                    return;
                }
                foundPrimitive = true;
                refEntity = m_selectedEntities[i];
                entityType = refEntity->getClassID();
            }
        } else if (m_selectedEntities[i]->isKindOf(CV_TYPES::POINT_CLOUD)) {
            clouds.push_back(m_selectedEntities[i]);
        }
    }

    if (!foundPrimitive) {
        ecvConsole::Error(
                "[Compute Primitive Distances] Select at least one "
                "Plane/Box/Sphere/Cylinder/Cone/Disc/Polyline Primitive!");
        return;
    }
    if (clouds.size() <= 0) {
        ecvConsole::Error(
                "[Compute Primitive Distances] Select at least one cloud!");
        return;
    }

    ecvPrimitiveDistanceDlg pDD{this};
    if (refEntity->isA(CV_TYPES::PLANE)) {
        pDD.treatPlanesAsBoundedCheckBox->setUpdatesEnabled(true);
    }
    bool execute = true;
    if (!refEntity->isA(CV_TYPES::POLY_LINE)) {
        execute = pDD.exec();
    }
    if (execute) {
        bool signedDist = pDD.signedDistances();
        bool flippedNormals = signedDist && pDD.flipNormals();
        bool treatPlanesAsBounded = pDD.treatPlanesAsBounded();
        for (auto& cloud : clouds) {
            ccPointCloud* compEnt = ccHObjectCaster::ToPointCloud(cloud);
            int sfIdx = compEnt->getScalarFieldIndexByName(
                    CC_TEMP_DISTANCES_DEFAULT_SF_NAME);
            if (sfIdx < 0) {
                // we need to create a new scalar field
                sfIdx = compEnt->addScalarField(
                        CC_TEMP_DISTANCES_DEFAULT_SF_NAME);
                if (sfIdx < 0) {
                    CVLog::Error(
                            QString("[Compute Primitive Distances] [Cloud: %1] "
                                    "Couldn't allocate a new scalar field for "
                                    "computing distances! Try to free some "
                                    "memory ...")
                                    .arg(compEnt->getName()));
                    continue;
                }
            }
            compEnt->setCurrentScalarField(sfIdx);
            compEnt->enableScalarField();
            compEnt->forEach(
                    cloudViewer::ScalarFieldTools::SetScalarValueToNaN);
            int returnCode;
            switch (entityType) {
                case CV_TYPES::SPHERE: {
                    if (!(returnCode = cloudViewer::DistanceComputationTools::
                                  computeCloud2SphereEquation(
                                          compEnt,
                                          refEntity->getOwnBB().getCenter(),
                                          static_cast<ccSphere*>(refEntity)
                                                  ->getRadius(),
                                          signedDist)))
                        ecvConsole::Error(errString, "Sphere", returnCode);
                    break;
                }
                case CV_TYPES::PLANE: {
                    ccPlane* plane = static_cast<ccPlane*>(refEntity);
                    if (treatPlanesAsBounded) {
                        cloudViewer::SquareMatrix rotationTransform(
                                plane->getTransformation().data(), true);
                        if (!(returnCode =
                                      cloudViewer::DistanceComputationTools::
                                              computeCloud2RectangleEquation(
                                                      compEnt,
                                                      plane->getXWidth(),
                                                      plane->getYWidth(),
                                                      rotationTransform,
                                                      plane->getCenter(),
                                                      signedDist)))
                            ecvConsole::Error(errString, "Bounded Plane",
                                              returnCode);
                    } else {
                        if (!(returnCode =
                                      cloudViewer::DistanceComputationTools::
                                              computeCloud2PlaneEquation(
                                                      compEnt,
                                                      static_cast<ccPlane*>(
                                                              refEntity)
                                                              ->getEquation(),
                                                      signedDist)))
                            ecvConsole::Error(errString, "Infinite Plane",
                                              returnCode);
                    }
                    break;
                }
                case CV_TYPES::CYLINDER: {
                    if (!(returnCode = cloudViewer::DistanceComputationTools::
                                  computeCloud2CylinderEquation(
                                          compEnt,
                                          static_cast<ccCylinder*>(refEntity)
                                                  ->getBottomCenter(),
                                          static_cast<ccCylinder*>(refEntity)
                                                  ->getTopCenter(),
                                          static_cast<ccCylinder*>(refEntity)
                                                  ->getBottomRadius(),
                                          signedDist)))
                        ecvConsole::Error(errString, "Cylinder", returnCode);
                    break;
                }
                case CV_TYPES::CONE: {
                    if (!(returnCode = cloudViewer::DistanceComputationTools::
                                  computeCloud2ConeEquation(
                                          compEnt,
                                          static_cast<ccCone*>(refEntity)
                                                  ->getLargeCenter(),
                                          static_cast<ccCone*>(refEntity)
                                                  ->getSmallCenter(),
                                          static_cast<ccCone*>(refEntity)
                                                  ->getLargeRadius(),
                                          static_cast<ccCone*>(refEntity)
                                                  ->getSmallRadius(),
                                          signedDist)))
                        ecvConsole::Error(errString, "Cone", returnCode);
                    break;
                }
                case CV_TYPES::BOX: {
                    const ccGLMatrix& glTransform =
                            refEntity->getGLTransformationHistory();
                    cloudViewer::SquareMatrix rotationTransform(
                            glTransform.data(), true);
                    CCVector3 boxCenter = glTransform.getColumnAsVec3D(3);
                    if (!(returnCode = cloudViewer::DistanceComputationTools::
                                  computeCloud2BoxEquation(
                                          compEnt,
                                          static_cast<ccBox*>(refEntity)
                                                  ->getDimensions(),
                                          rotationTransform, boxCenter,
                                          signedDist)))
                        ecvConsole::Error(errString, "Box", returnCode);
                    break;
                }
                case CV_TYPES::DISC: {
                    ccDisc* disc = static_cast<ccDisc*>(refEntity);
                    cloudViewer::SquareMatrix rotationTransform(
                            disc->getTransformation().data(), true);
                    if (!(returnCode = cloudViewer::DistanceComputationTools::
                                  computeCloud2DiscEquation(
                                          compEnt,
                                          refEntity->getOwnBB().getCenter(),
                                          static_cast<ccDisc*>(refEntity)
                                                  ->getRadius(),
                                          rotationTransform, signedDist)))
                        ecvConsole::Error(errString, "Disc", returnCode);
                    break;
                }
                case CV_TYPES::POLY_LINE: {
                    signedDist = false;
                    flippedNormals = false;
                    ccPolyline* line = static_cast<ccPolyline*>(refEntity);
                    returnCode = cloudViewer::DistanceComputationTools::
                            computeCloud2PolylineEquation(compEnt, line);
                    if (!returnCode) {
                        ecvConsole::Error(errString, "Polyline", returnCode);
                    }
                    break;
                }
                default: {
                    ecvConsole::Error(
                            "[Compute Primitive Distances] Unsupported "
                            "primitive type");  // Shouldn't ever reach here...
                    break;
                }
            }
            QString sfName;
            sfName.clear();
            sfName = QString(
                    signedDist
                            ? CC_CLOUD2PRIMITIVE_SIGNED_DISTANCES_DEFAULT_SF_NAME
                            : CC_CLOUD2PRIMITIVE_DISTANCES_DEFAULT_SF_NAME);
            if (flippedNormals) {
                compEnt->forEach(
                        cloudViewer::ScalarFieldTools::SetScalarValueInverted);
                sfName += QString("[-]");
            }

            int _sfIdx = compEnt->getScalarFieldIndexByName(qPrintable(sfName));
            if (_sfIdx >= 0) {
                compEnt->deleteScalarField(_sfIdx);
                // we update sfIdx because indexes are all messed up after
                // deletion
                sfIdx = compEnt->getScalarFieldIndexByName(
                        CC_TEMP_DISTANCES_DEFAULT_SF_NAME);
            }
            compEnt->renameScalarField(sfIdx, qPrintable(sfName));

            ccScalarField* sf =
                    static_cast<ccScalarField*>(compEnt->getScalarField(sfIdx));
            if (sf) {
                ScalarType mean;
                ScalarType variance;
                sf->computeMinAndMax();
                sf->computeMeanAndVariance(mean, &variance);
                CVLog::Print(
                        "[Compute Primitive Distances] [Primitive: %s] [Cloud: "
                        "%s] [%s] Mean distance = %f / std deviation = %f",
                        qPrintable(refEntity->getName()),
                        qPrintable(compEnt->getName()), qPrintable(sfName),
                        mean, sqrt(variance));
            }
            compEnt->setCurrentDisplayedScalarField(sfIdx);
            compEnt->showSF(sfIdx >= 0);
            compEnt->setRedrawFlagRecursive(true);
        }

        refreshAll();
        MainWindow::UpdateUI();
    }
}

void MainWindow::deactivateComparisonMode(int result) {
    // DGM: a bug apperead with recent changes (from CC or QT?)
    // which prevent us from deleting the dialog right away...
    //(it seems that QT has not yet finished the dialog closing
    // when the 'finished' signal is sent).
    // if(m_compDlg)
    //	delete m_compDlg;
    // m_compDlg = 0;

    // if the comparison is a success, we select only the compared entity
    if (m_compDlg && result == QDialog::Accepted && m_ccRoot) {
        ccHObject* compEntity = m_compDlg->getComparedEntity();
        if (compEntity) {
            m_ccRoot->selectEntity(compEntity);
        }
    }

    freezeUI(false);

    updateUI();
}

void MainWindow::doActionExportPlaneInfo() {
    ccHObject::Container planes;

    const ccHObject::Container& selectedEntities = getSelectedEntities();
    if (selectedEntities.size() == 1 &&
        selectedEntities.front()->isA(CV_TYPES::HIERARCHY_OBJECT)) {
        // a group
        selectedEntities.front()->filterChildren(planes, true, CV_TYPES::PLANE,
                                                 false);
    } else {
        for (ccHObject* ent : selectedEntities) {
            if (ent->isKindOf(CV_TYPES::PLANE)) {
                // a single plane
                planes.push_back(static_cast<ccPlane*>(ent));
            }
        }
    }

    if (planes.empty()) {
        CVLog::Error(tr("No plane in selection"));
        return;
    }

    // persistent settings
    QString currentPath =
            ecvSettingManager::getValue(ecvPS::SaveFile(), ecvPS::CurrentPath(),
                                        ecvFileUtils::defaultDocPath())
                    .toString();

    QString outputFilename = QFileDialog::getSaveFileName(
            this, tr("Select output file"), currentPath, "*.csv", nullptr,
            ECVFileDialogOptions());

    if (outputFilename.isEmpty()) {
        // process cancelled by the user
        return;
    }

    QFile csvFile(outputFilename);
    if (!csvFile.open(QFile::WriteOnly | QFile::Text)) {
        ecvConsole::Error(tr(
                "Failed to open file for writing! (check file permissions)"));
        return;
    }

    // save last saving location
    ecvSettingManager::setValue(ecvPS::SaveFile(), ecvPS::CurrentPath(),
                                QFileInfo(outputFilename).absolutePath());

    // write CSV header
    QTextStream csvStream(&csvFile);
    csvStream << "Name,";
    csvStream << "Width,";
    csvStream << "Height,";
    csvStream << "Cx,";
    csvStream << "Cy,";
    csvStream << "Cz,";
    csvStream << "Nx,";
    csvStream << "Ny,";
    csvStream << "Nz,";
    csvStream << "Dip,";
    csvStream << "Dip dir,";
    csvStream << QtCompat::endl;

    QChar separator(',');

    // write one line per plane
    for (ccHObject* ent : planes) {
        ccPlane* plane = static_cast<ccPlane*>(ent);

        CCVector3 C = plane->getOwnBB().getCenter();
        CCVector3 N = plane->getNormal();
        PointCoordinateType dip_deg = 0, dipDir_deg = 0;
        ccNormalVectors::ConvertNormalToDipAndDipDir(N, dip_deg, dipDir_deg);

        csvStream << plane->getName() << separator;    // Name
        csvStream << plane->getXWidth() << separator;  // Width
        csvStream << plane->getYWidth() << separator;  // Height
        csvStream << C.x << separator;                 // Cx
        csvStream << C.y << separator;                 // Cy
        csvStream << C.z << separator;                 // Cz
        csvStream << N.x << separator;                 // Nx
        csvStream << N.y << separator;                 // Ny
        csvStream << N.z << separator;                 // Nz
        csvStream << dip_deg << separator;             // Dip
        csvStream << dipDir_deg << separator;          // Dip direction
        csvStream << QtCompat::endl;
    }

    ecvConsole::Print(tr("[I/O] File '%1' successfully saved (%2 plane(s))")
                              .arg(outputFilename)
                              .arg(planes.size()));
    csvFile.close();
}

void MainWindow::doActionExportCloudInfo() {
    // look for clouds
    ccHObject::Container clouds;

    const ccHObject::Container& selectedEntities = getSelectedEntities();
    if (selectedEntities.size() == 1 &&
        selectedEntities.front()->isA(CV_TYPES::HIERARCHY_OBJECT)) {
        // a group
        selectedEntities.front()->filterChildren(clouds, true,
                                                 CV_TYPES::POINT_CLOUD, true);
    } else {
        for (ccHObject* entity : selectedEntities) {
            ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(entity);
            if (cloud) {
                clouds.push_back(cloud);
            }
        }
    }

    if (clouds.empty()) {
        ecvConsole::Error(tr("Select at least one point cloud!"));
        return;
    }

    // persistent settings
    QString currentPath =
            ecvSettingManager::getValue(ecvPS::SaveFile(), ecvPS::CurrentPath(),
                                        ecvFileUtils::defaultDocPath())
                    .toString();

    QString outputFilename = QFileDialog::getSaveFileName(
            this, tr("Select output file"), currentPath, "*.csv", nullptr,
            ECVFileDialogOptions());
    if (outputFilename.isEmpty()) {
        // process cancelled by the user
        return;
    }

    QFile csvFile(outputFilename);
    if (!csvFile.open(QFile::WriteOnly | QFile::Text)) {
        ecvConsole::Error(tr(
                "Failed to open file for writing! (check file permissions)"));
        return;
    }

    // save last saving location
    ecvSettingManager::setValue(ecvPS::SaveFile(), ecvPS::CurrentPath(),
                                QFileInfo(outputFilename).absolutePath());

    // determine the maximum number of SFs
    unsigned maxSFCount = 0;
    for (ccHObject* entity : clouds) {
        maxSFCount = std::max<unsigned>(
                maxSFCount,
                static_cast<ccPointCloud*>(entity)->getNumberOfScalarFields());
    }

    // write CSV header
    QTextStream csvStream(&csvFile);
    csvStream << "Name,";
    csvStream << "Points,";
    csvStream << "meanX,";
    csvStream << "meanY,";
    csvStream << "meanZ,";
    {
        for (unsigned i = 0; i < maxSFCount; ++i) {
            QString sfIndex = QString("SF#%1").arg(i + 1);
            csvStream << sfIndex << " name,";
            csvStream << sfIndex << " valid values,";
            csvStream << sfIndex << " mean,";
            csvStream << sfIndex << " std.dev.,";
            csvStream << sfIndex << " sum,";
        }
    }
    csvStream << QtCompat::endl;

    // write one line per cloud
    {
        for (ccHObject* entity : clouds) {
            ccPointCloud* cloud = static_cast<ccPointCloud*>(entity);

            CCVector3 G = *cloudViewer::Neighbourhood(cloud).getGravityCenter();
            csvStream << cloud->getName() << "," /*"Name;"*/;
            csvStream << cloud->size() << "," /*"Points;"*/;
            csvStream << G.x << "," /*"meanX;"*/;
            csvStream << G.y << "," /*"meanY;"*/;
            csvStream << G.z << "," /*"meanZ;"*/;
            for (unsigned j = 0; j < cloud->getNumberOfScalarFields(); ++j) {
                cloudViewer::ScalarField* sf = cloud->getScalarField(j);
                csvStream << sf->getName() << "," /*"SF name;"*/;

                unsigned validCount = 0;
                double sfSum = 0.0;
                double sfSum2 = 0.0;
                for (unsigned k = 0; k < sf->currentSize(); ++k) {
                    const ScalarType& val = sf->getValue(k);
                    if (cloudViewer::ScalarField::ValidValue(val)) {
                        ++validCount;
                        sfSum += val;
                        sfSum2 += val * val;
                    }
                }
                csvStream << validCount << "," /*"SF valid values;"*/;
                double mean = sfSum / validCount;
                csvStream << mean << "," /*"SF mean;"*/;
                csvStream << sqrt(std::abs(sfSum2 / validCount - mean * mean))
                          << "," /*"SF std.dev.;"*/;
                csvStream << sfSum << "," /*"SF sum;"*/;
            }
            csvStream << QtCompat::endl;
        }
    }

    ecvConsole::Print(tr("[I/O] File '%1' successfully saved (%2 cloud(s))")
                              .arg(outputFilename)
                              .arg(clouds.size()));
    csvFile.close();
}

void MainWindow::doActionComputeCPS() {
    if (m_selectedEntities.size() != 2) {
        ecvConsole::Error(tr("Select 2 point clouds!"));
        return;
    }

    if (!m_selectedEntities[0]->isKindOf(CV_TYPES::POINT_CLOUD) ||
        !m_selectedEntities[1]->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ecvConsole::Error(tr("Select 2 point clouds!"));
        return;
    }

    ccOrderChoiceDlg dlg(m_selectedEntities[0], tr("Compared"),
                         m_selectedEntities[1], tr("Reference"), this);
    if (!dlg.exec()) return;

    ccGenericPointCloud* compCloud =
            ccHObjectCaster::ToGenericPointCloud(dlg.getFirstEntity());
    ccGenericPointCloud* srcCloud =
            ccHObjectCaster::ToGenericPointCloud(dlg.getSecondEntity());

    if (!compCloud->isA(CV_TYPES::POINT_CLOUD))  // TODO
    {
        ecvConsole::Error(tr("Compared cloud must be a real point cloud!"));
        return;
    }
    ccPointCloud* cmpPC = static_cast<ccPointCloud*>(compCloud);

    static const char DEFAULT_CPS_TEMP_SF_NAME[] = "CPS temporary";
    int sfIdx = cmpPC->getScalarFieldIndexByName(DEFAULT_CPS_TEMP_SF_NAME);
    if (sfIdx < 0) sfIdx = cmpPC->addScalarField(DEFAULT_CPS_TEMP_SF_NAME);
    if (sfIdx < 0) {
        ecvConsole::Error(
                tr("Couldn't allocate a new scalar field for computing "
                   "distances! Try to free some memory ..."));
        return;
    }
    cmpPC->setCurrentScalarField(sfIdx);
    if (!cmpPC->enableScalarField()) {
        ecvConsole::Error(tr("Not enough memory!"));
        return;
    }
    // cmpPC->forEach(cloudViewer::ScalarFieldTools::SetScalarValueToNaN); //now
    // done by default by computeCloud2CloudDistances

    cloudViewer::ReferenceCloud CPSet(srcCloud);
    ecvProgressDialog pDlg(true, this);
    cloudViewer::DistanceComputationTools::Cloud2CloudDistancesComputationParams
            params;
    params.CPSet = &CPSet;
    int result =
            cloudViewer::DistanceComputationTools::computeCloud2CloudDistances(
                    compCloud, srcCloud, params, &pDlg);
    cmpPC->deleteScalarField(sfIdx);

    if (result >= 0) {
        ccPointCloud* newCloud = nullptr;
        // if the source cloud is a "true" cloud, the extracted CPS
        // will also get its attributes
        newCloud = srcCloud->isA(CV_TYPES::POINT_CLOUD)
                           ? static_cast<ccPointCloud*>(srcCloud)->partialClone(
                                     &CPSet)
                           : ccPointCloud::From(&CPSet, srcCloud);

        newCloud->setName(
                QString("[%1]->CPSet(%2)")
                        .arg(srcCloud->getName(), compCloud->getName()));
        addToDB(newCloud);

        // we hide the source cloud (for a clearer display)
        srcCloud->setEnabled(false);
    }
}

void MainWindow::doActionFitSphere() {
    double outliersRatio = 0.5;
    double confidence = 0.99;

    ecvProgressDialog pDlg(true, this);
    pDlg.setAutoClose(false);

    for (ccHObject* entity : getSelectedEntities()) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(entity);
        if (!cloud) continue;

        CCVector3 center;
        PointCoordinateType radius;
        double rms;
        if (cloudViewer::GeometricalAnalysisTools::DetectSphereRobust(
                    cloud, outliersRatio, center, radius, rms, &pDlg,
                    confidence) !=
            cloudViewer::GeometricalAnalysisTools::NoError) {
            CVLog::Warning(
                    tr("[Fit sphere] Failed to fit a sphere on cloud '%1'")
                            .arg(cloud->getName()));
            continue;
        }

        CVLog::Print(tr("[Fit sphere] Cloud '%1': center (%2,%3,%4) - radius = "
                        "%5 [RMS = %6]")
                             .arg(cloud->getName())
                             .arg(center.x)
                             .arg(center.y)
                             .arg(center.z)
                             .arg(radius)
                             .arg(rms));

        ccGLMatrix trans;
        trans.setTranslation(center);
        ccSphere* sphere =
                new ccSphere(radius, &trans,
                             tr("Sphere r=%1 [rms %2]").arg(radius).arg(rms));
        cloud->addChild(sphere);
        sphere->setOpacity(GLOBAL_OPACITY);
        addToDB(sphere, false, false, false);
    }
}

void MainWindow::doActionFitCircle() {
    ecvProgressDialog pDlg(true, this);
    pDlg.setAutoClose(false);

    ecvDisplayTools::SetRedrawRecursive(false);
    for (ccHObject* entity : getSelectedEntities()) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(entity);
        if (!cloud) continue;

        CCVector3 center;
        CCVector3 normal;
        PointCoordinateType radius = 0;
        double rms = std::numeric_limits<double>::quiet_NaN();
        if (cloudViewer::GeometricalAnalysisTools::DetectCircle(
                    cloud, center, normal, radius, rms, &pDlg) !=
            cloudViewer::GeometricalAnalysisTools::NoError) {
            CVLog::Warning(
                    tr("[Fit circle] Failed to fit a circle on cloud '%1'")
                            .arg(cloud->getName()));
            continue;
        }

        CVLog::Print(tr("[Fit circle] Cloud '%1': center (%2,%3,%4) - radius = "
                        "%5 [RMS = %6]")
                             .arg(cloud->getName())
                             .arg(center.x)
                             .arg(center.y)
                             .arg(center.z)
                             .arg(radius)
                             .arg(rms));

        CVLog::Print(tr("[Fit circle] Normal (%1,%2,%3)")
                             .arg(normal.x)
                             .arg(normal.y)
                             .arg(normal.z));

        // create the circle representation as a polyline
        ccCircle* circle = new ccCircle(radius, 128);
        if (circle) {
            circle->setName(QObject::tr("Circle r=%1").arg(radius));
            cloud->addChild(circle);
            circle->copyGlobalShiftAndScale(*cloud);
            circle->setMetaData("RMS", rms);

            ccGLMatrix trans =
                    ccGLMatrix::FromToRotation(CCVector3(0, 0, 1), normal);
            trans.setTranslation(center);
            circle->applyGLTransformation_recursive(&trans);
            circle->setRedrawFlagRecursive(true);

            addToDB(circle, false, false, false);
        }
    }

    refreshAll();
}

void MainWindow::doActionFitPlane() { doComputePlaneOrientation(false); }

void MainWindow::doComputePlaneOrientation(bool fitFacet) {
    if (!haveSelection()) return;

    double maxEdgeLength = 0.0;
    if (fitFacet) {
        bool ok = true;
        static double s_polygonMaxEdgeLength = 0.0;
        maxEdgeLength = QInputDialog::getDouble(
                this, tr("Fit facet"), tr("Max edge length (0 = no limit)"),
                s_polygonMaxEdgeLength, 0, 1.0e9, 8, &ok);
        if (!ok) return;
        s_polygonMaxEdgeLength = maxEdgeLength;
    }

    ccHObject::Container selectedEntities =
            getSelectedEntities();  // warning, getSelectedEntites may change
                                    // during this loop!
    bool firstEntity = true;

    for (ccHObject* entity : selectedEntities) {
        ccShiftedObject* shifted = nullptr;
        cloudViewer::GenericIndexedCloudPersist* cloud = nullptr;

        if (entity->isKindOf(CV_TYPES::POLY_LINE)) {
            ccPolyline* poly = ccHObjectCaster::ToPolyline(entity);
            cloud = static_cast<cloudViewer::GenericIndexedCloudPersist*>(poly);
            shifted = poly;
        } else {
            ccGenericPointCloud* gencloud =
                    ccHObjectCaster::ToGenericPointCloud(entity);
            if (gencloud) {
                cloud = static_cast<cloudViewer::GenericIndexedCloudPersist*>(
                        gencloud);
                shifted = gencloud;
            }
        }

        if (cloud) {
            double rms = 0.0;
            CCVector3 C, N;

            ccHObject* plane = nullptr;
            if (fitFacet) {
                ccFacet* facet = ccFacet::Create(
                        cloud, static_cast<PointCoordinateType>(maxEdgeLength));
                if (facet) {
                    facet->getPolygon()->setOpacity(GLOBAL_OPACITY);
                    facet->getPolygon()->setTempColor(ecvColor::darkGrey);
                    facet->getContour()->setColor(ecvColor::green);
                    facet->getContour()->showColors(true);

                    plane = static_cast<ccHObject*>(facet);
                    N = facet->getNormal();
                    C = facet->getCenter();
                    rms = facet->getRMS();

                    // manually copy shift & scale info!
                    if (shifted) {
                        ccPolyline* contour = facet->getContour();
                        if (contour) {
                            contour->setGlobalScale(shifted->getGlobalScale());
                            contour->setGlobalShift(shifted->getGlobalShift());
                        }
                    }
                }
            } else {
                ccPlane* pPlane = ccPlane::Fit(cloud, &rms);
                if (pPlane) {
                    plane = static_cast<ccHObject*>(pPlane);
                    N = pPlane->getNormal();
                    C = *cloudViewer::Neighbourhood(cloud).getGravityCenter();
                    pPlane->enableStippling(true);
                }
            }

            // as all information appears in Console...
            forceConsoleDisplay();

            if (plane) {
                ecvConsole::Print(
                        tr("[Orientation] Entity '%1'").arg(entity->getName()));
                ecvConsole::Print(tr("\t- plane fitting RMS: %1").arg(rms));

                // We always consider the normal with a positive 'Z' by default!
                if (N.z < 0.0) N *= -1.0;
                ecvConsole::Print(tr("\t- normal: (%1,%2,%3)")
                                          .arg(N.x)
                                          .arg(N.y)
                                          .arg(N.z));

                // we compute strike & dip by the way
                PointCoordinateType dip = 0.0f;
                PointCoordinateType dipDir = 0.0f;
                ccNormalVectors::ConvertNormalToDipAndDipDir(N, dip, dipDir);
                QString dipAndDipDirStr =
                        ccNormalVectors::ConvertDipAndDipDirToString(dip,
                                                                     dipDir);
                ecvConsole::Print(QString("\t- %1").arg(dipAndDipDirStr));

                // hack: output the transformation matrix that would make this
                // normal points towards +Z
                ccGLMatrix makeZPosMatrix =
                        ccGLMatrix::FromToRotation(N, CCVector3(0, 0, PC_ONE));
                CCVector3 Gt = C;
                makeZPosMatrix.applyRotation(Gt);
                makeZPosMatrix.setTranslation(C - Gt);
                ecvConsole::Print(
                        tr("[Orientation] A matrix that would make this plane "
                           "horizontal (normal towards Z+) is:"));
                ecvConsole::Print(
                        makeZPosMatrix.toString(12, ' '));  // full precision
                ecvConsole::Print(
                        tr("[Orientation] You can copy this matrix values "
                           "(CTRL+C) and paste them in the 'Apply "
                           "transformation tool' dialog"));

                plane->setName(dipAndDipDirStr);
                plane->applyGLTransformation_recursive();  // not yet in DB
                plane->setVisible(true);
                plane->setSelectionBehavior(ccHObject::SELECTION_FIT_BBOX);

                entity->addChild(plane);
                plane->setTempColor(ecvColor::darkGrey);
                plane->setOpacity(GLOBAL_OPACITY);
                addToDB(plane);

                if (firstEntity) {
                    m_ccRoot->unselectAllEntities();
                    m_ccRoot->selectEntity(plane);
                }
            } else {
                ecvConsole::Warning(
                        tr("Failed to fit a plane/facet on entity '%1'")
                                .arg(entity->getName()));
            }
        }
    }

    updateUI();
}

void MainWindow::doActionFitFacet() { doComputePlaneOrientation(true); }

void MainWindow::doActionFitQuadric() {
    bool errors = false;

    // for all selected entities
    for (ccHObject* entity : getSelectedEntities()) {
        // look for clouds
        if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
            ccGenericPointCloud* cloud =
                    ccHObjectCaster::ToGenericPointCloud(entity);

            double rms = 0.0;
            ccQuadric* quadric = ccQuadric::Fit(cloud, &rms);
            if (quadric) {
                cloud->addChild(quadric);
                quadric->setName(tr("Quadric (%1)").arg(cloud->getName()));
                quadric->setOpacity(GLOBAL_OPACITY);
                addToDB(quadric);

                ecvConsole::Print(
                        tr("[doActionFitQuadric] Quadric local coordinate "
                           "system:"));
                ecvConsole::Print(quadric->getTransformation().toString(
                        12, ' '));  // full precision
                ecvConsole::Print(tr("[doActionFitQuadric] Quadric equation "
                                     "(in local coordinate system): ") +
                                  quadric->getEquationString());
                ecvConsole::Print(tr("[doActionFitQuadric] RMS: %1").arg(rms));

#if 0
				//test: project the input cloud on the quadric
				if (cloud->isA(CV_TYPES::POINT_CLOUD))
				{
					ccPointCloud* newCloud = static_cast<ccPointCloud*>(cloud)->cloneThis();
					if (newCloud)
					{
						const PointCoordinateType* eq = quadric->getEquationCoefs();
						const Tuple3ub& dims = quadric->getEquationDims();

						const unsigned char dX = dims.x;
						const unsigned char dY = dims.y;
						const unsigned char dZ = dims.z;

						const ccGLMatrix& trans = quadric->getTransformation();
						ccGLMatrix invTrans = trans.inverse();
						for (unsigned i = 0; i < newCloud->size(); ++i)
						{
							CCVector3* P = const_cast<CCVector3*>(newCloud->getPoint(i));
							CCVector3 Q = invTrans * (*P);
							Q.u[dZ] = eq[0] + eq[1] * Q.u[dX] + eq[2] * Q.u[dY] + eq[3] * Q.u[dX] * Q.u[dX] + eq[4] * Q.u[dX] * Q.u[dY] + eq[5] * Q.u[dY] * Q.u[dY];
							*P = trans * Q;
						}
						newCloud->invalidateBoundingBox();
						newCloud->setName(newCloud->getName() + ".projection_on_quadric");
						addToDB(newCloud);
					}
				}
#endif
            } else {
                ecvConsole::Warning(
                        tr("Failed to compute quadric on cloud '%1'")
                                .arg(cloud->getName()));
                errors = true;
            }
        }
    }

    if (errors) {
        ecvConsole::Error(tr("Error(s) occurred: see console"));
    }
}

void MainWindow::doActionUnroll() {
    // there should be only one point cloud with sensor in current selection!
    if (!haveOneSelection()) {
        ecvConsole::Error(tr("Select one and only one entity!"));
        return;
    }

    // if selected entity is a mesh, the method will be applied to its vertices
    bool lockedVertices;
    ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(
            m_selectedEntities[0], &lockedVertices);
    if (lockedVertices) {
        ecvUtils::DisplayLockedVerticesWarning(m_selectedEntities[0]->getName(),
                                               true);
        return;
    }

    // for "real" point clouds only
    if (!cloud || !cloud->isA(CV_TYPES::POINT_CLOUD)) {
        ecvConsole::Error(
                tr("Method can't be applied on locked vertices or virtual "
                   "point clouds!"));
        return;
    }
    ccPointCloud* pc = static_cast<ccPointCloud*>(cloud);

    ccUnrollDlg unrollDlg(this);
    unrollDlg.fromPersistentSettings();
    if (!unrollDlg.exec()) return;
    unrollDlg.toPersistentSettings();

    ccPointCloud::UnrollMode mode = unrollDlg.getType();
    PointCoordinateType radius =
            static_cast<PointCoordinateType>(unrollDlg.getRadius());
    unsigned char dim =
            static_cast<unsigned char>(unrollDlg.getAxisDimension());
    bool exportDeviationSF = unrollDlg.exportDeviationSF();
    CCVector3 center = unrollDlg.getAxisPosition();

    // let's rock unroll ;)
    ecvProgressDialog pDlg(true, this);

    double startAngle_deg = 0.0;
    double stopAngle_deg = 360.0;
    unrollDlg.getAngleRange(startAngle_deg, stopAngle_deg);
    if (startAngle_deg >= stopAngle_deg) {
        QMessageBox::critical(this, "Error", "Invalid angular range");
        return;
    }

    ccPointCloud* output = nullptr;
    switch (mode) {
        case ccPointCloud::CYLINDER: {
            ccPointCloud::UnrollCylinderParams params;
            params.radius = radius;
            params.axisDim = dim;
            if (unrollDlg.isAxisPositionAuto()) {
                center = pc->getOwnBB().getCenter();
            }
            params.center = center;
            output = pc->unroll(mode, &params, exportDeviationSF,
                                startAngle_deg, stopAngle_deg, &pDlg);
        } break;

        case ccPointCloud::CONE:
        case ccPointCloud::STRAIGHTENED_CONE:
        case ccPointCloud::STRAIGHTENED_CONE2: {
            ccPointCloud::UnrollConeParams params;
            params.radius = (mode == ccPointCloud::CONE ? 0 : radius);
            params.apex = center;
            params.coneAngle_deg = unrollDlg.getConeHalfAngle();
            params.axisDim = dim;
            output = pc->unroll(mode, &params, exportDeviationSF,
                                startAngle_deg, stopAngle_deg, &pDlg);
        } break;

        default:
            assert(false);
            break;
    }

    if (output) {
        if (m_selectedEntities[0]->isA(CV_TYPES::MESH)) {
            ccMesh* mesh = ccHObjectCaster::ToMesh(m_selectedEntities[0]);
            mesh->setEnabled(false);
            ecvConsole::Warning(
                    "[Unroll] Original mesh has been automatically hidden");
            ccMesh* outputMesh = mesh->cloneMesh(output);
            outputMesh->addChild(output);
            outputMesh->setEnabled(true);
            outputMesh->setVisible(true);
            addToDB(outputMesh, true, true, false, true);
        } else {
            pc->setEnabled(false);
            ecvConsole::Warning(
                    "[Unroll] Original cloud has been automatically hidden");
            if (pc->getParent()) {
                pc->getParent()->addChild(output);
            }
            addToDB(output, true, true, false, true);
        }
        updateUI();
    }
}

void MainWindow::doComputeGeometricFeature() {
    static ccLibAlgorithms::GeomCharacteristicSet s_selectedCharacteristics;
    static CCVector3 s_upDir(0, 0, 1);
    static bool s_upDirDefined = false;

    ccGeomFeaturesDlg gfDlg(this);
    double radius =
            ccLibAlgorithms::GetDefaultCloudKernelSize(m_selectedEntities);
    gfDlg.setRadius(radius);

    // restore semi-persistent parameters
    gfDlg.setSelectedFeatures(s_selectedCharacteristics);
    if (s_upDirDefined) {
        gfDlg.setUpDirection(s_upDir);
    }

    if (!gfDlg.exec()) return;

    radius = gfDlg.getRadius();
    if (!gfDlg.getSelectedFeatures(s_selectedCharacteristics)) {
        CVLog::Error(tr("Not enough memory"));
        return;
    }

    CCVector3* upDir = gfDlg.getUpDirection();

    // remember semi-persistent parameters
    s_upDirDefined = (upDir != nullptr);
    if (s_upDirDefined) {
        s_upDir = *upDir;
    }

    ccLibAlgorithms::ComputeGeomCharacteristics(
            s_selectedCharacteristics, static_cast<PointCoordinateType>(radius),
            m_selectedEntities, upDir, this);

    refreshSelected();
    updateUI();
}

// Helper function to recursively collect functional actions from a menu
// Only collects leaf actions (actions without submenus) that are not excluded
static void collectActionsFromMenu(QMenu* menu,
                                   QList<QAction*>& actions,
                                   QSet<QAction*>& collected,
                                   const QSet<QMenu*>& excludedMenus) {
    if (!menu || excludedMenus.contains(menu)) {
        return;
    }

    for (QAction* action : menu->actions()) {
        if (!action) {
            continue;
        }

        // Skip separators
        if (action->isSeparator()) {
            continue;
        }

        // Skip actions without text (not user-visible)
        if (action->text().isEmpty()) {
            continue;
        }

        // Skip if already collected (avoid duplicates)
        if (collected.contains(action)) {
            continue;
        }

        // Skip toolbar toggle actions by objectName pattern
        // All toolbar toggle actions have objectName starting with
        // "actionDisplay"
        if (!action->objectName().isEmpty() &&
            action->objectName().startsWith("actionDisplay")) {
            continue;
        }

        // If action has a submenu, recursively process it
        QMenu* submenu = action->menu();
        if (submenu) {
            collectActionsFromMenu(submenu, actions, collected, excludedMenus);
            continue;  // Don't add menu items themselves
        }

        // This is a functional action (leaf node), add it
        actions.append(action);
        collected.insert(action);
    }
}

void MainWindow::populateActionList() {
    m_actions.clear();

    // Build set of excluded menus (dynamic menus that shouldn't have shortcuts)
    QSet<QMenu*> excludedMenus;

    // Exclude recent files menu
    if (m_recentFiles) {
        QMenu* recentFilesMenu = m_recentFiles->menu();
        if (recentFilesMenu) {
            excludedMenus.insert(recentFilesMenu);
        }
    }

    // Exclude 3D mouse manager menu
#ifdef CC_3DXWARE_SUPPORT
    if (m_3DMouseManager) {
        QMenu* mouseMenu = m_3DMouseManager->menu();
        if (mouseMenu) {
            excludedMenus.insert(mouseMenu);
        }
    }
#endif

    // Exclude gamepad manager menu
#ifdef CC_GAMEPAD_SUPPORT
    if (m_gamepadManager) {
        QMenu* gamepadMenu = m_gamepadManager->menu();
        if (gamepadMenu) {
            excludedMenus.insert(gamepadMenu);
        }
    }
#endif

    // Exclude toolbar menu (contains toolbar toggle actions)
    excludedMenus.insert(m_ui->menuToolbars);

    // Track collected actions to avoid duplicates
    QSet<QAction*> collected;

    // Collect actions from menuBar menus (efficient: only traverse menu
    // structure)
    for (QAction* menuBarAction : m_ui->menuBar->actions()) {
        QMenu* menu = menuBarAction->menu();
        if (menu) {
            collectActionsFromMenu(menu, m_actions, collected, excludedMenus);
        }
    }

    // Also collect actions from toolbars (for toolbar actions not in menus)
    for (QToolBar* toolbar : findChildren<QToolBar*>()) {
        // Skip plugin toolbars (they contain plugin actions)
        QString toolbarName = toolbar->objectName();
        if (toolbarName.contains("Plugin", Qt::CaseInsensitive) ||
            toolbarName.contains("PCL", Qt::CaseInsensitive)) {
            continue;
        }

        for (QAction* action : toolbar->actions()) {
            if (!action || action->isSeparator() || action->text().isEmpty()) {
                continue;
            }

            // Skip if already collected
            if (collected.contains(action)) {
                continue;
            }

            // Skip toolbar toggle actions
            if (!action->objectName().isEmpty() &&
                action->objectName().startsWith("actionDisplay")) {
                continue;
            }

            // Skip actions with submenus
            if (action->menu()) {
                continue;
            }

            m_actions.append(action);
            collected.insert(action);
        }
    }
}

void MainWindow::showShortcutDialog() {
    if (m_shortcutDlg) {
        m_shortcutDlg->exec();
    }
}

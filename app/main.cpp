// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// LOCAL
#include "MainWindow.h"
#include "ecvApplication.h"
#include "ecvCommandLineParser.h"
#include "ecvPersistentSettings.h"
#include "ecvSettingManager.h"
#include "ecvUIManager.h"

// CV_CORE_LIB
#include <CPUInfo.h>
#include <CVLog.h>
#include <CVTools.h>
#include <MemoryInfo.h>

// CV_DB_LIB
#include <ecvColorScalesManager.h>
#include <ecvGuiParameters.h>
#include <ecvNormalVectors.h>

// CV_IO_LIB
#include <FileIOFilter.h>
#include <ecvGlobalShiftManager.h>

// QT
#include <QDir>
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QSurfaceFormat>
#else
#include <QGLFormat>
#endif
#include <QMessageBox>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QPixmap>
#include <QSplashScreen>
#include <QStorageInfo>
#include <QSysInfo>
#include <QTime>
#include <QTimer>
#include <QTranslator>
#include <QtWidgets/QApplication>
#ifdef CC_GAMEPAD_SUPPORT
#include <QGamepadManager>
#endif

// System-specific includes
#ifdef _WIN32
#include <windows.h>  // For GetFileType, GetStdHandle, AttachConsole

#include <cstdio>  // For freopen
#endif
#ifdef Q_OS_MAC
#include <cstdlib>  // For putenv
#endif

// COMMON
#include "CommonSettings.h"
#include "ecvTranslationManager.h"

// PLUGINS
#include "ecvPluginInterface.h"
#include "ecvPluginManager.h"

#ifdef USE_VLD
#include <vld.h>
#endif

// #if defined(_MSC_VER) && (_MSC_VER >= 1600)
// #pragma execution_character_set("utf-8")
// #endif

// Function to get total system memory in GB
QString GetTotalMemoryInfo() {
    using namespace cloudViewer::system;

    MemoryInfo memInfo = getMemoryInfo();

    if (memInfo.totalRam > 0) {
        double totalGB = static_cast<double>(memInfo.totalRam) /
                         (1024.0 * 1024.0 * 1024.0);
        return QString("RAM: %1 GB").arg(totalGB, 0, 'f', 2);
    }

    return QString("RAM: Unable to detect");
}

// Function to get storage information
QString GetStorageInfo() {
    QStorageInfo storage = QStorageInfo::root();

    if (storage.isValid() && storage.isReady()) {
        qint64 totalBytes = storage.bytesTotal();
        qint64 availableBytes = storage.bytesAvailable();

        double totalGB = totalBytes / (1024.0 * 1024.0 * 1024.0);
        double availableGB = availableBytes / (1024.0 * 1024.0 * 1024.0);
        double usedGB = totalGB - availableGB;

        return QString("Storage: %1 GB total, %2 GB used, %3 GB available")
                .arg(totalGB, 0, 'f', 2)
                .arg(usedGB, 0, 'f', 2)
                .arg(availableGB, 0, 'f', 2);
    }

    return QString("Storage: Unable to detect");
}

// Function to get CPU information
QString GetCPUInfo() {
    using namespace cloudViewer::utility;

    const CPUInfo& cpuInfo = CPUInfo::GetInstance();
    int numCores = cpuInfo.NumCores();
    int numThreads = cpuInfo.NumThreads();
    std::string modelName = cpuInfo.ModelName();

    if (numCores <= 0 && numThreads <= 0) {
        return QString("CPU: Unable to detect");
    }

    QString result;
    if (!modelName.empty()) {
        // Show model name with cores and threads
        if (numCores > 0) {
            result = QString("CPU: %1 (%2 cores, %3 threads)")
                             .arg(QString::fromStdString(modelName))
                             .arg(numCores)
                             .arg(numThreads);
        } else {
            result = QString("CPU: %1 (%2 threads)")
                             .arg(QString::fromStdString(modelName))
                             .arg(numThreads);
        }
    } else {
        // Fallback: show only cores and threads
        if (numCores > 0) {
            result = QString("CPU: %1 cores, %2 threads")
                             .arg(numCores)
                             .arg(numThreads);
        } else {
            result = QString("CPU: %1 threads").arg(numThreads);
        }
    }

    return result;
}

// Function to get GPU information
QString GetGPUInfo() {
    QString gpuInfo = "GPU: ";

    // Create a temporary OpenGL context to query GPU info
    QOffscreenSurface surface;
    surface.create();

    QOpenGLContext context;
    if (context.create() && context.makeCurrent(&surface)) {
        QOpenGLFunctions* functions = context.functions();
        if (functions) {
            const GLubyte* vendor = functions->glGetString(GL_VENDOR);
            const GLubyte* renderer = functions->glGetString(GL_RENDERER);
            const GLubyte* version = functions->glGetString(GL_VERSION);

            if (vendor && renderer && version) {
                gpuInfo += QString("%1 %2 (OpenGL %3)")
                                   .arg(reinterpret_cast<const char*>(vendor))
                                   .arg(reinterpret_cast<const char*>(renderer))
                                   .arg(reinterpret_cast<const char*>(version));
            } else {
                gpuInfo += "Unable to query details";
            }
        }
        context.doneCurrent();
    } else {
        gpuInfo += "Unable to create OpenGL context";
    }

    return gpuInfo;
}

// Function to print all system hardware information
void PrintSystemHardwareInfo() {
    CVLog::Print(
            "=================================================================="
            "==============");
    CVLog::Print("System Hardware Information");
    CVLog::Print(
            "=================================================================="
            "==============");

    // Operating System
    CVLog::Print(QString("OS: %1 %2 (%3)")
                         .arg(QSysInfo::productType())
                         .arg(QSysInfo::productVersion())
                         .arg(QSysInfo::currentCpuArchitecture()));

    // Kernel version
    CVLog::Print(QString("Kernel: %1").arg(QSysInfo::kernelVersion()));

    // CPU Information
    CVLog::Print(GetCPUInfo());

    // Memory Information
    CVLog::Print(GetTotalMemoryInfo());

    // Storage Information
    CVLog::Print(GetStorageInfo());

    // GPU Information
    CVLog::Print(GetGPUInfo());

    CVLog::Print(
            "=================================================================="
            "==============");
}

void InitEnvironment() {
    // fix OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
    // already initialized.
#ifdef Q_OS_MAC
    char ompEnv[] = "KMP_DUPLICATE_LIB_OK=True";
    putenv(ompEnv);
#endif

    // store the log message until a valid logging instance is registered
    CVLog::EnableMessageBackup(true);

    // global structures initialization
    FileIOFilter::InitInternalFilters();  // load all known I/O filters

    // force pre-computed normals array initialization
    ccNormalVectors::GetUniqueInstance();
    // force pre-computed color tables initialization
    ccColorScalesManager::GetUniqueInstance();

    // load the plugins
    ccPluginManager::get().loadPlugins();

    ecvSettingManager::Init(Settings::CONFIG_PATH);  // init setting manager for
                                                     // persistent settings
    {  // restore some global parameters
        double maxAbsCoord =
                ecvSettingManager::getValue(
                        ecvPS::GlobalShift(), ecvPS::MaxAbsCoord(),
                        ecvGlobalShiftManager::MaxCoordinateAbsValue())
                        .toDouble();
        double maxAbsDiag =
                ecvSettingManager::getValue(
                        ecvPS::GlobalShift(), ecvPS::MaxAbsDiag(),
                        ecvGlobalShiftManager::MaxBoundgBoxDiagonal())
                        .toDouble();

        CVLog::Print(QObject::tr("Restore [Global Shift] Max abs. coord = %1 / "
                                 "max abs. diag = %2")
                             .arg(maxAbsCoord, 0, 'e', 0)
                             .arg(maxAbsDiag, 0, 'e', 0));

        ecvGlobalShiftManager::SetMaxCoordinateAbsValue(maxAbsCoord);
        ecvGlobalShiftManager::SetMaxBoundgBoxDiagonal(maxAbsDiag);
    }
}

int HandleQuickFlags(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-v") == 0) {
            printf("ACloudViewer %s\n", CLOUDVIEWER_VERSION);
            return 0;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("ACloudViewer %s - 3D Point Cloud & Mesh Processing\n"
                   "\n"
                   "Usage:\n"
                   "  ACloudViewer [files...]                  Open files in "
                   "GUI\n"
                   "  ACloudViewer -SILENT [commands...]       Headless CLI "
                   "mode\n"
                   "  ACloudViewer --version | -v              Print version\n"
                   "  ACloudViewer --help   | -h               Show this help\n"
                   "\n"
                   "I/O:\n"
                   "  -O <file>                     Open/load a file\n"
                   "    -SKIP <n>                     Skip first n lines "
                   "(ASCII)\n"
                   "    -GLOBAL_SHIFT AUTO|FIRST|<x> <y> <z>  Global shift on "
                   "load\n"
                   "  -SAVE_CLOUDS [FILE <path>]    Save point clouds\n"
                   "  -SAVE_MESHES [FILE <path>]    Save meshes\n"
                   "  -AUTO_SAVE ON|OFF             Toggle auto-save\n"
                   "  -NO_TIMESTAMP                 Disable filename "
                   "timestamps\n"
                   "  -LOG_FILE <path>              Write log to file\n"
                   "\n"
                   "Export format:\n"
                   "  -C_EXPORT_FMT <fmt>           Cloud format "
                   "(PLY,PCD,LAS,E57,BIN,ASC,SBF,DRC,...)\n"
                   "  -M_EXPORT_FMT <fmt>           Mesh format "
                   "(OBJ,STL,OFF,PLY,FBX,DXF,VTK,...)\n"
                   "  -H_EXPORT_FMT <fmt>           Hierarchy format\n"
                   "  -EXT <ext>                    Override output file "
                   "extension\n"
                   "  -PLY_EXPORT_FMT ASCII|BINARY_LE|BINARY_BE\n"
                   "  -PCD_OUTPUT_FORMAT <0|1>      PCD output format "
                   "(0=ASCII, 1=binary)\n"
                   "  -PREC <n>                     ASCII coord precision "
                   "(default: 12)\n"
                   "  -SEP <SPACE|SEMICOLON|COMMA|TAB>\n"
                   "  -ADD_HEADER                   Add column names header\n"
                   "  -ADD_PTS_COUNT                Add point count header\n"
                   "\n"
                   "Subsampling & filtering:\n"
                   "  -SS RANDOM|SPATIAL|OCTREE <param>  Subsample\n"
                   "  -EXTRACT_CC                   Extract connected "
                   "components\n"
                   "  -SOR <knn> <sigma>            Statistical Outlier "
                   "Removal\n"
                   "  -FILTER_SF <min> <max>        Filter by scalar field\n"
                   "  -CROP <Xmin:Ymin:Zmin:Xmax:Ymax:Zmax>  Crop bounding "
                   "box\n"
                   "    -OUTSIDE                      Keep outside region\n"
                   "  -CROP2D <ortho> <n> X1 Y1 ... Xn Yn  Crop by 2D polygon\n"
                   "  -CROSS_SECTION <file>         Cross section from "
                   "polyline file\n"
                   "\n"
                   "Normals:\n"
                   "  -COMPUTE_NORMALS              Compute normals (gridded)\n"
                   "  -OCTREE_NORMALS <radius>      Compute normals (octree)\n"
                   "  -ORIENT_NORMS_MST <knn>       Orient normals (MST)\n"
                   "  -INVERT_NORMALS               Invert normals\n"
                   "  -CLEAR_NORMALS                Remove normals\n"
                   "  -NORMALS_TO_DIP               Convert normals to "
                   "dip/dip-dir\n"
                   "  -NORMALS_TO_SFS               Convert normals to scalar "
                   "fields\n"
                   "\n"
                   "Scalar fields:\n"
                   "  -SET_ACTIVE_SF <idx>          Set active scalar field\n"
                   "  -SF_ARITHMETIC <idx> <op>     SF arithmetic "
                   "(ADD,SUB,MUL,...)\n"
                   "  -SF_OP <idx> <op> <val>       SF operation\n"
                   "  -RENAME_SF <old> <new>        Rename scalar field\n"
                   "  -COORD_TO_SF <X|Y|Z>          Export coordinate to SF\n"
                   "  -SF_COLOR_SCALE <file>        Apply color scale to SF\n"
                   "  -SF_CONVERT_TO_RGB             Convert SF to RGB\n"
                   "  -SF_GRAD [EUCLIDEAN]          SF gradient\n"
                   "  -RGB_CONVERT_TO_SF             Convert RGB to scalar "
                   "fields\n"
                   "  -REMOVE_SF <name>             Remove scalar field\n"
                   "  -REMOVE_ALL_SFS               Remove all scalar fields\n"
                   "  -REMOVE_RGB                   Remove colors\n"
                   "  -REMOVE_SCAN_GRIDS            Remove scan grids\n"
                   "\n"
                   "Geometry:\n"
                   "  -CURV <MEAN|GAUSS> <radius>   Compute curvature\n"
                   "  -DENSITY <radius>             Compute point density\n"
                   "  -APPROX_DENSITY               Approximate density\n"
                   "  -ROUGH <radius>               Compute roughness\n"
                   "    -UP_DIR <x> <y> <z>           Roughness up direction\n"
                   "  -MOMENT <order> <radius>      Compute moment\n"
                   "  -FEATURE <type> <radius>      Compute geometric feature\n"
                   "  -BEST_FIT_PLANE               Fit plane to cloud\n"
                   "    -MAKE_HORIZ                   Make plane horizontal\n"
                   "    -KEEP_LOADED                  Keep plane entity\n"
                   "  -MESH_VOLUME                  Compute mesh volume\n"
                   "    -TO_FILE <path>               Write volume to file\n"
                   "  -CBANDING <dim> <freq>        Apply color banding\n"
                   "\n"
                   "Raster / 2.5D:\n"
                   "  -RASTERIZE                    Rasterize to grid\n"
                   "    -VERT_DIR <0|1|2>             Vertical direction "
                   "(X/Y/Z)\n"
                   "    -GRID_STEP <val>              Grid cell size\n"
                   "    -OUTPUT_CLOUD                 Output as cloud\n"
                   "    -OUTPUT_MESH                  Output as mesh\n"
                   "    -OUTPUT_RASTER_Z <file>       Export raster (Z "
                   "heights)\n"
                   "    -OUTPUT_RASTER_RGB <file>     Export raster (RGB)\n"
                   "    -PROJ <MIN|MAX|AVG>           Projection type\n"
                   "    -EMPTY_FILL <method>          Empty cell filling\n"
                   "    -RESAMPLE                     Resample cloud\n"
                   "  -VOLUME                       2.5D volume calculation\n"
                   "    -GROUND_IS_FIRST              First cloud is ground\n"
                   "    -CONST_HEIGHT <val>           Use constant height\n"
                   "\n"
                   "Mesh:\n"
                   "  -DELAUNAY [AA|BEST_FIT]       Delaunay triangulation\n"
                   "    -MAX_EDGE_LENGTH <len>        Max edge constraint\n"
                   "  -SAMPLE_MESH POINTS <n>       Sample points from mesh\n"
                   "  -EXTRACT_VERTICES             Extract mesh vertices\n"
                   "  -FLIP_TRI                     Flip triangle normals\n"
                   "\n"
                   "Registration & transform:\n"
                   "  -ICP                          Iterative Closest Point\n"
                   "    -REFERENCE_IS_FIRST           Reference is first "
                   "cloud\n"
                   "    -MIN_ERROR_DIFF <val>         Convergence threshold\n"
                   "    -ITER <n>                     Max iterations\n"
                   "    -OVERLAP <pct>                Expected overlap "
                   "(0-100)\n"
                   "    -ADJUST_SCALE                 Allow scale adjustment\n"
                   "    -RANDOM_SAMPLING_LIMIT <n>    Random sampling limit\n"
                   "    -FARTHEST_REMOVAL             Enable farthest point "
                   "removal\n"
                   "    -MODEL_SF_AS_WEIGHTS <idx>    Model SF weights\n"
                   "    -DATA_SF_AS_WEIGHTS <idx>     Data SF weights\n"
                   "    -ROT <XYZ|X|Y|Z|NONE>        Constrain rotation axis\n"
                   "  -APPLY_TRANS <file>           Apply 4x4 transformation "
                   "matrix\n"
                   "  -DROP_GLOBAL_SHIFT            Remove global shift\n"
                   "  -MATCH_CENTERS                Match bounding box "
                   "centers\n"
                   "\n"
                   "Distance:\n"
                   "  -C2C_DIST                     Cloud-to-cloud distance\n"
                   "    -SPLIT_XYZ                    Split X/Y/Z components\n"
                   "  -C2M_DIST                     Cloud-to-mesh distance\n"
                   "    -FLIP_NORMS                   Flip normals\n"
                   "    -MODEL <LS|TRI|HF>            Local model type\n"
                   "  -M3C2                         M3C2 distance (plugin)\n"
                   "  -MAX_DIST <val>               Max comparison distance\n"
                   "  -OCTREE_LEVEL <n>             Octree level for "
                   "comparison\n"
                   "  -CLOSEST_POINT_SET            Extract closest point set\n"
                   "  -STAT_TEST <distrib> <p> <n>  Chi2 statistical test\n"
                   "\n"
                   "Cloud management:\n"
                   "  -MERGE_CLOUDS                 Merge all loaded clouds\n"
                   "  -MERGE_MESHES                 Merge all loaded meshes\n"
                   "  -CLEAR                        Clear all entities\n"
                   "  -CLEAR_CLOUDS                 Clear all clouds\n"
                   "  -CLEAR_MESHES                 Clear all meshes\n"
                   "  -POP_CLOUDS                   Remove last cloud\n"
                   "  -POP_MESHES                   Remove last mesh\n"
                   "\n"
                   "Plugins:\n"
                   "  -RANSAC [options]             RANSAC shape detection\n"
                   "  -CSF [options]                Cloth Simulation Filter "
                   "(ground)\n"
                   "  -CANUPO_CLASSIFY <file>       CANUPO classification\n"
                   "  -3DMASC_CLASSIFY <file>       3DMASC classification\n"
                   "  -TREEISO [options]            Tree isolation\n"
                   "  -PCV [options]                ShadeVis / ambient "
                   "occlusion\n"
                   "  -PYTHON_SCRIPT <file>         Run Python script\n"
                   "  -FBX [options]                FBX export options\n"
                   "  -FWF_O <file>                 Load LAS with full "
                   "waveform\n"
                   "  -FWF_SAVE_CLOUDS              Save LAS with full "
                   "waveform\n"
                   "  -BUNDLER_IMPORT <file>        Import Bundler .out\n"
                   "\n"
                   "Colmap reconstruction (separate binary):\n"
                   "  colmap automatic_reconstructor  Full SfM+MVS pipeline\n"
                   "  colmap feature_extractor        Extract image features\n"
                   "  colmap exhaustive_matcher        Feature matching\n"
                   "  colmap mapper                    SfM sparse "
                   "reconstruction\n"
                   "  colmap patch_match_stereo        Dense stereo matching\n"
                   "  colmap stereo_fusion             Fuse depth maps\n"
                   "  colmap poisson_mesher            Poisson surface "
                   "reconstruction\n"
                   "  colmap delaunay_mesher           Delaunay meshing\n"
                   "  colmap model_converter           Convert model format\n"
                   "  colmap image_undistorter         Undistort images\n"
                   "  (run 'colmap help' for full list of 40+ subcommands)\n"
                   "\n"
                   "System:\n"
                   "  -SILENT                       Headless mode (required "
                   "first arg)\n"
                   "  -VERBOSE                      Enable verbose logging\n"
                   "  -VERBOSITY <level>            Set verbosity (0-3)\n"
                   "  -DEBUG                        Enable debug mode\n"
                   "  -MAX_TCOUNT <n>               Max thread count\n"
                   "  -HELP                         List all registered "
                   "commands\n"
                   "\n"
                   "Examples:\n"
                   "  ACloudViewer -SILENT -O in.ply -SS SPATIAL 0.05 "
                   "-SAVE_CLOUDS\n"
                   "  ACloudViewer -SILENT -O in.pcd -C_EXPORT_FMT PLY "
                   "-SAVE_CLOUDS FILE out.ply\n"
                   "  ACloudViewer -SILENT -O mesh.obj -SAMPLE_MESH POINTS "
                   "50000 -SAVE_CLOUDS\n"
                   "  ACloudViewer -SILENT -O a.ply -O b.ply -ICP -ITER 50 "
                   "-SAVE_CLOUDS\n"
                   "  ACloudViewer -SILENT -O cloud.ply -SOR 6 1.0 "
                   "-SAVE_CLOUDS\n"
                   "  ACloudViewer -SILENT -O cloud.ply -CSF -SAVE_CLOUDS\n"
                   "  ACloudViewer -SILENT -O cloud.ply -RANSAC -SAVE_CLOUDS\n"
                   "  ACloudViewer -SILENT -HELP    (full list including "
                   "plugin commands)\n"
                   "\n"
                   "See: https://asher-1.github.io/ACloudViewer/\n",
                   CLOUDVIEWER_VERSION);
            return 0;
        }
    }
    return -1;
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    // Set UTF-8 code page for console output to properly display Chinese
    // characters CP_UTF8 = 65001
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    // This will allow printf to function on windows when opened from command
    // line
    DWORD stdout_type = GetFileType(GetStdHandle(STD_OUTPUT_HANDLE));
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        if (stdout_type ==
            FILE_TYPE_UNKNOWN)  // this will allow std redirection (./executable
                                // > out.txt)
        {
            freopen("CONOUT$", "w", stdout);
            freopen("CONOUT$", "w", stderr);
        }
    }
#endif

    if (int rc = HandleQuickFlags(argc, argv); rc >= 0) return rc;

#ifdef Q_OS_MAC
    // On macOS, when double-clicking the application, the Finder (sometimes!)
    // adds a command-line parameter like "-psn_0_582385" which is a "process
    // serial number". We need to recognize this and discount it when
    // determining if we are running on the command line or not.

    int numRealArgs = argc;

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "-psn_", 5) == 0) {
            --numRealArgs;
        }
    }

    bool commandLine = (numRealArgs > 1) && (argv[1][0] == '-');

    // macOS: for -SILENT (headless) mode, try offscreen platform if no
    // QT_QPA_PLATFORM is set.  The "minimal" plugin is not shipped in the
    // macOS bundle (only "cocoa" is), so we fall back gracefully:
    //   offscreen > minimal > (leave unset → Qt picks cocoa, which works
    //   on macOS runners because WindowServer is always available).
    if (commandLine && numRealArgs > 1) {
        QString firstArg = QString(argv[1]).toUpper();
        if (firstArg == "-SILENT" &&
            qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
            QStringList candidates = {"offscreen", "minimal"};
            QString pluginDir = QCoreApplication::applicationDirPath() +
                                "/../PlugIns/platforms";
            if (!QDir(pluginDir).exists())
                pluginDir =
                        QCoreApplication::applicationDirPath() + "/platforms";
            bool found = false;
            for (const auto& name : candidates) {
                QString pattern = QString("libq%1*").arg(name);
                if (QDir(pluginDir).entryList({pattern}).size() > 0) {
                    qputenv("QT_QPA_PLATFORM", name.toUtf8());
                    found = true;
                    break;
                }
            }
            (void)found;
        }
    }
#else
    bool commandLine = (argc > 1) && (argv[1][0] == '-');
#endif

    ecvApplication::InitOpenGL();

#ifdef CC_GAMEPAD_SUPPORT
    QGamepadManager::instance();  // potential workaround to bug
                                  // https://bugreports.qt.io/browse/QTBUG-61553
#endif

    ecvApplication app(argc, argv, commandLine);

    // Set UTF-8 encoding for QString conversion from/to std::string
    // This ensures proper handling of Chinese and other Unicode characters
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));
#endif

    // QApplication docs suggest resetting to "C" after the QApplication is
    // initialized.
    setlocale(LC_NUMERIC, "C");

    // However, this is needed to address BUG #17225, #17226.
    QLocale::setDefault(QLocale::c());

    // specific commands
    int lastArgumentIndex = 1;
    QTranslator translator;
    if (commandLine) {
        // translation file selection
        if (QString(argv[lastArgumentIndex]).toUpper() == "-LANG") {
            QString langFilename = QString::fromLocal8Bit(argv[2]);

            ccTranslationManager::get().loadTranslation(langFilename);
            commandLine = false;
            lastArgumentIndex += 2;
        }
    }

    // splash screen
    QScopedPointer<QSplashScreen> splash(nullptr);
    QTimer splashTimer;
    // standard mode
    if (!commandLine) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        QOpenGLContext context;
        QSurfaceFormat format;
        format.setVersion(2, 1);
        context.setFormat(format);
        if (!context.create() || !context.isValid()) {
            QMessageBox::critical(nullptr, QObject::tr("Error"),
                                  QObject::tr("This application needs OpenGL "
                                              "2.1 at least to run!"));
            return EXIT_FAILURE;
        }
#else
        if ((QGLFormat::openGLVersionFlags() & QGLFormat::OpenGL_Version_2_1) ==
            0) {
            QMessageBox::critical(nullptr, QObject::tr("Error"),
                                  QObject::tr("This application needs OpenGL "
                                              "2.1 at least to run!"));
            return EXIT_FAILURE;
        }
#endif

        // splash screen
        QPixmap pixmap(QString::fromUtf8(
                CVTools::FromQString(Settings::APP_START_LOGO).c_str()));
        splash.reset(new QSplashScreen(pixmap, Qt::WindowStaysOnTopHint));
        splash->show();
        QApplication::processEvents();
    }

    // In command-line mode, suppress verbose startup logs (plugin loading,
    // global shift restore, color scale manager, etc.) unless the user
    // explicitly passes -VERBOSE.
    if (commandLine) {
        bool hasVerboseFlag = false;
        for (int i = 1; i < argc; ++i) {
            if (QString(argv[i]).toUpper() == "-VERBOSE") {
                hasVerboseFlag = true;
                break;
            }
        }
        if (!hasVerboseFlag) {
            CVLog::SetVerbosityLevel(CVLog::LOG_WARNING);
        }
    }

    // init environment
    InitEnvironment();

    int result = 0;
    // UI settings
    QUIWidget* qui = nullptr;

    // command line mode
    if (commandLine) {
        // command line processing (no GUI)
        result = ccCommandLineParser::Parse(
                argc, argv, ccPluginManager::get().pluginList());
    } else {
        // main window init.
        MainWindow* mainWindow = MainWindow::TheInstance();

        if (!mainWindow) {
            QMessageBox::critical(nullptr, QObject::tr("Error"),
                                  QObject::tr("Failed to initialize the main "
                                              "application window?!"));
            return EXIT_FAILURE;
        }

        mainWindow->initPlugins();

        // Print system hardware information
        PrintSystemHardwareInfo();

        if (Settings::UI_WRAPPER) {
            // use UIManager instead
            QUIWidget::setCode();
            qui = new QUIWidget();
            // set main weindow
            mainWindow->setUiManager(qui);
            qui->setMainWidget(mainWindow);

            qui->setTitle(Settings::APP_TITLE);

            // set align center - center the window title text
            qui->setAlignment(Qt::AlignCenter);

            // set dragable
            qui->setSizeGripEnabled(true);

            // set icon visibility
            qui->setVisible(QUIWidget::Lab_Ico, true);
            qui->setVisible(QUIWidget::BtnMenu, true);

            // persistent Theme settings
            QString qssfile =
                    ecvSettingManager::getValue(ecvPS::ThemeSettings(),
                                                ecvPS::CurrentTheme(),
                                                Settings::DEFAULT_STYLE)
                            .toString();
            qui->setStyle(qssfile);

            // qui.setIconMain(QChar(0xf099), 11);

            qui->setPixmap(QUIWidget::Lab_Ico, Settings::APP_LOGO);
            qui->setWindowIcon(QIcon(Settings::APP_LOGO));

            qui->createTrayMenu();

            qui->show();
        } else {
            // use default ui
            mainWindow->setWindowIcon(QIcon(Settings::APP_LOGO));
            mainWindow->setWindowTitle(Settings::APP_TITLE);
            // persistent Theme settings
            QString qssfile =
                    ecvSettingManager::getValue(ecvPS::ThemeSettings(),
                                                ecvPS::CurrentTheme(),
                                                Settings::DEFAULT_STYLE)
                            .toString();
            MainWindow::ChangeStyle(qssfile);
            mainWindow->show();
        }

        // close start logo according timer
        QApplication::processEvents();

        // show current Global Shift parameters in Console
        {
            CVLog::Print(
                    QObject::tr("Current [Global Shift] Max abs. coord = %1 / "
                                "max abs. diag = %2")
                            .arg(ecvGlobalShiftManager::MaxCoordinateAbsValue(),
                                 0, 'e', 0)
                            .arg(ecvGlobalShiftManager::MaxBoundgBoxDiagonal(),
                                 0, 'e', 0));
        }

        if (argc > lastArgumentIndex) {
            if (splash) {
                splash->close();
            }

            // any additional argument is assumed to be a filename --> we try to
            // load it/them
            QStringList filenames;
            for (int i = lastArgumentIndex; i < argc; ++i) {
                QString arg(argv[i]);
                // special command: auto start a plugin
                if (arg.startsWith(":start-plugin:")) {
                    QString pluginName = arg.mid(14);
                    QString pluginNameUpper = pluginName.toUpper();
                    // look for this plugin
                    bool found = false;
                    for (ccPluginInterface* plugin :
                         ccPluginManager::get().pluginList()) {
                        if (plugin->getName().replace(' ', '_').toUpper() ==
                            pluginNameUpper) {
                            found = true;
                            bool success = plugin->start();
                            if (!success) {
                                CVLog::Error(QObject::tr("Failed to start the "
                                                         "plugin '%1'")
                                                     .arg(plugin->getName()));
                            }
                            break;
                        }
                    }

                    if (!found) {
                        CVLog::Error(
                                QObject::tr("Couldn't find the plugin '%1'")
                                        .arg(pluginName.replace('_', ' ')));
                    }
                } else {
                    filenames << QString::fromLocal8Bit(argv[i]);
                }
            }

            mainWindow->addToDB(filenames);
        } else if (splash) {
            QObject::connect(&splashTimer, &QTimer::timeout, [&]() {
                splashTimer.stop();
                if (splash) splash->close();
                QCoreApplication::processEvents();
                splash.reset();
            });
            splashTimer.setInterval(1000);
            splashTimer.setSingleShot(true);
            splashTimer.start();
        }

        // change the default path to the application one (do this AFTER
        // processing the command line)
        QDir workingDir = QCoreApplication::applicationDirPath();

#ifdef Q_OS_MAC
        // This makes sure that our "working directory" is not within the
        // application bundle
        if (workingDir.dirName() == "MacOS") {
            workingDir.cdUp();
            workingDir.cdUp();
            workingDir.cdUp();
        }
#endif

        QDir::setCurrent(workingDir.absolutePath());

        // let's rock!
        try {
            // mainWindow->drawWidgets();
            result = app.exec();
        } catch (const std::exception& e) {
            QMessageBox::warning(
                    nullptr, QObject::tr("ECV crashed!"),
                    QObject::tr("Hum, it seems that ECV has crashed... Sorry "
                                "about that :)\n") +
                            e.what());
        } catch (...) {
            QMessageBox::warning(nullptr, QObject::tr("ECV crashed!"),
                                 QObject::tr("Hum, it seems that ECV has "
                                             "crashed... Sorry about that :)"));
        }

        // release the plugins
        for (ccPluginInterface* plugin : ccPluginManager::get().pluginList()) {
            plugin->stop();  // just in case
        }
    }

    /**
     * release global structures
     */

    // release main window
    MainWindow::DestroyInstance();

    // release io filters
    FileIOFilter::UnregisterAll();

    // release setting manager
    ecvSettingManager::ReleaseInstance();

    // release ui manager in this case
    if (qui != nullptr) {
        delete qui;
        qui = nullptr;
    }

#ifdef CV_TRACK_ALIVE_SHARED_OBJECTS
    // for debug purposes
    unsigned alive = CCShareable::GetAliveCount();
    if (alive > 1) {
        printf("Error: some shared objects (%u) have not been released on "
               "program end!",
               alive);
        system("PAUSE");
    }
#endif

    return result;
}

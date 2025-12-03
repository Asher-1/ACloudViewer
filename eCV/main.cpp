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
#include <CVLog.h>
#include <CVTools.h>

// ECV_DB_LIB
#include <ecvColorScalesManager.h>
#include <ecvGuiParameters.h>
#include <ecvNormalVectors.h>

// ECV_IO_LIB
#include <FileIOFilter.h>
#include <ecvGlobalShiftManager.h>

// QT
#include <QDir>
#include <QGLFormat>
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

// System-specific includes for hardware info
#ifdef Q_OS_LINUX
#include <sys/sysinfo.h>
#include <unistd.h>
#endif
#ifdef Q_OS_WIN
#include <sysinfoapi.h>
#include <windows.h>
#endif
#ifdef Q_OS_MAC
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
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
    QString memInfo;

#ifdef Q_OS_LINUX
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        unsigned long long totalRAM = si.totalram * si.mem_unit;
        double totalGB = totalRAM / (1024.0 * 1024.0 * 1024.0);
        memInfo = QString("RAM: %1 GB").arg(totalGB, 0, 'f', 2);
    }
#elif defined(Q_OS_WIN)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        double totalGB = statex.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
        memInfo = QString("RAM: %1 GB").arg(totalGB, 0, 'f', 2);
    }
#elif defined(Q_OS_MAC)
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    int64_t physical_memory;
    size_t length = sizeof(int64_t);
    if (sysctl(mib, 2, &physical_memory, &length, NULL, 0) == 0) {
        double totalGB = physical_memory / (1024.0 * 1024.0 * 1024.0);
        memInfo = QString("RAM: %1 GB").arg(totalGB, 0, 'f', 2);
    }
#endif

    if (memInfo.isEmpty()) {
        memInfo = "RAM: Unable to detect";
    }

    return memInfo;
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
    QString cpuInfo;

#ifdef Q_OS_LINUX
    // Get number of processors
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);

    // Try to read CPU model from /proc/cpuinfo
    QFile file("/proc/cpuinfo");
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&file);
        while (!in.atEnd()) {
            QString line = in.readLine();
            if (line.startsWith("model name")) {
                QStringList parts = line.split(":");
                if (parts.size() >= 2) {
                    cpuInfo = QString("CPU: %1 (%2 cores)")
                                      .arg(parts[1].trimmed())
                                      .arg(nprocs);
                    break;
                }
            }
        }
        file.close();
    }

    if (cpuInfo.isEmpty()) {
        cpuInfo = QString("CPU: %1 cores").arg(nprocs);
    }
#elif defined(Q_OS_WIN)
    SYSTEM_INFO siSysInfo;
    GetSystemInfo(&siSysInfo);
    cpuInfo = QString("CPU: %1 cores").arg(siSysInfo.dwNumberOfProcessors);
#elif defined(Q_OS_MAC)
    int mib[2] = {CTL_HW, HW_NCPU};
    int numCPU = 0;
    size_t len = sizeof(numCPU);
    if (sysctl(mib, 2, &numCPU, &len, NULL, 0) == 0) {
        cpuInfo = QString("CPU: %1 cores").arg(numCPU);
    }
#endif

    if (cpuInfo.isEmpty()) {
        cpuInfo = "CPU: Unable to detect";
    }

    return cpuInfo;
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

int main(int argc, char* argv[]) {
#ifdef _WIN32  // This will allow printf to function on windows when opened from
               // command line
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
#else
    bool commandLine = (argc > 1) && (argv[1][0] == '-');
#endif

    ecvApplication::InitOpenGL();

#ifdef CC_GAMEPAD_SUPPORT
    QGamepadManager::instance();  // potential workaround to bug
                                  // https://bugreports.qt.io/browse/QTBUG-61553
#endif

    ecvApplication app(argc, argv, commandLine);

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
#ifdef Q_OS_MAC
        if ((QGLFormat::openGLVersionFlags() & QGLFormat::OpenGL_Version_3_3) ==
            0) {
            QMessageBox::critical(nullptr, QObject::tr("Error"),
                                  QObject::tr("This application needs OpenGL "
                                              "3.3 at least to run!"));
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

            // set align center
            qui->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);

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
            // count-down to hide the timer (only effective once the application
            // will have actually started!)
            QObject::connect(&splashTimer, &QTimer::timeout, [&]() {
                if (splash) splash->close();
                QCoreApplication::processEvents();
                splash.reset();
            });
            splashTimer.setInterval(1000);
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

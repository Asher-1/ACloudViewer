// LOCAL
#include "MainWindow.h"
#include "ecvApplication.h"
#include "ecvUIManager.h"
#include "ecvCommandLineParser.h"
#include "ecvSettingManager.h"
#include "ecvPersistentSettings.h"

// CV_CORE_LIB
#include <CVLog.h>

// ECV_DB_LIB
#include <ecvColorScalesManager.h>
#include <ecvNormalVectors.h>
#include <ecvGuiParameters.h>

// ECV_IO_LIB
#include <FileIOFilter.h>
#include <ecvWidgetsInterface.h>
#include <ecvGlobalShiftManager.h>

// QPCL_ENGINE_LIB
#include <VtkUtils/vtkWidgetsFactory.h>

// QT
#include <QDir>
#include <QtWidgets/QApplication>
#include <QMessageBox>
#include <QPixmap>
#include <QSplashScreen>
#include <QTime>
#include <QTimer>
#include <QGLFormat>
#include <QTranslator>

// COMMON
#include <CommonSettings.h>

// PLUGINS
#include "ecvPluginInterface.h"
#include "ecvPluginManager.h"

#ifdef USE_VLD
//VLD
#include <vld.h>
#endif

#ifdef Q_OS_MAC
#include <unistd.h>
#endif

void InitEnvironment()
{
	//store the log message until a valid logging instance is registered
	CVLog::EnableMessageBackup(true);

	//global structures initialization
	FileIOFilter::InitInternalFilters(); //load all known I/O filters

	DBLib::ecvWidgetsInterface::InitInternalInterfaces();
	DBLib::ecvWidgetsInterface::Register(VtkWidgetsFactory::GetFilterWidgetInterface());
	DBLib::ecvWidgetsInterface::Register(VtkWidgetsFactory::GetSmallWidgetsInterface());
	DBLib::ecvWidgetsInterface::Register(VtkWidgetsFactory::GetSurfaceWidgetsInterface());

	// force pre-computed normals array initialization
	ccNormalVectors::GetUniqueInstance(); 
	// force pre-computed color tables initialization
	ccColorScalesManager::GetUniqueInstance(); 

	//load the plugins
	ccPluginManager::get().loadPlugins();

	ecvSettingManager::Init(Settings::CONFIG_PATH);  // init setting manager for persistent settings
	{   // restore some global parameters
		double maxAbsCoord = ecvSettingManager::getValue(ecvPS::GlobalShift(),
			ecvPS::MaxAbsCoord(), ecvGlobalShiftManager::MaxCoordinateAbsValue()).toDouble();
		double maxAbsDiag = ecvSettingManager::getValue(ecvPS::GlobalShift(),
			ecvPS::MaxAbsDiag(), ecvGlobalShiftManager::MaxBoundgBoxDiagonal()).toDouble();

		CVLog::Print(QObject::tr("Restore [Global Shift] Max abs. coord = %1 / max abs. diag = %2").
			arg(maxAbsCoord, 0, 'e', 0).arg(maxAbsDiag, 0, 'e', 0));

		ecvGlobalShiftManager::SetMaxCoordinateAbsValue(maxAbsCoord);
		ecvGlobalShiftManager::SetMaxBoundgBoxDiagonal(maxAbsDiag);
	}
}

int main(int argc, char *argv[])
{
#ifdef _WIN32 //This will allow printf to function on windows when opened from command line	
	DWORD stdout_type = GetFileType(GetStdHandle(STD_OUTPUT_HANDLE));
	if (AttachConsole(ATTACH_PARENT_PROCESS))
	{
		if (stdout_type == FILE_TYPE_UNKNOWN) // this will allow std redirection (./executable > out.txt)
		{
			freopen("CONOUT$", "w", stdout);
			freopen("CONOUT$", "w", stderr);
		}
	}
#endif

#ifdef Q_OS_MAC
	bool commandLine = isatty(fileno(stdin));
#else
	bool commandLine = (argc > 1) && (argv[1][0] == '-');
#endif

	ecvApplication::init(commandLine);

	ecvApplication app(argc, argv, commandLine);

	// QApplication docs suggest resetting to "C" after the QApplication is initialized.
	setlocale(LC_NUMERIC, "C");

	// However, this is needed to address BUG #17225, #17226.
	QLocale::setDefault(QLocale::c());

	//specific commands
	int lastArgumentIndex = 1;
	QTranslator translator;
	if (commandLine)
	{
		//translation file selection
		if (QString(argv[lastArgumentIndex]).toUpper() == "-LANG")
		{
			QString langFilename = QString(argv[2]);

			//Load translation file
			if (translator.load(langFilename, QCoreApplication::applicationDirPath()))
			{
				qApp->installTranslator(&translator);
			}
			else
			{
				QMessageBox::warning(0, QObject::tr("Translation"), QObject::tr("Failed to load language file '%1'").arg(langFilename));
			}
			commandLine = false;
			lastArgumentIndex += 2;
		}
	}

	//splash screen
	QScopedPointer<QSplashScreen> splash(nullptr);
	QTimer splashTimer;
	//standard mode
	if (!commandLine)
	{
		if ((QGLFormat::openGLVersionFlags() & QGLFormat::OpenGL_Version_2_1) == 0)
		{
			QMessageBox::critical(0, QObject::tr("Error"),
				QObject::tr("This application needs OpenGL 2.1 at least to run!"));
			return EXIT_FAILURE;
		}

		//splash screen
		QPixmap pixmap(QString::fromUtf8(Settings::APP_START_LOGO));
		splash.reset(new QSplashScreen(pixmap, Qt::WindowStaysOnTopHint));
		splash->show();
		QApplication::processEvents();
	}

	// init environment
	InitEnvironment();

	int result = 0;
	// UI settings
	QUIWidget* qui = nullptr;

	//command line mode
	if (commandLine)
	{
		//command line processing (no GUI)
		result = ccCommandLineParser::Parse(argc, argv, ccPluginManager::get().pluginList());
	}
	else
	{
		//main window init.
		MainWindow* mainWindow = MainWindow::TheInstance();
		if (!mainWindow)
		{
			QMessageBox::critical(0, QObject::tr("Error"),
				QObject::tr("Failed to initialize the main application window?!"));
			return EXIT_FAILURE;
		}

		mainWindow->initPlugins();

		if (Settings::UI_WRAPPER) {
			// use UIManager instead
			QUIWidget::setCode();
			qui = new QUIWidget();
			//����������
			mainWindow->setUiManager(qui);
			qui->setMainWidget(mainWindow);

			qui->setTitle(Settings::APP_TITLE);

			//���ñ����ı�����
			qui->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);

			//���ô�����϶���С
			qui->setSizeGripEnabled(true);

			//���û��������˵��ɼ�
			qui->setVisible(QUIWidget::Lab_Ico, true);
			qui->setVisible(QUIWidget::BtnMenu, true);

			//persistent Theme settings
			QString qssfile = ecvSettingManager::getValue(ecvPS::ThemeSettings(),
				ecvPS::CurrentTheme(),
				Settings::DEFAULT_STYLE).toString();
			qui->setStyle(qssfile);

			//�������Ͻ�ͼ��-ͼ������
			//qui.setIconMain(QChar(0xf099), 11);

			//�������Ͻ�ͼ��-ͼƬ�ļ�
			qui->setPixmap(QUIWidget::Lab_Ico, Settings::APP_LOGO);
			qui->setWindowIcon(QIcon(Settings::APP_LOGO));

			qui->createTrayMenu();

			qui->show();
		}
		else {
			// use default ui
			mainWindow->setWindowIcon(QIcon(Settings::APP_LOGO));
			mainWindow->setWindowTitle(Settings::APP_TITLE);
			//persistent Theme settings
			QString qssfile = ecvSettingManager::getValue(
				ecvPS::ThemeSettings(), ecvPS::CurrentTheme(),
				Settings::DEFAULT_STYLE).toString();
			MainWindow::ChangeStyle(qssfile);
			mainWindow->show();
		}

		// close start logo according timer
		QApplication::processEvents();

		//show current Global Shift parameters in Console
		{
			CVLog::Print(QObject::tr("Current [Global Shift] Max abs. coord = %1 / max abs. diag = %2")
				.arg(ecvGlobalShiftManager::MaxCoordinateAbsValue(), 0, 'e', 0)
				.arg(ecvGlobalShiftManager::MaxBoundgBoxDiagonal(), 0, 'e', 0));
		}

		if (argc > lastArgumentIndex)
		{
			if (splash)
			{
				splash->close();
			}

			//any additional argument is assumed to be a filename --> we try to load it/them
			QStringList filenames;
			for (int i = lastArgumentIndex; i < argc; ++i)
			{
				QString arg(argv[i]);
				//special command: auto start a plugin
				if (arg.startsWith(":start-plugin:"))
				{
					QString pluginName = arg.mid(14);
					QString pluginNameUpper = pluginName.toUpper();
					//look for this plugin
					bool found = false;
					for (ccPluginInterface *plugin : ccPluginManager::get().pluginList())
					{
						if (plugin->getName().replace(' ', '_').toUpper() == pluginNameUpper)
						{
							found = true;
							bool success = plugin->start();
							if (!success)
							{
								CVLog::Error(QObject::tr("Failed to start the plugin '%1'").arg(plugin->getName()));
							}
							break;
						}
					}

					if (!found)
					{
						CVLog::Error(QObject::tr("Couldn't find the plugin '%1'").arg(pluginName.replace('_', ' ')));
					}
				}
				else
				{
					filenames << QString::fromLocal8Bit(argv[i]);
				}
			}

			mainWindow->addToDB(filenames);
		}
		else if (splash)
		{
			//count-down to hide the timer (only effective once the application will have actually started!)
			QObject::connect(&splashTimer, &QTimer::timeout, [&]() { if (splash) splash->close(); QCoreApplication::processEvents(); splash.reset(); });
			splashTimer.setInterval(1000);
			splashTimer.start();
		}

		//change the default path to the application one (do this AFTER processing the command line)
		QDir  workingDir = QCoreApplication::applicationDirPath();
		
#ifdef Q_OS_MAC
		// This makes sure that our "working directory" is not within the application bundle	
		if (workingDir.dirName() == "MacOS")
		{
			workingDir.cdUp();
			workingDir.cdUp();
			workingDir.cdUp();
		}
#endif
		
		QDir::setCurrent(workingDir.absolutePath());

		//let's rock!
		try
		{
			//mainWindow->drawWidgets();
			result = app.exec();
		}
		catch (const std::exception& e)
		{
			QMessageBox::warning(0, QObject::tr("ECV crashed!"), QObject::tr("Hum, it seems that ECV has crashed... Sorry about that :)\n") + e.what());
		}
		catch (...)
		{
			QMessageBox::warning(0, QObject::tr("ECV crashed!"), QObject::tr("Hum, it seems that ECV has crashed... Sorry about that :)"));
		}

		//release the plugins
		for (ccPluginInterface *plugin : ccPluginManager::get().pluginList())
		{
			plugin->stop(); //just in case
		}
	}

	/**
	 * release global structures
	 */
	
	// release main window
	MainWindow::DestroyInstance();

	// release widgets resource
	DBLib::ecvWidgetsInterface::UnregisterAll();

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
	//for debug purposes
	unsigned alive = CCShareable::GetAliveCount();
	if (alive > 1)
	{
		printf("Error: some shared objects (%u) have not been released on program end!", alive);
		system("PAUSE");
	}
#endif

	return result;
}

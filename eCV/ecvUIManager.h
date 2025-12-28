// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#ifdef Q_OS_WIN
#define NEWLINE "\r\n"
#else
#define NEWLINE "\n"
#endif

/**
 * QUI Frameless Window Widget
 * 1: Built-in N >= 12 beautiful styles, can be switched directly, or custom
 * style paths can be set 2: Can set icons or images for widgets (top-left
 * icon/minimize button/maximize button/close button) and their visibility 3:
 * Can integrate designer plugin, drag and drop directly, WYSIWYG 4: If window
 * resizing is needed, set setSizeGripEnabled(true); 5: Can set global style
 * with setStyle 6: Can show message box with showMessageBoxInfo 7: Can show
 * error box with showMessageBoxError 8: Can show question box with
 * showMessageBoxQuestion 9: Can show input box with showInputBox 10: Integrated
 * graphics font setting methods and get image from specified text 11:
 * Integrated static methods for setting window center display/setting
 * translation files/setting encoding/setting delay/setting system time, etc.
 * 12: Integrated methods for getting application file name, etc.
 */

#include <QPainterPath>
#include <QSystemTrayIcon>

#include "ecvHead.h"

class MainWindow;

#ifdef quc
#if (QT_VERSION < QT_VERSION_CHECK(5, 7, 0))
#include <QtDesigner/QDesignerExportWidget>
#else
#include <QtUiPlugin/QDesignerExportWidget>
#endif

class QDESIGNER_WIDGET_EXPORT QUIWidget : public QDialog
#else
class QUIWidget : public QDialog
#endif

{
    Q_OBJECT
    Q_ENUMS(Style)
    Q_PROPERTY(QString title READ getTitle WRITE setTitle)
    Q_PROPERTY(Qt::Alignment alignment READ getAlignment WRITE setAlignment)

public:
    // Expose some objects as enum values to external
    enum Widget {
        Lab_Ico = 0,         // Top-left icon
        BtnMenu = 1,         // Dropdown menu button
        BtnMenu_Min = 2,     // Minimize button
        BtnMenu_Max = 3,     // Maximize button
        BtnMenu_Normal = 4,  // Restore button
        BtnMenu_Close = 5    // Close button
    };

    // Style enumeration
    enum Style {
        Style_Silvery = 0,     // Silvery style
        Style_Blue = 1,        // Blue style
        Style_LightBlue = 2,   // Light blue style
        Style_DarkBlue = 3,    // Dark blue style
        Style_Gray = 4,        // Gray style
        Style_LightGray = 5,   // Light gray style
        Style_DarkGray = 6,    // Dark gray style
        Style_Black = 7,       // Black style
        Style_LightBlack = 8,  // Light black style
        Style_DarkBlack = 9,   // Dark black style
        Style_PSBlack = 10,    // PS black style
        Style_FlatBlack = 11,  // Flat black style
        Style_FlatWhite = 12   // Flat white style
    };

    // Global static methods collection start--------------------------------
public:
    // Desktop width and height
    static int deskWidth();
    static int deskHeight();

    // Application file name
    static QString appName();
    // Application current path
    static QString appPath();

    // Create new directory
    static void newDir(const QString &dirName);

    // Write message to additional log file
    static void writeInfo(const QString &info, const QString &filePath = "log");

    // Set global style
    static void setStyle(QUIWidget::Style style);
    static void setStyle(QString &qssFile);
    static void setStyle(const QString &qssFile,
                         QString &paletteColor,
                         QString &textColor);
    static void setStyle(const QString &qssFile,
                         QString &textColor,
                         QString &panelColor,
                         QString &borderColor,
                         QString &normalColorStart,
                         QString &normalColorEnd,
                         QString &darkColorStart,
                         QString &darkColorEnd,
                         QString &highColor);

    // Get corresponding color values from QSS style
    static void getQssColor(const QString &qss,
                            QString &textColor,
                            QString &panelColor,
                            QString &borderColor,
                            QString &normalColorStart,
                            QString &normalColorEnd,
                            QString &darkColorStart,
                            QString &darkColorEnd,
                            QString &highColor);

    // Set form to center display
    static void setFormInCenter(QWidget *frm);
    // Set translation file
    static void setTranslator(const QString &qmFile = ":/image/qt_zh_CN.qm");
    // Set encoding
    static void setCode();
    // Set delay
    static void sleep(int sec);
    // Set system time
    static void setSystemDateTime(const QString &year,
                                  const QString &month,
                                  const QString &day,
                                  const QString &hour,
                                  const QString &min,
                                  const QString &sec);
    // Set auto-start on boot
    static void runWithSystem(const QString &strName,
                              const QString &strPath,
                              bool autoRun = true);

    // Check if it's an IP address
    static bool isIP(const QString &ip);

    // Check if it's a MAC address
    static bool isMac(const QString &mac);

    // Check if it's a valid phone number
    static bool isTel(const QString &tel);

    // Check if it's a valid email address
    static bool isEmail(const QString &email);

    // Convert hex string to decimal
    static int strHexToDecimal(const QString &strHex);

    // Convert decimal string to decimal
    static int strDecimalToDecimal(const QString &strDecimal);

    // Convert binary string to decimal
    static int strBinToDecimal(const QString &strBin);

    // Convert hex string to binary string
    static QString strHexToStrBin(const QString &strHex);

    // Convert decimal to binary string (one byte)
    static QString decimalToStrBin1(int decimal);

    // Convert decimal to binary string (two bytes)
    static QString decimalToStrBin2(int decimal);

    // Convert decimal to hex string, zero-padded
    static QString decimalToStrHex(int decimal);

    // Convert int to byte array
    static QByteArray intToByte(int i);

    // Convert byte array to int
    static int byteToInt(const QByteArray &data);

    // Convert ushort to byte array
    static QByteArray ushortToByte(ushort i);

    // Convert byte array to ushort
    static int byteToUShort(const QByteArray &data);

    // XOR encryption algorithm
    static QString getXorEncryptDecrypt(const QString &str, char key);

    // XOR checksum
    static uchar getOrCode(const QByteArray &data);

    // Calculate checksum
    static uchar getCheckCode(const QByteArray &data);

    // Convert byte array to ASCII string
    static QString byteArrayToAsciiStr(const QByteArray &data);

    // Convert hex string to byte array
    static QByteArray hexStrToByteArray(const QString &str);
    static char convertHexChar(char ch);

    // Convert ASCII string to byte array
    static QByteArray asciiStrToByteArray(const QString &str);

    // Convert byte array to hex string
    static QString byteArrayToHexStr(const QByteArray &data);

    // Get selected file
    static QString getFileName(
            const QString &filter,
            QString defaultDir = QCoreApplication::applicationDirPath());

    // Get selected file collection
    static QStringList getFileNames(
            const QString &filter,
            QString defaultDir = QCoreApplication::applicationDirPath());

    // Get selected directory
    static QString getFolderName();

    // Get file name with extension
    static QString getFileNameWithExtension(const QString &strFilePath);

    // Get files in selected folder
    static QStringList getFolderFileNames(const QStringList &filter);

    // Check if folder exists
    static bool folderIsExist(const QString &strFolder);

    // Check if file exists
    static bool fileIsExist(const QString &strFile);

    // Copy file
    static bool copyFile(const QString &sourceFile, const QString &targetFile);

    // Delete all files in folder
    static void deleteDirectory(const QString &path);

    // Check if IP address and port are online
    static bool ipLive(const QString &ip, int port, int timeout = 1000);

    // Get all source code of web page
    static QString getHtml(const QString &url);

    // Get local public IP address
    static QString getNetIP(const QString &webCode);

    // Get local IP
    static QString getLocalIP();

    // Convert URL address to IP address
    static QString urlToIP(const QString &url);

    // Check if internet access is available
    static bool isWebOk();

    // Show message box
    static void showMessageBoxInfo(const QString &info, int closeSec = 0);
    // Show error box
    static void showMessageBoxError(const QString &info, int closeSec = 0);
    // Show question box
    static int showMessageBoxQuestion(const QString &info);

    // Show input box
    static QString showInputBox(bool &ok,
                                const QString &title,
                                int type = 0,
                                int closeSec = 0,
                                QString defaultValue = QString(),
                                bool pwd = false);

    // Global static methods collection end--------------------------------

public:
    explicit QUIWidget(QWidget *parent = 0);
    ~QUIWidget();

    void createTrayMenu();

protected:
    // bool eventFilter(QObject *obj, QEvent *evt);

private:
    QVBoxLayout *verticalLayout1;
    QWidget *widgetMain;
    QVBoxLayout *verticalLayout2;
    QWidget *widgetTitle;
    QHBoxLayout *horizontalLayout4;
    QLabel *labIco;
    QLabel *labTitle;
    QWidget *widgetMenu;
    QHBoxLayout *horizontalLayout;
    QToolButton *btnMenu;
    QPushButton *btnMenu_Min;
    QPushButton *btnMenu_Max;
    QPushButton *btnMenu_Close;
    QWidget *widget;
    QVBoxLayout *verticalLayout3;

    QSystemTrayIcon *m_systemTray;
    QMenu *trayMenu;
    QAction *quitAction;
    QAction *restoreWinAction;

private:
    bool max;        // Whether in maximized state
    QRect location;  // Coordinate position after moving window with mouse

    QString title;            // Title
    Qt::Alignment alignment;  // Title text alignment
    bool minHide;             // Minimize and hide
    MainWindow *mainWidget;   // Main window object

public:
    QLabel *getLabIco() const;
    QLabel *getLabTitle() const;
    QToolButton *getBtnMenu() const;
    QPushButton *getBtnMenuMin() const;
    QPushButton *getBtnMenuMax() const;
    QPushButton *getBtnMenuMClose() const;

    Style getStyle() const;
    QString getTitle() const;
    Qt::Alignment getAlignment() const;

    QSize sizeHint() const;
    QSize minimumSizeHint() const;

private:
    void closeEvent(QCloseEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void moveEvent(QMoveEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    bool eventFilter(QObject *obj, QEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private slots:
    void initControl();  // Initialize controls
    void initForm();     // Initialize form
    void changeStyle();  // Change style

    void on_btnMenu_Min_clicked();
    void on_btnMenu_Max_clicked();
    void on_btnMenu_Close_clicked();

    ////Minimize and hide interface
    // void changeEvent(QEvent *event) override {
    //	if (event->type()==QEvent::WindowStateChange &&
    //		this->windowState() == Qt::WindowMinimized) {
    //		m_systemTray->showMessage("Information",//Message window title
    //			"ACloudViewer",//Message content
    //			QSystemTrayIcon::MessageIcon::Information,//Message window icon
    //			5000);//Message window display duration
    //	}
    //	QDialog::changeEvent(event);
    // };

    // Restore application window
    void showWindow() {
        // this->widget->show();
        this->setWindowState((this->windowState() & ~Qt::WindowMinimized) |
                             Qt::WindowActive);
    };

    void activeTray(QSystemTrayIcon::ActivationReason reason);

public slots:
    //! toggles full screen
    void toggleFullScreen(bool state);

public Q_SLOTS:
    // Set widget icon
    void setIcon(QUIWidget::Widget widget, QChar str, quint32 size = 9);
    void setIconMain(QChar str, quint32 size = 9);
    // Set widget image
    void setPixmap(QUIWidget::Widget widget,
                   const QString &file,
                   const QSize &size = QSize(32, 32));
    void setWindowLogo(const QString &icon);
    // Set widget visibility
    void setVisible(QUIWidget::Widget widget, bool visible = true);
    // Set only close button
    void setOnlyCloseBtn();

    // Set title bar height
    void setTitleHeight(int height);
    // Set unified button width
    void setBtnWidth(int width);

    // Set title and text style
    void setTitle(const QString &title);
    void setAlignment(Qt::Alignment alignment);

    // Set minimize and hide
    void setMinHide(bool minHide);

    // Set main window
    void setMainWidget(MainWindow *mainWidget);

Q_SIGNALS:
    void changeStyle(const QString &qssFile);
    void closing();
};

// Message box class
class QUIMessageBox : public QDialog {
    Q_OBJECT

public:
    explicit QUIMessageBox(QWidget *parent = 0);
    ~QUIMessageBox();

    static QUIMessageBox *Instance() {
        static QMutex mutex;

        if (!self) {
            QMutexLocker locker(&mutex);

            if (!self) {
                self = new QUIMessageBox;
            }
        }

        return self;
    }

protected:
    void closeEvent(QCloseEvent *);
    bool eventFilter(QObject *obj, QEvent *evt);

private:
    static QUIMessageBox *self;

    QVBoxLayout *verticalLayout1;
    QWidget *widgetTitle;
    QHBoxLayout *horizontalLayout3;
    QLabel *labIco;
    QLabel *labTitle;
    QLabel *labTime;
    QWidget *widgetMenu;
    QHBoxLayout *horizontalLayout4;
    QPushButton *btnMenu_Close;
    QWidget *widgetMain;
    QVBoxLayout *verticalLayout2;
    QFrame *frame;
    QVBoxLayout *verticalLayout4;
    QHBoxLayout *horizontalLayout1;
    QLabel *labIcoMain;
    QSpacerItem *horizontalSpacer1;
    QLabel *labInfo;
    QHBoxLayout *horizontalLayout2;
    QSpacerItem *horizontalSpacer2;
    QPushButton *btnOk;
    QPushButton *btnCancel;

private:
    int closeSec;    // Total display time
    int currentSec;  // Current displayed time

private slots:
    void initControl();  // Initialize controls
    void initForm();     // Initialize form
    void checkSec();     // Check countdown

private slots:
    void on_btnOk_clicked();
    void on_btnMenu_Close_clicked();

public Q_SLOTS:
    void setIconMain(QChar str, quint32 size = 9);
    void setMessage(const QString &msg, int type, int closeSec = 0);
};

// Input box class
class QUIInputBox : public QDialog {
    Q_OBJECT

public:
    explicit QUIInputBox(QWidget *parent = 0);
    ~QUIInputBox();

    static QUIInputBox *Instance() {
        static QMutex mutex;

        if (!self) {
            QMutexLocker locker(&mutex);

            if (!self) {
                self = new QUIInputBox;
            }
        }

        return self;
    }

protected:
    void closeEvent(QCloseEvent *);
    bool eventFilter(QObject *obj, QEvent *evt);

private:
    static QUIInputBox *self;

    QVBoxLayout *verticalLayout1;
    QWidget *widgetTitle;
    QHBoxLayout *horizontalLayout1;
    QLabel *labIco;
    QLabel *labTitle;
    QLabel *labTime;
    QWidget *widgetMenu;
    QHBoxLayout *horizontalLayout2;
    QPushButton *btnMenu_Close;
    QWidget *widgetMain;
    QVBoxLayout *verticalLayout2;
    QFrame *frame;
    QVBoxLayout *verticalLayout3;
    QLabel *labInfo;
    QLineEdit *txtValue;
    QComboBox *cboxValue;
    QHBoxLayout *lay;
    QSpacerItem *horizontalSpacer;
    QPushButton *btnOk;
    QPushButton *btnCancel;

private:
    int closeSec;    // Total display time
    int currentSec;  // Current displayed time
    QString value;   // Current value

private slots:
    void initControl();  // Initialize controls
    void initForm();     // Initialize form
    void checkSec();     // Check countdown

private slots:
    void on_btnOk_clicked();
    void on_btnMenu_Close_clicked();

public:
    QString getValue() const;

public Q_SLOTS:
    void setIconMain(QChar str, quint32 size = 9);
    void setParameter(const QString &title,
                      int type = 0,
                      int closeSec = 0,
                      QString defaultValue = QString(),
                      bool pwd = false);
};

// Icon font processing class
class IconHelper : public QObject {
    Q_OBJECT

public:
    explicit IconHelper(QObject *parent = 0);
    static IconHelper *Instance() {
        static QMutex mutex;

        if (!self) {
            QMutexLocker locker(&mutex);

            if (!self) {
                self = new IconHelper;
            }
        }

        return self;
    }

    void setIcon(QLabel *lab, QChar c, quint32 size = 9);
    void setIcon(QAbstractButton *btn, QChar c, quint32 size = 9);
    QPixmap getPixmap(const QString &color,
                      QChar c,
                      quint32 size = 9,
                      quint32 pixWidth = 10,
                      quint32 pixHeight = 10);

    // Get corresponding icon for button
    QPixmap getPixmap(QToolButton *btn, bool normal);

    // Set navigation panel style without icons
    static void setStyle(QWidget *widget,
                         const QString &type = "left",
                         int borderWidth = 3,
                         const QString &borderColor = "#029FEA",
                         const QString &normalBgColor = "#292F38",
                         const QString &darkBgColor = "#1D2025",
                         const QString &normalTextColor = "#54626F",
                         const QString &darkTextColor = "#FDFDFD");

    // Set navigation panel style with icons and effect switching
    void setStyle(QWidget *widget,
                  QList<QToolButton *> btns,
                  QList<int> pixChar,
                  quint32 iconSize = 9,
                  quint32 iconWidth = 10,
                  quint32 iconHeight = 10,
                  const QString &type = "left",
                  int borderWidth = 3,
                  const QString &borderColor = "#029FEA",
                  const QString &normalBgColor = "#292F38",
                  const QString &darkBgColor = "#1D2025",
                  const QString &normalTextColor = "#54626F",
                  const QString &darkTextColor = "#FDFDFD");

    // Set navigation button style with icons and effect switching
    void setStyle(QFrame *frame,
                  QList<QToolButton *> btns,
                  QList<int> pixChar,
                  quint32 iconSize = 9,
                  quint32 iconWidth = 10,
                  quint32 iconHeight = 10,
                  const QString &normalBgColor = "#2FC5A2",
                  const QString &darkBgColor = "#3EA7E9",
                  const QString &normalTextColor = "#EEEEEE",
                  const QString &darkTextColor = "#FFFFFF");

protected:
    bool eventFilter(QObject *watched, QEvent *event);

private:
    static IconHelper *self;    // Object itself
    QFont iconFont;             // Icon font
    QList<QToolButton *> btns;  // Button queue
    QList<QPixmap> pixNormal;   // Normal image queue
    QList<QPixmap> pixDark;     // Darkened image queue
};

// Global variable control
class QUIConfig {
public:
    // Global icons
    static QChar IconMain;    // Top-left icon in title bar
    static QChar IconMenu;    // Dropdown menu icon
    static QChar IconMin;     // Minimize icon
    static QChar IconMax;     // Maximize icon
    static QChar IconNormal;  // Restore icon
    static QChar IconClose;   // Close icon

    static QString FontName;    // Global font name
    static int FontSize;        // Global font size
    static QString ConfigFile;  // Config file path and name

    // Stylesheet color values
    static QString TextColor;         // Text color
    static QString PanelColor;       // Panel color
    static QString BorderColor;      // Border color
    static QString NormalColorStart;  // Normal state start color
    static QString NormalColorEnd;    // Normal state end color
    static QString DarkColorStart;   // Darkened state start color
    static QString DarkColorEnd;     // Darkened state end color
    static QString HighColor;        // Highlight color

    static void ReadConfig();   // Read config file, called at the beginning of main function when loading application
    static void WriteConfig();  // Write config file, called when closing application after changing config
    static void NewConfig();    // Create new config file with initial values
    static bool CheckConfig();  // Validate config file
};

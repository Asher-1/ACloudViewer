"""Exercise the configured CVAppCommon owner and its actual Custom backend.

This is not an isolated style-class test: rebuild CVAppCommon first. Native
Fusion must have no global QSS/custom style; Custom must use Qlementine, with no
legacy QSS overlay. All Qt diagnostics are retained and categorized.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path.home() / 'ACloudViewer-icon-reviews/qt-components'
CPP = r'''#include "ecvApplicationBase.h"
#include "ecvAICoreUiHelper.h"
#include <QActionGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDockWidget>
#include <QDateTimeEdit>
#include <QFontComboBox>
#include <QCalendarWidget>
#include <QTextEdit>
#include <QDialogButtonBox>
#include <QDialog>
#include <QFile>
#include <QGroupBox>
#include <QHeaderView>
#include <QLabel>
#include <QMainWindow>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QMenuBar>
#include <QStyleOptionTitleBar>
#include <QMap>
#include <QMenu>
#include <QPlainTextEdit>
#include <QPainter>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QScrollBar>
#include <QScreen>
#include <QSpinBox>
#include <QSplitter>
#include <QStyleFactory>
#include <QWindow>
#include <QStyleOptionSpinBox>
#include <QStyleOptionToolButton>
#include <QStyledItemDelegate>
#include <QTabBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QLineEdit>
#include <QToolBar>
#include <QToolButton>
#include <QTimer>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QtTest/QTest>
#include <cmath>
#include <cstdio>
#include <iostream>

static QMap<QString,int> warnings;
static QString category(const QString& s) {
    if(s.contains("Could not parse",Qt::CaseInsensitive))return "stylesheet-parse";
    if(s.contains("Unknown property",Qt::CaseInsensitive))return "unsupported-property";
    if(s.contains("regular expression",Qt::CaseInsensitive))return "invalid-regex";
    if(s.contains("pixmap",Qt::CaseInsensitive)||s.contains("image",Qt::CaseInsensitive)||s.contains("resource",Qt::CaseInsensitive))return "image-resource";
    if(s.contains("This plugin does not support propagateSizeHints")||s.contains("This plugin does not support raise")||s.contains("This plugin does not support grabbing the keyboard")||s.contains("This plugin does not support setting window masks"))return "offscreen-platform";
    return "other";
}
static void messages(QtMsgType type,const QMessageLogContext&,const QString& s){
    const QString c=category(s);
    if(type==QtWarningMsg||type==QtCriticalMsg||type==QtFatalMsg||c!="other")++warnings[c];
    fprintf(stderr,"QT_MESSAGE[%s]: %s\n",qPrintable(c),qPrintable(s));
    if(type==QtFatalMsg)abort();
}
static bool healthy(const char* phase){
    for(auto i=warnings.cbegin();i!=warnings.cend();++i)fprintf(stderr,"QT_WARNING_SUMMARY[%s][%s]=%d\n",phase,qPrintable(i.key()),i.value());
    return !warnings.value("stylesheet-parse")&&!warnings.value("unsupported-property")&&!warnings.value("invalid-regex")&&!warnings.value("image-resource");
}
#define CHECK(x) do{if(!(x)){fprintf(stderr,"ASSERTION_FAILED line=%d: %s\n",__LINE__,#x);return 2;}}while(0)
class ApplicationDelegate : public QStyledItemDelegate { public: explicit ApplicationDelegate(QObject* parent):QStyledItemDelegate(parent){} };
static bool isQlementine(QStyle* s){return s&&QString::fromLatin1(s->metaObject()->className()).contains("Qlementine",Qt::CaseInsensitive);}
static QImage iconImage(QIcon icon,QIcon::Mode mode){QImage im(48,48,QImage::Format_ARGB32_Premultiplied);im.fill(Qt::transparent);QPainter p(&im);p.setRenderHint(QPainter::Antialiasing);icon.paint(&p,QRect(0,0,48,48),Qt::AlignCenter,mode);return im;}
static bool hasInk(const QImage& im){for(int y=0;y<im.height();++y)for(int x=0;x<im.width();++x)if(qAlpha(im.pixel(x,y)))return true;return false;}
static void settle(){qApp->processEvents();QTest::qWait(220);qApp->processEvents();}
static bool capture(QWidget& w,const QString& path){const auto before=w.size();const auto pm=w.grab();const qreal dpr=w.devicePixelRatioF();if(qAbs(pm.width()-w.width()*dpr)>1||qAbs(pm.height()-w.height()*dpr)>1){fprintf(stderr,"CAPTURE_DPR_MISMATCH %s before=%dx%d after=%dx%d pixels=%dx%d dpr=%g pixmapDpr=%g\n",qPrintable(path),before.width(),before.height(),w.width(),w.height(),pm.width(),pm.height(),dpr,pm.devicePixelRatioF());return false;}return pm.save(path);}

int main(int argc,char**argv){
#if QT_VERSION < QT_VERSION_CHECK(6,0,0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif
    ecvApplicationBase app(argc,argv,false,QStringLiteral("ComponentOwnerProbe"));
    if(argc!=5)return 1;
    const QString theme=argv[1],themeId=argv[2],mode=argv[3],out=argv[4];
    qInstallMessageHandler(messages);
    auto requested=app.appearance();requested.styleKey=mode=="custom"?"Custom":"Fusion";requested.themeId=themeId;
    CHECK(app.applyAppearance(requested,false));CHECK(app.styleSheet().isEmpty());
    CHECK(app.isCustomAppearance()==(mode=="custom"));CHECK(isQlementine(app.style())==(mode=="custom"));CHECK(healthy("apply"));
    const auto disabledHighlight=app.palette().color(QPalette::Disabled,QPalette::Highlight);
    const auto disabledSelectionText=app.palette().color(QPalette::Disabled,QPalette::HighlightedText);
    CHECK(disabledHighlight!=disabledSelectionText);
    const bool wasDark=app.palette().color(QPalette::Window).lightnessF()<.48;

    QMainWindow window;window.setWindowTitle("ACloudViewer actual Custom backend / Qt widgets");
    auto* toolbar=window.addToolBar("Actions");toolbar->setObjectName("probeToolBar");
    toolbar->addAction(app.style()->standardIcon(QStyle::SP_FileDialogNewFolder),"New folder");toolbar->addSeparator();
    toolbar->addAction(app.style()->standardIcon(QStyle::SP_DialogSaveButton),"Save");
    for(int i=0;i<12;++i)toolbar->addAction(QString("Action %1").arg(i+1));
    auto* vertical=new QToolBar("Vertical",&window);vertical->setObjectName("probeVertical");window.addToolBar(Qt::LeftToolBarArea,vertical);vertical->addAction(app.style()->standardIcon(QStyle::SP_BrowserReload),"Reload");
    auto* densityBar=new QToolBar(&window);densityBar->setIconSize(QSize(24,24));window.addToolBar(densityBar);auto* densityAction=densityBar->addAction(app.style()->standardIcon(QStyle::SP_DialogOpenButton),"Density reference");auto* densityButton=qobject_cast<QToolButton*>(densityBar->widgetForAction(densityAction));CHECK(densityButton);densityButton->ensurePolished();CHECK(densityButton->sizeHint().width()<=32);CHECK(app.style()->pixelMetric(QStyle::PM_ToolBarItemSpacing)==1);CHECK(app.style()->pixelMetric(QStyle::PM_ToolBarItemMargin)==2);
    QMenu splitPopup;splitPopup.addAction("Split-menu test");auto* splitAction=densityBar->addAction(app.style()->standardIcon(QStyle::SP_DialogOpenButton),"Split reference");splitAction->setMenu(&splitPopup);auto* splitButton=qobject_cast<QToolButton*>(densityBar->widgetForAction(splitAction));CHECK(splitButton);splitButton->setPopupMode(QToolButton::MenuButtonPopup);
    auto* central=new QWidget;window.setCentralWidget(central);auto* layout=new QVBoxLayout(central);
    auto* check=new QCheckBox("Visible cloud");auto* checked=new QCheckBox("Checked reference");checked->setChecked(true);
    auto* partial=new QCheckBox("Partial selection");partial->setTristate(true);partial->setCheckState(Qt::PartiallyChecked);
    auto* checks=new QHBoxLayout;checks->addWidget(check);checks->addWidget(checked);checks->addWidget(partial);layout->addLayout(checks);
    auto* radio=new QRadioButton("Point cloud");radio->setChecked(true);layout->addWidget(radio);
    auto* combo=new QComboBox;combo->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Fixed);combo->setMinimumContentsLength(8);combo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    for(int i=0;i<20;++i)combo->addItem(QString("OpenPCDet nuScenes BEVFusion (LiDAR) - F16 - long model checkpoint %1").arg(i+1));
    layout->addWidget(combo);
    QComboBox customCombo(&window);customCombo.hide();auto* applicationDelegate=new ApplicationDelegate(&customCombo);customCombo.setItemDelegate(applicationDelegate);customCombo.addItem("Application-owned delegate without Q_OBJECT");customCombo.ensurePolished();CHECK(customCombo.itemDelegate()==applicationDelegate);
    auto* spin=new QSpinBox;spin->setRange(0,10);spin->setValue(1);layout->addWidget(spin);
    auto* scroll=new QScrollBar(Qt::Horizontal);scroll->setRange(0,100);scroll->setValue(40);layout->addWidget(scroll);
    auto* tabs=new QTabWidget;tabs->setTabsClosable(true);tabs->setMovable(true);for(int i=0;i<9;++i)tabs->addTab(new QLabel(QString("View %1").arg(i+1)),QString("Reconstruction view %1").arg(i+1));layout->addWidget(tabs);
    auto* splitter=new QSplitter;auto* tree=new QTreeWidget;tree->setHeaderLabels({"Scene hierarchy"});auto* branch=new QTreeWidgetItem(tree,{"Cloud group"});new QTreeWidgetItem(branch,{"Point cloud"});auto* second=new QTreeWidgetItem(tree,{"Mesh group"});new QTreeWidgetItem(second,{"Triangle mesh"});splitter->addWidget(tree);
    auto* table=new QTableWidget(3,2);table->setHorizontalHeaderLabels({"Name","Value"});for(int i=0;i<3;++i){table->setItem(i,0,new QTableWidgetItem(QString("Object %1").arg(3-i)));table->setItem(i,1,new QTableWidgetItem(QString::number(i)));}table->setSortingEnabled(true);table->sortItems(0);splitter->addWidget(table);layout->addWidget(splitter);
    auto* group=new QGroupBox("Model options");group->setCheckable(true);auto* groupLayout=new QHBoxLayout(group);groupLayout->addWidget(new QCheckBox("Export cloud"));groupLayout->addWidget(ecvAICoreUi::makeSampleDataBtn(group));layout->addWidget(group);
    auto* progress=new QProgressBar;progress->setValue(64);layout->addWidget(progress);
    auto* menuButton=new QToolButton;auto* menu=new QMenu(menuButton);auto* action=menu->addAction("Show labels");action->setCheckable(true);action->setChecked(true);auto* exclusiveGroup=new QActionGroup(menu);exclusiveGroup->setExclusive(true);auto* exclusive=menu->addAction("Exclusive option");exclusive->setCheckable(true);exclusive->setChecked(true);exclusiveGroup->addAction(exclusive);menu->addMenu("Nested tools")->addAction("Sub action");menuButton->setMenu(menu);menuButton->setText("Tools");menuButton->setPopupMode(QToolButton::MenuButtonPopup);layout->addWidget(menuButton);
    auto* dock=new QDockWidget("Properties",&window);dock->setObjectName("probeProperties");dock->setWidget(new QLabel("Property panel"));window.addDockWidget(Qt::RightDockWidgetArea,dock);
    auto* otherDock=new QDockWidget("Information",&window);otherDock->setObjectName("probeInformation");otherDock->setWidget(new QLabel("Information panel"));window.addDockWidget(Qt::RightDockWidgetArea,otherDock);window.tabifyDockWidget(dock,otherDock);dock->raise();
    window.resize(1100,850);window.show();settle();CHECK(capture(window,out+"-widgets.png"));CHECK(healthy("shown"));

    QStyleOptionToolBar handleOption;handleOption.initFrom(toolbar);handleOption.rect=toolbar->rect();handleOption.state|=QStyle::State_Horizontal;handleOption.features=QStyleOptionToolBar::Movable;const auto handleRect=toolbar->style()->subElementRect(QStyle::SE_ToolBarHandle,&handleOption,toolbar);CHECK(toolbar->isMovable());CHECK(!handleRect.isEmpty());
    // At natural size a labelled toolbutton must paint the same caption as
    // when extra width is available (compare nontransparent ink, not a mock).
    if(mode=="custom") {
        QToolBar logBar; logBar.setIconSize(QSize(18,18)); logBar.setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        for(const auto* caption:{"Refresh","Open Folder","Export","Delete","Clean Old","Close","Open &Recent Projects"}) {
            auto* action=logBar.addAction(app.style()->standardIcon(QStyle::SP_DirOpenIcon),caption);
            auto* button=qobject_cast<QToolButton*>(logBar.widgetForAction(action)); CHECK(button);
            QStyleOptionToolButton opt; opt.initFrom(button); opt.text=action->text();opt.icon=action->icon();opt.iconSize=logBar.iconSize();opt.toolButtonStyle=Qt::ToolButtonTextBesideIcon;
            auto ink=[&](int extra) { QImage image(button->sizeHint()+QSize(extra,0),QImage::Format_ARGB32_Premultiplied);image.fill(Qt::transparent); opt.rect=image.rect(); QPainter painter(&image);painter.setFont(button->font());button->style()->drawControl(QStyle::CE_ToolButtonLabel,&opt,&painter,button); painter.end();int count=0;for(int y=0;y<image.height();++y)for(int x=0;x<image.width();++x)if(qAlpha(image.pixel(x,y)))++count;return count; };
            const int normalInk=ink(0); CHECK(normalInk>0&&normalInk==ink(100));
        }
        logBar.resize(1040,logBar.sizeHint().height());logBar.show();settle();CHECK(capture(logBar,out+"-labelled-toolbar.png"));logBar.close();
        QTabWidget verticalTabs;verticalTabs.setTabPosition(QTabWidget::West);verticalTabs.addTab(new QLabel("Vertical page"),"Project Browser");verticalTabs.addTab(new QLabel("Second page"),"Script Output Console");verticalTabs.resize(420,420);verticalTabs.show();settle();auto* verticalBar=verticalTabs.tabBar();for(int i=0;i<verticalBar->count();++i){CHECK(verticalBar->tabRect(i).height()>verticalBar->fontMetrics().horizontalAdvance(verticalBar->tabText(i)));}CHECK(capture(verticalTabs,out+"-vertical-tabs.png"));verticalTabs.close();
    }
    // Explicit XYZ multiline data must retain three readable lines.
    QTableWidget geometryTable(2,2);geometryTable.resize(500,300);geometryTable.setColumnWidth(0,170);geometryTable.setColumnWidth(1,260);auto* multiData=new QTableWidgetItem("X: 1.000\nY: 2.000\nZ: 3.000");geometryTable.setItem(0,0,new QTableWidgetItem("Bounding-box dimensions"));geometryTable.setItem(0,1,multiData);auto* emptyData=new QTableWidgetItem(QString());geometryTable.setItem(1,1,emptyData);geometryTable.show();geometryTable.openPersistentEditor(emptyData);geometryTable.resizeRowsToContents();settle();const auto multiCell=geometryTable.visualItemRect(multiData);CHECK(multiCell.height()>=geometryTable.fontMetrics().lineSpacing()*3);auto* emptyEditor=geometryTable.indexWidget(geometryTable.model()->index(1,1));CHECK(emptyEditor);const auto emptyCell=geometryTable.visualItemRect(emptyData);CHECK(emptyEditor->geometry().height()>0&&emptyEditor->geometry().intersects(emptyCell));CHECK(emptyEditor->geometry().top()>=emptyCell.top());CHECK(capture(geometryTable,out+"-multiline-empty-editor.png"));geometryTable.close();
    // Real mouse/keyboard events exercise controls rather than setting values.
    QStyleOptionToolButton splitOption;splitOption.initFrom(splitButton);splitOption.rect=splitButton->rect();splitOption.iconSize=splitButton->iconSize();splitOption.features=QStyleOptionToolButton::HasMenu|QStyleOptionToolButton::MenuButtonPopup;const auto splitRect=splitButton->style()->subControlRect(QStyle::CC_ToolButton,&splitOption,QStyle::SC_ToolButtonMenu,splitButton);CHECK(!splitRect.isEmpty());CHECK(splitButton->style()->hitTestComplexControl(QStyle::CC_ToolButton,&splitOption,splitRect.center(),splitButton)==QStyle::SC_ToolButtonMenu);bool splitOpened=false;QTimer::singleShot(120,&splitPopup,[&](){splitOpened=splitPopup.isVisible();splitPopup.hide();});QTest::mouseClick(splitButton,Qt::LeftButton,Qt::NoModifier,splitRect.center());settle();CHECK(splitOpened);
    QTest::mouseClick(check,Qt::LeftButton,Qt::NoModifier,QPoint(10,check->height()/2));settle();CHECK(check->isChecked());
    QTest::mouseClick(check,Qt::LeftButton,Qt::NoModifier,QPoint(10,check->height()/2));settle();CHECK(!check->isChecked());
    QTest::mouseMove(checked,QPoint(10,checked->height()/2));settle();CHECK(capture(window,out+"-hover.png"));
    checked->setEnabled(false);settle();CHECK(capture(window,out+"-disabled.png"));checked->setEnabled(true);
    QTest::mouseClick(combo,Qt::LeftButton,Qt::NoModifier,QPoint(combo->width()-8,combo->height()/2));settle();CHECK(combo->view()->isVisible());CHECK(capture(*combo->view()->window(),out+"-combo-popup.png"));QTest::keyClick(combo->view(),Qt::Key_Down);QTest::keyClick(combo->view(),Qt::Key_Return);settle();CHECK(combo->currentIndex()==1);
    QStyleOptionSpinBox spinOption;spinOption.initFrom(spin);spinOption.rect=spin->rect();spinOption.frame=true;spinOption.buttonSymbols=spin->buttonSymbols();spinOption.stepEnabled=QAbstractSpinBox::StepUpEnabled|QAbstractSpinBox::StepDownEnabled;
    QRect up=spin->style()->subControlRect(QStyle::CC_SpinBox,&spinOption,QStyle::SC_SpinBoxUp,spin);CHECK(!up.isEmpty());QTest::mouseClick(spin,Qt::LeftButton,Qt::NoModifier,up.center());settle();CHECK(spin->value()==2);
    QRect branchRect=tree->visualItemRect(branch);CHECK(!branchRect.isEmpty());QTest::mouseClick(tree->viewport(),Qt::LeftButton,Qt::NoModifier,QPoint(branchRect.left()-tree->indentation()/2,branchRect.center().y()));settle();CHECK(branch->isExpanded());
    menu->popup(QGuiApplication::primaryScreen()->availableGeometry().center());settle();CHECK(capture(*menu,out+"-menu.png"));QTest::mouseClick(menu,Qt::LeftButton,Qt::NoModifier,menu->actionGeometry(action).center());settle();CHECK(!action->isChecked());
    int closeRequest=-1;QObject::connect(tabs,&QTabWidget::tabCloseRequested,[&](int i){closeRequest=i;tabs->removeTab(i);});QTabBar* tabbar=tabs->tabBar();QWidget* closeButton=tabbar->tabButton(0,QTabBar::RightSide);if(!closeButton)closeButton=tabbar->tabButton(0,QTabBar::LeftSide);CHECK(closeButton);QTest::mouseClick(closeButton,Qt::LeftButton);settle();CHECK(closeRequest==0&&tabs->count()==8);
    dock->setFloating(true);settle();CHECK(dock->isFloating());CHECK(capture(*dock,out+"-floating-dock.png"));dock->close();settle();CHECK(!dock->isVisible());CHECK(healthy("input-states"));

    QMainWindow mdiHost;mdiHost.setWindowTitle("Python editor MDI contract");mdiHost.resize(720,480);mdiHost.menuBar()->addMenu("Window");auto* mdiArea=new QMdiArea(&mdiHost);mdiHost.setCentralWidget(mdiArea);auto* mdiChild=mdiArea->addSubWindow(new QPlainTextEdit);mdiChild->setWindowTitle("untitled.py");mdiChild->resize(400,270);mdiHost.show();mdiChild->show();settle();QStyleOptionTitleBar titleOpt;titleOpt.initFrom(mdiChild);titleOpt.rect=mdiChild->rect();titleOpt.rect.setHeight(mdiChild->style()->pixelMetric(QStyle::PM_TitleBarHeight,&titleOpt,mdiChild));titleOpt.titleBarFlags=mdiChild->windowFlags();titleOpt.titleBarState=int(mdiChild->windowState());titleOpt.subControls=QStyle::SC_All;for(auto sc:{QStyle::SC_TitleBarMinButton,QStyle::SC_TitleBarMaxButton,QStyle::SC_TitleBarCloseButton,QStyle::SC_TitleBarLabel}){const auto rect=mdiChild->style()->subControlRect(QStyle::CC_TitleBar,&titleOpt,sc,mdiChild);CHECK(!rect.isEmpty());CHECK(mdiChild->style()->hitTestComplexControl(QStyle::CC_TitleBar,&titleOpt,rect.center(),mdiChild)==sc||sc==QStyle::SC_TitleBarLabel);}CHECK(capture(mdiHost,out+"-python-mdi.png"));mdiHost.close();
    // A disabled standard icon must be visible but not pixel-identical.
    const auto normalIcon=app.style()->standardIcon(QStyle::SP_DialogSaveButton);
    const auto normal=iconImage(normalIcon,QIcon::Normal),disabled=iconImage(normalIcon,QIcon::Disabled);
    CHECK(hasInk(normal)&&hasInk(disabled));CHECK(normal!=disabled);normal.save(out+"-icon-normal.png");disabled.save(out+"-icon-disabled.png");
    if(mode=="custom"){
        QDialog extra(&window);extra.setWindowTitle("Calendar, date, font and rich text coverage");auto* extraLayout=new QVBoxLayout(&extra);auto* date=new QDateTimeEdit(QDateTime(QDate(2026,11,22),QTime(12,0)),&extra);date->setCalendarPopup(true);extraLayout->addWidget(date);auto* fonts=new QFontComboBox(&extra);extraLayout->addWidget(fonts);auto* calendar=new QCalendarWidget(&extra);calendar->setSelectedDate(QDate(2026,11,22));extraLayout->addWidget(calendar);auto* rich=new QTextEdit(&extra);rich->setHtml("<b>Rich text</b> selection and local content colors");extraLayout->addWidget(rich);auto* dialogButtons=new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel,&extra);extraLayout->addWidget(dialogButtons);extra.show();settle();CHECK(capture(extra,out+"-extended-controls.png"));date->setFocus();date->setCurrentSection(QDateTimeEdit::DaySection);QTest::keyClick(date,Qt::Key_Up);settle();CHECK(date->date()==QDate(2026,11,23));auto* calendarView=calendar->findChild<QAbstractItemView*>();CHECK(calendarView);calendarView->setFocus();QTest::keyClick(calendarView,Qt::Key_Right);settle();CHECK(calendar->selectedDate()==QDate(2026,11,23));extra.close();CHECK(healthy("extended-controls"));
        CHECK(app.style()->setProperty("animationsEnabled",false));settle();CHECK(!app.style()->property("animationsEnabled").toBool());
        CHECK(app.style()->setProperty("animationsEnabled",true));settle();CHECK(app.style()->property("animationsEnabled").toBool());
        auto opposite=requested;opposite.themeId=wasDark?"QDarkStyleSheet::Light":"QDarkStyleSheet::Dark";
        CHECK(app.applyAppearance(opposite,false));settle();CHECK(isQlementine(app.style()));CHECK(app.styleSheet().isEmpty());CHECK((app.palette().color(QPalette::Window).lightnessF()<.48)!=wasDark);CHECK(capture(window,out+"-opposite-theme.png"));
        CHECK(app.applyAppearance(requested,false));settle();CHECK(app.appearance().themeId==requested.themeId);CHECK((app.palette().color(QPalette::Window).lightnessF()<.48)==wasDark);CHECK(healthy("theme-restored"));
    }
    // Exercise exits with the same already-open window and a newly-created one.
    for(int i=0;i<3;++i){auto native=requested;native.styleKey="Fusion";CHECK(app.applyAppearance(native,false));settle();CHECK(!app.isCustomAppearance()&&!isQlementine(app.style())&&app.styleSheet().isEmpty());if(i<2){CHECK(app.applyAppearance({"Custom",themeId},false));settle();CHECK(isQlementine(app.style())&&app.styleSheet().isEmpty());}}
    CHECK(customCombo.itemDelegate()==applicationDelegate);
    auto nativeState=requested;nativeState.styleKey="Fusion";
    CHECK(app.applyAppearance({"Custom",themeId},false));menu->popup(QGuiApplication::primaryScreen()->availableGeometry().center());CHECK(app.applyAppearance(nativeState,false));settle();CHECK(!isQlementine(app.style()));menu->hide();
    CHECK(app.applyAppearance({"Custom",themeId},false));settle();menu->popup(QGuiApplication::primaryScreen()->availableGeometry().center());settle();const bool checkedBeforeCancel=action->isChecked();QTest::mouseClick(menu,Qt::LeftButton,Qt::NoModifier,menu->actionGeometry(action).center());CHECK(action->property("qlementine_flashing").toBool());CHECK(app.applyAppearance(nativeState,false));settle();CHECK(!menu->signalsBlocked());CHECK(!action->property("qlementine_flashing").toBool());CHECK(action->isChecked()==checkedBeforeCancel);menu->hide();
    CHECK(app.applyAppearance({"Custom",themeId},false));settle();bool switchedInsideAction=false;auto* switchAction=menu->addAction("Switch Native from action");QObject::connect(switchAction,&QAction::triggered,&window,[&](){switchedInsideAction=app.applyAppearance(nativeState,false);});menu->popup(QGuiApplication::primaryScreen()->availableGeometry().center());settle();QTest::mouseClick(menu,Qt::LeftButton,Qt::NoModifier,menu->actionGeometry(switchAction).center());settle();CHECK(switchedInsideAction&&!isQlementine(app.style()));CHECK(!menu->signalsBlocked());CHECK(customCombo.itemDelegate()==applicationDelegate);
    QMenu existingNativeMenu;existingNativeMenu.addAction("Existing native popup");existingNativeMenu.winId();const bool nativeHadNoAlpha=existingNativeMenu.windowHandle()->format().alphaBufferSize()<=0;
    CHECK(app.applyAppearance({"Custom",themeId},false));existingNativeMenu.ensurePolished();if(nativeHadNoAlpha){CHECK(app.style()->pixelMetric(QStyle::PM_MenuPanelWidth,nullptr,&existingNativeMenu)==0);existingNativeMenu.popup(QGuiApplication::primaryScreen()->availableGeometry().center());settle();const auto raster=existingNativeMenu.grab().toImage();CHECK(!raster.isNull());CHECK(qAlpha(raster.pixel(0,raster.height()/2))==255);CHECK(capture(existingNativeMenu,out+"-opaque-native-menu.png"));existingNativeMenu.hide();}CHECK(app.applyAppearance(nativeState,false));settle();
    QComboBox fresh;fresh.addItems({"Native first","Native second"});fresh.show();settle();CHECK(!isQlementine(fresh.style()));CHECK(capture(window,out+"-native-exit.png"));CHECK(healthy("native-exit"));
    std::cout<<mode.toStdString()<<" "<<theme.toStdString()<<" PASS Qt="<<qVersion()<<" DPR="<<window.devicePixelRatioF()<<" backend-owner; mouse checkbox/combo/spin/tree/menu/tab; floating dock; disabled icon; theme switch/Native exit; warnings retained\n";
    return 0;
}
'''


def cache_values(path):
    values = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line and not line.startswith(
                ("#", "//")) and "=" in line and ":" in line.split("=", 1)[0]:
                name, value = line.split("=", 1)
                values[name.split(":", 1)[0]] = value
    return values


def configured_qt(build, explicit_prefix):
    cache = cache_values(build / 'CMakeCache.txt')
    major = 6 if cache.get('USE_QT6') == 'ON' else 5
    if explicit_prefix:
        prefix = explicit_prefix.resolve()
        installed = [
            n for n in dict.fromkeys((major, 6, 5))
            if any((prefix / lib / 'cmake' / f'Qt{n}').is_dir()
                   for lib in ('lib', 'lib64')) or
            any(prefix.glob(f'lib/*/cmake/Qt{n}/Qt{n}Config.cmake'))
        ]
        if not installed:
            raise RuntimeError(f'No Qt CMake package below {prefix}')
        major = installed[0]
    else:
        package = cache.get(f'Qt{major}_DIR') or cache.get(f'Qt{major}Core_DIR')
        if not package or package.endswith('-NOTFOUND'):
            raise RuntimeError(
                'Cannot discover configured Qt; pass --build or --qt-prefix')
        prefixes = [
            p for p in Path(package).resolve().parents
            if (p / 'include').is_dir() and any(
                (p / lib).is_dir() for lib in ('lib', 'lib64'))
        ]
        if not prefixes:
            raise RuntimeError('Cannot derive Qt prefix; pass --qt-prefix')
        prefix = prefixes[0]
    return major, prefix, cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build_app')
    parser.add_argument('--qt-prefix', type=Path)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument(
        '--theme',
        action='append',
        help='native, a bundled CSS stem, or QDarkStyleSheet::Light/Dark')
    args = parser.parse_args(argv)
    build, out = args.build.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    major, prefix, cache = configured_qt(build, args.qt_prefix)
    lib_dirs = [build / 'bin', build / 'bin/Release', build / 'lib']
    patterns = ('CVAppCommon.lib',) if os.name == 'nt' else (
        'libCVAppCommon.dylib',) if sys.platform == 'darwin' else (
            'libCVAppCommon.so',)
    library = next((d / name
                    for d in lib_dirs
                    for name in patterns
                    if (d / name).is_file()), None)
    if library is None:
        raise RuntimeError(
            f'Build CVAppCommon first; shared library not found below {build}')
    probe_dir = out / 'component-owner-source'
    probe_dir.mkdir(exist_ok=True)
    (probe_dir / 'component_probe.cpp').write_text(CPP)
    quoted = lambda p: '"' + str(p).replace('\\', '/') + '"'
    cmake = [
        'cmake_minimum_required(VERSION 3.16)',
        'project(ComponentOwnerProbe LANGUAGES CXX)',
        f'find_package(Qt{major} REQUIRED COMPONENTS Widgets Test)',
        'add_library(CVAppCommonProbe UNKNOWN IMPORTED)',
        'set_target_properties(CVAppCommonProbe PROPERTIES IMPORTED_LOCATION ' +
        quoted(library) + ')',
        'add_executable(component_probe component_probe.cpp)',
        'target_compile_features(component_probe PRIVATE cxx_std_17)',
        'target_include_directories(component_probe PRIVATE ' +
        quoted(ROOT / 'libs/CVAppCommon/include') + ' ' +
        quoted(ROOT / 'libs/CVPluginAPI/include') + ')',
        f'target_link_libraries(component_probe PRIVATE CVAppCommonProbe Qt{major}::Widgets Qt{major}::Test)',
        'set_target_properties(component_probe PROPERTIES BUILD_RPATH ' +
        quoted(library.parent) + ')',
        f'file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/qt-core-path.txt" CONTENT "$<TARGET_FILE:Qt{major}::Core>")',
        f'if(TARGET Qt{major}::qmake)',
        f'  get_target_property(_qmake Qt{major}::qmake IMPORTED_LOCATION)',
        '  execute_process(COMMAND "${_qmake}" -query QT_INSTALL_PLUGINS OUTPUT_VARIABLE _plugins OUTPUT_STRIP_TRAILING_WHITESPACE)',
        '  file(WRITE "${CMAKE_BINARY_DIR}/qt-plugin-path.txt" "${_plugins}")',
        'endif()',
    ]
    (probe_dir / 'CMakeLists.txt').write_text('\n'.join(cmake) + '\n')
    probe_build = out / 'component-owner-build'
    configure = [
        'cmake', '-S',
        str(probe_dir), '-B',
        str(probe_build), f'-DCMAKE_PREFIX_PATH={prefix}'
    ]
    if cache.get('CMAKE_CXX_COMPILER'):
        configure.append('-DCMAKE_CXX_COMPILER=' + cache['CMAKE_CXX_COMPILER'])
    subprocess.run(configure, check=True)
    subprocess.run([
        'cmake', '--build',
        str(probe_build), '--config', 'Release', '--parallel', '2'
    ],
                   check=True)
    binary = probe_build / ('component_probe.exe'
                            if os.name == 'nt' else 'component_probe')
    if not binary.is_file():
        binary = probe_build / 'Release' / binary.name
    if not binary.is_file():
        raise RuntimeError(f'Probe executable not found below {probe_build}')
    themes = [('native', None)] + [
        (p.stem, f':/qss/{p.stem}.css')
        for p in sorted((ROOT / 'libs/CVAppCommon/resources/qlementine-themes'
                        ).glob('*.json'))
        if not p.stem.startswith('qdarkstyle-')
    ]
    themes += [(f'QDarkStyleSheet::{name}', f'QDarkStyleSheet::{name}')
               for name in ('Light', 'Dark')]
    if args.theme:
        unknown = set(args.theme) - {name for name, _ in themes}
        if unknown:
            parser.error('Unknown theme(s): ' + ', '.join(sorted(unknown)))
        themes = [(n, t) for n, t in themes if n in args.theme]
    env = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
    core_dir = Path(
        (probe_build / 'qt-core-path.txt').read_text().strip()).parent
    plugin_file = probe_build / 'qt-plugin-path.txt'
    plugins = Path(plugin_file.read_text().strip()) if plugin_file.is_file(
    ) else prefix / 'plugins'
    runtime = os.pathsep.join([str(library.parent), str(core_dir)])
    if sys.platform.startswith('linux'):
        env['LD_LIBRARY_PATH'] = runtime + os.pathsep + env.get(
            'LD_LIBRARY_PATH', '')
    elif sys.platform == 'darwin':
        env['DYLD_LIBRARY_PATH'] = runtime + os.pathsep + env.get(
            'DYLD_LIBRARY_PATH', '')
    else:
        env['PATH'] = runtime + os.pathsep + env.get('PATH', '')
    env['QT_PLUGIN_PATH'] = str(plugins)
    env['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugins / 'platforms')
    representative = {
        'native', 'paraview', 'nord', 'macos', 'macos-dark',
        'QDarkStyleSheet::Light', 'QDarkStyleSheet::Dark'
    }
    results, failures = [], []
    for name, theme_id in themes:
        scales = (1, 1.5, 2) if name in representative else (1,)
        # Each Custom preset has an actual Fusion baseline, never legacy QSS.
        modes = ('native',) if name == 'native' else ('native', 'custom')
        for scale in scales:
            for mode in modes:
                label = name.replace(':', '_') + '-' + mode + f'-{scale}x'
                run = subprocess.run([
                    str(binary), name, theme_id or ':/qss/paraview.css', mode,
                    str(out / label)
                ],
                                     env={
                                         **env, 'QT_SCALE_FACTOR': str(scale)
                                     },
                                     capture_output=True,
                                     text=True)
                text = run.stdout + run.stderr
                (out / (label + '.log')).write_text(text)
                results.append({
                    'theme': name,
                    'theme_id': theme_id,
                    'mode': mode,
                    'scale': scale,
                    'exit_code': run.returncode,
                    'warning_log': label + '.log',
                    'stdout': run.stdout.strip()
                })
                print(run.stdout.strip())
                if run.stderr:
                    print(run.stderr, end='', file=sys.stderr)
                if run.returncode:
                    failures.append(label)
    summary = {
        'configured_build':
            str(build),
        'configured_library':
            str(library),
        'qt_major':
            major,
        'qt_prefix':
            str(prefix),
        'scope':
            'Real ecvApplicationBase backend integration and widget input/gallery; no whole-app scene or persistence claim',
        'custom_contract':
            'Qlementine backend; zero global QSS; explicit Custom only',
        'results':
            results,
        'failed':
            failures,
        'warning_policy':
            'Every Qt message retained; unsupported-property/image-resource/parse/regex fail. Only explicit offscreen platform limitations are classified separately.',
        'platform_claim':
            'Only actual runtime recorded in each probe; no cross-platform or Qt6 inference'
    }
    (out / 'component-verification.json'
    ).write_text(json.dumps(summary, indent=2) + '\n')
    if failures:
        print('FAILED: ' + ', '.join(failures), file=sys.stderr)
        return 1
    print(
        f'Actual-owner widget assertions passed for {len(results)} runs; inspect retained Qt warnings.'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

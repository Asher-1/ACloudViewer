"""Offscreen verification/gallery for theme-layered Qt component glyphs."""
from pathlib import Path
import os,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[2]
OUT=Path.home()/'ACloudViewer-icon-reviews/qt-components'
CPP=r'''#include <QActionGroup>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDockWidget>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QDirIterator>
#include <QRegularExpression>
#include <QHeaderView>
#include <QIcon>
#include <QImage>
#include <QLabel>
#include <QResource>
#include <QMainWindow>
#include <QMenu>
#include <QPainter>
#include <QPushButton>
#include <QRadioButton>
#include <QScreen>
#include <QScrollBar>
#include <QSpinBox>
#include <QSplitter>
#include <QTreeWidget>
#include <QGroupBox>
#include <QToolButton>
#include <QProgressBar>
#include <QSizeGrip>
#include <QStyleFactory>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextStream>
#include <QToolBar>
#include <QVBoxLayout>
#include <QtTest/QTest>
#include <iostream>
#include "ecvComponentStyle.h"
static int parseErrors=0;
static void qtMessageHandler(QtMsgType type,const QMessageLogContext&,const QString&message){if(message.contains(QStringLiteral("Could not parse application stylesheet")))++parseErrors;if(type==QtFatalMsg)abort();}
int main(int argc,char**argv){QApplication app(argc,argv);if(argc!=7)return 1;
QString flavor=argv[1],qssPath=argv[2],out=argv[3],rcc=argv[4],mode=argv[5];
qInstallMessageHandler(qtMessageHandler);
if(!QResource::registerResource(rcc))return 2;QFile f(qssPath);QString base;
if(flavor!="native"){if(!f.open(QIODevice::ReadOnly))return 3;base=QString::fromUtf8(f.readAll());}
QStyle *native=QStyleFactory::create("Fusion");if(!native)return 4;app.setStyle(native);
if(mode=="after")ecvComponentStyle::applyTheme(&app,base,flavor);else app.setStyleSheet(base);
if(parseErrors)return 16;
QString css=app.styleSheet();if(mode=="after"&&(css.count("ECV_COMPONENT_ICON_OVERLAY_BEGIN")!=1||!css.startsWith(base)))return 6;
QFile effective(out+"-effective.qss");effective.open(QIODevice::WriteOnly);effective.write(css.toUtf8());effective.close();
QDirIterator resourceFiles(QStringLiteral(":/ecv/components"),QStringList()<<"*.svg",QDir::Files);int assets=0;
while(resourceFiles.hasNext()){QString url=resourceFiles.next();QFile bundled(url),physical(QString::fromLocal8Bit(argv[6])+"/libs/CVAppCommon/resources/components/"+QFileInfo(url).fileName());if(!bundled.open(QIODevice::ReadOnly)||!physical.open(QIODevice::ReadOnly)||bundled.readAll()!=physical.readAll()||QImage(url).isNull())return 18;++assets;}
if(assets!=72)return 19;
if(mode=="after"){auto matches=QRegularExpression(QStringLiteral("url\\(:/ecv/components/([^)]*)\\)")).globalMatch(css);while(matches.hasNext()){auto match=matches.next();if(!QFile::exists(":/ecv/components/"+match.captured(1)))return 20;}}

QIcon standard=app.style()->standardIcon(QStyle::SP_BrowserReload);QImage first(40,40,QImage::Format_ARGB32_Premultiplied);first.fill(Qt::transparent);{QPainter p(&first);standard.paint(&p,QRect(2,2,36,36));}first.save(out+"-standard-before.png");
QWidget root;auto *vl=new QVBoxLayout(&root);auto *cb0=new QCheckBox("Unchecked");auto *cb1=new QCheckBox("Checked");cb1->setChecked(true);auto *cb2=new QCheckBox("Partial");cb2->setTristate(true);cb2->setCheckState(Qt::PartiallyChecked);vl->addWidget(cb0);vl->addWidget(cb1);vl->addWidget(cb2);auto *radio=new QRadioButton("Radio");radio->setChecked(true);vl->addWidget(radio);
auto *combo=new QComboBox;combo->addItems({"First","Second"});vl->addWidget(combo);auto *spin=new QSpinBox;spin->setRange(1,10);vl->addWidget(spin);auto *scroll=new QScrollBar(Qt::Horizontal);scroll->setRange(0,10);vl->addWidget(scroll);auto *vscroll=new QScrollBar(Qt::Vertical);vscroll->setRange(0,10);vscroll->setFixedHeight(72);vl->addWidget(vscroll);
auto *tabs=new QTabWidget;tabs->setTabsClosable(true);tabs->addTab(new QLabel("Tab A"),"Alpha");tabs->addTab(new QLabel("Tab B"),"Beta");vl->addWidget(tabs);
auto *toolbar=new QToolBar;toolbar->addAction(app.style()->standardIcon(QStyle::SP_FileDialogNewFolder),"New folder");toolbar->addSeparator();toolbar->addAction(app.style()->standardIcon(QStyle::SP_DialogSaveButton),"Save");vl->addWidget(toolbar);
auto *table=new QTableWidget(2,2);table->setHorizontalHeaderLabels({"Name","Value"});table->setItem(0,0,new QTableWidgetItem("row"));vl->addWidget(table);
auto *tree=new QTreeWidget;tree->setHeaderLabels({"Hierarchy"});auto *branch=new QTreeWidgetItem(tree,{"Scene"});new QTreeWidgetItem(branch,{"Point cloud"});branch->setExpanded(true);vl->addWidget(tree);
auto *group=new QGroupBox("Options");group->setCheckable(true);group->setChecked(true);auto *groupLayout=new QVBoxLayout(group);auto *groupCheck=new QCheckBox("Enable");groupCheck->setTristate(true);groupLayout->addWidget(groupCheck);vl->addWidget(group);
auto *tool=new QToolButton;auto *toolMenu=new QMenu(tool);auto *menuCheck=toolMenu->addAction("Checked option");menuCheck->setCheckable(true);menuCheck->setChecked(true);auto *radioGroup=new QActionGroup(toolMenu);radioGroup->setExclusive(true);auto *exclusive=toolMenu->addAction("Radio option");exclusive->setCheckable(true);exclusive->setChecked(true);radioGroup->addAction(exclusive);auto *submenu=toolMenu->addMenu("Nested menu");submenu->addAction("Sub-action");tool->setMenu(toolMenu);tool->setPopupMode(QToolButton::MenuButtonPopup);tool->setText("More");vl->addWidget(tool);
auto *dock=new QDockWidget("Dock",&root);dock->setFeatures(QDockWidget::DockWidgetClosable|QDockWidget::DockWidgetFloatable);dock->setWidget(new QLabel("Dock content"));vl->addWidget(dock);
auto *progress=new QProgressBar;progress->setValue(64);vl->addWidget(progress);vl->addWidget(new QSizeGrip(&root));
root.resize(640,980);root.show();root.setFocus();app.processEvents();QPixmap capture=root.grab();if(qApp->devicePixelRatio()>1.5&&capture.size()!=QSize(root.width()*2,root.height()*2))return 17;capture.save(out+"-widgets.png");toolMenu->popup(root.mapToGlobal(QPoint(360,500)));app.processEvents();toolMenu->grab().save(out+"-menu.png");toolMenu->hide();
cb0->click();if(!cb0->isChecked())return 11;cb0->click();if(cb0->isChecked())return 12;combo->setCurrentIndex(1);if(combo->currentIndex()!=1)return 13;spin->stepUp();if(spin->value()!=2)return 14;groupCheck->setCheckState(Qt::PartiallyChecked);if(groupCheck->checkState()!=Qt::PartiallyChecked)return 15;
QTest::mouseMove(cb1,cb1->rect().center());app.processEvents();root.grab().save(out+"-widgets-hover.png");cb1->setEnabled(false);app.processEvents();root.grab().save(out+"-widgets-disabled.png");cb1->setEnabled(true);
if(mode=="before"){std::cout<<"before "<<flavor.toStdString()<<" PASS: base QSS and component gallery rendered\n";return 0;}
QString alt=flavor=="native"?QStringLiteral("QPalette{background:#121212;}*{color:#EEEEEE;}"):QStringLiteral("QPalette{background:#FFFFFF;}*{color:#111111;}");ecvComponentStyle::applyTheme(&app,alt,flavor+"-switch");if(app.styleSheet().count("ECV_COMPONENT_ICON_OVERLAY_BEGIN")!=1)return 7;
QImage second(40,40,QImage::Format_ARGB32_Premultiplied);second.fill(Qt::transparent);{QPainter p(&second);standard.paint(&p,QRect(2,2,36,36));}if(first==second)return 10;second.save(out+"-standard-after.png");
// Restoring original theme twice must not nest overlays or lose base QSS.
ecvComponentStyle::applyTheme(&app,base,flavor);ecvComponentStyle::applyTheme(&app,base,flavor);if(app.styleSheet().count("ECV_COMPONENT_ICON_OVERLAY_BEGIN")!=1)return 8;
QImage stable(40,40,QImage::Format_ARGB32_Premultiplied);stable.fill(Qt::transparent);{QPainter p(&stable);standard.paint(&p,QRect(2,2,36,36));}stable.save(out+"-standard-restored.png");
if(standard.isNull()||standard.pixmap(32,32).isNull())return 9;
std::cout<<flavor.toStdString()<<" PASS: base QSS retained, one overlay, standard QIcon, checkbox/radio/combo/spin/scroll/tab/header/toolbar widgets rendered, theme switch/reapply idempotent\n";return 0;}
'''
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    qt_env={**os.environ,'PKG_CONFIG_PATH':'/opt/qt515/lib/pkgconfig:'+os.environ.get('PKG_CONFIG_PATH','')}
    flags=subprocess.check_output(['pkg-config','--cflags','--libs','Qt5Widgets','Qt5Test'],text=True,env=qt_env).split()
    source=OUT/'component_probe.cpp';source.write_text(CPP)
    binary=OUT/'component_probe';
    subprocess.run(['g++','-std=c++17','-fPIC','-I'+str(ROOT/'libs/CVAppCommon/include'),str(source),'-L'+str(ROOT/'build_app/bin'),'-lCVAppCommon',f"-Wl,-rpath,{ROOT/'build_app/bin'}",'-o',str(binary),*flags],check=True)
    rcc=OUT/'components.rcc';subprocess.run(['rcc','--binary',str(ROOT/'libs/CVAppCommon/resources/component-icons.qrc'),'-o',str(rcc)],check=True)
    themes=[('native','')]
    themes.extend((p.stem,str(p.relative_to(ROOT))) for p in sorted((ROOT/'app/Resources/themes/qss').glob('*.css')))
    themes.extend([('QDarkStyleSheet::Light','libs/CVAppCommon/QDarkStyleSheet/qdarkstyle/light/lightstyle.qss'),
                   ('QDarkStyleSheet::Dark','libs/CVAppCommon/QDarkStyleSheet/qdarkstyle/dark/darkstyle.qss')])
    env={**os.environ,'QT_QPA_PLATFORM':'offscreen','LD_LIBRARY_PATH':'/opt/qt515/lib:'+os.environ.get('LD_LIBRARY_PATH',''),
         'QT_QPA_PLATFORM_PLUGIN_PATH':'/opt/qt515/plugins/platforms'}
    hi_dpi_themes={'native','blue','nord','paraview','QDarkStyleSheet::Light','QDarkStyleSheet::Dark'}
    for name,relative in themes:
        qss=str(ROOT/relative) if relative else ''
        for scale in (1,2) if name in hi_dpi_themes else (1,):
            for mode in ('before','after'):
                out=str(OUT/(name+'-'+mode+(f'-{scale}x' if scale==2 else '')))
                run_env={**env,'QT_SCALE_FACTOR':str(scale)}
                result=subprocess.run([str(binary),name,qss,out,str(rcc),mode,str(ROOT)],env=run_env,capture_output=True,text=True)
                if result.returncode:
                    raise RuntimeError(f'{mode}/{name}/{scale}x Qt probe failed ({result.returncode}):\\n{result.stdout}\\n{result.stderr}')
                print(result.stdout.strip())
                if result.stderr.strip():
                    print(result.stderr.strip())
if __name__=='__main__':main()

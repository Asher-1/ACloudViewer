"""Independent Qt tests for both reconstruction media QRC bundles."""
from pathlib import Path
import os,subprocess,tempfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
MEDIA=ROOT/'app/reconstruction/media'
QRC=[ROOT/'app/reconstruction/resources.qrc',ROOT/'libs/Reconstruction/src/ui/resources.qrc']
CPP=r'''#include <QApplication>
#include <QResource>
#include <QFile>
#include <QImage>
#include <QIcon>
#include <QSvgRenderer>
#include <QPainter>
#include <QTextStream>
#include <iostream>
int main(int argc,char**argv){QApplication app(argc,argv);if(argc<3)return 1;for(int i=1;i<argc-1;i++)if(!QResource::registerResource(argv[i]))return 2;QFile list(argv[argc-1]);if(!list.open(QIODevice::ReadOnly))return 3;QTextStream in(&list);int n=0;
while(!in.atEnd()){auto row=in.readLine().split('\t');if(row.size()!=2)return 4;QString url=row[0],disk=row[1];QFile a(url),b(disk);if(!a.open(QIODevice::ReadOnly)||!b.open(QIODevice::ReadOnly)||a.readAll()!=b.readAll())return 5;QImage image(url);QIcon icon(url);if(image.isNull()||image.size()!=QSize(128,128)||icon.isNull())return 6;if(url.endsWith(".svg")){QSvgRenderer svg(url);if(!svg.isValid()||svg.defaultSize()!=QSize(128,128))return 7;for(int s:{16,24,32,48,64,128,384,1024}){QImage out(s,s,QImage::Format_ARGB32_Premultiplied);out.fill(Qt::transparent);QPainter p(&out);svg.render(&p);p.end();bool ink=false;for(int y=0;y<s&&!ink;y+=2)for(int x=0;x<s;x+=2)if(qAlpha(out.pixel(x,y))>0){ink=true;break;}if(!ink||icon.pixmap(s,s).isNull()||icon.pixmap(s,s,QIcon::Disabled).isNull())return 8;}}++n;}std::cout<<"PASS: "<<n<<" reconstruction media URLs; 128 default plus 8 direct SVG/QIcon sizes, normal/disabled, exact bytes"<<std::endl;return 0;}
'''
def main():
 flags=subprocess.check_output(['pkg-config','--cflags','--libs','Qt5Widgets','Qt5Svg'],text=True).split()
 listing=[]
 for p in sorted(MEDIA.glob('*.svg')):listing.append(f':/media/{p.name}\t{p}')
 with tempfile.TemporaryDirectory(prefix='reconstruction-icons-') as tmp:
  work=Path(tmp);source=work/'probe.cpp';source.write_text(CPP);binary=work/'probe';subprocess.run(['g++','-std=c++17','-fPIC',str(source),'-o',str(binary),*flags],check=True)
  resources=[]
  for i,qrc in enumerate(QRC):
   r=work/f'{i}.rcc';subprocess.run(['rcc','--binary',str(qrc),'-o',str(r)],check=True);resources.append(r)
  tsv=work/'paths.tsv';tsv.write_text('\n'.join(listing)+'\n');env={**os.environ,'QT_QPA_PLATFORM':'offscreen'}
  for label,args in [('app-only',[resources[0]]),('library-only',[resources[1]]),('combined',resources)]:
   result=subprocess.run([str(binary),*[str(r) for r in args],str(tsv)],env=env,capture_output=True,text=True,check=True);print(label+': '+result.stdout.strip())
if __name__=='__main__':main()

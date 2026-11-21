"""Actual offscreen Qt loading for every app and qPCL image registration."""
from pathlib import Path
import os
import subprocess
from tempfile import TemporaryDirectory

if __package__:
    from .audit_svg_inventory import ROOT, registration_map
else:
    from audit_svg_inventory import ROOT, registration_map
CPP=r'''#include <QApplication>
#include <QResource>
#include <QFile>
#include <QIcon>
#include <QImage>
#include <QPainter>
#include <QSvgRenderer>
#include <QTextStream>
int main(int argc,char**argv){QApplication app(argc,argv);
 if(argc!=3||!QResource::registerResource(argv[1]))return 2;
 QFile f(argv[2]);if(!f.open(QIODevice::ReadOnly))return 3;QTextStream in(&f),out(stdout);int n=0,renders=0;
 while(!in.atEnd()){auto a=in.readLine().split('\t');if(a.size()!=2)return 4;
 QFile p(a[0]),d(a[1]);if(!p.open(QIODevice::ReadOnly)||!d.open(QIODevice::ReadOnly)||p.readAll()!=d.readAll())return 5;
 QImage im(a[0]);QIcon icon(a[0]);if(im.isNull()||icon.isNull()){out<<"FAIL "<<a[0]<<"\n";return 6;}
 for(int s:{16,24,32,48,96,192}){if(icon.pixmap(s,s).isNull()||icon.pixmap(s,s,QIcon::Disabled).isNull())return 7;
 if(a[0].endsWith(".svg")){QSvgRenderer v(a[0]);if(!v.isValid())return 8;QImage x(s,s,QImage::Format_ARGB32_Premultiplied);x.fill(Qt::transparent);QPainter q(&x);v.render(&q);q.end();bool visible=false;for(int y=0;y<s;y++)for(int z=0;z<s;z++)visible=visible||qAlpha(x.pixel(z,y))>0;if(!visible)return 9;}++renders;}++n;}
 out<<"PASS: "<<n<<" resource URLs, "<<renders<<" Qt size/normal/disabled checks and canonical byte equality\n";return 0;}
'''
def main(rcc="rcc"):
    qrcs = {
        "app": ROOT / "app/MainWindow.qrc",
        "qPCL": ROOT / "plugins/core/Standard/qPCL/qPCL.qrc",
    }
    _, registrations, missing, malformed = registration_map(
        [(label, path, "") for label, path in qrcs.items()]
    )
    duplicates = [url for url, rows in registrations.items() if len(rows) != 1]
    if missing or malformed or duplicates:
        raise SystemExit(f"Invalid QRC registrations: {missing + malformed + duplicates}")

    flags = subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", "Qt5Widgets", "Qt5Svg"], text=True
    ).split()
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    with TemporaryDirectory(prefix="acloudviewer-icons-") as folder:
        work = Path(folder)
        source = work / "probe.cpp"
        source.write_text(CPP)
        probe = work / "probe"
        subprocess.run(["g++", "-std=c++17", "-fPIC", str(source), "-o", str(probe), *flags], check=True)
        for label, qrc in qrcs.items():
            listing = work / f"{label}.tsv"
            images = {".svg", ".png", ".gif", ".jpg", ".jpeg", ".ico"}
            listing.write_text("".join(
                f"{url}\t{ROOT / row['file']}\n"
                for url, (row,) in registrations.items()
                if row["qrc"] == label and Path(row["file"]).suffix.lower() in images
            ))
            bundle = work / f"{label}.rcc"
            subprocess.run([rcc, "--binary", str(qrc), "-o", str(bundle)], check=True)
            result = subprocess.run([str(probe), str(bundle), str(listing)], env=env, check=True, stdout=subprocess.PIPE, text=True)
            print(f"{label}: {result.stdout.strip()}")
    return 0

#include <QCoreApplication>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSslConfiguration>
#include <QTimer>
#include <QEventLoop>
#include <cstdio>
void pump() { QEventLoop loop; QTimer::singleShot(100,&loop,&QEventLoop::quit); loop.exec(); }
int main(int argc,char **argv) {
 QCoreApplication app(argc,argv);
 const QString mode=argc>1?argv[1]:"rapid";
 std::fprintf(stderr,"mode=%s begin\n",qPrintable(mode));
 {
  QNetworkAccessManager net;
  net.setProxy(QNetworkProxy(QNetworkProxy::HttpProxy,"127.0.0.1",9));
  for(int i=0;i<3;++i) {
   std::fprintf(stderr,"request %d begin\n",i);
   QNetworkRequest req{QUrl(mode=="http"?"http://huggingface.co/test":"https://huggingface.co/test")};
   req.setSslConfiguration(QSslConfiguration::defaultConfiguration());
   auto *reply=net.get(req);
   std::fprintf(stderr,"request %d created\n",i);
   if(mode=="before")pump();
   reply->abort(); reply->deleteLater();
   std::fprintf(stderr,"request %d cancelled\n",i);
   if(mode=="after")pump();
  }
  std::fprintf(stderr,"manager destroy begin\n");
 }
 std::fprintf(stderr,"manager destroy done\n");
 return 0;
}

#include <QCoreApplication>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <cstdio>
extern "C" __declspec(dllimport) void static_request();
extern "C" __declspec(dllimport) void owned_request();
int main(int argc,char **argv) {
 {
 QCoreApplication app(argc,argv);
 const QString mode=argc>1?argv[1]:"rapid";
 std::fprintf(stderr,"mode=%s begin\n",qPrintable(mode));
 if(mode=="static")static_request();
 else if(mode=="owned")owned_request();
 else {
  QNetworkAccessManager net;
  net.setProxy(QNetworkProxy(QNetworkProxy::HttpProxy,"127.0.0.1",9));
  auto *reply=net.get(QNetworkRequest(QUrl(mode=="http"?"http://huggingface.co/test":"https://huggingface.co/test")));
  reply->abort(); reply->deleteLater();
 }
 std::fprintf(stderr,"application destroy begin\n");
 }
 std::fprintf(stderr,"application destroy done, returning from main\n");
 return 0;
}

#include <QCoreApplication>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <cstdio>
void request(QNetworkAccessManager &net) {
 net.setProxy(QNetworkProxy(QNetworkProxy::HttpProxy,"127.0.0.1",9));
 auto *reply=net.get(QNetworkRequest(QUrl("https://huggingface.co/test")));
 reply->abort(); reply->deleteLater();
 std::fprintf(stderr,"DLL request cancelled\n");
}
extern "C" __declspec(dllexport) void static_request() {
 static QNetworkAccessManager net;
 request(net);
}
extern "C" __declspec(dllexport) void owned_request() {
 auto *net=new QNetworkAccessManager(QCoreApplication::instance());
 request(*net);
}

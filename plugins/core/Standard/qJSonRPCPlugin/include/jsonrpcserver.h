#pragma once

//Qt
#include <QWebSocketServer>
#include <QWebSocket>
#include <QObject>
#include <QMap>
//STL
#include <functional>

struct JsonRPCResult
{
    static JsonRPCResult error(int code, QString message)
	{
		JsonRPCResult result;
		{
			result.isError = true;
			result.error_code = code;
			result.error_message = message;
		}
        return result;
    }

	static JsonRPCResult success(QVariant value)
	{
		JsonRPCResult result;
		{
			result.isError = false;
			result.result = value;
		}
        return result;
    }

	bool isError{true};
    int error_code{-32601};
    QString error_message = "Method not found";
    QVariant result;

};

class JsonRPCServer : public QObject
{
    Q_OBJECT
public:
    explicit JsonRPCServer(QObject *parent = nullptr);
    ~JsonRPCServer();

    void listen(unsigned int port);
    void close();

signals:
    JsonRPCResult execute(QString method, QMap<QString, QVariant> params);
private slots:
    void onNewConnection();
    void onClosed();
    void processTextMessage(QString message);
    void processBinaryMessage(QByteArray message);
    void socketDisconnected();

private:
    QWebSocketServer *ws_server{nullptr};
    QList<QWebSocket * > connections;
};

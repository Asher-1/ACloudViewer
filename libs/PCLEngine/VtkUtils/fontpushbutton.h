#ifndef FONTPUSHBUTTON_H
#define FONTPUSHBUTTON_H

#include <QPushButton>

#include "../qPCL.h"

namespace Widgets
{

class QPCL_ENGINE_LIB_API FontPushButton : public QPushButton
{
    Q_OBJECT
public:
    explicit FontPushButton(QWidget *parent = 0);
    explicit FontPushButton(const QString& text, QWidget* parent = 0);

signals:
    void fontSelected(const QFont& font);

private slots:
    void onClicked();

private:
    void init();

};

}
#endif // FONTPUSHBUTTON_H

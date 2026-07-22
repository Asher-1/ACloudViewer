// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "colorcombobox.h"

#include <QCoreApplication>
#include <QPainter>
#include <QPixmap>
#include <QSettings>
#include <algorithm>

namespace Widgets {
/*!
  \class ColorBox
  \brief ColorBox, an extension of QComboBox.

  \ingroup Pictureui
  */
const QColor ColorComboBox::stColors[] = {
        QColor(Qt::black),       QColor(Qt::red),        QColor(Qt::green),
        QColor(Qt::blue),        QColor(Qt::cyan),       QColor(Qt::magenta),
        QColor(Qt::yellow),      QColor(Qt::darkYellow), QColor(Qt::darkBlue),
        QColor(Qt::darkMagenta), QColor(Qt::darkRed),    QColor(Qt::darkGreen),
        QColor(Qt::darkCyan),    QColor("#0000A0"),      QColor("#FF8000"),
        QColor("#8000FF"),       QColor("#FF0080"),      QColor(Qt::white),
        QColor(Qt::lightGray),   QColor(Qt::gray),       QColor("#FFFF80"),
        QColor("#80FFFF"),       QColor("#FF80FF"),      QColor(Qt::darkGray),
};

/*!
 * \brief Constructs the color combo box and initializes it.
 * \param parent Parent widget.
 */
ColorComboBox::ColorComboBox(QWidget* parent) : QComboBox(parent) {
    setEditable(false);
    init();
}

/*!
 * \brief Initializes the color combo box.
 */
void ColorComboBox::init() {
    QList<QColor> indexedColors = colorList();
    QStringList color_names = colorNames();

    QPixmap icon = QPixmap(28, 16);
    QRect r = QRect(0, 0, 27, 15);

    QPainter p;
    p.begin(&icon);

    for (int i = 0; i < indexedColors.size(); i++) {
        p.setBrush(QBrush(indexedColors[i]));
        p.drawRect(r);
        this->addItem(icon, color_names[i]);
    }
    p.end();
}

/*!
 * \brief Sets the current color.
 * \param c Color.
 */
void ColorComboBox::setColor(const QColor& c) {
    setCurrentIndex(colorIndex(c));
}

/*!
 * \brief Returns the current color.
 * \return The current color.
 */
QColor ColorComboBox::color() const { return color(this->currentIndex()); }

/*!
 * \brief Returns the index of a color.
 * \param c Color to look up.
 * \return Index of the color.
 */
int ColorComboBox::colorIndex(const QColor& c) {
    if (!isValidColor(c)) return 0;

    return colorList().indexOf(c);
}

/*!
 * \brief Returns the color at the given index.
 * \param colorIndex Color index.
 * \return Color at the given index, or black if the index is out of range
 * or not found in the combo box.
 */
QColor ColorComboBox::color(int colorIndex) {
    QList<QColor> colorsList = colorList();
    if (colorIndex >= 0 && colorIndex < colorsList.size())
        return colorsList[colorIndex];

    return Qt::black;
}

/*!
 * \brief Returns the color list for the combo box.
 * \return List of colors in the combo box.
 */
QList<QColor> ColorComboBox::colorList() {
    QSettings settings(QCoreApplication::applicationDirPath() + "\\config.ini",
                       QSettings::IniFormat);

    settings.beginGroup("/General");

    QList<QColor> indexedColors;
    QStringList lst = settings.value("/IndexedColors").toStringList();
    if (!lst.isEmpty()) {
        for (int i = 0; i < lst.size(); i++) indexedColors << QColor(lst[i]);
    } else {
        for (int i = 0; i < stColorsCount; i++) indexedColors << stColors[i];
    }
    settings.endGroup();

    return indexedColors;
}

/*!
 * \brief Returns the color name list for the combo box.
 * \return List of color names in the combo box.
 */
QStringList ColorComboBox::colorNames() {
    QSettings settings(QCoreApplication::applicationDirPath() + "\\config.ini",
                       QSettings::IniFormat);

    settings.beginGroup("/General");
    QStringList color_names =
            settings.value("/IndexedColorNames", defaultColorNames())
                    .toStringList();
    settings.endGroup();
    return color_names;
}

/*!
 * \brief Returns the default color at the given index.
 * \param colorIndex Color index.
 * \return Default color at the given index, or black if the index is out of
 * range or not found in the combo box.
 */
QColor ColorComboBox::defaultColor(int colorIndex) {
    if (colorIndex >= 0 && colorIndex < (int)sizeof(stColors))
        return stColors[colorIndex];

    return Qt::black;
}

/*!
 * \brief Returns whether a color is valid.
 * \param color Color to check.
 * \return true if the color is valid, false otherwise.
 */
bool ColorComboBox::isValidColor(const QColor& color) {
    return colorList().contains(color);
}

/*!
 * \brief Returns the number of predefined colors in the combo box.
 * \return Number of predefined colors.
 */
int ColorComboBox::numPredefinedColors() { return stColorsCount; }

/*!
 * \brief Returns the default color name list.
 * \return Default color name list.
 */
QStringList ColorComboBox::defaultColorNames() {
    QStringList color_names = QStringList() << tr("black");
    color_names << tr("red");
    color_names << tr("green");
    color_names << tr("blue");
    color_names << tr("cyan");
    color_names << tr("magenta");
    color_names << tr("yellow");
    color_names << tr("dark yellow");
    color_names << tr("navy");
    color_names << tr("purple");
    color_names << tr("wine");
    color_names << tr("olive");
    color_names << tr("dark cyan");
    color_names << tr("royal");
    color_names << tr("orange");
    color_names << tr("violet");
    color_names << tr("pink");
    color_names << tr("white");
    color_names << tr("light gray");
    color_names << tr("gray");
    color_names << tr("light yellow");
    color_names << tr("light cyan");
    color_names << tr("light magenta");
    color_names << tr("dark gray");
    return color_names;
}

/*!
 * \brief Returns the default color list.
 * \return Default color list.
 */
QList<QColor> ColorComboBox::defaultColors() {
    QList<QColor> lst;
    for (int i = 0; i < stColorsCount; i++) lst << stColors[i];

    return lst;
}

}  // namespace Widgets

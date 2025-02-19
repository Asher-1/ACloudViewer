#ifndef COLORSERIES_H
#define COLORSERIES_H

#include <QColor>

#include "../qPCL.h"

namespace Utils
{

class ColorSeriesPrivate;
class QPCL_ENGINE_LIB_API ColorSeries
{
public:
    enum Scheme { Spectrum, Warm, Cool, Blues, WildFlower, Citrus };

    ColorSeries();
    ~ColorSeries();

    void setScheme(Scheme scheme);
    Scheme scheme() const;

    QColor color(int index) const;
    QColor nextColor() const;

private:
    ColorSeriesPrivate* d_ptr;
    Q_DISABLE_COPY(ColorSeries)
};

} // namespace Utils
#endif // COLORSERIES_H

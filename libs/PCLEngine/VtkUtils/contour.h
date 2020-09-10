#ifndef CONTOUR_H
#define CONTOUR_H

#include "surface.h"

#include "vector4f.h"

namespace VtkUtils
{
class ContourPrivate;
class QPCL_ENGINE_LIB_API Contour : public Surface
{
    Q_OBJECT
public:
    explicit Contour(QWidget* parent = nullptr);
    ~Contour();

    void setVectors(const QList<Vector4F>& vectors);

    void setNumberOfContours(int num);
    int numberOfContours() const;

    void setPlaneVisible(bool visible = true);
    bool planeVisible() const;

    void setPlaneDistance(qreal distance);
    bool planeDistance() const;

protected:
    void renderSurface();

private:
    ContourPrivate* d_ptr;
    Q_DISABLE_COPY(Contour)
};

} // namespace VtkUtils
#endif // CONTOUR_H

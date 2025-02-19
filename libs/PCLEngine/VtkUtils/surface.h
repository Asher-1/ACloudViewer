#ifndef SURFACE_H
#define SURFACE_H

#include "../qPCL.h"

#include "vtkwidget.h"
#include "point3f.h"

class vtkActor;
namespace VtkUtils
{

class SurfacePrivate;
class QPCL_ENGINE_LIB_API Surface : public VtkWidget
{
    Q_OBJECT
public:
    explicit Surface(QWidget* parent = nullptr);
    virtual ~Surface();

    void setPoints(const QList<Point3F>& points);

protected:
    virtual void renderSurface();
    vtkActor* surfaceActor() const;

private:
    SurfacePrivate* d_ptr;
    Q_DISABLE_COPY(Surface)
};

} // namespace VtkUtils
#endif // SURFACE_H

#ifndef MODELTOVECTORSCONVERTER_H
#define MODELTOVECTORSCONVERTER_H

#include "signalledrunable.h"
#include "utils.h"
#include "vector4f.h"

namespace VtkUtils
{
class TableModel;
class QPCL_ENGINE_LIB_API ModelToVectorsConverter : public SignalledRunnable
{
    Q_OBJECT
public:
    ModelToVectorsConverter(TableModel* model);

    void run();

    QList<Vector4F> vectors() const;

private:
    QList<Vector4F> m_vectors;
    TableModel* m_model = nullptr;
};

} // namespace Utils
#endif // MODELTOVECTORSCONVERTER_H

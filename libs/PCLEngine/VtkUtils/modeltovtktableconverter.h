#ifndef MODELTOVTKTABLECONVERTER_H
#define MODELTOVTKTABLECONVERTER_H

#include "signalledrunable.h"

#include "../qPCL.h"

class vtkTable;
namespace VtkUtils
{
class TableModel;
class QPCL_ENGINE_LIB_API ModelToVtkTableConverter : public SignalledRunnable
{
    Q_OBJECT
public:
    explicit ModelToVtkTableConverter(TableModel* model);

    void setLabels(const QStringList& labels);
    QStringList labels() const;

    void run();

    vtkTable* table() const;

private:
	TableModel* m_model = nullptr;
    vtkTable* m_table = nullptr;
    QStringList m_labels;
};

} // namespace VtkUtils
#endif // MODELTOVTKTABLECONVERTER_H

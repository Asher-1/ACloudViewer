// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkSmartPointer.h>

#include <QDebug>
#include <QWidget>

#include "ui_generalfilterwindow.h"

// namespace Ui
//{
//	class GeneralFilterWindow;
// }

namespace VtkUtils {
class TableModel;
class VtkWidget;
}  // namespace VtkUtils

class vtkActor;
class vtkDataObject;
class vtkScalarBarActor;
class vtkLODActor;
class vtkLookupTable;
class ccHObject;
class FilterWindow : public QWidget {
    Q_OBJECT
public:
    enum { DefaultRows = 10 };
    explicit FilterWindow(QWidget* parent = nullptr);
    virtual ~FilterWindow();

    void setFileName(const QString& fileName);
    QString fileName() const;

    virtual void update();
    virtual void apply() = 0;

    virtual bool setInput(const ccHObject* obj);
    virtual ccHObject* getOutput() const;

protected slots:
    void onObjFileReaderFinished();
    void onDynaFileReaderFinished();
    void onFluentFileReaderFinished();
    void onVrmlFileReaderFinished();
    void onStlFileReaderFinished();
    void onVtkFileReaderFinished();
    void onNastranFileReaderFinished();
    void onAnsysFileReaderFinished();
    void onPlyFileReaderFinished();
    void randomTableModel();
    void fireupModelToPointsConverter();
    void onModelToPointsConverterFinished();
    void onPointsToPolyDataConverterFinished();

protected:
    enum DisplayEffect { Transparent, Points, Opaque, Wireframe };
    void setDisplayEffect(DisplayEffect effect);
    DisplayEffect displayEffect() const;

    void applyDisplayEffect();
    void readFile();
    QString fileFilter() const;
    void browseFile();
    void showOrientationMarker(bool show = true);
    void showScalarBar(bool show = true);
    void showOutline(bool show = true);
    void setOutlineColor(const QColor& clr);

    bool isValidPolyData() const;
    bool isValidDataSet() const;

    void setResultData(vtkDataObject* data);
    vtkDataObject* resultData() const;

    void setScalarBarColors(const QColor& clr1, const QColor& clr2);
    QColor color1() const;
    QColor color2() const;

    void setScalarRange(double min, double max);
    double scalarMin() const;
    double scalarMax() const;

    vtkSmartPointer<vtkLookupTable> createLookupTable(double min, double max);

    virtual void modelReady();
    virtual void createUi();
    virtual void dataChanged();
    virtual void colorsChanged();

    template <class DataObject, class Mapper>
    void createActorFromData(vtkDataObject* dataObj);

    template <class ConfigClass>
    void setupConfigWidget(ConfigClass* cc) {
        QWidget* configWidget = new QWidget(this);
        cc->setupUi(configWidget);
        m_ui->setupUi(this);
        m_ui->configLayout->addWidget(configWidget);
    }

    void initTableModel();
    bool initTableModel(const ccHObject* obj);

protected:
    Ui::GeneralFilterWindow* m_ui = nullptr;
    QString m_fileName;
    DisplayEffect m_displayEffect = Transparent;
    vtkDataObject* m_dataObject = nullptr;
    vtkDataObject* m_resultData = nullptr;
    VtkUtils::VtkWidget* m_vtkWidget = nullptr;
    VtkUtils::TableModel* m_tableModel = nullptr;

    vtkSmartPointer<vtkLODActor> m_modelActor;
    vtkSmartPointer<vtkLODActor> m_filterActor;
    vtkSmartPointer<vtkScalarBarActor> m_scalarBar;
    vtkSmartPointer<vtkActor> m_outlineActor;

    QColor m_color1 = Qt::blue;
    QColor m_color2 = Qt::red;
    double m_scalarMin = 0.0;
    double m_scalarMax = 1.0;

    bool meshMode = true;
};

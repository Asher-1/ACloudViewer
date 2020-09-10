#ifndef RENDERSURFACE_H
#define RENDERSURFACE_H

#include <QWidget>

#include <CVGeom.h>
#include <VtkUtils/point3f.h>
#include <vtkSmartPointer.h>

namespace Ui
{
    class RenderSurface;
}

namespace VtkUtils
{
    class TableModel;
}

namespace VtkUtils
{
    class VtkWidget;
}

class vtkActor;
class vtkActor2D;
class vtkPolyData;
class vtkDelaunay2D;
class vtkCubeAxesActor;
class vtkScalarBarActor;
class vtkPolyDataMapper;
class vtkCubeAxesActor2D;
class vtkOrientationMarkerWidget;

class QMenu;
class ccHObject;
class RenderSurface : public QWidget
{
    Q_OBJECT

public:
    explicit RenderSurface(QWidget *parent = 0);
    ~RenderSurface();
	
	bool setInput(const ccHObject* obj);
	void reset();
	ccHObject* getOutput() const;

private slots:
    void onSourceActionTriggered();
    void onPointReaderFinished();
    void onPointsToModelConverterFinished();
    void onTableModelDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight);
    void onModelToPointsConverterFinished();
    void onColorChanged(const QColor& clr);
    void onPointsToPolyDataConverterFinished();

    void showOrientationMarker(bool show = true);
    void showScalarBar(bool show = true);
    void setScalarBarColors(int num);
    void showCubeAxes(bool show = true);
    void showOutline(bool show = true);
    void showContour(bool show = true);
    void showContourLines(bool show = true);
    void showContourLabels(bool show = true);
    void showBandedContour(bool show = true);
    void showEdges(bool show = true);

    void setOutlineColor(const QColor& clr);
    void setCubeAxesColor(const QColor& clr);
    void setContourLinesColor(const QColor& clr);
    void setContourLabelsColor(const QColor& clr);
    void setScalarBarFrameColor(const QColor& clr);
    void setEdgeColor(const QColor& clr);

    void onGroupBoxToggled(bool on);

    void on_viewFitButton_clicked();
    void on_showOriMarkerButton_toggled(bool checked);
    void on_showScalarBarButton_toggled(bool checked);
    void on_numOfColorsSpinBox_valueChanged(int arg1);
    void on_showCubeAxesButton_toggled(bool checked);
    void on_showOutlineButton_toggled(bool checked);
    void on_showContourButton_toggled(bool checked);
    void on_randomButton_clicked();
    void on_rowsSpinBox_valueChanged(int arg1);
    void on_loadFileButton_clicked();
    void on_checkBox_toggled(bool checked);
    void on_showLabelsCheckBox_toggled(bool checked);
    void on_showContourLinesCheckBox_toggled(bool checked);
    void on_numOfContoursSpinBox_valueChanged(int arg1);
    void on_scalarBarOrientationComboBox_currentIndexChanged(int index);
    void on_flyModeComboBox_currentIndexChanged(int index);

    void on_numOfLabelsSpinBox_valueChanged(int arg1);
    void on_showXAxisCheckBox_toggled(bool checked);
    void on_showYAxisCheckBox_toggled(bool checked);
    void on_showZAxisCheckBox_toggled(bool checked);
    void on_xLabelEdit_textChanged(const QString &arg1);
    void on_yLabelEdit_textChanged(const QString &arg1);

    void on_zLabelEdit_textChanged(const QString &arg1);
    void on_genFacesCheckBox_toggled(bool checked);
    void on_outlineOpacitySpinBox_valueChanged(double arg1);
    void on_showFrameCheckBox_toggled(bool checked);
    void on_showColorBarCheckBox_toggled(bool checked);
    void on_showTickLabelsCheckBox_toggled(bool checked);
    void on_frameWidthSpinBox_valueChanged(int arg1);
    void on_outlineLineWidthSpinBox_valueChanged(int arg1);
    void on_showEdgeCheckBox_toggled(bool checked);
    void on_configGroupBox_toggled(bool arg1);
    void on_gradientCombo_activated(int index);

private:
    void init();
	bool initTableModel(const ccHObject * obj);
    void initConnections();
    void initSettings();
    void renderSurfaceFromPoints(const QVector<VtkUtils::Point3F>& points, const QVector<Tuple3ui>& vertices = QVector<Tuple3ui>());
    void fireupModelToPointsConverter();
    void fireupPointsReader(const QString& file);

private:
    Ui::RenderSurface *m_ui;
    QMenu* m_sourcesMenu;
	VtkUtils::TableModel* m_tableModel = nullptr;

    vtkSmartPointer<vtkActor> m_surfaceActor;
    vtkSmartPointer<vtkActor> m_outlineActor;
    vtkSmartPointer<vtkActor> m_contourActor;
    vtkSmartPointer<vtkActor> m_contourLinesActor;
    vtkSmartPointer<vtkActor> m_edgeActor;
    vtkSmartPointer<vtkActor2D> m_contourLabelsActor;

    VtkUtils::VtkWidget* m_vtkWidget = nullptr;
    vtkSmartPointer<vtkOrientationMarkerWidget> m_oriMarkerWidget;
    vtkSmartPointer<vtkScalarBarActor> m_scalarBar;
    vtkSmartPointer<vtkCubeAxesActor2D> m_cubeAxes;
    vtkSmartPointer<vtkDelaunay2D> m_del;
	vtkPolyData* m_polyData = nullptr;

    qreal m_scalarMin = 0.0;
    qreal m_scalarMax = 0.0;

    int m_numOfScalarColors = 10;
    int m_numOfContourLines = 10;

	bool meshMode = true;
};

#endif // RENDERSURFACE_H

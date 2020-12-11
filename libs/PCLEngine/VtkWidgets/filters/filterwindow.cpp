#include "filterwindow.h"
#include "ui_generalfilterwindow.h"

#include <VtkUtils/point3f.h>
#include <VtkUtils/tablemodel.h>
#include <VtkUtils/modeltopointsconverter.h>

#include <VtkUtils/pointstopolydataconverter.h>
#include <VtkUtils/filereaderutils.h>
#include <VtkUtils/vtkwidget.h>
#include <VtkUtils/vtkutils.h>

#include <vtkDataSetMapper.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkDataSet.h>
#include <vtkUnstructuredGrid.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkScalarBarActor.h>
#include <vtkLookupTable.h>
#include <vtkTextProperty.h>
#include <vtkLODActor.h>
#include <vtkOutlineFilter.h>

#include <QFileInfo>
#include <QThreadPool>
#include <QDebug>
#include <QFileDialog>
#include <QThreadPool>

// ECV_DB_LIB
#include <ecvHObject.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

FilterWindow::FilterWindow(QWidget *parent) 
	: QWidget(parent)
	, m_ui(new Ui::GeneralFilterWindow)
{
	setObjectName("FilterWindow");
}

FilterWindow::~FilterWindow()
{
	VtkUtils::vtkSafeDelete(m_dataObject);

	if (m_vtkWidget)
	{
		m_vtkWidget->deleteLater();
	}

	if (m_tableModel)
	{
		delete m_tableModel;
		m_tableModel = nullptr;
	}

	if (m_ui)
	{
		delete m_ui;
		m_ui = nullptr;
	}
}

void FilterWindow::setFileName(const QString& fileName)
{
	if (m_fileName != fileName) {
		m_fileName = fileName;
		readFile();
	}
}

QString FilterWindow::fileName() const
{
	return m_fileName;
}

void FilterWindow::update()
{
	m_vtkWidget->update();
	QWidget::update();
}

bool FilterWindow::setInput(const ccHObject * obj)
{
	if (!initTableModel(obj))
	{
		return false;
	}
	fireupModelToPointsConverter();
	return true;
}

ccHObject* FilterWindow::getOutput() const
{
	vtkDataObject* vtkData = resultData();
	if (!vtkData)
	{
		return nullptr;
	}
	vtkPolyData * polydata = vtkPolyData::SafeDownCast(vtkData);
    if (!polydata)
	{
		return nullptr;
	}

	ccHObject* container = new ccHObject();
    ccMesh* mesh = nullptr;
    ccPointCloud* vertices = nullptr;

	vtkIdType ptsCount = polydata->GetNumberOfPoints();
	vtkIdType polyCount = polydata->GetNumberOfPolys();
	vtkPoints* vtkpoints = polydata->GetPoints();
	vtkCellArray* cell_array = polydata->GetPolys();
	if (ptsCount > 0)
	{
		vertices = new ccPointCloud("vertices");
        if (!vertices->reserve(static_cast<unsigned>(ptsCount)))
		{
			CVLog::Error("not enough memory to allocate vertices...");
			return nullptr;
		}

		for (vtkIdType iPt = 0; iPt < ptsCount; ++iPt)
		{
			CCVector3 P = CCVector3::fromArray(vtkpoints->GetPoint(iPt));
			vertices->addPoint(P);
		}
		// end POINTS
	}
	if (polyCount > 0)
	{
		mesh = new ccMesh(vertices);

		//add the triangles
		vtkIdType indexes[3];
		vtkIdType * tempInd = indexes;
		vtkIdType * &tempIndexes = tempInd;
		vtkIdType iPt = 0;
		while (cell_array->GetNextCell(iPt, tempIndexes) != 0)
		{
			if (mesh->size() + 1 > mesh->capacity())
			{
				if (!mesh->reserve(mesh->size() + 1 + 256)) //take some advance to avoid too many allocations
				{
					CVLog::Error("not enough memory to allocate mesh...");
					return nullptr;
				}
			}

			//triangle or quad
            mesh->addTriangle(static_cast<unsigned>(indexes[0]),
                              static_cast<unsigned>(indexes[1]),
                              static_cast<unsigned>(indexes[2]));
		}

		if (mesh->size() != 0)
		{
			mesh->shrinkToFit();
		}
		// end POLYGONS or TRIANGLE_STRIPS
	}

	if (vertices && vertices->size() == 0)
	{
		delete vertices;
        vertices = nullptr;
	}

    if (mesh && (mesh->size() == 0 || !vertices))
	{
		delete mesh;
        mesh = nullptr;
	}

	if (mesh)
	{
		container->addChild(mesh);
		mesh->setVisible(true);

		mesh->addChild(vertices);
		vertices->setEnabled(false);
		vertices->setName("Vertices");
		vertices->setLocked(true); //DGM: no need to lock it as it is only used by one mesh!

		mesh->showNormals(mesh->hasNormals());
		if (vertices->hasScalarFields())
		{
			vertices->setCurrentDisplayedScalarField(0);
			mesh->showSF(true);
		}
		if (vertices->hasColors())
			mesh->showColors(true);
	}
	else if (vertices)
	{
		container->addChild(vertices);
		vertices->setVisible(true);
		if (vertices->hasNormals())
			vertices->showNormals(true);
		if (vertices->hasScalarFields())
		{
			vertices->setCurrentDisplayedScalarField(0);
			vertices->showSF(true);
		}
		if (vertices->hasColors())
			vertices->showColors(true);
	}
	else
	{
		container = nullptr;
	}
	return container;
}

bool FilterWindow::initTableModel(const ccHObject * obj)
{
	if (!m_tableModel)
	{
		m_tableModel = new VtkUtils::TableModel(obj, this);
	}
	else
	{
		m_tableModel->updateData(obj);
	}

	if (!m_tableModel)
	{
		return false;
	}
	
	m_tableModel->setHorizontalHeaderData(QVariantList() << "X" << "Y" << "Z");
	emit m_tableModel->layoutChanged();
	return true;
}

void FilterWindow::initTableModel()
{
	if (m_tableModel)
		return;

	m_tableModel = new VtkUtils::TableModel(3, DefaultRows, this);
	m_tableModel->setHorizontalHeaderData(QVariantList() << "X" << "Y" << "Z");
	m_tableModel->random();
	emit m_tableModel->layoutChanged();
}

void FilterWindow::randomTableModel()
{
	if (!m_tableModel)
		return;

	m_tableModel->random();
	emit m_tableModel->layoutChanged();
	fireupModelToPointsConverter();
}

void FilterWindow::onObjFileReaderFinished()
{
	VtkUtils::ObjFileReader* stlReader = qobject_cast<VtkUtils::ObjFileReader*>(sender());
	createActorFromData<vtkPolyData, vtkPolyDataMapper>(stlReader->dataObject());
	stlReader->deleteLater();
}

void FilterWindow::onDynaFileReaderFinished()
{
}

void FilterWindow::onFluentFileReaderFinished()
{
	VtkUtils::FluentFileReader* fluentReader = qobject_cast<VtkUtils::FluentFileReader*>(sender());
	vtkMultiBlockDataSet* output = vtkMultiBlockDataSet::New();
	output->ShallowCopy(fluentReader->dataObject());

	createActorFromData<vtkUnstructuredGrid, vtkDataSetMapper>(output->GetBlock(0));
	fluentReader->deleteLater();
}

void FilterWindow::onVrmlFileReaderFinished()
{
}

void FilterWindow::onStlFileReaderFinished()
{
	VtkUtils::StlFileReader* stlReader = qobject_cast<VtkUtils::StlFileReader*>(sender());
	createActorFromData<vtkPolyData, vtkPolyDataMapper>(stlReader->dataObject());
	stlReader->deleteLater();
}

void FilterWindow::onVtkFileReaderFinished()
{
	VtkUtils::VtkFileReader* vtkReader = qobject_cast<VtkUtils::VtkFileReader*>(sender());
	vtkDataSet* dataSet = vtkDataSet::SafeDownCast(vtkReader->dataObject());
	vtkPolyData* polyData = vtkPolyData::SafeDownCast(vtkReader->dataObject());
	if (dataSet)
		createActorFromData<vtkDataSet, vtkDataSetMapper>(vtkReader->dataObject());
	else if (polyData)
		createActorFromData<vtkPolyData, vtkPolyDataMapper>(vtkReader->dataObject());
	vtkReader->deleteLater();
}

void FilterWindow::onNastranFileReaderFinished()
{
	VtkUtils::NastranFileReader* nastranReader = qobject_cast<VtkUtils::NastranFileReader*>(sender());
	createActorFromData<vtkDataSet, vtkDataSetMapper>(nastranReader->dataObject());
	nastranReader->deleteLater();
}

void FilterWindow::onAnsysFileReaderFinished()
{
	VtkUtils::AnsysFileReader* ansysReader = qobject_cast<VtkUtils::AnsysFileReader*>(sender());
	vtkDataSet* dataSet = vtkDataSet::SafeDownCast(ansysReader->dataObject());
	vtkPolyData* polyData = vtkPolyData::SafeDownCast(ansysReader->dataObject());
	if (dataSet)
		createActorFromData<vtkDataSet, vtkDataSetMapper>(ansysReader->dataObject());
	else if (polyData)
		createActorFromData<vtkPolyData, vtkPolyDataMapper>(ansysReader->dataObject());
	ansysReader->deleteLater();
}

void FilterWindow::onPlyFileReaderFinished()
{
	VtkUtils::PlyFileReader* plyReader = qobject_cast<VtkUtils::PlyFileReader*>(sender());
	createActorFromData<vtkPolyData, vtkPolyDataMapper>(plyReader->dataObject());
	plyReader->deleteLater();
}

void FilterWindow::setDisplayEffect(FilterWindow::DisplayEffect effect)
{
	if (m_displayEffect != effect) {
		m_displayEffect = effect;
		applyDisplayEffect();
	}
}

FilterWindow::DisplayEffect FilterWindow::displayEffect() const
{
	return m_displayEffect;
}

void FilterWindow::applyDisplayEffect()
{
	if (m_modelActor) {
		switch (m_displayEffect) {
		case Transparent:
			m_modelActor->GetProperty()->SetOpacity(0.3);
			m_modelActor->SetVisibility(1);
			m_modelActor->GetProperty()->SetRepresentationToSurface();
			break;

		case Opaque:
			m_modelActor->SetVisibility(0);
			break;

		case Points:
			m_modelActor->SetVisibility(1);
			m_modelActor->GetProperty()->SetRepresentationToPoints();
			break;

		case Wireframe:
			m_modelActor->SetVisibility(1);
			m_modelActor->GetProperty()->SetRepresentationToWireframe();
			break;
		}

		vtkSmartPointer<vtkLookupTable> lut = createLookupTable(m_scalarMin, m_scalarMax);
		m_modelActor->GetMapper()->SetLookupTable(lut);
		update();
	}
}

void FilterWindow::readFile()
{
	const QFileInfo fi(m_fileName);
	const QString ext = fi.suffix().toLower();

	VtkUtils::AbstractFileReader* fileReader = nullptr;

	if (ext == "obj") {
		fileReader = new VtkUtils::ObjFileReader();
		connect(fileReader, SIGNAL(finished()), this, SLOT(onObjFileReaderFinished()));
	} else if (ext == "vtk") {
		fileReader = new VtkUtils::VtkFileReader();
		connect(fileReader, SIGNAL(finished()), this, SLOT(onVtkFileReaderFinished()));
	} else if (ext == "stl") {
		fileReader = new VtkUtils::StlFileReader();
		connect(fileReader, SIGNAL(finished()), this, SLOT(onStlFileReaderFinished()));
	} else if (ext == "cdb") {
		fileReader = new VtkUtils::AnsysFileReader();
		connect(fileReader, SIGNAL(finished()), this, SLOT(onAnsysFileReaderFinished()));
	} else if (ext == "bdf") {
		fileReader = new VtkUtils::NastranFileReader();
		connect(fileReader, SIGNAL(finished()), this, SLOT(onNastranFileReaderFinished()));
	} else if (ext == "cas") {
		fileReader = new VtkUtils::FluentFileReader();
		connect(fileReader, SIGNAL(finished()), this, SLOT(onFluentFileReaderFinished()));
	} else if (ext == "vrml") {

	} else if (ext == "ply") {
		fileReader = new VtkUtils::PlyFileReader();
		connect(fileReader, SIGNAL(finished()), this, SLOT(onPlyFileReaderFinished()));
	}

	if (!fileReader) {
		qDebug() << "importFile: null file reader.";
		return;
	}

	fileReader->setFileName(m_fileName);
	fileReader->setAutoDelete(false);

	QThreadPool::globalInstance()->start(fileReader);
}

QString FilterWindow::fileFilter() const
{
	static QString filter = "ANSYS Files(*.obj);;VTK Files(*.vtk);;STL Files(*.stl);;ANSYS Files(*.cdb);;"
					 "Fluent Files(*.cas);;Ply Files(*.ply)";
	return filter;
}

void FilterWindow::browseFile()
{
	static QString previousDir = ".";
	QString file = QFileDialog::getOpenFileName(this, tr("Open File"), previousDir, fileFilter());
	if (file.isEmpty())
		return;

	QFileInfo fi(file);
	previousDir = fi.canonicalFilePath();

	setFileName(file);
}

void FilterWindow::showOrientationMarker(bool show)
{
	m_vtkWidget->showOrientationMarker(show);
}

void FilterWindow::showScalarBar(bool show)
{
	if (!m_scalarBar) {
		vtkSmartPointer<vtkLookupTable> lut = createLookupTable(m_scalarMin, m_scalarMax);
		lut->Build();

		m_scalarBar = vtkScalarBarActor::New();
		m_scalarBar->SetWidth(.1); // fraction of window width
		m_scalarBar->SetHeight(.8); // fraction of window height
		m_scalarBar->GetTitleTextProperty()->SetFontSize(3);
		m_scalarBar->GetTitleTextProperty()->SetBold(0);
		m_scalarBar->GetTitleTextProperty()->SetItalic(0);
		m_scalarBar->SetOrientationToVertical();
		m_scalarBar->SetTitle("Scalar");
		m_vtkWidget->addActor(m_scalarBar);
		m_scalarBar->SetLookupTable(lut);
	}

	m_scalarBar->SetVisibility(show);
	update();
}

void FilterWindow::showOutline(bool show)
{
	if (!show && !m_outlineActor)
		return;

	if (!m_dataObject) {
		qDebug() << "FitlerWindow::showOutline: null data object, quit.";
		return;
	}

	VTK_CREATE(vtkOutlineFilter, outline);
	outline->SetInputData(m_dataObject);

	VTK_CREATE(vtkPolyDataMapper, mapper);
	mapper->SetInputConnection(outline->GetOutputPort());

	VtkUtils::vtkInitOnce(m_outlineActor);
	m_outlineActor->SetMapper(mapper);

	m_vtkWidget->defaultRenderer()->AddActor(m_outlineActor);
	update();
}

void FilterWindow::setOutlineColor(const QColor &clr)
{
	if (!m_outlineActor)
		return;

	double vtkClr[3];
	Utils::vtkColor(clr, vtkClr);
	m_outlineActor->GetProperty()->SetColor(vtkClr);
	update();
}

bool FilterWindow::isValidPolyData() const
{
	return vtkPolyData::SafeDownCast(m_dataObject) != nullptr;
}

bool FilterWindow::isValidDataSet() const
{
	return vtkDataSet::SafeDownCast(m_dataObject) != nullptr;
}

void FilterWindow::setResultData(vtkDataObject* data)
{
	m_resultData = data;
}

vtkDataObject* FilterWindow::resultData() const
{
	return m_resultData;
}

void FilterWindow::setScalarBarColors(const QColor& clr1, const QColor& clr2)
{
	if (m_color1 == clr1 && m_color2 == clr2)
		return;

	m_color1 = clr1;
	m_color2 = clr2;
	colorsChanged();
}

QColor FilterWindow::color1() const
{
	return m_color1;
}

QColor FilterWindow::color2() const
{
	return m_color2;
}

void FilterWindow::setScalarRange(double min, double max)
{
	m_scalarMin = qMin(min, max);
	m_scalarMax = qMax(min, max);
}

double FilterWindow::scalarMin() const
{
	return m_scalarMin;
}

double FilterWindow::scalarMax() const
{
	return m_scalarMax;
}

vtkSmartPointer<vtkLookupTable> FilterWindow::createLookupTable(double min, double max)
{
	double hsv1[3];
	double hsv2[3];
	Utils::qColor2HSV(m_color1, hsv1);
	Utils::qColor2HSV(m_color2, hsv2);

	VTK_CREATE(vtkLookupTable, lut);
	lut->SetHueRange(hsv1[0], hsv2[0]);
	lut->SetSaturationRange(hsv1[1], hsv2[1]);
	lut->SetValueRange(hsv1[2], hsv2[2]);
	lut->SetTableRange(min, max);
	lut->Build();

	return lut;
}

void FilterWindow::modelReady()
{
	showOrientationMarker();
	showOutline();
	m_vtkWidget->defaultRenderer()->ResetCamera();
}

void FilterWindow::createUi()
{
	m_vtkWidget = new VtkUtils::VtkWidget(this);
	m_vtkWidget->showMaximized();
	m_vtkWidget->update();
	m_ui->previewLayout->addWidget(m_vtkWidget);
}

void FilterWindow::dataChanged()
{
	// no impl
}

void FilterWindow::colorsChanged()
{
	vtkSmartPointer<vtkLookupTable> lut = createLookupTable(m_scalarMin, m_scalarMax);
	if (m_scalarBar)
		m_scalarBar->SetLookupTable(lut);
	if (m_modelActor)
		m_modelActor->GetMapper()->SetLookupTable(lut);
	if (m_filterActor)
		m_filterActor->GetMapper()->SetLookupTable(lut);
	if (m_vtkWidget)
		m_vtkWidget->update();
}

void FilterWindow::fireupModelToPointsConverter()
{
	VtkUtils::ModelToPointsConverter* converter = new VtkUtils::ModelToPointsConverter(m_tableModel);
	converter->setAutoDelete(false);
	connect(converter, SIGNAL(finished()), this, SLOT(onModelToPointsConverterFinished()));
	QThreadPool::globalInstance()->start(converter);
}

void FilterWindow::onModelToPointsConverterFinished()
{
	VtkUtils::ModelToPointsConverter* lastConverter = qobject_cast<VtkUtils::ModelToPointsConverter*>(sender());
	QVector<VtkUtils::Point3F> points = lastConverter->points().toVector();
	QVector<Tuple3ui> vertices = lastConverter->vertices();
	meshMode = vertices.isEmpty() ? false : true;

	VtkUtils::PointsToPolyDataConverter* newConverter = new VtkUtils::PointsToPolyDataConverter(points, vertices);
	newConverter->setAutoDelete(false);
	connect(newConverter, SIGNAL(finished()), this, SLOT(onPointsToPolyDataConverterFinished()));
	QThreadPool::globalInstance()->start(newConverter);

	lastConverter->deleteLater();
}

void FilterWindow::onPointsToPolyDataConverterFinished()
{
	if (m_dataObject) {
		m_dataObject->Delete();
		m_dataObject = nullptr;
	}

	VtkUtils::PointsToPolyDataConverter* converter = qobject_cast<VtkUtils::PointsToPolyDataConverter*>(sender());
	vtkPolyData* polydata = converter->polyData();

	VtkUtils::vtkInitOnce(m_modelActor);
	VTK_CREATE(vtkPolyDataMapper, mapper);

	if (!m_dataObject)
		m_dataObject = vtkPolyData::New();

	// copy surface data
	if (meshMode)
	{
		m_dataObject->DeepCopy(polydata);
		mapper->SetInputData(polydata);
		m_modelActor->SetMapper(mapper);
	}
	else {
		VTK_CREATE(vtkCellArray, vertices);
		polydata->SetVerts(vertices);
		// Create the supporting structures
		vertices = polydata->GetVerts();
		if (!vertices)
			vertices = vtkSmartPointer<vtkCellArray>::New();

		vtkSmartPointer<vtkIdTypeArray> cells = vertices->GetData ();

		// If no init cells and cells has not been initialized...
		if (!cells)
			cells = vtkSmartPointer<vtkIdTypeArray>::New();

		vtkIdType nr_points = polydata->GetNumberOfPoints();
		// If we have less values then we need to recreate the array
		if (cells->GetNumberOfTuples() < nr_points)
		{
			cells = vtkSmartPointer<vtkIdTypeArray>::New();

			// If the number of tuples is still too small, we need to recreate the array
			cells->SetNumberOfComponents(2);
			cells->SetNumberOfTuples(nr_points);
			vtkIdType *cell = cells->GetPointer(0);
			for (vtkIdType i = 0; i < nr_points; ++i, cell += 2)
			{
				*cell = 1;
				*(cell + 1) = i;
			}
		}
		else
		{
			// The assumption here is that the current set of cells has more data than needed
			cells->SetNumberOfComponents(2);
			cells->SetNumberOfTuples(nr_points);
		}

		// Set the cells and the vertices
		vertices->SetCells(nr_points, cells);
		m_vtkWidget->createActorFromVTKDataSet(polydata, m_modelActor);

		vtkPolyData* tempData = reinterpret_cast<vtkPolyDataMapper*>(m_modelActor->GetMapper())->GetInput();
		m_dataObject->DeepCopy(tempData);
	}

	
	m_vtkWidget->defaultRenderer()->AddActor(m_modelActor);
	update();

	converter->deleteLater();

	apply();
	modelReady();
	dataChanged();
}

template <class DataObject, class Mapper>
void FilterWindow::createActorFromData(vtkDataObject* dataObj)
{
    if (!dataObj)
        return;

    m_dataObject = dataObj;
    DataObject* data = DataObject::SafeDownCast(dataObj);

    if (!data) {
        qDebug() << "ansys parsing: null data set.";
        return;
    }

    VTK_CREATE(Mapper, mapper);
    mapper->SetInputData(data);
    mapper->Update();

    VtkUtils::vtkInitOnce(m_modelActor);
    m_modelActor->SetMapper(mapper);

    m_modelActor->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, data->GetNumberOfPoints() / 10)));
    m_modelActor->GetProperty()->SetInterpolationToFlat();

    m_vtkWidget->defaultRenderer()->AddActor(m_modelActor);
    m_vtkWidget->defaultRenderer()->ResetCamera();
    update();

    modelReady();
    dataChanged();
}

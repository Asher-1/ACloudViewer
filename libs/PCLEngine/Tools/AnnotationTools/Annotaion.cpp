#include <Tools/AnnotationTools/Annotaion.h>
#include <Tools/AnnotationTools/vtkAnnotationBoxSource.h>
#include <Tools/AnnotationTools/vtkBoxWidgetRestricted.h>
#include <Tools/AnnotationTools/vtkBoxWidgetCallback.h>

#include <vtkBalloonWidget.h>
#include <vtkBalloonRepresentation.h>

#include <vtkLookupTable.h>
#include <vtkFloatArray.h>
#include <vtkProperty.h>
#include <vtkCellData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>

// CV_CORE_LIB
#include <CVLog.h>
#include <CVTools.h>

// ECV_DB_LIB
#include <ecvColorTypes.h>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif

// SYSTEM
#include <QFile>
#include <map>

using namespace std;

// static variable
std::vector<std::string>* Annotation::types = nullptr;

Annotation::Annotation(const std::vector<int>& slice, std::string type_)
{
	assert(!slice.empty());
	this->type = type_;
	this->m_slice.clear();
	this->m_slice.assign(slice.begin(), slice.end());
	this->actor = nullptr;
	this->boxWidget = nullptr;
}

Annotation::Annotation(const BoxLabel &label, bool visible_, bool lock_)
	:visible(visible_),lock(lock_){
	//type
	type = label.type;

	// init variable
	initial();

	// apply transform
	vtkSmartPointer<vtkTransform> cubeTransform = vtkSmartPointer<vtkTransform>::New();
	cubeTransform->PostMultiply();
	cubeTransform->Scale(label.detail.length,label.detail.width,label.detail.height);
	cubeTransform->RotateY(label.detail.yaw * 180 / vtkMath::Pi());
	cubeTransform->Translate(label.detail.center_x,label.detail.center_y,label.detail.center_z);

	applyTransform(cubeTransform);
}

Annotation::Annotation(const PointCloudI::Ptr cloud, vector<int> &slice, std::string type_)  
{
	double p1[3];
	double p2[3];
	ComputeOBB(cloud, slice, p1,p2);
	BoxLabel label(p1, p2, type_);

	setAnchorPoint(cloud, slice);

	this->type=type_;

	this->m_slice.clear();
	this->m_slice.assign(slice.begin(), slice.end());

	// init variable
	initial();

	// apply transform
	vtkSmartPointer<vtkTransform> cubeTransform = vtkSmartPointer<vtkTransform>::New();
	cubeTransform->PostMultiply();
	cubeTransform->Scale(label.detail.length,label.detail.width,label.detail.height);
	cubeTransform->RotateY(label.detail.yaw * 180 / vtkMath::Pi());
	cubeTransform->Translate(label.detail.center_x,label.detail.center_y,label.detail.center_z);

	applyTransform(cubeTransform);
}

Annotation::~Annotation(){
	// release anchorPoints
	for (auto p:anchorPoints){
		delete[] p;
	}
	anchorPoints.clear();
}

void Annotation::initial() {
	// Cube
	source = vtkSmartPointer<vtkAnnotationBoxSource>::New();
	mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(source->GetOutputPort());
	actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	transform = vtkSmartPointer<vtkTransform>::New();
	colorAnnotation();
}

void Annotation::colorAnnotation(int color_index) {
	if (!this->actor) return;

	vtkSmartPointer<vtkLookupTable> lut =
		vtkSmartPointer<vtkLookupTable>::New();
	lut->SetNumberOfTableValues(2);
	lut->Build();
	if (color_index < 0) {
		ecvColor::Rgb c = ecvColor::LookUpTable::at(GetTypeIndex(type));
		lut->SetTableValue(0, c.r / 255.0, c.g / 255.0, c.b / 255.0, 0);
		lut->SetTableValue(1, c.r / 255.0, c.g / 255.0, c.b / 255.0, 1);
	}
	else {
		//pcl::RGB c = pcl::GlasbeyLUT::at(color_index);
		ecvColor::Rgb c = ecvColor::LookUpTable::at(color_index);
		lut->SetTableValue(0, c.r / 255.0, c.g / 255.0, c.b / 255.0, 0);
		lut->SetTableValue(1, c.r / 255.0, c.g / 255.0, c.b / 255.0, 1);
	}

	vtkSmartPointer<vtkFloatArray> cellData =
		vtkSmartPointer<vtkFloatArray>::New();

	//line color
	for (int i = 0; i < 12; i++)
	{
		cellData->InsertNextValue(1);
	}

	//face color
	for (int i = 0; i < 6; i++)
	{
		cellData->InsertNextValue(0);
	}

	// plusX face
	cellData->InsertValue(12, 1);

	source->Update();
	source->GetOutput()->GetCellData()->SetScalars(cellData);

	actor->GetProperty()->SetLineWidth(2);
	actor->GetProperty()->SetLighting(false);

	mapper->SetLookupTable(lut);
}

void Annotation::setAnchorPoint(const PointCloudI::Ptr cloud, const std::vector<int> &slice)
{
	if (cloud->size() == 0 || slice.empty()) return;

	int num = static_cast<int>(slice.size());
	anchorPoints.clear();
	anchorPoints.resize(num);
	
	if (num != 0)
	{
#ifdef USE_TBB
		tbb::parallel_for(0, num, [&](int dataIndex)
#else

#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int dataIndex = 0; dataIndex < num; ++dataIndex)
#endif
		{
			double* p = new double[3];
			p[0] = cloud->points[slice[dataIndex]].x;
			p[1] = cloud->points[slice[dataIndex]].y;
			p[2] = cloud->points[slice[dataIndex]].z;
			anchorPoints[dataIndex] = p;
		}
#ifdef USE_TBB
		);
#endif
	}
}

BoxLabel Annotation::getBoxLabel(){
	BoxLabel label;
	label.type=type;

	double p[3];
	double s[3];
	double o[3];
	transform->GetPosition(p);
	transform->GetScale(s);
	transform->GetOrientation(o);
	memcpy(label.data, p, 3*sizeof(double));
	memcpy(label.data+3, s,3*sizeof(double));
	label.detail.yaw=o[1] / 180 * vtkMath::Pi();
	return label;
}

void Annotation::applyTransform(vtkSmartPointer<vtkTransform> t){
	if (this->actor)
	{
		if (t == transform) return;

		transform = t;
		actor->SetUserTransform(t);
	}
}

void Annotation::picked(vtkRenderWindowInteractor *interactor){
	// enable box widget
	if (!this->actor) return;

	// enable box widget
	if (!boxWidget)
	{
		boxWidget = vtkSmartPointer<vtkBoxWidgetRestricted>::New();
		boxWidgetCallback0 = vtkSmartPointer<vtkBoxWidgetCallback0>::New();
		boxWidgetCallback0->setAnno(this);
		boxWidgetCallback1 = vtkSmartPointer<vtkBoxWidgetCallback1>::New();
		boxWidgetCallback1->setAnno(this);

		boxWidget->SetInteractor(interactor);
		// boxWidget->SetProp3D( cubeActor );
		// boxWidget->SetPlaceFactor( 1.25 ); // Make the box 1.25x larger than the actor
		// boxWidget->PlaceWidget();

		// default is [-0.5, 0.5], NOTE
		// [-1,1] makes boxwidget fit to annotion,
		// but [-0.5, 0.5] should be in the correct way, may be some bug
		double bounds[6] = { -1,1,-1,1,-1,1 };
		boxWidget->PlaceWidget(bounds);

		boxWidget->SetHandleSize(0.01);
		boxWidget->GetOutlineProperty()->SetAmbientColor(1.0, 0.0, 0.0);
		boxWidget->AddObserver(vtkCommand::InteractionEvent, boxWidgetCallback0);
		boxWidget->AddObserver(vtkCommand::EndInteractionEvent, boxWidgetCallback1);
	}

	vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();
	t->DeepCopy(actor->GetUserTransform());
	boxWidget->SetTransform(t);
	boxWidget->On();
}

void Annotation::unpicked(){
	if (this->boxWidget)
	{
		boxWidget->Off();
	}
}

void Annotation::adjustToAnchor(){
	if (anchorPoints.size()==0) return;

	transform->GetPosition(center);

	double r[3], x[3], z[3], y[3] = {0,1,0};
	double s[3]; // scale

	transform->GetOrientation(r);
	// direction
	z[0] = std::sin(vtkMath::RadiansFromDegrees(r[1]));
	z[1] = 0;
	z[2] = std::cos(vtkMath::RadiansFromDegrees(r[1]));
	vtkMath::Cross(y, z, x);

	double scs[2];
	s[0]=computeScaleAndCenterShift(x,scs);
	vtkMath::MultiplyScalar(x,scs[1]);
	s[1]=computeScaleAndCenterShift(y,scs);
	vtkMath::MultiplyScalar(y,scs[1]);
	s[2]=computeScaleAndCenterShift(z,scs);
	vtkMath::MultiplyScalar(z,scs[1]);

	// apply center shift
	vtkMath::Add(center,x,center);
	vtkMath::Add(center,y,center);
	vtkMath::Add(center,z,center);

	vtkSmartPointer<vtkTransform> t=vtkSmartPointer<vtkTransform>::New();
	t->Translate(center);
	t->RotateY(r[1]);
	t->Scale(s);

	boxWidget->SetTransform(t);
	applyTransform(t);
}

std::string Annotation::getType() const
{
	return type;
}

const std::vector<int>& Annotation::getSlice() const
{
	return m_slice;
}

void Annotation::setType(const std::string value)
{
	if (value!=type){
		type = value;
		if (this->actor)
		{
			colorAnnotation();
		}
	}
}

vtkSmartPointer<vtkActor> Annotation::getActor() const
{
	return actor;
}

double Annotation::computeScaleAndCenterShift(double o[], double scs[]){
	vtkMath::Normalize(o);

	double a,b;
	a=-std::numeric_limits <double>::max ();
	b=std::numeric_limits <double>::max ();

	double t[3];
	for (auto x:anchorPoints){
		vtkMath::Subtract(x,this->center,t);
		double s=vtkMath::Dot(t,o);
		a=std::max(a,s);
		b=std::min(b,s);
	}
	scs[0]=a-b;scs[1]=(a+b)/2;
	return a-b;
}

std::vector<std::string>* Annotation::GetTypes()
{
	if (!types)
	{
		types = new vector<string>();
	}

	return types;
}

int Annotation::GetTypeIndex(std::string type_) 
{
	assert(types);
	for (int i=0;i<types->size();i++)
	{
		if (types->at(i)==type_) return i;
	}

	types->push_back(type_);
	return types->size() - 1;
}

std::string Annotation::GetTypeByIndex(size_t index)
{
	assert(types);
	if (index > types->size() - 1 || index < 0)
	{
		return "";
	}

	return types->at(index);
}

void Annotation::ComputeOBB(const  PointCloudI::Ptr cloud, std::vector<int>& slice,double p1[3], double p2[3])
{
	p1[0]=std::numeric_limits <double>::max ();
	p1[1]=std::numeric_limits <double>::max ();
	p1[2]=std::numeric_limits <double>::max ();

	//std::numeric_limits <double>::min (); is a number close enough to 0
	p2[0]=-std::numeric_limits <double>::max ();
	p2[1]=-std::numeric_limits <double>::max ();
	p2[2]=-std::numeric_limits <double>::max ();

	for (auto i:slice){
		p1[0]=std::min(p1[0],(double)cloud->points[i].x);
		p1[1]=std::min(p1[1],(double)cloud->points[i].y);
		p1[2]=std::min(p1[2],(double)cloud->points[i].z);

		p2[0]=std::max(p2[0],(double)cloud->points[i].x);
		p2[1]=std::max(p2[1],(double)cloud->points[i].y);
		p2[2]=std::max(p2[2],(double)cloud->points[i].z);
	}
}

Annotaions::Annotaions(vtkRenderWindowInteractor * interactor)
	:m_interactor(interactor)
{
	if (interactor && !m_balloonWidget)
	{
		vtkSmartPointer<vtkBalloonRepresentation> balloonRep =
			vtkSmartPointer<vtkBalloonRepresentation>::New();
		balloonRep->SetBalloonLayoutToImageRight();

		m_balloonWidget = vtkSmartPointer<vtkBalloonWidget>::New();
		m_balloonWidget->SetInteractor(interactor);
		m_balloonWidget->SetRepresentation(balloonRep);
		m_balloonWidget->On();
	}
}

void Annotaions::preserve(size_t num)
{
	if (num != 0)
	{
		m_capacity = num;
		m_labeledCloudIndex = new int[m_capacity];
	}
	memset(m_labeledCloudIndex, 0, m_capacity * sizeof(int));
}

void Annotaions::release()
{
	if (m_labeledCloudIndex) {
		delete[] m_labeledCloudIndex;
		m_labeledCloudIndex = nullptr;
	}

	if (m_balloonWidget)
	{
		m_balloonWidget->Off();
	}
	m_capacity = 0;

	if (Annotation::GetTypes())
	{
		delete Annotation::types;
		Annotation::types = nullptr;
	}
}

void Annotaions::add(Annotation *anno) {
	// add balloon tip
	if (m_balloonWidget)
	{
		m_balloonWidget->AddBalloon(anno->getActor(), anno->getType().c_str(), NULL);
	}
	
	// update internal labels
	updateLabels(anno, false);
	// add this anno to annotation manager
	m_annotations.push_back(anno);
}

void Annotaions::remove(Annotation *anno){
	if (m_balloonWidget)
	{
		m_balloonWidget->RemoveBalloon(anno->getActor());
	}
	// reset internal labels
	updateLabels(anno, true);
	m_annotations.erase(std::remove(m_annotations.begin(), m_annotations.end(), anno), m_annotations.end());
}

void Annotaions::clear(){
	for (auto p:m_annotations){
		if (m_balloonWidget)
		{
			m_balloonWidget->RemoveBalloon(p->getActor());
		}
		delete p;
	}
	m_annotations.clear();
	preserve();
}

int Annotaions::getSize(){
	return m_annotations.size();
}

void Annotaions::updateLabels(Annotation * anno, bool resetFlag)
{
	if (!anno) return;

	int num = static_cast<int>(anno->getSlice().size());
	int Annotype = Annotation::GetTypeIndex(anno->getType());
	if (m_labeledCloudIndex && num != 0)
	{
#ifdef USE_TBB
		tbb::parallel_for(0, num, [&](int dataIndex)
#else

#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int dataIndex = 0; dataIndex < num; ++dataIndex)
#endif
		{
			if (resetFlag) // reset labels which are related to this annotation
			{
				m_labeledCloudIndex[anno->getSlice()[dataIndex]] = 0;
			}
			else
			{
				m_labeledCloudIndex[anno->getSlice()[dataIndex]] = Annotype;
			}

		}
#ifdef USE_TBB
		);
#endif
	}
}

typedef map<size_t, std::vector<int> > AnnotationMap;

void Annotaions::loadAnnotations(string filename, int mode)
{
	m_annotations.clear();

	if (mode == 0) // semantic annotation
	{
		CVTools::TimeStart();
		std::vector<size_t> indices;
		if (!CVTools::QMappingReader(filename, indices))
		{
			return;
		}

		AnnotationMap annotationsMap;
		size_t i = 0;
		for (i = 0; i < indices.size(); ++i)
		{
			if (annotationsMap.find(indices[i]) == annotationsMap.end())
			{
				annotationsMap[indices[i]] = vector<int>();
			}
			annotationsMap[indices[i]].push_back(static_cast<int>(i));
		}

		CVLog::Print(QString("loadAnnotations: finish cost %1 s").arg(CVTools::TimeOff()));
		if (i != m_capacity)
		{
			CVLog::Warning("label files are probably corrupted and drop it!");
			return;
		}

		if (!annotationsMap.empty())
		{
			for (AnnotationMap::iterator iter_int = annotationsMap.begin(); 
				iter_int != annotationsMap.end(); iter_int++)
			{
				size_t typeIndex = iter_int->first;
				string type = Annotation::GetTypeByIndex(typeIndex);
				if (type == "") continue;
				vector<int> slice = iter_int->second;
				add(new Annotation(slice, type));
			}
		}
		
	}
	else if (mode == 1) // bbox annotation
	{
		std::ifstream input(filename.c_str(), std::ios::in);
		if(!input.good()){
			CVLog::Error(QString("Cannot open file : %1").arg(filename.c_str()));
			return;
		}
		std::string type;
		while (input >> type) {
			BoxLabel label;
			label.type = type;
			for (int j = 0; j < 7; j++) {
				input >> label.data[j];
			}
			add(new Annotation(label));
		}
		input.close();
	}
}

void Annotaions::saveAnnotations(string filename, int mode)
{
	if (m_annotations.empty()) return;

#ifdef _WIN32
	CVTools::TimeStart();
	std::string ss;
	if (mode == 0) // semantic annotation
	{
		for (size_t i = 0; i < m_capacity; ++i)
		{
			ss += (std::to_string(m_labeledCloudIndex[i]) + "\n");
		}
	}
	else if (mode == 1) // bbox annotation
	{
		for (auto anno : m_annotations) {
			ss += (anno->getBoxLabel().toString() + "\n");
		}
	}
	else
	{
		CVLog::Warning(QString("unknown mode: %1, only semantic and box supported!").arg(mode));
		return;
	}

	if (!CVTools::FileMappingWriter(filename, (void*)ss.c_str(), strlen(ss.c_str())))
	{
		CVLog::Warning("save annotation file failed!");
		return;
	}
	CVLog::Print(QString("Save annotation file cost %1 s").arg(CVTools::TimeOff()));

#else
	std::ofstream output(filename.c_str(), std::ios_base::out);
	if (mode == 0) // semantic annotation
	{
		for (size_t i = 0; i < m_capacity; ++i)
		{
			output << m_labeledCloudIndex[i] << std::endl;
		}
	}
	else if (mode == 1) // bbox annotation
	{
		for (auto anno : m_annotations) {
			output << anno->getBoxLabel().toString() << std::endl;
		}
	}

	output.close();
#endif
}

bool Annotaions::getAnnotations(std::vector<int>& annos) const
{
	if (!m_labeledCloudIndex || m_capacity == 0) return false;

	annos.resize(m_capacity);
	memcpy(annos.data(), m_labeledCloudIndex, m_capacity * sizeof(int));

	return true;
}

std::vector<Annotation *>& Annotaions::getAnnotations()
{
	return m_annotations;
}

Annotation *Annotaions::getAnnotation(vtkActor *actor) {
	for (auto* anno : m_annotations)
	{
		if (anno->getActor() == actor) {
			return anno;
		}
	}
	return 0;
}

int Annotaions::getAnnotationIndex(Annotation * anno)
{
	int index = -1;
	if (!anno->getActor()) return index;

	for (auto anno2 : m_annotations)
	{
		index++;
		if (anno2->getActor() == anno->getActor()) {
			return index;
		}
	}
	return -1;
}

Annotation *Annotaions::getAnnotation(int index) {
	if (index < 0 || index > getSize() - 1) return nullptr;
	return m_annotations[index];
}

void Annotaions::getAnnotations(
	const std::string& type, 
	std::vector<Annotation*>& annotations)
{
	for (auto anno2 : m_annotations)
	{
		if (anno2->getType() == type) {
			annotations.push_back(anno2);
		}
	}
}

void Annotaions::updateBalloonByIndex(int index)
{
	if (index < 0 || index > getSize() - 1) return;
	if (!getAnnotation(index) || !getAnnotation(index)->getActor()) return;

	if (m_balloonWidget)
	{
		m_balloonWidget->RemoveBalloon(getAnnotation(index)->getActor());
		m_balloonWidget->AddBalloon(getAnnotation(index)->getActor(), getAnnotation(index)->getType().c_str(), NULL);
	}
}

void Annotaions::updateBalloonByAnno(Annotation * anno)
{
	if (!m_balloonWidget || !anno->getActor()) return;

	updateBalloonByIndex(getAnnotationIndex(anno));
}

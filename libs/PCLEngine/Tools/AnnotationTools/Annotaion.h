#ifndef Annotaions_H
#define Annotaions_H

#include <PclUtils/PCLCloud.h>
#include <vtkSmartPointer.h>
#include <vector>

struct BoxLabel
{
	BoxLabel(){
		type="unknown";
		this->detail.center_x=this->detail.center_y=this->detail.center_z=0;
		this->detail.yaw=2;
		this->detail.length=this->detail.width=this->detail.height=1;
	}
	BoxLabel(const double p1[3],const double p2[3], std::string type_="unknown"){
		type=type_;
		this->detail.center_x=(p1[0]+p2[0])/2;
		this->detail.center_y=(p1[1]+p2[1])/2;
		this->detail.center_z=(p1[2]+p2[2])/2;
		this->detail.yaw=0;
		this->detail.length=p2[0]-p1[0];
		this->detail.width=p2[1]-p1[1];
		this->detail.height=p2[2]-p1[2];
	}
	std::string type;
	union{
		double data[7];
		struct{
			double center_x;
			double center_y;
			double center_z;
			double length;
			double width;
			double height;
			double yaw;
		} detail;
	};

	std::string toString(){
		char buffer [200];
		sprintf(buffer,"%s %f %f %f %f %f %f %f",
				type.c_str(),data[0],data[1],data[2],data[3],data[4],data[5],data[6]);
		return std::string(buffer);
	}
};


class vtkBalloonWidget;
class vtkBoxWidgetCallback0;
class vtkBoxWidgetCallback1;
class vtkAnnotationBoxSource;
class vtkBoxWidgetRestricted;

class vtkTransform;
class vtkRenderWindowInteractor;
class vtkActor;
class vtkPolyDataMapper;

class Annotation
{
	friend class Annotaions;
public:
	/**
	 * @brief Annotation  construct from slice which load from label file
	 * @param slice
	 * @param type_
	 */
	Annotation(const std::vector<int>& slice, std::string type_);

	/**
	 * @brief Annotation  construct from boxlabel which load from label file
	 * @param label
	 * @param visible_
	 * @param lock_
	 */
	Annotation(const BoxLabel &label, bool visible_ = true, bool lock_ = false);

	/**
	 * @brief Annotation construct from part of cloud points
	 * @param cloud
	 * @param slice
	 * @param type_
	 */
	Annotation(const PointCloudI::Ptr cloud, std::vector<int> & slice, std::string type_);

	~Annotation();

	/**
	 * @brief getBoxLabel get boxLabel from annotaion tranformation
	 * @return
	 */
	BoxLabel getBoxLabel();

	/**
	 * @brief apply transform to annotation
	 * @param t
	 */
	void applyTransform(vtkSmartPointer<vtkTransform> t);

	/**
	 * @brief enter picked state, show boxwidget which allow to adjust annotation
	 * @param interactor
	 */
	void picked(vtkRenderWindowInteractor* interactor);

	/**
	 * @brief disable boxWidget
	 */
	void unpicked();

	/**
	 * @brief keep current orientation, re-compute the center and scale
	 * to make annotation fit to selected point well enough
	 */
	void adjustToAnchor();

	/**
	 * @brief change the type of annotation, and color too
	 * @param value
	 */
	void setType(const std::string value);
	vtkSmartPointer<vtkActor> getActor() const;
	std::string getType() const;

	const std::vector<int>& getSlice() const;

protected:
	void initial();

	/**
	 * @brief color the annotation with given color
	 * @param color_index
	 * if color_index>=0,refer to @ref pcl::GlasbeyLUT
	 * otherwise use color already mapped by type
	 */
	void colorAnnotation(int color_index=-1);

	/**
	 * @brief copy selected points as anchor to current annotation
	 * @param cloud
	 * @param slice
	 */
	void setAnchorPoint(const PointCloudI::Ptr cloud, const std::vector<int> &slice);

	/**
	 * @brief computeScaleAndCenterShift
	 * @param o direction
	 * @param scs ["scale", "center shift"]
	 * @return scale
	 */
	double computeScaleAndCenterShift(double o[3], double scs[2]);

private:
	std::string type;
	vtkSmartPointer<vtkAnnotationBoxSource> source;
	vtkSmartPointer<vtkActor> actor;
	vtkSmartPointer<vtkPolyDataMapper> mapper;
	vtkSmartPointer<vtkTransform> transform;

	vtkSmartPointer<vtkBoxWidgetRestricted> boxWidget;
	vtkSmartPointer<vtkBoxWidgetCallback0> boxWidgetCallback0;
	vtkSmartPointer<vtkBoxWidgetCallback1> boxWidgetCallback1;

	std::vector<double*> anchorPoints;

	std::vector<int> m_slice;

	double center[3];

	// NOTE not used
	bool visible;
	bool lock;

public:
	/**
	 * @brief get types vector pointer
	 * @return
	 */
	static std::vector<std::string>* GetTypes();

	/**
	 * @brief GetTypeIndex  auto add to vector map if has not
	 * @param type_
	 * @return
	 */
	static int GetTypeIndex(std::string type_);

	/**
	 * @brief GetTypeByIndex  auto add to vector map if has not
	 * @param index
	 * @return
	 */
	static std::string GetTypeByIndex(size_t index);

	/**
	 * @brief ComputeOBB compute max,min [x,y,z] aligned to xyz axis
	 * @param cloud
	 * @param slice
	 * @param p1 min [x,y,z]
	 * @param p2 max [x,y,z]
	 */
	static void ComputeOBB(const PointCloudI::Ptr cloud, std::vector<int>& slice, double p1[3], double p2[3]);

protected:
	/**
	 * @brief types all annotation type here
	 */
	static std::vector<std::string>* types;

};

class Annotaions
{
public:
	explicit Annotaions(vtkRenderWindowInteractor* interactor = nullptr);

	void preserve(size_t num = 0);
	void release();

	void add(Annotation* anno);
	void remove(Annotation* anno);
	void clear();
	int getSize();

	void updateLabels(Annotation* anno, bool resetFlag = false);

	/**
	 * @brief load annotations from file
	 * @param filename
	 */
	void loadAnnotations(std::string filename, int mode);

	/**
	 * @brief save annotations to file
	 * @param filename
	 */
	void saveAnnotations(std::string filename, int mode);

	bool getAnnotations(std::vector<int>& annos) const;

	/**
	 * @brief from annotatin box actor to find annotation itself
	 * @param actor
	 * @return
	 */
	Annotation* getAnnotation(vtkActor* actor);
	Annotation *getAnnotation(int index);
	void getAnnotations(
		const std::string & type, 
		std::vector<Annotation*>& annotations);
	int getAnnotationIndex(Annotation * anno);
	std::vector<Annotation *>& getAnnotations();

	inline int getLabelByIndex(int index) 
	{ 
		if (index >= m_capacity || index < 0)
		{
			return -1;
		}
		return m_labeledCloudIndex[index];
	}

	void updateBalloonByIndex(int index);
	void updateBalloonByAnno(Annotation * anno);

protected:
	/**
	 * @brief keep all annotation from current cloud
	 */
	std::vector<Annotation*> m_annotations;

	vtkRenderWindowInteractor* m_interactor;
	vtkSmartPointer<vtkBalloonWidget> m_balloonWidget;

	int* m_labeledCloudIndex = nullptr;
	size_t m_capacity = 0;

};



#endif //Annotaions_H

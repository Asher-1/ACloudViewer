//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

//Local
#include "ImageVis.h"
#include "PCLConv.h"
#include "Tools/ecvTools.h"
#include "Tools/PclTools.h"
#include "PclUtils/CustomContextItem.h"

// CV_CORE_LIB
#include <CVPlatform.h>
#include <CVTools.h>
#include <ecvGLMatrix.h>

// ECV_DB_LIB
#include <ecvBBox.h>

// VTK
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>
#include <vtkRendererCollection.h>
#include <vtkCamera.h>
#include <vtkAxes.h>
#include <vtkAxesActor.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCaptionActor2D.h>
#include <vtkProperty.h>
#include <vtkPropAssembly.h>
#include <vtkUnsignedCharArray.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkJPEGReader.h>
#include <vtkBMPReader.h>
#include <vtkPNMReader.h>
#include <vtkPNGReader.h>
#include <vtkTIFFReader.h>
#include <vtkLookupTable.h>
#include <vtkTextureUnitManager.h>
#include <vtkContext2D.h>
#include <vtkTextProperty.h>

#if VTK_MAJOR_VERSION >= 6
#include <vtkImageSlice.h>
#include <vtkImageSliceMapper.h>
#endif

// PCL
#include <pcl/common/transforms.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/visualization/vtk/pcl_context_item.h>
#include <pcl/visualization/vtk/vtkRenderWindowInteractorFix.h>

// Support for VTK 7.1 upwards
#ifdef vtkGenericDataArray_h
#define SetTupleValue SetTypedTuple
#define InsertNextTupleValue InsertNextTypedTuple
#define GetTupleValue GetTypedTuple
#endif

using namespace std;

namespace PclUtils
{
	ImageVis::ImageVis(const string &viewerName, bool initIterator)
		: pcl::visualization::ImageViewer(viewerName)
	{
	}

	vtkSmartPointer<vtkRenderWindow> ImageVis::getRenderWindow()
	{
		return this->win_;
	}

	void ImageVis::setupInteractor(vtkSmartPointer<vtkRenderWindowInteractor> interactor, vtkSmartPointer<vtkRenderWindow> win)
	{
		if (!win || !interactor)
		{
			return;
		}

		setRenderWindow(win);
		setRenderWindowInteractor(interactor);
		getRenderWindow()->Render();
	}

	void ImageVis::enable2Dviewer(bool state)
	{
#ifdef CV_LINUX
        CVLog::Warning("[ImageVis::enable2Dviewer] Do not support 2D viewer on Linux or Mac platform now!");
        return;
#endif
		if (state)
		{
			m_mainInteractor = getRenderWindowInteractor();
			setRenderWindowInteractor(vtkSmartPointer<vtkRenderWindowInteractor>::Take(vtkRenderWindowInteractorFixNew()));
			getRenderWindow()->SetInteractor(getRenderWindowInteractor());
			getRenderWindowInteractor()->SetRenderWindow(getRenderWindow());
			m_mouseConnection = registerMouseCallback(&ImageVis::mouseEventProcess, *this);
		}
		else
		{
			setupInteractor(m_mainInteractor, getRenderWindow());
			getRenderWindow()->SetInteractor(getRenderWindowInteractor());
			getRenderWindowInteractor()->SetRenderWindow(getRenderWindow());
			m_mouseConnection.disconnect();
		}
	}

	void ImageVis::mouseEventProcess(const pcl::visualization::MouseEvent& event, void * args)
	{
		if (event.getButton() == pcl::visualization::MouseEvent::RightButton
			&& event.getType() == pcl::visualization::MouseEvent::MouseMove) {
			std::string id = pickItem(event);
			if (id != "")
			{
				CVLog::Print(QString("Picked item id : %1").arg(id.c_str()));
			}
		}
	}

	std::string ImageVis::pickItem(const pcl::visualization::MouseEvent& event)
	{
		int x = event.getX();
		int y = event.getY();

		return pickItem(x, y);
	}

	std::string ImageVis::pickItem(int x, int y)
	{
		for (int i = 0; i < layer_map_.size(); ++i)
		{
			Layer* layer = &layer_map_[i];
			int index = 0;
			while (layer->actor->GetScene()->GetItem(index))
			{
				pcl::visualization::context_items::Rectangle * context =
					reinterpret_cast<pcl::visualization::context_items::Rectangle*>(layer->actor->GetScene()->GetItem(index));
				if (context && context->params.size() == 4)
				{
					bool containFlag = (x >= context->params[0] && x <= context->params[0] + context->params[2] &&
						y >= context->params[1] && y <= context->params[1] + context->params[3]);
					if (containFlag)
					{
						return layer->layer_name;
					}
				}
				index++;
			}
		}
		return std::string("");
	}


	void ImageVis::setRenderWindow(vtkSmartPointer<vtkRenderWindow> win)
	{
		this->win_ = win;
	}

	vtkSmartPointer<vtkRenderWindowInteractor> ImageVis::getRenderWindowInteractor()
	{
		return this->interactor_;
	}

	void ImageVis::setRenderWindowInteractor(vtkSmartPointer<vtkRenderWindowInteractor> interactor)
	{
		this->interactor_ = interactor;
		timer_id_ = this->interactor_->CreateRepeatingTimer(5000L);

		// Set the exit callbacks
		exit_main_loop_timer_callback_ = vtkSmartPointer<ExitMainLoopTimerCallback>::New();
		exit_main_loop_timer_callback_->window = this;
		exit_main_loop_timer_callback_->right_timer_id = -1;
		this->interactor_->AddObserver(vtkCommand::TimerEvent, exit_main_loop_timer_callback_);

		exit_callback_ = vtkSmartPointer<ExitCallback>::New();
		exit_callback_->window = this;
		this->interactor_->AddObserver(vtkCommand::ExitEvent, exit_callback_);

		// Reset camera (flip it vertically)
#if ((VTK_MAJOR_VERSION == 5) && (VTK_MINOR_VERSION > 10))
  //ren_->GetActiveCamera ()->SetViewUp (0.0, -1.0, 0.0);
		vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
		transform->Scale(1.0, -1.0, 1.0);
		ren_->GetActiveCamera()->SetUserTransform(transform);
		ren_->GetActiveCamera()->ParallelProjectionOn();
		ren_->ResetCamera();
		ren_->ResetCameraClippingRange();
#endif

		resetStoppedFlag();
	}

	vtkSmartPointer<vtkRenderer> ImageVis::getRender()
	{
		return this->ren_;
	}


	void ImageVis::setRender(vtkSmartPointer<vtkRenderer> render)
	{
		this->ren_ = render;
		this->ren_->AddViewProp(slice_);
	}

	bool ImageVis::contains(const std::string & id) const
	{
		LayerMap::const_iterator am_it = std::find_if(layer_map_.begin(), layer_map_.end(), LayerComparator(id));
		return am_it != layer_map_.end();
	}

	pcl::visualization::ImageViewer::Layer* ImageVis::getLayer(const std::string& id)
	{
		for (auto & l : layer_map_)
		{
			if (l.layer_name == id)
			{
				return &l;
			}
		}
		return nullptr;
	}

	void ImageVis::hideShowActors(bool visibility, const std::string & viewID)
	{
		double opacity = visibility ? 1.0 : 0.0;
		Layer* layer = getLayer(viewID);
		if (layer)
		{
			int index = 0;
			while (layer->actor->GetScene()->GetItem(index))
			{
				layer->actor->GetScene()->GetItem(index)->SetVisible(visibility);
				index++;
			}
			layer->actor->SetVisibility(opacity);
			layer->actor->Modified();
		}
	}


	void ImageVis::changeOpacity(const std::string & viewID, double opacity)
	{
		Layer* layer = getLayer(viewID);
		if (layer)
		{
			layer->actor->SetVisibility(1);
			int index = 0;
			while (layer->actor->GetScene()->GetItem(index))
			{
				pcl::visualization::PCLContextItem * context = reinterpret_cast<pcl::visualization::PCLContextItem*>(layer->actor->GetScene()->GetItem(index));
				if (context)
				{
					context->SetVisible(1);
					context->setOpacity(opacity);
				}
				index++;
			}
			layer->actor->Modified();
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////
	pcl::visualization::ImageViewer::LayerMap::iterator
		ImageVis::createLayer(
			const std::string &layer_id, int x, int y,
			int width, int height, double opacity, bool fill_box)
	{
		Layer l;
		l.layer_name = layer_id;
		// Create a new layer
		l.actor = vtkSmartPointer<vtkContextActor>::New();
		l.actor->PickableOff();
		l.actor->DragableOff();
		if (fill_box)
		{
			vtkSmartPointer<pcl::visualization::context_items::FilledRectangle> rect = vtkSmartPointer<pcl::visualization::context_items::FilledRectangle>::New();
			rect->setColors(0, 0, 0);
			rect->setOpacity(opacity);
			rect->set(x, y, static_cast<float> (width), static_cast<float> (height));
			l.actor->GetScene()->AddItem(rect);
		}
#if VTK_MAJOR_VERSION < 6
		image_viewer_->GetRenderer()->AddActor(l.actor);
#else
		ren_->AddActor(l.actor);
#endif
		// Add another element
		layer_map_.push_back(l);

		return (layer_map_.end() - 1);
	}

	void ImageVis::addRGBImage(
		const unsigned char* rgb_data,
		unsigned x, unsigned y, unsigned width, unsigned height,
		const std::string &layer_id, double opacity)
	{
		if (unsigned(getSize()[0]) != width ||
			unsigned(getSize()[1]) != height)
			setSize(width, height);

		// Check to see if this ID entry already exists (has it been already added to the visualizer?)
		pcl::visualization::ImageViewer::LayerMap::iterator am_it = std::find_if(layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
		if (am_it == layer_map_.end())
		{
			//PCL_DEBUG("[pcl::visualization::ImageViewer::addRGBImage] No layer with ID='%s' found. Creating new one...\n", layer_id.c_str());
			am_it = createLayer(layer_id, x, y, width, height, opacity, false);
		}

		void* data = const_cast<void*> (reinterpret_cast<const void*> (rgb_data));

		vtkSmartPointer<vtkImageData> image = vtkSmartPointer<vtkImageData>::New();
		image->SetExtent(x, x + width - 1, y, y + height - 1, 0, 0);
#if VTK_MAJOR_VERSION < 6
		image->SetScalarTypeToUnsignedChar();
		image->SetNumberOfScalarComponents(3);
		image->AllocateScalars();
#else
		image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
#endif
		image->GetPointData()->GetScalars()->SetVoidArray(data, 3 * width * height, 1);
#if ((VTK_MAJOR_VERSION == 5) && (VTK_MINOR_VERSION <= 10))
		// Now create filter and set previously created transformation
		algo_->SetInput(image);
		algo_->Update();
#  if (VTK_MINOR_VERSION <= 6)
		image_viewer_->SetInput(algo_->GetOutput());
#  else
		image_viewer_->SetInputConnection(algo_->GetOutputPort());
#  endif
#elif VTK_MAJOR_VERSION < 6
		image_viewer_->SetInputData(image);
		interactor_style_->adjustCamera(image, ren_);
#else
		algo_->SetInputData(image);
		algo_->Update();
		slice_->GetMapper()->SetInputConnection(algo_->GetOutputPort());
		ren_->ResetCamera();
		ren_->GetActiveCamera()->SetParallelScale(0.5 * win_->GetSize()[1]);
#endif
	}

	bool ImageVis::addText(unsigned int x, unsigned int y,
		const std::string& text_string,
		double r, double g, double b,
        const std::string& layer_id,
        double opacity, int fontSize, bool bold)
	{
		//bool sucess = pcl::visualization::ImageViewer::addText(x, y, text_string, r, g, b, layer_id, opacity);

		  // Check to see if this ID entry already exists (has it been already added to the visualizer?)
		LayerMap::iterator am_it = std::find_if(layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
		if (am_it == layer_map_.end())
		{
			PCL_DEBUG("[pcl::visualization::ImageViewer::addText] No layer with ID='%s' found. Creating new one...\n", layer_id.c_str());
			am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1, opacity, false);
#if ((VTK_MAJOR_VERSION == 5) && (VTKOR_VERSION > 10))
			interactor_style_->adjustCamera(ren_);
#endif
		}

		vtkSmartPointer<PclUtils::context_items::Text> text = vtkSmartPointer<PclUtils::context_items::Text>::New();
		text->setColors(static_cast<unsigned char> (255.0 * r),
			static_cast<unsigned char> (255.0 * g),
			static_cast<unsigned char> (255.0 * b));
		text->setOpacity(opacity);
        text->setBold(bold);
        text->setFontSize(fontSize);
#if ((VTK_MAJOR_VERSION >= 6) || ((VTK_MAJOR_VERSION == 5) && (VTK_MINOR_VERSION > 7)))
		text->set(static_cast<float> (x), static_cast<float> (y), text_string);
#else
		text->set(static_cast<float> (x), static_cast<float> (getSize()[1] - y), text_string);
#endif
		am_it->actor->GetScene()->AddItem(text);

		return (true);
	}
}

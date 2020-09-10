/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

// LOCAL
#include "vtkCustomInteractorStyle.h"
#include "vtkCameraManipulator.h"

// CV_CORE_LIB
#include <CVLog.h>

#include <list>
#include <pcl/visualization/common/io.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkVersion.h>
#include <vtkLODActor.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkCellArray.h>
#include <vtkTextProperty.h>
#include <vtkAbstractPropPicker.h>
#include <vtkCamera.h>
#include <vtkCollectionIterator.h>
#include <vtkCollection.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkScalarBarActor.h>
#include <vtkPNGWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkRendererCollection.h>
#include <vtkActorCollection.h>
#include <vtkLegendScaleActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkObjectFactory.h>
#include <vtkProperty.h>
#include <vtkPointData.h>
#include <vtkAssemblyPath.h>
#include <vtkAbstractPicker.h>
#include <vtkPointPicker.h>
#include <vtkAreaPicker.h>

#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
#include <pcl/visualization/vtk/vtkVertexBufferObjectMapper.h>
#endif

#define ORIENT_MODE 0
#define SELECT_MODE 1

#define VTKISRBP_ORIENT 0
#define VTKISRBP_SELECT 1

namespace VTKExtensions
{
	vtkCustomInteractorStyle::vtkCustomInteractorStyle()
		: pcl::visualization::PCLVisualizerInteractorStyle()
		, CameraManipulators(vtkCollection::New())
		, CurrentManipulator(nullptr)
		, RotationFactor(1.0)
		, lut_actor_id_("")
	{
		this->CenterOfRotation[0] = 
			this->CenterOfRotation[1] = 
			this->CenterOfRotation[2] = 0;
	}

	//-------------------------------------------------------------------------
	vtkCustomInteractorStyle::~vtkCustomInteractorStyle()
	{
		this->CameraManipulators->Delete();
		this->CameraManipulators = nullptr;
	}

	//-------------------------------------------------------------------------
	void vtkCustomInteractorStyle::RemoveAllManipulators()
	{
		this->CameraManipulators->RemoveAllItems();
	}

	//-------------------------------------------------------------------------
	void vtkCustomInteractorStyle::AddManipulator(vtkCameraManipulator* m)
	{
		this->CameraManipulators->AddItem(m);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::zoomIn()
	{
		FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);
		// Zoom in
		StartDolly();
		double factor = 10.0 * 0.2 * .5;
		Dolly(pow(1.1, factor));
		EndDolly();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::zoomOut()
	{
		FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);
		// Zoom out
		StartDolly();
		double factor = 10.0 * -0.2 * .5;
		Dolly(pow(1.1, factor));
		EndDolly();
	}

	void vtkCustomInteractorStyle::toggleAreaPicking()
	{
		CurrentMode = (CurrentMode == ORIENT_MODE) ? SELECT_MODE : ORIENT_MODE;
		if (CurrentMode == SELECT_MODE)
		{
			// Save the point picker
			point_picker_ = static_cast<vtkPointPicker*> (Interactor->GetPicker());
			// Switch for an area picker
			vtkSmartPointer<vtkAreaPicker> area_picker = vtkSmartPointer<vtkAreaPicker>::New();
			Interactor->SetPicker(area_picker);
		}
		else
		{
			// Restore point picker
			Interactor->SetPicker(point_picker_);
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnChar()
	{
		// Make sure we ignore the same events we handle in OnKeyDown to avoid calling things twice
		FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);
		if (Interactor->GetKeyCode() >= '0' && Interactor->GetKeyCode() <= '9')
			return;
		std::string key(Interactor->GetKeySym());
		if (key.find("XF86ZoomIn") != std::string::npos)
			zoomIn();
		else if (key.find("XF86ZoomOut") != std::string::npos)
			zoomOut();

		bool keymod = false;
		switch (modifier_)
		{
		case pcl::visualization::INTERACTOR_KB_MOD_ALT:
		{
			keymod = Interactor->GetAltKey();
			break;
		}
		case pcl::visualization::InteractorKeyboardModifier::INTERACTOR_KB_MOD_CTRL:
		{
			keymod = Interactor->GetControlKey();
			break;
		}
		case pcl::visualization::InteractorKeyboardModifier::INTERACTOR_KB_MOD_SHIFT:
		{
			keymod = Interactor->GetShiftKey();
			break;
		}
		}

		switch (Interactor->GetKeyCode())
		{
			// All of the options below simply exit
		case 'a': case 'A':
		case 'h': case 'H':
		case 'l': case 'L':
		case 'p': case 'P':
		case 'j': case 'J':
		case 'c': case 'C':
		case 43:        // KEY_PLUS
		case 45:        // KEY_MINUS
		case 'f': case 'F':
		case 'g': case 'G':
		case 'o': case 'O':
		case 'u': case 'U':
		case 'q': case 'Q':
		case 'x': case 'X':
		case 'r': case 'R':
		{
			break;
		}
		// S have special !ALT case
		case 's': case 'S':
		{
			if (!keymod)
				vtkInteractorStyleRubberBandPick::OnChar();
			break;
		}
		default:
		{
			vtkInteractorStyleRubberBandPick::OnChar();
			break;
		}
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnKeyDown()
	{
		if (!rens_)
		{
			CVLog::Error("[vtkCustomInteractorStyle] No renderer collection given! Use SetRendererCollection () before continuing.");
			return;
		}

		// Look for a matching camera interactor.
		this->CameraManipulators->InitTraversal();
		vtkCameraManipulator* manipulator = NULL;
		while ((manipulator = (vtkCameraManipulator*)this->CameraManipulators->GetNextItemAsObject()))
		{
			manipulator->OnKeyDown(this->Interactor);
		}

		FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);

		if (wif_->GetInput() == NULL)
		{
			wif_->SetInput(Interactor->GetRenderWindow());
			wif_->Modified();
			snapshot_writer_->Modified();
		}

		// Save the initial windows width/height
		if (win_height_ == -1 || win_width_ == -1)
		{
			int *win_size = Interactor->GetRenderWindow()->GetSize();
			win_height_ = win_size[0];
			win_width_ = win_size[1];
		}

		// Get the status of special keys (Cltr+Alt+Shift)
		bool shift = Interactor->GetShiftKey();
		bool ctrl = Interactor->GetControlKey();
		bool alt = Interactor->GetAltKey();

		bool keymod = false;
		switch (modifier_)
		{
		case pcl::visualization::INTERACTOR_KB_MOD_ALT:
		{
			keymod = alt;
			break;
		}
		case pcl::visualization::INTERACTOR_KB_MOD_CTRL:
		{
			keymod = ctrl;
			break;
		}
		case pcl::visualization::INTERACTOR_KB_MOD_SHIFT:
		{
			keymod = shift;
			break;
		}
		}

		// ---[ Check the rest of the key codes

		// Save camera parameters
		if ((Interactor->GetKeySym()[0] == 'S' || Interactor->GetKeySym()[0] == 's') && ctrl && !alt && !shift)
		{
			if (camera_file_.empty())
			{
				getCameraParameters(camera_);
				camera_saved_ = true;
				CVLog::Print("Camera parameters saved, you can press CTRL + R to restore.");
			}
			else
			{
				if (saveCameraParameters(camera_file_))
				{
					CVLog::Print("Save camera parameters to %s, you can press CTRL + R to restore.", camera_file_.c_str());
				}
				else
				{
					CVLog::Error("[vtkCustomInteractorStyle] Can't save camera parameters to file: %s.", camera_file_.c_str());
				}
			}
		}

		// Restore camera parameters
		if ((Interactor->GetKeySym()[0] == 'R' || Interactor->GetKeySym()[0] == 'r') && ctrl && !alt && !shift)
		{
			if (camera_file_.empty())
			{
				if (camera_saved_)
				{
					setCameraParameters(camera_);
					CVLog::Print("Camera parameters restored.");
				}
				else
				{
					CVLog::Print("No camera parameters saved for restoring.");
				}
			}
			else
			{
				if (boost::filesystem::exists(camera_file_))
				{
					if (loadCameraParameters(camera_file_))
					{
						CVLog::Print("Restore camera parameters from %s.", camera_file_.c_str());
					}
					else
					{
						CVLog::Error("Can't restore camera parameters from file: %s.", camera_file_.c_str());
					}
				}
				else
				{
					CVLog::Print("No camera parameters saved in %s for restoring.", camera_file_.c_str());
				}
			}
		}

		// Switch between point color/geometry handlers
		if (Interactor->GetKeySym() && Interactor->GetKeySym()[0] >= '0' && Interactor->GetKeySym()[0] <= '9')
		{
			pcl::visualization::CloudActorMap::iterator it;
			int index = Interactor->GetKeySym()[0] - '0' - 1;
			if (index == -1) index = 9;

			// Add 10 more for CTRL+0..9 keys
			if (ctrl)
				index += 10;

			// Geometry ?
			if (keymod)
			{
				for (it = cloud_actors_->begin(); it != cloud_actors_->end(); ++it)
				{
					pcl::visualization::CloudActor *act = &(*it).second;
					if (index >= static_cast<int> (act->geometry_handlers.size()))
						continue;

					// Save the geometry handler index for later usage
					act->geometry_handler_index_ = index;

					// Create the new geometry
					pcl::visualization::PointCloudGeometryHandler<pcl::PCLPointCloud2>::ConstPtr geometry_handler = act->geometry_handlers[index];

					// Use the handler to obtain the geometry
					vtkSmartPointer<vtkPoints> points;
					geometry_handler->getGeometry(points);

					// Set the vertices
					vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();
					for (vtkIdType i = 0; i < static_cast<vtkIdType> (points->GetNumberOfPoints()); ++i)
						vertices->InsertNextCell(static_cast<vtkIdType>(1), &i);

					// Create the data
					vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
					data->SetPoints(points);
					data->SetVerts(vertices);
					// Modify the mapper
#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
					if (use_vbos_)
					{
						vtkVertexBufferObjectMapper* mapper = static_cast<vtkVertexBufferObjectMapper*>(act->actor->GetMapper());
						mapper->SetInput(data);
						// Modify the actor
						act->actor->SetMapper(mapper);
					}
					else
#endif
					{
						vtkPolyDataMapper* mapper = static_cast<vtkPolyDataMapper*>(act->actor->GetMapper());
#if VTK_MAJOR_VERSION < 6
						mapper->SetInput(data);
#else
						mapper->SetInputData(data);
#endif
						// Modify the actor
						act->actor->SetMapper(mapper);
					}
					act->actor->Modified();
				}
			}
			else
			{
				for (it = cloud_actors_->begin(); it != cloud_actors_->end(); ++it)
				{
					pcl::visualization::CloudActor *act = &(*it).second;
					// Check for out of bounds
					if (index >= static_cast<int> (act->color_handlers.size()))
						continue;

					// Save the color handler index for later usage
					act->color_handler_index_ = index;

					// Get the new color
					pcl::visualization::PointCloudColorHandler<pcl::PCLPointCloud2>::ConstPtr color_handler = act->color_handlers[index];

					vtkSmartPointer<vtkDataArray> scalars;
					color_handler->getColor(scalars);
					double minmax[2];
					scalars->GetRange(minmax);
					// Update the data
					vtkPolyData *data = static_cast<vtkPolyData*>(act->actor->GetMapper()->GetInput());
					data->GetPointData()->SetScalars(scalars);
					// Modify the mapper
#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
					if (use_vbos_)
					{
						vtkVertexBufferObjectMapper* mapper = static_cast<vtkVertexBufferObjectMapper*>(act->actor->GetMapper());
						mapper->SetScalarRange(minmax);
						mapper->SetScalarModeToUsePointData();
						mapper->SetInput(data);
						// Modify the actor
						act->actor->SetMapper(mapper);
					}
					else
#endif
					{
						vtkPolyDataMapper* mapper = static_cast<vtkPolyDataMapper*>(act->actor->GetMapper());
						mapper->SetScalarRange(minmax);
						mapper->SetScalarModeToUsePointData();
#if VTK_MAJOR_VERSION < 6
						mapper->SetInput(data);
#else
						mapper->SetInputData(data);
#endif
						// Modify the actor
						act->actor->SetMapper(mapper);
					}
					act->actor->Modified();
				}
			}

			Interactor->Render();
			return;
		}

		std::string key(Interactor->GetKeySym());
		if (key.find("XF86ZoomIn") != std::string::npos)
			zoomIn();
		else if (key.find("XF86ZoomOut") != std::string::npos)
			zoomOut();

		switch (Interactor->GetKeyCode())
		{
		case 'h': case 'H':
		{
			CVLog::Print("| Help:"
				"-------"
				"          p, P   : switch to a point-based representation"
				"          w, W   : switch to a wireframe-based representation (where available)"
				"          s, S   : switch to a surface-based representation (where available)"
				""
				"          j, J   : take a .PNG snapshot of the current window view"
				"          c, C   : display current camera/window parameters"
				"          f, F   : fly to point mode"
				""
				"          e, E   : exit the interactor"
				"          q, Q   : stop and call VTK's TerminateApp"
				""
				"           +/-   : increment/decrement overall point size"
				"     +/- [+ ALT] : zoom in/out "
				""
				"          g, G   : display scale grid (on/off)"
				"          u, U   : display lookup table (on/off)"
				""
				"    o, O         : switch between perspective/parallel projection (default = perspective)"
				"    r, R [+ ALT] : reset camera [to viewpoint = {0, 0, 0} -> center_{x, y, z}]"
				"    CTRL + s, S  : save camera parameters"
				"    CTRL + r, R  : restore camera parameters"
				""
				"    ALT + s, S   : turn stereo mode on/off"
				"    ALT + f, F   : switch between maximized window mode and original size"
				""
				"          l, L           : list all available geometric and color handlers for the current actor map"
				"    ALT + 0..9 [+ CTRL]  : switch between different geometric handlers (where available)"
				"          0..9 [+ CTRL]  : switch between different color handlers (where available)"
				""
				"    SHIFT + left click   : select a point (start with -use_point_picking)"
				""
				"          x, X   : toggle rubber band selection mode for left mouse button"
			);
			break;
		}

		// Get the list of available handlers
		case 'l': case 'L':
		{
			// Iterate over the entire actors list and extract the geomotry/color handlers list
			for (pcl::visualization::CloudActorMap::iterator it = cloud_actors_->begin(); it != cloud_actors_->end(); ++it)
			{
				std::list<std::string> geometry_handlers_list, color_handlers_list;
				pcl::visualization::CloudActor *act = &(*it).second;
				for (size_t i = 0; i < act->geometry_handlers.size(); ++i)
					geometry_handlers_list.push_back(act->geometry_handlers[i]->getFieldName());
				for (size_t i = 0; i < act->color_handlers.size(); ++i)
					color_handlers_list.push_back(act->color_handlers[i]->getFieldName());

				if (!geometry_handlers_list.empty())
				{
					int i = 0;
					CVLog::Print("List of available geometry handlers for actor "); 
					CVLog::Print("%s: ", (*it).first.c_str());
					for (std::list<std::string>::iterator git = geometry_handlers_list.begin(); git != geometry_handlers_list.end(); ++git)
						CVLog::Print("%s(%d) ", (*git).c_str(), ++i);
				}
				if (!color_handlers_list.empty())
				{
					int i = 0;
					CVLog::Print("List of available color handlers for actor "); 
					CVLog::Print("%s: ", (*it).first.c_str());
					for (std::list<std::string>::iterator cit = color_handlers_list.begin(); cit != color_handlers_list.end(); ++cit)
						CVLog::Print("%s(%d) ", (*cit).c_str(), ++i);
				}
			}

			break;
		}

		// Switch representation to points
		case 'p': case 'P':
		{
			vtkSmartPointer<vtkActorCollection> ac = CurrentRenderer->GetActors();
			vtkCollectionSimpleIterator ait;
			for (ac->InitTraversal(ait); vtkActor* actor = ac->GetNextActor(ait); )
			{
				for (actor->InitPathTraversal(); vtkAssemblyPath* path = actor->GetNextPath(); )
				{
					vtkSmartPointer<vtkActor> apart = reinterpret_cast <vtkActor*> (path->GetLastNode()->GetViewProp());
					apart->GetProperty()->SetRepresentationToPoints();
				}
			}
			break;
		}

		// Switch representation to wireframe (override default behavior)
		case 'w': case 'W':
		{
			vtkSmartPointer<vtkActorCollection> ac = CurrentRenderer->GetActors();
			vtkCollectionSimpleIterator ait;
			for (ac->InitTraversal(ait); vtkActor* actor = ac->GetNextActor(ait); )
			{
				for (actor->InitPathTraversal(); vtkAssemblyPath* path = actor->GetNextPath(); )
				{
					vtkSmartPointer<vtkActor> apart = reinterpret_cast <vtkActor*> (path->GetLastNode()->GetViewProp());
					apart->GetProperty()->SetRepresentationToWireframe();
					apart->GetProperty()->SetLighting(false);
				}
			}
			break;
		}

		// Save a PNG snapshot with the current screen
		case 'j': case 'J':
		{
			char cam_fn[80], snapshot_fn[80];
			unsigned t = static_cast<unsigned> (time(0));
			sprintf(snapshot_fn, "screenshot-%d.png", t);
			saveScreenshot(snapshot_fn);

			sprintf(cam_fn, "screenshot-%d.cam", t);
			saveCameraParameters(cam_fn);

			CVLog::Print("Screenshot (%s) and camera information (%s) successfully captured.", snapshot_fn, cam_fn);
			break;
		}
		// display current camera settings/parameters
		case 'c': case 'C':
		{
			vtkSmartPointer<vtkCamera> cam = Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera();
			double clip[2], focal[3], pos[3], view[3];
			cam->GetClippingRange(clip);
			cam->GetFocalPoint(focal);
			cam->GetPosition(pos);
			cam->GetViewUp(view);
			int *win_pos = Interactor->GetRenderWindow()->GetPosition();
			int *win_size = Interactor->GetRenderWindow()->GetSize();
			std::cerr << "Clipping plane [near,far] " << clip[0] << ", " << clip[1] << endl <<
				"Focal point [x,y,z] " << focal[0] << ", " << focal[1] << ", " << focal[2] << endl <<
				"Position [x,y,z] " << pos[0] << ", " << pos[1] << ", " << pos[2] << endl <<
				"View up [x,y,z] " << view[0] << ", " << view[1] << ", " << view[2] << endl <<
				"Camera view angle [degrees] " << cam->GetViewAngle() << endl <<
				"Window size [x,y] " << win_size[0] << ", " << win_size[1] << endl <<
				"Window position [x,y] " << win_pos[0] << ", " << win_pos[1] << endl;
			break;
		}
		case '=':
		{
			zoomIn();
			break;
		}
		case 43:        // KEY_PLUS
		{
			if (alt)
				zoomIn();
			else
			{
				vtkSmartPointer<vtkActorCollection> ac = CurrentRenderer->GetActors();
				vtkCollectionSimpleIterator ait;
				for (ac->InitTraversal(ait); vtkActor* actor = ac->GetNextActor(ait); )
				{
					for (actor->InitPathTraversal(); vtkAssemblyPath* path = actor->GetNextPath(); )
					{
						vtkSmartPointer<vtkActor> apart = reinterpret_cast <vtkActor*> (path->GetLastNode()->GetViewProp());
						float psize = apart->GetProperty()->GetPointSize();
						if (psize < 63.0f)
							apart->GetProperty()->SetPointSize(psize + 1.0f);
					}
				}
			}
			break;
		}
		case 45:        // KEY_MINUS
		{
			if (alt)
				zoomOut();
			else
			{
				vtkSmartPointer<vtkActorCollection> ac = CurrentRenderer->GetActors();
				vtkCollectionSimpleIterator ait;
				for (ac->InitTraversal(ait); vtkActor* actor = ac->GetNextActor(ait); )
				{
					for (actor->InitPathTraversal(); vtkAssemblyPath* path = actor->GetNextPath(); )
					{
						vtkSmartPointer<vtkActor> apart = static_cast<vtkActor*> (path->GetLastNode()->GetViewProp());
						float psize = apart->GetProperty()->GetPointSize();
						if (psize > 1.0f)
							apart->GetProperty()->SetPointSize(psize - 1.0f);
					}
				}
			}
			break;
		}
		// Switch between maximize and original window size
		case 'f': case 'F':
		{
			if (keymod)
			{
				// Get screen size
				int *temp = Interactor->GetRenderWindow()->GetScreenSize();
				int scr_size[2]; scr_size[0] = temp[0]; scr_size[1] = temp[1];

				// Get window size
				temp = Interactor->GetRenderWindow()->GetSize();
				int win_size[2]; win_size[0] = temp[0]; win_size[1] = temp[1];
				// Is window size = max?
				if (win_size[0] == max_win_height_ && win_size[1] == max_win_width_)
				{
					// Set the previously saved 'current' window size
					Interactor->GetRenderWindow()->SetSize(win_height_, win_width_);
					// Set the previously saved window position
					Interactor->GetRenderWindow()->SetPosition(win_pos_x_, win_pos_y_);
					Interactor->GetRenderWindow()->Render();
					Interactor->Render();
				}
				// Set to max
				else
				{
					int *win_pos = Interactor->GetRenderWindow()->GetPosition();
					// Save the current window position
					win_pos_x_ = win_pos[0];
					win_pos_y_ = win_pos[1];
					// Save the current window size
					win_height_ = win_size[0];
					win_width_ = win_size[1];
					// Set the maximum window size
					Interactor->GetRenderWindow()->SetSize(scr_size[0], scr_size[1]);
					Interactor->GetRenderWindow()->Render();
					Interactor->Render();
					int *win_size = Interactor->GetRenderWindow()->GetSize();
					// Save the maximum window size
					max_win_height_ = win_size[0];
					max_win_width_ = win_size[1];
				}
			}
			else
			{
				AnimState = VTKIS_ANIM_ON;
				vtkAssemblyPath *path = NULL;
				Interactor->GetPicker()->Pick(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1], 0.0, CurrentRenderer);
				vtkAbstractPropPicker *picker;
				if ((picker = vtkAbstractPropPicker::SafeDownCast(Interactor->GetPicker())))
					path = picker->GetPath();
				if (path != NULL)
					Interactor->FlyTo(CurrentRenderer, picker->GetPickPosition());
				AnimState = VTKIS_ANIM_OFF;
			}
			break;
		}
		// 's'/'S' w/out ALT
		case 's': case 'S':
		{
			if (keymod)
			{
				int stereo_render = Interactor->GetRenderWindow()->GetStereoRender();
				if (!stereo_render)
				{
					if (stereo_anaglyph_mask_default_)
					{
						Interactor->GetRenderWindow()->SetAnaglyphColorMask(4, 3);
						stereo_anaglyph_mask_default_ = false;
					}
					else
					{
						Interactor->GetRenderWindow()->SetAnaglyphColorMask(2, 5);
						stereo_anaglyph_mask_default_ = true;
					}
				}
				Interactor->GetRenderWindow()->SetStereoRender(!stereo_render);
				Interactor->GetRenderWindow()->Render();
				Interactor->Render();
			}
			else
			{
				vtkInteractorStyleRubberBandPick::OnKeyDown();
				vtkSmartPointer<vtkActorCollection> ac = CurrentRenderer->GetActors();
				vtkCollectionSimpleIterator ait;
				for (ac->InitTraversal(ait); vtkActor* actor = ac->GetNextActor(ait);)
				{
					for (actor->InitPathTraversal(); vtkAssemblyPath* path = actor->GetNextPath();)
					{
						vtkSmartPointer<vtkActor> apart = reinterpret_cast<vtkActor*>(path->GetLastNode()->GetViewProp());
						apart->GetProperty()->SetRepresentationToSurface();
						apart->GetProperty()->SetLighting(true);
					}
				}
			}
			break;
		}

		// Display a grid/scale over the screen
		case 'g': case 'G':
		{
			if (!grid_enabled_)
			{
				grid_actor_->TopAxisVisibilityOn();
				CurrentRenderer->AddViewProp(grid_actor_);
				grid_enabled_ = true;
			}
			else
			{
				CurrentRenderer->RemoveViewProp(grid_actor_);
				grid_enabled_ = false;
			}
			break;
		}

		case 'o': case 'O':
		{
			vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();
			int flag = cam->GetParallelProjection();
			cam->SetParallelProjection(!flag);

			CurrentRenderer->SetActiveCamera(cam);
			CurrentRenderer->Render();
			break;
		}
		// Display a LUT actor on screen
		case 'u': case 'U':
		{
			this->updateLookUpTableDisplay(true);
			break;
		}

		// Overwrite the camera reset
		case 'r': case 'R':
		{
			if (!keymod)
			{
				FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);
				if (CurrentRenderer != 0)
					CurrentRenderer->ResetCamera();
				else
					PCL_WARN("no current renderer on the interactor style.");

				CurrentRenderer->Render();
				break;
			}

			vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();

			static pcl::visualization::CloudActorMap::iterator it = cloud_actors_->begin();
			// it might be that some actors don't have a valid transformation set -> we skip them to avoid a seg fault.
			bool found_transformation = false;
			for (unsigned idx = 0; idx < cloud_actors_->size(); ++idx, ++it)
			{
				if (it == cloud_actors_->end())
					it = cloud_actors_->begin();

				const pcl::visualization::CloudActor& actor = it->second;
				if (actor.viewpoint_transformation_.GetPointer())
				{
					found_transformation = true;
					break;
				}
			}

			// if a valid transformation was found, use it otherwise fall back to default view point.
			if (found_transformation)
			{
				const pcl::visualization::CloudActor& actor = it->second;
				cam->SetPosition(actor.viewpoint_transformation_->GetElement(0, 3),
					actor.viewpoint_transformation_->GetElement(1, 3),
					actor.viewpoint_transformation_->GetElement(2, 3));

				cam->SetFocalPoint(actor.viewpoint_transformation_->GetElement(0, 3) - actor.viewpoint_transformation_->GetElement(0, 2),
					actor.viewpoint_transformation_->GetElement(1, 3) - actor.viewpoint_transformation_->GetElement(1, 2),
					actor.viewpoint_transformation_->GetElement(2, 3) - actor.viewpoint_transformation_->GetElement(2, 2));

				cam->SetViewUp(actor.viewpoint_transformation_->GetElement(0, 1),
					actor.viewpoint_transformation_->GetElement(1, 1),
					actor.viewpoint_transformation_->GetElement(2, 1));
			}
			else
			{
				cam->SetPosition(0, 0, 0);
				cam->SetFocalPoint(0, 0, 1);
				cam->SetViewUp(0, -1, 0);
			}

			// go to the next actor for the next key-press event.
			if (it != cloud_actors_->end())
				++it;
			else
				it = cloud_actors_->begin();

			CurrentRenderer->SetActiveCamera(cam);
			CurrentRenderer->ResetCameraClippingRange();
			CurrentRenderer->Render();
			break;
		}

		case 'a': case 'A':
		{
			CurrentMode = (CurrentMode == ORIENT_MODE) ? SELECT_MODE : ORIENT_MODE;
			if (CurrentMode == SELECT_MODE)
			{
				// Save the point picker
				point_picker_ = static_cast<vtkPointPicker*> (Interactor->GetPicker());
				// Switch for an area picker
				vtkSmartPointer<vtkAreaPicker> area_picker = vtkSmartPointer<vtkAreaPicker>::New();
				Interactor->SetPicker(area_picker);
			}
			else
			{
				// Restore point picker
				Interactor->SetPicker(point_picker_);
			}
			break;
		}

		case 'q': case 'Q':
		{
			Interactor->ExitCallback();
			return;
		}
		default:
		{
			vtkInteractorStyleRubberBandPick::OnKeyDown();
			break;
		}
		}

		pcl::visualization::KeyboardEvent event(true, Interactor->GetKeySym(), Interactor->GetKeyCode(), Interactor->GetAltKey(), Interactor->GetControlKey(), Interactor->GetShiftKey());
		keyboard_signal_(event);

		rens_->Render();
		Interactor->Render();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnKeyUp()
	{
		pcl::visualization::KeyboardEvent event(false, 
			Interactor->GetKeySym(), 
			Interactor->GetKeyCode(), 
			Interactor->GetAltKey(), 
			Interactor->GetControlKey(),
			Interactor->GetShiftKey());
		keyboard_signal_(event);

		// Look for a matching camera interactor.
		this->CameraManipulators->InitTraversal();
		vtkCameraManipulator* manipulator = NULL;
		while ((manipulator = (vtkCameraManipulator*)this->CameraManipulators->GetNextItemAsObject()))
		{
			manipulator->OnKeyUp(this->Interactor);
		}

		vtkInteractorStyleRubberBandPick::OnKeyUp();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnMouseMove()
	{
		int x = this->Interactor->GetEventPosition()[0];
		int y = this->Interactor->GetEventPosition()[1];
		pcl::visualization::MouseEvent event(
			pcl::visualization::MouseEvent::MouseMove,
			pcl::visualization::MouseEvent::NoButton, 
			x, y, Interactor->GetAltKey(), 
			Interactor->GetControlKey(), 
			Interactor->GetShiftKey(), 
			vtkInteractorStyleRubberBandPick::CurrentMode);
		mouse_signal_(event);

		if (this->CurrentMode != VTKISRBP_SELECT/* && !Interactor->GetControlKey() && !Interactor->GetShiftKey()*/)
		{
			if (this->CurrentRenderer && this->CurrentManipulator)
			{
				// When an interaction is active, we should not change the renderer being
				// interacted with.
			}
			else
			{
				this->FindPokedRenderer(
					this->Interactor->GetEventPosition()[0], this->Interactor->GetEventPosition()[1]);
			}

			if (this->CurrentManipulator)
			{
				this->CurrentManipulator->OnMouseMove(this->Interactor->GetEventPosition()[0],
					this->Interactor->GetEventPosition()[1], this->CurrentRenderer, this->Interactor);
				this->InvokeEvent(vtkCommand::InteractionEvent);
			}
		}
		else
		{
			vtkInteractorStyleRubberBandPick::OnMouseMove();
		}

	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnLeftButtonDown()
	{

		int x = this->Interactor->GetEventPosition()[0];
		int y = this->Interactor->GetEventPosition()[1];

		if (Interactor->GetRepeatCount() == 0)
		{
			pcl::visualization::MouseEvent event(
				pcl::visualization::MouseEvent::MouseButtonPress, 
				pcl::visualization::MouseEvent::LeftButton, x, y, 
				Interactor->GetAltKey(), Interactor->GetControlKey(),
				Interactor->GetShiftKey(), vtkInteractorStyleRubberBandPick::CurrentMode);
			mouse_signal_(event);
		}
		else
		{
			pcl::visualization::MouseEvent event(
				pcl::visualization::MouseEvent::MouseDblClick, 
				pcl::visualization::MouseEvent::LeftButton, x, y,
				Interactor->GetAltKey(), Interactor->GetControlKey(), 
				Interactor->GetShiftKey(), vtkInteractorStyleRubberBandPick::CurrentMode);
			mouse_signal_(event);
		}
		this->OnButtonDown(1, this->Interactor->GetShiftKey(), this->Interactor->GetControlKey());
		vtkInteractorStyleRubberBandPick::OnLeftButtonDown();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnLeftButtonUp()
	{
		int x = this->Interactor->GetEventPosition()[0];
		int y = this->Interactor->GetEventPosition()[1];
		pcl::visualization::MouseEvent event(
			pcl::visualization::MouseEvent::MouseButtonRelease, 
			pcl::visualization::MouseEvent::LeftButton, x, y,
			Interactor->GetAltKey(), Interactor->GetControlKey(), 
			Interactor->GetShiftKey(), vtkInteractorStyleRubberBandPick::CurrentMode);
		mouse_signal_(event);
		this->OnButtonUp(1);
		vtkInteractorStyleRubberBandPick::OnLeftButtonUp();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnMiddleButtonDown()
	{
		int x = this->Interactor->GetEventPosition()[0];
		int y = this->Interactor->GetEventPosition()[1];
		if (Interactor->GetRepeatCount() == 0)
		{
			pcl::visualization::MouseEvent event(
				pcl::visualization::MouseEvent::MouseButtonPress, 
				pcl::visualization::MouseEvent::MiddleButton, x, y, 
				Interactor->GetAltKey(), Interactor->GetControlKey(), 
				Interactor->GetShiftKey(), vtkInteractorStyleRubberBandPick::CurrentMode);
			mouse_signal_(event);
		}
		else
		{
			pcl::visualization::MouseEvent event(
				pcl::visualization::MouseEvent::MouseDblClick, 
				pcl::visualization::MouseEvent::MiddleButton, x, y, 
				Interactor->GetAltKey(), Interactor->GetControlKey(),
				Interactor->GetShiftKey(), vtkInteractorStyleRubberBandPick::CurrentMode);
			mouse_signal_(event);
		}
		this->OnButtonDown(2, this->Interactor->GetShiftKey(), this->Interactor->GetControlKey());
		vtkInteractorStyleRubberBandPick::OnMiddleButtonDown();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnMiddleButtonUp()
	{
		int x = this->Interactor->GetEventPosition()[0];
		int y = this->Interactor->GetEventPosition()[1];
		pcl::visualization::MouseEvent event(
			pcl::visualization::MouseEvent::MouseButtonRelease,
			pcl::visualization::MouseEvent::MiddleButton, x, y, 
			Interactor->GetAltKey(), Interactor->GetControlKey(), 
			Interactor->GetShiftKey(), vtkInteractorStyleRubberBandPick::CurrentMode);
		mouse_signal_(event);
		this->OnButtonUp(2);
		vtkInteractorStyleRubberBandPick::OnMiddleButtonUp();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnRightButtonDown()
	{
		int x = this->Interactor->GetEventPosition()[0];
		int y = this->Interactor->GetEventPosition()[1];
		if (Interactor->GetRepeatCount() == 0)
		{
			pcl::visualization::MouseEvent event(
				pcl::visualization::MouseEvent::MouseButtonPress, 
				pcl::visualization::MouseEvent::RightButton, x, y,
				Interactor->GetAltKey(), Interactor->GetControlKey(), 
				Interactor->GetShiftKey(), vtkInteractorStyleRubberBandPick::CurrentMode);
			mouse_signal_(event);
		}
		else
		{
			pcl::visualization::MouseEvent event(
				pcl::visualization::MouseEvent::MouseDblClick,
				pcl::visualization::MouseEvent::RightButton, x, y, 
				Interactor->GetAltKey(), Interactor->GetControlKey(),
				Interactor->GetShiftKey(), vtkInteractorStyleRubberBandPick::CurrentMode);
			mouse_signal_(event);
		}

		this->OnButtonDown(3, this->Interactor->GetShiftKey(), this->Interactor->GetControlKey());
		vtkInteractorStyleRubberBandPick::OnRightButtonDown();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnRightButtonUp()
	{
		int x = this->Interactor->GetEventPosition()[0];
		int y = this->Interactor->GetEventPosition()[1];
		pcl::visualization::MouseEvent event(
			pcl::visualization::MouseEvent::MouseButtonRelease, 
			pcl::visualization::MouseEvent::RightButton, x, y, 
			Interactor->GetAltKey(), Interactor->GetControlKey(), 
			Interactor->GetShiftKey(), vtkInteractorStyleRubberBandPick::CurrentMode);
		mouse_signal_(event);
		this->OnButtonUp(3);
		vtkInteractorStyleRubberBandPick::OnRightButtonUp();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnMouseWheelForward()
	{
		int x = this->Interactor->GetEventPosition()[0];
		int y = this->Interactor->GetEventPosition()[1];
		pcl::visualization::MouseEvent event(
			pcl::visualization::MouseEvent::MouseScrollUp, 
			pcl::visualization::MouseEvent::VScroll, x, y, 
			Interactor->GetAltKey(), Interactor->GetControlKey(), 
			Interactor->GetShiftKey(), vtkInteractorStyleRubberBandPick::CurrentMode);
		mouse_signal_(event);
		if (Interactor->GetRepeatCount())
			mouse_signal_(event);

		if (Interactor->GetAltKey())
		{
			// zoom
			vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();
			double opening_angle = cam->GetViewAngle();
			if (opening_angle > 15.0)
				opening_angle -= 1.0;

			cam->SetViewAngle(opening_angle);
			cam->Modified();
			CurrentRenderer->SetActiveCamera(cam);
			CurrentRenderer->ResetCameraClippingRange();
			CurrentRenderer->Modified();
			CurrentRenderer->Render();
			rens_->Render();
			Interactor->Render();
		}
		else
			vtkInteractorStyleRubberBandPick::OnMouseWheelForward();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	void vtkCustomInteractorStyle::OnMouseWheelBackward()
	{
		int x = this->Interactor->GetEventPosition()[0];
		int y = this->Interactor->GetEventPosition()[1];
		pcl::visualization::MouseEvent event(
			pcl::visualization::MouseEvent::MouseScrollDown, 
			pcl::visualization::MouseEvent::VScroll, x, y, 
			Interactor->GetAltKey(), Interactor->GetControlKey(), 
			Interactor->GetShiftKey(), vtkInteractorStyleRubberBandPick::CurrentMode);
		mouse_signal_(event);
		if (Interactor->GetRepeatCount())
			mouse_signal_(event);

		if (Interactor->GetAltKey())
		{
			// zoom
			vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();
			double opening_angle = cam->GetViewAngle();
			if (opening_angle < 170.0)
				opening_angle += 1.0;

			cam->SetViewAngle(opening_angle);
			cam->Modified();
			CurrentRenderer->SetActiveCamera(cam);
			CurrentRenderer->ResetCameraClippingRange();
			CurrentRenderer->Modified();
			CurrentRenderer->Render();
			rens_->Render();
			Interactor->Render();
		}
		else
			vtkInteractorStyleRubberBandPick::OnMouseWheelBackward();
	}

	//-------------------------------------------------------------------------
	void vtkCustomInteractorStyle::ResetLights()
	{
		if (!this->CurrentRenderer)
		{
			return;
		}

		vtkLight* light;

		vtkLightCollection* lights = this->CurrentRenderer->GetLights();
		vtkCamera* camera = this->CurrentRenderer->GetActiveCamera();

		lights->InitTraversal();
		light = lights->GetNextItem();
		if (!light)
		{
			return;
		}
		light->SetPosition(camera->GetPosition());
		light->SetFocalPoint(camera->GetFocalPoint());
	}

	//-------------------------------------------------------------------------
	void vtkCustomInteractorStyle::OnButtonDown(int button, int shift, int control)
	{
		// Must not be processing an interaction to start another.
		if (this->CurrentManipulator)
		{
			return;
		}

		// Get the renderer.
		this->FindPokedRenderer(
			this->Interactor->GetEventPosition()[0], this->Interactor->GetEventPosition()[1]);
		if (this->CurrentRenderer == NULL)
		{
			return;
		}

		// Look for a matching camera interactor.
		this->CurrentManipulator = this->FindManipulator(button, shift, control);
		if (this->CurrentManipulator)
		{
			this->CurrentManipulator->Register(this);
			this->InvokeEvent(vtkCommand::StartInteractionEvent);
			this->CurrentManipulator->SetCenter(this->CenterOfRotation);
			this->CurrentManipulator->SetRotationFactor(this->RotationFactor);
			this->CurrentManipulator->StartInteraction();
			this->CurrentManipulator->OnButtonDown(this->Interactor->GetEventPosition()[0],
				this->Interactor->GetEventPosition()[1], this->CurrentRenderer, this->Interactor);
		}
	}

	//-------------------------------------------------------------------------
	void vtkCustomInteractorStyle::OnButtonUp(int button)
	{
		if (this->CurrentManipulator == NULL)
		{
			return;
		}
		if (this->CurrentManipulator->GetButton() == button)
		{
			this->CurrentManipulator->OnButtonUp(this->Interactor->GetEventPosition()[0],
				this->Interactor->GetEventPosition()[1], this->CurrentRenderer, this->Interactor);
			this->CurrentManipulator->EndInteraction();
			this->InvokeEvent(vtkCommand::EndInteractionEvent);
			this->CurrentManipulator->UnRegister(this);
			this->CurrentManipulator = NULL;
		}
	}

	//-------------------------------------------------------------------------
	vtkCameraManipulator* vtkCustomInteractorStyle::FindManipulator(int button, int shift, int control)
	{
		// Look for a matching camera interactor.
		this->CameraManipulators->InitTraversal();
		vtkCameraManipulator* manipulator = NULL;
		while ((manipulator = (vtkCameraManipulator*)this->CameraManipulators->GetNextItemAsObject()))
		{
			if (manipulator->GetButton() == button && manipulator->GetShift() == shift &&
				manipulator->GetControl() == control)
			{
				return manipulator;
			}
		}
		return NULL;
	}

	void vtkCustomInteractorStyle::Dolly(double fact)
	{
		if (this->Interactor->GetControlKey())
		{
			vtkCustomInteractorStyle::DollyToPosition(
				fact, this->Interactor->GetEventPosition(), this->CurrentRenderer);
		}
		else
		{
			this->vtkInteractorStyleRubberBandPick::Dolly(fact);
		}
	}

	//-------------------------------------------------------------------------
	void vtkCustomInteractorStyle::DollyToPosition(double fact, int* position, vtkRenderer* renderer)
	{
		vtkCamera* cam = renderer->GetActiveCamera();
		if (cam->GetParallelProjection())
		{
			int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
			// Zoom relatively to the cursor
			int* aSize = renderer->GetRenderWindow()->GetSize();
			int w = aSize[0];
			int h = aSize[1];
			x0 = w / 2;
			y0 = h / 2;
			x1 = position[0];
			y1 = position[1];
			vtkCustomInteractorStyle::TranslateCamera(renderer, x0, y0, x1, y1);
			cam->SetParallelScale(cam->GetParallelScale() / fact);
			vtkCustomInteractorStyle::TranslateCamera(renderer, x1, y1, x0, y0);
		}
		else
		{
			// Zoom relatively to the cursor position
			double viewFocus[4], originalViewFocus[3], cameraPos[3], newCameraPos[3];
			double newFocalPoint[4], norm[3];

			// Move focal point to cursor position
			cam->GetPosition(cameraPos);
			cam->GetFocalPoint(viewFocus);
			cam->GetFocalPoint(originalViewFocus);
			cam->GetViewPlaneNormal(norm);

			vtkCustomInteractorStyle::ComputeWorldToDisplay(
				renderer, viewFocus[0], viewFocus[1], viewFocus[2], viewFocus);

			vtkCustomInteractorStyle::ComputeDisplayToWorld(
				renderer, double(position[0]), double(position[1]), viewFocus[2], newFocalPoint);

			cam->SetFocalPoint(newFocalPoint);

			// Move camera in/out along projection direction
			cam->Dolly(fact);

			// Find new focal point
			cam->GetPosition(newCameraPos);

			double newPoint[3];
			newPoint[0] = originalViewFocus[0] + newCameraPos[0] - cameraPos[0];
			newPoint[1] = originalViewFocus[1] + newCameraPos[1] - cameraPos[1];
			newPoint[2] = originalViewFocus[2] + newCameraPos[2] - cameraPos[2];

			cam->SetFocalPoint(newPoint);
		}
	}

	//-------------------------------------------------------------------------
	void vtkCustomInteractorStyle::TranslateCamera(
		vtkRenderer* renderer, int toX, int toY, int fromX, int fromY)
	{
		vtkCamera* cam = renderer->GetActiveCamera();
		double viewFocus[4], focalDepth, viewPoint[3];
		double newPickPoint[4], oldPickPoint[4], motionVector[3];
		cam->GetFocalPoint(viewFocus);

		vtkCustomInteractorStyle::ComputeWorldToDisplay(
			renderer, viewFocus[0], viewFocus[1], viewFocus[2], viewFocus);
		focalDepth = viewFocus[2];

		vtkCustomInteractorStyle::ComputeDisplayToWorld(
			renderer, double(toX), double(toY), focalDepth, newPickPoint);
		vtkCustomInteractorStyle::ComputeDisplayToWorld(
			renderer, double(fromX), double(fromY), focalDepth, oldPickPoint);

		// camera motion is reversed
		motionVector[0] = oldPickPoint[0] - newPickPoint[0];
		motionVector[1] = oldPickPoint[1] - newPickPoint[1];
		motionVector[2] = oldPickPoint[2] - newPickPoint[2];

		cam->GetFocalPoint(viewFocus);
		cam->GetPosition(viewPoint);
		cam->SetFocalPoint(
			motionVector[0] + viewFocus[0], motionVector[1] + viewFocus[1], motionVector[2] + viewFocus[2]);

		cam->SetPosition(
			motionVector[0] + viewPoint[0], motionVector[1] + viewPoint[1], motionVector[2] + viewPoint[2]);
	}


	//////////////////////////////////////////////////////////////////////////////////////////////
	// Update the look up table displayed when 'u' is pressed
	void vtkCustomInteractorStyle::updateLookUpTableDisplay(bool add_lut)
	{
		pcl::visualization::CloudActorMap::iterator am_it;
		pcl::visualization::ShapeActorMap::iterator sm_it;
		bool actor_found = false;

		if (!lut_enabled_ && !add_lut)
			return;

		if (lut_actor_id_ != "")  // Search if provided actor id is in CloudActorMap or ShapeActorMap
		{
			am_it = cloud_actors_->find(lut_actor_id_);
			if (am_it == cloud_actors_->end())
			{
				sm_it = shape_actors_->find(lut_actor_id_);
				if (sm_it == shape_actors_->end())
				{
					PCL_WARN("[updateLookUpTableDisplay] Could not find any actor with id <%s>!", lut_actor_id_.c_str());
					if (lut_enabled_)
					{  // Remove LUT and exit
						CurrentRenderer->RemoveActor(lut_actor_);
						lut_enabled_ = false;
					}
					return;
				}

				// ShapeActor found
				vtkSmartPointer<vtkProp> *act = &(*sm_it).second;
				vtkSmartPointer<vtkActor> actor = vtkActor::SafeDownCast(*act);
				if (!actor || !actor->GetMapper()->GetInput()->GetPointData()->GetScalars())
				{
					PCL_WARN("[updateLookUpTableDisplay] id <%s> does not hold any color information!", lut_actor_id_.c_str());
					if (lut_enabled_)
					{  // Remove LUT and exit
						CurrentRenderer->RemoveActor(lut_actor_);
						lut_enabled_ = false;
					}
					return;
				}

				lut_actor_->SetLookupTable(actor->GetMapper()->GetLookupTable());
				lut_actor_->Modified();
				actor_found = true;
			}
			else
			{
				// CloudActor
				pcl::visualization::CloudActor *act = &(*am_it).second;
				if (!act->actor->GetMapper()->GetLookupTable() && !act->actor->GetMapper()->GetInput()->GetPointData()->GetScalars())
				{
					PCL_WARN("[updateLookUpTableDisplay] id <%s> does not hold any color information!", lut_actor_id_.c_str());
					if (lut_enabled_)
					{  // Remove LUT and exit
						CurrentRenderer->RemoveActor(lut_actor_);
						lut_enabled_ = false;
					}
					return;
				}

				vtkScalarsToColors* lut = act->actor->GetMapper()->GetLookupTable();
				lut_actor_->SetLookupTable(lut);
				lut_actor_->Modified();
				actor_found = true;
			}
		}
		else  // lut_actor_id_ == "", the user did not specify which cloud/shape LUT should be displayed
		// Circling through all clouds/shapes and displaying first LUT found
		{
			for (am_it = cloud_actors_->begin(); am_it != cloud_actors_->end(); ++am_it)
			{
				pcl::visualization::CloudActor *act = &(*am_it).second;
				if (!act->actor->GetMapper()->GetLookupTable())
					continue;

				if (!act->actor->GetMapper()->GetInput()->GetPointData()->GetScalars())
					continue;

				vtkScalarsToColors* lut = act->actor->GetMapper()->GetLookupTable();
				lut_actor_->SetLookupTable(lut);
				lut_actor_->Modified();
				actor_found = true;
				break;
			}

			if (!actor_found)
			{
				for (sm_it = shape_actors_->begin(); sm_it != shape_actors_->end(); ++sm_it)
				{
					vtkSmartPointer<vtkProp> *act = &(*sm_it).second;
					vtkSmartPointer<vtkActor> actor = vtkActor::SafeDownCast(*act);
					if (!actor)
						continue;

					if (!actor->GetMapper()->GetInput()->GetPointData()->GetScalars())  // Check if actor has scalars
						continue;
					lut_actor_->SetLookupTable(actor->GetMapper()->GetLookupTable());
					lut_actor_->Modified();
					actor_found = true;
					break;
				}
			}
		}

		if ((!actor_found && lut_enabled_) || (lut_enabled_ && add_lut))  // Remove actor
		{
			CurrentRenderer->RemoveActor(lut_actor_);
			lut_enabled_ = false;
		}
		else if (!lut_enabled_ && add_lut && actor_found)  // Add actor
		{
			CurrentRenderer->AddActor(lut_actor_);
			lut_actor_->SetVisibility(true);
			lut_enabled_ = true;
		}
		else if (lut_enabled_)  // Update actor (if displayed)
		{
			CurrentRenderer->RemoveActor(lut_actor_);
			CurrentRenderer->AddActor(lut_actor_);
		}
		else
			return;

		CurrentRenderer->Render();
		return;
	}

	//-------------------------------------------------------------------------
	void vtkCustomInteractorStyle::PrintSelf(ostream& os, vtkIndent indent)
	{
		this->vtkInteractorStyleRubberBandPick::PrintSelf(os, indent);
		os << indent << "CenterOfRotation: " << this->CenterOfRotation[0] << ", "
			<< this->CenterOfRotation[1] << ", " << this->CenterOfRotation[2] << endl;
		os << indent << "RotationFactor: " << this->RotationFactor << endl;
		os << indent << "CameraManipulators: " << this->CameraManipulators << endl;
	}

}

namespace VTKExtensions
{
	// Standard VTK macro for *New ()
	vtkStandardNewMacro (vtkCustomInteractorStyle);
}


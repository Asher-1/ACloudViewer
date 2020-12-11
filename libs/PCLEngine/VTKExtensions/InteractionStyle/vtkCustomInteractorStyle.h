/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
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
 * $Id$
 *
 */
#ifndef ECV_VTK_CUSTOM_INTERACTOR_STYLE_H
#define ECV_VTK_CUSTOM_INTERACTOR_STYLE_H

#include "qPCL.h"

#include <vtkInteractorStyleRubberBandPick.h>
#include <pcl/visualization/interactor_style.h>

class vtkRendererCollection;
class vtkPointPicker;

class vtkCameraManipulator;
class vtkCollection;

namespace VTKExtensions
{
	/** \brief vtkCustomInteractorStyle defines an unique, custom VTK
	  * based interactory style for PCL Visualizer applications. Besides
	  * defining the rendering style, we also create a list of custom actions
	  * that are triggered on different keys being pressed:
	  *
	  * -        p, P   : switch to a point-based representation
	  * -        w, W   : switch to a wireframe-based representation (where available)
	  * -        s, S   : switch to a surface-based representation (where available)
	  * -        j, J   : take a .PNG snapshot of the current window view
	  * -        c, C   : display current camera/window parameters
	  * -        f, F   : fly to point mode
	  * -        e, E   : exit the interactor\
	  * -        q, Q   : stop and call VTK's TerminateApp
	  * -       + / -   : increment/decrement overall point size
	  * -        g, G   : display scale grid (on/off)
	  * -        u, U   : display lookup table (on/off)
	  * -  r, R [+ ALT] : reset camera [to viewpoint = {0, 0, 0} -> center_{x, y, z}]
	  * -  CTRL + s, S  : save camera parameters
	  * -  CTRL + r, R  : restore camera parameters
	  * -  ALT + s, S   : turn stereo mode on/off
	  * -  ALT + f, F   : switch between maximized window mode and original size
	  * -        l, L           : list all available geometric and color handlers for the current actor map
	  * -  ALT + 0..9 [+ CTRL]  : switch between different geometric handlers (where available)
	  * -        0..9 [+ CTRL]  : switch between different color handlers (where available)
	  * -
	  * -  SHIFT + left click   : select a point
	  * -        x, X   : toggle rubber band selection mode for left mouse button
	  *
	  * \author Radu B. Rusu
	  * \ingroup visualization
	  */
	class QPCL_ENGINE_LIB_API vtkCustomInteractorStyle : public pcl::visualization::PCLVisualizerInteractorStyle
	{
	public:
		static vtkCustomInteractorStyle *New();
		// this macro defines Superclass, the isA functionality and the safe downcast method
		vtkTypeMacro(vtkCustomInteractorStyle, pcl::visualization::PCLVisualizerInteractorStyle);
		void PrintSelf(ostream& os, vtkIndent indent) override;

		/** \brief Empty constructor. */
		vtkCustomInteractorStyle();

		/** \brief Empty destructor */
		virtual ~vtkCustomInteractorStyle() override;

		void toggleAreaPicking();

        /** \brief Set render window. */
        inline void setRenderWindow(const vtkSmartPointer<vtkRenderWindow>& win) {
            win_ = win;
        }

	public:
		/**
		 * Access to adding or removing manipulators.
		 */
		void AddManipulator(vtkCameraManipulator* m);

		/**
		 * Removes all manipulators.
		 */
		void RemoveAllManipulators();

		//@{
		/**
		 * Accessor for the collection of camera manipulators.
		 */
		vtkGetObjectMacro(CameraManipulators, vtkCollection);
		//@}

		//@{
		/**
		 * Propagates the center to the manipulators.
		 * This simply sets an internal ivar.
		 * It is propagated to a manipulator before the event
		 * is sent to it.
		 * Also changing the CenterOfRotation during interaction
		 * i.e. after a button press but before a button up
		 * has no effect until the next button press.
		 */
		vtkSetVector3Macro(CenterOfRotation, double);
		vtkGetVector3Macro(CenterOfRotation, double);
		//@}

		//@{
		/**
		 * Propagates the rotation factor to the manipulators.
		 * This simply sets an internal ivar.
		 * It is propagated to a manipulator before the event
		 * is sent to it.
		 * Also changing the RotationFactor during interaction
		 * i.e. after a button press but before a button up
		 * has no effect until the next button press.
		 */
		vtkSetMacro(RotationFactor, double);
		vtkGetMacro(RotationFactor, double);
		//@}

		/**
		 * Returns the chosen manipulator based on the modifiers.
		 */
		virtual vtkCameraManipulator* FindManipulator(int button, int shift, int control);

		/**
		 * Dolly the renderer's camera to a specific point
		 */
		static void DollyToPosition(double fact, int* position, vtkRenderer* renderer);

		/**
		 * Translate the renderer's camera
		 */
		static void TranslateCamera(vtkRenderer* renderer, int toX, int toY, int fromX, int fromY);

		using vtkInteractorStyleTrackballCamera::Dolly;

	protected:
        /** \brief Interactor style internal method. Zoom in. */
        void zoomIn();

        /** \brief Interactor style internal method. Zoom out. */
        void zoomOut();

        // Keyboard events
        virtual void OnKeyDown() override;
        virtual void OnKeyUp() override;

        /** \brief Interactor style internal method. Gets called
            * whenever a key is pressed. */
        virtual void OnChar() override;

        // mouse button events
        virtual void OnMouseMove() override;
        virtual void OnLeftButtonDown() override;
        virtual void OnLeftButtonUp() override;
        virtual void OnMiddleButtonDown() override;
        virtual void OnMiddleButtonUp() override;
        virtual void OnRightButtonDown() override;
        virtual void OnRightButtonUp() override;
        virtual void OnMouseWheelForward() override;
        virtual void OnMouseWheelBackward() override;

        friend class PointPickingCallback;
        friend class PCLVisualizer;

		void Dolly(double factor) override;

		/** \brief ID used to fetch/display the look up table on the visualizer
		 * It should be set by PCLVisualizer \ref setLookUpTableID
		 * @note If empty, a random actor added to the interactor will be used */
		std::string lut_actor_id_;

		/** \brief Add/remove the look up table displayed when 'u' is pressed, can also be used to update the current LUT displayed
		 * \ref lut_actor_id_ is used (if not empty) to chose which cloud/shape actor LUT will be updated (depending on what is available)
		 * If \ref lut_actor_id_ is empty the first actor with LUT support found will be used. */
		void updateLookUpTableDisplay(bool add_lut = false);

		vtkCameraManipulator* CurrentManipulator;
		double CenterOfRotation[3];
		double RotationFactor;

		// The CameraInteractors also store there button and modifier.
		vtkCollection* CameraManipulators;

		void OnButtonDown(int button, int shift, int control);
		void OnButtonUp(int button);
		void ResetLights();

		vtkCustomInteractorStyle(const vtkCustomInteractorStyle&) = delete;
		void operator=(const vtkCustomInteractorStyle&) = delete;
	};

}

#endif // ECV_VTK_CUSTOM_INTERACTOR_STYLE_H

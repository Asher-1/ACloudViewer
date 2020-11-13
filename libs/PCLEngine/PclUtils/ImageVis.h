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

#ifndef ECV_IMAGE_VIS_HEADER
#define ECV_IMAGE_VIS_HEADER

// LOCAL
#include "qPCL.h"
#include "PCLCloud.h"

// ECV_DB_LIB
#include <ecvGenericVisualizer2D.h>
#include <ecvColorTypes.h>
#include <ecvDrawContext.h>

// PCL
#include <visualization/include/pcl/visualization/image_viewer.h>
//#include <pcl/visualization/image_viewer.h>

// VTK
#include <vtkSmartPointer.h>

class vtkRenderer;
class vtkRenderWindow;
class vtkRenderWindowInteractor;

namespace PclUtils
{
	class QPCL_ENGINE_LIB_API ImageVis : public ecvGenericVisualizer2D, public pcl::visualization::ImageViewer
	{
		//Q_OBJECT
	public:
		//! Default constructor
		/** Constructor is protected to avoid using this object as a non static class.
		**/
		ImageVis(const std::string &viewerName, bool autoInit = false);
		/** \brief Set up our unique PCL interactor style for a given vtkRenderWindowInteractor object
		* attached to a given vtkRenderWindow
		* \param[in,out] interactor the vtkRenderWindowInteractor object to set up
		* \param[in,out] win a vtkRenderWindow object that the interactor is attached to
		*/
		void setupInteractor(vtkSmartPointer<vtkRenderWindowInteractor> interactor, vtkSmartPointer<vtkRenderWindow> win);
		void setRenderWindow(vtkSmartPointer<vtkRenderWindow> win);
		vtkSmartPointer<vtkRenderWindow> getRenderWindow();
		vtkSmartPointer<vtkRenderWindowInteractor> getRenderWindowInteractor();
		void setRenderWindowInteractor(vtkSmartPointer<vtkRenderWindowInteractor> interactor);
		/** \brief The renderer. */
		vtkSmartPointer<vtkRenderer> getRender();
		void setRender(vtkSmartPointer<vtkRenderer> render);

		/** \brief Check if the image with the given id was already added to this visualizer.
		* \param[in] id the id of the image to check
		* \return true if a image with the specified id was found
		*/
		bool contains(const std::string &id) const;

		Layer* getLayer(const std::string& id);
		void changeOpacity(const std::string & viewID, double opacity);
		void hideShowActors(bool visibility, const std::string & viewID);

		/** \brief Add a new 2D rendering layer to the viewer.
		* \param[in] layer_id the name of the layer
		* \param[in] width the width of the layer
		* \param[in] height the height of the layer
		* \param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 0.5)
		* \param[in] fill_box set to true to fill in the image with one black box before starting
		*/
		LayerMap::iterator createLayer(const std::string &layer_id, int x, int y,
			int width, int height, double opacity = 0.5, bool fill_box = true);


		void addRGBImage(
			const unsigned char* rgb_data, unsigned x, unsigned y, unsigned width, unsigned height,
			const std::string &layer_id, double opacity);

        bool addText(unsigned int x, unsigned int y,
			const std::string& text_string,
			double r, double g, double b,
			const std::string &layer_id = "line", 
			double opacity = 1.0, int fontSize = 10);

	public:
		void enable2Dviewer(bool state);

		std::string pickItem(int x, int y);

	private:
		void mouseEventProcess(const pcl::visualization::MouseEvent& event, void * args);
		boost::signals2::connection m_mouseConnection;
		std::string pickItem(const pcl::visualization::MouseEvent& event);

		vtkSmartPointer <vtkRenderWindowInteractor> m_mainInteractor;

	};

	typedef boost::shared_ptr<ImageVis> ImageVisPtr;
}

#endif // ECV_IMAGE_VIS_HEADER

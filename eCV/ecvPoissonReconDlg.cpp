//##########################################################################
//#                                                                        #
//#                CLOUDVIEWER  PLUGIN: ecvPoissonReconDlg                      #
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
//#                  COPYRIGHT: Daniel Girardeau-Montaut                   #
//#                                                                        #
//##########################################################################

#include "ecvPoissonReconDlg.h"
#include "ecvEntityAction.h"

// ECV_DB_LIB
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvHObjectCaster.h>

// QT
#include <QtGui>
#include <QtCore>
#include <QDialog>
#include <QMessageBox>
#include <QMainWindow>
#include <QInputDialog>
#include <QProgressDialog>
#include <QtConcurrentRun>

//System
#if defined(CV_WINDOWS)
#include "Windows.h"
#else
#include <time.h>
#include <unistd.h>
#endif

static double s_defaultResolution = 0.0;
static bool s_depthMode = true;
static ccPointCloud* s_cloud = nullptr;
static ccMesh* s_mesh = nullptr;

static std::vector<double> s_density;

//! Algorithm parameters
struct Parameters
{
	//! Default initializer
	Parameters() = default;

	//! Boundary types
	enum BoundaryType {
		FREE, DIRICHLET, NEUMANN, COUNT
	};

	//! Boundary type for the finite elements
	BoundaryType boundary = NEUMANN;

	//! The maximum depth of the tree that will be used for surface reconstruction
	/** Running at depth d corresponds to solving on a 2^d x 2^d x 2^d.
		Note that since the reconstructor adapts the octree to the sampling density,
		the specified reconstruction depth is only an upper bound.
	**/
	int depth = 8;

	//! The target width of the finest level octree cells (ignored if depth is specified)
	float finestCellWidth = 0.0f;

	//! The ratio between the diameter of the cube used for reconstruction and the diameter of the samples' bounding cube.
	/** Specifies the factor of the bounding cube that the input samples should fit into.
	**/
	float scale = 1.1f;

	//! The minimum number of sample points that should fall within an octree node as the octree construction is adapted to sampling density.
	/** This parameter specifies the minimum number of points that should fall within an octree node.
		For noise-free samples, small values in the range [1.0 - 5.0] can be used. For more noisy samples, larger values
		in the range [15.0 - 20.0] may be needed to provide a smoother, noise-reduced, reconstruction.
	**/
	float samplesPerNode = 1.5f;

	//! The importance that interpolation of the point samples is given in the formulation of the screened Poisson equation.
	/** The results of the original (unscreened) Poisson Reconstruction can be obtained by setting this value to 0.
	**/
	float pointWeight = 2.0f;

	//! The number of solver iterations
	/** Number of Gauss-Seidel relaxations to be performed at each level of the octree hierarchy.
	**/
	int iters = 8;

	//! If this flag is enabled, the sampling density is written out with the vertices
	bool density = false;

	//! This flag tells the reconstructor to read in color values with the input points and extrapolate those to the vertices of the output.
	bool withColors = true;

	//! Data pull factor
	/** If withColors is rue, this floating point value specifies the relative importance of finer color estimates over lower ones.
	**/
	float colorPullFactor = 32.0f;

	//! Normal confidence exponent
	/** Exponent to be applied to a point's confidence to adjust its weight. (A point's confidence is defined by the magnitude of its normal.)
	**/
	float normalConfidence = 0.0;

	//! Normal confidence bias exponent
	/** Exponent to be applied to a point's confidence to bias the resolution at which the sample contributes to the linear system. (Points with lower confidence are biased to contribute at coarser resolutions.)
	**/
	float normalConfidenceBias = 0.0;

	//! Enabling this flag has the reconstructor use linear interpolation to estimate the positions of iso-vertices.
	bool linearFit = false;

	//! This parameter specifies the number of threads across which the solver should be parallelized
	int threads = 1;

	/** The parameters below are accessible via the command line but are not described in the official documentation **/

	//! The depth beyond which the octree will be adapted.
	/** At coarser depths, the octree will be complete, containing all 2^d x 2^d x 2^d nodes.
	**/
	int fullDepth = 5;

	//! Coarse MG solver depth
	int baseDepth = 0;

	//! Coarse MG solver v-cycles
	int baseVCycles = 1;

	//! This flag specifies the accuracy cut-off to be used for CG
	float cgAccuracy = 1.0e-3f;

};
static Parameters s_params;

bool doReconstruct()
{
	//invalid parameters
	if (!s_cloud)
	{
		return false;
	}

	std::tuple<std::shared_ptr<ccMesh>, std::vector<double>> result =
		ccMesh::CreateFromPointCloudPoisson(
			*s_cloud,
			(size_t)s_params.depth,
			(size_t)s_params.finestCellWidth,
			s_params.scale,
			s_params.linearFit,
			s_params.pointWeight,
			s_params.samplesPerNode,
			s_params.boundary);

	s_mesh = new ccMesh();
	s_mesh->createInternalCloud();
	*s_mesh = *std::get<0>(result);
	
	if (s_params.density)
	{
		s_density = std::get<1>(result);
	}

	return !s_mesh->IsEmpty();
}

ecvPoissonReconDlg::ecvPoissonReconDlg(QWidget* parent)
	: QDialog(parent, Qt::Tool)
	, Ui::PoissonReconParamDialog()
	, m_app(parent)
	, m_applyAllClouds(false)
{
	setupUi(this);
	m_clouds.clear();
	m_normalsMask.clear();
	m_result.clear();
}

void ecvPoissonReconDlg::adjustParams(ccPointCloud* cloud)
{
	s_defaultResolution = cloud->getOwnBB().getDiagNormd() / 200.0;
	bool cloudHasColors = cloud->hasColors();
	importColorsCheckBox->setVisible(cloudHasColors);
	if (s_depthMode)
		depthRadioButton->setChecked(true);
	else
		resolutionRadioButton->setChecked(true);

	//init dialog with semi-persistent settings
	depthSpinBox->setValue(s_params.depth);
	resolutionDoubleSpinBox->setValue(s_defaultResolution);
	samplesPerNodeSpinBox->setValue(s_params.samplesPerNode);
	importColorsCheckBox->setChecked(s_params.withColors);
	densityCheckBox->setChecked(s_params.density);
	weightDoubleSpinBox->setValue(s_params.pointWeight);
	linearFitCheckBox->setChecked(s_params.linearFit);
	switch (s_params.boundary)
	{
	case Parameters::FREE:
		boundaryComboBox->setCurrentIndex(0);
		break;
	case Parameters::DIRICHLET:
		boundaryComboBox->setCurrentIndex(1);
		break;
	case Parameters::NEUMANN:
		boundaryComboBox->setCurrentIndex(2);
		break;
	case Parameters::COUNT:
		boundaryComboBox->setCurrentIndex(3);
		break;
	default:
		assert(false);
		break;
	}

}

bool ecvPoissonReconDlg::doComputation()
{
	bool result = false;
	{
		//progress dialog (Qtconcurrent::run can't be canceled!)
		QProgressDialog pDlg(tr("Initialization"), QString(), 0, 0, m_app);
		pDlg.setWindowTitle(tr("Poisson Reconstruction"));
		pDlg.show();
		//QApplication::processEvents();
		if (!s_cloud) return result;

		QString progressLabel(tr("Reconstruction for [%1] in progress\n").arg(s_cloud->getName()));
		if (s_depthMode)
			progressLabel += tr("level: %1").arg(s_params.depth);
		else
			progressLabel += tr("resolution: %1").arg(s_params.finestCellWidth);
		progressLabel += tr(" [%1 thread(s)]").arg(s_params.threads);

		pDlg.setLabelText(progressLabel);
		QApplication::processEvents();

		//run in a separate thread

		QFuture<bool> future = QtConcurrent::run(doReconstruct);

		//wait until process is finished!
		while (!future.isFinished())
		{
#if defined(CV_WINDOWS)
			::Sleep(500);
#else
			usleep(500 * 1000);
#endif

			pDlg.setValue(pDlg.value() + 1);
			QApplication::processEvents();
		}

		result = future.result();

		s_cloud = nullptr;

		pDlg.hide();
		QApplication::processEvents();
	}

	return result;
}

bool ecvPoissonReconDlg::addEntity(ccHObject * ent)
{
	if (!ent->isKindOf(CV_TYPES::POINT_CLOUD)) return false;
	
	m_clouds.push_back(ent);
	if (ent->isA(CV_TYPES::POINT_CLOUD))
	{
		m_normalsMask.push_back(static_cast<ccPointCloud*>(ent)->hasNormals());
	}
	else
	{
		m_normalsMask.push_back(false);
	}
	return false;
}

ccHObject::Container& ecvPoissonReconDlg::getReconstructions()
{
	return m_result;
}

bool ecvPoissonReconDlg::start()
{
	int nCount = std::count(m_normalsMask.begin(), m_normalsMask.end(), false);
	bool updateNormals = false;
	if (nCount > 0)
	{
		updateNormals = (QMessageBox::question(m_app,
			tr("Some clouds have no normals?"),
			tr("Clouds must have normals before poisson reconstruction. Do you want to compute normals?"),
			QMessageBox::Yes,
			QMessageBox::No) == QMessageBox::Yes);
		if (!updateNormals)
		{
			CVLog::Error(tr("m_clouds must have normals before poisson reconstruction!"));
			return false;
		}
		else
		{
			assert(m_normalsMask.size() == m_clouds.size());
			ccHObject::Container toBeEstimateNormals;
			for (size_t i = 0; i < m_normalsMask.size(); ++i)
			{
				if (!m_normalsMask[i])
					toBeEstimateNormals.push_back(m_clouds[i]);
			}
			if (!ccEntityAction::computeNormals(toBeEstimateNormals, m_app))
			{
				CVLog::Error(tr("Computer normals failed!"));
				return false;
			}

		}

	}

	m_applyAllClouds = false;
	for (size_t i = 0; i < m_clouds.size(); ++i)
	{
		ccHObject* ent = m_clouds[i];
		assert(ent->isKindOf(CV_TYPES::POINT_CLOUD));

		// poisson reconstruction parameters
		ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(ent);
		adjustParams(cloud);

		if (!m_applyAllClouds)
		{
			if (!showDialog())
			{
				continue;
			}
		}
		updateParams();

		// clear history
		s_cloud = cloud;
		s_mesh = nullptr;
		s_density.clear();

		// run in a separate thread
		// compute mesh
		if (!doComputation())
		{
			CVLog::Warning("");
			continue;
		}

		if (s_mesh)
		{
			cloud->setEnabled(false); //can't disable the cloud as the resulting mesh will be its child!
			if (cloud->getParent())
			{
				cloud->getParent()->addChild(s_mesh);
			}

			ccPointCloud* newPC =
				ccHObjectCaster::ToPointCloud(s_mesh->getAssociatedCloud());

			CVLog::Print(
				tr("[PoissonRecon] reconstruction from [%1] (%2 triangles, %3 vertices)").
				arg(cloud->getName()).arg(s_mesh->size()).arg(newPC->size()));

			s_mesh->setName(tr("Mesh[%1] (level %2)").arg(cloud->getName()).arg(s_params.depth));
			newPC->setEnabled(false);
			s_mesh->setVisible(true);
			s_mesh->computeNormals(true);
			if (!s_params.withColors) {
				newPC->unallocateColors();
				newPC->showColors(false);
			}
			s_mesh->showColors(newPC->hasColors());

			//copy Global Shift & Scale information
			newPC->setGlobalShift(cloud->getGlobalShift());
			newPC->setGlobalScale(cloud->getGlobalScale());

			if (s_params.density)
			{
				if (s_density.size() != newPC->size())
				{
					CVLog::Warning("[doActionPoissonReconstruction] density dimension does not match!");
					s_mesh->showColors(newPC->colorsShown());
					s_mesh->showSF(false);
					m_result.push_back(s_mesh);
					continue;
				}

				std::vector<std::vector<ScalarType>> scalarsVector;
				std::vector<std::vector<double>> tempScalarsvector;
				std::vector<ccHObject*> tempClouds;
				tempClouds.push_back(s_mesh);
				tempScalarsvector.push_back(s_density);
				ccEntityAction::ConvertToScalarType<double>(tempScalarsvector, scalarsVector);
				if (!ccEntityAction::importToSF(tempClouds, scalarsVector, "Density"))
				{
					CVLog::Error("[doActionPoissonReconstruction] import sf failed!");
					s_mesh->showSF(false);
				}
				else
				{
					s_mesh->showSF(true);
				}
				newPC->showSF(true);
				s_mesh->showColors(newPC->colorsShown());
				s_mesh->showSF(true);

			}

			m_result.push_back(s_mesh);
		}
		else
		{
			CVLog::Warning(
				tr("[ecvPoissonReconDlg] Poisson reconstruction from [%1] failed!").arg(ent->getName()));
		}
	}

	return true;
}

bool ecvPoissonReconDlg::showDialog()
{
	if (!this->exec())
	{
		return false;
	}

	//set parameters with dialog settings
	updateParams();

	return true;
}

void ecvPoissonReconDlg::updateParams()
{
	// Set parameters with dialog settings
	s_depthMode = depthRadioButton->isChecked();
	s_defaultResolution = resolutionDoubleSpinBox->value();
	s_params.depth = (s_depthMode ? depthSpinBox->value() : 0);
	s_params.finestCellWidth = static_cast<float>(s_depthMode ? 0.0 : s_defaultResolution);
	s_params.samplesPerNode = static_cast<float>(samplesPerNodeSpinBox->value());
	s_params.withColors = importColorsCheckBox->isChecked();
	s_params.density = densityCheckBox->isChecked();
	s_params.pointWeight = static_cast<float>(weightDoubleSpinBox->value());
	s_params.linearFit = linearFitCheckBox->isChecked();
	switch (boundaryComboBox->currentIndex())
	{
	case 0:
		s_params.boundary = Parameters::FREE;
		break;
	case 1:
		s_params.boundary = Parameters::DIRICHLET;
		break;
	case 2:
		s_params.boundary = Parameters::NEUMANN;
		break;
	case 3:
		s_params.boundary = Parameters::COUNT;
		break;
	default:
		assert(false);
		break;
	}

	m_applyAllClouds = applyParamsAllCloudsCheckBox->isChecked();
}

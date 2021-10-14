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

#include "ecvGraphicalTransformationTool.h"
#include "MainWindow.h"

//ECV_DB_LIB
#include <CVLog.h>
#include <ecvMesh.h>
#include <ecvFacet.h>
#include <ecvPolyline.h>
#include <ecvDisplayTools.h>
#include <ecvGenericTransformTool.h>

ccGraphicalTransformationTool::ccGraphicalTransformationTool(QWidget* parent)
	: ccOverlayDialog(parent)
	, Ui::GraphicalTransformationDlg()
	, m_toTransform("transformed")
{
	setupUi(this);

	connect(pauseButton,    &QAbstractButton::toggled,	this, &ccGraphicalTransformationTool::pause);
	connect(okButton,       &QAbstractButton::clicked,	this, &ccGraphicalTransformationTool::apply);
    connect(razButton,	    &QAbstractButton::clicked,	this, &ccGraphicalTransformationTool::reset);
	connect(cancelButton,   &QAbstractButton::clicked,	this, &ccGraphicalTransformationTool::cancel);

	//add shortcuts
	addOverridenShortcut(Qt::Key_Space); //space bar for the "pause" button
	addOverridenShortcut(Qt::Key_Escape); //escape key for the "cancel" button
	addOverridenShortcut(Qt::Key_Return); //return key for the "ok" button
	connect(this, &ccOverlayDialog::shortcutTriggered, this, &ccGraphicalTransformationTool::onShortcutTriggered);
	connect(this->rotComboBox, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &ccGraphicalTransformationTool::onRotationModeChanged);
	connect(this->TxCheckBox, &QCheckBox::toggled, this, &ccGraphicalTransformationTool::onTranlationModeChanged);
	connect(this->TyCheckBox, &QCheckBox::toggled, this, &ccGraphicalTransformationTool::onTranlationModeChanged);
	connect(this->TzCheckBox, &QCheckBox::toggled, this, &ccGraphicalTransformationTool::onTranlationModeChanged);
	connect(this->scaleCheckBox, &QCheckBox::toggled, this, &ccGraphicalTransformationTool::onScaleEnabled);
	connect(this->shearCheckBox, &QCheckBox::toggled, this, &ccGraphicalTransformationTool::onShearEnabled);
}

ccGraphicalTransformationTool::~ccGraphicalTransformationTool()
{
	clear();
}

void ccGraphicalTransformationTool::onShortcutTriggered(int key)
{
	switch(key)
	{
	case Qt::Key_Space:
		pauseButton->toggle();
		return;

	case Qt::Key_Return:
		okButton->click();
		return;

	case Qt::Key_Escape:
		cancelButton->click();
		return;

	default:
		//nothing to do
		break;
	}
}

void ccGraphicalTransformationTool::onScaleEnabled(bool dummy)
{
	if (!m_tool)
	{
		return;
	}

	m_tool->setScaleEnabled(this->scaleCheckBox->isChecked());
}

void ccGraphicalTransformationTool::onShearEnabled(bool dummy)
{
	if (!m_tool)
	{
		return;
	}

	m_tool->setShearEnabled(this->shearCheckBox->isChecked());
}

void ccGraphicalTransformationTool::onRotationModeChanged(int dummy)
{
	if (!m_tool)
	{
		return;
	}

	switch (rotComboBox->currentIndex())
	{
	case 0: //XYZ
		m_tool->setRotationMode(ecvGenericTransformTool::RotationMode::R_XYZ);
		break;
	case 1: //X
		m_tool->setRotationMode(ecvGenericTransformTool::RotationMode::R_X);
		break;
	case 2: //Y
		m_tool->setRotationMode(ecvGenericTransformTool::RotationMode::R_Y);
		break;
	case 3: //Z
		m_tool->setRotationMode(ecvGenericTransformTool::RotationMode::R_Z);
		break;
	}
}

void ccGraphicalTransformationTool::onTranlationModeChanged(bool dummy)
{
	if (!m_tool)
	{
		return;
	}

	int flag = 0;
	flag += TxCheckBox->isChecked() * 1;
	flag += TyCheckBox->isChecked() * 2;
	flag += TzCheckBox->isChecked() * 3;
	switch (flag)
	{
	case 0:
		m_tool->setTranlationMode(ecvGenericTransformTool::TranslationMOde::T_NONE);
		break;
	case 1:
		m_tool->setTranlationMode(ecvGenericTransformTool::TranslationMOde::T_X);
		break;
	case 2:
		m_tool->setTranlationMode(ecvGenericTransformTool::TranslationMOde::T_Y);
		break;
	case 3:
	{
		if (TzCheckBox->isChecked())
		{
			m_tool->setTranlationMode(ecvGenericTransformTool::TranslationMOde::T_Z);
		}
		else
		{
			m_tool->setTranlationMode(ecvGenericTransformTool::TranslationMOde::T_XY);
		}
	}
		break;
	case 4:
		m_tool->setTranlationMode(ecvGenericTransformTool::TranslationMOde::T_XZ);
		break;
	case 5:
		m_tool->setTranlationMode(ecvGenericTransformTool::TranslationMOde::T_ZY);
		break;
	case 6:
		m_tool->setTranlationMode(ecvGenericTransformTool::TranslationMOde::T_XYZ);
		break;

	default:
		break;
	}
}

void ccGraphicalTransformationTool::pause(bool state)
{
	if (!ecvDisplayTools::GetCurrentScreen())
		return;

	if (state)
	{
		//ecvDisplayTools::SetInteractionMode(ecvDisplayTools::TRANSFORM_CAMERA());
		ecvDisplayTools::DisplayNewMessage("Transformation [PAUSED]", ecvDisplayTools::UPPER_CENTER_MESSAGE, false, 3600, ecvDisplayTools::MANUAL_TRANSFORMATION_MESSAGE);
		ecvDisplayTools::DisplayNewMessage("Unpause to transform again", ecvDisplayTools::UPPER_CENTER_MESSAGE, true, 3600, ecvDisplayTools::MANUAL_TRANSFORMATION_MESSAGE);
	}
	else
	{
		//ecvDisplayTools::SetInteractionMode(ecvDisplayTools::TRANSFORM_ENTITIES());
		ecvDisplayTools::DisplayNewMessage("[Rotation/Translation mode]", ecvDisplayTools::UPPER_CENTER_MESSAGE, false, 3600, ecvDisplayTools::MANUAL_TRANSFORMATION_MESSAGE);
	}

	if (m_tool)
	{
		m_tool->showInteractor(!state);
	}

	//update mini-GUI
	pauseButton->blockSignals(true);
	pauseButton->setChecked(state);
	pauseButton->blockSignals(false);

	ecvDisplayTools::SetRedrawRecursive(false);
	ecvDisplayTools::RedrawDisplay(true, false);
}

void ccGraphicalTransformationTool::clear()
{
	m_toTransform.detatchAllChildren();

	if (m_tool)
	{
		m_tool->clear();
	}

	ecvDisplayTools::DisplayNewMessage(QString(), ecvDisplayTools::UPPER_CENTER_MESSAGE); //clear the area

}

bool ccGraphicalTransformationTool::addEntity(ccHObject* entity)
{
	assert(entity);
	if (!entity)
		return false;

	//we can't transform locked entities
	if (entity->isLocked())
	{
		CVLog::Warning(QString("[Graphical Transformation Tool] Can't transform entity '%1' cause it's locked!").arg(entity->getName()));
		return false;
	}

	//we can't transform child meshes
	if (entity->isA(CV_TYPES::MESH) && entity->getParent() && entity->getParent()->isKindOf(CV_TYPES::MESH))
	{
		CVLog::Warning(QString("[Graphical Transformation Tool] Entity '%1' can't be modified as it is part of a mesh group. You should 'clone' it first.").arg(entity->getName()));
		return false;
	}

	//eventually, we must check that there is no "parent + sibling" in the selection!
	//otherwise, the sibling will be rotated twice!
	unsigned n = m_toTransform.getChildrenNumber();
	for (unsigned i=0; i<n; )
	{
		ccHObject* previous = m_toTransform.getChild(i);
		if (previous->isAncestorOf(entity))
		{
			//we have found a parent, we won't add this entity
			return false;
		}
		//if the inverse is true, then we get rid of the current element!
		else if (entity->isAncestorOf(previous))
		{
			m_toTransform.detachChild(previous);
			--n;
		}
		else
		{
			//proceed
			++i;
		}
	}

    // TODO+ add ccFacet, ccCoordinateSystem and ccSensor support!
    if (entity->isA(CV_TYPES::FACET) ||
        entity->isKindOf(CV_TYPES::SENSOR) ||
        entity->isA(CV_TYPES::COORDINATESYSTEM))
    {
        CVLog::Warning(QString("[Graphical Transformation Tool] Can't transform entity '%1' cause it's not supported now!").arg(entity->getName()));
        return false;
    }
    else {
        m_toTransform.addChild(entity,ccHObject::DP_NONE);
    }

	return true;
}

unsigned ccGraphicalTransformationTool::getNumberOfValidEntities() const
{
    return m_toTransform.getChildrenNumber();
}

void ccGraphicalTransformationTool::setRotationCenter(CCVector3d & center)
{
	m_rotationCenter = center;
}

bool ccGraphicalTransformationTool::linkWith(QWidget* win)
{
	if (!ccOverlayDialog::linkWith(win))
	{
		return false;
	}
	
	//assert(m_toTransform.getChildrenNumber() == 0);
	return true;
}

bool ccGraphicalTransformationTool::start()
{
	assert(!m_processing);
	if (!ecvDisplayTools::GetCurrentScreen() || !m_tool)
		return false;

	unsigned childNum = m_toTransform.getChildrenNumber();
	if (childNum == 0)
		return false;

	if (!m_tool->setInputData(&m_toTransform))
	{
		return false;
	}

	if (!m_tool->start())
	{
		return false;
	}
	
	// update m_tool
	onScaleEnabled(true);
	onShearEnabled(false);
	onRotationModeChanged(0);
	onTranlationModeChanged(0);

	pause(this->pauseButton->isChecked());

	return ccOverlayDialog::start();
}

void ccGraphicalTransformationTool::stop(bool state)
{
	if (m_tool)
	{
		m_tool->stop();
		m_tool->clear();
		delete m_tool;
		m_tool = nullptr;
	}

	ccOverlayDialog::stop(state);
}

bool ccGraphicalTransformationTool::setTansformTool(ecvGenericTransformTool * tool)
{
	if (!tool)
	{
		return false;
	}

	m_tool = tool;
	return true;
}

void ccGraphicalTransformationTool::reset()
{
	if (!m_tool)
	{
		CVLog::Warning(QString("[ccGraphicalTransformationTool::reset] transformation tool has not been initialized!"));
		return;
	}

	m_tool->reset();
}

void ccGraphicalTransformationTool::exportNewEntities()
{
	if (!m_tool || !MainWindow::TheInstance())
		return;

	ccHObject::Container tranformedEntities;
	m_tool->getOutput(tranformedEntities);
	if (tranformedEntities.size() != getNumberOfValidEntities())
	{
		assert(false);
		return;
	}

	for (unsigned ci = 0; ci != getNumberOfValidEntities(); ++ci)
	{
		ccHObject* obj = m_toTransform.getChild(ci);
		if (!obj)
		{
			assert(false);
			continue;
		}

		ccHObject* result = tranformedEntities[ci];
		if (result)
		{
			result->setEnabled(true);
			obj->setEnabled(false);
			result->setName(obj->getName() + QString(".ScaleShear"));
			if (obj->getParent())
				obj->getParent()->addChild(result);
			MainWindow::TheInstance()->addToDB(result);
		}
		else
		{
			CVLog::Warning(QString("ignore entity [ID:%1] transformation!").arg(obj->getUniqueID()));
		}
	}

	// reset interactors or model if necessary
	reset();
}

void ccGraphicalTransformationTool::apply()
{
	//we recompute global GL transformation matrix and display it in console
	if (!m_tool)
	{
		CVLog::Warning(QString("[ccGraphicalTransformationTool::apply] transformation tool has not been initialized!"));
		stop(true);
		clear();
		return;
	}

	// non rigid transform
	if (this->scaleCheckBox->isChecked() || this->shearCheckBox->isChecked())
	{
		exportNewEntities();
		stop(true);
		clear();
		return;
	}

	// rigid transform
	ccGLMatrixd finalTrans;
	finalTrans = m_tool->getFinalTransformation();

	ccGLMatrixd finalTransCorrected = finalTrans;
#define NORMALIZE_TRANSFORMATION_MATRIX_WITH_EULER
#ifdef NORMALIZE_TRANSFORMATION_MATRIX_WITH_EULER
	{
		//convert matrix back and forth so as to be sure to get a 'true' rotation matrix
		//DGM: we use Euler angles, as the axis/angle method (formerly used) is not robust
		//enough! Shifts could be perceived by the user.
		double phi_rad,theta_rad,psi_rad;
		CCVector3d t3D;
		finalTrans.getParameters(phi_rad,theta_rad,psi_rad,t3D);
		finalTransCorrected.initFromParameters(phi_rad,theta_rad,psi_rad,t3D);

#ifdef QT_DEBUG
		CVLog::Print("[GraphicalTransformationTool] Final transformation (before correction):");
		CVLog::Print(finalTrans.toString(12,' ')); //full precision
		CVLog::Print(QString("Angles(%1,%2,%3) T(%5,%6,%7)").arg(phi_rad).arg(theta_rad).arg(psi_rad).arg(t3D.x).arg(t3D.y).arg(t3D.z));
#endif
	}
#endif //NORMALIZE_TRANSFORMATION_MATRIX_WITH_EULER

#ifdef QT_DEBUG
	//test: compute rotation "norm" (as it may not be exactly 1 due to numerical (in)accuracy!)
	{
		ccGLMatrixd finalRotation = finalTransCorrected;
		finalRotation.setTranslation(CCVector3(0,0,0));
		ccGLMatrixd finalRotationT = finalRotation.transposed();
		ccGLMatrixd idTrans = finalRotation * finalRotationT;
		double norm = idTrans.data()[0] * idTrans.data()[5] * idTrans.data()[10];
		CVLog::PrintDebug("[GraphicalTransformationTool] T*T-1:");
		CVLog::PrintDebug(idTrans.toString(12,' ')); //full precision
		CVLog::PrintDebug(QString("Rotation norm = %1").arg(norm,0,'f',12));
	}
#endif

	//update GL transformation for all entities
	ccGLMatrix correctedFinalTrans(finalTransCorrected.data());

	ecvDisplayTools::SetRedrawRecursive(false);
	for (unsigned i=0; i<m_toTransform.getChildrenNumber(); ++i)
	{
		ccHObject* toTransform = m_toTransform.getChild(i);
		toTransform->setGLTransformation(correctedFinalTrans);

		//DGM: warning, applyGLTransformation may delete the associated octree!
		MainWindow::ccHObjectContext objContext = MainWindow::TheInstance()->removeObjectTemporarilyFromDBTree(toTransform);

		toTransform->applyGLTransformation_recursive();
		//toTransform->prepareDisplayForRefresh_recursive();
		MainWindow::TheInstance()->putObjectBackIntoDBTree(toTransform, objContext);

		toTransform->setRedrawFlagRecursive(true);

		//special case: if the object is a mesh vertices set, we may have to update the mesh normals!
		if (toTransform->isA(CV_TYPES::POINT_CLOUD) && toTransform->getParent() && toTransform->getParent()->isKindOf(CV_TYPES::MESH))
		{
			ccMesh* mesh = static_cast<ccMesh*>(toTransform->getParent());
			if (mesh->hasTriNormals() && !m_toTransform.isAncestorOf(mesh))
			{
				mesh->transformTriNormals(correctedFinalTrans);
			}
		}
	}

	stop(true);

	clear();

	ecvDisplayTools::RedrawDisplay();

	//output resulting transformation matrix
	CVLog::Print("[GraphicalTransformationTool] Applied transformation:");
	CVLog::Print(correctedFinalTrans.toString(12,' ')); //full precision

#ifdef QT_DEBUG
	{
		float phi_rad,theta_rad,psi_rad;
		CCVector3f t3D;
		correctedFinalTrans.getParameters(phi_rad,theta_rad,psi_rad,t3D);
		CVLog::Print(QString("Angles(%1,%2,%3) T(%5,%6,%7)").arg(phi_rad).arg(theta_rad).arg(psi_rad).arg(t3D.x).arg(t3D.y).arg(t3D.z));
	}
#endif

}

void ccGraphicalTransformationTool::cancel()
{
	for (unsigned i=0; i<m_toTransform.getChildrenNumber(); ++i)
	{
		ccHObject* child = m_toTransform.getChild(i);
		child->resetGLTransformation();
		child->setRedraw(true);
	}

	reset();

	stop(false);

	clear();
}

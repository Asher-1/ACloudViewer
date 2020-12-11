/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkScalarBarRepresentation.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/*
 * Copyright 2008 Sandia Corporation.
 * Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
 * license for use of this work by or on behalf of the
 * U.S. Government. Redistribution and use in source and binary forms, with
 * or without modification, are permitted provided that this Notice and any
 * statement of authorship are reproduced on all copies.
 */

#include "vtkScalarBarWidgetCustom.h"

#include "VTKExtensions/Views/vtkScalarBarActorCustom.h"
#include "VTKExtensions/Views/vtkContext2DScalarBarActor.h"
#include "VTKExtensions/Views/vtkScalarBarRepresentationCustom.h"

#include "vtkTextProperty.h"
#include "vtkCallbackCommand.h"
#include "vtkCoordinate.h"
#include "vtkObjectFactory.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkScalarBarActor.h"
#include "vtkScalarBarRepresentation.h"
#include "vtkWidgetCallbackMapper.h"
#include "vtkWidgetEvent.h"
//-----------------------------------------------------------------------------
vtkStandardNewMacro(vtkScalarBarWidgetCustom);

//-------------------------------------------------------------------------
vtkScalarBarWidgetCustom::vtkScalarBarWidgetCustom()
{
	this->Selectable = 0;
	this->Repositionable = 1;

	// Override the subclasses callback to handle the Repositionable flag.
	this->CallbackMapper->SetCallbackMethod(
		vtkCommand::MouseMoveEvent, vtkWidgetEvent::Move, this, Superclass::MoveAction);
}

//-------------------------------------------------------------------------
vtkScalarBarWidgetCustom::~vtkScalarBarWidgetCustom() = default;

//-----------------------------------------------------------------------------
void vtkScalarBarWidgetCustom::SetRepresentation(vtkScalarBarRepresentation* rep)
{
	this->SetWidgetRepresentation(rep);
}

//-----------------------------------------------------------------------------
void vtkScalarBarWidgetCustom::SetScalarBarActor(vtkScalarBarActor* actor)
{
	vtkScalarBarRepresentation* rep = this->GetScalarBarRepresentation();
	if (!rep)
	{
		this->CreateDefaultRepresentation();
		rep = this->GetScalarBarRepresentation();
	}

	if (rep->GetScalarBarActor() != actor)
	{
		rep->SetScalarBarActor(actor);
		this->Modified();
	}
}

//-----------------------------------------------------------------------------
vtkScalarBarActor* vtkScalarBarWidgetCustom::GetScalarBarActor()
{
	vtkScalarBarRepresentation* rep = this->GetScalarBarRepresentation();
	if (!rep)
	{
		this->CreateDefaultRepresentation();
		rep = this->GetScalarBarRepresentation();
	}

	return rep->GetScalarBarActor();
}

//-----------------------------------------------------------------------------
void vtkScalarBarWidgetCustom::CreateDefaultRepresentation()
{
	if (!this->WidgetRep)
	{
		vtkScalarBarRepresentationCustom* rep = vtkScalarBarRepresentationCustom::New();
		rep->SetWindowLocation(vtkScalarBarRepresentationCustom::LowerRightCorner);
		this->SetRepresentation(rep);
		rep->Delete();
	}
}

void vtkScalarBarWidgetCustom::CreateDefaultScalarBarActor()
{
	vtkContext2DScalarBarActor* lut_actor = vtkContext2DScalarBarActor::New();
	lut_actor->SetTitle("");
	lut_actor->SetOrientationToVertical();
	vtkSmartPointer<vtkTextProperty> prop = lut_actor->GetLabelTextProperty();
	prop->SetFontSize(10);
	lut_actor->SetLabelTextProperty(prop);
	lut_actor->SetTitleTextProperty(prop);
	this->SetScalarBarActor(lut_actor);
	lut_actor->Delete();
}

//-------------------------------------------------------------------------
void vtkScalarBarWidgetCustom::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os, indent);

	os << indent << "Repositionable: " << this->Repositionable << endl;
}

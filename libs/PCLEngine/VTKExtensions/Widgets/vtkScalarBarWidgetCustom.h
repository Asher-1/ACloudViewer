/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkScalarBarWidgetCustom.h

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

/**
 * @class   vtkScalarBarWidgetCustom
 * @brief   Represent scalar bar for vtkScalarBarWidgetCustom.
 *
 * Subclass of vtkScalarBarRepresentation that provides convenience functions
 * for placing the scalar bar widget in the scene.
 */

#ifndef vtkPVScalarBarRepresentationCustom_h
#define vtkPVScalarBarRepresentationCustom_h

#include "vtkScalarBarWidget.h"
#include "vtkScalarBarRepresentation.h"

class vtkScalarBarWidgetCustom : public vtkScalarBarWidget
{
public:
	static vtkScalarBarWidgetCustom* New();
	vtkTypeMacro(vtkScalarBarWidgetCustom, vtkBorderWidget);
	void PrintSelf(ostream& os, vtkIndent indent) override;

	/**
	 * Specify an instance of vtkWidgetRepresentation used to represent this
	 * widget in the scene. Note that the representation is a subclass of vtkProp
	 * so it can be added to the renderer independent of the widget.
	 */
	virtual void SetRepresentation(vtkScalarBarRepresentation* rep) override;

	/**
	 * Return the representation as a vtkScalarBarRepresentation.
	 */
	vtkScalarBarRepresentation* GetScalarBarRepresentation()
	{
		return reinterpret_cast<vtkScalarBarRepresentation*>(this->GetRepresentation());
	}

	//@{
	/**
	 * Get the ScalarBar used by this Widget. One is created automatically.
	 */
	virtual void SetScalarBarActor(vtkScalarBarActor* actor) override;
	virtual vtkScalarBarActor* GetScalarBarActor() override;

	/**
	 * Create the default widget representation if one is not set.
	 */
	void CreateDefaultRepresentation() override;

	void CreateDefaultScalarBarActor();

protected:
	vtkScalarBarWidgetCustom();
	~vtkScalarBarWidgetCustom() override;

private:
	vtkScalarBarWidgetCustom(const vtkScalarBarWidgetCustom&) = delete;
	void operator=(const vtkScalarBarWidgetCustom&) = delete;
};

#endif // vtkPVScalarBarRepresentationCustom_h

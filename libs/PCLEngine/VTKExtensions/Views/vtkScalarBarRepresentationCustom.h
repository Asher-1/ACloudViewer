/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkScalarBarRepresentationCustom.h

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
 * @class   vtkScalarBarRepresentationCustom
 * @brief   Represent scalar bar for vtkScalarBarWidget.
 *
 * Subclass of vtkScalarBarRepresentation that provides convenience functions
 * for placing the scalar bar widget in the scene.
 */

#ifndef vtkScalarBarRepresentationCustom_h
#define vtkScalarBarRepresentationCustom_h

#include "qPCL.h"

#include "vtkScalarBarRepresentation.h"

class QPCL_ENGINE_LIB_API vtkScalarBarRepresentationCustom
    : public vtkScalarBarRepresentation {
public:
  vtkTypeMacro(vtkScalarBarRepresentationCustom, vtkScalarBarRepresentation) void PrintSelf(
    ostream& os, vtkIndent indent) override;
  static vtkScalarBarRepresentationCustom* New();

  enum
  {
    AnyLocation = 0,
    LowerLeftCorner,
    LowerRightCorner,
    LowerCenter,
    UpperLeftCorner,
    UpperRightCorner,
    UpperCenter
  };

  //@{
  /**
   * Set the scalar bar position, by enumeration (
   * AnyLocation = 0,
   * LowerLeftCorner,
   * LowerRightCorner,
   * LowerCenter,
   * UpperLeftCorner,
   * UpperRightCorner,
   * UpperCenter)
   * related to the render window.
   */
  vtkSetMacro(WindowLocation, int);
  vtkGetMacro(WindowLocation, int);
  //@}

  /**
   * Override to obtain viewport size and potentially adjust placement
   * of the representation.
   */
  int RenderOverlay(vtkViewport*) override;

protected:
  vtkScalarBarRepresentationCustom();
  ~vtkScalarBarRepresentationCustom() override;

  int WindowLocation;

private:
  vtkScalarBarRepresentationCustom(const vtkScalarBarRepresentationCustom&) = delete;
  void operator=(const vtkScalarBarRepresentationCustom&) = delete;
};

#endif // vtkScalarBarRepresentationCustom

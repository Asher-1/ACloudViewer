// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------


#import <Cocoa/Cocoa.h>
#include "vtkRenderWindowInteractorFix.h"
#include <vtkCocoaRenderWindow.h>
#include <vtkCocoaRenderWindowInteractor.h>
#include <vtkObjectFactory.h>

vtkRenderWindowInteractor* vtkRenderWindowInteractorFixNew ()
{
  return (vtkCocoaRenderWindowInteractor::New ());
}

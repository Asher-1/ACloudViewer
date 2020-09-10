/*=========================================================================

   Program: ParaView
   Module:    EditCameraTool.h

   Copyright (c) 2005-2008 Sandia Corporation, Kitware Inc.
   All rights reserved.

   ParaView is a free software; you can redistribute it and/or modify it
   under the terms of the ParaView license version 1.2.

   See License_v1.2.txt for the full ParaView license.
   A copy of this license can be obtained by contacting
   Kitware Inc.
   28 Corporate Drive
   Clifton Park, NY 12065
   USA

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/

#ifndef QPCL_VTK_EDITCAMERA_TOOL_HEADER
#define QPCL_VTK_EDITCAMERA_TOOL_HEADER

#include "qPCL.h"
#include "ecvGenericCameraTool.h"

#include <QObject>

class QPCL_ENGINE_LIB_API EditCameraTool : public ecvGenericCameraTool
{
  Q_OBJECT
public:
  EditCameraTool(ecvGenericVisualizer3D* viewer);
  ~EditCameraTool() override;

  static void UpdateCameraInfo();
  static void UpdateCamera();
  static void SetVisualizer(ecvGenericVisualizer3D* viewer);

private slots:
	// Description:
	// Choose a file and load/save camera properties.
	virtual void saveCameraConfiguration(const std::string& file) override;
	virtual void loadCameraConfiguration(const std::string& file) override;

	virtual void resetViewDirection(
	double look_x, double look_y, double look_z,
		double up_x, double up_y, double up_z) override;

	virtual void updateCamera() override;
	virtual void updateCameraParameters() override;

private:
  virtual void adjustCamera(CameraAdjustmentType enType, double value) override;

};

#endif // QPCL_VTK_EDITCAMERA_TOOL_HEADER

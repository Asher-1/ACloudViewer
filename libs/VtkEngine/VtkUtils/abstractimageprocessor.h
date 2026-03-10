// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file abstractimageprocessor.h
/// @brief Base class for VTK image processing algorithms run as
/// SignalledRunnable.

#include <vtkImageGradient.h>
#include <vtkImageLaplacian.h>
#include <vtkJPEGReader.h>

#include "qVTK.h"
#include "signalledrunable.h"

namespace VtkUtils {

/// @class AbstractImageProcessor
/// @brief Base for VTK image processors; runs in thread and emits finished().
class QVTK_ENGINE_LIB_API AbstractImageProcessor : public SignalledRunnable {
public:
    AbstractImageProcessor();

    /// @param input VTK image data to process
    void setInputData(vtkImageData* input) { m_imageData = input; }
    /// @return Input image data
    vtkImageData* inputData() const { return m_imageData; }

protected:
    vtkImageData* m_imageData = nullptr;
};

/// @class ImageProcessorTempl
/// @brief Template image processor wrapping a VTK image algorithm.
/// @tparam T VTK image algorithm type (e.g. vtkImageGradient)
template <class T>
class ImageProcessorTempl : public AbstractImageProcessor {
public:
    ImageProcessorTempl() { m_algorithm = T::New(); }

    /// @return Underlying VTK algorithm
    T* algorithm() const { return m_algorithm; }

    void run() {
        if (!m_imageData) {
            emit finished();
            return;
        }

        m_algorithm->SetInputData(m_imageData);
        m_algorithm->Update();
        emit finished();
    }

protected:
    T* m_algorithm = nullptr;
};

}  // namespace VtkUtils

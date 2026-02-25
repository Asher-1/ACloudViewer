// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkImageGradient.h>
#include <vtkImageLaplacian.h>
#include <vtkJPEGReader.h>

#include "qPCL.h"
#include "signalledrunable.h"

namespace VtkUtils {

class QPCL_ENGINE_LIB_API AbstractImageProcessor : public SignalledRunnable {
public:
    AbstractImageProcessor();

    void setInputData(vtkImageData* input) { m_imageData = input; }
    vtkImageData* inputData() const { return m_imageData; }

protected:
    vtkImageData* m_imageData = nullptr;
};

template <class T>
class ImageProcessorTempl : public AbstractImageProcessor {
public:
    ImageProcessorTempl() { m_algorithm = T::New(); }

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

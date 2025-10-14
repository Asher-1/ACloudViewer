// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_TEMPLATEALIGNMENT_HEADER
#define Q_PCL_PLUGIN_TEMPLATEALIGNMENT_HEADER

#include "BasePclModule.h"

// Qt
#include <QString>

class TemplateAlignmentDialog;

namespace PCLModules {
class TemplateMatching;
}

//! Template Alignment
class TemplateAlignment : public BasePclModule {
public:
    TemplateAlignment();
    virtual ~TemplateAlignment();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    void applyTransformation(ccHObject* entity, const ccGLMatrixd& mat);

    ccPointCloud* m_targetCloud;
    TemplateAlignmentDialog* m_dialog;
    PCLModules::TemplateMatching* m_templateMatch;

    float m_leafSize;
    bool m_useVoxelGrid;

    // PCLUtils::FeatureCloud parameters
    int m_maxThreadCount;
    float m_normalRadius;
    float m_featureRadius;

    // PCLUtils::TemplateMatching parameters
    int m_maxIterations;
    float m_minSampleDistance;
    float m_maxCorrespondenceDistance;

    std::vector<float> m_scales;
    std::vector<ccPointCloud*> m_templateClouds;
};

#endif  // Q_PCL_PLUGIN_TEMPLATEALIGNMENT_HEADER

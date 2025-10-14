// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_GENERIC_TRANSFORM_TOOL_HEADER
#define ECV_GENERIC_TRANSFORM_TOOL_HEADER

#include "eCV_db.h"
#include "ecvGLMatrix.h"

// QT
#include <QFile>

class ccHObject;
class ecvGenericVisualizer3D;

//! Generic Annotation Tool interface
class ECV_DB_LIB_API ecvGenericTransformTool : public QObject {
    Q_OBJECT
public:
    //! Default constructor
    /**
            \param mode Annotation mode
    **/

    enum RotationMode { R_XYZ, R_X, R_Y, R_Z };

    enum TranslationMOde { T_X, T_Y, T_Z, T_XY, T_XZ, T_ZY, T_XYZ, T_NONE };

    ecvGenericTransformTool();
    virtual ~ecvGenericTransformTool() = default;

    ccHObject* getAssociatedEntity() { return m_associatedEntity; }

public:
    virtual void setVisualizer(ecvGenericVisualizer3D* viewer = nullptr) = 0;
    virtual bool setInputData(ccHObject* entity, int viewport = 0);

    virtual void showInteractor(bool state) = 0;

    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual void reset() = 0;
    virtual void clear() = 0;
    virtual void setTranlationMode(TranslationMOde mode) = 0;
    virtual void setRotationMode(RotationMode mode) = 0;
    virtual void setScaleEnabled(bool state) = 0;
    virtual void setShearEnabled(bool state) = 0;

    virtual const ccGLMatrixd getFinalTransformation() = 0;

    virtual void getOutput(std::vector<ccHObject*>& out) = 0;

signals:
    void tranformMatrix(const ccGLMatrixd& transMatrix);

protected:
    ccHObject* m_associatedEntity;
};

#endif  // ECV_GENERIC_TRANSFORM_TOOL_HEADER

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPlanarEntityInterface.h"

// Local
#include "ecvCone.h"
#include "ecvCylinder.h"
#include "ecvDisplayTools.h"

// Qt
#include <QSharedPointer>

ccPlanarEntityInterface::ccPlanarEntityInterface()
    : m_showNormalVector(false), m_uniqueId(0) {}

// unit normal representation
static QSharedPointer<ccCylinder> c_unitNormalSymbol(0);
static QSharedPointer<ccCone> c_unitNormalHeadSymbol(0);
static const QString SEPARATOR = "_";

ccPlanarEntityInterface::ccPlanarEntityInterface(unsigned int id)
    : m_showNormalVector(false), m_uniqueId(id) {}

void ccPlanarEntityInterface::glDrawNormal(CC_DRAW_CONTEXT& context,
                                           const CCVector3& pos,
                                           float scale,
                                           const ecvColor::Rgb* color /*=0*/) {
    // get the set of OpenGL functions (version 2.1)

    if (ecvDisplayTools::GetCurrentScreen() == nullptr) return;

    // delete history
    clearNormalVector(context);

    if (!normalVectorIsShown()) {
        return;
    }

    if (!c_unitNormalSymbol) {
        c_unitNormalSymbol = QSharedPointer<ccCylinder>(
                new ccCylinder(0.02f, 0.9f, 0, "UnitNormal", 12));
        c_unitNormalSymbol->showColors(true);
        c_unitNormalSymbol->setVisible(true);
        c_unitNormalSymbol->setEnabled(true);
        c_unitNormalSymbol->setTempColor(ecvColor::green);
        c_unitNormalSymbol->setFixedId(true);
    }
    if (!c_unitNormalHeadSymbol) {
        c_unitNormalHeadSymbol = QSharedPointer<ccCone>(
                new ccCone(0.05f, 0.0f, 0.1f, 0, 0, 0, "UnitNormalHead", 12));
        c_unitNormalHeadSymbol->showColors(true);
        c_unitNormalHeadSymbol->setVisible(true);
        c_unitNormalHeadSymbol->setEnabled(true);
        c_unitNormalHeadSymbol->setTempColor(ecvColor::green);
        c_unitNormalHeadSymbol->setFixedId(true);
    }

    if (c_unitNormalHeadSymbol) {
        m_headId = QString::number(m_uniqueId) + SEPARATOR +
                   c_unitNormalHeadSymbol->getViewId();
    }

    if (c_unitNormalSymbol) {
        m_bodyId = QString::number(m_uniqueId) + SEPARATOR +
                   c_unitNormalSymbol->getViewId();
    }

    // build-up the normal representation own 'context'
    CC_DRAW_CONTEXT normalContext = context;
    // we must remove the 'push name flag' so that the primitives don't push
    // their own!
    normalContext.drawingFlags &= (~CC_ENTITY_PICKING);

    if (color) {
        c_unitNormalSymbol->setTempColor(*color, true);
        c_unitNormalHeadSymbol->setTempColor(*color, true);
    } else {
        c_unitNormalSymbol->enableTempColor(false);
        c_unitNormalHeadSymbol->enableTempColor(false);
    }

    c_unitNormalSymbol->setRedraw(true);
    c_unitNormalHeadSymbol->setRedraw(true);

    CCVector3 posVec(pos.x, pos.y, pos.z);
    normalContext.transformInfo.setTranslationStart(posVec);
    CCVector3 direction(0, 0, PC_ONE * scale);

    ccGLMatrixd mat = ccGLMatrixd(
            ccGLMatrix::FromToRotation(CCVector3(0, 0, PC_ONE), getNormal())
                    .data());
    mat.applyRotation(direction);
    normalContext.transformInfo.setTransformation(mat, false, false);

    // ccGL::Scale(glFunc, scale, scale, scale);
    normalContext.transformInfo.setScale(CCVector3(scale, scale, scale));

    // glFunc->glTranslatef(0, 0, 0.45f);
    normalContext.transformInfo.setTranslationEnd(0.45f * direction);
    normalContext.viewID = m_bodyId;
    c_unitNormalSymbol->draw(normalContext);

    // glFunc->glTranslatef(0, 0, 0.45f);
    normalContext.transformInfo.setTranslationEnd(0.9f * direction);
    normalContext.viewID = m_headId;
    c_unitNormalHeadSymbol->draw(normalContext);
}

void ccPlanarEntityInterface::clearNormalVector(CC_DRAW_CONTEXT& context) {
    context.removeEntityType = ENTITY_TYPE::ECV_MESH;
    if (c_unitNormalSymbol) {
        context.removeViewID = m_bodyId;
        ecvDisplayTools::RemoveEntities(context);
    }
    if (c_unitNormalHeadSymbol) {
        context.removeViewID = m_headId;
        ecvDisplayTools::RemoveEntities(context);
    }
}

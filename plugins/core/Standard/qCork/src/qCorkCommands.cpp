// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qCorkCommands.h"

#include <ecvHObjectCaster.h>

#include <QObject>
#include <algorithm>

#include "ccCorkDlg.h"
#include "qCorkInternal.h"

static const char COMMAND_CORK[] = "CORK";
static const char COMMAND_CORK_OPERATION[] = "OPERATION";
static const char COMMAND_CORK_SWAP[] = "SWAP";

CommandCork::CommandCork()
    : ccCommandLineInterface::Command("Cork Boolean", COMMAND_CORK) {}

bool CommandCork::process(ccCommandLineInterface& cmd) {
    cmd.print("[CORK]");

    if (cmd.meshes().size() < 2) {
        return cmd.error(QObject::tr("At least two meshes are required (use "
                                     "\"-O\" before \"-%1\")")
                                 .arg(COMMAND_CORK));
    }

    ccCorkDlg::CSG_OPERATION operation = ccCorkDlg::UNION;
    bool swap = false;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_CORK_OPERATION)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty()) {
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CORK_OPERATION));
            }
            const QString v = cmd.arguments().takeFirst().toUpper();
            if (v == "UNION") {
                operation = ccCorkDlg::UNION;
            } else if (v == "INTERSECT") {
                operation = ccCorkDlg::INTERSECT;
            } else if (v == "DIFF") {
                operation = ccCorkDlg::DIFF;
            } else if (v == "SYM_DIFF") {
                operation = ccCorkDlg::SYM_DIFF;
            } else {
                return cmd.error(QObject::tr("Invalid \"-%1\": use UNION, "
                                             "INTERSECT, DIFF, or "
                                             "SYM_DIFF")
                                         .arg(COMMAND_CORK_OPERATION));
            }
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CORK_SWAP)) {
            cmd.arguments().pop_front();
            swap = true;
        } else {
            break;
        }
    }

    ccMesh* meshA = ccHObjectCaster::ToMesh(cmd.meshes()[0].mesh);
    ccMesh* meshB = ccHObjectCaster::ToMesh(cmd.meshes()[1].mesh);
    if (!meshA || !meshB) {
        return cmd.error(QObject::tr("Invalid mesh entity"));
    }
    if (swap) {
        std::swap(meshA, meshB);
    }

    CorkMesh corkA;
    CorkMesh corkB;
    if (!ToCorkMesh(meshA, corkA, nullptr)) {
        return cmd.error(QObject::tr("Failed to convert mesh A to Cork mesh"));
    }
    if (!ToCorkMesh(meshB, corkB, nullptr)) {
        return cmd.error(QObject::tr("Failed to convert mesh B to Cork mesh"));
    }

    if (!qCorkPerformBooleanOp(operation, corkA, corkB, meshA->getName(),
                               meshB->getName(), nullptr)) {
        return cmd.error(QObject::tr("Cork boolean operation failed"));
    }

    ccMesh* result = FromCorkMesh(corkA, nullptr);
    if (!result) {
        return cmd.error(QObject::tr("Cork produced an empty result"));
    }

    QString opName;
    switch (operation) {
        case ccCorkDlg::UNION:
            opName = "union";
            break;
        case ccCorkDlg::INTERSECT:
            opName = "isect";
            break;
        case ccCorkDlg::DIFF:
            opName = "diff";
            break;
        case ccCorkDlg::SYM_DIFF:
            opName = "sym_diff";
            break;
        default:
            opName = "op";
            break;
    }

    result->setName(QString("(%1).%2.(%3)")
                            .arg(meshA->getName())
                            .arg(opName)
                            .arg(meshB->getName()));

    bool hasNormals = false;
    if (meshA->hasTriNormals())
        hasNormals = result->computePerTriangleNormals();
    else if (meshA->hasNormals())
        hasNormals = result->computePerVertexNormals();
    meshA->showNormals(hasNormals && meshA->normalsShown());

    const QString baseName =
            cmd.meshes()[0].basename + QString("_CORK_%1").arg(opName);
    CLMeshDesc meshDesc(result, baseName, cmd.meshes()[0].path);
    cmd.meshes().push_back(meshDesc);

    if (cmd.autoSaveMode()) {
        QString errorStr = cmd.exportEntity(cmd.meshes().back());
        if (!errorStr.isEmpty()) {
            return cmd.error(errorStr);
        }
    }

    return true;
}

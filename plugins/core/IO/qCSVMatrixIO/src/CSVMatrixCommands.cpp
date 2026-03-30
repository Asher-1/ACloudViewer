// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CSVMatrixCommands.h"

#include <QObject>

static const char COMMAND_CSV_MATRIX[] = "CSV_MATRIX";
static const char COMMAND_CM_SEPARATOR[] = "SEPARATOR";
static const char COMMAND_CM_SKIP_HEADER[] = "SKIP_HEADER";
static const char COMMAND_CM_INVERT_ROWS[] = "INVERT_ROWS";

CommandCSVMatrix::CommandCSVMatrix()
        : ccCommandLineInterface::Command("CSV Matrix", COMMAND_CSV_MATRIX) {}

bool CommandCSVMatrix::process(ccCommandLineInterface& cmd) {
    cmd.print("[CSV_MATRIX]");

    QChar separator = ',';
    bool skipHeader = false;
    bool invertRows = false;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_CM_SEPARATOR)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CM_SEPARATOR));
            QString sep = cmd.arguments().takeFirst();
            if (!sep.isEmpty()) separator = sep.at(0);
            cmd.print(QObject::tr("[CSV_MATRIX] Separator: '%1'")
                              .arg(separator));
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_CM_SKIP_HEADER)) {
            cmd.arguments().pop_front();
            skipHeader = true;
            cmd.print("[CSV_MATRIX] Skip header enabled");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_CM_INVERT_ROWS)) {
            cmd.arguments().pop_front();
            invertRows = true;
            cmd.print("[CSV_MATRIX] Invert rows enabled");
        } else {
            break;
        }
    }

    return true;
}

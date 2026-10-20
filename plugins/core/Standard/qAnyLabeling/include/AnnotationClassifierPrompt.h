// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QStringList>

inline QString annotationClassifierPrompt(const QStringList& labels,
                                          bool singleLabel) {
    QStringList categories, flags;
    for (const auto& label : labels) {
        categories.append('"' + label + '"');
        flags.append('"' + label + QStringLiteral("\": false"));
    }
    const QString instruction =
            singleLabel
                    ? QStringLiteral(
                              "Set exactly ONE category to 'true' that best "
                              "matches the image, keep all others as 'false'.")
                    : QStringLiteral(
                              "Set ALL applicable categories to 'true', keep "
                              "non-applicable ones as 'false'.");
    return QStringLiteral(
                   "@image\nYou are an expert image classifier. Your task is "
                   "to perform %1 classification.\n\n"
                   "Task Definition: Analyze the given image and classify it "
                   "based on the provided categories.\n\n"
                   "Available Categories: [%2]\n\n"
                   "Instructions:\n"
                   "1. Carefully examine the image and identify the main "
                   "subject and their activity\n"
                   "2. Be precise - only select categories that clearly "
                   "match what you observe\n\n"
                   "Return your result in strict JSON format:\n{%3}\n\n%4")
            .arg(singleLabel ? QStringLiteral("multi-class")
                             : QStringLiteral("multi-label"),
                 categories.join(QStringLiteral(", ")),
                 flags.join(QStringLiteral(", ")), instruction);
}

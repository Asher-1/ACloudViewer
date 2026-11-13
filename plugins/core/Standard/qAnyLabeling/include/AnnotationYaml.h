// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>
#include <QVector>

class QStringList;

/** Minimal YAML subset shared by the pose-config and label-flags template
 *  parsers. Both consume files the source application reads with
 *  yaml.safe_load, so this parser reproduces the constructs those files can
 *  legally use: block mappings/sequences with any consistent indentation
 *  (sequence items may sit at their parent key's indentation), flow
 *  mappings/sequences (single- or multi-line), single/double quoted scalars,
 *  plain scalars, comments, anchors and aliases within one document,
 *  document start/end markers, and YAML 1.1 boolean resolution. Tabs in
 *  indentation, merge keys, block scalars, tags, and multiple documents stay
 *  rejected with a line-numbered error. */
struct AnnotationYamlNode {
    enum Type { Null, Scalar, Sequence, Mapping };
    Type type = Null;
    QString scalar;
    QVector<AnnotationYamlNode> sequence;
    // Ordered mapping entries; duplicate keys keep the first position with
    // the last value, matching dict behavior under yaml.safe_load.
    QVector<QPair<QString, AnnotationYamlNode>> mapping;
    int line = 0;

    bool isBool() const;
    bool boolValue() const;
};

struct AnnotationYamlResult {
    AnnotationYamlNode root;
    QString error;
};

AnnotationYamlResult parseAnnotationYaml(const QString& text);

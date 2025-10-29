// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FooFilter.h"

#include <QString>

FooFilter::FooFilter()
    : FileIOFilter({"Foo Filter",
                    DEFAULT_PRIORITY,  // priority
                    QStringList{"foo", "txt"}, "foo",
                    QStringList{"Foo file (*.foo)", "Text file (*.txt)"},
                    QStringList(), Import}) {}

CC_FILE_ERROR FooFilter::loadFile(const QString &fileName,
                                  ccHObject &container,
                                  LoadParameters &parameters) {
    Q_UNUSED(container);
    Q_UNUSED(parameters);

    QFile file(fileName);

    if (!file.open(QIODevice::ReadOnly)) {
        return CC_FERR_READING;
    }

    QTextStream stream(&file);

    // ...do some stuff with the contents of the file

    // In this example, we treat the file as text and count the occurenecs of
    // the string 'foo'
    QString line;
    int count = 0;

    while (stream.readLineInto(&line)) {
        if (line.contains(QStringLiteral("foo"), Qt::CaseInsensitive)) {
            ++count;
        }
    }

    CVLog::Print(
            QStringLiteral("[foo] The file %1 has %2 lines containing 'foo'")
                    .arg(file.fileName(), QString::number(count)));

    return CC_FERR_NO_ERROR;
}

bool FooFilter::canSave(CV_CLASS_ENUM type,
                        bool &multiple,
                        bool &exclusive) const {
    Q_UNUSED(type);
    Q_UNUSED(multiple);
    Q_UNUSED(exclusive);

    // ... can we save this?
    return false;
}
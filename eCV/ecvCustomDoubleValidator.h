// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_CUSTOM_DOUBLE_VALIDATOR_HEADER
#define ECV_CUSTOM_DOUBLE_VALIDATOR_HEADER

// Qt
#include <QString>
#include <QValidator>

//! Validator class (accepts only double numbers and replaces the comma by a
//! point automatically)
class ccCustomDoubleValidator : public QValidator {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccCustomDoubleValidator(QObject* parent = 0)
        : QValidator(parent) {}

    // reimplemented from QValidator
    State validate(QString& input, int& pos) const {
        for (int i = 0; i < input.size(); ++i) {
            QChar c = input[i];
            if (c == ',') {
                input[i] = '.';
                continue;
            } else if (c == '.' || c == '-' || c.isDigit()) {
                continue;
            } else {
                return Invalid;
            }
        }
        return Acceptable;
    }
};

#endif  // ECV_CUSTOM_DOUBLE_VALIDATOR_HEADER

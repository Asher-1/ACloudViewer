// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Python View — a lightweight script editor + matplotlib output panel.
// Modeled after ParaView's PythonView (vtkPythonView).

#pragma once

#include <QWidget>
#include <functional>

class ccHObject;
class ecvPythonCodeEditor;
class QComboBox;
class QCompleter;
class QLabel;
class QMenu;
class QPlainTextEdit;
class QPushButton;
class QSplitter;
class QTableWidget;
class QToolButton;
class ecvPythonSyntaxHighlighter;

class ecvPythonView : public QWidget {
    Q_OBJECT

public:
    explicit ecvPythonView(QWidget* parent = nullptr);
    ~ecvPythonView() override;

    QString title() const { return m_title; }

protected:
    bool eventFilter(QObject* obj, QEvent* event) override;

public slots:
    void setEntity(ccHObject* entity);

private slots:
    void onRunScript();
    void onClear();
    void onLoadScript();
    void onSaveScript();
    void onExportEntityAndRun();
    void onEntitySelectionChanged(ccHObject* entity);
    void showSnippetMenu();
    void insertSnippet(const QString& code);

private:
    QString exportEntityToTempCsv();

    void setupCompleter();
    void insertCompletion(const QString& completion);
    void inspectVariables();
    void highlightErrorLine(const QString& stderrText);
    void clearErrorHighlights();

    using EntityListProvider = std::function<QList<ccHObject*>()>;

public:
    void setEntityListProvider(EntityListProvider provider);

private:
    void refreshSourceCombo();
    void onSourceComboChanged(int index);

    QComboBox* m_sourceCombo = nullptr;
    EntityListProvider m_entityListProvider;
    ecvPythonCodeEditor* m_scriptEditor = nullptr;
    QPlainTextEdit* m_outputPanel = nullptr;
    QLabel* m_imageLabel = nullptr;
    QLabel* m_statusLabel = nullptr;
    QTableWidget* m_variableTable = nullptr;
    QToolButton* m_snippetBtn = nullptr;
    ecvPythonSyntaxHighlighter* m_highlighter = nullptr;
    QCompleter* m_completer = nullptr;
    ccHObject* m_entity = nullptr;
    QString m_lastExportPath;
    QString m_lastImagePath;
    QString m_pythonBin;
    bool m_pythonChecked = false;

    QString m_viewTypeKey;
    QString m_title;

    QString findPythonBinary();
    void checkPythonEnvironment();
};

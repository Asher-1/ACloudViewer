// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPythonView.h"

#include "ecvPythonCodeEditor.h"
#include "ecvPythonSyntaxHighlighter.h"

#include <ecvGenericPointCloud.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvViewManager.h>

#include <QAbstractItemView>
#include <QCompleter>
#include <QDir>
#include <QKeyEvent>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QScrollBar>
#include <QSplitter>
#include <QStringListModel>
#include <QTemporaryDir>
#include <QTextStream>
#include <QVBoxLayout>

ecvPythonView::ecvPythonView(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    auto btnSS = QStringLiteral(
            "QPushButton { background: #3a5f8f; color: #ddd; border: 1px "
            "solid #555; border-radius: 3px; padding: 2px 8px; }"
            "QPushButton:hover { background: #4a7fbf; }");

    auto* toolbar = new QWidget(this);
    auto* tbLayout = new QHBoxLayout(toolbar);
    tbLayout->setContentsMargins(4, 2, 4, 2);
    tbLayout->setSpacing(4);
    toolbar->setStyleSheet("QWidget { background: #333; }");

    auto* titleLabel = new QLabel(tr("<b>Python View</b>"), toolbar);
    titleLabel->setStyleSheet("QLabel { color: #ccc; }");
    tbLayout->addWidget(titleLabel);

    auto* runBtn = new QPushButton(tr("Run"), toolbar);
    runBtn->setStyleSheet(btnSS);
    runBtn->setToolTip(tr("Execute Python script (Ctrl+Enter)"));
    runBtn->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Return));
    tbLayout->addWidget(runBtn);

    auto* exportRunBtn = new QPushButton(tr("Export+Run"), toolbar);
    exportRunBtn->setStyleSheet(btnSS);
    exportRunBtn->setToolTip(
            tr("Export selected entity to CSV, set DATA_FILE env var, then "
               "run script"));
    tbLayout->addWidget(exportRunBtn);

    auto* clearBtn = new QPushButton(tr("Clear"), toolbar);
    clearBtn->setStyleSheet(btnSS);
    tbLayout->addWidget(clearBtn);

    auto* loadBtn = new QPushButton(tr("Load"), toolbar);
    loadBtn->setStyleSheet(btnSS);
    tbLayout->addWidget(loadBtn);

    auto* saveBtn = new QPushButton(tr("Save"), toolbar);
    saveBtn->setStyleSheet(btnSS);
    tbLayout->addWidget(saveBtn);

    m_statusLabel = new QLabel(toolbar);
    m_statusLabel->setStyleSheet("QLabel { color: #999; font-size: 11px; }");
    tbLayout->addWidget(m_statusLabel);
    tbLayout->addStretch(1);

    layout->addWidget(toolbar);

    auto* splitter = new QSplitter(Qt::Vertical, this);

    m_scriptEditor = new ecvPythonCodeEditor(splitter);
    m_scriptEditor->setPlaceholderText(
            tr("# Python View — with entity data access\n"
               "# Click 'Export+Run' to auto-export the selected entity\n"
               "# as CSV and set DATA_FILE environment variable.\n"
               "#\n"
               "# Example:\n"
               "#   import os, numpy as np\n"
               "#   data = np.loadtxt(os.environ['DATA_FILE'],\n"
               "#                     delimiter=',', skiprows=1)\n"
               "#   print(f'Points: {len(data)}')\n"
               "#   print(f'X range: [{data[:,0].min():.3f}, "
               "{data[:,0].max():.3f}]')"));
    m_scriptEditor->setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; "
            "font-family: 'Consolas', 'Monaco', 'Courier New', monospace; "
            "font-size: 12px; border: none; }");
    m_scriptEditor->setTabStopDistance(32);
    splitter->addWidget(m_scriptEditor);

    m_outputPanel = new QPlainTextEdit(splitter);
    m_outputPanel->setReadOnly(true);
    m_outputPanel->setStyleSheet(
            "QPlainTextEdit { background: #1a1a1a; color: #b5cea8; "
            "font-family: 'Consolas', 'Monaco', 'Courier New', monospace; "
            "font-size: 11px; border: none; }");
    m_outputPanel->setPlaceholderText(tr("Output will appear here..."));
    splitter->addWidget(m_outputPanel);

    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 1);
    layout->addWidget(splitter, 1);

    connect(runBtn, &QPushButton::clicked, this, &ecvPythonView::onRunScript);
    connect(exportRunBtn, &QPushButton::clicked, this,
            &ecvPythonView::onExportEntityAndRun);
    connect(clearBtn, &QPushButton::clicked, this, &ecvPythonView::onClear);
    connect(loadBtn, &QPushButton::clicked, this, &ecvPythonView::onLoadScript);
    connect(saveBtn, &QPushButton::clicked, this, &ecvPythonView::onSaveScript);

    m_highlighter =
            new ecvPythonSyntaxHighlighter(m_scriptEditor->document());

    setupCompleter();

    connect(&ecvViewManager::instance(),
            &ecvViewManager::entitySelectionChanged, this,
            &ecvPythonView::onEntitySelectionChanged);
}

ecvPythonView::~ecvPythonView() {
    if (!m_lastExportPath.isEmpty()) {
        QFile::remove(m_lastExportPath);
    }
}

void ecvPythonView::setEntity(ccHObject* entity) {
    m_entity = entity;
    if (entity) {
        m_statusLabel->setText(
                tr("Entity: %1").arg(entity->getName()));
    } else {
        m_statusLabel->setText(tr("No entity"));
    }
}

void ecvPythonView::onEntitySelectionChanged(ccHObject* entity) {
    setEntity(entity);
}

QString ecvPythonView::exportEntityToTempCsv() {
    if (!m_entity) return {};

    auto* cloud = ccHObjectCaster::ToGenericPointCloud(m_entity);
    if (!cloud) return {};

    auto* pc = ccHObjectCaster::ToPointCloud(m_entity);

    QString tempDir = QDir::tempPath();
    QString csvPath = tempDir + QStringLiteral("/acv_entity_%1.csv")
                              .arg(reinterpret_cast<quintptr>(m_entity), 0, 16);

    QFile file(csvPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return {};

    QTextStream out(&file);

    QStringList headers;
    headers << "X" << "Y" << "Z";

    int sfCount = pc ? pc->getNumberOfScalarFields() : 0;
    for (int i = 0; i < sfCount; ++i) {
        auto* sf = pc->getScalarField(i);
        if (sf) headers << QString::fromStdString(sf->getName());
    }

    bool hasNormals = cloud->hasNormals();
    if (hasNormals) {
        headers << "Nx" << "Ny" << "Nz";
    }
    bool hasColors = cloud->hasColors();
    if (hasColors) {
        headers << "R" << "G" << "B";
    }

    out << headers.join(",") << "\n";

    unsigned count = cloud->size();
    for (unsigned i = 0; i < count; ++i) {
        const CCVector3* pt = cloud->getPoint(i);
        out << pt->x << "," << pt->y << "," << pt->z;

        for (int s = 0; s < sfCount; ++s) {
            auto* sf = pc->getScalarField(s);
            out << "," << (sf ? sf->getValue(i) : 0.0);
        }

        if (hasNormals) {
            const CCVector3& n = cloud->getPointNormal(i);
            out << "," << n.x << "," << n.y << "," << n.z;
        }
        if (hasColors) {
            const ecvColor::Rgb& c = cloud->getPointColor(i);
            out << "," << c.r << "," << c.g << "," << c.b;
        }
        out << "\n";
    }

    if (!m_lastExportPath.isEmpty() && m_lastExportPath != csvPath) {
        QFile::remove(m_lastExportPath);
    }
    m_lastExportPath = csvPath;
    return csvPath;
}

void ecvPythonView::onExportEntityAndRun() {
    QString csvPath = exportEntityToTempCsv();
    if (csvPath.isEmpty()) {
        m_outputPanel->appendPlainText(
                tr("Error: No entity selected or entity has no point data."));
        return;
    }

    m_outputPanel->clear();
    m_outputPanel->appendPlainText(
            tr("Exported entity to: %1").arg(csvPath));
    m_outputPanel->appendPlainText(
            tr("DATA_FILE env variable set. Running script...\n"));

    QString script = m_scriptEditor->toPlainText().trimmed();
    if (script.isEmpty()) {
        script = QStringLiteral(
                "import os, sys\n"
                "path = os.environ.get('DATA_FILE', '')\n"
                "print(f'DATA_FILE = {path}')\n"
                "try:\n"
                "    import numpy as np\n"
                "    data = np.loadtxt(path, delimiter=',', skiprows=1)\n"
                "    print(f'Shape: {data.shape}')\n"
                "    print(f'X: [{data[:,0].min():.4f}, {data[:,0].max():.4f}]')\n"
                "    print(f'Y: [{data[:,1].min():.4f}, {data[:,1].max():.4f}]')\n"
                "    print(f'Z: [{data[:,2].min():.4f}, {data[:,2].max():.4f}]')\n"
                "except ImportError:\n"
                "    import csv\n"
                "    with open(path) as f:\n"
                "        reader = csv.reader(f)\n"
                "        header = next(reader)\n"
                "        print(f'Columns: {header}')\n"
                "        rows = sum(1 for _ in reader)\n"
                "        print(f'Rows: {rows}')\n");
        m_scriptEditor->setPlainText(script);
    }

    m_statusLabel->setText(tr("Running..."));

    QProcess proc;
    proc.setProgram("python3");
    proc.setArguments({"-c", script});

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("DATA_FILE", csvPath);
    proc.setProcessEnvironment(env);
    proc.start();

    if (!proc.waitForFinished(30000)) {
        m_outputPanel->appendPlainText(
                tr("Error: Script timed out (30s limit)"));
        proc.kill();
        m_statusLabel->setText(tr("Timeout"));
        return;
    }

    QString stdOut = QString::fromUtf8(proc.readAllStandardOutput());
    QString stdErr = QString::fromUtf8(proc.readAllStandardError());

    if (!stdOut.isEmpty()) {
        m_outputPanel->appendPlainText(stdOut);
    }
    if (!stdErr.isEmpty()) {
        m_outputPanel->appendPlainText(tr("--- stderr ---"));
        m_outputPanel->appendPlainText(stdErr);
    }

    int exitCode = proc.exitCode();
    m_statusLabel->setText(tr("Exit code: %1").arg(exitCode));
}

void ecvPythonView::onRunScript() {
    QString script = m_scriptEditor->toPlainText().trimmed();
    if (script.isEmpty()) return;

    m_outputPanel->clear();
    m_statusLabel->setText(tr("Running..."));

    QProcess proc;
    proc.setProgram("python3");
    proc.setArguments({"-c", script});

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    if (!m_lastExportPath.isEmpty()) {
        env.insert("DATA_FILE", m_lastExportPath);
    }
    proc.setProcessEnvironment(env);
    proc.start();

    if (!proc.waitForFinished(30000)) {
        m_outputPanel->appendPlainText(
                tr("Error: Script timed out (30s limit)"));
        proc.kill();
        m_statusLabel->setText(tr("Timeout"));
        return;
    }

    QString stdOut = QString::fromUtf8(proc.readAllStandardOutput());
    QString stdErr = QString::fromUtf8(proc.readAllStandardError());

    if (!stdOut.isEmpty()) {
        m_outputPanel->appendPlainText(stdOut);
    }
    if (!stdErr.isEmpty()) {
        m_outputPanel->appendPlainText(tr("--- stderr ---"));
        m_outputPanel->appendPlainText(stdErr);
    }

    int exitCode = proc.exitCode();
    m_statusLabel->setText(tr("Exit code: %1").arg(exitCode));
}

void ecvPythonView::onClear() {
    m_outputPanel->clear();
    m_statusLabel->clear();
}

void ecvPythonView::onLoadScript() {
    QString path = QFileDialog::getOpenFileName(
            this, tr("Load Python Script"), QString(),
            tr("Python (*.py);;All files (*)"));
    if (path.isEmpty()) return;

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return;

    m_scriptEditor->setPlainText(QString::fromUtf8(file.readAll()));
    m_statusLabel->setText(tr("Loaded: %1").arg(path));
}

void ecvPythonView::onSaveScript() {
    QString path = QFileDialog::getSaveFileName(
            this, tr("Save Python Script"), QString(),
            tr("Python (*.py)"));
    if (path.isEmpty()) return;

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;

    QTextStream out(&file);
    out << m_scriptEditor->toPlainText();
    m_statusLabel->setText(tr("Saved: %1").arg(path));
}

bool ecvPythonView::eventFilter(QObject* obj, QEvent* event) {
    if (obj == m_scriptEditor && event->type() == QEvent::KeyPress) {
        auto* ke = static_cast<QKeyEvent*>(event);

        if (m_completer && m_completer->popup()->isVisible()) {
            switch (ke->key()) {
                case Qt::Key_Enter:
                case Qt::Key_Return:
                case Qt::Key_Escape:
                case Qt::Key_Tab:
                case Qt::Key_Backtab:
                    return false;
                default:
                    break;
            }
        }

        bool isShortcut =
                (ke->modifiers() & Qt::ControlModifier) && ke->key() == Qt::Key_Space;

        if (!isShortcut) {
            QWidget::eventFilter(obj, event);
        }

        if (m_completer) {
            static const QString eow =
                    QStringLiteral("~!@#$%^&*()+{}|:\"<>?,/;'[]\\-=");
            bool hasModifier = (ke->modifiers() != Qt::NoModifier) &&
                               !isShortcut;
            QTextCursor tc = m_scriptEditor->textCursor();
            tc.select(QTextCursor::WordUnderCursor);
            QString prefix = tc.selectedText();

            if (!isShortcut &&
                (hasModifier || ke->text().isEmpty() ||
                 prefix.length() < 2 ||
                 eow.contains(ke->text().right(1)))) {
                m_completer->popup()->hide();
            } else {
                if (prefix != m_completer->completionPrefix()) {
                    m_completer->setCompletionPrefix(prefix);
                    m_completer->popup()->setCurrentIndex(
                            m_completer->completionModel()->index(0, 0));
                }
                QRect cr = m_scriptEditor->cursorRect();
                cr.setWidth(m_completer->popup()->sizeHintForColumn(0) +
                            m_completer->popup()
                                    ->verticalScrollBar()
                                    ->sizeHint()
                                    .width());
                m_completer->complete(cr);
            }
        }

        if (isShortcut) return true;
        return true;
    }
    return QWidget::eventFilter(obj, event);
}

void ecvPythonView::setupCompleter() {
    QStringList words = {
            // Python keywords
            "False",    "None",      "True",      "and",       "as",
            "assert",   "async",     "await",     "break",     "class",
            "continue", "def",       "del",       "elif",      "else",
            "except",   "finally",   "for",       "from",      "global",
            "if",       "import",    "in",        "is",        "lambda",
            "nonlocal", "not",       "or",        "pass",      "raise",
            "return",   "try",       "while",     "with",      "yield",
            // Built-in functions
            "abs",      "all",       "any",       "bin",       "bool",
            "bytes",    "callable",  "chr",       "dict",      "dir",
            "enumerate","eval",      "exec",      "filter",    "float",
            "format",   "getattr",   "globals",   "hasattr",   "hash",
            "help",     "hex",       "id",        "input",     "int",
            "isinstance","issubclass","iter",     "len",       "list",
            "locals",   "map",       "max",       "min",       "next",
            "object",   "oct",       "open",      "ord",       "pow",
            "print",    "property",  "range",     "repr",      "reversed",
            "round",    "set",       "setattr",   "slice",     "sorted",
            "staticmethod","str",    "sum",       "super",     "tuple",
            "type",     "vars",      "zip",
            // Common modules
            "os",       "sys",       "numpy",     "np",        "matplotlib",
            "plt",      "pandas",    "pd",        "math",      "json",
            "csv",      "pathlib",   "Path",      "collections",
            // numpy common
            "np.array", "np.zeros",  "np.ones",   "np.linspace",
            "np.arange","np.loadtxt","np.savetxt","np.mean",
            "np.std",   "np.max",    "np.min",    "np.sum",
            // matplotlib common
            "plt.plot", "plt.scatter","plt.hist", "plt.bar",
            "plt.figure","plt.savefig","plt.show","plt.xlabel",
            "plt.ylabel","plt.title","plt.legend","plt.grid",
            // os/env
            "os.environ","os.path",  "os.getcwd", "os.listdir",
    };
    words.sort(Qt::CaseInsensitive);
    words.removeDuplicates();

    m_completer = new QCompleter(words, this);
    m_completer->setWidget(m_scriptEditor);
    m_completer->setCompletionMode(QCompleter::PopupCompletion);
    m_completer->setCaseSensitivity(Qt::CaseInsensitive);
    m_completer->setModelSorting(
            QCompleter::CaseInsensitivelySortedModel);

    connect(m_completer,
            QOverload<const QString&>::of(&QCompleter::activated), this,
            &ecvPythonView::insertCompletion);

    m_scriptEditor->installEventFilter(this);
}

void ecvPythonView::insertCompletion(const QString& completion) {
    QTextCursor tc = m_scriptEditor->textCursor();
    int extra = completion.length() - m_completer->completionPrefix().length();
    tc.movePosition(QTextCursor::Left);
    tc.movePosition(QTextCursor::EndOfWord);
    tc.insertText(completion.right(extra));
    m_scriptEditor->setTextCursor(tc);
}

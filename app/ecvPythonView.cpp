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
#include <QComboBox>
#include <QCompleter>
#include <QDir>
#include <QKeyEvent>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QSplitter>
#include <QStringListModel>
#include <QTemporaryDir>
#include <QTextStream>
#include <QVBoxLayout>

ecvPythonView::ecvPythonView(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    // === Row 1: Showing combo (ParaView-style) ===
    auto* decoratorBar = new QWidget(this);
    auto* decLayout = new QHBoxLayout(decoratorBar);
    decLayout->setContentsMargins(2, 1, 2, 1);
    decLayout->setSpacing(2);

    auto* showingLabel = new QLabel(QStringLiteral("<b>Showing</b>"), decoratorBar);
    decLayout->addWidget(showingLabel);

    m_sourceCombo = new QComboBox(decoratorBar);
    m_sourceCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    m_sourceCombo->addItem(tr("None"));
    decLayout->addWidget(m_sourceCombo);
    decLayout->addStretch(1);
    layout->addWidget(decoratorBar);

    connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ecvPythonView::onSourceComboChanged);

    auto btnSS = QStringLiteral(
            "QPushButton { background: #3a5f8f; color: #ddd; border: 1px "
            "solid #555; border-radius: 3px; padding: 2px 8px; }"
            "QPushButton:hover { background: #4a7fbf; }");

    // === Row 2: Script toolbar ===
    auto* toolbar = new QWidget(this);
    auto* tbLayout = new QHBoxLayout(toolbar);
    tbLayout->setContentsMargins(4, 2, 4, 2);
    tbLayout->setSpacing(4);
    toolbar->setStyleSheet("QWidget { background: #333; }");

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
            tr("# Python View — matplotlib rendering + entity data\n"
               "# Click 'Export+Run' to export entity to CSV & run.\n"
               "# Use plt.savefig(os.environ['PLOT_FILE']) to render inline.\n"
               "#\n"
               "# Example:\n"
               "#   import os, numpy as np, matplotlib\n"
               "#   matplotlib.use('Agg')\n"
               "#   import matplotlib.pyplot as plt\n"
               "#   data = np.loadtxt(os.environ['DATA_FILE'],\n"
               "#                     delimiter=',', skiprows=1)\n"
               "#   plt.scatter(data[:,0], data[:,1], s=1, c=data[:,2])\n"
               "#   plt.colorbar(); plt.title('XY colored by Z')\n"
               "#   plt.savefig(os.environ['PLOT_FILE'], dpi=150)"));
    m_scriptEditor->setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; "
            "font-family: 'Consolas', 'Monaco', 'Courier New', monospace; "
            "font-size: 12px; border: none; }");
    m_scriptEditor->setTabStopDistance(32);
    splitter->addWidget(m_scriptEditor);

    auto* outputSplitter = new QSplitter(Qt::Horizontal, splitter);

    m_outputPanel = new QPlainTextEdit(outputSplitter);
    m_outputPanel->setReadOnly(true);
    m_outputPanel->setStyleSheet(
            "QPlainTextEdit { background: #1a1a1a; color: #b5cea8; "
            "font-family: 'Consolas', 'Monaco', 'Courier New', monospace; "
            "font-size: 11px; border: none; }");
    m_outputPanel->setPlaceholderText(tr("Output will appear here..."));
    outputSplitter->addWidget(m_outputPanel);

    m_imageLabel = new QLabel(outputSplitter);
    m_imageLabel->setAlignment(Qt::AlignCenter);
    m_imageLabel->setStyleSheet("QLabel { background: white; }");
    m_imageLabel->setText(tr("Plot output\n(use plt.savefig(os.environ['PLOT_FILE']))"));
    m_imageLabel->setScaledContents(false);
    m_imageLabel->setMinimumSize(100, 100);
    outputSplitter->addWidget(m_imageLabel);

    outputSplitter->setStretchFactor(0, 1);
    outputSplitter->setStretchFactor(1, 2);
    splitter->addWidget(outputSplitter);

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
    if (!m_lastImagePath.isEmpty()) {
        QFile::remove(m_lastImagePath);
    }
}

void ecvPythonView::setEntityListProvider(EntityListProvider provider) {
    m_entityListProvider = std::move(provider);
    refreshSourceCombo();
}

void ecvPythonView::refreshSourceCombo() {
    if (!m_sourceCombo) return;
    QSignalBlocker blocker(m_sourceCombo);
    m_sourceCombo->clear();
    m_sourceCombo->addItem(tr("None"));

    if (m_entityListProvider) {
        auto entities = m_entityListProvider();
        for (auto* e : entities) {
            if (e) {
                m_sourceCombo->addItem(e->getName(),
                                       QVariant::fromValue(
                                               reinterpret_cast<quintptr>(e)));
            }
        }
    }

    if (m_entity) {
        for (int i = 1; i < m_sourceCombo->count(); ++i) {
            auto ptr = m_sourceCombo->itemData(i).value<quintptr>();
            if (reinterpret_cast<ccHObject*>(ptr) == m_entity) {
                m_sourceCombo->setCurrentIndex(i);
                return;
            }
        }
    }
}

void ecvPythonView::onSourceComboChanged(int index) {
    if (index <= 0) {
        setEntity(nullptr);
        return;
    }
    auto ptr = m_sourceCombo->itemData(index).value<quintptr>();
    setEntity(reinterpret_cast<ccHObject*>(ptr));
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

    QString plotPath = QDir::tempPath() +
                       QStringLiteral("/acv_plot_%1.png")
                               .arg(reinterpret_cast<quintptr>(this), 0, 16);

    QString script = m_scriptEditor->toPlainText().trimmed();
    if (script.isEmpty()) {
        script = QStringLiteral(
                "import os, numpy as np, matplotlib\n"
                "matplotlib.use('Agg')\n"
                "import matplotlib.pyplot as plt\n"
                "path = os.environ.get('DATA_FILE', '')\n"
                "data = np.loadtxt(path, delimiter=',', skiprows=1)\n"
                "print(f'Shape: {data.shape}')\n"
                "print(f'X: [{data[:,0].min():.4f}, {data[:,0].max():.4f}]')\n"
                "print(f'Y: [{data[:,1].min():.4f}, {data[:,1].max():.4f}]')\n"
                "print(f'Z: [{data[:,2].min():.4f}, {data[:,2].max():.4f}]')\n"
                "fig, ax = plt.subplots(figsize=(8, 6))\n"
                "sc = ax.scatter(data[:,0], data[:,1], s=1, c=data[:,2], cmap='viridis')\n"
                "plt.colorbar(sc); ax.set_xlabel('X'); ax.set_ylabel('Y')\n"
                "ax.set_title(os.path.basename(path))\n"
                "plt.tight_layout()\n"
                "plt.savefig(os.environ['PLOT_FILE'], dpi=150)\n"
                "print(f'Plot saved to {os.environ[\"PLOT_FILE\"]}')\n");
        m_scriptEditor->setPlainText(script);
    }

    m_statusLabel->setText(tr("Running..."));

    QFile::remove(plotPath);

    QProcess proc;
    proc.setProgram("python3");
    proc.setArguments({"-c", script});

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("DATA_FILE", csvPath);
    env.insert("PLOT_FILE", plotPath);
    proc.setProcessEnvironment(env);
    proc.start();

    if (!proc.waitForFinished(60000)) {
        m_outputPanel->appendPlainText(
                tr("Error: Script timed out (60s limit)"));
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

    if (QFile::exists(plotPath)) {
        QPixmap pix(plotPath);
        if (!pix.isNull() && m_imageLabel) {
            m_imageLabel->setPixmap(pix.scaled(m_imageLabel->size(),
                                               Qt::KeepAspectRatio,
                                               Qt::SmoothTransformation));
        }
        if (!m_lastImagePath.isEmpty() && m_lastImagePath != plotPath) {
            QFile::remove(m_lastImagePath);
        }
        m_lastImagePath = plotPath;
    }
}

void ecvPythonView::onRunScript() {
    QString script = m_scriptEditor->toPlainText().trimmed();
    if (script.isEmpty()) return;

    m_outputPanel->clear();
    m_statusLabel->setText(tr("Running..."));

    QString plotPath = QDir::tempPath() +
                       QStringLiteral("/acv_plot_%1.png")
                               .arg(reinterpret_cast<quintptr>(this), 0, 16);
    QFile::remove(plotPath);

    QProcess proc;
    proc.setProgram("python3");
    proc.setArguments({"-c", script});

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    if (!m_lastExportPath.isEmpty()) {
        env.insert("DATA_FILE", m_lastExportPath);
    }
    env.insert("PLOT_FILE", plotPath);
    proc.setProcessEnvironment(env);
    proc.start();

    if (!proc.waitForFinished(60000)) {
        m_outputPanel->appendPlainText(
                tr("Error: Script timed out (60s limit)"));
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

    if (QFile::exists(plotPath)) {
        QPixmap pix(plotPath);
        if (!pix.isNull() && m_imageLabel) {
            m_imageLabel->setPixmap(pix.scaled(m_imageLabel->size(),
                                               Qt::KeepAspectRatio,
                                               Qt::SmoothTransformation));
        }
        if (!m_lastImagePath.isEmpty() && m_lastImagePath != plotPath) {
            QFile::remove(m_lastImagePath);
        }
        m_lastImagePath = plotPath;
    }
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

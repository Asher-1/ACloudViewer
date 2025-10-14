// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PythonActionLauncher.h"
#include "PythonInterpreter.h"
#include "Resources.h"
#include <ui_ActionLauncher.h>

#include <QIcon>
#include <QVBoxLayout>

#undef slots
#include "PythonPluginManager.h"
#include "PythonStdErrOutRedirect.h"

/// Implementation of Model so that, we can handle double clicks on plugin actions
/// better thant if we used QListWidget
class PluginListModel final : public QAbstractListModel
{
    Q_OBJECT

  public:
    explicit PluginListModel(const Runtime::RegisteredPlugin *plugin, PythonActionLauncher *parent)
        : QAbstractListModel(parent), plugin(plugin)
    {
    }

    int rowCount(const QModelIndex &parent) const override
    {
        Q_ASSERT(plugin != nullptr);
        return static_cast<int>(plugin->actions.size());
    };

    QVariant data(const QModelIndex &index, int role) const override
    {
        Q_ASSERT(plugin != nullptr);
        if (role != Qt::DisplayRole || !index.isValid() || index.row() >= plugin->actions.size())
        {
            return {};
        }
        return plugin->actions[index.row()].name;
    };

    void handleDoubleClick(const QModelIndex &index)
    {
        if (!index.isValid())
        {
            return;
        }

        PythonInterpreter *interpreter =
            static_cast<PythonActionLauncher *>(parent())->m_interpreter;
        interpreter->executeFunction(plugin->actions[index.row()].target);
    }

  private:
    const Runtime::RegisteredPlugin *plugin{nullptr};
};

PythonActionLauncher::PythonActionLauncher(const PythonPluginManager *pluginManager,
                                           PythonInterpreter *interpreter,
                                           QWidget *parent)
    : QWidget(parent),
      m_ui(new Ui_ActionLauncher),
      m_pluginManager(pluginManager),
      m_interpreter(interpreter)
{
    setWindowTitle("ActionLauncher");
    m_ui->setupUi(this);
    connect(
        m_interpreter, &PythonInterpreter::executionStarted, this, &PythonActionLauncher::disable);
    connect(
        m_interpreter, &PythonInterpreter::executionFinished, this, &PythonActionLauncher::enable);

    setWindowIcon(QIcon(ACTION_LAUNCHER_ICON_PATH));
}

void PythonActionLauncher::showEvent(QShowEvent *event)
{
    clearToolBox();
    populateToolBox();
    QWidget::showEvent(event);
}

void PythonActionLauncher::disable()
{
    m_ui->toolBox->setDisabled(true);
}
void PythonActionLauncher::enable()
{
    m_ui->toolBox->setDisabled(false);
}

void PythonActionLauncher::clearToolBox()
{
    QToolBox *toolBox = m_ui->toolBox;
    for (int i{toolBox->count() - 1}; i >= 0; --i)
    {
        QWidget *widget = toolBox->widget(i);
        toolBox->removeItem(i);
        delete widget;
    }
}

void PythonActionLauncher::populateToolBox()
{
    QToolBox *toolBox = m_ui->toolBox;
    for (const Runtime::RegisteredPlugin &plugin : m_pluginManager->plugins())
    {
        if (plugin.actions.empty())
        {
            continue;
        }

        auto *view = new QListView(this);
        auto *model = new PluginListModel(&plugin, this);
        connect(view, &QListView::doubleClicked, model, &PluginListModel::handleDoubleClick);
        view->setModel(model);

        toolBox->addItem(view, plugin.name);
    }
}

#include "PythonActionLauncher.moc"

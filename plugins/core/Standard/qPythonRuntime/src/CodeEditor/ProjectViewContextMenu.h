// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QMenu>
#include <QModelIndex>

class QTreeView;
class QFileSystemModel;

class ProjectView;

class ProjectViewContextMenu final : public QMenu
{
    Q_OBJECT

  public:
    explicit ProjectViewContextMenu(ProjectView *view);

  public Q_SLOTS:
    void requested(const QPoint &pos);

  private Q_SLOTS:
    void renameFile() const;
    void deleteElement() const;
    void createFile() const;
    void createFolder() const;

  private:
    ProjectView *m_treeView;
    QModelIndex m_currentIndex;

    QAction m_renameAction;
    QAction m_deleteAction;
    QAction m_createFileAction;
    QAction m_createFolderAction;
};

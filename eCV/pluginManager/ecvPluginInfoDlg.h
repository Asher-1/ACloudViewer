// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>
#include <QList>

class QModelIndex;
class QStandardItem;
class QStandardItemModel;
class QSortFilterProxyModel;

class ccPluginInterface;

namespace Ui {
class ccPluginInfoDlg;
}

class ccPluginInfoDlg : public QDialog {
    Q_OBJECT

public:
    explicit ccPluginInfoDlg(QWidget *parent = nullptr);
    ~ccPluginInfoDlg();

    void setPluginPaths(const QStringList &pluginPaths);
    void setPluginList(const QList<ccPluginInterface *> &pluginList);

private:
    enum { PLUGIN_PTR = Qt::UserRole + 1 };

    const ccPluginInterface *pluginFromItemData(
            const QStandardItem *item) const;

    void selectionChanged(const QModelIndex &current,
                          const QModelIndex &previous);
    void itemChanged(QStandardItem *item);

    void updatePluginInfo(const ccPluginInterface *plugin);

    Ui::ccPluginInfoDlg *m_UI;

    QStandardItemModel *m_ItemModel;
    QSortFilterProxyModel *m_ProxyModel;
};

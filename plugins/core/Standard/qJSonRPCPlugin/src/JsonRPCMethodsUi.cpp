// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Generic GUI drivers: ui.listActions, ui.triggerAction, qt.invokeMethod.
// They reach plugin dialogs through Qt reflection, never by plugin type.

#include <ecvMainAppInterface.h>

#include <QAction>
#include <QApplication>
#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMenu>
#include <QMetaMethod>
#include <QPluginLoader>
#include <QToolBar>
#include <QWidget>

#include "JsonRPCCommon.h"
#include "JsonRPCPlugin.h"

using jsonrpc::D;

// ═══════════════════════════════════════════════════════════════════════════
// ui.triggerAction
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// Plugin objects are created by QPluginLoader WITHOUT a parent, and their
// actions/dialogs hang below them — a qApp-relative walk never reaches
// them. Union three domains: the qApp tree, all top-level widgets
// (parent-less dialogs are still top-level), and every menu's action list
// (menus hold actions by reference, not by parenting).
QList<QObject*> collectAllObjects() {
    QList<QObject*> all;
    QSet<QObject*> seen;
    QList<QObject*> queue{QCoreApplication::instance()};
    seen.insert(QCoreApplication::instance());
    for (QWidget* w : QApplication::topLevelWidgets()) {
        if (w && !seen.contains(w)) {
            seen.insert(w);
            queue.append(w);
        }
    }
    while (!queue.isEmpty()) {
        QObject* cur = queue.takeFirst();
        if (!cur) continue;
        all.append(cur);
        for (QObject* child : cur->children()) {
            if (child && !seen.contains(child)) {
                seen.insert(child);
                queue.append(child);
            }
        }
    }
    for (QObject* obj : all) {
        // Union every widget's action list: plugin actions reach the UI via
        // QMenu::addAction AND QToolBar::addAction, neither of which
        // reparents the action to the widget.
        if (auto* widget = qobject_cast<QWidget*>(obj)) {
            for (QAction* action : widget->actions()) {
                if (action && !seen.contains(action)) {
                    seen.insert(action);
                    all.append(action);
                }
            }
        }
    }
    return all;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// ui.listActions — enumerate every triggerable QAction so an agent can
// discover the full plugin/menu surface without guessing texts.
// ═══════════════════════════════════════════════════════════════════════════
JsonRPCResult JsonRPCPlugin::rpcUiListActions(
        const QMap<QString, QVariant>& params) {
    Q_UNUSED(params);
    if (!m_app) {
        return JsonRPCResult::error(2, "App interface unavailable");
    }
    QJsonArray actions;
    QSet<QObject*> seen;
    const auto objects = collectAllObjects();
    for (QObject* obj : objects) {
        auto* action = qobject_cast<QAction*>(obj);
        if (!action || action->text().isEmpty() || seen.contains(action))
            continue;
        seen.insert(action);
        QJsonObject o;
        o["text"] = action->text();
        if (!action->objectName().isEmpty())
            o["objectName"] = action->objectName();
        o["enabled"] = action->isEnabled();
        o["checkable"] = action->isCheckable();
        if (action->isCheckable()) o["checked"] = action->isChecked();
        if (!action->shortcut().isEmpty())
            o["shortcut"] = action->shortcut().toString();
        if (auto* parentWidget = qobject_cast<QWidget*>(action->parent()))
            o["container"] = parentWidget->objectName().isEmpty()
                                     ? parentWidget->metaObject()->className()
                                     : parentWidget->objectName();
        actions.append(o);
    }
    QJsonObject result;
    result["count"] = static_cast<qint64>(actions.size());
    result["actions"] = actions;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

JsonRPCResult JsonRPCPlugin::rpcUiTriggerAction(
        const QMap<QString, QVariant>& params) {
    const QString text = params["text"].toString();
    if (text.isEmpty()) {
        return JsonRPCResult::error(1, "Missing parameter: text",
                                    D("param", "text"));
    }
    if (!m_app) {
        return JsonRPCResult::error(2, "App interface unavailable");
    }

    QAction* exact = nullptr;
    QList<QAction*> partial;
    const auto objects = collectAllObjects();
    for (QObject* obj : objects) {
        auto* action = qobject_cast<QAction*>(obj);
        if (!action || action->text().isEmpty()) continue;
        if (action->text() == text) {
            exact = action;
            break;
        }
        if (action->text().contains(text, Qt::CaseInsensitive)) {
            partial.append(action);
        }
    }
    QAction* hit = exact;
    if (!hit && partial.size() == 1) {
        hit = partial.first();
    }
    if (!hit) {
        QJsonObject data;
        data["text"] = text;
        data["partial_matches"] = partial.size();
        return JsonRPCResult::error(2,
                                    partial.isEmpty() ? "Action not found"
                                                      : "Ambiguous action text",
                                    data);
    }
    hit->trigger();

    QJsonObject result;
    result["triggered"] = hit->text();
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// qt.invokeMethod
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcQtInvokeMethod(
        const QMap<QString, QVariant>& params) {
    const QString classContains = params.value("class_contains").toString();
    const QString objectContains = params.value("object_contains").toString();
    const QString method = params["method"].toString();
    const QVariantList args = params.value("args").toList();
    if (method.isEmpty()) {
        return JsonRPCResult::error(1, "Missing parameter: method",
                                    D("param", "method"));
    }
    if (args.size() > 5) {
        return JsonRPCResult::error(1, "Too many args (max 5)",
                                    D("args", args.size()));
    }
    if (!m_app) {
        return JsonRPCResult::error(2, "App interface unavailable");
    }

    QObject* hit = nullptr;
    QMetaMethod hitMethod;
    const bool invokeAll = params.value("all", false).toBool();
    QList<QPair<QObject*, QMetaMethod>> hits;
    const QByteArray methodName = method.toLatin1();
    const auto objects = collectAllObjects();
    for (QObject* obj : objects) {
        if (!obj) continue;
        if (!classContains.isEmpty() &&
            !QString::fromLatin1(obj->metaObject()->className())
                     .contains(classContains, Qt::CaseInsensitive)) {
            continue;
        }
        if (!objectContains.isEmpty() &&
            !obj->objectName().contains(objectContains, Qt::CaseInsensitive)) {
            continue;
        }
        const QMetaObject* mo = obj->metaObject();
        for (int i = 0; i < mo->methodCount(); ++i) {
            const QMetaMethod mm = mo->method(i);
            if (mm.name() != methodName) continue;
            if (mm.parameterCount() != args.size()) continue;
            // Driver-side convention: parameterized targets take QVariant
            // parameters (so the JSON types marshal losslessly); the 0-arg
            // form has no type constraint.
            bool typesOk = true;
            for (int p = 0; p < mm.parameterCount(); ++p) {
                if (mm.parameterType(p) != QMetaType::QVariant) {
                    typesOk = false;
                    break;
                }
            }
            if (!typesOk) continue;
            if (invokeAll) {
                hits.append({obj, mm});
                continue;
            }
            if (hit) {
                QJsonObject data;
                data["method"] = method;
                data["first_match_class"] = hit->metaObject()->className();
                return JsonRPCResult::error(2, "Ambiguous target", data);
            }
            hit = obj;
            hitMethod = mm;
        }
    }
    if (!hit && hits.isEmpty()) {
        QJsonObject data;
        data["method"] = method;
        data["class_contains"] = classContains;
        return JsonRPCResult::error(2, "No matching invokable method", data);
    }

    bool ok = false;
    if (invokeAll) {
        // Batch mode (e.g. closing every stacked modal of one class).
        const QVariant a0 = args.value(0);
        const QVariant a1 = args.value(1);
        const QVariant a2 = args.value(2);
        const QVariant a3 = args.value(3);
        const QVariant a4 = args.value(4);
        int invoked = 0;
        for (const auto& h : hits) {
            const QMetaMethod& mm = h.second;
            switch (mm.parameterCount()) {
                case 0:
                    ok = mm.invoke(h.first, Qt::DirectConnection);
                    break;
                case 1:
                    ok = mm.invoke(h.first, Qt::DirectConnection,
                                   Q_ARG(QVariant, a0));
                    break;
                case 2:
                    ok = mm.invoke(h.first, Qt::DirectConnection,
                                   Q_ARG(QVariant, a0), Q_ARG(QVariant, a1));
                    break;
                case 3:
                    ok = mm.invoke(h.first, Qt::DirectConnection,
                                   Q_ARG(QVariant, a0), Q_ARG(QVariant, a1),
                                   Q_ARG(QVariant, a2));
                    break;
                case 4:
                    ok = mm.invoke(h.first, Qt::DirectConnection,
                                   Q_ARG(QVariant, a0), Q_ARG(QVariant, a1),
                                   Q_ARG(QVariant, a2), Q_ARG(QVariant, a3));
                    break;
                case 5:
                    ok = mm.invoke(h.first, Qt::DirectConnection,
                                   Q_ARG(QVariant, a0), Q_ARG(QVariant, a1),
                                   Q_ARG(QVariant, a2), Q_ARG(QVariant, a3),
                                   Q_ARG(QVariant, a4));
                    break;
            }
            if (ok) ++invoked;
        }
        QJsonObject result;
        result["invoked"] = method;
        result["count"] = invoked;
        return JsonRPCResult::success(QJsonDocument(result).toVariant());
    }
    const QVariant a0 = args.value(0);
    const QVariant a1 = args.value(1);
    const QVariant a2 = args.value(2);
    const QVariant a3 = args.value(3);
    const QVariant a4 = args.value(4);
    switch (hitMethod.parameterCount()) {
        case 0:
            ok = hitMethod.invoke(hit, Qt::DirectConnection);
            break;
        case 1:
            ok = hitMethod.invoke(hit, Qt::DirectConnection,
                                  Q_ARG(QVariant, a0));
            break;
        case 2:
            ok = hitMethod.invoke(hit, Qt::DirectConnection,
                                  Q_ARG(QVariant, a0), Q_ARG(QVariant, a1));
            break;
        case 3:
            ok = hitMethod.invoke(hit, Qt::DirectConnection,
                                  Q_ARG(QVariant, a0), Q_ARG(QVariant, a1),
                                  Q_ARG(QVariant, a2));
            break;
        case 4:
            ok = hitMethod.invoke(hit, Qt::DirectConnection,
                                  Q_ARG(QVariant, a0), Q_ARG(QVariant, a1),
                                  Q_ARG(QVariant, a2), Q_ARG(QVariant, a3));
            break;
        case 5:
            ok = hitMethod.invoke(hit, Qt::DirectConnection,
                                  Q_ARG(QVariant, a0), Q_ARG(QVariant, a1),
                                  Q_ARG(QVariant, a2), Q_ARG(QVariant, a3),
                                  Q_ARG(QVariant, a4));
            break;
            break;
    }
    if (!ok) {
        return JsonRPCResult::error(3, "invoke failed", D("method", method));
    }

    QJsonObject result;
    result["invoked"] = method;
    result["class"] = hit->metaObject()->className();
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

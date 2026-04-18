// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvMultiViewFrameManager.h"

#include <CVLog.h>
#include <Visualization/ecvGLView.h>

#include <QApplication>
#include <QDrag>
#include <QInputDialog>
#include <QMenu>
#include <QMimeData>
#include <QMouseEvent>
#include <QStyle>
#include <QTimer>

ecvMultiViewFrameManager::ecvMultiViewFrameManager(QWidget* parent)
    : QObject(parent), m_parentWidget(parent) {
    m_mdiArea = new QMdiArea(parent);
    m_mdiArea->setViewMode(QMdiArea::TabbedView);
    m_mdiArea->setTabsClosable(true);
    m_mdiArea->setTabsMovable(true);
    m_mdiArea->setDocumentMode(false);
}

ecvMultiViewFrameManager::~ecvMultiViewFrameManager() = default;

QWidget* ecvMultiViewFrameManager::createViewFrame(
        QWidget* innerWidget,
        const QString& title,
        std::function<void(QWidget* toolbar, QWidget* viewWidget)>
                perViewToolbarCallback) {
    auto* frame = new QWidget(m_parentWidget);
    auto* frameLayout = new QVBoxLayout(frame);
    frameLayout->setContentsMargins(0, 0, 0, 0);
    frameLayout->setSpacing(1);

    // --- Title bar ---
    auto* titleBar = new QWidget(frame);
    titleBar->setObjectName("ViewTitleBar");
    titleBar->setContextMenuPolicy(Qt::CustomContextMenu);
    auto* titleLayout = new QHBoxLayout(titleBar);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(1);

    // --- Per-view toolbar (plain QWidget, not QToolBar) ---
    auto* viewToolBar = new QWidget(titleBar);
    viewToolBar->setObjectName("ViewSelectionToolBar");
    viewToolBar->setMinimumHeight(PV_ICON_SIZE + 4);
    viewToolBar->setStyleSheet("QToolButton{padding:1px;margin:0;}");
    auto* tbLayout = new QHBoxLayout(viewToolBar);
    tbLayout->setContentsMargins(0, 0, 0, 0);
    tbLayout->setSpacing(1);

    auto makeToolBtn = [viewToolBar](const QIcon& icon, const QString& tip) {
        auto* btn = new QToolButton(viewToolBar);
        btn->setIcon(icon);
        btn->setToolTip(tip);
        btn->setAutoRaise(true);
        btn->setIconSize(QSize(PV_ICON_SIZE, PV_ICON_SIZE));
        btn->setFixedSize(PV_ICON_SIZE + 6, PV_ICON_SIZE + 6);
        return btn;
    };

    // 3D/2D toggle
    auto* view3DBtn =
            makeToolBtn(QIcon(":/Resources/images/3D3.png"), tr("3D View"));
    view3DBtn->setObjectName("btn3DView");
    view3DBtn->setCheckable(true);
    view3DBtn->setChecked(true);
    tbLayout->addWidget(view3DBtn);

    // Capture screenshot
    auto* captureBtn =
            makeToolBtn(QIcon(":/Resources/images/svg/pqCaptureScreenshot.svg"),
                        tr("Capture Screenshot"));
    captureBtn->setObjectName("btnCaptureScreenshot");
    tbLayout->addWidget(captureBtn);

    // Edit camera
    auto* editCamBtn =
            makeToolBtn(QIcon(":/Resources/images/svg/pqEditCamera.svg"),
                        tr("Adjust Camera"));
    editCamBtn->setObjectName("btnAdjustCamera");
    tbLayout->addWidget(editCamBtn);

    // Separator
    auto addSeparator = [viewToolBar, tbLayout]() {
        auto* sep = new QFrame(viewToolBar);
        sep->setFrameShape(QFrame::VLine);
        sep->setFrameShadow(QFrame::Sunken);
        sep->setFixedWidth(2);
        tbLayout->addWidget(sep);
    };
    addSeparator();

    // Let the caller populate selection actions on the toolbar
    if (perViewToolbarCallback) {
        perViewToolbarCallback(viewToolBar, innerWidget);
    }

    titleLayout->addWidget(viewToolBar, 0);

    titleLayout->addStretch(1);

    auto* titleLabel = new QLabel(title, titleBar);
    titleLabel->setObjectName("ViewTitleLabel");
    titleLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    titleLabel->setProperty("plainTitle", title);
    titleLayout->addWidget(titleLabel);

    titleLayout->addSpacing(10);

    const int btnIconPx = PV_ICON_SIZE;
    const int btnSizePx = btnIconPx + 6;

    auto makeSvgBtn = [titleBar, btnIconPx, btnSizePx](const QString& iconPath,
                                                       const QString& tip) {
        auto* btn = new QToolButton(titleBar);
        btn->setIcon(QIcon(iconPath));
        btn->setToolTip(tip);
        btn->setAutoRaise(true);
        btn->setIconSize(QSize(btnIconPx, btnIconPx));
        btn->setFixedSize(btnSizePx, btnSizePx);
        return btn;
    };

    // Split Left|Right
    auto* splitLRBtn = makeSvgBtn(
            QStringLiteral(":/Resources/images/svg/pqSplitHorizontal.svg"),
            tr("Split Left|Right"));
    splitLRBtn->setObjectName("btnSplitHorizontal");
    titleLayout->addWidget(splitLRBtn);

    // Split Top|Bottom
    auto* splitTBBtn = makeSvgBtn(
            QStringLiteral(":/Resources/images/svg/pqSplitVertical.svg"),
            tr("Split Top|Bottom"));
    splitTBBtn->setObjectName("btnSplitVertical");
    titleLayout->addWidget(splitTBBtn);

    // Maximize
    auto* maxBtn = makeSvgBtn(QString(), tr("Maximize"));
    maxBtn->setIcon(
            titleBar->style()->standardIcon(QStyle::SP_TitleBarMaxButton));
    maxBtn->setProperty("maximized", false);
    connect(maxBtn, &QToolButton::clicked, this, [this, frame, maxBtn]() {
        toggleMaximizeViewFrame(frame, maxBtn);
    });
    titleLayout->addWidget(maxBtn);

    // Close
    auto* closeBtn =
            makeSvgBtn(QStringLiteral(":/Resources/images/svg/pqCloseView.svg"),
                       tr("Close View"));
    connect(closeBtn, &QToolButton::clicked, this,
            [this, frame]() { emit viewFrameCloseRequested(frame); });
    titleLayout->addWidget(closeBtn);

    frameLayout->addWidget(titleBar, 0);

    // --- Content frame ---
    auto* contentFrame = new QFrame(frame);
    contentFrame->setObjectName("CentralWidgetFrame");
    contentFrame->setFrameShape(QFrame::StyledPanel);
    contentFrame->setFrameShadow(QFrame::Sunken);
    auto* contentLayout = new QVBoxLayout(contentFrame);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(0);
    contentLayout->addWidget(innerWidget);
    frameLayout->addWidget(contentFrame, 1);

    // --- Context menu (rename) ---
    connect(titleBar, &QWidget::customContextMenuRequested, this,
            [this, frame, titleLabel](const QPoint& pos) {
                QMenu menu;
                menu.addAction(tr("Rename"), [this, frame, titleLabel]() {
                    bool ok = false;
                    QString current =
                            titleLabel->property("plainTitle").toString();
                    QString newName = QInputDialog::getText(
                            m_parentWidget, tr("Rename View"), tr("New name:"),
                            QLineEdit::Normal, current, &ok);
                    if (ok && !newName.isEmpty()) {
                        titleLabel->setProperty("plainTitle", newName);
                        titleLabel->setText(newName);
                        emit viewRenamed(frame, newName);
                    }
                });
                menu.exec(frame->findChild<QWidget*>("ViewTitleBar")
                                  ->mapToGlobal(pos));
            });

    // --- Drag-drop for swapping views ---
    titleBar->setAcceptDrops(true);
    const QString dragMime = QStringLiteral("application/x-acv-viewframe-%1")
                                     .arg(reinterpret_cast<quintptr>(qApp));

    struct DragEventFilter : public QObject {
        QWidget* frame;
        ecvMultiViewFrameManager* mgr;
        QString mime;
        QPoint startPos;

        DragEventFilter(QWidget* f,
                        ecvMultiViewFrameManager* m,
                        const QString& mimeType,
                        QObject* parent)
            : QObject(parent), frame(f), mgr(m), mime(mimeType) {}

        bool eventFilter(QObject* obj, QEvent* ev) override {
            auto* w = qobject_cast<QWidget*>(obj);
            if (!w) return false;

            switch (ev->type()) {
                case QEvent::MouseButtonPress: {
                    auto* me = static_cast<QMouseEvent*>(ev);
                    if (me->button() == Qt::LeftButton) startPos = me->pos();
                    break;
                }
                case QEvent::MouseMove: {
                    auto* me = static_cast<QMouseEvent*>(ev);
                    if (!(me->buttons() & Qt::LeftButton)) break;
                    if ((me->pos() - startPos).manhattanLength() <
                        QApplication::startDragDistance())
                        break;
                    auto* drag = new QDrag(w);
                    auto* mimeData = new QMimeData;
                    mimeData->setText(mime);
                    mimeData->setData(
                            "application/x-acv-frame-ptr",
                            QByteArray::number(
                                    reinterpret_cast<quintptr>(frame)));
                    drag->setMimeData(mimeData);
                    drag->setPixmap(
                            frame->grab().scaled(120, 80, Qt::KeepAspectRatio,
                                                 Qt::SmoothTransformation));
                    drag->exec(Qt::MoveAction);
                    break;
                }
                case QEvent::DragEnter: {
                    auto* de = static_cast<QDragEnterEvent*>(ev);
                    if (de->mimeData()->text() == mime) {
                        auto srcPtr = de->mimeData()
                                              ->data("application/x-acv-frame-"
                                                     "ptr")
                                              .toLongLong();
                        auto* srcFrame = reinterpret_cast<QWidget*>(
                                static_cast<quintptr>(srcPtr));
                        if (srcFrame != frame) de->acceptProposedAction();
                    }
                    break;
                }
                case QEvent::Drop: {
                    auto* de = static_cast<QDropEvent*>(ev);
                    if (de->mimeData()->text() == mime) {
                        auto srcPtr = de->mimeData()
                                              ->data("application/x-acv-frame-"
                                                     "ptr")
                                              .toLongLong();
                        auto* srcFrame = reinterpret_cast<QWidget*>(
                                static_cast<quintptr>(srcPtr));
                        if (srcFrame && srcFrame != frame) {
                            mgr->swapViewFrames(srcFrame, frame);
                        }
                    }
                    break;
                }
                default:
                    break;
            }
            return false;
        }
    };

    auto* dragFilter = new DragEventFilter(frame, this, dragMime, titleBar);
    titleBar->installEventFilter(dragFilter);

    emit viewFrameCreated(frame, innerWidget);
    return frame;
}

QMdiSubWindow* ecvMultiViewFrameManager::addFrameToMdi(
        QWidget* frame, const QString& tabTitle) {
    QMdiSubWindow* subWin = m_mdiArea->addSubWindow(frame);
    subWin->setWindowTitle(tabTitle);
    return subWin;
}

void ecvMultiViewFrameManager::splitViewFrame(
        QWidget* frameToSplit,
        Qt::Orientation orientation,
        std::function<ecvGLView*()> viewFactory) {
    if (!m_mdiArea || !frameToSplit || !viewFactory) return;

    auto* view = viewFactory();
    if (!view) return;

    // The new frame is created via createViewFrame — the caller's
    // perViewToolbarCallback is wired at new3DView / split level.
    // We emit splitViewCreated so the caller can register + wire up.

    // Create a new splitter
    auto* splitter = new QSplitter(orientation);
    splitter->setChildrenCollapsible(false);

    QWidget* parent = frameToSplit->parentWidget();
    auto* parentSplitter = qobject_cast<QSplitter*>(parent);
    auto* parentMdi = qobject_cast<QMdiSubWindow*>(parent);

    QList<int> savedParentSizes;
    int frameIdx = -1;
    if (parentSplitter) {
        savedParentSizes = parentSplitter->sizes();
        frameIdx = parentSplitter->indexOf(frameToSplit);
        parentSplitter->insertWidget(frameIdx, splitter);
    } else if (parentMdi) {
        parentMdi->setWidget(splitter);
    } else {
        QLayout* layout = parent ? parent->layout() : nullptr;
        if (layout) {
            layout->replaceWidget(frameToSplit, splitter);
        } else {
            view->deleteLater();
            delete splitter;
            return;
        }
    }

    splitter->addWidget(frameToSplit);
    frameToSplit->show();

    if (parentSplitter && !savedParentSizes.isEmpty()) {
        parentSplitter->setSizes(savedParentSizes);
    }

    // 50/50 split
    QTimer::singleShot(0, splitter, [splitter]() {
        int total = (splitter->orientation() == Qt::Horizontal)
                            ? splitter->width()
                            : splitter->height();
        int half = total / 2;
        splitter->setSizes({half, half});
    });

    // Caller creates frame and adds to splitter via splitViewCreated
    emit splitViewCreated(nullptr, view);
}

void ecvMultiViewFrameManager::toggleMaximizeViewFrame(QWidget* frame,
                                                       QToolButton* btn) {
    if (!frame || !btn || !m_mdiArea) return;

    bool isMaximized = btn->property("maximized").toBool();

    auto* splitter = qobject_cast<QSplitter*>(frame->parentWidget());
    if (!splitter) {
        for (auto* sub : m_mdiArea->subWindowList()) {
            if (sub->widget() == frame) {
                if (sub->isMaximized()) {
                    sub->showNormal();
                    btn->setIcon(frame->style()->standardIcon(
                            QStyle::SP_TitleBarMaxButton));
                    btn->setToolTip(tr("Maximize"));
                    btn->setProperty("maximized", false);
                } else {
                    sub->showMaximized();
                    btn->setIcon(frame->style()->standardIcon(
                            QStyle::SP_TitleBarNormalButton));
                    btn->setToolTip(tr("Restore"));
                    btn->setProperty("maximized", true);
                }
                break;
            }
        }
        return;
    }

    if (isMaximized) {
        for (int i = 0; i < splitter->count(); ++i) {
            QWidget* child = splitter->widget(i);
            if (child != frame) child->show();
        }
        auto savedSizes = splitter->property("savedSizes");
        if (savedSizes.isValid()) {
            splitter->setSizes(savedSizes.value<QList<int>>());
        }
        btn->setIcon(
                frame->style()->standardIcon(QStyle::SP_TitleBarMaxButton));
        btn->setToolTip(tr("Maximize"));
        btn->setProperty("maximized", false);
    } else {
        splitter->setProperty("savedSizes",
                              QVariant::fromValue(splitter->sizes()));
        for (int i = 0; i < splitter->count(); ++i) {
            QWidget* child = splitter->widget(i);
            if (child != frame) child->hide();
        }
        btn->setIcon(
                frame->style()->standardIcon(QStyle::SP_TitleBarNormalButton));
        btn->setToolTip(tr("Restore"));
        btn->setProperty("maximized", true);
    }
}

void ecvMultiViewFrameManager::swapViewFrames(QWidget* frameA,
                                              QWidget* frameB) {
    if (!frameA || !frameB || frameA == frameB) return;

    auto* splitterA = qobject_cast<QSplitter*>(frameA->parentWidget());
    auto* splitterB = qobject_cast<QSplitter*>(frameB->parentWidget());

    if (splitterA && splitterB) {
        int idxA = splitterA->indexOf(frameA);
        int idxB = splitterB->indexOf(frameB);

        if (splitterA == splitterB) {
            QList<int> sizes = splitterA->sizes();
            if (idxA < idxB) {
                splitterA->insertWidget(idxA, frameB);
                splitterA->insertWidget(idxB, frameA);
            } else {
                splitterA->insertWidget(idxB, frameA);
                splitterA->insertWidget(idxA, frameB);
            }
            splitterA->setSizes(sizes);
        } else {
            QList<int> sizesA = splitterA->sizes();
            QList<int> sizesB = splitterB->sizes();

            splitterA->insertWidget(idxA, frameB);
            splitterB->insertWidget(idxB, frameA);

            splitterA->setSizes(sizesA);
            splitterB->setSizes(sizesB);
        }

        frameA->show();
        frameB->show();
    }
}

void ecvMultiViewFrameManager::equalizeSplitter(QSplitter* splitter,
                                                bool horizontal,
                                                bool vertical) {
    if (!splitter || splitter->count() < 2) return;

    auto equalize = [](QSplitter* s) {
        int total =
                (s->orientation() == Qt::Horizontal) ? s->width() : s->height();
        int perChild = total / s->count();
        QList<int> sizes;
        sizes.reserve(s->count());
        for (int i = 0; i < s->count(); ++i) sizes << perChild;
        s->setSizes(sizes);
    };

    if ((splitter->orientation() == Qt::Horizontal && horizontal) ||
        (splitter->orientation() == Qt::Vertical && vertical)) {
        equalize(splitter);
    }

    for (int i = 0; i < splitter->count(); ++i) {
        auto* child = qobject_cast<QSplitter*>(splitter->widget(i));
        if (child) equalizeSplitter(child, horizontal, vertical);
    }
}

void ecvMultiViewFrameManager::lockViewSize(const QSize& size) {
    m_lockedViewSize = size;
    QSize maxSz =
            size.isEmpty() ? QSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX) : size;

    if (!m_mdiArea) return;
    for (auto* sub : m_mdiArea->subWindowList()) {
        QWidget* frame = sub->widget();
        if (!frame) continue;
        auto contentFrames = frame->findChildren<QFrame*>("CentralWidgetFrame");
        for (auto* cf : contentFrames) {
            cf->setMaximumSize(maxSz);
        }
    }
}

void ecvMultiViewFrameManager::markActiveViewFrame(QWidget* activeViewWidget) {
    if (!m_mdiArea || !activeViewWidget) return;

    QColor activeColor = m_parentWidget->palette().link().color();
    QString activeSS = QString("QFrame#CentralWidgetFrame "
                               "{ color: rgb(%1, %2, %3); }")
                               .arg(activeColor.red())
                               .arg(activeColor.green())
                               .arg(activeColor.blue());

    QList<QFrame*> allContentFrames;
    for (auto* sub : m_mdiArea->subWindowList()) {
        auto frames = sub->findChildren<QFrame*>("CentralWidgetFrame");
        allContentFrames.append(frames);
    }

    for (auto* contentFrame : allContentFrames) {
        bool isActive = contentFrame->isAncestorOf(activeViewWidget);
        contentFrame->setStyleSheet(isActive ? activeSS : QString());

        QWidget* parentFrame = contentFrame->parentWidget();
        if (!parentFrame) continue;
        auto* titleLabel = parentFrame->findChild<QLabel*>("ViewTitleLabel");
        if (!titleLabel) continue;
        QString plain = titleLabel->property("plainTitle").toString();
        if (plain.isEmpty()) plain = titleLabel->text();
        titleLabel->setText(isActive ? QString("<b><u>%1</u></b>").arg(plain)
                                     : plain);
    }

    emit activeFrameChanged(activeViewWidget);
}

void ecvMultiViewFrameManager::updateTabBarVisibility() {
    QTabBar* tb = m_mdiArea->findChild<QTabBar*>();
    if (tb) {
        tb->setVisible(true);
        auto* plusBtn = tb->findChild<QToolButton*>("TabBarPlusButton");
        if (plusBtn) {
            plusBtn->setVisible(true);
            int lastIdx = tb->count() - 1;
            QRect lastTabRect = lastIdx >= 0 ? tb->tabRect(lastIdx) : QRect();
            int x = lastTabRect.right() + 4;
            int y = (tb->height() - plusBtn->height()) / 2;
            plusBtn->move(x, qMax(0, y));
        }
    }
}

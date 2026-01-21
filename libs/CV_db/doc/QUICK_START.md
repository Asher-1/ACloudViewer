# Modal Shortcuts - Quick Start

## 30-Second Overview

ParaView-style shortcuts that **prevent conflicts** through mutual exclusion.

## The Problem

```cpp
// Traditional QShortcut (BAD)
new QShortcut(Qt::Key_S, widgetA);  // Selection tool
new QShortcut(Qt::Key_S, widgetB);  // VTK interaction
// Result: BOTH triggered! ❌ Ambiguous!
```

## The Solution

```cpp
// Modal shortcut (GOOD)
ecvKeySequences::instance().addModalShortcut(Qt::Key_S, action, widgetA);
ecvKeySequences::instance().addModalShortcut(Qt::Key_S, action, widgetB);
// Result: Only ONE active at a time! ✅ No ambiguity!
```

## Basic Usage

### 1. Include Header

```cpp
#include <Shortcuts/ecvKeySequences.h>
#include <Shortcuts/ecvModalShortcut.h>
```

### 2. Register Shortcut

```cpp
auto* shortcut = ecvKeySequences::instance().addModalShortcut(
    QKeySequence(Qt::Key_S),  // Key
    myAction,                 // QAction (or nullptr)
    myWidget                  // Context widget
);
shortcut->setObjectName("MyShortcut_S");
```

### 3. Connect Signal (if no action)

```cpp
connect(shortcut, &ecvModalShortcut::activated, 
        this, &MyClass::onShortcutPressed);
```

## Common Patterns

### Application-Wide Shortcut

```cpp
auto* shortcut = ecvKeySequences::instance().addModalShortcut(
    QKeySequence(Qt::ALT | Qt::Key_S),
    m_ui->actionSelectCells,
    this
);
shortcut->setContextWidget(this, Qt::ApplicationShortcut);
```

### Widget-Specific Shortcut

```cpp
auto* shortcut = ecvKeySequences::instance().addModalShortcut(
    QKeySequence(Qt::Key_S),
    nullptr,
    myVTKWidget
);
shortcut->setContextWidget(myVTKWidget, Qt::WidgetWithChildrenShortcut);
connect(shortcut, &ecvModalShortcut::activated, this, &MyClass::onS);
```

### Visual Feedback

```cpp
auto* decorator = new ecvShortcutDecorator(myFrame);
decorator->addShortcut(shortcut);
// myFrame will show a blue border when shortcut is active
```

## Debugging

### Check Active Shortcut

```cpp
auto* active = ecvKeySequences::instance().active(QKeySequence(Qt::Key_S));
CVLog::Print(active ? active->objectName() : "No active 'S' shortcut");
```

### Dump All Shortcuts for a Key

```cpp
ecvKeySequences::instance().dumpShortcuts(QKeySequence(Qt::Key_S));
// Output:
// [ecvKeySequences] Shortcuts for S:
//   - SelectTool_S: enabled
//   - VTKWidget_S: disabled
```

## Key Benefits

✅ **No Conflicts**: Automatic mutual exclusion  
✅ **Context-Aware**: Shortcuts respect widget focus  
✅ **Visual Feedback**: Optional border highlighting  
✅ **ParaView-Compatible**: Battle-tested design  
✅ **Easy to Debug**: Built-in utilities

## Full Documentation

- **README.md** - Complete API reference
- **SHORTCUT_INTEGRATION_EXAMPLE.md** - Step-by-step guide
- **PARAVIEW_SHORTCUT_MIGRATION.md** - Migration strategy

## Example: Selection Tools

```cpp
// MainWindow.cpp

void MainWindow::initializeShortcuts()
{
    // Selection tool (Alt+S) - application-wide
    auto* selShortcut = ecvKeySequences::instance().addModalShortcut(
        QKeySequence(Qt::ALT | Qt::Key_S),
        m_ui->actionSelectSurfaceCells,
        this
    );
    selShortcut->setObjectName("SelectSurfaceCells");
    
    // VTK interaction (S) - widget-specific
    auto* vtkShortcut = ecvKeySequences::instance().addModalShortcut(
        QKeySequence(Qt::Key_S),
        nullptr,
        m_vtkWidget
    );
    vtkShortcut->setObjectName("VTK_Surface");
    vtkShortcut->setContextWidget(m_vtkWidget, Qt::WidgetWithChildrenShortcut);
    connect(vtkShortcut, &ecvModalShortcut::activated, 
            this, &MainWindow::onVTKSurface);
    
    // No conflict! Alt+S and S are different keys
    // If two widgets register the same key, only one is active at a time
}
```

That's it! 🎉


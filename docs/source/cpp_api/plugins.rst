.. _cpp_plugins:

Plugin System
=============

ACloudViewer features a comprehensive plugin system that allows extending the application's functionality without modifying the core code.

Overview
--------

The plugin system is designed to provide:

* **Extensibility**: Add new features without recompiling the core application
* **Modularity**: Plugins are self-contained and can be enabled/disabled independently
* **Standard API**: All plugins follow a common interface defined by the Plugin API
* **Two Categories**: IO plugins and Standard (processing) plugins

Plugin Architecture
-------------------

.. code-block:: text

   ACloudViewer Application
   ├── Core Libraries
   │   ├── cloudViewer (3D processing)
   │   ├── eCV_db (database)
   │   ├── eCV_io (I/O operations)
   │   └── PCLEngine (PCL integration)
   │
   └── Plugin System
       ├── CVPluginAPI (Plugin interface)
       ├── CVPluginStub (Plugin base classes)
       └── plugins/core/
           ├── IO Plugins (file format support)
           └── Standard Plugins (processing algorithms)

Plugin API
----------

All plugins must implement the base plugin interface defined in ``CVPluginAPI``:

.. code-block:: cpp

   class ccPluginInterface {
   public:
       // Plugin metadata
       virtual QString getName() const = 0;
       virtual QString getDescription() const = 0;
       virtual QIcon getIcon() const = 0;
       
       // Plugin lifecycle
       virtual void onStartup() = 0;
       virtual void onShutdown() = 0;
       
       // Action registration
       virtual void registerActions(QMenu* menu) = 0;
   };

IO Plugins
----------

IO plugins extend file format support for point clouds, meshes, and other 3D data.

Available IO Plugins
~~~~~~~~~~~~~~~~~~~~

**Core IO**

* **qCoreIO**: Core file format support (PLY, PCD, XYZ, etc.)
* **qAdditionalIO**: Additional formats (OBJ, STL, OFF, etc.)
* **qMeshIO**: Advanced mesh formats (PLY, OBJ, STL, 3DS, etc.)

**Point Cloud Formats**

* **qLASIO**: LAS/LAZ point cloud format
* **qLASFWFIO**: LAS with full waveform support
* **qE57IO**: ASTM E57 format for terrestrial laser scanning
* **qPDALIO**: PDAL-based format support (multiple formats)

**CAD/3D Formats**

* **qDracoIO**: Google Draco compressed geometry
* **qFBXIO**: Autodesk FBX format
* **qStepCADImport**: STEP CAD format
* **qPhotoscanIO**: Agisoft Photoscan format
* **qRDBIO**: RIEGL RDB format

**Data Exchange**

* **qCSVMatrixIO**: CSV matrix data import/export

IO Plugin Interface
~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   class FileIOFilter : public ccPluginInterface {
   public:
       // File capabilities
       virtual QStringList getFileFilters(bool onImport) const = 0;
       virtual QString getDefaultExtension() const = 0;
       virtual bool canLoadExtension(const QString& upperCaseExt) const = 0;
       virtual bool canSave(CC_CLASS_ENUM type) const = 0;
       
       // Load/Save operations
       virtual CC_FILE_ERROR loadFile(
           const QString& filename,
           ccHObject& container,
           LoadParameters& parameters) = 0;
           
       virtual CC_FILE_ERROR saveToFile(
           ccHObject* entity,
           const QString& filename,
           const SaveParameters& parameters) = 0;
   };

Example: Using IO Plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <FileIOFilter.h>
   #include <ccPointCloud.h>
   
   // Load a point cloud using the plugin system
   ccHObject* container = new ccHObject("container");
   FileIOFilter::LoadParameters params;
   
   CC_FILE_ERROR result = FileIOFilter::LoadFromFile(
       "mydata.las",
       *container,
       params
   );
   
   if (result == CC_FERR_NO_ERROR) {
       ccPointCloud* cloud = static_cast<ccPointCloud*>(
           container->getChild(0)
       );
       // Process the cloud...
   }

Standard Plugins
----------------

Standard plugins provide processing algorithms and analysis tools.

Available Standard Plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Segmentation & Classification**

* **qCSF**: Cloth Simulation Filter for ground point filtering
* **qCanupo**: Multi-scale classification plugin
* **qColorimetricSegmenter**: Segmentation based on color information
* **qCloudLayers**: Layer-based point cloud organization
* **qMasonry**: Masonry structure analysis (auto/manual segmentation)

**Geometric Analysis**

* **qCompass**: Structural geology analysis and compass tools
* **qFacets**: Planar facet extraction and analysis
* **qHoughNormals**: Normal estimation using Hough transform
* **qM3C2**: Multiscale Model to Model Cloud Comparison
* **qMPlane**: Multiple plane detection and fitting
* **qRANSAC_SD**: RANSAC-based shape detection
* **qSRA**: Surface of Revolution Analysis

**Reconstruction & Modeling**

* **qPoissonRecon**: Poisson surface reconstruction
* **qCork**: Boolean operations on meshes (union, difference, intersection)
* **q3DMASC**: 3D Modeling And Shape Classification

**Animation & Visualization**

* **qAnimation**: Create animations and camera paths
* **qG3Point**: Geotechnical 3-point problem solver

**Advanced Processing**

* **qPCL**: Integration with Point Cloud Library (PCL) algorithms
* **qTreeIso**: Tree isolation and individual tree detection
* **qVoxFall**: Voxel-based analysis and processing

**Scripting & Automation**

* **qJSonRPCPlugin**: JSON-RPC interface for automation
* **qPythonRuntime**: Python scripting support

Standard Plugin Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   class ccStdPluginInterface : public ccPluginInterface {
   public:
       // Plugin actions
       virtual QList<QAction*> getActions() = 0;
       
       // Action execution
       virtual void onNewSelection(const ccHObject::Container& selectedEntities) = 0;
       
       // Processing
       virtual void doAction(QAction* action) = 0;
   };

Example: Using Standard Plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <ccPluginInterface.h>
   #include <ccPointCloud.h>
   
   // Example: Using CSF plugin for ground filtering
   ccPluginInterface* csfPlugin = getPluginByName("CSF");
   
   if (csfPlugin) {
       ccPointCloud* cloud = loadPointCloud("terrain.las");
       
       // Set input
       ccHObject::Container selected;
       selected.push_back(cloud);
       csfPlugin->onNewSelection(selected);
       
       // Execute ground filtering
       QAction* filterAction = csfPlugin->getActions().first();
       csfPlugin->doAction(filterAction);
   }

Plugin Metadata (info.json)
----------------------------

Each plugin includes an ``info.json`` file with metadata:

.. code-block:: json

   {
       "type": "Standard",
       "name": "CSF",
       "icon": ":/CC/plugin/qCSF/icon.png",
       "description": "Cloth Simulation Filter for ground point filtering",
       "authors": [
           {
               "name": "Wuming Zhang",
               "email": "wumingzh@gmail.com"
           }
       ],
       "maintainers": [
           {
               "name": "ACloudViewer Team",
               "email": "support@cloudviewer.org"
           }
       ],
       "version": "1.5",
       "min_version": "3.0.0",
       "max_version": "4.0.0",
       "references": [
           {
               "text": "CSF Algorithm Paper",
               "url": "https://doi.org/10.1016/j.isprsjprs.2016.01.011"
           }
       ]
   }

Creating a New Plugin
---------------------

Step 1: Plugin Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

Create the following directory structure:

.. code-block:: text

   plugins/core/Standard/qMyPlugin/
   ├── CMakeLists.txt
   ├── info.json
   ├── include/
   │   └── qMyPlugin.h
   ├── src/
   │   └── qMyPlugin.cpp
   ├── ui/
   │   └── myPluginDialog.ui (optional)
   └── images/
       └── icon.png

Step 2: Plugin Header
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // qMyPlugin.h
   #ifndef Q_MY_PLUGIN_H
   #define Q_MY_PLUGIN_H
   
   #include <ccStdPluginInterface.h>
   
   class qMyPlugin : public QObject, public ccStdPluginInterface {
       Q_OBJECT
       Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
       Q_PLUGIN_METADATA(IID "cccorp.cloudviewer.plugin.qMyPlugin" FILE "info.json")
       
   public:
       explicit qMyPlugin(QObject* parent = nullptr);
       virtual ~qMyPlugin() = default;
       
       // ccPluginInterface
       QString getName() const override { return "My Plugin"; }
       QString getDescription() const override { return "My custom plugin"; }
       QIcon getIcon() const override;
       
       void onStartup() override;
       void onShutdown() override;
       void registerActions(QMenu* menu) override;
       
       // ccStdPluginInterface
       QList<QAction*> getActions() override;
       void onNewSelection(const ccHObject::Container& selectedEntities) override;
       void doAction(QAction* action) override;
       
   private slots:
       void doMyAction();
       
   private:
       QAction* m_action;
   };
   
   #endif // Q_MY_PLUGIN_H

Step 3: Plugin Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // qMyPlugin.cpp
   #include "qMyPlugin.h"
   #include <ccPointCloud.h>
   
   qMyPlugin::qMyPlugin(QObject* parent)
       : QObject(parent)
       , ccStdPluginInterface(":/CC/plugin/qMyPlugin/info.json")
       , m_action(nullptr)
   {
   }
   
   QIcon qMyPlugin::getIcon() const {
       return QIcon(":/CC/plugin/qMyPlugin/icon.png");
   }
   
   void qMyPlugin::onStartup() {
       // Plugin initialization
   }
   
   void qMyPlugin::onShutdown() {
       // Plugin cleanup
   }
   
   void qMyPlugin::registerActions(QMenu* menu) {
       if (!menu) return;
       
       m_action = new QAction(getName(), this);
       m_action->setToolTip(getDescription());
       m_action->setIcon(getIcon());
       connect(m_action, &QAction::triggered, this, &qMyPlugin::doMyAction);
       
       menu->addAction(m_action);
   }
   
   QList<QAction*> qMyPlugin::getActions() {
       return QList<QAction*>() << m_action;
   }
   
   void qMyPlugin::onNewSelection(const ccHObject::Container& selectedEntities) {
       // Update action state based on selection
       bool hasPointCloud = false;
       for (ccHObject* entity : selectedEntities) {
           if (entity->isA(CC_TYPES::POINT_CLOUD)) {
               hasPointCloud = true;
               break;
           }
       }
       m_action->setEnabled(hasPointCloud);
   }
   
   void qMyPlugin::doAction(QAction* action) {
       if (action == m_action) {
           doMyAction();
       }
   }
   
   void qMyPlugin::doMyAction() {
       // Your plugin logic here
       ccPointCloud* cloud = getCurrentPointCloud();
       if (!cloud) return;
       
       // Process the cloud...
   }

Step 4: CMakeLists.txt
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

   # CMakeLists.txt
   cmake_minimum_required(VERSION 3.18)
   
   project(qMyPlugin VERSION 1.0)
   
   include(../../cmake/plugin.cmake)
   
   # Source files
   set(PLUGIN_SOURCES
       src/qMyPlugin.cpp
   )
   
   set(PLUGIN_HEADERS
       include/qMyPlugin.h
   )
   
   # UI files (optional)
   set(PLUGIN_UI
       ui/myPluginDialog.ui
   )
   
   # Resources (optional)
   set(PLUGIN_QRC
       qMyPlugin.qrc
   )
   
   # Create plugin
   add_cloudviewer_plugin(
       NAME ${PROJECT_NAME}
       TYPE STANDARD
       SOURCES ${PLUGIN_SOURCES}
       HEADERS ${PLUGIN_HEADERS}
       UI ${PLUGIN_UI}
       QRC ${PLUGIN_QRC}
   )

Step 5: Build and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Configure with plugin enabled
   cd build
   cmake -DPLUGIN_STANDARD_QMYPLUGIN=ON ..
   
   # Build
   make qMyPlugin -j$(nproc)
   
   # The plugin will be installed to:
   # - Linux: <install_prefix>/lib/cloudViewer/plugins/
   # - macOS: ACloudViewer.app/Contents/PlugIns/
   # - Windows: <install_prefix>/plugins/

Plugin Loading
--------------

Plugins are automatically discovered and loaded at application startup:

1. **Plugin Discovery**: Application scans plugin directories
2. **Metadata Loading**: Reads ``info.json`` for each plugin
3. **Version Checking**: Verifies compatibility with application version
4. **Plugin Instantiation**: Creates plugin instances
5. **Registration**: Registers plugin actions in menus/toolbars

Plugin Directories
~~~~~~~~~~~~~~~~~~

Default plugin search paths:

.. code-block:: text

   Linux:
   - /usr/lib/cloudViewer/plugins/
   - ~/.local/share/cloudViewer/plugins/
   - $CLOUDVIEWER_PLUGINS_PATH
   
   macOS:
   - ACloudViewer.app/Contents/PlugIns/
   - ~/Library/Application Support/cloudViewer/plugins/
   - $CLOUDVIEWER_PLUGINS_PATH
   
   Windows:
   - C:\Program Files\ACloudViewer\plugins\
   - %APPDATA%\cloudViewer\plugins\
   - %CLOUDVIEWER_PLUGINS_PATH%

Best Practices
--------------

1. **Thread Safety**: Use Qt's signal/slot mechanism for thread-safe operations
2. **Error Handling**: Always check return values and handle errors gracefully
3. **Memory Management**: Use Qt's parent-child relationship for automatic cleanup
4. **Progress Reporting**: Use progress dialogs for long-running operations
5. **Undo Support**: Implement undo/redo for destructive operations
6. **Documentation**: Provide clear documentation and examples

Debugging Plugins
-----------------

Enable plugin debugging:

.. code-block:: bash

   # Set environment variable
   export CLOUDVIEWER_PLUGIN_DEBUG=1
   
   # Run application
   ./ACloudViewer

Debug output will show:
- Plugin discovery process
- Loading successes/failures
- Plugin initialization
- Action registration

Further Reading
---------------

* :doc:`../getting_started/build_from_source` - Build ACloudViewer with plugins
* :doc:`overview` - C++ API overview
* `Plugin Examples <https://github.com/Asher-1/ACloudViewer/tree/main/plugins/example>`_
* `CVPluginAPI Documentation <../cpp_api/doxygen/html/>`_

---

**See Also:**

* Complete C++ API: `Doxygen Documentation <../cpp_api/doxygen/html/index.html>`_
* Plugin source code: `plugins/core/ <https://github.com/Asher-1/ACloudViewer/tree/main/plugins/core>`_

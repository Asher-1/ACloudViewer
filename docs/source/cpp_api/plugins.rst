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
   │   ├── CV_db (database)
   │   ├── CV_io (I/O operations)
   │   └── VtkEngine (VTK rendering engine)
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

**Image-Based Rendering & Neural Radiance**

* **qSIBR**: SIBR framework integration — real-time 3D Gaussian Splatting (CUDA), ULR novel-view synthesis, textured mesh and point-based viewing, remote training viewer, and dataset preprocessing tools. Bidirectional interaction with ACloudViewer: entity selection auto-fills viewer paths, results import back with zoom-to-fit. Enable with ``-DPLUGIN_STANDARD_QSIBR=ON`` (not supported on macOS due to OpenGL-on-Metal limitations 4.1 +).

**AI inference (AICore)**

Twenty-three plugins share one native inference library — ``libAICore.so``
(``core/AICore/``), whose ggml runtime is vendored from llama.cpp. All require
``-DAICore_ENABLED=ON``; quantized **GGUF** models run on CUDA / Vulkan / Metal /
CPU with no Python or PyTorch at runtime, and results land directly in the DB
tree. Plugins consume only the public ``aicore/`` C headers (``depth_capi.h``,
``gaussian_capi.h``, ``image_view.h``, ``pipeline_timing.h``, ``runtime_capi.h``,
``asset_digests.h`` and the per-task ``*_capi.h``); Qt, STL, OpenCV, exceptions
and ggml types never cross the ABI. User-facing guides live in
``docs/guides/plugins/`` (synced into Sphinx at build time); the full developer
README for each plugin is under ``plugins/core/Standard/``.

*Perception — depth, lines, matching, faces, text*

* **qDA3**: Depth Anything V3 — monocular/multiview depth, camera pose, GLB/COLMAP export, Automatic Reconstruction sparse/dense hooks. CMake: ``PLUGIN_STANDARD_QDA3``. Guide: :doc:`../guides/plugins/qDA3`.
* **qDeepLSD**: DeepLSD wireframe — line-segment extraction from GGUF distance + angle fields with AFM-guided LSD post-processing. CMake: ``PLUGIN_STANDARD_QDEEPLSD``. Guide: :doc:`../guides/plugins/qDeepLSD`.
* **qLightGlue**: SIFT/ALIKED LightGlue GGUF — sub-second sparse feature matching with live visualization. CMake: ``PLUGIN_STANDARD_QLIGHTGLUE``. Guide: :doc:`../guides/plugins/qLightGlue`.
* **qFaceDetect**: face-detect.cpp GGUF packs — SCRFD/YuNet detection, ArcFace/SFace verification, identity registry, live multi-face video. CMake: ``PLUGIN_STANDARD_QFACEDETECT``. Guide: :doc:`../guides/plugins/qFaceDetect`.
* **qOCR**: PP-OCRv4 GGUF (EN / JA / CH+EN recipes) — text recognition with opt-in SLANet_plus layout and table stages, PP-Structure style DB overlay. CMake: ``PLUGIN_STANDARD_QOCR``. Guide: :doc:`../guides/plugins/qOCR`.

*Detection, segmentation & open-vocabulary understanding*

* **qYOLO**: YOLOv8 / YOLO26 GGUF (63 models) — detection, instance segmentation, metric depth, pose, OBB, classification + open-vocab World / YOLOE prompts. CMake: ``PLUGIN_STANDARD_QYOLO``. Guide: :doc:`../guides/plugins/qYOLO`.
* **qSAM3**: SAM2 / 2.1 / 3 GGUF (39 models) — point/box/text promptable segmentation plus video object tracking. CMake: ``PLUGIN_STANDARD_QSAM3``. Guide: :doc:`../guides/plugins/qSAM3`.
* **qRFDetr**: RF-DETR GGUF (44 models) — real-time detection & instance masks, per-class allowlist, 13 ms GPU latency. CMake: ``PLUGIN_STANDARD_QRFDETR``. Guide: :doc:`../guides/plugins/qRFDetr`.
* **qRMBG**: RMBG-2.0 (BiRefNet-Swin-L) GGUF — transparent background removal with raw alpha matte, image + live video. CMake: ``PLUGIN_STANDARD_QRMBG``. Guide: :doc:`../guides/plugins/qRMBG`.
* **qRAM**: RAM / RAM++ / Tag2Text GGUF — open-set image tagging & captioning, all logits + thresholds, JSON/CSV/text export. CMake: ``PLUGIN_STANDARD_QRAM``. Guide: :doc:`../guides/plugins/qRAM`.
* **qRelateAnything**: RelSGG ViT / ConvNeXt GGUF (24 models) — subject–predicate–object scene relation graphs over tracked objects (image & video). CMake: ``PLUGIN_STANDARD_QRELATEANYTHING``. Guide: :doc:`../guides/plugins/qRelateAnything`.
* **qGKD**: GKDT-L GGUF (ECCV 2026) — open-world keypoint detection with text / visual 1-shot / multimodal / multi-object prompts. CMake: ``PLUGIN_STANDARD_QGKD``. Guide: :doc:`../guides/plugins/qGKD`.
* **qVLXSeek**: VLX-Seek 1.5-10B GGUF — open-vocab grounding: detection, counting, VQA and region OCR via region retrieval. CMake: ``PLUGIN_STANDARD_QVLXSEEK``. Guide: :doc:`../guides/plugins/qVLXSeek`.
* **qWeVisOne**: WeVisOne-VL 2B/4B (Qwen3-VL) GGUF — unified perception: detection, referring, pointing, counting, OCR, segmentation, matting and visual-prompt grounding via structured box tokens plus a pixel decoder for alpha mattes. CMake: ``PLUGIN_STANDARD_QWEVISONE``. Guide: :doc:`../guides/plugins/qWeVisOne`.

*3D generation & reconstruction*

* **qFreeSplatter**: FreeSplatter 3D Gaussian Splatting — uncalibrated images to per-pixel 3D Gaussians, PnP pose recovery, SIBR-compatible PLY export, guided Face Capture; optional in-app viewer when qSIBR is enabled. CMake: ``PLUGIN_STANDARD_QFREESPLATTER``. Guide: :doc:`../guides/plugins/qFreeSplatter`.
* **qTrellis**: TRELLIS.2 GGUF (DINOv3 + flow-matching DiTs) — single image to PBR-textured 3D mesh & GLB. CMake: ``PLUGIN_STANDARD_QTRELLIS``. Guide: :doc:`../guides/plugins/qTrellis`.
* **qInstantMesh**: InstantMesh GGUF (14 files) — Zero123++ 6-view diffusion to FlexiCubes mesh + UV atlas (GLB/OBJ). CMake: ``PLUGIN_STANDARD_QINSTANTMESH``. Guide: :doc:`../guides/plugins/qInstantMesh`.
* **qSAM3D**: SAM 3D Objects GGUF — single image to Gaussian splat PLY + optional FlexiCubes mesh. CMake: ``PLUGIN_STANDARD_QSAM3D``. Guide: :doc:`../guides/plugins/qSAM3D`.
* **qQuerySplat**: QuerySplat + VGGT-Omega GGUF — 2–16 unposed photos to a 3DGS scene (524 288 gaussians), optional test-time optimization. CMake: ``PLUGIN_STANDARD_QQUERYSPLAT``. Guide: :doc:`../guides/plugins/qQuerySplat`.
* **qMapAnything**: VGGT-Omega / MapAnything / Pi3 / Pi3X / VGGT-1B / DUSt3R GGUF — feed-forward metric multi-view reconstruction, ETH3D benchmark anchoring. CMake: ``PLUGIN_STANDARD_QMAPANYTHING``. Guide: :doc:`../guides/plugins/qMapAnything`.
* **qLingbotMap**: LingBot-Map GCT GGUF — streaming RGB-D reconstruction from image folders / video with live per-frame map, frustums and sky masking. CMake: ``PLUGIN_STANDARD_QLINGBOTMAP``. Guide: :doc:`../guides/plugins/qLingbotMap`.
* **qOpenPCDet**: OpenPCDet GGUF (54 models) — PointPillars / SECOND / PV-RCNN LiDAR 3D detection on KITTI + nuScenes, wireframe boxes + CSV export. CMake: ``PLUGIN_STANDARD_QOPENPCDET``. Guide: :doc:`../guides/plugins/qOpenPCDet`.

*Annotation workbench*

* **qAnyLabeling**: X-AnyLabeling-style annotation workbench — shapes, brush, magic wand, video tracking + 3D cuboids; every AI assist runs locally through the AICore catalogs (YOLO, SAM3, RF-DETR, …). CMake: ``PLUGIN_STANDARD_QANYLABELING``. Guide: :doc:`../guides/plugins/qAnyLabeling`.

See :doc:`../guides/plugins/README` for the complete overview and build
prerequisites, and :doc:`overview` for the AICore module architecture.

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

Plugin Documentation
--------------------

Detailed documentation for each plugin, including CLI usage, parameters, and references.
Each plugin's README is maintained alongside its source code under ``plugins/core/``.
The catalog and build index is at ``plugins/README.md`` in the repository root.

.. toctree::
   :maxdepth: 1
   :caption: AI Inference (AICore)

   ../guides/plugins/README
   ../guides/plugins/qDA3
   ../guides/plugins/qDeepLSD
   ../guides/plugins/qFaceDetect
   ../guides/plugins/qLightGlue
   ../guides/plugins/qFreeSplatter
   ../guides/plugins/qSAM3
   ../guides/plugins/qTrellis
   ../guides/plugins/qYOLO
   ../guides/plugins/qRFDetr
   ../guides/plugins/qRMBG
   ../guides/plugins/qGKD
   ../guides/plugins/qInstantMesh
   ../guides/plugins/qLingbotMap
   ../guides/plugins/qMapAnything
   ../guides/plugins/qOCR
   ../guides/plugins/qOpenPCDet
   ../guides/plugins/qQuerySplat
   ../guides/plugins/qRAM
   ../guides/plugins/qRelateAnything
   ../guides/plugins/qSAM3D
   ../guides/plugins/qVLXSeek
   ../guides/plugins/qWeVisOne
   ../guides/plugins/qAnyLabeling
   ../plugins/qDA3/README
   ../plugins/qDeepLSD/README
   ../plugins/qFaceDetect/README
   ../plugins/qLightGlue/README
   ../plugins/qFreeSplatter/README
   ../plugins/qSAM3/README
   ../plugins/qTrellis/README
   ../plugins/qYOLO/README
   ../plugins/qRFDetr/README
   ../plugins/qRMBG/README
   ../plugins/qGKD/README
   ../plugins/qInstantMesh/README
   ../plugins/qLingbotMap/README
   ../plugins/qMapAnything/README
   ../plugins/qOCR/README
   ../plugins/qOpenPCDet/README
   ../plugins/qQuerySplat/README
   ../plugins/qRAM/README
   ../plugins/qRelateAnything/README
   ../plugins/qSAM3D/README
   ../plugins/qVLXSeek/README
   ../plugins/qAnyLabeling/README

.. toctree::
   :maxdepth: 1
   :caption: Processing & Analysis

   ../plugins/qPCV/README
   ../plugins/qCSF/README
   ../plugins/qM3C2/README
   ../plugins/qRANSAC_SD/README
   ../plugins/qCanupo/README
   ../plugins/qFacets/README
   ../plugins/qHoughNormals/README
   ../plugins/qPoissonRecon/README
   ../plugins/qCork/README
   ../plugins/qVoxFall/README
   ../plugins/q3DMASC/README
   ../plugins/qTreeIso/README
   ../plugins/qG3Point/README

.. toctree::
   :maxdepth: 1
   :caption: Structural Geology

   ../plugins/qCompass/README
   ../plugins/qSRA/README

.. toctree::
   :maxdepth: 1
   :caption: Segmentation & Classification

   ../plugins/qCloudLayers/README
   ../plugins/qColorimetricSegmenter/README
   ../plugins/qMasonry/README
   ../plugins/qMPlane/README

.. toctree::
   :maxdepth: 1
   :caption: Visualization & Integration

   ../plugins/qAnimation/README
   ../plugins/qSIBR/README
   ../plugins/qPCL/README
   ../plugins/qPythonRuntime/README
   ../plugins/qJSonRPCPlugin/README

.. toctree::
   :maxdepth: 1
   :caption: I/O Plugins

   ../plugins/qLASIO/README
   ../plugins/qE57IO/README
   ../plugins/qDracoIO/README
   ../plugins/qFBXIO/README
   ../plugins/qMeshIO/README
   ../plugins/qLASFWFIO/README
   ../plugins/qCSVMatrixIO/README
   ../plugins/qPhotoscanIO/README
   ../plugins/qCoreIO/README
   ../plugins/qAdditionalIO/README
   ../plugins/qPDALIO/README
   ../plugins/qRDBIO/README
   ../plugins/qStepCADImport/README

Further Reading
---------------

* :doc:`../getting_started/build_from_source` - Build ACloudViewer with plugins
* :doc:`overview` - C++ API overview
* `Plugin catalog (plugins/README.md) <https://github.com/Asher-1/ACloudViewer/blob/main/plugins/README.md>`_
* `Plugin Examples <https://github.com/Asher-1/ACloudViewer/tree/main/plugins/example>`_
* `CVPluginAPI Documentation <api/>`_

---

**See Also:**

* Complete C++ API: `Doxygen Documentation <api/index.html>`_
* Plugin source code: `plugins/core/ <https://github.com/Asher-1/ACloudViewer/tree/main/plugins/core>`_

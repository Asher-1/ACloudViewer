.. ACloudViewer documentation master file

.. image:: ../images/ACloudViewer_logo_horizontal.png
   :alt: ACloudViewer Logo
   :width: 400px
   :align: center

-----------

ACloudViewer: A Modern Library for 3D Point Cloud Processing
=============================================================

**ACloudViewer** is a powerful open-source library for 3D point cloud and mesh processing, built on top of CloudCompare, Open3D, ParaView, and COLMAP.

.. note::
   **Latest Release:** |version| | `Download <https://github.com/Asher-1/ACloudViewer/releases>`_ | `GitHub <https://github.com/Asher-1/ACloudViewer>`_

.. raw:: html

   <div style="margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 8px; border-left: 4px solid #2196F3;">
       <p style="margin: 0 0 10px 0; font-weight: 600; color: #333;">
           📚 <strong>Documentation Version:</strong>
       </p>
       <select id="docs-version-select-main" 
               style="width: 100%; max-width: 300px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; background: white; cursor: pointer;"
               onchange="if(window.ACloudViewerVersionSwitcher) { window.ACloudViewerVersionSwitcher.switchVersion(this.value); }">
           <option value="latest">Latest (main)</option>
       </select>
       <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">
           Switch between different documentation versions. Use the version selector in the sidebar for quick access.
       </p>
   </div>
   <script>
   // Initialize main page version selector
   (function() {
       function updateMainSelector() {
           const select = document.getElementById('docs-version-select-main');
           if (!select) {
               setTimeout(updateMainSelector, 200);
               return;
           }
           
           if (!window.ACloudViewerVersionSwitcher) {
               setTimeout(updateMainSelector, 200);
               return;
           }
           
           const versions = window.ACloudViewerVersionSwitcher.getVersions();
           const currentVersion = window.ACloudViewerVersionSwitcher.getCurrentVersion();
           
           if (versions.length > 0) {
               select.innerHTML = '';
               versions.forEach(v => {
                   const option = document.createElement('option');
                   option.value = v.value;
                   option.textContent = v.display;
                   if (v.value === currentVersion || (currentVersion === 'latest' && v.value === 'latest')) {
                       option.selected = true;
                   }
                   select.appendChild(option);
               });
           }
       }
       
       if (document.readyState === 'loading') {
           document.addEventListener('DOMContentLoaded', updateMainSelector);
       } else {
           updateMainSelector();
       }
       
       document.addEventListener('versionsLoaded', updateMainSelector);
   })();
   </script>

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/introduction
   getting_started/installation
   getting_started/quickstart
   getting_started/build_from_source

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   tutorial/core/index
   tutorial/geometry/index
   tutorial/t_geometry/index
   tutorial/data/index
   tutorial/visualization/index
   tutorial/pipelines/index
   tutorial/t_pipelines/index
   tutorial/reconstruction_system/index
   tutorial/t_reconstruction_system/index
   tutorial/sensor/index
   tutorial/ml/index
   tutorial/advanced/index
   tutorial/reference

.. toctree::
   :maxdepth: 1
   :caption: Python API

   python_api/cloudViewer.camera
   python_api/cloudViewer.core
   python_api/cloudViewer.data
   python_api/cloudViewer.geometry
   python_api/cloudViewer.io
   python_api/cloudViewer.t
   python_api/cloudViewer.ml
   python_api/cloudViewer.pipelines
   python_api/cloudViewer.reconstruction
   python_api/cloudViewer.utility
   python_api/cloudViewer.visualization

.. toctree::
   :maxdepth: 2
   :caption: Python Examples

   python_example/benchmark/index
   python_example/camera/index
   python_example/core/index
   python_example/geometry/index
   python_example/io/index
   python_example/pipelines/index
   python_example/reconstruction/index
   python_example/reconstruction_system/index
   python_example/t_reconstruction_system/index
   python_example/utility/index
   python_example/visualization/index

.. toctree::
   :maxdepth: 1
   :caption: C++ Examples

   examples/cpp_examples

.. toctree::
   :maxdepth: 1
   :caption: C++ API

   cpp_api

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer/contributing
   developer/docker
   developer/ci_cd

.. toctree::
   :maxdepth: 1
   :caption: Resources

   resources/changelog
   resources/faq
   resources/support

..
   Note: Python API and Examples sections will be auto-generated from docstrings
   when Python bindings include proper documentation.

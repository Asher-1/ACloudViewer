.. ACloudViewer documentation master file

ACloudViewer Documentation
==========================

.. image:: ../images/ACloudViewer_logo_horizontal.png
   :alt: ACloudViewer Logo
   :align: center
   :width: 400px

|

**ACloudViewer** is a powerful open-source library for 3D point cloud and mesh processing, providing a comprehensive set of tools for visualization, registration, reconstruction, and machine learning applications.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   introduction
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic
   tutorials/visualization
   tutorials/registration
   tutorials/reconstruction
   tutorials/machine_learning

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/cpp_api
   api/python_api

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   building_from_source
   contributing
   docker

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   release_notes
   faq
   support


Key Features
------------

🎯 **Point Cloud Processing**
   Advanced algorithms for point cloud filtering, sampling, and segmentation

📊 **3D Visualization**
   Interactive visualization tools with support for multiple rendering backends

🔧 **Registration & Alignment**
   ICP, GICP, and other state-of-the-art registration algorithms

🏗️ **3D Reconstruction**
   Surface reconstruction from point clouds using Poisson, Ball Pivoting, and more

🤖 **Machine Learning**
   Integration with modern ML frameworks for 3D understanding

🐳 **Docker Support**
   Pre-configured Docker images for easy deployment


Quick Links
-----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* `GitHub Repository <https://github.com/Asher-1/ACloudViewer>`_
* `Download Releases <https://asher-1.github.io/ACloudViewer/>`_


Installation
------------

Quick install via pip::

    # Download wheel from GitHub Releases
    # https://github.com/Asher-1/ACloudViewer/releases
    pip install cloudviewer-*.whl

Or build from source::

    git clone https://github.com/Asher-1/ACloudViewer.git
    cd ACloudViewer
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    make install

See :doc:`installation` for detailed instructions.


Quick Example
-------------

.. code-block:: python

   import cloudViewer as cv3d
   import numpy as np

   # Create a point cloud
   points = np.random.rand(1000, 3)
   pcd = cv3d.geometry.PointCloud()
   pcd.points = cv3d.utility.Vector3dVector(points)

   # Visualize
   cv3d.visualization.draw_geometries([pcd])


Community & Support
-------------------

* **Issues**: Report bugs on `GitHub Issues <https://github.com/Asher-1/ACloudViewer/issues>`_
* **Discussions**: Join our `GitHub Discussions <https://github.com/Asher-1/ACloudViewer/discussions>`_
* **Contributing**: See our :doc:`contributing` guide


License
-------

ACloudViewer is released under the MIT License. See the LICENSE file for details.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


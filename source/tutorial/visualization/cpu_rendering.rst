.. _cpu_rendering:

CPU (Software) Rendering
========================

CloudViewer's new visualization functionality (:class:`.O3DVisualizer` class,
:func:`.draw()` function and :mod:`cloudViewer.visualization.gui` and
:mod:`cloudViewer.visualization.rendering` modules) requires a recent GPU with
support for OpenGL 4.1 or higher. This is not available in certain situations:

    - GPU is too old to support OpenGL 4.1.
    - No GPU is available, for example on cloud servers that do not have any GPU
      (integrated or discrete) installed, or in a docker container that does not
      have access to the host GPU. This is often the case for many cloud based
      Jupyter notebooks such as Google Colab, Kaggle, etc.
    - A GPU is available, but it only supports computation, not graphics. This
      is a common scenario for cloud based Jupyter notebooks deployed in docker
      containers.

CloudViewer supports CPU or software rendering in such situations. Note that this
usually produces slower and less responsive rendering, so a GPU is recommended.
Currently, this is available only for Linux. There are two separate ways to
use CPU rendering depending on whether interactive or headless rendering is
desired. Both methods are described below.

Headless CPU Rendering
----------------------

For Python code, you can enable CPU rendering for headless rendering when using
the :class: `.OffscreenRenderer` for a process by setting an environment
variable  before importing CloudViewer::

 - ``EGL_PLATFORM=surfaceless`` for Ubuntu 20.04+ (Mesa v20.2 or newer)

Here are the different ways to do that:

.. code:: bash

    # from the command line (Ubuntu 20.04+)
    EGL_PLATFORM=surfaceless python examples/Python/visualization/render_to_image.py

.. code:: python

    # In Python code
    import os
    os.environ['EGL_PLATFORM'] = 'surfaceless'   # Ubuntu 20.04+
    import cloudViewer as cv3d

    # In a Jupyter notebook
    %env EGL_PLATFORM surfaceless   # Ubuntu 20.04+
    import cloudViewer as cv3d

.. note:: Setting the environment variable after importing ``cloudViewer`` will not work,
    even if ``cloudViewer`` is re-imported. In this case, if no usable GPU is present, the
    Python interpreter or Jupyter kernel will crash when visualization functions are
    used.

.. note:: This method will **not** work for interactive rendering scripts such
   as ``examples/Python/visualization/draw.py``. For interactive rendering see
   the next section.

Interactive CPU Rendering
-------------------------

The method for enabling interactive CPU rendering depends on your system:

1.  **You use Mesa drivers v20.2 or higher.** This is the case for all
    Intel GPUs and some AMD and Nvidia GPUs. You should be running a recent Linux
    OS, such as Ubuntu 20.04. Check your Mesa version from your package manager
    (e.g. run ``dpkg -s libglx-mesa0 | grep Version`` in Debian or Ubuntu). In this
    case, you can switch to CPU rendering by simply setting an environment
    variable before starting your application. For example, start the CloudViewer
    visualizer app in CPU rendering mode with:

    .. code:: bash

        LIBGL_ALWAYS_SOFTWARE=true CloudViewer

    Or for Python code:

    .. code:: bash

        LIBGL_ALWAYS_SOFTWARE=true python examples/Python/visualization/draw.py

.. note:: Mesa drivers must be in use for this method to work; just having
   them installed is not sufficient. You can check the drivers in use with the
   ``glxinfo`` command.

2.  **You use Nvidia or AMD drivers, but your OS comes with recent Mesa drivers (>= v20.2).** 
    Install Mesa drivers if they are not installed in your system (e.g. `sudo apt install libglx0-mesa`
    in Ubuntu). Preload the Mesa driver library before running any CloudViewer application requiring CPU rendering.
    For example:

    .. code:: bash

        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLX_mesa.so.0
        CloudViewer

    Or with Python code:

    .. code:: bash

        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLX_mesa.so.0
        python examples/Python/visualization/draw.py

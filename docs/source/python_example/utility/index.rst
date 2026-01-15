Utility Examples
=================

Python examples for utility functions and helper classes.

Vector Operations
-----------------

Working with vector types:

.. code-block:: python

   import cloudViewer as cv3d
   import numpy as np
   
   # Create vector types
   vec2d = cv3d.utility.Vector2dVector(np.array([[1.0, 2.0], [3.0, 4.0]]))
   vec3d = cv3d.utility.Vector3dVector(np.array([[1.0, 2.0, 3.0], 
                                                   [4.0, 5.0, 6.0]]))
   vec3i = cv3d.utility.Vector3iVector(np.array([[1, 2, 3], [4, 5, 6]]))
   
   # Convert to numpy array
   arr = np.asarray(vec3d)

Matrix Operations
-----------------

Working with transformation matrices:

.. code-block:: python

   import cloudViewer as cv3d
   import numpy as np
   
   # Create 4x4 transformation matrix
   matrix = cv3d.utility.Matrix4dVector([
       np.eye(4),  # Identity matrix
       np.eye(4)   # Another identity matrix
   ])
   
   # Access matrices
   for mat in matrix:
       print(np.asarray(mat))

Scalar Fields
-------------

Working with scalar fields:

.. code-block:: python

   import cloudViewer as cv3d
   import numpy as np
   
   # Create scalar field
   values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
   scalar_field = cv3d.utility.ccScalarField("intensity", values)
   
   # Access values
   print(scalar_field.as_array())

Verbosity Control
-----------------

Controlling output verbosity:

.. code-block:: python

   import cloudViewer as cv3d
   
   # Set verbosity level
   cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)
   
   # Get current verbosity
   level = cv3d.utility.get_verbosity_level()
   
   # Use context manager for temporary verbosity
   with cv3d.utility.VerbosityContextManager(
           cv3d.utility.VerbosityLevel.Error):
       # Only errors will be printed here
       pass

Example Code
------------

For complete examples, see:

**Vector Operations:**
- `vector.py <../../../examples/Python/utility/vector.py>`_

See Also
--------

- :doc:`../../python_api/cloudViewer.utility` - Utility API Reference
- :doc:`../../tutorial/geometry/index` - Geometry Tutorials

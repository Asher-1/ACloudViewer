import pycc
import os

CC = pycc.GetInstance()

params = pycc.FileIOFilter.LoadParameters()
params.parentWidget = CC.getMainWindow()

# Use the test data file shipped with the plugin
_script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(_script_dir, "..", "tests", "data", "a_cloud.bin")
path = os.path.normpath(path)


# This automatically adds the loaded entities
# to the DB on success, and raises an exception if an error occured
obj = CC.loadFile(path, params)


# This does not add to the DB, and returns None if the loading failed
obj = pycc.FileIOFilter.LoadFromFile(path, params)

if obj is not None:
    print("Success to load the file")
    CC.addToDB(obj)

params = pycc.FileIOFilter.SaveParameters()
result = pycc.FileIOFilter.SaveToFile(obj, r'savedFromPlugin.bin', params)
result = pycc.FileIOFilter.SaveToFile(obj.getChild(0), r'savedFromPlugin.laz', params)

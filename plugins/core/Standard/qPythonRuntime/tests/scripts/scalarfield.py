import cccorelib
import pycc
import numpy as np

cc = pycc.GetInstance()
cloud = cc.clouds()[0].pc

# Independent a_cloud.bin Intensity totals: 10683 values, sum 87645995,
# squared sum 993988375593. Round using ScalarType, not the tested mean.
expected_mean = float(np.asarray(87645995 / 10683, dtype=pycc.ScalarType))
expected_mean_square = float(np.asarray(993988375593 / 10683, dtype=pycc.ScalarType))
assert cccorelib.ScalarFieldTools.computeMeanScalarValue(cloud) == expected_mean
assert cccorelib.ScalarFieldTools.computeMeanSquareScalarValue(cloud) == expected_mean_square
assert cccorelib.ScalarFieldTools.computeScalarFieldGradient(cloud, 0, False) == 0

assert cccorelib.ScalarFieldTools.computeScalarFieldExtremas(cloud) == (0, 37522.00)
assert cccorelib.ScalarFieldTools.countScalarFieldValidValues(cloud) == 10_683


classificationSf = cloud.getScalarField(cloud.getScalarFieldIndexByName("Classification"))
assert classificationSf is not None
assert classificationSf.getName() == "Classification"

classificationSf.setName("classification")
assert classificationSf.getName() == "classification"

classificationSf.computeMinAndMax()

assert classificationSf.getMax() == 11
assert classificationSf.getMin() == 11

classificationSf.fill(0)

classificationSf.computeMinAndMax()

assert classificationSf.getMax() == 0
assert classificationSf.getMin() == 0


classificationArray = classificationSf.asArray()
classificationArray[:] = 17

classificationSf.computeMinAndMax()

assert classificationSf.getMax() == 17
assert classificationSf.getMin() == 17

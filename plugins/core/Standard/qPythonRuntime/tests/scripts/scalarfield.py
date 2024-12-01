import cvcorelib
import pycc

cc = pycc.GetInstance()
cloud = cc.clouds()[0].pc

assert cvcorelib.ScalarFieldTools.computeMeanScalarValue(cloud) == 8204.2490234375
assert cvcorelib.ScalarFieldTools.computeMeanSquareScalarValue(cloud) == 93043936.0
assert cvcorelib.ScalarFieldTools.computeScalarFieldGradient(cloud, 0, False) == 0

assert cvcorelib.ScalarFieldTools.computeScalarFieldExtremas(cloud) == (0, 37522.00)
assert cvcorelib.ScalarFieldTools.countScalarFieldValidValues(cloud) == 10_683


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

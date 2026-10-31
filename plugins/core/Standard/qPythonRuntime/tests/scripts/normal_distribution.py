import cccorelib
import numpy as np
import pycc

d = cccorelib.NormalDistribution(1.0, 0.5)

mu, sigma2 = d.getParameters()
assert mu == 1.0 == d.getMu()
assert sigma2 == 0.5 == d.getSigma2()

d.setParameters(0.0, 0.0)

mu, sigma2 = d.getParameters()
assert mu == 0.0 == d.getMu()
assert sigma2 == 0.0 == d.getSigma2()

CC = pycc.GetCmdLineInstance()
cloud = CC.clouds()[0].pc

d.computeParameters(cloud)

# Identify the core ScalarType using a separate, real scalar input boundary.
# Do not derive expectations from the distribution or toArray's fixed double dtype.
scalar_probe = cccorelib.ScalarField("ScalarType probe")
scalar_probe.resize(1, 1.1)
stored = scalar_probe.getValue(0)
assert stored in (1.1, float(np.float32(1.1))), "Unsupported ScalarType conversion"
scalar_type = np.float64 if stored == 1.1 else np.float32
expected_mu = float(scalar_type(1.1))
expected_sigma2 = float(scalar_type(0.6))

d.setParameters(1.1, 0.6)
assert d.getMu() == expected_mu
assert d.getSigma2() == expected_sigma2
assert d.getParameters() == (expected_mu, expected_sigma2)

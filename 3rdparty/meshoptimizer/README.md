# meshoptimizer

This directory vendors the meshoptimizer 1.2 implementation subset used by
TRELLIS and the reconstruction mesh cleanup pipeline. The source is distributed
under the MIT license; each source file retains its SPDX header. It is built as
the `3rdparty_meshoptimizer` CMake target so AICore and Reconstruction share
one implementation on every supported platform.

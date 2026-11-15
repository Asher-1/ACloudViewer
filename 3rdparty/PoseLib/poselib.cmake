include(FetchContent)

# Keep the exact revision pinned by COLMAP (a2bd15c2 src/thirdparty/CMakeLists.txt,
# bumped in d7ffb9c4): it carries the gen_relpose_5p1pt gamma/cheirality fix
# (PoseLib#214) that the GR5P1P fast path of the generalized relative pose
# estimator relies on. PoseLib is populated by default with reconstruction so
# the minimal solvers share the same pinned dependency as COLMAP.
FetchContent_Declare(
    poselib
    URL https://github.com/PoseLib/PoseLib/archive/1bb8881f08e68f0b3d9b2ecc411fc15e7766ef1b.zip
    URL_HASH SHA256=ee46322581c7901fe24e91434eeec27163f820d481cba5faa43d37651fabc8e8
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/poselib"
)

set(MARCH_NATIVE OFF CACHE BOOL "" FORCE)
set(POSELIB_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# Upstream parity (d3ccaf35): upstream builds its fetched PoseLib with the
# compiler defaults — no SSE/AVX512 instruction-set toggles and no Eigen
# alignment overrides. The fork-wide SIMD flags (-msse4.2 -mfpmath=sse
# -mno-avx512*) and the PCL-compatibility macro EIGEN_MAX_ALIGN_BYTES=0 both
# change PoseLib's Eigen codegen; on relpose_6pt_shared_focal the result is a
# last-bit-different essential matrix / focal (observed 1e-4-relative
# divergence, 3296.78 vs 3293.38 px) which chaotically diverges RANSAC trial
# counts and two-view-geometry inlier sets. Strip both for this subproject
# only and restore them afterwards; PoseLib owns no headers shared with the
# rest of the build, so the codegen change cannot leak across targets.
string(REPLACE " -msse4.2" "" POSELIB_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -mfpmath=sse" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512f" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512cd" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512vl" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512bw" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512dq" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512ifma" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512pf" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512er" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512vbmi2" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512vbmi" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512vnni" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512bitalg" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
string(REPLACE " -mno-avx512vpopcntdq" "" POSELIB_CXX_FLAGS "${POSELIB_CXX_FLAGS}")
set(POSELIB_CXX_FLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${POSELIB_CXX_FLAGS}")

get_directory_property(POSELIB_DIR_DEFS COMPILE_DEFINITIONS)
set(POSELIB_FILTERED_DEFS "${POSELIB_DIR_DEFS}")
list(REMOVE_ITEM POSELIB_FILTERED_DEFS "EIGEN_MAX_ALIGN_BYTES=0")
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS "${POSELIB_FILTERED_DEFS}")

FetchContent_MakeAvailable(poselib)

set(CMAKE_CXX_FLAGS "${POSELIB_CXX_FLAGS_BACKUP}")
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS "${POSELIB_DIR_DEFS}")

include(ExternalProject)

# Retry-friendly download for CUTLASS. The GitHub release tarball can be
# flaky in CI/Docker environments (transient HTTP 22 / timeout).
# TLS_VERIFY OFF works around host cert issues in some Docker base images;
# the SHA256 hash below still guarantees integrity on the downloaded content.
# NOTE: ExternalProject_Add does NOT implement retry natively — if the
# download fails the entire CMake configure step errors out.
ExternalProject_Add(ext_cutlass
    PREFIX cutlass
    URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v1.3.3.tar.gz
    URL_HASH SHA256=12d5b4c913063625154019b0a03a253c5b9339c969939454b81f6baaf82b34ca
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/cutlass"
    DOWNLOAD_NO_PROGRESS 1
    TLS_VERIFY OFF
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_cutlass SOURCE_DIR)
set(CUTLASS_INCLUDE_DIRS ${SOURCE_DIR}/) # "/" is critical.

# include(FetchContent)

# FetchContent_Declare(
#     ext_pybind11
#     PREFIX pybind11
#     URL https://github.com/pybind/pybind11/archive/refs/tags/v2.6.2.tar.gz
#     URL_HASH SHA256=8ff2fff22df038f5cd02cea8af56622bc67f5b64534f1b83b9f133b8366acff2
#     DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/pybind11"
# )

# FetchContent_MakeAvailable(ext_pybind11)

include(FetchContent)

FetchContent_Declare(
    ext_pybind11
    PREFIX pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.1.tar.gz
    URL_HASH SHA256=51631e88960a8856f9c497027f55c9f2f9115cafb08c0005439838a05ba17bfc
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/pybind11"
)

FetchContent_MakeAvailable(ext_pybind11)

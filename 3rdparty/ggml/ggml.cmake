# Existing production ggml source owner. Backend and install behavior is shared
# with the llama.cpp owner so there is one implementation of that contract.
set(GGML_VERSION "0.21.0")
set(GGML_URL "https://github.com/ggml-org/ggml/archive/refs/tags/v${GGML_VERSION}.tar.gz")
set(GGML_SHA256 "3b0d4f1fe7c278824d4bb753b7402733576985689bd40e9cc719eca627131d24")
set(_GGML_OWNER_TARGET ext_ggml)
set(_GGML_OWNER_PREFIX ggml)
set(_GGML_OWNER_SUBDIR "")
set(_GGML_OWNER_PATCH_DIR "${CMAKE_CURRENT_LIST_DIR}/../llama/ggml-patches")
set(_GGML_OWNER_PATCH_MANIFEST "${_GGML_OWNER_PATCH_DIR}/manifest.yaml")
set(_GGML_OWNER_CMAKE_ARGS "")
set(_GGML_OWNER_BUILD_BYPRODUCTS "")
include("${CMAKE_CURRENT_LIST_DIR}/../llama/llama_runtime.cmake")

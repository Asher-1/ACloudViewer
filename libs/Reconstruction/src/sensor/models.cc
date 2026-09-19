// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "sensor/models.h"

#include "util/hash_containers.h"

namespace colmap {

bool ExistsCameraModelWithName(const std::string& model_name) {
  return CameraModelNameToId(model_name) != CameraModelId::kInvalid;
}

bool ExistsCameraModelWithId(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) case CameraModel::model_id:
    CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
    return true;
    default:
      return false;
  }
}

CameraModelId CameraModelNameToId(const std::string& model_name) {
  // Function-local static: built once on first use (thread-safe), keeping
  // the name lookup O(1) without paying for eager static initialization.
  static const NodeHashMap<std::string, CameraModelId> kNameToId = [] {
    NodeHashMap<std::string, CameraModelId> name_to_id;
#define CAMERA_MODEL_CASE(CameraModel) \
  name_to_id.emplace(CameraModel::model_name, CameraModel::model_id);

    CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE

    return name_to_id;
  }();

  const auto it = kNameToId.find(model_name);
  if (it == kNameToId.end()) {
    return CameraModelId::kInvalid;
  } else {
    return it->second;
  }
}

const std::string& CameraModelIdToName(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return CameraModel::model_name;

    CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
    default:
      break;
  }

  const static std::string kEmptyModelName = "";
  return kEmptyModelName;
}

std::vector<double> CameraModelInitializeParams(const CameraModelId model_id,
                                                const double focal_length,
                                                const size_t width,
                                                const size_t height) {
  // Assuming that image measurements are within [0, dim], i.e. that the
  // upper left corner is the (0, 0) coordinate (rather than the center of
  // the upper left pixel). This complies with the default SiftGPU convention.
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::model_id:                                          \
    return CameraModel::InitializeParams(focal_length, width, height); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

const std::string& CameraModelParamsInfo(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return CameraModel::params_info;   \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  const static std::string kEmptyParamsInfo = "";
  return kEmptyParamsInfo;
}

// Fork adaptation: upstream returns span<const size_t> over the constexpr
// std::array groups; the fork has no span type and hands out references to
// per-model vector copies of those arrays (built once, thread-safe).
const std::vector<size_t>& CameraModelFocalLengthIdxs(
        const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                          \
  case CameraModel::model_id: {                                 \
    static const std::vector<size_t> kIdxs(                     \
        CameraModel::focal_length_idxs.begin(),                 \
        CameraModel::focal_length_idxs.end());                  \
    return kIdxs;                                               \
  }                                                             \
  break;

    PERSPECTIVE_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    break;
    SPHERICAL_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
    default:
      CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION
  }

  const static std::vector<size_t> kEmptyIdxs;
  return kEmptyIdxs;
}

const std::vector<size_t>& CameraModelPrincipalPointIdxs(
        const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                          \
  case CameraModel::model_id: {                                 \
    static const std::vector<size_t> kIdxs(                     \
        CameraModel::principal_point_idxs.begin(),              \
        CameraModel::principal_point_idxs.end());               \
    return kIdxs;                                               \
  }                                                             \
  break;

    PERSPECTIVE_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    break;
    SPHERICAL_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
    default:
      CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION
  }

  const static std::vector<size_t> kEmptyIdxs;
  return kEmptyIdxs;
}

const std::vector<size_t>& CameraModelExtraParamsIdxs(
        const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                          \
  case CameraModel::model_id: {                                 \
    static const std::vector<size_t> kIdxs(                     \
        CameraModel::extra_params_idxs.begin(),                 \
        CameraModel::extra_params_idxs.end());                  \
    return kIdxs;                                               \
  }                                                             \
  break;

    PERSPECTIVE_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    break;
    SPHERICAL_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
    default:
      CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION
  }

  const static std::vector<size_t> kEmptyIdxs;
  return kEmptyIdxs;
}

const std::vector<size_t>& CameraModelMetaDataParamsIdxs(
        const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                    \
  case CameraModel::model_id: {                           \
    static const std::vector<size_t> kIdxs(               \
        CameraModel::metadata_idxs.begin(),               \
        CameraModel::metadata_idxs.end());                \
    return kIdxs;                                         \
  }                                                       \
  break;

    SPHERICAL_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    break;
    PERSPECTIVE_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
    default:
      CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION
  }

  const static std::vector<size_t> kEmptyIdxs;
  return kEmptyIdxs;
}

size_t CameraModelNumParams(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return CameraModel::num_params;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return 0;
}

bool CameraModelVerifyParams(const CameraModelId model_id,
                             const std::vector<double>& params) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)              \
  case CameraModel::model_id:                       \
    if (params.size() == CameraModel::num_params) { \
      return true;                                  \
    }                                               \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return false;
}

bool CameraModelHasBogusParams(const CameraModelId model_id,
                               const std::vector<double>& params,
                               const size_t width,
                               const size_t height,
                               const double min_focal_length_ratio,
                               const double max_focal_length_ratio,
                               const double max_extra_param) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                         \
  case CameraModel::model_id:                                  \
    return CameraModel::HasBogusParams(params,                 \
                                       width,                  \
                                       height,                 \
                                       min_focal_length_ratio, \
                                       max_focal_length_ratio, \
                                       max_extra_param);       \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return false;
}

}  // namespace colmap

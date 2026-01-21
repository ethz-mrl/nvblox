/*
Copyright 2024 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include "nvblox/core/types.h"
#include "nvblox/utils/params.h"

namespace nvblox {

/// @brief Defines how emptyness is determined for each block.
/// @details
/// - kStrict: A block is considered NOT empty iff at least one voxel within
///     has been observed (weight > 0) and is within truncation distance.
/// - kBlockWiseMinWeight: A block is considered empty if the sum of all its
///     voxel weights does not reach a minimum threshold.
enum class EmptynessClassifierType { kStrict, kBlockWiseMinWeight };

// Specialization placed here to not pollute types.h
template <>
inline std::string toString(const EmptynessClassifierType& classifier_type) {
  switch (classifier_type) {
    case EmptynessClassifierType::kStrict:
      return "kStrict";
      break;
    case EmptynessClassifierType::kBlockWiseMinWeight:
      return "kBlockWiseMinWeight";
      break;
    default:
      LOG(FATAL) << "Not implemented";
      break;
  }
  return "";
}

const Param<LayerTypeBitMask>::Description
    kEmptySpaceIntegratorLayersToClearParamDesc{
        "empty_space_integrator_layers_to_clear",
        LayerTypeBitMask(LayerType::kTsdf) | LayerType::kEsdf |
            LayerType::kColor | LayerType::kColorMesh |
            LayerType::kFeatureMesh | LayerType::kOccupancy |
            LayerType::kFeature,
        "Layer types that should be cleared by empty space clearing logic."};

constexpr Param<EmptynessClassifierType>::Description
    kEmptynessClassifierTypeParamDesc{
        "emptyness_classifier_type",
        EmptynessClassifierType::kBlockWiseMinWeight,
        "What method to use for determining the emptyness of a block."};

constexpr Param<float>::Description kVoxelWeightThresholdParamDesc{
    "voxel_weight_threshold", 0.5f,
    "Threshold for accumulated voxel weight within a block to still be "
    "considered empty. Used by EmptynessClassifierType::kBlockWiseMinWeight."};

struct EmptySpaceIntegratorParams {
  Param<LayerTypeBitMask> layers_to_clear{
      kEmptySpaceIntegratorLayersToClearParamDesc};
  Param<EmptynessClassifierType> emptyness_classifier_type{
      kEmptynessClassifierTypeParamDesc};
  Param<float> voxel_weight_threshold{kVoxelWeightThresholdParamDesc};
};

}  // namespace nvblox

/*
Copyright 2022 NVIDIA CORPORATION

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
#include "nvblox/map/common_names.h"

namespace nvblox {

/// Template trait to convert layer types to LayerType enum.
template <typename LayerTypeT>
struct LayerTypeFromType {
  // Cause compiler error if not specialized for type.
  static_assert(std::is_same_v<LayerTypeT, void>,
                "LayerTypeFromType not specialized for this layer type.");
};

/// Template trait to convert LayerType enum to layer type.
template <LayerType EnumVal>
struct TypeFromLayerType {
  // Cause compiler error if not specialized for type.
  static_assert(std::is_same_v<EnumVal, void>,
                "TypeFromLayerType not specialized for this layer type.");
};

// Specializations for each type.
template <>
struct LayerTypeFromType<TsdfLayer> {
  static constexpr LayerType value = LayerType::kTsdf;
};

template <>
struct TypeFromLayerType<LayerType::kTsdf> {
  using type = TsdfLayer;
};

template <>
struct LayerTypeFromType<EsdfLayer> {
  static constexpr LayerType value = LayerType::kEsdf;
};

template <>
struct TypeFromLayerType<LayerType::kEsdf> {
  using type = EsdfLayer;
};

template <>
struct LayerTypeFromType<ColorLayer> {
  static constexpr LayerType value = LayerType::kColor;
};

template <>
struct TypeFromLayerType<LayerType::kColor> {
  using type = ColorLayer;
};

template <>
struct LayerTypeFromType<ColorMeshLayer> {
  static constexpr LayerType value = LayerType::kColorMesh;
};

template <>
struct TypeFromLayerType<LayerType::kColorMesh> {
  using type = ColorMeshLayer;
};

template <>
struct LayerTypeFromType<FeatureMeshLayer> {
  static constexpr LayerType value = LayerType::kFeatureMesh;
};

template <>
struct TypeFromLayerType<LayerType::kFeatureMesh> {
  using type = FeatureMeshLayer;
};

template <>
struct LayerTypeFromType<FreespaceLayer> {
  static constexpr LayerType value = LayerType::kFreespace;
};

template <>
struct TypeFromLayerType<LayerType::kFreespace> {
  using type = FreespaceLayer;
};

template <>
struct LayerTypeFromType<OccupancyLayer> {
  static constexpr LayerType value = LayerType::kOccupancy;
};

template <>
struct TypeFromLayerType<LayerType::kOccupancy> {
  using type = OccupancyLayer;
};

template <>
struct LayerTypeFromType<FeatureLayer> {
  static constexpr LayerType value = LayerType::kFeature;
};

template <>
struct TypeFromLayerType<LayerType::kFeature> {
  using type = FeatureLayer;
};

template <>
struct LayerTypeFromType<EmptySpaceLayer> {
  static constexpr LayerType value = LayerType::kEmptySpace;
};

template <>
struct TypeFromLayerType<LayerType::kEmptySpace> {
  using type = EmptySpaceLayer;
};

/// Helper function to get LayerType from a layer type.
template <typename LayerTypeT>
constexpr LayerType getLayerType() {
  return LayerTypeFromType<LayerTypeT>::value;
}

/// Helper alias for cleaner access.
template <LayerType EnumVal>
using TypeFromLayerType_t = typename TypeFromLayerType<EnumVal>::type;

}  // namespace nvblox
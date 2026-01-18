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

const Param<LayerTypeBitMask>::Description
    kEmptySpaceIntegratorLayersToClearParamDesc{
        "empty_space_integrator_layers_to_clear",
        // TODO(@bmicha) check which ones we should actually clear.
        LayerTypeBitMask(LayerType::kTsdf) | LayerType::kEsdf |
            LayerType::kColor | LayerType::kColorMesh |
            LayerType::kFeatureMesh | LayerType::kFreespace |
            LayerType::kOccupancy | LayerType::kFeature,
        "Layer types that should be cleared by empty space clearing logic."};

struct EmptySpaceIntegratorParams {
  Param<LayerTypeBitMask> layers_to_clear{
      kEmptySpaceIntegratorLayersToClearParamDesc};
};

}  // namespace nvblox

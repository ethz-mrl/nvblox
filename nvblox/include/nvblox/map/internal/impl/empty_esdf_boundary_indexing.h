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

#include "nvblox/map/blox.h"

namespace nvblox {

constexpr int kInvalidBoundaryFlatIndex = -1;

inline Index3D getBoundaryVoxelIndexFromFlatIndexOnHost(int flat_index) {
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  constexpr int kNumVoxelsPerFace = kVoxelsPerSide * kVoxelsPerSide;
  constexpr int kNumVoxelsPerStrip = (kVoxelsPerSide - 2) * kVoxelsPerSide;
  constexpr int kNumVoxelsPerInnerFace =
      (kVoxelsPerSide - 2) * (kVoxelsPerSide - 2);

  CHECK(flat_index >= 0);
  CHECK(flat_index < EmptyEsdfBlock::kNumBoundaryVoxels);

  if (flat_index < kNumVoxelsPerFace) {
    return Index3D(0, flat_index / kVoxelsPerSide, flat_index % kVoxelsPerSide);
  }
  flat_index -= kNumVoxelsPerFace;

  if (flat_index < kNumVoxelsPerFace) {
    return Index3D(kVoxelsPerSide - 1, flat_index / kVoxelsPerSide,
                   flat_index % kVoxelsPerSide);
  }
  flat_index -= kNumVoxelsPerFace;

  if (flat_index < kNumVoxelsPerStrip) {
    return Index3D(1 + flat_index / kVoxelsPerSide, 0,
                   flat_index % kVoxelsPerSide);
  }
  flat_index -= kNumVoxelsPerStrip;

  if (flat_index < kNumVoxelsPerStrip) {
    return Index3D(1 + flat_index / kVoxelsPerSide, kVoxelsPerSide - 1,
                   flat_index % kVoxelsPerSide);
  }
  flat_index -= kNumVoxelsPerStrip;

  if (flat_index < kNumVoxelsPerInnerFace) {
    return Index3D(1 + flat_index / (kVoxelsPerSide - 2),
                   1 + flat_index % (kVoxelsPerSide - 2), 0);
  }
  flat_index -= kNumVoxelsPerInnerFace;

  return Index3D(1 + flat_index / (kVoxelsPerSide - 2),
                 1 + flat_index % (kVoxelsPerSide - 2), kVoxelsPerSide - 1);
}

__device__ __constant__ int
    kBoundaryVoxelLut[EmptyEsdfBlock::kNumBoundaryVoxels][3];
__device__ __constant__ int
    kBoundaryFlatIndexLut[VoxelBlock<bool>::kVoxelsPerSide]
                         [VoxelBlock<bool>::kVoxelsPerSide]
                         [VoxelBlock<bool>::kVoxelsPerSide];

inline void initializeBoundaryVoxelLutOnGPU() {
  static bool initialized = false;
  if (initialized) {
    return;
  }

  int host_lut_from_flat[EmptyEsdfBlock::kNumBoundaryVoxels][3];

  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  int host_lut_to_flat[kVoxelsPerSide][kVoxelsPerSide][kVoxelsPerSide];
  // Fill LUT with invalid sentinel.
  std::fill_n(&host_lut_to_flat[0][0][0],
              kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide,
              kInvalidBoundaryFlatIndex);

  for (int i = 0; i < EmptyEsdfBlock::kNumBoundaryVoxels; ++i) {
    const Index3D voxel_index = getBoundaryVoxelIndexFromFlatIndexOnHost(i);
    host_lut_from_flat[i][0] = voxel_index.x();
    host_lut_from_flat[i][1] = voxel_index.y();
    host_lut_from_flat[i][2] = voxel_index.z();

    host_lut_to_flat[voxel_index.x()][voxel_index.y()][voxel_index.z()] = i;
  }

  checkCudaErrors(cudaMemcpyToSymbol(kBoundaryVoxelLut, host_lut_from_flat,
                                     sizeof(host_lut_from_flat)));

  checkCudaErrors(cudaMemcpyToSymbol(kBoundaryFlatIndexLut, host_lut_to_flat,
                                     sizeof(host_lut_to_flat)));
  initialized = true;
}

__device__ inline Index3D getBoundaryVoxelIndexFromFlatIndex(int flat_index) {
  return Index3D(static_cast<int>(kBoundaryVoxelLut[flat_index][0]),
                 static_cast<int>(kBoundaryVoxelLut[flat_index][1]),
                 static_cast<int>(kBoundaryVoxelLut[flat_index][2]));
}

__device__ inline int getFlatIndexFromBoundaryVoxelIndex(Index3D voxel_index) {
  int idx =
      kBoundaryFlatIndexLut[voxel_index.x()][voxel_index.y()][voxel_index.z()];
  NVBLOX_CHECK(idx != kInvalidBoundaryFlatIndex,
               "Out of bounds boundary voxel flat index.");
  return idx;
}

}  // namespace nvblox

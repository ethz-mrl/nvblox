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

__host__ __device__ inline Index3D getBoundaryVoxelIndexFromFlatIndex(
    int flat_index) {
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  constexpr int kNumVoxelsPerFace = kVoxelsPerSide * kVoxelsPerSide;
  constexpr int kNumVoxelsPerStrip = (kVoxelsPerSide - 2) * kVoxelsPerSide;
  constexpr int kNumVoxelsPerInnerFace =
      (kVoxelsPerSide - 2) * (kVoxelsPerSide - 2);

  NVBLOX_CHECK(flat_index >= 0, "Invalid flat_index.");
  NVBLOX_CHECK(flat_index < EmptyEsdfBlock::kNumBoundaryVoxels,
               "Invalid flat_index.");

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

__host__ __device__ inline int getFlatIndexFromBoundaryVoxelIndex(
    const Index3D& idx) {
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  constexpr int kNumVoxelsPerFace = kVoxelsPerSide * kVoxelsPerSide;
  constexpr int kNumVoxelsPerStrip = (kVoxelsPerSide - 2) * kVoxelsPerSide;
  constexpr int kNumVoxelsPerInnerFace =
      (kVoxelsPerSide - 2) * (kVoxelsPerSide - 2);

  const int x = idx.x(), y = idx.y(), z = idx.z();

  if (x == 0) {
    return y * kVoxelsPerSide + z;
  }
  if (x == kVoxelsPerSide - 1) {
    return kNumVoxelsPerFace + y * kVoxelsPerSide + z;
  }
  if (y == 0) {
    return 2 * kNumVoxelsPerFace + (x - 1) * kVoxelsPerSide + z;
  }
  if (y == kVoxelsPerSide - 1) {
    return 2 * kNumVoxelsPerFace + kNumVoxelsPerStrip +
           (x - 1) * kVoxelsPerSide + z;
  }
  if (z == 0) {
    return 2 * kNumVoxelsPerFace + 2 * kNumVoxelsPerStrip +
           (x - 1) * (kVoxelsPerSide - 2) + (y - 1);
  }
  if (z == kVoxelsPerSide - 1) {
    return 2 * kNumVoxelsPerFace + 2 * kNumVoxelsPerStrip +
           kNumVoxelsPerInnerFace + (x - 1) * (kVoxelsPerSide - 2) + (y - 1);
  }

  return kInvalidBoundaryFlatIndex;
}

__host__ __device__ inline bool isBoundaryVoxelIndex(
    const Index3D& voxel_index) {
  constexpr int kLast = VoxelBlock<bool>::kVoxelsPerSide - 1;
  return voxel_index.x() == 0 || voxel_index.x() == kLast ||
         voxel_index.y() == 0 || voxel_index.y() == kLast ||
         voxel_index.z() == 0 || voxel_index.z() == kLast;
}

}  // namespace nvblox

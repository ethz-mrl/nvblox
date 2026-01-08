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

#include "nvblox/io/ply_writer.h"
#include "nvblox/map/accessors.h"

namespace nvblox {
namespace io {

/// Outputs a voxel layer as a pointcloud with the lambda function deciding the
/// intensity.
template <typename VoxelType>
bool outputVoxelLayerToPly(
    const VoxelBlockLayer<VoxelType>& layer, const std::string& filename,
    std::function<bool(const VoxelType* voxel, float* intensity)> lambda) {
  // Create a ply writer object.
  io::PlyWriter writer(filename);

  // Combine all the voxels in the mesh into a pointcloud.
  std::vector<Vector3f> points;
  std::vector<float> intensities;

  const float block_size = layer.block_size();
  const float voxel_size = layer.voxel_size();

  auto new_lambda = [&points, &intensities, &block_size, &voxel_size, &lambda](
                        const Index3D& block_index, const Index3D& voxel_index,
                        const VoxelType* voxel) {
    float intensity = 0.0f;
    if (lambda(voxel, &intensity)) {
      points.push_back(getCenterPositionFromBlockIndexAndVoxelIndex(
          block_size, block_index, voxel_index));
      intensities.push_back(intensity);
    }
  };

  // Call above lambda on every voxel in the layer.
  callFunctionOnAllVoxels<VoxelType>(layer, new_lambda);

  // Add the pointcloud to the ply writer.
  writer.setPoints(&points);
  writer.setIntensities(&intensities);

  // Write out the ply.
  return writer.write();
}

/// Outputs a block layer as a pointcloud with the lambda function deciding the
/// intensity. Each block will be one point.
template <typename BlockType>
bool outputBlockLayerToPly(
    const BlockLayer<BlockType>& layer, const std::string& filename,
    std::function<bool(const BlockType* block, float* intensity)> lambda) {
  // Create a ply writer objcet.
  io::PlyWriter writer(filename);

  // Combine all the blocks in the mesh into a pointcloud.
  std::vector<Vector3f> points;
  std::vector<float> intensities;

  const float block_size = layer.block_size();

  auto new_lambda = [&points, &intensities, &block_size, &lambda](
                        const Index3D& block_index, const BlockType* block) {
    float intensity = 0.0f;
    if (lambda(block, &intensity)) {
      points.push_back(getCenterPositionFromBlockIndex(
          block_size, block_index));
      intensities.push_back(intensity);
    }
  };

  // Call above lambda on every block in the layer.
  callFunctionOnAllBlocks<BlockType>(layer, new_lambda);

  // Add the pointcloud to the ply writer.
  writer.setPoints(&points);
  writer.setIntensities(&intensities);

  // Write out the ply.
  return writer.write();
}

}  // namespace io
}  // namespace nvblox

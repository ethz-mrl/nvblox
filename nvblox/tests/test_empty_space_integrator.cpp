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
#include <gtest/gtest.h>

#include <cmath>

#include "nvblox/integrators/empty_space_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/weighting_function.h"
#include "nvblox/interpolation/interpolation_3d.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

DECLARE_bool(alsologtostderr);

// Check that
//  1. The right blocks are marked (not) empty.
//  2. All cleared blocks are actually empty.
//  3. All still present blocks are marked not empty.

class EmptySpaceIntegratorTest : public ::testing::Test {
 protected:
  EmptySpaceIntegratorTest()
      : layer_(voxel_size_m_, MemoryType::kUnified),
        camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

  // Test layer
  constexpr static float voxel_size_m_ = 0.2;
  TsdfLayer layer_;

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;
};

TEST_F(EmptySpaceIntegratorTest, SphereSceneTest) {
  constexpr float kTrajectoryRadius = 4.0f;
  constexpr float kTrajectoryHeight = 2.0f;
  constexpr int kNumTrajectoryPoints = 80;
  constexpr float kTruncationDistanceVox = 2;
  constexpr float kTruncationDistanceMeters =
      kTruncationDistanceVox * voxel_size_m_;
  // Maximum distance to consider for scene generation.
  constexpr float kMaxDist = 10.0;

  // Get the ground truth SDF of a sphere in a box.
  primitives::Scene scene = test_utils::getSphereInBox();
  TsdfLayer gt_layer(voxel_size_m_, MemoryType::kUnified);
  scene.generateLayerFromScene(kTruncationDistanceMeters, &gt_layer);

  // Create integrators.
  EmptySpaceIntegrator empty_space_integrator;
  ProjectiveTsdfIntegrator tsdf_integrator_baseline;
  ProjectiveTsdfIntegrator tsdf_integrator_clearing;
  tsdf_integrator_baseline.truncation_distance_vox(kTruncationDistanceVox);
  tsdf_integrator_clearing.truncation_distance_vox(kTruncationDistanceVox);

  // Create layers.
  EmptySpaceLayer empty_space_layer(layer_.voxel_size(), MemoryType::kUnified);

  // Two layers, one for baseline integration and one for clearing integration.
  TsdfLayer tsdf_layer_baseline(layer_.voxel_size(), MemoryType::kUnified);
  TsdfLayer tsdf_layer_clearing(layer_.voxel_size(), MemoryType::kUnified);

  // Create a depth frame.
  // We share this memory buffer for the entire trajectory.
  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);

  // Simulate trajectory of points, on a circle around the sphere.
  const float radians_increment = 2 * M_PI / (kNumTrajectoryPoints);

  // Integrate depth frames into layers.
  for (size_t i = 0; i < kNumTrajectoryPoints; i++) {
    const float theta = radians_increment * i;
    // Convert polar to cartesian coordinates.
    Vector3f cartesian_coordinates(kTrajectoryRadius * std::cos(theta),
                                   kTrajectoryRadius * std::sin(theta),
                                   kTrajectoryHeight);
    // The camera has its z axis pointing towards the origin.
    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaternionf rotation_theta(
        Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

    // Construct a transform from camera to scene with this.
    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_theta * rotation_base);
    T_S_C.pretranslate(cartesian_coordinates);

    // Generate a depth image of the scene.
    scene.generateDepthImageFromScene(camera_, T_S_C, kMaxDist, &depth_frame);

    // Integrate this depth image.
    tsdf_integrator_baseline.integrateFrame(
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        camera_, &tsdf_layer_baseline);
    tsdf_integrator_clearing.integrateFrame(
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        camera_, &tsdf_layer_clearing);
  }

  // Mark empty blocks as such in empty space layer.
  empty_space_integrator.updateEmptySpaceLayer(
      tsdf_layer_clearing.getAllBlockIndices(), tsdf_layer_clearing,
      &empty_space_layer, kTruncationDistanceMeters);

  // Check the kernel logic of marking blocks as empty before we clear the
  // layer.
  // Check the empty flag for each block in the empty space layer.
  bool all_voxels_are_empty;
  for (const Index3D& block_index : tsdf_layer_clearing.getAllBlockIndices()) {
    // Reset flag.
    all_voxels_are_empty = true;

    const auto empty_space_block =
        empty_space_layer.getBlockAtIndex(block_index);

    const auto tsdf_clearing_layer_block =
        tsdf_layer_clearing.getBlockAtIndex(block_index);

    for (int x = 0; x < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; x++) {
      for (int y = 0; y < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; y++) {
        for (int z = 0; z < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; z++) {
          const TsdfVoxel tsdf_voxel =
              tsdf_clearing_layer_block->voxels[x][y][z];
          // Voxel has been observed and is occupied.
          if (tsdf_voxel.weight > 0.0f &&
              kTruncationDistanceMeters > tsdf_voxel.distance) {
            all_voxels_are_empty = false;
          }
        }
      }
    }
    if (empty_space_block->is_empty) {
      EXPECT_TRUE(all_voxels_are_empty);
    } else {
      EXPECT_FALSE(all_voxels_are_empty);
    }
  }

  // Clear empty blocks in tsdf layer.
  empty_space_integrator.clearEmptyBlocksFromLayer(&tsdf_layer_clearing,
                                                   &empty_space_layer);

  // Check that all empty blocks have been removed and that all non-empty blocks
  // have not been removed.
  for (const Index3D& block_index : tsdf_layer_baseline.getAllBlockIndices()) {
    const auto cleared_layer_block =
        tsdf_layer_clearing.getBlockAtIndex(block_index);

    const auto empty_space_block =
        empty_space_layer.getBlockAtIndex(block_index);

    // Check blocks that have been cleared / no longer exist.
    if (!cleared_layer_block) {
      // We did not clear blocks in this layer...
      ASSERT_FALSE(!empty_space_block);
      // ... and the block should be flagged empty.
      EXPECT_TRUE(empty_space_block->is_empty);
    } else {
      // We did not clear blocks in this layer...
      ASSERT_FALSE(!empty_space_block);
      // ... and the block should be flagged NOT empty.
      EXPECT_FALSE(empty_space_block->is_empty);
    }
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
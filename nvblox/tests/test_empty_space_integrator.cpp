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

#include "nvblox/core/layer_type_traits.h"
#include "nvblox/integrators/empty_space_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/weighting_function.h"
#include "nvblox/interpolation/interpolation_3d.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

DECLARE_bool(alsologtostderr);

class EmptySpaceIntegratorTest : public ::testing::Test {
 protected:
  EmptySpaceIntegratorTest()
      : camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)),
        depth_frame_(DepthImage(camera_.height(), camera_.width(),
                                MemoryType::kUnified)) {}

  // Voxel size
  constexpr static float voxel_size_m_ = 0.2;

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;

  // Depth frame
  // We can share this memory buffer for the entire trajectory.
  DepthImage depth_frame_;
};

class EmptySpaceIntegratorTestSphereScene : public EmptySpaceIntegratorTest {
 protected:
  EmptySpaceIntegratorTestSphereScene()
      : scene_(test_utils::getSphereInBox()),
        gt_tsdf_layer_(TsdfLayer{voxel_size_m_, MemoryType::kUnified}) {
    // Generate groundtruth tsdf layer from scene.
    scene_.generateLayerFromScene(kTruncationDistanceMeters_, &gt_tsdf_layer_);
  }
  constexpr static float kTrajectoryRadius_ = 4.0f;
  constexpr static float kTrajectoryHeight_ = 2.0f;
  constexpr static int kNumTrajectoryPoints_ = 80;
  constexpr static float kTruncationDistanceVox_ = 2;
  constexpr static float kTruncationDistanceMeters_ =
      kTruncationDistanceVox_ * voxel_size_m_;
  // Maximum distance to consider for scene generation.
  constexpr static float kMaxDist_ = 10.0;

  // Simulate trajectory of points, on a circle around the sphere.
  const float radians_increment_ = 2 * M_PI / (kNumTrajectoryPoints_);

  // Ground truth scene and tsdf layer of sphere in a box.
  primitives::Scene scene_;
  TsdfLayer gt_tsdf_layer_;
};

// Check bare integrator functionality.
//  1. The right blocks are marked (not) empty.
//  2. All cleared blocks are actually empty.
//  3. All still present blocks are marked not empty.
TEST_F(EmptySpaceIntegratorTestSphereScene, StandaloneIntegrator) {
  // Create integrators.
  EmptySpaceIntegrator empty_space_integrator;
  ProjectiveTsdfIntegrator tsdf_integrator_baseline;
  ProjectiveTsdfIntegrator tsdf_integrator_clearing;
  tsdf_integrator_baseline.truncation_distance_vox(kTruncationDistanceVox_);
  tsdf_integrator_clearing.truncation_distance_vox(kTruncationDistanceVox_);

  // Create layers.
  EmptySpaceLayer empty_space_layer(voxel_size_m_, MemoryType::kUnified);

  // Two layers, one for baseline integration and one for clearing integration.
  TsdfLayer tsdf_layer_baseline(voxel_size_m_, MemoryType::kUnified);
  TsdfLayer tsdf_layer_clearing(voxel_size_m_, MemoryType::kUnified);

  // Integrate depth frames into layers.
  for (size_t i = 0; i < kNumTrajectoryPoints_; i++) {
    const float theta = radians_increment_ * i;
    // Convert polar to cartesian coordinates.
    Vector3f cartesian_coordinates(kTrajectoryRadius_ * std::cos(theta),
                                   kTrajectoryRadius_ * std::sin(theta),
                                   kTrajectoryHeight_);
    // The camera has its z axis pointing towards the origin.
    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaternionf rotation_theta(
        Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

    // Construct a transform from camera to scene with this.
    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_theta * rotation_base);
    T_S_C.pretranslate(cartesian_coordinates);

    // Generate a depth image of the scene.
    scene_.generateDepthImageFromScene(camera_, T_S_C, kMaxDist_,
                                       &depth_frame_);

    // Integrate this depth image.
    tsdf_integrator_baseline.integrateFrame(
        MaskedDepthImageConstView(depth_frame_, kMaskActiveEverywhere), T_S_C,
        camera_, &tsdf_layer_baseline);
    tsdf_integrator_clearing.integrateFrame(
        MaskedDepthImageConstView(depth_frame_, kMaskActiveEverywhere), T_S_C,
        camera_, &tsdf_layer_clearing);
  }

  // Mark empty blocks as such in empty space layer.
  empty_space_integrator.updateEmptySpaceLayer(
      tsdf_layer_clearing.getAllBlockIndices(), tsdf_layer_clearing,
      &empty_space_layer, kTruncationDistanceVox_);

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
              kTruncationDistanceMeters_ > tsdf_voxel.distance) {
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

// Check integrator functionality within mapper with intermediate empty space
// layer updates.
//  1. The right blocks are marked (not) empty.
//  2. All cleared blocks are actually empty.
//  3. All still present blocks are marked not empty.
TEST_F(EmptySpaceIntegratorTestSphereScene, MapperTest) {
  // Create baseline tsdf layer and integrator.
  TsdfLayer tsdf_layer_baseline =
      TsdfLayer(voxel_size_m_, MemoryType::kUnified);
  ProjectiveTsdfIntegrator tsdf_integrator_baseline;
  tsdf_integrator_baseline.truncation_distance_vox(kTruncationDistanceVox_);

  Mapper mapper(voxel_size_m_, MemoryType::kUnified,
                ProjectiveLayerType::kTsdfWithEmptySpace);
  mapper.tsdf_integrator().truncation_distance_vox(kTruncationDistanceVox_);

  constexpr int emptySpaceLayerUpdateInterval = kNumTrajectoryPoints_ / 4;
  // Integrate depth frames into layers.
  for (size_t i = 0; i < kNumTrajectoryPoints_; i++) {
    const float theta = radians_increment_ * i;
    // Convert polar to cartesian coordinates.
    Vector3f cartesian_coordinates(kTrajectoryRadius_ * std::cos(theta),
                                   kTrajectoryRadius_ * std::sin(theta),
                                   kTrajectoryHeight_);
    // The camera has its z axis pointing towards the origin.
    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaternionf rotation_theta(
        Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

    // Construct a transform from camera to scene with this.
    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_theta * rotation_base);
    T_S_C.pretranslate(cartesian_coordinates);

    // Generate a depth image of the scene.
    scene_.generateDepthImageFromScene(camera_, T_S_C, kMaxDist_,
                                       &depth_frame_);

    // Integrate depth image into baseline at every step.
    tsdf_integrator_baseline.integrateFrame(
        MaskedDepthImageConstView(depth_frame_, kMaskActiveEverywhere), T_S_C,
        camera_, &tsdf_layer_baseline);

    // Integrate this depth image.
    mapper.integrateDepth(
        MaskedDepthImageConstView(depth_frame_, kMaskActiveEverywhere), T_S_C,
        camera_);

    // Update empty space layer after emptySpaceLayerUpdateInterval frames.
    if (i && i % emptySpaceLayerUpdateInterval == 0) {
      mapper.updateEmptySpace();
      mapper.clearEmptySpaceBlocksInLayers(UpdateFullLayer::kYes);
    }
  }
  // Also include the potentially off-by-one frames.
  mapper.updateEmptySpace();
  mapper.clearEmptySpaceBlocksInLayers(UpdateFullLayer::kYes);

  // Check the empty flag with the values in the baseline tsdf layer for each
  // block in the empty space layer.
  bool all_voxels_are_empty;
  for (const Index3D& block_index : tsdf_layer_baseline.getAllBlockIndices()) {
    // Reset flag.
    all_voxels_are_empty = true;

    const auto empty_space_block =
        mapper.empty_space_layer().getBlockAtIndex(block_index);

    const auto tsdf_block = tsdf_layer_baseline.getBlockAtIndex(block_index);

    for (int x = 0; x < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; x++) {
      for (int y = 0; y < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; y++) {
        for (int z = 0; z < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; z++) {
          const TsdfVoxel tsdf_voxel = tsdf_block->voxels[x][y][z];
          // Voxel has been observed and is occupied.
          if (tsdf_voxel.weight > 0.0f &&
              kTruncationDistanceMeters_ > tsdf_voxel.distance) {
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

  // Check that all empty blocks have been removed and that all non-empty blocks
  // have not been removed.
  for (const Index3D& block_index : tsdf_layer_baseline.getAllBlockIndices()) {
    const auto cleared_layer_block =
        mapper.tsdf_layer().getBlockAtIndex(block_index);

    const auto empty_space_block =
        mapper.empty_space_layer().getBlockAtIndex(block_index);

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

// A wrapper that exposes the protected members for testing.
class MapperTestWrapper : public Mapper {
 public:
  // Inherit the constructors
  using Mapper::Mapper;

  // Expose protected layers_ pointer publicly.
  const LayerCake& layers() const { return layers_; }
  LayerCake& layers() { return layers_; }
};

template <typename T>
class EmptySpaceIntegratorMapperClearing
    : public EmptySpaceIntegratorTestSphereScene {
 protected:
  MapperTestWrapper mapper_;

 public:
  EmptySpaceIntegratorMapperClearing()
      : mapper_(voxel_size_m_, MemoryType::kUnified,
                ProjectiveLayerType::kTsdfWithEmptySpace) {
    // Only passed layer type will be cleared.
    LayerTypeBitMask layers_to_clear_mask{getLayerType<T>()};
    mapper_.empty_space_integrator().layers_to_clear(layers_to_clear_mask);
    mapper_.tsdf_integrator().truncation_distance_vox(kTruncationDistanceVox_);
  }
};

using LayerToTest = ::testing::Types<TsdfLayer, EsdfLayer>;
TYPED_TEST_SUITE(EmptySpaceIntegratorMapperClearing, LayerToTest);

// Check selective layer clearing logic for different layer types.
TYPED_TEST(EmptySpaceIntegratorMapperClearing, ClearingSelectedLayer) {
  constexpr int emptySpaceLayerUpdateInterval = this->kNumTrajectoryPoints_ / 4;

  // Integrate depth frames into layers.
  for (size_t i = 0; i < this->kNumTrajectoryPoints_; i++) {
    const float theta = this->radians_increment_ * i;
    // Convert polar to cartesian coordinates.
    Vector3f cartesian_coordinates(this->kTrajectoryRadius_ * std::cos(theta),
                                   this->kTrajectoryRadius_ * std::sin(theta),
                                   this->kTrajectoryHeight_);
    // The camera has its z axis pointing towards the origin.
    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaternionf rotation_theta(
        Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

    // Construct a transform from camera to scene with this.
    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_theta * rotation_base);
    T_S_C.pretranslate(cartesian_coordinates);

    // Generate a depth image of the scene.
    this->scene_.generateDepthImageFromScene(
        this->camera_, T_S_C, this->kMaxDist_, &this->depth_frame_);

    // Integrate this depth image.
    this->mapper_.integrateDepth(
        MaskedDepthImageConstView(this->depth_frame_, kMaskActiveEverywhere),
        T_S_C, this->camera_);

    // Update empty space layer after emptySpaceLayerUpdateInterval frames.
    if ((i == this->kNumTrajectoryPoints_ - 1) ||
        (i && i % emptySpaceLayerUpdateInterval == 0)) {
      this->mapper_.updateEmptySpace();
      this->mapper_.updateEsdf();
      this->mapper_.clearEmptySpaceBlocksInLayers(UpdateFullLayer::kYes);
    }
  }

  // Check that all empty blocks have been removed and that all non-empty blocks
  // have not been removed.
  for (const Index3D& block_index :
       this->mapper_.empty_space_layer().getAllBlockIndices()) {
    const auto cleared_layer_block =
        this->mapper_.layers().template get<TypeParam>().getBlockAtIndex(
            block_index);

    const auto empty_space_block =
        this->mapper_.empty_space_layer().getBlockAtIndex(block_index);

    // Check blocks that have been cleared / no longer exist.
    if (!cleared_layer_block) {
      EXPECT_TRUE(empty_space_block->is_empty);
    } else {
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

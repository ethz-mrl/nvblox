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
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "nvblox/integrators/empty_space_integrator.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

DECLARE_bool(alsologtostderr);

struct EmptySpaceTestConfig {
  EmptynessClassifierType classifier;
  LayerType layer_type_to_clear;

  // Parameters
  float voxel_weight_threshold;
  bool do_empty_space_clearing;
};

std::vector<EmptySpaceTestConfig> test_case_matrix = {
    {EmptynessClassifierType::kStrict, LayerType::kTsdf, 1.0f, true},
    {EmptynessClassifierType::kStrict, LayerType::kTsdf, 1.0f, false},
    {EmptynessClassifierType::kStrict, LayerType::kEsdf, 1.0f, true},
    {EmptynessClassifierType::kStrict, LayerType::kEsdf, 1.0f, false},
    {EmptynessClassifierType::kBlockWiseMinWeight, LayerType::kTsdf, 1.0f,
     true},
    {EmptynessClassifierType::kBlockWiseMinWeight, LayerType::kTsdf, 1.0f,
     false},
    {EmptynessClassifierType::kBlockWiseMinWeight, LayerType::kEsdf, 1.0f,
     true},
    {EmptynessClassifierType::kBlockWiseMinWeight, LayerType::kEsdf, 1.0f,
     false},
    {EmptynessClassifierType::kBlockWiseMinWeight, LayerType::kTsdf, 1.5f,
     true},
    {EmptynessClassifierType::kBlockWiseMinWeight, LayerType::kTsdf, 1.5f,
     false},
    {EmptynessClassifierType::kBlockWiseMinWeight, LayerType::kEsdf, 1.5f,
     true},
    {EmptynessClassifierType::kBlockWiseMinWeight, LayerType::kEsdf, 1.5f,
     false}};

class EmptySpaceParamTest
    : public ::testing::TestWithParam<EmptySpaceTestConfig> {
 protected:
  EmptySpaceParamTest()
      : camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)),
        depth_frame_(DepthImage(camera_.height(), camera_.width(),
                                MemoryType::kUnified)),
        scene_(test_utils::getSphereInBox()),
        gt_tsdf_layer_(TsdfLayer{voxel_size_m_, MemoryType::kUnified}) {
    // Generate groundtruth tsdf layer from scene.
    scene_.generateLayerFromScene(kTruncationDistanceMeters_, &gt_tsdf_layer_);
  }

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

  // Depth frame which we use for every single frame.
  DepthImage depth_frame_;

  // Camera trajectory parameters
  constexpr static float kTrajectoryRadius_ = 4.0f;
  constexpr static float kTrajectoryHeight_ = 2.0f;
  constexpr static int kNumTrajectoryPoints_ = 80;
  const float radians_increment_ = 2 * M_PI / (kNumTrajectoryPoints_);
  constexpr static int kLayerUpdateInterval = kNumTrajectoryPoints_ / 4;

  // General integration parameters
  constexpr static float kMaxDist_ = 10.0;
  constexpr static float kTruncationDistanceVox_ = 2;
  constexpr static float kTruncationDistanceMeters_ =
      kTruncationDistanceVox_ * voxel_size_m_;

  // Ground truth scene and tsdf layer.
  primitives::Scene scene_;
  TsdfLayer gt_tsdf_layer_;
};

void checkEmptyFlagConsistency(const EmptySpaceTestConfig& cfg,
                               const TsdfLayer& tsdf_layer,
                               const EmptySpaceLayer& empty_space_layer,
                               const float truncation_distance_m) {
  // Setup flags and accumulators
  float qualified_voxel_weight_in_block;
  bool all_voxels_empty;

  for (const Index3D& block_index : tsdf_layer.getAllBlockIndices()) {
    // Reset flags and accumulators
    qualified_voxel_weight_in_block = 0.0f;
    all_voxels_empty = true;

    const auto empty_space_block =
        empty_space_layer.getBlockAtIndex(block_index);

    const auto tsdf_block = tsdf_layer.getBlockAtIndex(block_index);

    for (int x = 0; x < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; x++) {
      for (int y = 0; y < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; y++) {
        for (int z = 0; z < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; z++) {
          const TsdfVoxel tsdf_voxel = tsdf_block->voxels[x][y][z];
          // kStrict
          if (tsdf_voxel.weight > 0.0f &&
              tsdf_voxel.distance < truncation_distance_m) {
            all_voxels_empty = false;
          }
          // kBlockWiseMinWeight
          if (std::abs(tsdf_voxel.distance) < truncation_distance_m) {
            qualified_voxel_weight_in_block += tsdf_voxel.weight;
          }
        }
      }
    }

    bool is_block_empty;

    if (cfg.classifier == EmptynessClassifierType::kStrict) {
      is_block_empty = (!!all_voxels_empty);
    } else if (cfg.classifier == EmptynessClassifierType::kBlockWiseMinWeight) {
      is_block_empty =
          (qualified_voxel_weight_in_block <= cfg.voxel_weight_threshold);
    }

    if (empty_space_block->is_empty) {
      EXPECT_TRUE(is_block_empty);
    } else {
      EXPECT_FALSE(is_block_empty);
    }
  }
}

// Check that removed blocks are flagged empty and present blocks
// non-empty.
void checkBlockClearingMatchesFlags(const EmptySpaceTestConfig& cfg,
                                    Mapper& mapper) {
  auto run_for_layer = [&](auto& clearing_layer) {
    for (const Index3D& block_index :
         mapper.empty_space_layer().getAllBlockIndices()) {
      auto cleared_layer_block = clearing_layer.getBlockAtIndex(block_index);
      auto empty_space_block =
          mapper.empty_space_layer().getBlockAtIndex(block_index);

      // No blocks should have been cleared.
      if (!cfg.do_empty_space_clearing) {
        EXPECT_NE(cleared_layer_block, nullptr);
        continue;
      }

      // Check that cleared blocks have empty flag and vice versa.
      if (!cleared_layer_block) {
        EXPECT_TRUE(empty_space_block->is_empty);
      } else {
        EXPECT_FALSE(empty_space_block->is_empty);
      }
    }
  };

  // NOTE(@bmicha) currently we only test with tsdf and esdf.
  // To add more layer types, add integration and update logic to test.
  switch (cfg.layer_type_to_clear) {
    case LayerType::kTsdf:
      return run_for_layer(mapper.tsdf_layer());
    case LayerType::kEsdf:
      return run_for_layer(mapper.esdf_layer());
    default:
      CHECK(false) << "Unsupported layer_type_to_clear";
  }
}

TEST_P(EmptySpaceParamTest, GenericEmptySpaceBehavior) {
  EmptySpaceTestConfig cfg = GetParam();

  // Setup map with test config.
  Mapper mapper{voxel_size_m_, MemoryType::kUnified,
                ProjectiveLayerType::kTsdf};
  mapper.empty_space_integrator().emptyness_classifier_type(cfg.classifier);
  mapper.empty_space_integrator().layers_to_clear(
      LayerTypeBitMask(cfg.layer_type_to_clear));
  mapper.empty_space_integrator().voxel_weight_threshold(
      cfg.voxel_weight_threshold);
  mapper.do_empty_space_clearing(cfg.do_empty_space_clearing);
  mapper.tsdf_integrator().truncation_distance_vox(kTruncationDistanceVox_);

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
    mapper.integrateDepth(
        MaskedDepthImageConstView(depth_frame_, kMaskActiveEverywhere), T_S_C,
        camera_);

    // Update and clear empty space layer after kLayerUpdateInterval
    // frames.
    if ((i == kNumTrajectoryPoints_ - 1) ||
        (i && i % kLayerUpdateInterval == 0)) {
      // Update.
      mapper.updateEmptySpace();
      mapper.updateEsdf();
      checkEmptyFlagConsistency(cfg, mapper.tsdf_layer(),
                                mapper.empty_space_layer(),
                                kTruncationDistanceMeters_);
      // Clear.
      mapper.clearEmptySpaceBlocksInLayers(UpdateFullLayer::kYes);
      checkBlockClearingMatchesFlags(cfg, mapper);
    }
  }
}

INSTANTIATE_TEST_CASE_P(EmptySpaceTestMatrix, EmptySpaceParamTest,
                        ::testing::ValuesIn(test_case_matrix));

// Test parameter setters and getters
TEST(EmptySpaceParameters, ParamSetterAndGetter) {
  Mapper mapper(0.05f, MemoryType::kDevice);

  mapper.do_empty_space_clearing(false);
  EXPECT_FALSE(mapper.do_empty_space_clearing());
  mapper.do_empty_space_clearing(true);
  EXPECT_TRUE(mapper.do_empty_space_clearing());

  mapper.empty_space_integrator().emptyness_classifier_type(
      EmptynessClassifierType::kStrict);
  EXPECT_EQ(mapper.empty_space_integrator().emptyness_classifier_type(),
            EmptynessClassifierType::kStrict);
  mapper.empty_space_integrator().emptyness_classifier_type(
      EmptynessClassifierType::kBlockWiseMinWeight);
  EXPECT_EQ(mapper.empty_space_integrator().emptyness_classifier_type(),
            EmptynessClassifierType::kBlockWiseMinWeight);

  mapper.empty_space_integrator().voxel_weight_threshold(2.f);
  EXPECT_EQ(mapper.empty_space_integrator().voxel_weight_threshold(), 2.f);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#pragma once

#include "nvblox/core/types.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"

namespace nvblox {

class EmptySpaceIntegrator {
 public:
  EmptySpaceIntegrator();
  EmptySpaceIntegrator(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~EmptySpaceIntegrator() = default;

  void launchIntegrationKernel(float truncation_distance_m);

  /// @brief Updates an EmptySpace layer according to a tsdf layer.
  /// @param block_indices_to_update The block indices that should be updated.
  /// @param tsdf_layer The tsdf layer that is used to check wether blocks
  /// are completely empty or not.
  /// @param empty_space_layer_ptr The emptySpace layer that will be updated.
  void updateEmptySpaceLayer(
      const std::vector<Index3D>& block_indices_to_update,
      const TsdfLayer& tsdf_layer, EmptySpaceLayer* empty_space_layer_ptr,
      float truncation_distance_vox);

  /// @brief Gets all indices of blocks marked empty in EmptySpace layer.
  /// @param empty_space_layer_ptr EmptySpace layer that holds the empty
  /// flags.
  ///@return std::vector<Index3D> holding indices of all blocks that are marked
  /// empty.
  std::vector<Index3D> getIndicesOfAllBlocksMarkedEmpty(
      const std::vector<Index3D>& block_indices_to_consider,
      EmptySpaceLayer* empty_space_layer_ptr);

  /// @brief Removes all blocks marked empty from a given layer by checking
  /// every single block.
  /// @param layer_to_modify_ptr Layer to update / clear blocks from.
  /// @param empty_space_layer_ptr EmptySpace layer that holds the empty
  /// flags.
  template <typename LayerType>
  void clearEmptyBlocksFromLayer(LayerType* layer_to_modify_ptr,
                                 EmptySpaceLayer* empty_space_layer_ptr);

  /// @brief Removes all blocks marked empty from a given layer.
  /// @param block_indices_to_consider Indices of blocks to consider for
  /// clearing.
  /// @param layer_to_modify_ptr Layer to update / clear blocks from.
  /// @param empty_space_layer_ptr EmptySpace layer that holds the empty
  /// flags.
  template <typename LayerType>
  void clearEmptyBlocksFromLayer(
      const std::vector<Index3D>& block_indices_to_consider,
      LayerType* layer_to_modify_ptr, EmptySpaceLayer* empty_space_layer_ptr);

  // Parameter getters/setters as needed
  // ...

 protected:
  std::shared_ptr<CudaStream> cuda_stream_;

  // Block index buffers
  host_vector<Index3D> block_indices_to_update_host_;
  device_vector<Index3D> block_indices_to_update_device_;

  // Block ptr buffers
  host_vector<EmptySpaceBlock*> empty_space_blocks_to_update_host_;
  device_vector<EmptySpaceBlock*> empty_space_blocks_to_update_device_;
  host_vector<const TsdfBlock*> tsdf_blocks_to_update_host_;
  device_vector<const TsdfBlock*> tsdf_blocks_to_update_device_;
};

template <typename LayerType>
void EmptySpaceIntegrator::clearEmptyBlocksFromLayer(
    LayerType* layer_to_modify_ptr, EmptySpaceLayer* empty_space_layer_ptr) {
  const std::vector<Index3D> all_block_indices =
      empty_space_layer_ptr->getAllBlockIndices();
  clearEmptyBlocksFromLayer(all_block_indices, layer_to_modify_ptr,
                            empty_space_layer_ptr);
}

template <typename LayerType>
void EmptySpaceIntegrator::clearEmptyBlocksFromLayer(
    const std::vector<Index3D>& block_indices_to_consider,
    LayerType* layer_to_modify_ptr, EmptySpaceLayer* empty_space_layer_ptr) {
  // Check inputs
  CHECK_NOTNULL(layer_to_modify_ptr);

  std::vector<Index3D> block_indices_to_be_cleared =
      getIndicesOfAllBlocksMarkedEmpty(block_indices_to_consider,
                                       empty_space_layer_ptr);

  if (block_indices_to_be_cleared.empty()) {
    return;
  }

  layer_to_modify_ptr->clearBlocks(block_indices_to_be_cleared);
}

}  // namespace nvblox

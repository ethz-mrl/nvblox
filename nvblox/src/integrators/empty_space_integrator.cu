#include "nvblox/integrators/empty_space_integrator.h"
#include "nvblox/integrators/internal/integrators_common.h"

#include "nvblox/utils/timing.h"

namespace nvblox {

EmptySpaceIntegrator::EmptySpaceIntegrator()
    : EmptySpaceIntegrator(std::make_shared<CudaStreamOwning>()) {}

EmptySpaceIntegrator::EmptySpaceIntegrator(
    std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

__global__ void updateEmptySpaceKernel(
    int num_block_indices_to_update, const TsdfBlock** tsdf_blocks_to_update,
    EmptySpaceBlock** empty_space_blocks_to_update, float truncation_distance) {
  if (blockIdx.x >= num_block_indices_to_update) {
    return;
  }

  // Shared flag: 0 means empty so far, 1 means occupancy detected
  __shared__ int occupied_flag;

  // In each block, the first thread handles the shared flag.
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    occupied_flag = 0;
  }
  __syncthreads();

  const Index3D voxel_index(threadIdx.x, threadIdx.y, threadIdx.z);

  const TsdfVoxel& tsdf_voxel =
      tsdf_blocks_to_update[blockIdx.x]
          ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];

  // A block is considered occupied if ANY voxel is:
  // 1. Observed (weight > 0) AND
  // 2. Within truncation distance of a surface
  if (tsdf_voxel.weight > 0.0f && truncation_distance > tsdf_voxel.distance) {
    atomicExch(&occupied_flag, 1);
  }

  __syncthreads();

  // The first thread of the block updates the shared flag.
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    empty_space_blocks_to_update[blockIdx.x]->is_empty = (occupied_flag == 0);
  }
}

void EmptySpaceIntegrator::launchIntegrationKernel(float truncation_distance) {
  const dim3 kThreadsPerBlock(TsdfBlock::kVoxelsPerSide,
                              TsdfBlock::kVoxelsPerSide,
                              TsdfBlock::kVoxelsPerSide);
  const int num_thread_blocks = block_indices_to_update_device_.size();

  // Launch Kernel for update
  updateEmptySpaceKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                           *cuda_stream_>>>(
      block_indices_to_update_device_.size(),
      tsdf_blocks_to_update_device_.data(),
      empty_space_blocks_to_update_device_.data(), truncation_distance);
  checkCudaErrors(cudaPeekAtLastError());
}

void EmptySpaceIntegrator::updateEmptySpaceLayer(
    const std::vector<Index3D>& block_indices_to_update,
    const TsdfLayer& tsdf_layer, EmptySpaceLayer* empty_space_layer_ptr,
    float truncation_distance) {
  timing::Timer integration_timer("empty_space/integrate");

  // Check inputs
  CHECK_NOTNULL(empty_space_layer_ptr);
  if (block_indices_to_update.empty()) {
    return;
  }
  const size_t num_block_to_update = block_indices_to_update.size();

  // Allocate missing blocks
  timing::Timer allocate_timer("empty_space/integrate/allocate");
  empty_space_layer_ptr->allocateBlocksAtIndices(block_indices_to_update,
                                                 *cuda_stream_);
  allocate_timer.Stop();

  // Expand the buffers when needed
  if (num_block_to_update > block_indices_to_update_device_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size =
        static_cast<int>(kBufferExpansionFactor * num_block_to_update);
    block_indices_to_update_device_.reserveAsync(new_size, *cuda_stream_);
    empty_space_blocks_to_update_device_.reserveAsync(new_size, *cuda_stream_);
    tsdf_blocks_to_update_device_.reserveAsync(new_size, *cuda_stream_);
  }

  timing::Timer transfer_timer("empty_space/integrate/transfer_blocks");

  // Transfer block indices
  transferBlocksIndicesToDevice(block_indices_to_update, *cuda_stream_,
                                &block_indices_to_update_host_,
                                &block_indices_to_update_device_);

  // Transfer empty_space block pointers
  transferBlockPointersToDevice(block_indices_to_update, *cuda_stream_,
                                empty_space_layer_ptr,
                                &empty_space_blocks_to_update_host_,
                                &empty_space_blocks_to_update_device_);

  // Transfer tsdf block pointers
  transferBlockPointersToDevice<TsdfBlock>(
      block_indices_to_update, *cuda_stream_, tsdf_layer,
      &tsdf_blocks_to_update_host_, &tsdf_blocks_to_update_device_);
  transfer_timer.Stop();

  timing::Timer update_timer("empty_space/integrate/update_blocks");
  launchIntegrationKernel(truncation_distance);

  cuda_stream_->synchronize();
  update_timer.Stop();
}

}  // namespace nvblox
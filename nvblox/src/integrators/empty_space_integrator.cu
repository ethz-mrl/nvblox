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
    EmptySpaceBlock** empty_space_blocks_to_update,
    float truncation_distance_m) {
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
  if (tsdf_voxel.weight > 0.0f && truncation_distance_m > tsdf_voxel.distance) {
    atomicExch(&occupied_flag, 1);
  }

  __syncthreads();

  // The first thread of the block updates the shared flag.
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    empty_space_blocks_to_update[blockIdx.x]->is_empty = (occupied_flag == 0);
  }
}

void EmptySpaceIntegrator::launchIntegrationKernel(
    float truncation_distance_m) {
  const dim3 kThreadsPerBlock(TsdfBlock::kVoxelsPerSide,
                              TsdfBlock::kVoxelsPerSide,
                              TsdfBlock::kVoxelsPerSide);
  const int num_thread_blocks = block_indices_to_update_device_.size();

  // Launch Kernel for update
  updateEmptySpaceKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                           *cuda_stream_>>>(
      block_indices_to_update_device_.size(),
      tsdf_blocks_to_update_device_.data(),
      empty_space_blocks_to_update_device_.data(), truncation_distance_m);
  checkCudaErrors(cudaPeekAtLastError());
}

void EmptySpaceIntegrator::updateEmptySpaceLayer(
    const std::vector<Index3D>& block_indices_to_update,
    const TsdfLayer& tsdf_layer, EmptySpaceLayer* empty_space_layer_ptr,
    float truncation_distance_vox) {
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
  launchIntegrationKernel(truncation_distance_vox * tsdf_layer.voxel_size());

  cuda_stream_->synchronize();
  update_timer.Stop();
}

__global__ void getIndicesOfAllBlocksMarkedEmptyKernel(
    int num_block_indices_to_check, Index3D* block_indices_to_check,
    EmptySpaceBlock** empty_space_blocks_to_check,
    Index3D* output_block_indices, int* output_count) {
  const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (block_idx >= num_block_indices_to_check) {
    return;
  }

  // Check if this block is empty
  const EmptySpaceBlock* block = empty_space_blocks_to_check[block_idx];
  if (block != nullptr && block->is_empty) {
    // Atomically add this block index to the output
    const int idx = atomicAdd(output_count, 1);
    if (idx <
        num_block_indices_to_check) {  // Safety check to avoid out-of-bounds
      output_block_indices[idx] = block_indices_to_check[block_idx];
    }
  }
}

std::vector<Index3D> EmptySpaceIntegrator::getIndicesOfAllBlocksMarkedEmpty(
    EmptySpaceLayer* empty_space_layer_ptr) {
  timing::Timer index_fetching_timer("empty_space/get_empy_block_indices");

  // Check inputs
  CHECK_NOTNULL(empty_space_layer_ptr);

  // For now, fetch indices of all blocks in the layer for the lookup.
  const std::vector<Index3D> all_block_indices =
      empty_space_layer_ptr->getAllBlockIndices();
  const size_t num_all_block_indices = all_block_indices.size();

  // Early return if no blocks available to check.
  if (num_all_block_indices == 0) {
    return std::vector<Index3D>();
  }

  // Expand the buffers when needed
  if (num_all_block_indices > block_indices_to_update_device_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size =
        static_cast<int>(kBufferExpansionFactor * num_all_block_indices);
    block_indices_to_update_device_.reserveAsync(new_size, *cuda_stream_);
    empty_space_blocks_to_update_device_.reserveAsync(new_size, *cuda_stream_);
    tsdf_blocks_to_update_device_.reserveAsync(new_size, *cuda_stream_);
  }

  timing::Timer transfer_blocks_to_device_timer(
      "empty_space/get_empy_block_indices/transfer_blocks_to_device");

  // Transfer block indices
  transferBlocksIndicesToDevice(all_block_indices, *cuda_stream_,
                                &block_indices_to_update_host_,
                                &block_indices_to_update_device_);

  // Transfer block pointers
  transferBlockPointersToDevice(all_block_indices, *cuda_stream_,
                                empty_space_layer_ptr,
                                &empty_space_blocks_to_update_host_,
                                &empty_space_blocks_to_update_device_);

  transfer_blocks_to_device_timer.Stop();

  timing::Timer allocate_output_buffers(
      "empty_space/get_empy_block_indices/"
      "allocate_output_buffers");

  device_vector<Index3D> output_indices_device;
  output_indices_device.resizeAsync(num_all_block_indices, *cuda_stream_);

  device_vector<int> output_count_device;
  output_count_device.resizeAsync(1, *cuda_stream_);
  output_count_device.setZeroAsync(*cuda_stream_);

  host_vector<int> output_count_host(1, 0);

  cuda_stream_->synchronize();
  allocate_output_buffers.Stop();

  timing::Timer fetching_timer("empty_space/get_empty_block_indices/fetch");

  // Launch kernel
  constexpr int kNumThreads = 512;
  const int num_thread_blocks =
      (num_all_block_indices + kNumThreads - 1) / kNumThreads;

  getIndicesOfAllBlocksMarkedEmptyKernel<<<num_thread_blocks, kNumThreads>>>(
      block_indices_to_update_device_.size(),
      block_indices_to_update_device_.data(),
      empty_space_blocks_to_update_device_.data(), output_indices_device.data(),
      output_count_device.data());

  checkCudaErrors(cudaPeekAtLastError());

  output_count_device.copyToAsync(output_count_host.data(), *cuda_stream_);
  cuda_stream_->synchronize();
  fetching_timer.Stop();

  const int num_empty = output_count_host[0];

  // Copy non-empty indices back to host
  std::vector<Index3D> resulting_empty_indices;
  if (num_empty > 0) {
    resulting_empty_indices =
        output_indices_device.toVectorAsync(*cuda_stream_);
    cuda_stream_->synchronize();

    // Our device buffer for indices output_indices_device is initialized to
    // full length with (0,0,0), which will also be copied into our return
    // buffer. We need to truncate this vector to only contain actual empty
    // indices.
    resulting_empty_indices.resize(num_empty);
  }

  return resulting_empty_indices;
}

}  // namespace nvblox

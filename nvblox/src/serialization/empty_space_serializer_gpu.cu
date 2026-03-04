#include "nvblox/serialization/empty_space_serializer_gpu.h"

namespace nvblox {

std::shared_ptr<SerializedEmptySpaceLayer> EmptySpaceSerializerGpu::serialize(
    const EmptySpaceLayer& empty_space_layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    const CudaStream& cuda_stream) {
  is_empty_flags_serializer_.serializeAsync(
      empty_space_layer, block_indices_to_serialize,
      serialized_empty_space_->is_empty_flags,
      serialized_empty_space_->is_empty_flags_block_offsets,
      [](const EmptySpaceBlock* empty_space_block)
          -> std::pair<const bool*, int> {
        return std::make_pair(&empty_space_block->is_empty, 1);
      },
      cuda_stream);

  // Create an unique identifier for each block.
  serialized_empty_space_->block_indices = block_indices_to_serialize;

  cuda_stream.synchronize();

  return serialized_empty_space_;
}

EmptySpaceSerializerGpu::EmptySpaceSerializerGpu()
    : serialized_empty_space_(std::make_shared<SerializedEmptySpaceLayer>()) {}

}  // namespace nvblox

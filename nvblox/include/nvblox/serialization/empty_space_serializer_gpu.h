#pragma once

#include <memory>
#include <vector>

// #include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
// #include "nvblox/core/unified_vector.h"
// #include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/serialization/internal/serialization_gpu.h"

namespace nvblox {

/// Container for storing a serialized mesh
struct SerializedEmptySpaceLayer {
  /// Serialized empty space block components
  host_vector<bool> is_empty_flags;

  /// Indices of serialized empty space blocks
  std::vector<Index3D> block_indices;

  /// Offsets for each mesh block in the output vector.
  /// Size of offsets is num_blocks+1. The first element is always
  /// zero and the last element always equals the total size of the serialized
  /// vector. The size of block n can be computed as offsets[n+1] -
  /// offsets[n]
  host_vector<int32_t> is_empty_flags_block_offsets;
};

/// Class for serialization
class EmptySpaceSerializerGpu {
 public:
  EmptySpaceSerializerGpu();
  virtual ~EmptySpaceSerializerGpu() = default;
  // This is needed for in layer streamer trait to resolve type.
  using SerializedLayerType = SerializedEmptySpaceLayer;

  /// Serialize an empty space layer
  ///
  /// All requested blocks will be serialized and placed in output host
  /// vectors. This implementation is more effective than issuing a memcpy
  /// for each block.
  ///
  /// @attention: Input empty space layer must be in device or unified memory
  ///
  /// @param empty_space_layer           Empty space layer to serialize
  /// @param block_indices_to_serialize  Requested block indices
  /// @param cuda_stream                 Cuda stream
  std::shared_ptr<SerializedEmptySpaceLayer> serialize(
      const EmptyBlockLayer& empty_space_layer,
      const std::vector<Index3D>& block_indices_to_serialize,
      const CudaStream& cuda_stream);

  /// Get the serialized mesh
  std::shared_ptr<SerializedEmptySpaceLayer> getSerializedLayer() const {
    return serialized_empty_space_;
  }

 private:
  LayerSerializerGpuInternal<EmptyBlockLayer, bool> is_empty_flags_serializer_;

  std::shared_ptr<SerializedEmptySpaceLayer> serialized_empty_space_;
};

}  // namespace nvblox

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

      void launchKernel(float truncation_distance);

      /// @brief Updates an EmptySpace layer according to a tsdf layer.
      /// @param block_indices_to_update The block indices that should be updated.
      /// @param tsdf_layer The tsdf layer that is used to check wether blocks
      /// are completely empty or not.
      /// @param empty_space_layer_ptr The emptySpace layer that will be updated.
      void updateEmptySpaceLayer(
          const std::vector<Index3D>& block_indices_to_update,
          const TsdfLayer& tsdf_layer,
          EmptyBlockLayer* empty_space_layer_ptr,
          float truncation_distance);

      // Parameter getters/setters as needed
      // ...
    
    protected:
      std::shared_ptr<CudaStream> cuda_stream_;
      
      // Block index buffers
      host_vector<Index3D> block_indices_to_update_host_;
      device_vector<Index3D> block_indices_to_update_device_;

      // Block ptr buffers
      host_vector<EmptyBlock*> empty_space_blocks_to_update_host_;
      device_vector<EmptyBlock*> empty_space_blocks_to_update_device_;
      host_vector<const TsdfBlock*> tsdf_blocks_to_update_host_;
      device_vector<const TsdfBlock*> tsdf_blocks_to_update_device_;
  };

} // namespace nvblox
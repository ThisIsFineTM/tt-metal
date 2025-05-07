// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

// forward declarations
namespace tt::tt_metal {
class IDevice;
class Tensor;
}  // namespace tt::tt_metal

namespace ttnn::operations::fused::normalization {

tt::tt_metal::operation::ProgramWithCallbacks frmsnorm_pre_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& b,  // residual
    Tensor& output,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config,
    // New Parameters
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ::ttnn::ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

tt::tt_metal::operation::ProgramWithCallbacks frmsnorm_post_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& gamma,  // weight
    const std::optional<const Tensor>& stats,  // stats
    Tensor& output,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config,
    const GlobalSemaphore& semaphore,
    const uint32_t ring_size,
    const uint32_t num_links);

struct RMSAllGather {
    float eps;
    MemoryConfig output_mem_config;
    ttnn::operations::normalization::LayerNormProgramConfig program_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    std::optional<DataType> dtype;
    const ttnn::ccl::Topology topology;
    const bool is_pre;
    const uint32_t num_links;
    const uint32_t ring_size;
    const GlobalSemaphore semaphore;
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    const uint32_t cluster_axis = 0;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;

    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;

    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& mesh_coordinate,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;

    RMSAllGather(
        float eps,
        MemoryConfig output_mem_config,
        ttnn::operations::normalization::LayerNormProgramConfig program_config,
        const DeviceComputeKernelConfig compute_kernel_config,
        std::optional<DataType> dtype,
        ::ttnn::ccl::Topology topology,
        const bool is_pre,
        const uint32_t num_links,
        const uint32_t ring_size,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        uint32_t cluster_axis);

    auto attributes() const -> std::vector<std::tuple<std::string, tt::stl::reflection::Attribute>>;

    tt::tt_metal::operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
};

}  // namespace ttnn::operations::fused::normalization

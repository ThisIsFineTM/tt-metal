// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "cpp/ttnn-nanobind/decorators.hpp"

#include "pad.hpp"

namespace nb = nanobind;

namespace ttnn::operations::data_movement::detail {

void bind_pad(nb::module_& module) {
    auto doc =
        R"doc(

            Returns a padded tensor, with a specified value at the specified location. If the input tensor is on host, the pad will be performed on host, and if its on device it will be performed on device.

            Equivalent pytorch code:

            .. code-block:: python

                torch.pad(input_tensor, padding, value)
                torch.pad(input_tensor, output_tensor_shape, input_tensor_start, value)

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                padding (ttnn.Tensor): padding to apply. Each element of padding should be a tuple of 2 integers, with the first integer specifying the number of values to add before the tensor and the second integer specifying the number of values to add after the tensor. Mutually exclusive to output_tensor_shape and input_tensor_start.
                output_tensor_shape (shape): Final shape of padded tensor. This along with input_tensor_start is mutually exclusive from padding.
                input_tensor_start (shape): Shape describing where to start padding. This along with output_tensor_shape is mutually exclusive from padding.
                value (number): value to pad with.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
               List of ttnn.Tensor: the output tensor.

        )doc";

    using OperationType = decltype(ttnn::pad);
    ttnn::bind_registered_operation(
        mod,
        ttnn::pad,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               ttnn::SmallVector<std::pair<uint32_t, uint32_t>> padding,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) { return self(queue_id, input_tensor, padding, value, use_multicore, memory_config); },
            nb::arg("input_tensor"),
            nb::arg("padding"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = true,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId,
        },
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::Array1D& output_padded_shape,
               const tt::tt_metal::Array1D& input_tensor_start,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    output_padded_shape,
                    input_tensor_start,
                    value,
                    use_multicore,
                    memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("output_padded_shape"),
            nb::arg("input_tensor_start"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = false,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId,
        },
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::Array2D& output_padded_shape,
               const tt::tt_metal::Array2D& input_tensor_start,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    output_padded_shape,
                    input_tensor_start,
                    value,
                    use_multicore,
                    memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("output_padded_shape"),
            nb::arg("input_tensor_start"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = false,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId,
        },
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::Array3D& output_padded_shape,
               const tt::tt_metal::Array3D& input_tensor_start,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    output_padded_shape,
                    input_tensor_start,
                    value,
                    use_multicore,
                    memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("output_padded_shape"),
            nb::arg("input_tensor_start"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = false,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId,
        },
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::Array4D& output_padded_shape,
               const tt::tt_metal::Array4D& input_tensor_start,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    output_padded_shape,
                    input_tensor_start,
                    value,
                    use_multicore,
                    memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("output_padded_shape"),
            nb::arg("input_tensor_start"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = false,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId,
        },
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::Array5D& output_padded_shape,
               const tt::tt_metal::Array5D& input_tensor_start,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    output_padded_shape,
                    input_tensor_start,
                    value,
                    use_multicore,
                    memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("output_padded_shape"),
            nb::arg("input_tensor_start"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = false,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId,
        },
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::Array6D& output_padded_shape,
               const tt::tt_metal::Array6D& input_tensor_start,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    output_padded_shape,
                    input_tensor_start,
                    value,
                    use_multicore,
                    memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("output_padded_shape"),
            nb::arg("input_tensor_start"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = false,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId,
        },
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::Array7D& output_padded_shape,
               const tt::tt_metal::Array7D& input_tensor_start,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    output_padded_shape,
                    input_tensor_start,
                    value,
                    use_multicore,
                    memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("output_padded_shape"),
            nb::arg("input_tensor_start"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = false,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId,
        },
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::Array8D& output_padded_shape,
               const tt::tt_metal::Array8D& input_tensor_start,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    output_padded_shape,
                    input_tensor_start,
                    value,
                    use_multicore,
                    memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("output_padded_shape"),
            nb::arg("input_tensor_start"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = false,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId,
        });
}
}  // namespace ttnn::operations::data_movement::detail

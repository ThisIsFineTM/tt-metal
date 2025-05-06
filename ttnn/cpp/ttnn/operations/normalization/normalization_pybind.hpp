// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::normalization {
namespace py = pybind11;
void py_module(py::module& module);
}  // namespace ttnn::operations::normalization

/*
void py_module(py::module& module) {
    detail::bind_normalization_softmax(module);
    detail::bind_normalization_layernorm(module);
    detail::bind_normalization_rms_norm(module);
    detail::bind_normalization_group_norm(module);
    detail::bind_normalization_layernorm_distributed(module);
    detail::bind_normalization_rms_norm_distributed(module);
    detail::bind_batch_norm_operation(module);
}
*/

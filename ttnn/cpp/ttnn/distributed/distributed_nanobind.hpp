// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace nb = nanobind;

namespace ttnn::distributed {

void py_module_types(nb::module& module);
void py_module(nb::module& module);

}  // namespace ttnn::distributed

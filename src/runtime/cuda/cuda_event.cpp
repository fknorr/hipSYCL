/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/runtime/cuda/cuda_event.hpp"
#include "hipSYCL/runtime/error.hpp"

#include <cuda_runtime_api.h>

namespace hipsycl {
namespace rt {


cuda_node_event::cuda_node_event(device_id dev, CUevent_st* evt, const timing_ref *ref)
: _dev{dev}, _evt{evt}, _ref(ref)
{}

cuda_node_event::~cuda_node_event()
{
  auto err = cudaEventDestroy(_evt);
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_node_event: Couldn't destroy event",
                              error_code{"CUDA", err}});
  }
}

bool cuda_node_event::is_complete() const
{
  cudaError_t err = cudaEventQuery(_evt);
  if (err != cudaErrorNotReady && err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_node_event: Couldn't query event status",
                              error_code{"CUDA", err}});
  }
  return err == cudaSuccess;
}

void cuda_node_event::wait()
{
  auto err = cudaEventSynchronize(_evt);
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_node_event: cudaEventSynchronize() failed",
                              error_code{"CUDA", err}});
  }
}

std::optional<cuda_node_event::clock::time_point> cuda_node_event::get_completion_time() const
{
  assert(_ref != nullptr);
  float ms = 0;
  auto err = cudaEventElapsedTime(&ms, _ref->ref_event->get_event(), _evt);
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
        error_info{"cuda_node_event: cudaEventElapsedTime() failed",
            error_code{"CUDA", err}});
  }
  auto elapsed = std::chrono::duration_cast<clock::duration>(std::chrono::duration<float, std::milli>(ms));
  return _ref->ref_time_point + elapsed;
}

CUevent_st* cuda_node_event::get_event() const
{
  return _evt;
}

device_id cuda_node_event::get_device() const
{
  return _dev;
}

}
}

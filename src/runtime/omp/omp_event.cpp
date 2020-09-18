/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#include "hipSYCL/runtime/omp/omp_event.hpp"

namespace hipsycl {
namespace rt {

bool omp_node_event::completion_flag::is_complete() const {
  return _atomic_timestamp.load(std::memory_order_acquire) != _incomplete_timestamp;
}

std::optional<std::chrono::steady_clock::time_point> omp_node_event::completion_flag::completion_time() const {
  auto timestamp = _atomic_timestamp.load(std::memory_order_acquire);
  if (timestamp != _incomplete_timestamp) {
    return time_point(duration(timestamp));
  } else {
    return std::nullopt;
  }
}

void omp_node_event::completion_flag::complete_now() {
  auto timestamp = clock::now().time_since_epoch().count();
  if (timestamp == _incomplete_timestamp) {
    ++timestamp;
  }
  auto expected = _incomplete_timestamp;
  _atomic_timestamp.compare_exchange_strong(expected, timestamp, std::memory_order_release,
      std::memory_order_relaxed);
}

omp_node_event::omp_node_event()
: _completion{std::make_shared<completion_flag>()}
{}

omp_node_event::~omp_node_event()
{}

bool omp_node_event::is_complete() const {
  return _completion->is_complete();
}

void omp_node_event::wait() {
  while(!_completion->is_complete()) ;
}

std::shared_ptr<omp_node_event::completion_flag>
omp_node_event::get_completion_flag() const {
  return _completion;
}

std::optional<omp_node_event::clock::time_point> omp_node_event::get_completion_time() const
{
  return _completion->completion_time();
}

}
}

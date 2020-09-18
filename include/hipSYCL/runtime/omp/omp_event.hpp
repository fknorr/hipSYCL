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

#ifndef HIPSYCL_OMP_EVENT_HPP
#define HIPSYCL_OMP_EVENT_HPP

#include <atomic>

#include "../event.hpp"

namespace hipsycl {
namespace rt {


class omp_node_event : public dag_node_event
{
public:
  struct completion_flag {
      bool is_complete() const;

      std::optional<std::chrono::steady_clock::time_point> completion_time() const;

      void complete_now();

  private:
      using clock = std::chrono::steady_clock;
      using time_point = std::chrono::steady_clock::time_point;
      using duration = time_point::duration;
      constexpr static time_point::rep _incomplete_timestamp = std::numeric_limits<time_point::rep>::lowest();
      static_assert(std::atomic<time_point::rep>::is_always_lock_free);
      std::atomic<time_point::rep> _atomic_timestamp{_incomplete_timestamp};
  };


  omp_node_event();
  ~omp_node_event();

  virtual bool is_complete() const override;
  virtual void wait() override;
  virtual std::optional<clock::time_point> get_completion_time() const override;

  std::shared_ptr<completion_flag> get_completion_flag() const;

private:
  std::shared_ptr<completion_flag> _completion;
};


}
}

#endif

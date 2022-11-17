/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Fabian Knorr
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

#ifndef HIPSYCL_PROFILE_HPP
#define HIPSYCL_PROFILE_HPP

#include <optional>
#include <string>
#include <variant>
#include <vector>


namespace hipsycl::sycl::profile {

enum class backend_queue_id : size_t {};
enum class command_group_id : size_t {};

enum class frontend_operation {
    submit_command_group,
    wait,
    host_access,
};

inline std::string_view frontend_operation_string(frontend_operation op) {
  switch (op) {
    case frontend_operation::submit_command_group: return "submit command group";
    case frontend_operation::wait: return "wait";
    case frontend_operation::host_access: return "host access";
  }
}

enum class runtime_operation {
    hipSYCL_flush_dag,
};

inline std::string_view runtime_operation_name(runtime_operation op) {
  switch (op) {
    case runtime_operation::hipSYCL_flush_dag: return "flush dag";
  }
}

enum class backend_operation {
    execute_kernel,
    execute_host_task,
    copy,
    fill,
    prefetch,
    hipSYCL_custom_operation,
};

inline std::string_view backend_operation_string(backend_operation op) {
  switch (op) {
    case backend_operation::execute_kernel: return "execute kernel";
    case backend_operation::execute_host_task: return "execute host task";
    case backend_operation::copy: return "copy";
    case backend_operation::fill: return "fill";
    case backend_operation::prefetch: return "prefetch";
    case backend_operation::hipSYCL_custom_operation: return "custom operation";
  }
}

class sink {
    public:
        virtual ~sink() = default;
        virtual void register_backend_queue(backend_queue_id id, std::string name, bool in_order) = 0;
        virtual void unregister_backend_queue(backend_queue_id id) = 0;
        virtual void register_command_group(command_group_id id, std::optional<std::string> name) = 0;
        virtual void unregister_command_group(command_group_id id) = 0;
        virtual void register_runtime_thread(std::string name) = 0;
        virtual void unregister_runtime_thread() = 0;

        virtual void frontend_thread_begin(frontend_operation, std::vector<command_group_id> cgs) = 0;
        virtual void frontend_thread_end() = 0;

        virtual void runtime_thread_begin(runtime_operation, std::vector<command_group_id> cgs) = 0;
        virtual void runtime_thread_end() = 0;

        virtual void backend_queue_begin(backend_queue_id id, backend_operation operation,
            std::vector<command_group_id> cgs) = 0;
        virtual void backend_queue_end(backend_queue_id id) = 0;
};

extern sink *the_sink; // hack

}

#endif

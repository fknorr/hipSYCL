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

#include "buffer.hpp"
#include "device.hpp"

#include <string>
#include <variant>
#include <vector>


namespace hipsycl::sycl::profile {

using identifier = uint64_t;

struct kernel_task_type {
    std::string kernel_name;
};
struct host_task_type {};
struct copy_task_type {
    identifier buffer_from;
    identifier buffer_to;
    size_t bytes;
};
struct fill_task_type {
    identifier buffer;
    size_t bytes;
};
struct update_host_task_type {
    identifier buffer;
};
enum class usm_task_type {
    malloc,
    malloc_host,
    malloc_device,
    free,
    copy,
    memset,
    fill,
    prefetch,
    mem_advice,
};
using task_type = std::variant<kernel_task_type, host_task_type, fill_task_type, copy_task_type, update_host_task_type,
    usm_task_type>;

enum class device {
    host = 0,
};
struct task {
    identifier id;
    task_type type;
    device executes_on;
    std::vector<identifier> dependencies;
};

enum class transfer_direction {
    host_to_device,
    device_to_host,
    device_to_device,
};
struct transfer {
    identifier id;
    identifier buffer_id;
    transfer_direction direction;
    size_t bytes;
    std::vector<identifier> dependencies;
};

struct host_access {
    identifier buffer_id;
    std::vector<identifier> dependencies;
};

class sink {
    public:
        virtual ~sink() = default;
        virtual void register_device(identifier device_id, const sycl::device &device) = 0;
        virtual void set_buffer_name(identifier buffer_id, std::string name) = 0;
        virtual void task_submit(task task) = 0;
        virtual void task_begin_execute(identifier task_id) = 0;
        virtual void task_end_execute(identifier task_id) = 0;
        virtual void transfer_begin(transfer transfer) = 0;
        virtual void transfer_end(identifier transfer_id) = 0;
        virtual void host_access_request(host_access access) = 0;
        virtual void host_access_begin(identifier access_id) = 0;
        virtual void host_access_end(identifier access_id) = 0;
        virtual void wait_begin() = 0;
        virtual void wait_begin(std::vector<identifier> dependencies) = 0;
        virtual void wait_end() = 0;
        virtual void idle_begin(device device) = 0;
        virtual void idle_end(device device) = 0;
};

template<typename T, int Dims>
void set_buffer_name(const buffer<T, Dims> &buf, std::string name);

inline sink *the_sink = nullptr; // hack

}

#endif

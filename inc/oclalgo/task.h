/*!
 * Copyright (c) 2014, Samsung Electronics Co.,Ltd.
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
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of Samsung Electronics Co.,Ltd..
 *
 * OCLAlgo - Framework based on C++11 and OpenCL API to provide simple access
 *           to OpenCL devices for asynchronous calculations.
 * URL:      https://github.com/seninds/OCLAlgo
 */

/*! @file task.h
 *  @brief Contains oclalgo::Task class.
 *  @author Senin Dmitry <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_OCLALGO_TASK_H_
#define INC_OCLALGO_TASK_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <type_traits>
#include <vector>

namespace oclalgo {

/*!
 * @brief Class to represent individual OpenCL task.
 *
 * Parses input and output OpenCL kernel arguments.
 */
class Task {
 public:
  template <typename... Args>
  Task(const cl::Kernel& kernel, const Args&... args)
      : kernel_(kernel) {
    SetArg(0, args...);
  }

  /** @brief Clears cl::Kernel object and all stored cl::Buffer objects. */
  void clear() noexcept {
    kernel_ = cl::Kernel();
    buffers_.clear();
    output_.clear();
  }

  cl::Kernel kernel() const noexcept { return kernel_; }
  std::vector<cl::Buffer> buffers() const noexcept { return buffers_; }
  std::vector<cl::Buffer> output() const noexcept { return output_; }

 private:
  void SetArg(int /*index*/) {
  }

  template <typename T>
  void SetArg(int index, const T& arg) {
    kernel_.setArg(index, arg.data);
    if (std::is_same<T, KernelArg<cl::Buffer>>::value) {
      if (arg.arg_type == ArgType::OUT || arg.arg_type == ArgType::IN_OUT)
        output_.push_back(arg.data);
      else
        buffers_.push_back(arg.data);
    }
  }

  template <typename First, typename... Tail>
  void SetArg(int index, const First& data, const Tail&... args) {
    SetArg(index, data);
    SetArg(++index, args...);
  }

  cl::Kernel kernel_;
  std::vector<cl::Buffer> buffers_;
  std::vector<cl::Buffer> output_;
};

}  // namespace oclalgo

#endif  // INC_OCLALGO_TASK_H_

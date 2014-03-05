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

/*! @file kernel_arg.h
 *  @brief Contains oclalgo::KernelArg class (wrapper for kernel data).
 *  @author Senin Dmitry <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_OCLALGO_KERNEL_ARG_H_
#define INC_OCLALGO_KERNEL_ARG_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace oclalgo {

/** @brief Enum of allowed argument types for OpenCL kernel. */
enum class ArgType { IN, OUT, IN_OUT };

/** @brief Class for wrapping OpenCL kernel data with argument type. */
template <typename T>
struct KernelArg {
 public:
  KernelArg(const T& data, ArgType arg_type);
  KernelArg(T&& data, ArgType arg_type);

  KernelArg(const KernelArg<T>&) = delete;
  KernelArg<T>& operator=(const KernelArg<T>&) = delete;

  KernelArg(KernelArg<T>&& arg);
  KernelArg<T>& operator=(KernelArg<T>&& arg);

  const T& data() const noexcept { return data_; }
  T& data() noexcept { return data_; }
  ArgType arg_type() const noexcept { return arg_type_; }
  ArgType& arg_type() noexcept { return arg_type_; }

 private:
  T data_;
  ArgType arg_type_;
};

typedef KernelArg<cl::Buffer> BufferArg;
typedef KernelArg<cl::LocalSpaceArg> LocalArg;

template <typename T>
KernelArg<T>::KernelArg(const T& data, ArgType arg_type)
    : data_(data),
      arg_type_(arg_type) {
}

template <typename T>
KernelArg<T>::KernelArg(T&& data, ArgType arg_type)
    : data_(std::move(data)),
      arg_type_(arg_type) {
}

template <typename T>
KernelArg<T>::KernelArg(KernelArg<T>&& arg)
    : data_(std::move(arg.data_)),
      arg_type_(arg.arg_type_) {
}

template <typename T>
KernelArg<T>& KernelArg<T>::operator=(KernelArg<T>&& arg) {
  if (this != &arg) {
    data_ = std::move(arg.data_);
    arg_type_ = arg.arg_type_;
  }
  return this;
}

}  // namespace oclalgo

#endif  // INC_OCLALGO_KERNEL_ARG_H_

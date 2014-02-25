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
  KernelArg(const T& data, ArgType arg_type) : data(data), arg_type(arg_type) {
  }

  T data;
  ArgType arg_type;
};

typedef KernelArg<cl::Buffer> BufferArg;
typedef KernelArg<cl::LocalSpaceArg> LocalArg;

}  // namespace oclalgo

#endif  // INC_OCLALGO_KERNEL_ARG_H_

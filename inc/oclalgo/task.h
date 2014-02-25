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

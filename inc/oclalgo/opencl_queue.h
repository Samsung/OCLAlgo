/*! @file opencl_queue.h
 *  @brief Class for simple OpenCL API usage.
 *  @author Senin Dmitry <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  Use OpencCL C++ Wrapper API. Create one in-order OpenCL command queue.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

/*! @mainpage OCLAlgo framework
 *
 * @section Brief
 *
 * Framework based on C++11 and OpenCL API to provide simple access to OpenCL
 * devices for asynchronous calculations (for example, matrix multiplication).
 *
 * @section Details
 *
 * Contains OpenCLQueue and Matrix classes.
 *
 * OpenCLQueue provides simple interface for starting OpenCL tasks.
 *
 * <i>Code example:</i>
 * @code{.cpp}
 * OpenCLQueue queue("Intel(R) OpenCL", "Intel(R)");
 *
 * // allocate host data
 * shared_array<int> input_a(new int[1024], 1024);
 * shared_array<int> input_b(new int[1024], 1024);
 * shared_array<int> output (new int[1024], 1024);
 *
 * // ...host data initiallization...
 *
 * // initialize device data by host array and I/O specificator
 * cl_data_t<int, oclalgo::IN>  device_input_a(input_a);
 * cl_data_t<int, oclalgo::IN>  device_input_b(input_b);
 * cl_data_t<int, oclalgo::OUT> device_output(output);
 *
 * // start OpenCL task
 * auto future = queue.AddTask("vector_add.cl", "vector_add", "", cl::NullRange,
 *                             cl::NDRange(1024), cl::NullRange, device_input_a,
 *                             device_input_b, device_output);
 *
 * // get results of calculations as std::tuple of output parameters
 * std::tie(output) = future.get();
 * @endcode
 *
 * Matrix class provides asynchronous operations using OpenCL.
 *
 * <i>Code example:</i>
 * @code{.cpp}
 * Matrix<int> a(32, 32), b(32, 32), c(32, 16), d(32, 16);
 *
 * // ...host matrices initialization...
 *
 * // load OpenCL device
 * auto future = (a.future() + b.future()) * c.future();
 *
 * // ...make some other calculations on host side...
 *
 * // get results
 * d = future.get();
 * @endcode
 *
 * @section Requirements
 *
 * <ul>
 * <li>C++11 compatible compiler (g++-4.8 was used during the development)</li>
 * <li>OpenCL drivers for your device (OpenCL version >= 1.1)</li>
 * </ul>
 */

#ifndef OCLALGO_OPENCL_QUEUE_H_
#define OCLALGO_OPENCL_QUEUE_H_

#define __CL_ENABLE_EXCEPTIONS

#include <cstdint>

#include <vector>
#include <unordered_map>
#include <tuple>
#include <type_traits>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <CL/cl.hpp>

#include <oclalgo/shared_array.h>

namespace oclalgo {

/** @brief Types of argument in OpenCL kernel. */
enum DataType { IN, OUT, IN_OUT, LOCALE, VAR };

/** @brief Class for wrapping OpenCL kernel data. */
template<typename T, oclalgo::DataType DT>
struct cl_data_t {
  cl_data_t(const oclalgo::shared_array<T>& array)
      : host_array(array) {
  }

  oclalgo::shared_array<T> host_array;
  static constexpr oclalgo::DataType io_type = DT;
};

/** @brief Class for synchronization OpenCLQueue task with host thread. */
template<typename T>
class cl_future {
 public:
  /** @brief Constructor of cl_future.
   *  @param data stored object
   *  @param buffers OpenCL Buffer objects
   *  @param event OpenCL event for task synchronization
   */
  cl_future(const T& data, const std::vector<cl::Buffer>& buffers,
            const cl::Event event)
      : stored_data_(data),
        buffers_(buffers),
        event_(event),
        is_event_set_(true) {
  }

  cl_future(T&& data, const std::vector<cl::Buffer>& buffers,
            const cl::Event event)
      : stored_data_(std::move(data)),
        buffers_(buffers),
        event_(event),
        is_event_set_(true) {
  }

  cl_future(const T& data)
      : stored_data_(data),
        is_event_set_(false) {
  }

  cl_future(const cl_future&) = delete;
  cl_future& operator=(const cl_future&) = delete;

  cl_future(cl_future&& other)
      : stored_data_(other.stored_data_),
        buffers_(other.buffers_),
        event_(other.event_),
        is_event_set_(other.is_event_set_) {
  }

  virtual ~cl_future() = default;

  /**
   * @brief Stop host thread and wait the end of OpenCL task.
   * @return Returns the refreshed host data.
   */
  virtual T get() {
    if (is_event_set_) event_.wait();
    return stored_data_;
  }
  /** @brief Stop host thread and wait the end of OpenCL task. */
  virtual void wait() const {
    if (is_event_set_) event_.wait();
  }

  cl::Event event() const noexcept { return event_; }
  bool is_event_set() const noexcept { return is_event_set_; }
  const std::vector<cl::Buffer>& buffers() const noexcept { return buffers_; }
  const T& stored_data() const noexcept { return stored_data_; }

 private:
  T stored_data_;
  std::vector<cl::Buffer> buffers_;
  cl::Event event_;
  bool is_event_set_;
};

/**
 * @brief Class for simple execution of OpenCL kernels.
 *
 * Use OpenCL C++ Wrapper API. Provide a OpenCL task synchronization by cl_future<Args...>
 * objects, which have similar interface and functionality like std::future
 */
class OpenCLQueue {
 public:
  OpenCLQueue(const std::string& platformName, const std::string& deviceName);
  OpenCLQueue(const OpenCLQueue&) = delete;
  OpenCLQueue& operator=(const OpenCLQueue&) = delete;

  /**
   * @brief Add task in OpenCL in-order queue.
   *
   * Code example:
   * @code{.cpp}
   * // opencl_program.cl
   * kernel void vector_add(global const int *A, global const int *B,
   *                        global int *C) {
   *   int i = get_global_id(0);
   *   C[i] = A[i] + B[i];
   * }
   *
   * // your_source_file.cc
   * // ...
   * const int vector_size = 1024;
   * shared_array<int> a(new int[vector_size], vector_size);
   * shared_array<int> b(new int[vector_size], vector_size);
   * shared_array<int> c(new int[vector_size], vector_size);
   * // init a and b vectors...
   * cl_data_t<int, oclalgo::IN>  d_a(a);
   * cl_data_t<int, oclalgo::IN>  d_b(b);
   * cl_data_t<int, oclalgo::OUT> d_c(c);
   * auto future = queue.AddTask("vector_add.cl", "vector_add", "", cl::NullRange,
   *                             cl::NDRange(el_count), cl::NullRange, d_a, d_b, d_c);
   * std::tie(c) = future.get();
   * @endcode
   *
   * @param pathToProgram path to OpenCL program
   * @param kernelName kernel name in OpenCL program (without args and braces)
   * @param compileOptions compile options to build OpenCL program
   * @param offset offset for OpenCL global
   * @param global size of OpenCL global grid
   * @param local size of OpenCL group grid
   * @param args arguments to launch OpenCL kernel
   */
  template<typename... Args>
  auto AddTask(const std::string& pathToProgram, const std::string& kernelName,
               const std::string& compileOptions, const cl::NDRange& offset,
               const cl::NDRange& global, const cl::NDRange& local,
               const Args&... args)
               -> cl_future<decltype(ComposeOutTuple(args...))>;

  /** @brief Return info about available OpenCL devices. */
  static std::string OpenCLInfo(bool completeInfo);
  /** @brief Return status string by the error code. */
  static std::string StatusStr(cl_int status);

 private:
  /** @brief Return info about OpenCL platform by platformId. */
  static std::string PlatformInfo(const std::vector<cl::Platform>& platforms,
                                  size_t platformId, bool completeInfo);
  /** @brief Return info about OpenCL device by deviceId. */
  static std::string DeviceInfo(const std::vector<cl::Device>& devices,
                                size_t deviceId, bool completeInfo);

  template<typename First, typename ... Tail>
  void SetKernelArgs(uint32_t argIndex, cl::Kernel* kernel,
                     std::vector<cl::Buffer>* buffers, const First& clData,
                     const Tail&... args) const;
  template<typename First>
  void SetKernelArgs(uint32_t argIndex, cl::Kernel* kernel,
                     std::vector<cl::Buffer>* buffers,
                     const First& clData) const;
  void SetKernelArgs(uint32_t argIndex, cl::Kernel* kernel,
                     std::vector<cl::Buffer>* buffers) const;
  template<typename First, typename ... Tail>
  void GetResults(uint32_t argIndex, const std::vector<cl::Buffer>& buffers,
                  cl::Event* event, const First& clData,
                  const Tail&... args) const;
  template<typename First>
  void GetResults(uint32_t argIndex, const std::vector<cl::Buffer>& buffers,
                  cl::Event* event, const First& clData) const;
  void GetResults(uint32_t argIndex, const std::vector<cl::Buffer>& buffers,
                  cl::Event* event) const;

  size_t platform_id_;
  size_t device_id_;
  std::vector<cl::Platform> platforms_;
  std::vector<cl::Device> devices_;
  cl::Context context_;
  cl::CommandQueue queue_;
  std::unordered_map<std::string, cl::Program> programs_;
  std::unordered_map<std::string, cl::Kernel> kernels_;
};

template<typename T, oclalgo::DataType DT>
typename std::enable_if<DT != OUT && DT != IN_OUT,
std::tuple<>>::type ReturnOutData(const cl_data_t<T, DT>&) {
  return std::tuple<>();
}

template<typename T, oclalgo::DataType DT>
typename std::enable_if<DT == OUT || DT == IN_OUT,
std::tuple<shared_array<T>>>::type ReturnOutData(const cl_data_t<T, DT>& data) {
   return std::make_tuple(data.host_array);
}

template<typename First>
auto ComposeOutTuple(const First& data) -> decltype(ReturnOutData(data)) {
  return ReturnOutData(data);
}

template<typename First, typename ... Tail>
auto ComposeOutTuple(const First& data, const Tail& ... args)
    -> decltype(std::tuple_cat(ReturnOutData(data), ComposeOutTuple(args...))) {
  return std::tuple_cat(ReturnOutData(data), ComposeOutTuple(args...));
}

template<typename... Args>
auto OpenCLQueue::AddTask(const std::string& pathToProgram,
                          const std::string& kernelName,
                          const std::string& compileOptions,
                          const cl::NDRange& offset, const cl::NDRange& global,
                          const cl::NDRange& local, const Args&... args)
                          -> cl_future<decltype(ComposeOutTuple(args...))> {
  // load source code
  bool build_flag = false;
  std::string program_id = pathToProgram + compileOptions;
  if (programs_.find(program_id) == programs_.end()) {
    std::ifstream source_file(pathToProgram);
    std::string source_code(std::istreambuf_iterator<char>(source_file),
                            (std::istreambuf_iterator<char>()));
    cl::Program::Sources cl_source(
        1, std::make_pair(source_code.c_str(), source_code.length() + 1));

    // build program from source code
    programs_[program_id] = cl::Program(context_, cl_source);
    try {
      programs_[program_id].build({ devices_[device_id_] },
                                  compileOptions.c_str());
    } catch (const cl::Error& e) {
      std::cout << "Build log:" << std::endl
          << programs_[pathToProgram].
          getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices_[device_id_]) << std::endl;
      throw(e);
    }
    build_flag = true;
  }
  // create OpenCL kernel
  std::string kernel_id = program_id + "; " + kernelName;
  if (build_flag || kernels_.find(kernel_id) == kernels_.end()) {
    kernels_[kernel_id] = cl::Kernel(programs_[program_id],
                                         kernelName.c_str());
  }

  // fill OpenCL command queue
  std::vector<cl::Buffer> buffers;
  cl::Event event;
  SetKernelArgs(0, &kernels_[kernel_id], &buffers, args...);
  queue_.enqueueNDRangeKernel(kernels_[kernel_id], offset, global, local,
                              nullptr, &event);
  GetResults(0, buffers, &event, args...);

  // providing output data into the cl_future object
  auto t = ComposeOutTuple(args...);
  return cl_future<decltype(t)>(t, buffers, event);
}

template<typename First, typename ... Tail>
void OpenCLQueue::SetKernelArgs(uint32_t argIndex, cl::Kernel* kernel,
                                std::vector<cl::Buffer>* buffers,
                                const First& clData,
                                const Tail&... args) const {
  SetKernelArgs(argIndex, kernel, buffers, clData);
  SetKernelArgs(++argIndex, kernel, buffers, args...);
}

template<typename First>
void OpenCLQueue::SetKernelArgs(uint32_t argIndex, cl::Kernel* kernel,
                                std::vector<cl::Buffer>* buffers,
                                const First& clData) const {
  cl::Buffer buffer;
  switch (First::io_type) {
    case oclalgo::IN:
      buffer = cl::Buffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          clData.host_array.memsize(), clData.host_array.get());

      buffers->push_back(buffer);
      kernel->setArg(argIndex, buffer);
      break;
    case oclalgo::IN_OUT:
      buffer = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          clData.host_array.memsize(), clData.host_array.get());
      buffers->push_back(buffer);
      kernel->setArg(argIndex, buffer);
      break;
    case oclalgo::OUT:
      buffer = cl::Buffer(context_, CL_MEM_WRITE_ONLY, clData.host_array.memsize());
      buffers->push_back(buffer);
      kernel->setArg(argIndex, buffer);
      break;
    case oclalgo::LOCALE:
      kernel->setArg(argIndex, cl::Local(clData.host_array.memsize()));
      break;
    case oclalgo::VAR:
      kernel->setArg(argIndex, clData.host_array[0]);
      break;
  }
}

template<typename First, typename ... Tail>
void OpenCLQueue::GetResults(uint32_t argIndex,
                             const std::vector<cl::Buffer>& buffers,
                             cl::Event* event, const First& clData,
                             const Tail&... args) const {
  GetResults(argIndex, buffers, event, clData);
  GetResults(++argIndex, buffers, event, args...);
}

template<typename First>
void OpenCLQueue::GetResults(uint32_t argIndex,
                             const std::vector<cl::Buffer>& buffers,
                             cl::Event* event, const First& clData) const {
  if (First::io_type == oclalgo::OUT || First::io_type == oclalgo::IN_OUT) {
    queue_.enqueueReadBuffer(buffers[argIndex], CL_FALSE, 0,
                             clData.host_array.memsize(), clData.host_array.get(),
                             nullptr, event);
  }
}

} // namespace oclalgo

#endif // OCLALGO_OPENCL_QUEUE_H_

/*! @file queue.h
 *  @brief Contains oclalgo::Queue class.
 *  @author Senin Dmitry <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  Class for simple OpenCL API usage.
 *  Based on OpencCL C++ Wrapper API. Creates one in-order OpenCL command queue.
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
 * Contains Queue, Matrix and DMatrix classes.
 *
 * Queue provides simple interface for starting OpenCL tasks.
 *
 * <i>Code example:</i>
 * @code{.cpp}
 * Queue queue("NVIDIA", "TITAN");
 *
 * // allocate host data
 * int size = 4096;
 * shared_array<int> a(size), b(size);
 *
 * // ...host data initiallization...
 *
 * // initialize device data
 * BufferArg buff_a(queue.CreateBuffer(BufferType::ReadOnly, a), ArgType::IN);
 * BufferArg buff_b(queue.CreateBuffer(BufferType::ReadOnly, b), ArgType::IN);
 * BufferArg buff_c(queue.CreateBuffer<int>(BufferType::WriteOnly, size), ArgType::OUT);
 *
 * // create OpenCL task
 * Task task = queue.CreateTask(program, kernel, options, buff_a, buff_b, buff_c);
 * // create OpenCL grid
 * Grid grid = Grid(cl::NDRange(size));
 * // enqueue task
 * auto future = queue.EnqueueTask(task, grid);
 *
 * // copy result (it can be another thread)
 * queue.memcpy(a, future.get()[0]);
 * @endcode
 *
 * Matrix and DMatrix classes is an example of Queue usage.
 *
 * @section Requirements
 *
 * <ul>
 * <li>C++11 compatible compiler</li>
 * <li>OpenCL version >= 1.1</li>
 * </ul>
 */

#ifndef INC_OCLALGO_QUEUE_H_
#define INC_OCLALGO_QUEUE_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cstdio>
#include <unordered_map>
#include <string>
#include <fstream>
#include <vector>

#include <oclalgo/shared_array.h>
#include <oclalgo/kernel_arg.h>
#include <oclalgo/task.h>
#include <oclalgo/grid.h>
#include <oclalgo/future.h>

namespace oclalgo {

/** @brief Enum of OpenCL buffer types. */
enum class BufferType { ReadOnly, WriteOnly, ReadWrite };

/** @brief Enum of task execution type (with blocking or not). */
enum class BlockingType { Block, Unblock };

/*!
 * @brief Class for simple execution of OpenCL kernels.
 *
 * Uses OpenCL C++ Wrapper API. Provides to launch an OpenCL task asynchronously
 * with in-order OpenCL queue. Synchronization is based on oclalgo::future
 * objects, which have similar interface and functionality as std::future.
 */
class Queue {
 public:
  /*!
   * @brief Creates Queue object using platform part name and device part name.
   *
   * Finding isn't case sensitive. If it can't find corresponding platform or
   * device, it throws an exception cl::Error.
   */
  Queue(const std::string& platformPartName,
        const std::string& devicePartName);

  /*!
   * @brief Creates Queue object using platform and device ID in your system.
   *
   * It throws exception if there is no platform or device with
   * corresponding id.
   */
  Queue(int platformId, int deviceId);

  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;

  /*!
   * @brief Creates Task object by corresponding program and kernel names.
   *
   * Arguments <i>args</i> should be passed in the same order as in OpenCL
   * kernel. They should be a simple types (int, double, float, char) or
   * objects of KernelArg class.
   *
   * @param programName path to OpenCL program source file (*.cl)
   * @param kernelName function name in OpenCL program (*.cl source file)
   * @param options compilation options used for building OpenCL program
   * @param args list of kernel arguments   *
   */
  template <typename... Args>
  Task CreateTask(const std::string& programName, const std::string& kernelName,
                  const std::string& options, const Args&... args);

  /** @brief Creates OpenCL buffer with corresponding size and OpenCL flags. */
  template <typename T>
  cl::Buffer CreateBuffer(cl_mem_flags flags, size_t size);

  /** @brief Creates OpenCL buffer with corresponding size and type. */
  template <typename T>
  cl::Buffer CreateBuffer(BufferType type, size_t size);

  /*!
   * @brief Creates OpenCL buffer by host array with corresponding OpenCL flags.
   */
  template <typename T>
  cl::Buffer CreateBuffer(cl_mem_flags flags, const shared_array<T>& array);

  /** @brief Creates OpenCL buffer by host array with corresponding type. */
  template <typename T>
  cl::Buffer CreateBuffer(BufferType type, const shared_array<T>& array);

  /** @brief Creates OpenCL local buffer with corresponding size. */
  template <typename T>
  cl::LocalSpaceArg CreateLocalBuffer(size_t size);

  /** @brief Copies host memory to cl::Buffer object. */
  template <typename T>
  future<cl::Buffer, shared_array<T>> memcpy(
      const cl::Buffer& buffer, const shared_array<T>& array,
      BlockingType block = BlockingType::Block, size_t offset = 0,
      const std::vector<cl::Event>* events = nullptr);

  /** @brief Copies cl::Buffer object to host memory. */
  template <typename T>
  future<shared_array<T>, cl::Buffer> memcpy(
      const shared_array<T>& array, const cl::Buffer& buffer,
      BlockingType block = BlockingType::Block, size_t offset = 0,
      const std::vector<cl::Event>* events = nullptr);

  /** @brief Starts task in OpenCL queue. */
  template <typename... Args>
  future<std::vector<cl::Buffer>, std::vector<cl::Buffer>> EnqueueTask(
      const Task& task, const Grid& grid, const Args&...);

  /** @brief Returns string corresponding to the error code. */
  static std::string StatusStr(cl_int code);

  /** @brief Returns cl::Platform object of this queue. */
  cl::Platform platform() const noexcept { return platform_; }
  /** @brief Returns platform ID of this queue in system. */
  int platform_id() const noexcept { return platform_id_; }
  /** @brief Returns name of OpenCL platform of this queue. */
  std::string PlatformName() const noexcept {
    return platform_.getInfo<CL_PLATFORM_NAME>();
  }

  /** @brief Returns cl::Device object of this queue. */
  cl::Device device() const noexcept { return device_; }
  /** @brief Returns device ID of this queue in system. */
  int device_id() const noexcept { return device_id_; }
  /** @brief Returns name of OpenCL device of this queue. */
  std::string DeviceName() const noexcept {
    return device_.getInfo<CL_DEVICE_NAME>();
  }

  /** @brief Returns cl::Context object of this queue. */
  cl::Context context() const noexcept { return context_; }
  /** @brief Returns cl::CommandQueue object of this queue. */
  cl::CommandQueue queue() const noexcept { return queue_; }

 private:
  cl::Platform platform_;
  cl::Device device_;
  int platform_id_;
  int device_id_;
  cl::Context context_;
  cl::CommandQueue queue_;
  std::unordered_map<std::string, cl::Program> programs_;
};

template <typename T>
cl::Buffer Queue::CreateBuffer(cl_mem_flags flags, size_t size) {
  return cl::Buffer(context_, flags, size * sizeof(T), nullptr);
}

template <typename T>
cl::Buffer Queue::CreateBuffer(BufferType type, size_t size) {
  cl::Buffer buffer;
  switch (type) {
    case BufferType::ReadOnly:
      buffer = cl::Buffer(context_, CL_MEM_READ_ONLY, size * sizeof(T));
      break;
    case BufferType::WriteOnly:
      buffer = cl::Buffer(context_, CL_MEM_WRITE_ONLY, size * sizeof(T));
      break;
    case BufferType::ReadWrite:
      buffer = cl::Buffer(context_, CL_MEM_READ_WRITE, size * sizeof(T));
      break;
  }
  return buffer;
}

template <typename T>
cl::Buffer Queue::CreateBuffer(cl_mem_flags flags,
                               const shared_array<T>& array) {
  return cl::Buffer(context_, flags, array.memsize(), array.get_raw());
}

template <typename T>
cl::Buffer Queue::CreateBuffer(BufferType type,
                               const shared_array<T>& array) {
  cl::Buffer buffer;
  switch (type) {
    case BufferType::ReadOnly:
      buffer = cl::Buffer(context_, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                          array.memsize(), array.get_raw());
      break;
    case BufferType::WriteOnly:
      buffer = cl::Buffer(context_, CL_MEM_WRITE_ONLY, array.memsize(),
                          nullptr);
      break;
    case BufferType::ReadWrite:
      buffer = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                          array.memsize(), array.get_raw());
      break;
  }
  return buffer;
}

template <typename T>
cl::LocalSpaceArg CreateLocalBuffer(size_t size) {
  return cl::Local(size * sizeof(T));
}

template <typename T>
future<cl::Buffer, shared_array<T>> Queue::memcpy(
    const cl::Buffer& buffer, const shared_array<T>& array, BlockingType block,
    size_t offset, const std::vector<cl::Event>* events) {
  cl::Event event;
  queue_.enqueueWriteBuffer(buffer,
                            block == BlockingType::Block ? CL_TRUE : CL_FALSE,
                            offset, array.memsize(),
                            array.get_raw(), events, &event);
  return future<cl::Buffer, shared_array<T>>(buffer, array, event);
}

template <typename T>
future<shared_array<T>, cl::Buffer> Queue::memcpy(
    const shared_array<T>& array, const cl::Buffer& buffer, BlockingType block,
    size_t offset, const std::vector<cl::Event>* events) {
  cl::Event event;
  queue_.enqueueReadBuffer(buffer,
                           block == BlockingType::Block ? CL_TRUE : CL_FALSE,
                           offset, array.memsize(),
                           array.get_raw(), events, &event);
  return future<shared_array<T>, cl::Buffer>(array, buffer, event);
}

template <typename... Args>
Task Queue::CreateTask(const std::string& programName,
                       const std::string& kernelName,
                       const std::string& options, const Args&... args) {
  char buff[512] = {0};
  std::snprintf(buff, sizeof(buff), "program=\"%s\"\noptions=\"%s\"",
                programName.c_str(), options.c_str());
  std::string program_id(buff);
  cl::Program program;

  if (programs_.find(program_id) == programs_.end()) {
    std::ifstream source_file(programName);
    std::string source_code(std::istreambuf_iterator<char>(source_file),
                            (std::istreambuf_iterator<char>()));
    cl::Program::Sources cl_source(1, std::make_pair(source_code.c_str(),
                                                     source_code.length() + 1));
    // build program from source code
    program = cl::Program(context_, cl_source);
    try {
      program.build({ device_ }, options.c_str());
    } catch (const cl::Error& e) {
      std::printf("Build log:\n%s\n",
                  program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_).c_str());
      throw(e);
    }
    programs_[program_id] = program;
  } else {
    program = programs_[program_id];
  }

  cl::Kernel kernel = cl::Kernel(program, kernelName.c_str());
  return Task(kernel, args...);
}

std::vector<cl::Event> ExtractEvents();

template <typename T>
std::vector<cl::Event> ExtractEvents(const T& data);

template <typename First, typename... Tail>
std::vector<cl::Event> ExtractEvents(const First& first, const Tail&... tail);

template <typename... Args>
future<std::vector<cl::Buffer>, std::vector<cl::Buffer>> Queue::EnqueueTask(
    const Task& task, const Grid& grid, const Args&... args) {
  std::vector<cl::Event> events = ExtractEvents(args...), *pevents = nullptr;
  if (events.size()) pevents = &events;
  cl::Event event;
  queue_.enqueueNDRangeKernel(task.kernel(), grid.offset(), grid.global(),
                              grid.local(), pevents, &event);
  return future<std::vector<cl::Buffer>, std::vector<cl::Buffer>>
      (task.output(), task.buffers(), event);
}

template <typename T, typename U>
std::vector<cl::Event> ExtractEvents(const future<T, U>& f) {
  return std::vector<cl::Event>({ f.event() });
}

template <typename T, typename U>
std::vector<cl::Event> ExtractEvents(const std::vector<future<T, U>>& f) {
  std::vector<cl::Event> events;
  for (const auto& el : f)
    events.push_back(el.event());
  return events;
}

template <typename First, typename... Tail>
std::vector<cl::Event> ExtractEvents(const First& first, const Tail&... tail) {
  std::vector<cl::Event> events = ExtractEvents(first);
  std::vector<cl::Event> tail_events = ExtractEvents(tail...);
  events.insert(events.end(), tail_events.begin(), tail_events.end());
  return events;
}

}  // namespace oclalgo

#endif  // INC_OCLALGO_QUEUE_H_

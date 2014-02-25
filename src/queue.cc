/*! @file queue.cc
 *  @brief Queue class implementation.
 *  @author Senin Dmitry <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "inc/oclalgo/queue.h"

#include <algorithm>
#include <string>

namespace oclalgo {

Queue::Queue(const std::string& platformPartName,
             const std::string& devicePartName) {
  // convert input strings to upper case
  std::string pl_name = platformPartName, dev_name = devicePartName;
  std::transform(pl_name.begin(), pl_name.end(), pl_name.begin(), ::toupper);
  std::transform(dev_name.begin(), dev_name.end(), dev_name.begin(), ::toupper);
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  // try to find corresponding OpenCL platform
  auto pl_it = find_if(platforms.begin(), platforms.end(),
                       [&pl_name] (const cl::Platform& p) {
    std::string name = p.getInfo<CL_PLATFORM_NAME>();
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);
    return name.find(pl_name) != std::string::npos;
  });
  if (pl_it != platforms.end()) {
    platform_id_ = std::distance(platforms.begin(), pl_it);
    platform_ = *pl_it;
  } else {
    throw cl::Error(CL_INVALID_PLATFORM, "can't find OpenCL platform");
  }

  cl_context_properties properties[3] = {
      CL_CONTEXT_PLATFORM,
      reinterpret_cast<cl_context_properties>(platform_()),
      0
  };
  context_ = cl::Context(CL_DEVICE_TYPE_ALL, properties);
  std::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();

  // try to find corresponding OpenCL platform
  auto dev_it = find_if(devices.begin(), devices.end(),
                        [&dev_name] (const cl::Device& d) {
    std::string name = d.getInfo<CL_DEVICE_NAME>();
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);
    return name.find(dev_name) != std::string::npos;
  });
  if (dev_it != devices.end()) {
    device_id_ = std::distance(devices.begin(), dev_it);
    device_ = *dev_it;
  } else {
    throw cl::Error(CL_INVALID_DEVICE, "can't find OpenCL device");
  }

  queue_ = cl::CommandQueue(context_, device_);
}

Queue::Queue(int platformId, int deviceId) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  // try to get OpenCL platform with corresponding platformId
  if (static_cast<int>(platforms.size()) <= platformId)
    throw cl::Error(CL_INVALID_PLATFORM, "can't find OpenCL platform");
  platform_id_ = platformId;
  platform_ = platforms[platform_id_];

  cl_context_properties properties[3] = {
      CL_CONTEXT_PLATFORM,
      reinterpret_cast<cl_context_properties>(platforms[platform_id_]()),
      0
  };
  context_ = cl::Context(CL_DEVICE_TYPE_ALL, properties);
  std::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();

  // try to get OpenCL platform with corresponding deviceId
  if (static_cast<int>(devices.size()) <= deviceId)
    throw cl::Error(CL_INVALID_DEVICE, "can't find OpenCL device");
  device_id_ = deviceId;
  device_ = devices[device_id_];

  queue_ = cl::CommandQueue(context_, device_);
}

std::string Queue::StatusStr(cl_int code) {
  switch (code) {
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case -1001:
      return "-1001 (OpenCL is not configured or unavailable)";
    default: {
      char str[256];
      snprintf(str, sizeof(str), "a not recognized error code (%i)", code);
      return static_cast<std::string>(str);
    }
  }
}

std::vector<cl::Event> ExtractEvents() { return std::vector<cl::Event>(); }

std::vector<cl::Event> ExtractEvents(const cl::Event& event) {
  return std::vector<cl::Event>({ event });
}

std::vector<cl::Event> ExtractEvents(const std::vector<cl::Event>& events) {
  return events;
}


}  // namespace oclalgo

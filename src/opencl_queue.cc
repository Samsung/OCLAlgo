/*! @file opencl_queue.cc
 *  @brief OpenCLQueue class realization.
 *  @author Senin Dmitry <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "inc/oclalgo/opencl_queue.h"

#include <cstdint>
#include <iostream>
#include <sstream>

namespace oclalgo {

OpenCLQueue::OpenCLQueue(const std::string& platformName,
                         const std::string& deviceName) {
  cl::Platform::get(&platforms_);
  bool is_found = false;
  for (size_t i = 0; i < platforms_.size(); ++i) {
    std::string name = "";
    platforms_[i].getInfo(CL_PLATFORM_NAME, &name);
    if (name.find(platformName) != std::string::npos) {
      platform_id_ = i;
      is_found = true;
      break;
    }
  }
  if (!is_found) {
    throw cl::Error(CL_INVALID_PLATFORM,
                    "(OpenCLQueue) error: can't select OpenCL platform");
  }

  cl_context_properties properties[3];
  cl_platform_id platformIDs[platforms_.size()];
  if (clGetPlatformIDs(platforms_.size(), platformIDs, NULL) != CL_SUCCESS) {
    throw cl::Error(CL_INVALID_PLATFORM,
                    "(OpenCLQueue) error: clGetPlatformIDs()"
                    " can't get OpenCL platform IDs");
  }
  properties[0] = CL_CONTEXT_PLATFORM;
  properties[1] =
      reinterpret_cast<cl_context_properties>(platformIDs[platform_id_]);
  properties[2] = 0;
  context_ = cl::Context(CL_DEVICE_TYPE_ALL, properties);
  devices_ = context_.getInfo<CL_CONTEXT_DEVICES>();

  is_found = false;
  for (size_t j = 0; j < devices_.size(); ++j) {
    std::string name = "";
    devices_[j].getInfo(CL_DEVICE_NAME, &name);
    if (name.find(deviceName) != std::string::npos) {
      device_id_ = j;
      is_found = true;
      break;
    }
  }
  if (!is_found) {
    throw cl::Error(CL_INVALID_DEVICE,
                    "(OpenCLQueue) error: can't select OpenCL device");
  }
  queue_ = cl::CommandQueue(context_, devices_[device_id_]);
}

std::string OpenCLQueue::OpenCLInfo(bool completeInfo) {
  std::stringstream ss;
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  for (size_t i = 0; i < platforms.size(); ++i) {
    ss << OpenCLQueue::PlatformInfo(platforms, i, completeInfo);

    // Create context for all devices on current platform
    cl_context_properties properties[3];
    cl_platform_id platformIDs[platforms.size()];
    if (clGetPlatformIDs(platforms.size(), platformIDs, NULL) != CL_SUCCESS) {
      throw cl::Error(CL_INVALID_PLATFORM,
                      "(OpenCLQueue) error: clGetPlatformIDs()"
                      " can't get OpenCL platform IDs");
    }
    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = reinterpret_cast<cl_context_properties>(platformIDs[i]);
    properties[2] = 0;
    cl::Context context(CL_DEVICE_TYPE_ALL, properties);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    ss << "---------------- DEVICES ----------------" << std::endl;
    for (size_t j = 0; j < devices.size(); ++j) {
      ss << OpenCLQueue::DeviceInfo(devices, j, completeInfo) << std::endl;
    }
    ss << "-----------------------------------------" << std::endl;
    ss << std::endl;
  }
  return ss.str();
}

#define INFO_TO_OSTREAM(ostream, obj, clConst) {                        \
  (ostream) << #clConst ": " << (obj).getInfo<clConst>() << std::endl;  \
}

std::string OpenCLQueue::PlatformInfo(
    const std::vector<cl::Platform>& platforms, size_t platformId,
    bool completeInfo) {
  std::stringstream ss;
  INFO_TO_OSTREAM(ss, platforms[platformId], CL_PLATFORM_NAME);
  INFO_TO_OSTREAM(ss, platforms[platformId], CL_PLATFORM_VENDOR);
  INFO_TO_OSTREAM(ss, platforms[platformId], CL_PLATFORM_VERSION);
  if (completeInfo) {
    INFO_TO_OSTREAM(ss, platforms[platformId], CL_PLATFORM_EXTENSIONS);
    INFO_TO_OSTREAM(ss, platforms[platformId], CL_PLATFORM_PROFILE);
  }
  return ss.str();
}

std::string OpenCLQueue::DeviceInfo(const std::vector<cl::Device>& devices,
                                    size_t deviceId, bool completeInfo) {
  std::stringstream ss;
  INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_NAME);
  INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_VERSION);
  INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_VENDOR);
  INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_OPENCL_C_VERSION);
  INFO_TO_OSTREAM(ss, devices[deviceId], CL_DRIVER_VERSION);

  cl_device_type dev_type = devices[deviceId].getInfo<CL_DEVICE_TYPE>();
  ss << "CL_DEVICE_TYPE: ";
  switch (dev_type) {
    case CL_DEVICE_TYPE_CPU:
      ss << "CL_DEVICE_TYPE_CPU" << std::endl;
      break;
    case CL_DEVICE_TYPE_GPU:
      ss << "CL_DEVICE_TYPE_GPU" << std::endl;
      break;
    case CL_DEVICE_TYPE_ACCELERATOR:
      ss << "CL_DEVICE_TYPE_ACCELERATOR" << std::endl;
      break;
    case CL_DEVICE_TYPE_DEFAULT:
      ss << "CL_DEVICE_TYPE_DEFAULT" << std::endl;
      break;
  }

  if (completeInfo) {
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_MAX_COMPUTE_UNITS);
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_GLOBAL_MEM_SIZE);
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_LOCAL_MEM_SIZE);
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_MAX_CONSTANT_ARGS);
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_MAX_PARAMETER_SIZE);
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_MAX_WORK_GROUP_SIZE);
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);

    std::vector<size_t> param;
    devices[deviceId].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &param);
    ss << "CL_DEVICE_MAX_WORK_ITEM_SIZES: (";
    for (size_t i = 0; i < param.size() - 1; ++i)
      ss << param[i] << "; ";
    ss << *param.rbegin() << ")" << std::endl;

    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_PROFILE);
    INFO_TO_OSTREAM(ss, devices[deviceId], CL_DEVICE_EXTENSIONS);
  }
  return ss.str();
}

#undef INFO_TO_OSTREAM

std::string OpenCLQueue::StatusStr(cl_int status) {
  switch (status) {
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
      static char str[256];
      snprintf(str, sizeof(str), "a not recognized error code (%i)", status);
      return str;
    }
  }
}

} // namespace oclalgo

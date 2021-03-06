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

/*! @file queue.cc
 *  @brief Unit tests for oclalgo::Queue.
 *  @author Dmitry Senin <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  Tests for OpenCLQueue class
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include <algorithm>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include "src/gtest_main.cc"
#include "inc/oclalgo/queue.h"

std::string platform_name = "NVIDIA";
std::string device_name = "GeForce";

TEST(Queue, VectorAdd) {
  try {
    // create OpenCL queue for sync/async task running using
    // part platform and device names
    oclalgo::Queue queue(platform_name, device_name);

    // create and initialize input shared arrays
    int size = 128;
    oclalgo::shared_array<int> a(size), b(size);
    for (int i = 0; i < size; ++i) {
      a[i] = i;
      b[i] = size - i;
    }

    // initialize OpenCl kernel arguments
    using oclalgo::ArgType;
    using oclalgo::BufferArg;
    BufferArg a_arg = queue.CreateKernelArg(a, ArgType::IN);
    BufferArg b_arg = queue.CreateKernelArg(b, ArgType::IN);
    BufferArg c_arg = queue.CreateKernelArg<int>(size, ArgType::OUT);

    // create task using OpenCL program and kernel names, compilation options
    // and arguments in the same order as in OpenCL kernel
    oclalgo::Task task = queue.CreateTask("vector.cl", "vector_add", "",
                                          a_arg, b_arg, c_arg);

    // create grid to define dimensions of OpenCL task
    // in global and local (group size) space
    oclalgo::Grid grid = oclalgo::Grid(cl::NDRange(size));

    // enqueue OpenCL task (EnqueueTask() returns oclalgo::future object
    // for async task running)
    auto ocl_res = queue.EnqueueTask(task, grid);

    // copy device memory with result to host
    // (ocl_res.get() waits while OpenCL finished task
    // and returns std::vector with output OpenCL buffers,
    // which was marked as ArgType::OUT or ArgType::IN_OUT when was created)
    queue.memcpy(a, ocl_res.get()[0]);

    // check result
    auto it = std::find_if(a.get_raw(), a.get_raw() + a.size(),
                           [size](int x) { return x != size; });
    ASSERT_EQ(a.get_raw() + a.size(), it);
  } catch (const cl::Error& e) {
    std::cerr << e.what() << " (err_code = "
              << oclalgo::Queue::StatusStr(e.err()) << ")" << std::endl;
    throw e;
  }
}

TEST(Queue, MatrixAdd) {
  using oclalgo::BufferType;
  using oclalgo::BufferArg;
  using oclalgo::ArgType;
  try {
    oclalgo::Queue queue(platform_name, device_name);
    int rows = 128, cols = 512, size = rows * cols;
    oclalgo::shared_array<int> a(size), b(size);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        a[i * cols + j] = i * cols + j;
        b[i * cols + j] = cols * rows - (i * cols + j);
      }
    }

    BufferArg a_arg = queue.CreateKernelArg(a, ArgType::IN);
    BufferArg b_arg = queue.CreateKernelArg(b, ArgType::IN);
    BufferArg c_arg = queue.CreateKernelArg<int>(size, ArgType::OUT);

    oclalgo::Task task = queue.CreateTask("matrix.cl", "matrix_add", "", a_arg,
                                          b_arg, c_arg);
    oclalgo::Grid grid = oclalgo::Grid(cl::NDRange(rows, cols));
    auto future = queue.EnqueueTask(task, grid);
    queue.memcpy(a, future.get()[0]);

    // check results
    auto it = std::find_if(a.get_raw(), a.get_raw() + a.size(),
                           [size](int x) { return x != size; });
    ASSERT_EQ(a.get_raw() + a.size(), it);
  } catch (const cl::Error& e) {
    std::cerr << e.what() << " (err_code = "
              << oclalgo::Queue::StatusStr(e.err()) << ")" << std::endl;
    throw e;
  }
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum { ROW, COL } PackingType;

typedef struct tag_matrix_param_t {
  int rows;
  int cols;
  PackingType packing;
} matrix_param_t;

#ifdef __cplusplus
}
#endif  // __cplusplus

TEST(Queue, MatrixMul_Row) {
  using oclalgo::BufferType;
  using oclalgo::BufferArg;
  using oclalgo::ArgType;
  try {
    oclalgo::Queue queue(platform_name, device_name);
    matrix_param_t m1_param, m2_param;
    m1_param.cols = 4, m1_param.rows = 4, m1_param.packing = PackingType::ROW;
    m2_param.cols = 8, m2_param.rows = 4, m2_param.packing = PackingType::ROW;
    oclalgo::shared_array<int> m1(m1_param.cols * m1_param.rows);
    oclalgo::shared_array<int> m2(m2_param.cols * m2_param.rows);

    for (int i = 0; i < m1_param.rows * m1_param.cols; ++i)
      m1[i] = i + 1;
    for (int i = 0; i < m2_param.rows * m2_param.cols; ++i)
      m2[i] = i + 1;

    BufferArg A = queue.CreateKernelArg(m1, ArgType::IN);
    cl::Buffer A_param = cl::Buffer(queue.context(),
                                    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                    sizeof(m1_param), &m1_param);
    BufferArg A_param_arg(A_param, ArgType::IN);
    BufferArg B = queue.CreateKernelArg(m2, ArgType::IN);
    cl::Buffer B_param = cl::Buffer(queue.context(),
                                    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                    sizeof(m2_param), &m2_param);
    BufferArg B_param_arg(B_param, ArgType::IN);
    BufferArg C = queue.CreateKernelArg<int>(m1_param.rows * m2_param.cols,
                                             ArgType::OUT);

    oclalgo::Task task = queue.CreateTask("matrix.cl", "matrix_mul",
                                          "-D BLOCK_SIZE=2 -D VAR_TYPE=int", A,
                                          A_param_arg, B, B_param_arg, C);
    oclalgo::Grid grid = oclalgo::Grid(
        cl::NDRange(m2_param.cols, m1_param.rows), cl::NDRange(2, 2));
    auto future = queue.EnqueueTask(task, grid);

    queue.memcpy(m2, future.get()[0]);
    int gold_res[] = { 170, 180, 190, 200,  210,  220,  230,  240,
                     378, 404, 430, 456,  482,  508,  534,  560,
                     586, 628, 670, 712,  754,  796,  838,  880,
                     794, 852, 910, 968, 1026, 1084, 1142, 1200 };
    for (int i = 0; i < m1_param.rows * m2_param.cols; ++i)
      ASSERT_EQ(gold_res[i], m2[i]);
  } catch (const cl::Error& e) {
    std::cerr << e.what() << " (err_code = "
        << oclalgo::Queue::StatusStr(e.err()) << ")" << std::endl;
    throw e;
  }
}

TEST(Queue, MatrixMul_Col) {
  using oclalgo::BufferType;
  using oclalgo::BufferArg;
  using oclalgo::ArgType;
  try {
    oclalgo::Queue queue(platform_name, device_name);
    matrix_param_t m1_param, m2_param;
    m1_param.cols = 4, m1_param.rows = 4, m1_param.packing = PackingType::COL;
    m2_param.cols = 8, m2_param.rows = 4, m2_param.packing = PackingType::ROW;
    oclalgo::shared_array<int> m1(m1_param.cols * m1_param.rows);
    oclalgo::shared_array<int> m2(m2_param.cols * m2_param.rows);

    for (int i = 0; i < m1_param.rows; ++i)
      for (int j = 0; j < m1_param.cols; ++j)
        m1[j * m1_param.rows + i] = i * m1_param.cols + j + 1;
    for (int i = 0; i < m2_param.rows * m2_param.cols; ++i)
      m2[i] = i + 1;

    BufferArg A = queue.CreateKernelArg(m1, ArgType::IN);
    cl::Buffer A_param = cl::Buffer(queue.context(),
                                    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                    sizeof(m1_param), &m1_param);
    BufferArg A_param_arg(A_param, ArgType::IN);
    BufferArg B = queue.CreateKernelArg(m2, ArgType::IN);
    cl::Buffer B_param = cl::Buffer(queue.context(),
                                    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                    sizeof(m2_param), &m2_param);
    BufferArg B_param_arg(B_param, ArgType::IN);
    BufferArg C = queue.CreateKernelArg<int>(m1_param.rows * m2_param.cols,
                                             ArgType::OUT);

    oclalgo::Task task = queue.CreateTask("matrix.cl", "matrix_mul",
                                          "-D BLOCK_SIZE=2 -D VAR_TYPE=int", A,
                                          A_param_arg, B, B_param_arg, C);
    oclalgo::Grid grid = oclalgo::Grid(cl::NDRange(m2_param.cols,
                                                   m1_param.rows),
                                       cl::NDRange(2, 2));
    auto future = queue.EnqueueTask(task, grid);

    queue.memcpy(m2, future.get()[0]);
    int gold_res[] = { 170, 180, 190, 200,  210,  220,  230,  240,
                     378, 404, 430, 456,  482,  508,  534,  560,
                     586, 628, 670, 712,  754,  796,  838,  880,
                     794, 852, 910, 968, 1026, 1084, 1142, 1200 };
    for (int i = 0; i < m1_param.rows * m2_param.cols; ++i)
      ASSERT_EQ(gold_res[i], m2[i]);
  } catch (const cl::Error& e) {
    std::cerr << e.what() << " (err_code = "
        << oclalgo::Queue::StatusStr(e.err()) << ")" << std::endl;
    throw e;
  }
}

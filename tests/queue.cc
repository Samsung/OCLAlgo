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
  using oclalgo::BufferType;
  using oclalgo::BufferArg;
  using oclalgo::ArgType;
  try {
    oclalgo::Queue queue(platform_name, device_name);
    const int size = 1024;
    oclalgo::shared_array<int> a(size), b(size);
    for (int i = 0; i < size; ++i) {
      a[i] = i;
      b[i] = size - i;
    }

    BufferArg a_arg(queue.CreateBuffer(oclalgo::BufferType::ReadOnly, a),
                    ArgType::IN);
    BufferArg b_arg(queue.CreateBuffer(oclalgo::BufferType::ReadOnly, b),
                    ArgType::IN);
    BufferArg c_arg(queue.CreateBuffer<int>(BufferType::WriteOnly, size),
                    ArgType::OUT);

    oclalgo::Task task = queue.CreateTask("vector.cl", "vector_add", "", a_arg,
                                          b_arg, c_arg);
    oclalgo::Grid grid = oclalgo::Grid(cl::NDRange(size));
    auto future = queue.EnqueueTask(task, grid);
    queue.memcpy(a, future.get()[0]);

    // check results
    auto it = std::find_if(a.get_raw(), a.get_raw() + a.size(),
                           [size](int x) { return x != size; });
    ASSERT_EQ(a.get_raw() + a.size(), it);
  } catch (const cl::Error& e) {
    std::cerr << e.what() << " (err_code = "
              << oclalgo::Queue::StatusStr(e.err()) << ")" << std::endl;
    ASSERT_FALSE(true);
  }
}

TEST(Queue, MatrixAdd) {
  using oclalgo::BufferType;
  using oclalgo::BufferArg;
  using oclalgo::ArgType;
  try {
    oclalgo::Queue queue(platform_name, device_name);
    int rows = 1024, cols = 2048, size = rows * cols;
    oclalgo::shared_array<int> a(size), b(size);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        a[i * cols + j] = i * cols + j;
        b[i * cols + j] = cols * rows - (i * cols + j);
      }
    }

    BufferArg a_arg(queue.CreateBuffer(BufferType::ReadOnly, a), ArgType::IN);
    BufferArg b_arg(queue.CreateBuffer(BufferType::ReadOnly, b), ArgType::IN);
    BufferArg c_arg(queue.CreateBuffer<int>(BufferType::WriteOnly, size),
                    ArgType::OUT);

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
    ASSERT_FALSE(true);
  }
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum { ROW, COL } DataDir;

typedef struct tag_matrix_param_t {
  int rows;
  int cols;
  DataDir dir;
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
    matrix_param_t A_param, B_param;
    A_param.cols = 4, A_param.rows = 4, A_param.dir = DataDir::ROW;
    B_param.cols = 8, B_param.rows = 4, B_param.dir = DataDir::ROW;
    oclalgo::shared_array<int> m1(A_param.cols * A_param.rows);
    oclalgo::shared_array<int> m2(B_param.cols * B_param.rows);

    for (int i = 0; i < A_param.rows * A_param.cols; ++i)
      m1[i] = i + 1;
    for (int i = 0; i < B_param.rows * B_param.cols; ++i)
      m2[i] = i + 1;

    BufferArg A(queue.CreateBuffer(BufferType::ReadOnly, m1), ArgType::IN);
    cl::Buffer pA = cl::Buffer(queue.context(),
                               CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                               sizeof(matrix_param_t), &A_param);
    BufferArg buff_pA(pA, ArgType::IN);
    BufferArg B(queue.CreateBuffer(BufferType::ReadOnly, m2), ArgType::IN);
    cl::Buffer pB = cl::Buffer(queue.context(),
                               CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                               sizeof(matrix_param_t), &B_param);
    BufferArg buff_pB(pB, ArgType::IN);
    BufferArg C(queue.CreateBuffer<int>(BufferType::WriteOnly,
                                        A_param.rows * B_param.cols),
                ArgType::OUT);

    oclalgo::Task task = queue.CreateTask("matrix.cl", "matrix_mul",
                                 "-D BLOCK_SIZE=2 -D VAR_TYPE=int",
                                 A, buff_pA, B, buff_pB, C);
    oclalgo::Grid grid = oclalgo::Grid(cl::NDRange(B_param.cols, A_param.rows),
                                       cl::NDRange(2, 2));
    auto future = queue.EnqueueTask(task, grid);

    queue.memcpy(m2, future.get()[0]);
    int gold_res[] = { 170, 180, 190, 200,  210,  220,  230,  240,
                     378, 404, 430, 456,  482,  508,  534,  560,
                     586, 628, 670, 712,  754,  796,  838,  880,
                     794, 852, 910, 968, 1026, 1084, 1142, 1200 };
    for (int i = 0; i < A_param.rows * B_param.cols; ++i)
      ASSERT_EQ(gold_res[i], m2[i]);
  } catch (const cl::Error& e) {
    std::cerr << e.what() << " (err_code = "
        << oclalgo::Queue::StatusStr(e.err()) << ")" << std::endl;
    ASSERT_FALSE(true);
  }
}

TEST(Queue, MatrixMul_Col) {
  using oclalgo::BufferType;
  using oclalgo::BufferArg;
  using oclalgo::ArgType;
  try {
    oclalgo::Queue queue(platform_name, device_name);
    matrix_param_t A_param, B_param;
    A_param.cols = 4, A_param.rows = 4, A_param.dir = DataDir::COL;
    B_param.cols = 8, B_param.rows = 4, B_param.dir = DataDir::ROW;
    oclalgo::shared_array<int> m1(A_param.cols * A_param.rows);
    oclalgo::shared_array<int> m2(B_param.cols * B_param.rows);

    for (int i = 0; i < A_param.rows; ++i)
      for (int j = 0; j < A_param.cols; ++j)
        m1[j * A_param.rows + i] = i * A_param.cols + j + 1;
    for (int i = 0; i < B_param.rows * B_param.cols; ++i)
      m2[i] = i + 1;

    BufferArg A(queue.CreateBuffer(BufferType::ReadOnly, m1), ArgType::IN);
    cl::Buffer pA = cl::Buffer(queue.context(),
                               CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                               sizeof(matrix_param_t), &A_param);
    BufferArg buff_pA(pA, ArgType::IN);
    BufferArg B(queue.CreateBuffer(BufferType::ReadOnly, m2), ArgType::IN);
    cl::Buffer pB = cl::Buffer(queue.context(),
                               CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                               sizeof(matrix_param_t), &B_param);
    BufferArg buff_pB(pB, ArgType::IN);
    BufferArg C(queue.CreateBuffer<int>(BufferType::WriteOnly,
                                        A_param.rows * B_param.cols),
                ArgType::OUT);

    oclalgo::Task task = queue.CreateTask("matrix.cl", "matrix_mul",
                                 "-D BLOCK_SIZE=2 -D VAR_TYPE=int",
                                 A, buff_pA, B, buff_pB, C);
    oclalgo::Grid grid = oclalgo::Grid(cl::NDRange(B_param.cols, A_param.rows),
                                       cl::NDRange(2, 2));
    auto future = queue.EnqueueTask(task, grid);

    queue.memcpy(m2, future.get()[0]);
    int gold_res[] = { 170, 180, 190, 200,  210,  220,  230,  240,
                     378, 404, 430, 456,  482,  508,  534,  560,
                     586, 628, 670, 712,  754,  796,  838,  880,
                     794, 852, 910, 968, 1026, 1084, 1142, 1200 };
    for (int i = 0; i < A_param.rows * B_param.cols; ++i)
      ASSERT_EQ(gold_res[i], m2[i]);
  } catch (const cl::Error& e) {
    std::cerr << e.what() << " (err_code = "
        << oclalgo::Queue::StatusStr(e.err()) << ")" << std::endl;
    ASSERT_FALSE(true);
  }
}

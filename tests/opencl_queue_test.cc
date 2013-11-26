/*! @file openclqueue_tests.cc
 *  @brief OpenCLQueue class tests.
 *  @author Dmitry Senin <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  Tests for OpenCLQueue class
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include <iostream>
#include <string>
#include <memory>
#include <gtest/gtest.h>
#include "src/gtest_main.cc"
#include <oclalgo/opencl_queue.h>

TEST(OpenCLQueue, VectorAdd) {
  using namespace oclalgo;
  try {
    OpenCLQueue queue("Intel(R) OpenCL", "Intel(R)");
    const int el_count = 1024;
    shared_array<int> a(new int[el_count], el_count);
    shared_array<int> b(new int[el_count], el_count);
    shared_array<int> c(new int[el_count], el_count);
    for (int i = 0; i < el_count; ++i) {
      a[i] = i;
      b[i] = el_count - i;
    }
    cl_data_t<int, oclalgo::IN> d_a(a);
    cl_data_t<int, oclalgo::IN> d_b(b);
    cl_data_t<int, oclalgo::OUT> d_c(c);

    auto future = queue.AddTask("vector_add.cl", "vector_add", cl::NullRange,
                                cl::NDRange(el_count), cl::NullRange, d_a, d_b,
                                d_c);
    std::tie(c) = future.get();

    // check results
    bool is_correct = true;
    for (int i = 0; i < el_count; ++i) {
      if (d_c.host_array[i] != el_count) {
        is_correct = false;
        break;
      }
    }
    EXPECT_TRUE(is_correct);
  } catch (const cl::Error& e) {
    std::cerr << e.what() << " (err_code = " << OpenCLQueue::StatusStr(e.err()) << ")" << std::endl;
    ASSERT_FALSE(true)<< "===> OpenCLQueue.VectorAdd: exception raised" << std::endl;
  }
}

TEST(OpenCLQueue, MatrixAdd) {
  using namespace oclalgo;
  try {
    OpenCLQueue queue("Intel(R) OpenCL", "Intel(R)");
    uint32_t rows = 3, cols = 4;
    shared_array<int> a(new int[rows * cols], rows * cols);
    shared_array<int> b(new int[rows * cols], rows * cols);
    shared_array<int> c(new int[rows * cols], rows * cols);
    for (uint32_t i = 0; i < rows; ++i) {
      for (uint32_t j = 0; j < cols; ++j) {
        a[i * cols + j] = i * cols + j;
        b[i * cols + j] = cols * rows - (i * cols + j);
      }
    }
    cl_data_t<int, oclalgo::IN> d_a(a);
    cl_data_t<int, oclalgo::IN> d_b(b);
    cl_data_t<int, oclalgo::OUT> d_c(c);

    auto future = queue.AddTask("hblas.cl", "matrix_add", cl::NullRange,
                                cl::NDRange(rows, cols), cl::NullRange, d_a, d_b,
                                d_c);
    std::tie(c) = future.get();

    // check results
    bool is_correct = true;
    for (uint32_t i = 0; i < rows * cols; ++i) {
      if (c[i] != static_cast<int>(rows * cols)) {
        is_correct = false;
        break;
      }
    }
    EXPECT_TRUE(is_correct);
  } catch (const cl::Error& e) {
    std::cerr << e.what() << " (err_code = " << OpenCLQueue::StatusStr(e.err()) << ")" << std::endl;
    ASSERT_FALSE(true)<< "===> OpenCLQueue.MatrixAdd: exception raised" << std::endl;
  }
}

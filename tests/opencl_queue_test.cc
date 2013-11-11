/*! @file openclqueue_tests.cc
 *  @brief OpenCLQueue class tests
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
#include "inc/oclalgo/opencl_queue.h"

TEST(OpenCLQueue, TryVectorAddKernel) {
  using namespace oclalgo;
  try {
    std::string platform_name = "Intel(R) OpenCL";
    std::string device_name = "Intel(R)";
    OpenCLQueue queue(platform_name, device_name);

    const int size = 1024;
    std::unique_ptr<int[]> a(new int[size]);
    std::unique_ptr<int[]> b(new int[size]);
    std::unique_ptr<int[]> c(new int[size]);
    for (int i = 0; i < size; ++i) {
      a[i] = i;
      b[i] = size - i;
    }
    cl_data_t<int*, oclalgo::IN> d_a(a.get(), sizeof(int) * size);
    cl_data_t<int*, oclalgo::IN> d_b(b.get(), sizeof(int) * size);
    cl_data_t<int*, oclalgo::OUT> d_c(c.get(), sizeof(int) * size);

    auto future = queue.AddTask("vector_add.cl", "vector_add", cl::NullRange,
                                cl::NDRange(size), cl::NullRange, d_a, d_b, d_c);
    std::tie(d_c) = future.get();

    // check results
    bool is_correct = true;
    for (int i = 0; i < size; ++i) {
      if (d_c.host_ptr[i] != size) {
        is_correct = false;
        break;
      }
    }
    EXPECT_TRUE(is_correct);
  } catch (const cl::Error& e) {
    std::cerr << e.what() << " (err_code = " << OpenCLQueue::StatusStr(e.err()) << ")" << std::endl;
    ASSERT_FALSE(true)<< "===> OpenCLQueue test: " << "exception raised" << std::endl;
  }
}

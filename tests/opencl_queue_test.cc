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

TEST(OpenCLQueue, TryVectorAddKernel) {
  using namespace oclalgo;
  try {
    std::string platform_name = "Intel(R) OpenCL";
    std::string device_name = "Intel(R)";
    OpenCLQueue queue(platform_name, device_name);

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
                                cl::NDRange(el_count), cl::NullRange, d_a, d_b, d_c);
    std::tie(d_c) = future.get();

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
    ASSERT_FALSE(true)<< "===> OpenCLQueue test: " << "exception raised" << std::endl;
  }
}

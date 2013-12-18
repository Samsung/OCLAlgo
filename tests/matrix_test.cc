/*! @file matrix_tests.cc
 *  @brief hblas::matrix class tests.
 *  @author Dmitry Senin <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  Tests for hblas::matrix class
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include <gtest/gtest.h>
//#include <oclalgo/opencl_queue.h>
#include "inc/oclalgo/hblas/matrix.h"
#include "src/gtest_main.cc"

TEST(Matrix, CopyAndAssignment) {
  oclalgo::hblas::Matrix<int> m1(3, 4);
  for (uint32_t i = 1; i <= m1.rows(); ++i)
    for (uint32_t j = 1; j <= m1.cols(); ++j)
      m1(i, j) = (i - 1) * m1.cols() + j;

  oclalgo::hblas::Matrix<int> m2(m1);
  oclalgo::hblas::Matrix<int> m3;
  m3 = m1;
  bool is_copy_err = false, is_assignment_err = false;
  for (uint32_t i = 1; i < m1.rows(); ++i) {
    for (uint32_t j = 1; j < m1.cols(); ++j) {
      if (m2(i, j) != m1(i, j)) is_copy_err = true;
      if (m3(i, j) != m1(i, j)) is_assignment_err = true;
    }
  }
  EXPECT_FALSE(is_copy_err) << "Matrix<T> copy constructor test" << std::endl;
  EXPECT_FALSE(is_assignment_err) << "Matrix<T> assignment operator test"
                                  << std::endl;
}

TEST(Matrix, AddOperator) {
  oclalgo::hblas::Matrix<int> m1(32, 32), m2(32, 32);
  for (uint32_t i = 1; i <= m1.rows(); ++i) {
    for (uint32_t j = 1; j <= m1.cols(); ++j) {
      m1(i, j) = (i - 1) * m1.cols() + j;
      m2(i, j) = m2.rows() * m2.cols() - (i - 1) * m2.cols() - j;
    }
  }

  oclalgo::hblas::Matrix<int> host_res = m1 + m2;
  oclalgo::hblas::Matrix<int> ocl_res = (m1.future() + m2.future()).get();
  bool is_host_err = false, is_ocl_err = false;
  for (uint32_t i = 1; i <= m1.rows(); ++i) {
    for (uint32_t j = 1; j <= m1.cols(); ++j) {
      if (host_res(i, j) != static_cast<int>(m2.cols() * m2.rows()))
        is_host_err = true;
      if (ocl_res(i, j) != static_cast<int>(m2.cols() * m2.rows()))
        is_ocl_err = true;
    }
  }
  EXPECT_FALSE(is_host_err) << "Host Matrix<T>::operator+() test" << std::endl;
  EXPECT_FALSE(is_ocl_err) << "OpenCL Matrix<T>::operator+() test" << std::endl;
}

TEST(Matrix, SubOperator) {
  oclalgo::hblas::Matrix<int> m1(32, 32), m2(32, 32);
  for (uint32_t i = 1; i <= m1.rows(); ++i)
    for (uint32_t j = 1; j <= m1.cols(); ++j)
      m2(i, j) = m1(i, j) = (i - 1) * m1.cols() + j;

  oclalgo::hblas::Matrix<int> host_res = m1 - m2;
  oclalgo::hblas::Matrix<int> ocl_res = (m1.future() - m2.future()).get();
  bool is_host_err = false, is_ocl_err = false;
  for (uint32_t i = 1; i <= m1.rows(); ++i) {
    for (uint32_t j = 1; j <= m1.cols(); ++j) {
      if (host_res(i, j) != 0)
        is_host_err = true;
      if (ocl_res(i, j) != 0)
        is_ocl_err = true;
    }
  }
  EXPECT_FALSE(is_host_err) << "Host Matrix<T>::operator-() test" << std::endl;
  EXPECT_FALSE(is_ocl_err) << "OpenCL Matrix<T>::operator-() test" << std::endl;
}

TEST(Matrix, MulOperator) {
  uint32_t block_size = 4;
  oclalgo::hblas::Matrix<int> m1(4, 4, block_size), m2(4, 8, block_size);
  for (uint32_t i = 1; i <= m1.rows(); ++i)
    for (uint32_t j = 1; j <= m1.cols(); ++j)
      m1(i, j) = (i - 1) * m1.cols() + j;
  for (uint32_t i = 1; i <= m2.rows(); ++i)
    for (uint32_t j = 1; j <= m2.cols(); ++j)
      m2(i, j) = (i - 1) * m2.cols() + j;

  oclalgo::hblas::Matrix<int> host_res = m1 * m2;
  oclalgo::hblas::Matrix<int> ocl_res = (m1.future() * m2.future()).get();
  int gold_res[] = { 170, 180, 190, 200,  210,  220,  230,  240,
                     378, 404, 430, 456,  482,  508,  534,  560,
                     586, 628, 670, 712,  754,  796,  838,  880,
                     794, 852, 910, 968, 1026, 1084, 1142, 1200 };
  bool is_host_err = false, is_ocl_err = false;
  for (uint32_t i = 1; i <= host_res.rows(); ++i) {
    for (uint32_t j = 1; j <= host_res.cols(); ++j) {
      if (host_res(i, j) != gold_res[(i - 1) * host_res.cols() + j - 1])
        is_host_err = true;
      if (ocl_res(i, j) != gold_res[(i - 1) * host_res.cols() + j - 1])
        is_ocl_err = true;
    }
  }
  EXPECT_FALSE(is_host_err) << "Host Matrix<T>::operator*() test" << std::endl;
  EXPECT_FALSE(is_ocl_err) << "OpenCL Matrix<T>::operator*() test" << std::endl;
}

TEST(Matrix, Transpose) {
  oclalgo::hblas::Matrix<int> m1(32, 64), m2(64, 32);
  for (uint32_t i = 1; i <= m1.rows(); ++i)
    for (uint32_t j = 1; j <= m1.cols(); ++j)
      m1(i, j) = (i - 1) * m1.cols() + j;
  for (uint32_t i = 1; i <= m2.rows(); ++i)
    for (uint32_t j = 1; j <= m2.cols(); ++j)
      m2(i, j) = (j - 1) * m2.rows() + i;

  m1.transpose();
  bool is_err = false;
  for (uint32_t i = 1; i <= m1.rows(); ++i) {
    for (uint32_t j = 1; j <= m1.cols(); ++j) {
      if (m1(i, j) != m2(i, j))
        is_err = true;
    }
  }
  EXPECT_FALSE(is_err) << "Matrix<T>::transpose() test" << std::endl;
}

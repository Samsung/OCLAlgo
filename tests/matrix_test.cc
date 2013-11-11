/*! @file matrix_tests.cc
 *  @brief hblas::matrix class tests
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
#include "inc/oclalgo/hblas/matrix.h"
#include "src/gtest_main.cc"

TEST(Matrix, CopyAndAssignmentTest) {
  uint rows = 3, cols = 3;
  oclalgo::hblas::Matrix<int> m1(3, 3);
  for (uint i = 1; i <= rows; ++i) {
    for (uint j = 1; j <= cols; ++j) {
      m1(i, j) = (i - 1) * cols + j;
    }
  }
  oclalgo::hblas::Matrix<int> m2(m1);
  oclalgo::hblas::Matrix<int> m3;
  m3 = m1;
  bool copy_eq = true, ass_eq = true;
  for (uint i = 1; i < rows; ++i) {
    for (uint j = 1; j < cols; ++j) {
      if (m2(i, j) != m1(i, j)) copy_eq = false;
      if (m3(i, j) != m1(i, j)) ass_eq = false;
    }
  }
  EXPECT_TRUE(copy_eq) << "===> hblas::matrix copy constructor test"
      << std::endl << "hblas::matrix<int> m2(m1);" << std::endl << "m1"
      << std::endl << m1 << std::endl << "m2" << std::endl << m2;
  EXPECT_TRUE(ass_eq) << "===> hblas::matrix assignment operator test"
        << std::endl << "hblas::matrix<int> m3;\nm3 = m1;" << std::endl << "m1"
        << std::endl << m1 << std::endl << "m3" << std::endl << m3;
}

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

/*! @file dmatrix.cc
 *  @brief Unit tests for oclalgo::DMatrix class.
 *  @author Senin Dmitry <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include <gtest/gtest.h>
#include "inc/oclalgo/dmatrix.h"
#include "inc/oclalgo/matrix.h"
#include "src/gtest_main.cc"

TEST(DMatrix, CtorFromMatrix) {
  using oclalgo::Matrix;
  using oclalgo::DMatrix;
  int rows = 1024, cols = 2048;
  Matrix<int> m(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      m(i, j) = i * cols + j;

  DMatrix<int> dm(m);
  EXPECT_EQ(m.rows(), dm.rows());
  EXPECT_EQ(m.cols(), dm.cols());

  Matrix<int> res = dm.ToHost();
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      ASSERT_EQ(m(i, j), res(i, j));
}

TEST(DMatrix, CtorFromBuffer) {
  using oclalgo::Matrix;
  using oclalgo::DMatrix;
  int rows = 1024, cols = 2048;
  Matrix<int> m(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      m(i, j) = i * cols + j;

  cl::Buffer buff = oclalgo::MatrixQueue::instance()->CreateBuffer(
      m.data(), oclalgo::BufferType::ReadWrite);

  DMatrix<int> dm(rows, cols, buff);
  EXPECT_EQ(m.rows(), dm.rows());
  EXPECT_EQ(m.cols(), dm.cols());

  Matrix<int> res = dm.ToHost();
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      ASSERT_EQ(m(i, j), res(i, j));
}

TEST(DMatrix, ToHost) {
  using oclalgo::Matrix;
  using oclalgo::DMatrix;
  int rows = 1024, cols = 2048;
  Matrix<int> m(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      m(i, j) = i * cols + j;

  DMatrix<int> dm(m);
  EXPECT_EQ(m.rows(), dm.rows());
  EXPECT_EQ(m.cols(), dm.cols());

  Matrix<int> res1 = dm.ToHost();
  Matrix<int> res2 = dm.ToHost(oclalgo::BlockingType::Unblock).get();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      ASSERT_EQ(m(i, j), res1(i, j));
      ASSERT_EQ(m(i, j), res2(i, j));
    }
  }
}

TEST(DMatrix, UpdateData) {
  using oclalgo::Matrix;
  using oclalgo::DMatrix;
  int rows = 1024, cols = 2048;
  Matrix<int> m1(rows, cols);
  for (int i = 0; i < m1.rows(); ++i)
    for (int j = 0; j < m1.cols(); ++j)
      m1(i, j) = 1;

  DMatrix<int> dm(m1);
  for (int i = 0; i < m1.rows(); ++i)
    for (int j = 0; j < m1.cols(); ++j)
      m1(i, j) = i * m1.cols() + j;
  dm.UpdateData(m1);
  EXPECT_EQ(m1.rows(), dm.rows());
  EXPECT_EQ(m1.cols(), dm.cols());
  Matrix<int> res1 = dm.ToHost();

  Matrix<int> m2(rows * 3, cols);
  for (int i = 0; i < m2.rows(); ++i)
    for (int j = 0; j < m2.cols(); ++j)
      m2(i, j) = i * m2.cols() + j;
  dm.UpdateData(m2);
  EXPECT_EQ(m2.rows(), dm.rows());
  EXPECT_EQ(m2.cols(), dm.cols());
  Matrix<int> res2 = dm.ToHost();

  dm = dm.UpdateData(m1, oclalgo::BlockingType::Unblock).get();
  EXPECT_EQ(m1.rows(), dm.rows());
  EXPECT_EQ(m1.cols(), dm.cols());
  Matrix<int> res3 = dm.ToHost();

  for (int i = 0; i < m1.rows(); ++i) {
    for (int j = 0; j < m1.cols(); ++j) {
      ASSERT_EQ(m1(i, j), res1(i, j));
      ASSERT_EQ(m1(i, j), res3(i, j));
    }
  }
  for (int i = 0; i < m1.rows(); ++i)
    for (int j = 0; j < m1.cols(); ++j)
      ASSERT_EQ(m2(i, j), res2(i, j));
}

TEST(DMatrix, Add) {
  using oclalgo::Matrix;
  using oclalgo::DMatrix;
  int rows = 1024, cols = 2048;
  Matrix<int> m1(rows, cols), m2(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      m1(i, j) = i * cols + j;
      m2(i, j) = cols * rows - i * cols - j;
    }
  }

  DMatrix<int> dm1(m1), dm2(m2);
  DMatrix<int> dres = (dm1 + dm2).get();
  EXPECT_EQ(m1.rows(), dres.rows());
  EXPECT_EQ(m1.cols(), dres.cols());

  Matrix<int> res = dres.ToHost();
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      ASSERT_EQ(cols * rows, res(i, j));
}

TEST(DMatrix, Sub) {
  using oclalgo::Matrix;
  using oclalgo::DMatrix;
  int rows = 1024, cols = 2048;
  Matrix<int> m1(rows, cols), m2(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      m1(i, j) = i * cols + j + rows * cols;
      m2(i, j) = i * cols + j;
    }
  }

  DMatrix<int> dm1(m1), dm2(m2);
  DMatrix<int> dres = (dm1 - dm2).get();
  EXPECT_EQ(m1.rows(), dres.rows());
  EXPECT_EQ(m1.cols(), dres.cols());

  Matrix<int> res = dres.ToHost();
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      ASSERT_EQ(cols * rows, res(i, j));
}

TEST(DMatrix, MulIdentity) {
  using oclalgo::Matrix;
  using oclalgo::DMatrix;
  Matrix<int> m1(256, 256), I(256, 256);
  for (int i = 0; i < m1.rows(); ++i) {
    for (int j = 0; j < m1.cols(); ++j) {
      m1(i, j) = i * m1.cols() + j;
      I(i, j) = i == j ? 1 : 0;
    }
  }

  Matrix<int> I2 = I * I;
  EXPECT_EQ(I.rows(), I2.rows());
  EXPECT_EQ(I.cols(), I2.cols());
  for (int i = 0; i < I2.rows(); ++i)
    for (int j = 0; j < I2.cols(); ++j)
      ASSERT_EQ(I(i, j), I2(i, j));

  Matrix<int> m1_I = m1 * I;
  for (int i = 0; i < m1_I.rows(); ++i)
    for (int j = 0; j < m1_I.cols(); ++j)
      ASSERT_EQ(m1(i, j), m1_I(i, j));
}

TEST(DMatrix, MulInt) {
  using oclalgo::Matrix;
  using oclalgo::DMatrix;
  Matrix<int> m1(4, 4), m2(4, 8);
  for (int i = 0; i < m1.rows(); ++i)
    for (int j = 0; j < m1.cols(); ++j)
      m1(i, j) = i * m1.cols() + j + 1;
  for (int i = 0; i < m2.rows(); ++i)
    for (int j = 0; j < m2.cols(); ++j)
      m2(i, j) = i * m2.cols() + j + 1;

  Matrix<int> res = m1 * m2;
  int gold_res[] = { 170, 180, 190, 200,  210,  220,  230,  240,
                     378, 404, 430, 456,  482,  508,  534,  560,
                     586, 628, 670, 712,  754,  796,  838,  880,
                     794, 852, 910, 968, 1026, 1084, 1142, 1200 };
  for (int i = 0; i < res.rows(); ++i)
    for (int j = 0; j < res.cols(); ++j)
      ASSERT_EQ(gold_res[i * res.cols() + j], res(i, j));
}

TEST(DMatrix, MulFloat) {
  using oclalgo::Matrix;
  using oclalgo::DMatrix;
  Matrix<float> m1(2, 2), m2(2, 2);
  for (int i = 0; i < m1.rows(); ++i)
    for (int j = 0; j < m1.cols(); ++j)
      m1(i, j) = i * m1.cols() + j + 1.5F;
  for (int i = 0; i < m2.rows(); ++i)
    for (int j = 0; j < m2.cols(); ++j)
      m2(i, j) = i * m2.cols() + j + 1.5F + m1.cols() * m2.rows();

  Matrix<float> res = m1 * m2;
  float gold_res[] = { 27.F, 31.F,
                       53.F, 61.F };
  for (int i = 0; i < res.rows(); ++i)
    for (int j = 0; j < res.cols(); ++j)
      ASSERT_EQ(gold_res[i * res.cols() + j], res(i, j));
}

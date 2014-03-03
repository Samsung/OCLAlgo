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

/*! @file matrix.h
 *  @brief Contains oclalgo::Matrix class for host matrix operations.
 *  @author Dmitry Senin <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  Use OpenCL and Host resources to compute simplest linear algebra operations.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_OCLALGO_MATRIX_H_
#define INC_OCLALGO_MATRIX_H_

#include <cassert>
#include <ostream>
#include <functional>
#include <algorithm>

#include <oclalgo/shared_array.h>

namespace oclalgo {

template <typename T>
class Matrix;

template <typename T>
std::ostream& operator<<(std::ostream&, const Matrix<T>&);

/** @brief Template matrix class. */
template <typename T>
class Matrix {
  friend std::ostream& operator<< <T>(std::ostream& out, const Matrix<T>& m);

 public:
  Matrix();
  /** @brief Creates matrix with corresponding numbers of rows and columns. */
  Matrix(int rows, int cols);
  /*!
   * @brief Creates matrix with corresponding numbers of rows and columns using
   * transferred shared array.
   */
  Matrix(int rows, int cols, const shared_array<T>& array);

  Matrix(const Matrix<T>& m);
  Matrix(Matrix<T>&& m);

  virtual ~Matrix() = default;

  Matrix<T>& operator=(const Matrix<T>& m);
  Matrix<T>& operator=(Matrix<T>&& m);

  /** @brief Resizes matrix by new one with specified size. */
  virtual void resize(int rows, int cols);

  /** @brief Transposes matrix. */
  virtual void transpose();

  /** @brief Returns number of rows in matrix. */
  int rows() const noexcept { return rows_; }
  /** @brief Returns number of columns in matrix. */
  int cols() const noexcept { return cols_; }
  /** @brief Returns shared_array class object, which contains matrix data. */
  shared_array<T> data() const noexcept { return data_; }

  /*!
   * @brief Returns matrix element in position (i, j).
   *
   * i is in range [0, rows - 1]
   * j is in range [0, cols - 1]
   */
  const T& operator()(int i, int j) const noexcept {
    return data_[i * cols_ + j];
  }

  /*!
   * @brief Returns reference to matrix element in position (i, j).
   *
   * Can be used to modify matrix element in position (i, j).
   * i is in range [0, rows - 1]
   * j is in range [0, cols - 1]
   */
  T& operator()(int i, int j) noexcept {
    return data_[i * cols_ + j];
  }

 private:
  int rows_;
  int cols_;
  shared_array<T> data_;
};

template <typename T>
Matrix<T>::Matrix(): rows_(0), cols_(0) {
}

template <typename T>
Matrix<T>::Matrix(int rows, int cols)
    : rows_(rows),
      cols_(cols),
      data_(rows_ * cols_) {
}

template <typename T>
Matrix<T>::Matrix(int rows, int cols, const shared_array<T>& array)
    : rows_(rows),
      cols_(cols),
      data_(array) {
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T>& m)
    : rows_(m.rows_),
      cols_(m.cols_),
      data_(rows_ * cols_) {
  std::copy(m.data_.get_raw(), m.data_.get_raw() + m.rows_ * m.cols_,
            data_.get_raw());
}

template <typename T>
Matrix<T>::Matrix(Matrix<T>&& m)
    : rows_(m.rows_),
      cols_(m.cols_),
      data_(m.data_) {
  m.data_.reset();
  m.rows_ = m.cols_ = 0;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& m) {
  if (this != &m) {
    rows_ = m.rows_;
    cols_ = m.cols_;
    T* ptr = new T[rows_ * cols_];
    std::copy(m.data_.get_raw(), m.data_.get_raw() + m.rows_ * m.cols_, ptr);
    data_.reset(ptr, rows_ * cols_);
  }
  return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& m) {
  if (this != &m) {
    rows_ = m.rows_;
    cols_ = m.cols_;
    data_ = m.data_;
    m.data_.reset();
    m.rows_ = m.cols_ = 0;
  }
  return *this;
}

template <typename T>
void Matrix<T>::resize(int rows, int cols) {
  rows_ = rows;
  cols_ = cols;
  data_ = shared_array<T>(rows * cols);
}

template <typename T>
void Matrix<T>::transpose() {
  shared_array<T> new_data(rows_ * cols_);
  for (int i = 0; i < rows_; ++i)
    for (int j = 0; j < cols_; ++j)
      new_data[j * rows_ + i] = data_[i * cols_ + j];
  std::swap(rows_, cols_);
  data_ = new_data;
}

template <typename U, typename FunctorType>
Matrix<U> MatrixOperation(const Matrix<U>& m1, const Matrix<U>& m2,
                          const FunctorType& f) {
  Matrix<U> res(m1.rows(), m1.cols());
  for (int i = 0; i < m1.rows(); ++i)
    for (int j = 0; j < m1.cols(); ++j)
      res(i, j) = f(m1(i, j), m2(i, j));
  return res;
}

template <typename U>
Matrix<U> operator+(const Matrix<U>& m1, const Matrix<U>& m2) {
  assert(m1.cols() == m2.cols() && m1.rows() == m2.rows());
  return MatrixOperation(m1, m2, std::plus<U>());
}

template <typename U>
Matrix<U> operator-(const Matrix<U>& m1, const Matrix<U>& m2) {
  assert(m1.cols() == m2.cols() && m1.rows() == m2.rows());
  return MatrixOperation(m1, m2, std::minus<U>());
}

template <typename U>
Matrix<U> operator*(const Matrix<U>& m1, const Matrix<U>& m2) {
  assert(m1.cols() == m2.rows());
  Matrix<U> res(m1.rows(), m2.cols());
  for (int i = 0; i < m1.rows(); ++i) {
    for (int j = 0; j < m2.cols(); ++j) {
      res(i, j) = 0;
      for (int k = 0; k < m1.cols(); ++k)
        res(i, j) += m1(i, k) * m2(k, j);
    }
  }
  return res;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& m) {
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j)
      out << m(i, j) << "\t";
    out << std::endl;
  }
  return out;
}

}  // namespace oclalgo

#endif  // INC_OCLALGO_MATRIX_H_

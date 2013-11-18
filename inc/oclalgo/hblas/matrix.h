/*! @file matrix.h
 *  @brief Matrix Class with Heterogeneous Computing.
 *  @author Dmitry Senin <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  Use OpenCL and Host resources to compute simplest linear algebra operations.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef OCLALGO_HBLAS_MATRIX_H_
#define OCLALGO_HBLAS_MATRIX_H_

#include <cstdint>
#include <cassert>
#include <ostream>
#include <algorithm>
#include <memory>

namespace oclalgo {

namespace hblas {

template<typename T>
class Matrix;

template<typename T>
std::ostream& operator<<(std::ostream&, const Matrix<T>&);

template<typename T>
class Matrix {
  friend std::ostream& operator<<<T>(std::ostream& out, const Matrix<T>& m);

 public:
  Matrix()
      : rows_(0),
        cols_(0) {
  }
  Matrix(uint32_t rows, uint32_t cols)
      : rows_(rows),
        cols_(cols),
        data_(new T[rows_ * cols_]) {
  }
  Matrix(const Matrix& m)
      : rows_(m.rows_),
        cols_(m.cols_),
        data_(new T[m.rows_ * m.cols_]) {
    std::copy(m.data_.get(), m.data_.get() + m.rows_ * m.cols_, data_.get());
  }
  Matrix<T>& operator=(const Matrix<T>& m) {
    if (this != &m) {
      rows_ = m.rows();
      cols_ = m.cols();
      T* ptr = new T[rows_ * cols_];
      std::copy(m.data_.get(), m.data_.get() + m.rows_ * m.cols_, ptr);
      data_.reset(ptr);
    }
    return *this;
  }

  uint32_t rows() const noexcept { return rows_; }
  uint32_t cols() const noexcept { return cols_; }
  T* data() const noexcept { return data_; }

  const T& operator()(uint32_t i, uint32_t j) const noexcept {
    assert(i >= 1 && j >= 1 && i <= rows_ && j <= cols_);
    return data_[(i - 1) * cols_ + (j - 1)];
  }
  T& operator()(uint32_t i, uint32_t j) noexcept {
    assert(i >= 1 && j >= 1 && i <= rows_ && j <= cols_);
    return data_[(i - 1) * cols_ + (j - 1)];
  }

  const Matrix<T>& operator*(const Matrix<T>& m) const;
  void operator*=(const Matrix<T>& m);
  const Matrix<T>& operator+(const Matrix<T>& m) const;
  void operator+=(const Matrix<T>& m);
  const Matrix<T>& operator-(const Matrix<T>& m) const;
  void operator-=(const Matrix<T>& m);
  void transpose();

 private:
  uint32_t rows_;
  uint32_t cols_;
  std::unique_ptr<T[]> data_;
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& m) {
  for (uint32_t i = 0; i < m.rows_; ++i) {
    for (uint32_t j = 0; j < m.cols_; ++j) {
      out << m.data_[i * m.cols_ + j] << "\t";
    }
    out << std::endl;
  }
  return out;
}

} // namespace hblas

} // namespace oclalgo

#endif // OCLALGO_HBLAS_MATRIX_H_

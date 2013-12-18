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
#include <cmath>
#include <ostream>
#include <algorithm>
#include <memory>

#include <oclalgo/shared_array.h>
#include <oclalgo/opencl_queue.h>

namespace oclalgo {

namespace hblas {

class DeviceQueue {
 public:
  DeviceQueue() = delete;
  DeviceQueue(const DeviceQueue&) = delete;
  void operator=(const DeviceQueue&) = delete;

  static OpenCLQueue& getInstance() {
    static OpenCLQueue queue("Intel(R) OpenCL", "Intel(R)");
    return queue;
  }
};

template<typename T>
class Matrix;

template<typename T>
std::ostream& operator<<(std::ostream&, const Matrix<T>&);

template<typename T>
class Matrix {
  friend std::ostream& operator<< <T>(std::ostream& out, const Matrix<T>& m);
  template<typename U>
  friend cl_future<Matrix<U>> operator+(const cl_future<Matrix<U>>& futureM1,
                                        const cl_future<Matrix<U>>& futureM2);
  template<typename U>
  friend cl_future<Matrix<U>> operator-(const cl_future<Matrix<U>>& futureM1,
                                        const cl_future<Matrix<U>>& futureM2);
  template<typename U>
  friend cl_future<Matrix<U>> operator*(const cl_future<Matrix<U>>& futureM1,
                                        const cl_future<Matrix<U>>& futureM2);
 public:
  Matrix()
      : rows_(0),
        cols_(0),
        block_size_(0) {
  }

  Matrix(uint32_t rows, uint32_t cols, uint32_t block_size = 0)
      : rows_(rows),
        cols_(cols),
        data_(new T[rows_ * cols_], rows_ * cols_),
        block_size_(block_size) {
  }

  Matrix(uint32_t rows, uint32_t cols, const oclalgo::shared_array<T>& array)
      : rows_(rows),
        cols_(cols),
        data_(array),
        block_size_(0) {
  }

  Matrix(uint32_t rows, uint32_t cols, uint32_t block_size, const oclalgo::shared_array<T>& array)
      : rows_(rows),
        cols_(cols),
        data_(array),
        block_size_(block_size) {
  }

  Matrix(const Matrix<T>& m)
      : rows_(m.rows_),
        cols_(m.cols_),
        data_(new T[rows_ * cols_], rows_ * cols_),
        block_size_(m.block_size_) {
    std::copy(m.data_.get(), m.data_.get() + m.rows_ * m.cols_, data_.get());
  }

  Matrix(Matrix<T>&& m)
      : rows_(m.rows_),
        cols_(m.cols_),
        data_(m.data_),
        block_size_(m.block_size_) {
  }

  Matrix<T>& operator=(const Matrix<T>& m) {
    if (this != &m) {
      rows_ = m.rows_;
      cols_ = m.cols_;
      T* ptr = new T[rows_ * cols_];
      std::copy(m.data_.get(), m.data_.get() + m.rows_ * m.cols_, ptr);
      data_.reset(ptr, rows_ * cols_);
      block_size_ = m.block_size_;
    }
    return *this;
  }

  void resize(uint32_t rows, uint32_t cols) {
    rows_ = rows;
    cols_ = cols;
    data_.reset(new T[rows_ * cols_], rows_ * cols_);
    block_size_ = 0;
  }

  void resize(uint32_t rows, uint32_t cols, uint32_t block_size) {
    rows_ = rows;
    cols_ = cols;
    data_.reset(new T[rows_ * cols_], rows_ * cols_);
    block_size_ = block_size;
  }

  uint32_t rows() const noexcept { return rows_; }
  uint32_t cols() const noexcept { return cols_; }
  oclalgo::shared_array<T> data() const noexcept { return data_; }
  uint32_t block_size() const noexcept { return block_size_; }
  uint32_t& block_size() noexcept { return block_size_; }

  cl_future<Matrix<T>> future() const noexcept {
    Matrix<T> m(rows_, cols_, block_size_, data_);
    return cl_future<Matrix<T>>(std::move(m));
  }

  const T& operator()(uint32_t i, uint32_t j) const noexcept {
    assert(i >= 1 && j >= 1 && i <= rows_ && j <= cols_);
    return data_[(i - 1) * cols_ + (j - 1)];
  }

  T& operator()(uint32_t i, uint32_t j) noexcept {
    assert(i >= 1 && j >= 1 && i <= rows_ && j <= cols_);
    return data_[(i - 1) * cols_ + (j - 1)];
  }

  void transpose();

 private:
  uint32_t rows_;
  uint32_t cols_;
  oclalgo::shared_array<T> data_;
  uint32_t block_size_;
};

template<typename T>
void Matrix<T>::transpose() {
  oclalgo::shared_array<T> new_data(new T[rows_ * cols_], rows_ * cols_);
  for (uint32_t i = 1; i <= rows_; ++i)
    for (uint32_t j = 1; j <= cols_; ++j)
      new_data[(j - 1) * rows_ + i - 1] = this->operator ()(i, j);
  std::swap(rows_, cols_);
  data_ = new_data;
}

template<typename U, typename X>
Matrix<U> MatrixOperation(const Matrix<U>& m1, const Matrix<U>& m2,
                          const X& func) {
  Matrix<U> res(m1.rows(), m1.cols());
  for (uint32_t i = 1; i <= m1.rows(); ++i)
    for (uint32_t j = 1; j <= m1.cols(); ++j)
      res(i, j) = func(m1(i, j), m2(i, j));
  return res;
}

template<typename U>
Matrix<U> operator+(const Matrix<U>& m1, const Matrix<U>& m2) {
  assert(m1.cols() == m2.cols() && m1.rows() == m2.rows());
  return MatrixOperation(m1, m2, std::plus<U>());
}

template<typename U>
Matrix<U> operator-(const Matrix<U>& m1, const Matrix<U>& m2) {
  assert(m1.cols() == m2.cols() && m1.rows() == m2.rows());
  return MatrixOperation(m1, m2, std::minus<U>());
}

template<typename U>
Matrix<U> operator*(const Matrix<U>& m1, const Matrix<U>& m2) {
  assert(m1.cols() == m2.rows());
  Matrix<U> res(m1.rows(), m2.cols());
  for (uint32_t i = 1; i <= m1.rows(); ++i) {
    for (uint32_t j = 1; j <= m2.cols(); ++j) {
      res(i, j) = 0;
      for (uint32_t k = 1; k <= m1.cols(); ++k)
        res(i, j) += m1(i, k) * m2(k, j);
    }
  }
  return res;
}

template <typename T> std::string PrintType();
template <> std::string PrintType<int>() { return "int"; }
template <> std::string PrintType<float>() { return "float"; }
template <> std::string PrintType<double>() { return "double"; }

template<typename U>
cl_future<Matrix<U>> operator+(const cl_future<Matrix<U>>& futureM1,
                               const cl_future<Matrix<U>>& futureM2) {
  uint32_t cols = futureM1.stored_data().cols();
  uint32_t rows = futureM1.stored_data().rows();
  assert(cols == futureM2.stored_data().cols() &&
         rows == futureM2.stored_data().rows());

  cl_data_t<U, oclalgo::IN> dm1(futureM1.stored_data().data());
  cl_data_t<U, oclalgo::IN> dm2(futureM2.stored_data().data());
  Matrix<U> res(rows, cols);
  cl_data_t<U, oclalgo::OUT> dres(res.data());

  char compile_options[512] = {0};
  std::snprintf(compile_options, 512, "-D VAR_TYPE=%s", PrintType<U>().c_str());
  OpenCLQueue& queue = DeviceQueue::getInstance();
  auto future = queue.AddTask("hblas.cl", "matrix_add", compile_options,
                              cl::NullRange, cl::NDRange(rows, cols),
                              cl::NullRange, dm1, dm2, dres);
  return cl_future<Matrix<U>>(std::move(res), future.buffers(), future.event());
}

template<typename U>
cl_future<Matrix<U>> operator-(const cl_future<Matrix<U>>& futureM1,
                               const cl_future<Matrix<U>>& futureM2) {
  uint32_t cols = futureM1.stored_data().cols();
  uint32_t rows = futureM1.stored_data().rows();
  assert(cols == futureM2.stored_data().cols() &&
         rows == futureM2.stored_data().rows());

  cl_data_t<U, oclalgo::IN> dm1(futureM1.stored_data().data());
  cl_data_t<U, oclalgo::IN> dm2(futureM2.stored_data().data());
  Matrix<U> res(rows, cols);
  cl_data_t<U, oclalgo::OUT> dres(res.data());

  char compile_options[512] = {0};
  std::snprintf(compile_options, 512, "-D VAR_TYPE=%s", PrintType<U>().c_str());
  OpenCLQueue& queue = DeviceQueue::getInstance();
  auto future = queue.AddTask("hblas.cl", "matrix_sub", compile_options,
                              cl::NullRange, cl::NDRange(rows, cols),
                              cl::NullRange, dm1, dm2, dres);
  return cl_future<Matrix<U>>(std::move(res), future.buffers(), future.event());
}

template<typename U>
cl_future<Matrix<U>> operator*(const cl_future<Matrix<U>>& futureM1,
                               const cl_future<Matrix<U>>& futureM2) {
  uint32_t m1_cols = futureM1.stored_data().cols();
  uint32_t m1_rows = futureM1.stored_data().rows();
  uint32_t m2_cols = futureM2.stored_data().cols();
  uint32_t m2_rows = futureM2.stored_data().rows();
  assert(m1_cols == m2_rows);

  uint32_t block_size = futureM1.stored_data().block_size();
  cl_data_t<U, oclalgo::IN> dm1(futureM1.stored_data().data());
  cl_data_t<U, oclalgo::IN> dm2(futureM2.stored_data().data());
  oclalgo::shared_array<U> loc(nullptr, block_size * block_size);
  cl_data_t<U, oclalgo::LOCALE> dloc(loc);
  oclalgo::shared_array<int> cl_m1_cols(new int(m1_cols), 1);
  cl_data_t<int, oclalgo::VAR> dm1_cols(cl_m1_cols);
  oclalgo::shared_array<int> cl_m2_cols(new int(m2_cols), 1);
  cl_data_t<int, oclalgo::VAR> dm2_cols(cl_m2_cols);
  Matrix<U> res(m1_rows, m2_cols);
  cl_data_t<U, oclalgo::OUT> dres(res.data());

  char compile_options[512] = {0};
  std::snprintf(compile_options, 512, "-D BLOCK_SIZE=%d -D VAR_TYPE=%s",
                block_size, PrintType<U>().c_str());
  OpenCLQueue& queue = DeviceQueue::getInstance();
  auto future = queue.AddTask("hblas.cl", "matrix_mul", compile_options,
                              cl::NullRange, cl::NDRange(m2_cols, m1_rows),
                              cl::NDRange(block_size, block_size), dm1, dm2,
                              dres, dloc, dloc, dm1_cols, dm2_cols);
  return cl_future<Matrix<U>>(std::move(res), future.buffers(), future.event());
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& m) {
  for (uint32_t i = 0; i < m.rows_; ++i) {
    for (uint32_t j = 0; j < m.cols_; ++j)
      out << m.data_[i * m.cols_ + j] << "\t";
    out << std::endl;
  }
  return out;
}

} // namespace hblas

} // namespace oclalgo

#endif // OCLALGO_HBLAS_MATRIX_H_

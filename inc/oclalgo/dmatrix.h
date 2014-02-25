/*! @file dmatrix.h
 *  @brief Contains oclalgo::DMatrix class.
 *  @author Dmitry Senin <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_OCLALGO_DMATRIX_H_
#define INC_OCLALGO_DMATRIX_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <string>

#include <oclalgo/matrix.h>
#include <oclalgo/queue.h>

namespace oclalgo {

/** @brief Singleton to provide access to Queue object. */
class MatrixQueue {
 public:
  MatrixQueue() = delete;
  MatrixQueue(const MatrixQueue&) = delete;
  MatrixQueue& operator=(const MatrixQueue&) = delete;

  /** @brief Provides an instance of Queue object to launch tasks. */
  static Queue* instance() {
    static Queue queue("NVIDIA", "GeForce");
    return &queue;
  }
  constexpr static int block_size = 32;
};

/** @brief Class of matrix with data placed in OpenCL device memory. */
template <typename T>
class DMatrix {
  template <typename U>
  friend future<DMatrix<U>, VecBuffers> operator+(
      const future<DMatrix<U>, VecBuffers>& m1,
      const future<DMatrix<U>, VecBuffers>& m2);
  template <typename U>
  friend future<DMatrix<U>, VecBuffers> operator-(
      const future<DMatrix<U>, VecBuffers>& m1,
      const future<DMatrix<U>, VecBuffers>& m2);
  template <typename U>
  friend future<DMatrix<U>, VecBuffers> operator*(
      const future<DMatrix<U>, VecBuffers>& m1,
      const future<DMatrix<U>, VecBuffers>& m2);

 public:
  DMatrix();
  /** @brief Creates device matrix by using host matrix data. */
  explicit DMatrix(const Matrix<T>& m);
  /*!
   * @brief Creates device matrix with corresponding number of rows and columns.
   */
  DMatrix(int rows, int cols);
  /*!
   * @brief Creates device matrix with corresponding numbers of rows and columns
   * using transferred cl::Buffer object.
   */
  DMatrix(int rows, int cols, const cl::Buffer& buffer);

  DMatrix(const DMatrix<T>& m) = delete;
  DMatrix<T>& operator=(const DMatrix<T>& m) = delete;

  DMatrix(DMatrix<T>&& m);
  DMatrix<T>& operator=(DMatrix<T>&& m);

  virtual ~DMatrix() = default;

  /** @brief Creates host matrix based on device matrix data. */
  Matrix<T> ToHost() const;

  /*!
   * @brief Creates host matrix as blocking or unblocking operation
   * (depends on argument <i>block</i>). Host matrix based on device
   * matrix data.
   */
  future<Matrix<T>, cl::Buffer> ToHost(BlockingType block) const;

  /*!
   * @brief Modifies host matrix by coping data from device matrix.
   *
   * If host matrix has the same size as device matrix, only memcpy operation
   * will start. If host matrix size is different - resize operation will
   * start at first.
   */
  void ToHost(Matrix<T>* m) const;

  /** Updates device matrix using host matrix data. */
  void UpdateData(const Matrix<T>& m);

  /*!
   * @brief Updates device matrix using host matrix data as blocking or
   * unblocking operation (depends on argument <i>block</i>).
   */
  future<DMatrix<T>, shared_array<T>> UpdateData(const Matrix<T>& m,
                                                 BlockingType block);

  /** @brief Returns number of rows in device matrix. */
  int rows() const noexcept { return rows_; }
  /** @brief Returns number of columns in device matrix. */
  int cols() const noexcept { return cols_; }
  /** @brief Returns cl::Buffer object, which contains device matrix data. */
  cl::Buffer buffer() const noexcept { return buffer_; }

 private:
  int rows_;
  int cols_;
  cl::Buffer buffer_;
};

template <typename T>
DMatrix<T>::DMatrix(): rows_(0), cols_(0) {
}

template <typename T>
DMatrix<T>::DMatrix(const Matrix<T>& m): rows_(m.rows()), cols_(m.cols()) {
  buffer_ = cl::Buffer(MatrixQueue::instance()->context(),
                       CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                       m.rows() * m.cols() * sizeof(T), m.data().get_raw());
}

template <typename T>
DMatrix<T>::DMatrix(int rows, int cols): rows_(rows), cols_(cols) {
  buffer_ = cl::Buffer(MatrixQueue::instance()->context(), CL_MEM_READ_WRITE,
                       rows_ * cols_ * sizeof(T));
}

template <typename T>
DMatrix<T>::DMatrix(int rows, int cols, const cl::Buffer& buffer)
    : rows_(rows),
      cols_(cols),
      buffer_(buffer) {
}

template <typename T>
DMatrix<T>::DMatrix(DMatrix<T>&& m)
    : rows_(m.rows_),
      cols_(m.cols_),
      buffer_(m.buffer_) {
  m.rows_ = m.cols_ = 0;
  m.buffer_ = cl::Buffer();
}

template <typename T>
DMatrix<T>& DMatrix<T>::operator=(DMatrix<T>&& m) {
  if (this != &m) {
    rows_ = m.rows_;
    cols_ = m.cols_;
    buffer_ = m.buffer_;

    m.rows_ = m.cols_ = 0;
    m.buffer_ = cl::Buffer();
  }
  return *this;
}

template <typename T>
Matrix<T> DMatrix<T>::ToHost() const {
  shared_array<T> data(rows_ * cols_);
  MatrixQueue::instance()->memcpy(data, buffer_);
  return Matrix<T>(rows_, cols_, data);
}

template <typename T>
future<Matrix<T>, cl::Buffer> DMatrix<T>::ToHost(BlockingType block) const {
  shared_array<T> data(rows_ * cols_);
  auto f = MatrixQueue::instance()->memcpy(data, buffer_, block);
  return future<Matrix<T>, cl::Buffer>(Matrix<T>(rows_, cols_, data), buffer_,
                                       f.event());
}

template <typename T>
void DMatrix<T>::ToHost(Matrix<T>* m) const {
  if (m->rows() != rows_ || m->cols() != cols_)
    m->resize(rows_, cols_);
  MatrixQueue::instance()->memcpy(m->data(), buffer_);
}

template <typename T>
void DMatrix<T>::UpdateData(const Matrix<T>& m) {
  if (rows_ != m.rows() || cols_ != m.cols()) {
    rows_ = m.rows();
    cols_ = m.cols();
    buffer_ = cl::Buffer(MatrixQueue::instance()->context(),
                         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         rows_ * cols_ * sizeof(T), m.data().get_raw());
  } else {
    MatrixQueue::instance()->memcpy(buffer_, m.data());
  }
}

template <typename T>
future<DMatrix<T>, shared_array<T>> DMatrix<T>::UpdateData(const Matrix<T>& m,
                                                           BlockingType block) {
  if (rows_ != m.rows() || cols_ != m.cols()) {
    rows_ = m.rows();
    cols_ = m.cols();
    buffer_ = cl::Buffer(MatrixQueue::instance()->context(), CL_MEM_READ_WRITE,
                         rows_ * cols_ * sizeof(T));
  }
  auto f = MatrixQueue::instance()->memcpy(buffer_, m.data(), block);
  return future<DMatrix<T>, shared_array<T>>(DMatrix<T>(rows_, cols_, buffer_),
      m.data(), f.event());
}

template <typename T> std::string PrintType();
template <> std::string PrintType<int>() { return "int"; }
template <> std::string PrintType<float>() { return "float"; }
template <> std::string PrintType<double>() { return "double"; }

template <typename T>
future<DMatrix<T>, VecBuffers> operator+(const DMatrix<T>& m1,
                                         const DMatrix<T>& m2) {
  assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
  Queue *queue = MatrixQueue::instance();

  BufferArg m1_arg(m1.buffer(), ArgType::IN);
  BufferArg m2_arg(m2.buffer(), ArgType::IN);
  size_t size = sizeof(T) * m1.rows() * m1.cols();
  cl::Buffer out = queue->CreateBuffer<int>(BufferType::WriteOnly, size);
  BufferArg out_arg(out, ArgType::OUT);

  char options[512] = {0};
  std::snprintf(options, sizeof(options), "-D VAR_TYPE=%s",
                PrintType<T>().c_str());
  Task task = queue->CreateTask("matrix.cl", "matrix_add", options, m1_arg,
                               m2_arg, out_arg);
  Grid grid = Grid(cl::NDRange(m1.rows(), m1.cols()));
  auto f = queue->EnqueueTask(task, grid);
  return future<DMatrix<T>, VecBuffers>(
      DMatrix<T>(m1.rows(), m1.cols(), out),
      {m1.buffer(), m2.buffer()}, f.event());
}

template <typename T>
future<DMatrix<T>, VecBuffers> operator-(const DMatrix<T>& m1,
                                         const DMatrix<T>& m2) {
  assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
  Queue *queue = MatrixQueue::instance();

  BufferArg m1_arg(m1.buffer(), ArgType::IN);
  BufferArg m2_arg(m2.buffer(), ArgType::IN);
  size_t size = sizeof(T) * m1.rows() * m1.cols();
  cl::Buffer out = queue->CreateBuffer<int>(BufferType::WriteOnly, size);
  BufferArg out_arg(out, ArgType::OUT);

  char options[512] = {0};
  std::snprintf(options, sizeof(options), "-D VAR_TYPE=%s",
                PrintType<T>().c_str());
  Task task = queue->CreateTask("matrix.cl", "matrix_sub", options, m1_arg,
                               m2_arg, out_arg);
  Grid grid = Grid(cl::NDRange(m1.rows(), m1.cols()));
  auto f = queue->EnqueueTask(task, grid);
  return future<DMatrix<T>, VecBuffers>(
      DMatrix<T>(m1.rows(), m1.cols(), out),
      {m1.buffer(), m2.buffer()}, f.event());
}

enum DataDir { ROW, COL };

struct matrix_param_t {
  int rows;
  int cols;
  DataDir dir;
  matrix_param_t(int rows, int cols, DataDir dir)
      : rows(rows),
        cols(cols),
        dir(dir) {
  }
};

template <typename T>
future<DMatrix<T>, VecBuffers> operator*(const DMatrix<T>& m1,
                                         const DMatrix<T>& m2) {
  assert(m1.cols() == m2.rows());
  matrix_param_t m1_param(m1.rows(), m1.cols(), DataDir::ROW);
  matrix_param_t m2_param(m2.rows(), m2.cols(), DataDir::ROW);
  Queue *queue = MatrixQueue::instance();

  BufferArg m1_arg(m1.buffer(), ArgType::IN);
  BufferArg m2_arg(m2.buffer(), ArgType::IN);
  size_t size = sizeof(T) * m1.rows() * m2.cols();
  cl::Buffer out = queue->CreateBuffer<int>(BufferType::WriteOnly, size);
  BufferArg out_arg(out, ArgType::OUT);
  cl::Buffer m1p = cl::Buffer(queue->context(),
                              CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(matrix_param_t), &m1_param);
  BufferArg m1p_arg(m1p, ArgType::IN);
  cl::Buffer m2p = cl::Buffer(queue->context(),
                              CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(matrix_param_t), &m2_param);
  BufferArg m2p_arg(m2p, ArgType::IN);

  char options[512] = {0};
  int block_size = MatrixQueue::block_size;
  std::snprintf(options, sizeof(options), "-D BLOCK_SIZE=%d -D VAR_TYPE=%s",
                block_size, PrintType<T>().c_str());
  Task task = queue->CreateTask("matrix.cl", "matrix_mul",
                               options, m1_arg, m1p_arg, m2_arg, m2p_arg,
                               out_arg);
  Grid grid = Grid(cl::NDRange(m2.cols(), m1.rows()),
                   cl::NDRange(block_size, block_size));
  auto f = queue->EnqueueTask(task, grid);
  return future<DMatrix<T>, VecBuffers>(
      DMatrix<T>(m1.rows(), m2.cols(), out),
      {m1.buffer(), m2.buffer()}, f.event());
}

}  // namespace oclalgo

#endif  // INC_OCLALGO_DMATRIX_H_

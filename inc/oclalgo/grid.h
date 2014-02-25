/*! @file grid.h
 *  @brief Contains oclalgo::Grid class.
 *  @author Senin Dmitry <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_OCLALGO_GRID_H_
#define INC_OCLALGO_GRID_H_

namespace oclalgo {

/*!
 * @brief Class for setting global, local and offset dimensions for OpenCL task.
 */
class Grid {
 public:
  explicit Grid(const cl::NDRange& global)
      : global_(global),
        local_(cl::NullRange),
        offset_(cl::NullRange) {
  }

  Grid(const cl::NDRange& global, const cl::NDRange& local)
      : global_(global),
        local_(local),
        offset_(cl::NullRange) {
  }

  Grid(const cl::NDRange& offset, const cl::NDRange& global,
       const cl::NDRange& local)
      : global_(global),
        local_(local),
        offset_(offset) {
  }

  const cl::NDRange& global() const noexcept { return global_; }
  cl::NDRange& global() noexcept { return global_; }
  const cl::NDRange& local() const noexcept { return local_; }
  cl::NDRange& local() noexcept { return local_; }
  const cl::NDRange& offset() const noexcept { return offset_; }
  cl::NDRange& offset() noexcept { return offset_; }

 private:
  cl::NDRange global_;
  cl::NDRange local_;
  cl::NDRange offset_;
};

}  // namespace oclalgo

#endif  // INC_OCLALGO_GRID_H_

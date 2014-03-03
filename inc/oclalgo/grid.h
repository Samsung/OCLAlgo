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

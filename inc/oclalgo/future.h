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

/*! @file future.h
 *  @brief Contains oclalgo::future class.
 *  @author Senin Dmitry <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_OCLALGO_FUTURE_H_
#define INC_OCLALGO_FUTURE_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <vector>

namespace oclalgo {

/** @brief Class for synchronization tasks with host thread. */
template <typename T>
class future {
 public:
  /*!
   * @brief Creates object of oclalgo::future class using move copy of future
   * result and copy OpenCL event for synchronization.
   *
   * @param future_result stored future result object
   * @param event OpenCL event for task synchronization
   */
  future(T&& future_result, const cl::Event& event);

  future(const future&) = delete;
  future& operator=(const future&) = delete;

  future(future&& f);
  future& operator=(future&& f);

  virtual ~future() = default;

  /** @brief Stop host thread and wait the end of corresponding task. */
  T get();
  /** @brief Stop host thread and wait the end of corresponding task. */
  void wait() const;

  cl::Event event() const noexcept { return event_; }

 private:
  T future_result_;
  cl::Event event_;
};

template <typename T>
future<T>::future(T&& future_result, const cl::Event& event)
    : future_result_(std::move(future_result)),
      event_(event) {
}

template <typename T>
future<T>::future(future&& f)
    : future_result_(std::move(f.future_result_)),
      event_(f.event_) {
  f.future_result_ = T();
  f.event_ = cl::Event();
}

template <typename T>
T future<T>::get() {
  if (event_()) {
    event_.wait();
    return std::move(future_result_);
  } else {
    throw cl::Error(CL_INVALID_EVENT, "null event in future::get()");
  }
}

template <typename T>
void future<T>::wait() const {
  if (event_())
    event_.wait();
  else
    throw cl::Error(CL_INVALID_EVENT, "null event in future::wait()");
}

}  // namespace oclalgo

#endif  // INC_OCLALGO_FUTURE_H_

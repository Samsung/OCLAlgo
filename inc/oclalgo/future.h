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
template <typename T, typename U>
class future {
 public:
  /** @brief Constructor of oclalgo::future<T> class.
   *  @param future_result stored future result object
   *  @param stored_obj object, which is saved to exclude memory release
   *                    during the task execution
   *  @param event OpenCL event for task synchronization
   */
  future(const T& future_result, const U& stored_obj,
         const cl::Event& event);
  future(T&& future_result, const U& stored_obj, const cl::Event& event);
  future(T&& future_result, U&& stored_obj, const cl::Event& event);

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
  U stored_obj_;
  cl::Event event_;
};

template <typename T, typename U>
future<T, U>::future(const T& future_result, const U& stored_obj,
                     const cl::Event& event)
    : future_result_(future_result),
      stored_obj_(stored_obj),
      event_(event) {
}

template <typename T, typename U>
future<T, U>::future(T&& future_result, const U& stored_obj,
                     const cl::Event& event)
    : future_result_(std::move(future_result)),
      stored_obj_(stored_obj),
      event_(event) {
}

template <typename T, typename U>
future<T, U>::future(T&& future_result, U&& stored_obj,
                     const cl::Event& event)
    : future_result_(std::move(future_result)),
      stored_obj_(std::move(stored_obj)),
      event_(event) {
}

template <typename T, typename U>
future<T, U>::future(future&& f)
    : future_result_(std::move(f.future_result_)),
      stored_obj_(std::move(f.stored_obj_)),
      event_(f.event_) {
  f.future_result_ = T();
  f.stored_obj_ = U();
  f.event_ = cl::Event();
}

template <typename T, typename U>
T future<T, U>::get() {
  if (event_()) {
    event_.wait();
    return std::move(future_result_);
  } else {
    throw cl::Error(CL_INVALID_EVENT, "null event in future::get()");
  }
}

template <typename T, typename U>
void future<T, U>::wait() const {
  if (event_())
    event_.wait();
  else
    throw cl::Error(CL_INVALID_EVENT, "null event in future::wait()");
}

typedef std::vector<cl::Buffer> VecBuffers;

}  // namespace oclalgo

#endif  // INC_OCLALGO_FUTURE_H_

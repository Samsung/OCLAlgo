/*! @file shared_array.h
 *  @brief Implementation of template class oclalgo::shared_array.
 *  @author Senin Dmitry <d.senin@samsung.com>
 *  @version 1.0
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef OCLALGO_SHARED_ARRAY_H_
#define OCLALGO_SHARED_ARRAY_H_

#include <algorithm>

namespace oclalgo {

/** @brief Providing shared array storage. */
template<typename T>
class shared_array {
 public:
  typedef T element_type;
  shared_array() : size_(0) {
  }

  shared_array(T* arrayPtr, size_t size)
      : sptr_(arrayPtr, std::default_delete<T[]>()),
        size_(size) {
  }

  template<typename D>
  shared_array(T* arrayPtr, size_t size, const D& del)
      : sptr_(arrayPtr, del),
        size_(size) {
  }

  shared_array(const oclalgo::shared_array<T>& array)
      : sptr_(array.sptr_),
        size_(array.size_) {
  }

  shared_array& operator=(const oclalgo::shared_array<T>& array) {
    if (array != *this) {
      sptr_ = array.sptr_;
      size_ = array.size_;
    }
    return *this;
  }

  void reset() {
    sptr_.reset();
    size_ = 0;
  }

  void reset(T* arrayPtr, size_t size) {
    sptr_.reset(arrayPtr, std::default_delete<T[]>());
    size_ = size;
  }

  template<typename D>
  void reset(T* arrayPtr, size_t size, const D& del) {
    sptr_.reset(arrayPtr, del);
    size_ = size;
  }

  const T& operator[](std::ptrdiff_t i) const noexcept {
    return *(sptr_.get() + i);
  }
  T& operator[](std::ptrdiff_t i) noexcept {
    return *(sptr_.get() + i);
  }

  T* get() const noexcept { return sptr_.get(); }
  bool unique() const noexcept { return sptr_.unique(); }
  long use_count() const noexcept { return sptr_.use_count(); }

  void swap(shared_array<T>& array) {
    sptr_.swap(array.sptr_);
    std::swap(size_, array.size_);
  }

  size_t size() const noexcept { return size_; }
  size_t memsize() const noexcept { return sizeof(T) * size_; }

 private:
  std::shared_ptr<T> sptr_;  // ptr to the shared memory, which contains array
  size_t size_;              // number of elements in array
};

template<class T>
bool operator==(const oclalgo::shared_array<T>& a,
                const oclalgo::shared_array<T>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

template<class T>
bool operator!=(const oclalgo::shared_array<T>& a,
                const oclalgo::shared_array<T>& b) {
  return !(a == b);
}

template<class T>
bool operator<(const oclalgo::shared_array<T>& a,
               const oclalgo::shared_array<T>& b) {
  if (a.size() < b.size()) return true;
  else if (a.size() > b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] < b[i]) return true;
    else if (a[i] > b[i]) return false;
  }
  return false;
}

template<class T>
void swap(oclalgo::shared_array<T>& a, oclalgo::shared_array<T>& b) {
  a.swap(b);
}

} // namespace oclalgo

#endif  // OCLALGO_SHARED_ARRAY_H_

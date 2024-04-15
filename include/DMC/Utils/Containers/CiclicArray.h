#pragma once

//******************//
//---DEPENDENCIES---//
//******************//
#include <cmath>

namespace DMC {

template <typename T> class CyclicArray {
public:
  CyclicArray(int size = 1) : size_(size) { cont_ = new T[size]; }
  CyclicArray(const CyclicArray &right) : size_(right.size_) {
    cont_ = new T[size_];
    for (int i = 0; i != size_; i++)
      cont_[i] = right.cont_[i];
  }
  ~CyclicArray() { delete[] cont_; }

  CyclicArray &operator=(const CyclicArray &right) {
    resize(right.size_);
    for (int i = 0; i != size_; i++)
      cont_[i] = right.cont_[i];

    return *this;
  }

  void resize(int size) {
    delete[] cont_;
    cont_ = new T[size], size_ = size;
  }
  void fill(T value) {
    for (int i = 0; i != size_; i++)
      cont_[i] = value;
  }

  T &operator[](int idx) {
    return cont_[idx >= 0 ? idx % size_ : size_ + (idx % size_)];
  }
  T operator[](int idx) const {
    return cont_[idx >= 0 ? idx % size_ : size_ + (idx % size_)];
  }

  int size() const { return size_; }

private:
  T *cont_;
  int size_;

}; // CyclicArray

} // namespace DMC

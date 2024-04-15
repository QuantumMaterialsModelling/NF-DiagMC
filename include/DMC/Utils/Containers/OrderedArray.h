#pragma once

//******************//
//---DEPENDENCIES---//
//******************//
#include <array>
#include <cstdio>

namespace DMC {

template <typename T, int N> class OrderedArray {
public:
  OrderedArray(const T &beg, const T &end) {
    data_[0] = end;
    data_[1] = beg;

    next_[0] = prev_[1] = -1;
    next_[1] = 0;
    prev_[0] = 1;

    for (int i = 0; i != N; i++)
      perm_[i] = i;

    size_ = 2;
  }

  void add(const T &name, int where) {
    int idx = perm_[where];
    int ins = perm_[size_];

    printf("Before add with size %d\n", size_);
    print();

    prev_[ins] = idx;
    prev_[next_[idx]] = ins;
    next_[ins] = next_[idx];
    next_[idx] = ins;

    data_[ins] = name;

    printf("After add\n");
    print();
    printf("\n");
    size_++;
  }

  void rem(int del) {
    // Since the permutation is composed only of swaps it's the inverse of
    // itself, you can prove it by showing how a single swap is the inverse of
    // itself and it's also symmetric.
    int where = perm_[del];

    printf("Before rem whith size %d and %d\n", size_, where);
    print();

    if (where != size_ - 1) {
      std::swap(perm_[where], perm_[size_ - 1]);
    }

    prev_[next_[del]] = prev_[del];
    next_[prev_[del]] = next_[del];

    printf("After rem\n");
    print();
    printf("\n");
    size_--;
  }

  int operator()(int where) const { return perm_[where]; }
  T &operator[](int idx) { return data_[idx]; }

  int begin() const { return 1; }
  int end() const { return 0; }
  int next(int idx) const { return next_[idx]; }
  int prev(int idx) const { return prev_[idx]; }
  int size() const { return size_; }

  std::array<T, N> data() const { return data_; }

  void print() {
    for (int i = 0; i != 10; i++) {
      printf("%f %2d %2d %2d\n", data_[i], next_[i], prev_[i], perm_[i]);
    }
  }

  void clear() {
    next_[0] = prev_[1] = -1;
    next_[1] = 0;
    prev_[0] = 1;

    for (int i = 0; i != N; i++)
      perm_[i] = i;

    size_ = 2;
    size_ = 2;
  }

private:
  std::array<T, N> data_;
  std::array<int, N> perm_;
  std::array<int, N> prev_, next_;
  int size_{0};
};

} // namespace DMC

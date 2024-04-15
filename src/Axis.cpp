#include <DMC/Utils/Math/Axis.h>
#include <cstring>
#include <iostream>

using namespace DMC;

Axis::Axis(int size) : size_(size) { ax_ = new double[size]; }

Axis::~Axis() { delete[] ax_; }

Axis::Axis(Axis &&ax) {
  size_ = ax.size_;

  ax_ = new double[size_];
  memcpy(ax_, ax.ax_, size_ * sizeof(double));
}

Axis::Axis(const Axis &ax) {
  size_ = ax.size_;

  ax_ = new double[size_];
  memcpy(ax_, ax.ax_, size_ * sizeof(double));
}

Axis &Axis::operator=(const Axis &ax) {
  size_ = ax.size_;

  ax_ = new double[size_];
  memcpy(ax_, ax.ax_, size_ * sizeof(double));

  return *this;
}

double Axis::operator[](int i) const {
  if (i > size_ || i < -1) {
    std::cerr << "Axis: index out of bound!" << std::endl;
    exit(1);
  }

  if (i == -1)
    return ax_[size_ - 1];
  else
    return ax_[i];
}

int Axis::operator[](double x) const {
  // If out of bounds return -2
  if (x > ax_[size_ - 1] || x < ax_[0])
    return -2;

  // Starting binary search
  int beg = 0, end = size_ - 1, mid = (beg + end) * 0.5;

  while (beg != end - 1) {
    mid = (beg + end) * 0.5;

    if (x > ax_[mid])
      beg = mid;
    else
      end = mid;
  }

  return beg;
}

Axis Axis::linspace(double beg, double end, int size) {
  Axis res(size);

  for (int i = 0; i != size; i++)
    res.ax_[i] = beg + i * (end - beg) / (size - 1);

  return res;
}

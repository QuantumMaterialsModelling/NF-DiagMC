#pragma once

//******************//
//---DEPENDENCIES---//
//******************//
#include <cstdlib>
#include <iostream>

namespace DMC {

class Axis {
public:
  Axis(Axis &&ax);
  Axis(const Axis &ax);
  ~Axis();

  inline int size() const { return size_; }

  Axis &operator=(const Axis &ax);
  int operator[](double x) const;
  double operator[](int i) const;

  static Axis linspace(double beg, double end, int size);

private:
  Axis(int size);

private:
  double *ax_;
  int size_;
}; // Axis

} // namespace DMC

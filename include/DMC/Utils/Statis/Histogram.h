#pragma once

//******************//
//---DEPENDENCIES---//
//******************//
#include <DMC/Utils/Math/Axis.h>

#include <string>

namespace DMC {

class Histogram {
public:
  Histogram(int size, int n_dim = 1);
  Histogram(double beg, double end, int size, int n_dim = 1);
  ~Histogram();

  inline void add(int i, int dim = 0) { hist_[dim][i]++; }
  inline void add(double x, int dim = 0) {
    int i = ax_[x];
    if (i != -2)
      hist_[dim][i]++;
  }

  int sum(int dim = 0) const;
  void print(const std::string &path, bool print_ax = true);

  int dim() const { return n_dim_; }
  int size() const { return ax_.size(); }

  const int *operator[](int which) const { return hist_[which]; }

private:
  Axis ax_;
  int **hist_;
  int n_dim_;

}; // Histogram

} // namespace DMC

#pragma once

//******************//
//---DEPENDENCIES---//
//******************//
#include <DMC/Utils/Math/Axis.h>
#include <DMC/Utils/Math/D1Function.h>
#include <vector>

namespace DMC {

class LinearFunApprox : public D1Function {
public:
  LinearFunApprox();
  LinearFunApprox(const D1Function &fun, double beg, double end, int size);
  LinearFunApprox(const Axis &ax, double *y);
  LinearFunApprox(const Axis &x, std::vector<double> y);
  LinearFunApprox(const LinearFunApprox &x);
  ~LinearFunApprox();

  LinearFunApprox &operator=(const LinearFunApprox &x);

  inline double operator()(double x) const override {
    int i = (x - ax_[0]) / dx_;

    return table_[i] + der_[i] * (x - ax_[i]);
  }

  void setup(const D1Function &fun, double beg, double end, int size = 100);

  const Axis &axis() const { return ax_; }

  void print(const std::string &path);

private:
  void setup(const D1Function *fun);
  void setup(double *y);

private:
  Axis ax_;
  double *table_, *der_, dx_;

}; // LinearFunApprox

} // namespace DMC

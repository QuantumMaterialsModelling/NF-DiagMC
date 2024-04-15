#pragma once

//******************//
//---DEPENDENCIES---//
//******************//
#include <DMC/Utils/Math/Axis.h>
#include <DMC/Utils/Math/D1Function.h>
#include <vector>

namespace DMC {

class NewtonFunApprox : public D1Function {
public:
  NewtonFunApprox();
  NewtonFunApprox(const D1Function &fun, double beg, double end,
                  int size = 100);
  NewtonFunApprox(const Axis &x, double *y);
  NewtonFunApprox(const Axis &x, std::vector<double> y);
  NewtonFunApprox(const NewtonFunApprox &x);
  NewtonFunApprox(NewtonFunApprox &&x);
  ~NewtonFunApprox();

  NewtonFunApprox &operator=(const NewtonFunApprox &x);

  double operator()(double x) const override;

  void set_fun(const D1Function &fun);
  void set_ax(double beg, double end, int size);
  void set_ax(const Axis &ax);

  void setup(const D1Function &fun, double beg, double end, int size = 100);

  void print(const std::string &path);

private:
  void setup(const D1Function *fun);
  void setup(double *y);

private:
  Axis ax_;
  double *coeff_;
}; // NewtonFunApprox

} // namespace DMC

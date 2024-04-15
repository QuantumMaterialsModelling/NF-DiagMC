#pragma once

//******************//
//---DEPENDENCIES---//
//******************//
#include <DMC/Utils/Math/D1Function.h>
#include <DMC/Utils/Math/Interpolators/LinearFunApprox.h>

namespace DMC {

class Bijection : public D1Function {
public:
  Bijection() = default;
  Bijection(const D1Function &fun, double beg, double end, bool rising = true,
            int size = 100, double toll = 1E-10);

  inline double operator()(double x) const override { return inverse_(x); }

  void setup(const D1Function &fun, double beg, double end, bool rising = true,
             int size = 100, double toll = 1E-10);

  void print(const std::string &path);

private:
  LinearFunApprox create_inverse(const D1Function *fun, double beg, double end,
                                 bool rising, int size, double toll);

private:
  LinearFunApprox inverse_;

}; // Bijection

} // namespace DMC

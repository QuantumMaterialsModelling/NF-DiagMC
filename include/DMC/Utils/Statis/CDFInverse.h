#pragma once

//******************//
//---DEPENDENCIES---//
//******************//
#include <DMC/Utils/Math/D1Function.h>
#include <DMC/Utils/Math/Interpolators/LinearFunApprox.h>
#include <DMC/Utils/Math/Interpolators/NewtonFunApprox.h>

#include <cstdio>
#include <random>

namespace DMC {

namespace CDFInverse {

class Linear {
public:
  Linear() = default;
  Linear(const D1Function &pdf, double beg, double end, int size = 100,
         double toll = 1E-8);

  void setup(const D1Function &pdf, double beg, double end, int size = 100,
             double toll = 1E-8);

  template <typename URNG> inline double operator()(URNG &rng) {
    return inverse_(std::uniform_real_distribution<double>(0., 1.)(rng));
  };

  void print(const std::string &path) {
    FILE *file = fopen(path.data(), "w");

    auto x = Axis::linspace(0, 1., 1E3);
    for (int i = 0; i != 1E3; i++)
      fprintf(file, "%f\t%f\n", x[i], inverse_(x[i]));
    fclose(file);
  }

private:
  LinearFunApprox create_inverse(const D1Function *pdf, double beg, double end,
                                 int size, double toll);

private:
  LinearFunApprox inverse_;

}; // Linear

class Newton {
public:
  Newton(const D1Function &pdf, double beg, double end, int size = 100,
         double toll = 1E-8);

  void setup(const D1Function &pdf, double beg, double end, int size = 100,
             double toll = 1E-8);

  template <typename URNG> inline double operator()(URNG &rng) {
    return inverse_(std::uniform_real_distribution<double>(0., 1.)(rng));
  };

  void print(const std::string &path) {
    FILE *file = fopen(path.data(), "w");

    auto x = Axis::linspace(0, 1., 1E3);
    for (int i = 0; i != 1E3; i++)
      fprintf(file, "%f\t%f\n", x[i], inverse_(x[i]));
    fclose(file);
  }

private:
  NewtonFunApprox create_inverse(const D1Function *pdf, double beg, double end,
                                 int size, double toll);

private:
  NewtonFunApprox inverse_;

}; // Linear

} // namespace CDFInverse

} // namespace DMC

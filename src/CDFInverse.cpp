#include <DMC/Utils/Math/Interpolators/LinearFunApprox.h>
#include <DMC/Utils/Statis/CDFInverse.h>
#include <array>
#include <boost/math/quadrature/gauss.hpp>

using namespace DMC;
using namespace boost::math::quadrature;

CDFInverse::Linear::Linear(const D1Function &pdf, double beg, double end,
                           int size, double toll)
    : inverse_(create_inverse(&pdf, beg, end, size, toll)) {}

void CDFInverse::Linear::setup(const D1Function &pdf, double beg, double end,
                               int size, double toll) {
  inverse_ = create_inverse(&pdf, beg, end, size, toll);
}

LinearFunApprox CDFInverse::Linear::create_inverse(const D1Function *pdf,
                                                   double beg, double end,
                                                   int size, double toll) {
  std::vector<double> y(size);
  auto ax = Axis::linspace(0, 1, size);
  double mid, value = -1;
  auto f = [pdf](const double &x) { return pdf->operator()(x); };

  y[0] = beg, y[size - 1] = end;
  for (int i = 1; i != size - 1; i++) {
    beg = y[0], end = y[size - 1];

    while (std::abs(value - ax[i]) > toll) {
      mid = 0.5 * (beg + end);
      value = gauss<double, 30>::integrate(f, y[0], mid);

      if (value > ax[i])
        end = mid;
      else
        beg = mid;
    }

    y[i] = mid;
  }

  return LinearFunApprox(ax, &y[0]);
}

CDFInverse::Newton::Newton(const D1Function &pdf, double beg, double end,
                           int size, double toll)
    : inverse_(create_inverse(&pdf, beg, end, size, toll)) {}

void CDFInverse::Newton::setup(const D1Function &pdf, double beg, double end,
                               int size, double toll) {
  inverse_ = create_inverse(&pdf, beg, end, size, toll);
}

NewtonFunApprox CDFInverse::Newton::create_inverse(const D1Function *pdf,
                                                   double beg, double end,
                                                   int size, double toll) {
  std::vector<double> y(size);
  auto ax = Axis::linspace(0, 1, size);
  double mid, value = -1;
  auto f = [pdf](const double &x) { return pdf->operator()(x); };

  y[0] = beg, y[size - 1] = end;
  for (int i = 1; i != size - 1; i++) {
    beg = y[0], end = y[size - 1];

    while (std::abs(value - ax[i]) > toll) {
      mid = 0.5 * (beg + end);
      value = gauss<double, 20>::integrate(f, y[0], mid);

      if (value > ax[i])
        end = mid;
      else
        beg = mid;
    }

    y[i] = mid;
  }

  return NewtonFunApprox(ax, &y[0]);
}

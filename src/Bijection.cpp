#include <DMC/Utils/Math/D1Function.h>
#include <DMC/Utils/Math/Inverters/Bijection.h>
#include <cstdio>
#include <iterator>

using namespace DMC;

Bijection::Bijection(const D1Function &fun, double beg, double end, bool rising,
                     int size, double toll)
    : inverse_(create_inverse(&fun, beg, end, rising, size, toll)) {}

void Bijection::setup(const D1Function &fun, double beg, double end,
                      bool rising, int size, double toll) {
  inverse_ = create_inverse(&fun, beg, end, rising, size, toll);
}

LinearFunApprox Bijection::create_inverse(const D1Function *fun, double beg,
                                          double end, bool rising, int size,
                                          double toll) {
  std::vector<double> y(size);
  double mid, value = -1;

  auto ax = Axis::linspace(fun->operator()(beg), fun->operator()(end), size);
  y[0] = beg, y[size - 1] = end;

  if (!rising) {
    ax = Axis::linspace(fun->operator()(end), fun->operator()(beg), size);
    y[0] = end, y[size - 1] = beg;
  }

  for (int i = 1; i != size - 1; i++) {
    beg = y[0], end = y[size - 1];

    while (std::abs(value - ax[i]) > toll) {
      mid = 0.5 * (beg + end);
      value = fun->operator()(mid);

      if (value > ax[i])
        end = mid;
      else
        beg = mid;
    }

    y[i] = mid;
  }

  return LinearFunApprox(ax, &y[0]);
}

void Bijection::print(const std::string &path) {
  FILE *file = fopen(path.data(), "w");

  if (!file) {
    printf("Bijection: problems opening file!\n");
    exit(1);
  }

  auto x = Axis::linspace(inverse_.axis()[0], inverse_.axis()[-1], 1E3);
  for (int i = 0; i != 1E3; i++)
    fprintf(file, "%f\t%f\n", x[i], inverse_(x[i]));
  fclose(file);
}

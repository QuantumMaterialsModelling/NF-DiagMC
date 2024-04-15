#include <DMC/Utils/Math/Interpolators/NewtonFunApprox.h>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace DMC;

NewtonFunApprox::NewtonFunApprox() : ax_(Axis::linspace(0, 0, 2)) {}

NewtonFunApprox::NewtonFunApprox(const D1Function &fun, double beg, double end,
                                 int size)
    : ax_(Axis::linspace(beg, end, size)) {
  coeff_ = (double *)calloc(size, sizeof(double));

  setup(&fun);
}

NewtonFunApprox::NewtonFunApprox(const Axis &x, double *y) : ax_(x) {
  coeff_ = (double *)calloc(x.size(), sizeof(double));

  setup(y);
}

NewtonFunApprox::NewtonFunApprox(const Axis &x, std::vector<double> y)
    : ax_(x) {
  if (x.size() != y.size()) {
    printf("NewtonFunApprox: Axis and Ascis sizes are not equal!\n");
    exit(1);
  }

  coeff_ = (double *)calloc(x.size(), sizeof(double));

  setup(&y[0]);
}

NewtonFunApprox::NewtonFunApprox(const NewtonFunApprox &x) : ax_(x.ax_) {
  coeff_ = new double[x.ax_.size()];

  for (int i = 0; i != ax_.size(); i++) {
    coeff_[i] = x.coeff_[i];
  }
}

NewtonFunApprox::NewtonFunApprox(NewtonFunApprox &&x) : ax_(x.ax_) {
  coeff_ = new double[x.ax_.size()];

  for (int i = 0; i != ax_.size(); i++) {
    coeff_[i] = x.coeff_[i];
  }
}

NewtonFunApprox::~NewtonFunApprox() { delete[] coeff_; }

NewtonFunApprox &NewtonFunApprox::operator=(const NewtonFunApprox &x) {
  ax_ = x.ax_;

  coeff_ = new double[x.ax_.size()];

  for (int i = 0; i != ax_.size(); i++) {
    coeff_[i] = x.coeff_[i];
  }
  return *this;
}

double NewtonFunApprox::operator()(double x) const {
  double res = 0, prod = 1;

  for (int i = 0; i != ax_.size(); i++) {
    res += coeff_[i] * prod;
    prod *= (x - ax_[i]);
  }

  return res;
}

void NewtonFunApprox::setup(const D1Function *fun) {
  std::vector<double> y;

  for (int i = 0; i != ax_.size(); i++)
    y.push_back(fun->operator()(ax_[i]));

  setup(&y[0]);
}

void NewtonFunApprox::setup(double *y) {
  double den;

  for (int i = 0; i != ax_.size(); i++) {
    den = 1;
    for (int j = 0; j != i; j++)
      den *= (ax_[i] - ax_[j]);
    coeff_[i] = (y[i] - (*this)(ax_[i])) / den;
  }
}

void NewtonFunApprox::print(const std::string &path) {
  FILE *file = fopen(path.data(), "w");

  if (!file) {
    printf("LinearFunApprox: problems opening file!\n");
    exit(1);
  }

  auto x = Axis::linspace(ax_[0], ax_[-1], 1E3);
  for (int i = 0; i != 1E3; i++)
    fprintf(file, "%f\t%f\n", x[i], operator()(x[i]));
  fclose(file);
}

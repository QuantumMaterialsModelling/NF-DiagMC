#include <DMC/Utils/Math/Interpolators/LinearFunApprox.h>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace DMC;

LinearFunApprox::LinearFunApprox() : ax_(Axis::linspace(0, 0, 2)) {}

LinearFunApprox::LinearFunApprox(const D1Function &fun, double beg, double end,
                                 int size)
    : ax_(Axis::linspace(beg, end, size)) {
  table_ = new double[size];
  der_ = new double[size - 1];

  setup(&fun);
}

LinearFunApprox::LinearFunApprox(const LinearFunApprox &x) : ax_(x.ax_) {
  table_ = new double[x.ax_.size()];
  der_ = new double[x.ax_.size() - 1];

  memcpy(table_, x.table_, ax_.size() * sizeof(double));
  memcpy(der_, x.der_, (ax_.size() - 1) * sizeof(double));
}

LinearFunApprox::LinearFunApprox(const Axis &x, double *y) : ax_(x) {
  table_ = new double[x.size()];
  der_ = new double[x.size() - 1];

  setup(y);
}

LinearFunApprox::LinearFunApprox(const Axis &x, std::vector<double> y)
    : ax_(x) {
  if (ax_.size() != y.size()) {
    printf("LinearFunApprox: Axis and Ascis sizes are not equal!\n");
    exit(1);
  }
  table_ = new double[x.size()];
  der_ = new double[x.size() - 1];

  setup(&y[0]);
}

LinearFunApprox::~LinearFunApprox() {
  delete[] table_;
  delete[] der_;
}

LinearFunApprox &LinearFunApprox::operator=(const LinearFunApprox &x) {
  ax_ = x.ax_;
  dx_ = x.dx_;

  table_ = new double[x.ax_.size()];
  der_ = new double[x.ax_.size() - 1];

  memcpy(table_, x.table_, ax_.size() * sizeof(double));
  memcpy(der_, x.der_, (ax_.size() - 1) * sizeof(double));

  return *this;
}

void LinearFunApprox::setup(const D1Function &fun, double beg, double end,
                            int size) {
  ax_ = Axis::linspace(beg, end, size);

  table_ = new double[size];
  der_ = new double[size - 1];

  setup(&fun);
}

void LinearFunApprox::setup(const D1Function *fun) {
  std::vector<double> y;

  for (int i = 0; i != ax_.size(); i++)
    y.push_back(fun->operator()(ax_[i]));

  setup(&y[0]);
}

void LinearFunApprox::setup(double *y) {
  memcpy(table_, y, ax_.size() * sizeof(double));
  dx_ = ax_[1] - ax_[0];

  for (int i = 1; i != ax_.size(); i++)
    der_[i - 1] = (table_[i] - table_[i - 1]) / (ax_[i] - ax_[i - 1]);
}

void LinearFunApprox::print(const std::string &path) {
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

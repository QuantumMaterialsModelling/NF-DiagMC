#include <DMC/Utils/Statis/Histogram.h>
#include <cstdio>
#include <cstdlib>

using namespace DMC;

Histogram::Histogram(int size, int n_dim)
    : ax_(Axis::linspace(0, 1, size + 1)), n_dim_(n_dim) {
  hist_ = new int *[n_dim];
  for (int i = 0; i != n_dim; i++)
    // Allocates and sets to zero
    hist_[i] = (int *)calloc(size, sizeof(int));
}

Histogram::Histogram(double beg, double end, int size, int n_dim)
    : ax_(Axis::linspace(beg, end, size + 1)), n_dim_(n_dim) {
  hist_ = new int *[n_dim];
  for (int i = 0; i != n_dim; i++)
    // Allocates and sets to zero
    hist_[i] = (int *)calloc(size, sizeof(int));
}

Histogram::~Histogram() { delete[] hist_; }

int Histogram::sum(int dim) const {
  if (dim < 0 || dim >= n_dim_) {
    std::cerr << "Histogram: tried to sum on invalid dimension!" << std::endl;
    exit(1);
  }

  int res = 0;

  for (int i = 0; i != ax_.size(); i++)
    res += hist_[dim][i];

  return res;
}

void Histogram::print(const std::string &path, bool print_ax) {
  FILE *file = fopen(path.data(), "w");

  if (!file) {
    std::cerr << "Histogram: unable to open file!" << std::endl;
    exit(1);
  }

  for (int i = 0; i != ax_.size() - 1; i++) {
    if (print_ax)
      fprintf(file, "%f\t", 0.5 * (ax_[i + 1] + ax_[i]));

    for (int j = 0; j != n_dim_ - 1; j++) {
      fprintf(file, "%d\t", hist_[j][i]);
    }

    fprintf(file, "%d\n", hist_[n_dim_ - 1][i]);
  }

  fclose(file);
}

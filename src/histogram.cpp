#include <NF/Utils/histogram.h>
#include <bits/types/FILE.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>

Histogram::Histogram(int size, double beg, double end, int dim)
    : size(size), dim(dim) {
  dx_ = (end - beg) / size;

  hist_ = new int *[dim];
  for (int i = 0; i != dim + 1; i++) {
    hist_[i] = (int *)calloc(size, sizeof(int));
  }

  x_ = (double *)calloc(size, sizeof(double));
  for (int i = 0; i != size; i++)
    x_[i] = beg + dx_ * (i + 0.5);
}

void Histogram::add(double x, int which) {
  int idx = search(x);
  if (idx != size)
    hist_[which][idx]++;
}

void Histogram::add(int idx, int which) { hist_[which][idx]++; }

void Histogram::print(const char *path) {
  FILE *file;

  file = fopen(path, "w");
  if (file == NULL) {
    printf("Unable to open the file");
    exit(-1);
  }

  for (int i = 0; i != size; i++) {
    fprintf(file, "%f", x_[i]);
    for (int j = 0; j != dim; j++)
      fprintf(file, "\t%d", hist_[j][i]);

    fprintf(file, "\n");
  }

  fclose(file);
}

int Histogram::search(double x) {
  for (int i = 0; i != size; i++)
    if (std::abs(x - x_[i]) < 0.5 * dx_)
      return i;

  return size;
}

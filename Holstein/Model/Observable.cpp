#pragma once

#include "Diagram.cpp"
#include <cstdio>

namespace Holstein {

class Eg : public DMC::Observable<Diagram> {
public:
  Eg() : Observable("Eg") {}

  void eval() override {
    if (dia->t < 7)
      return;

    double res = 0;
    for (int i = 0; i != dia->n; i++)
      res += dia->Pe[i] - dia->Pb[i];

    val += (dia->o * res - 2 * dia->n) / dia->t;
    n++;
  }

  void conv() override { hist.push_back(val / n); }

  void print(const std::string &path) override {
    FILE *file = fopen(path.data(), "a");

    for (auto x : hist)
      fprintf(file, "%f ", x);
    fprintf(file, "\n");

    fclose(file);
  }

private:
  double val{0}, n{0};
  std::vector<double> hist;
};

class Green : public DMC::Observable<Diagram> {
public:
  Green(double beg = 0, double end = 10)
      : Observable("Green"), beg(beg), end(end) {
    for (int i = 0; i != 199; i++)
      hist[i] = 0;
  }

  void eval() override {
    if (dia->t > end) {
      n++;
      return;
    }

    int which = 200 * (dia->t - beg) / (end - beg);
    hist[which] += 1;
  }

  void print(const std::string &path) override {
    FILE *file = fopen(path.data(), "a");

    for (int i = 0; i != 199; i++)
      fprintf(file, "%f ", hist[i] / n);
    fprintf(file, "\n");

    fclose(file);
  }

private:
  double hist[199], n{0};
  double beg, end;
};

} // namespace Holstein

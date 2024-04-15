#pragma once

#include <string>

namespace MLHol {

struct Option {
  int NLayers{1};
  int NAffine{5};
  int Hidden{50};
  int MOrder{50};
  int NEpoch{10};
  int Batch{100};
  int Dsize{100};
  int n_bin{5};
  int RNGseed{2};

  double CPlateu{1E2};
  double lr{1e-4};
  double TEval{5};
  double bound{20};

  std::string Mpath;

  bool ResTrain{false};

  static Option read(const std::string &path = "./sim.ini");
};

}

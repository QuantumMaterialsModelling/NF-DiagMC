#pragma once

//******************//
//---DEPENDENCIES---//
//******************//
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace DMC {

//****************//
//---SimOptions---//
//****************//

class SimOptions {
public:
  SimOptions() = default;
  SimOptions(const std::string &path);

  void read(const std::string &path);

  double &operator[](const std::string &key) { return sim_param[key]; }
  double &operator()(const std::string &key) { return conf_param[key]; }

  auto get_conf() const { return conf_param; }
  auto get_sim() const { return sim_param; }

  auto &get_out_dir() { return out_dir; }
  auto collect_correlation() { return corr; }

private:
  static std::map<std::string, double> sim_default;

private:
  std::map<std::string, double> conf_param;
  std::map<std::string, double> sim_param;

  std::string out_dir{"null"};
  bool corr{false};
};

} // namespace DMC

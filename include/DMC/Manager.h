#pragma once

//******************//
//---DEPENDENCIES---//
//******************//
#include "Archetypes.h"
#include "SimOptions.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>

namespace DMC {

//*************//
//---MANAGER---//
//*************//

template <class DIA> class Manager {
public:
  template <typename... Args> Manager(Args &&...arg);
  Manager();
  ~Manager();

  template <class UPD, typename... Args> void add_update(Args &&...arg);
  template <class OBS, typename... Args> void add_observable(Args &&...arg);

  void set_seed(std::size_t seed);

  void simulate(const std::string &path = "./sim.ini");
  void simulate(SimOptions &options);
  void simulate(int N, int n = 1E5);
  std::pair<int, UpStatus> evolve();

  void print(const std::string &path);

private:
  void controll_unique_name();
  bool check_conv();

  static void simulate_(Manager<DIA> *manager, SimOptions options);

private:
  std::shared_ptr<DIA> dia_;
  std::mt19937_64 rng_;
  std::vector<Update<DIA> *> updates_;
  std::vector<Observable<DIA> *> observables_;

  // Inserire un vettore di interi che serve per tenere conto dell'evoluzione
  // del tempo di correlazione
  std::array<double, 20> corrTime_;
  std::vector<std::array<int, 4>> UpStatistic_;
  std::size_t seed_;
  int N_{0}, time_, thN_;
}; // Manager

//********************//
//---IMPLEMENTATION---//
//********************//

template <class DIA> Manager<DIA>::~Manager() {
  updates_.clear();
  updates_.shrink_to_fit();

  observables_.clear();
  observables_.shrink_to_fit();
}

template <class DIA> Manager<DIA>::Manager() {
  if constexpr (!std::is_base_of<Configuration, DIA>::value)
    static_assert(std::is_base_of<Configuration, DIA>::value,
                  "Manager: tried to use a Diagram that doesn't inherits from "
                  "Configuration class!");
  else {
    dia_ = std::make_shared<DIA>();
  }

  set_seed(std::random_device()());
}

template <class DIA> void Manager<DIA>::set_seed(std::size_t seed) {
  rng_.seed(seed), seed_ = seed;
  for (auto &up : updates_)
    up->set_rng(rng_);
}

template <class DIA>
template <typename... Args>
Manager<DIA>::Manager(Args &&...arg) {
  if constexpr (!std::is_base_of<Configuration, DIA>::value)
    static_assert(std::is_base_of<Configuration, DIA>::value,
                  "Manager: tried to use a Diagram that doesn't inherits from "
                  "Configuration class!");
  else {
    dia_ = std::make_shared<DIA>(arg...);
  }

  set_seed(std::random_device()());
}

template <class DIA>
template <class UPD, typename... Args>
void Manager<DIA>::add_update(Args &&...args) {
  if constexpr (!std::is_base_of<Update<DIA>, UPD>::value)
    static_assert(std::is_base_of<Update<DIA>, UPD>::value,
                  "Manager: tried to add update that doesn't inherits from "
                  "Update class!");
  else {
    updates_.push_back(new UPD(args...));
    updates_[updates_.size() - 1]->set_rng(rng_);
    updates_[updates_.size() - 1]->set_dia(dia_);
  }

  UpStatistic_.push_back({0, 0, 0, 0});
}

template <class DIA>
template <class OBS, typename... Args>
void Manager<DIA>::add_observable(Args &&...args) {
  if constexpr (!std::is_base_of<Observable<DIA>, OBS>::value)
    static_assert(std::is_base_of<Observable<DIA>, OBS>::value,
                  "Manager: tried to add observable that doesn't inherits from "
                  "Observable class!");
  else {
    observables_.push_back(new OBS(args...));
    observables_[observables_.size() - 1]->set_dia(dia_);
  }
}

template <class DIA> std::pair<int, UpStatus> Manager<DIA>::evolve() {
  int which = std::uniform_int_distribution<int>(0, updates_.size() - 1)(rng_);
  auto res = std::make_pair(which, UpStatus::ACCEPTED);

  double acc = updates_[which]->atempt();
  if (acc < 0)
    res.second = UpStatus::INVALID;
  else if (acc >= 1.)
    updates_[which]->accept();
  else if (std::uniform_real_distribution<double>(0., 1.)(rng_) < acc)
    updates_[which]->accept();
  else
    res.second = UpStatus::REJECTED;

  return res;
}

template <class DIA> void Manager<DIA>::simulate(const std::string &path) {
  // Read options
  DMC::SimOptions options(path);

  simulate(options);
}

template <class DIA> void Manager<DIA>::simulate(SimOptions &options) {
  // Define the Configuration if it is needed
  if (options.get_conf().size()) {
    dia_->set_param(options.get_conf());

    // Reset the diagram for all updates and or observables for possible
    // triggers
    for (auto up : updates_)
      up->set_dia(dia_);
    for (auto ob : observables_)
      ob->set_dia(dia_);
  }

  // Setting the random seed if needed
  if (options["RNG_SEED"] != -1)
    set_seed(options["RNG_SEED"]);

  // Simulate
  simulate_(this, options);

  // If out_dir is specified print
  if (options.get_out_dir() != "null")
    print(options.get_out_dir());
}

template <class DIA> void Manager<DIA>::simulate(int N, int n) {
  auto now = std::chrono::high_resolution_clock::now();
  N_ = N;

  // Thermalization
  for (int i = 0; i != n; i++)
    evolve();

  for (int i = 0; i != N; i++) {
    // Perform step
    auto [which, state] = evolve();

    // Evaluate observables
    for (auto obs : observables_)
      obs->eval();

    // Update statistics
    UpStatistic_[which][state]++;
    UpStatistic_[which][3]++;
  }
  auto duration = std::chrono::high_resolution_clock::now() - now;

  time_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

template <class DIA> void Manager<DIA>::print(const std::string &path) {
  // Print the simulation informations
  FILE *file = fopen((path + "simulation.out").data(), "w");

  if (!file) {
    printf("Manager: unable to open file!\n");
    exit(1);
  }

  fprintf(file, "----------Simulation informations----------\n");
  fprintf(file, "\nConfigurati name: %s\n", dia_->name());
  fprintf(file, "\nThermaliza steps: %d\n", thN_);
  fprintf(file, "\nTotal iterations: %d\n", N_);
  fprintf(file, "\nRandom seed used: %lu\n", seed_);
  fprintf(file, "\nTotal time taken: %d ms\n", time_);

  fprintf(file, "\n-------------Update Statistics-------------\n");
  fprintf(file, "\n%10s | %10s | %8s | %8s | %8s\n", "name", "inverse",
          "accepted", "rejected", "invalid");
  for (std::size_t i = 0; i != UpStatistic_.size(); i++)
    fprintf(file, "%10s | %10s | %8.6f | %8.6f | %8.6f\n", updates_[i]->name(),
            updates_[i]->inverse(),
            (double)UpStatistic_[i][0] / UpStatistic_[i][3],
            (double)UpStatistic_[i][1] / UpStatistic_[i][3],
            (double)UpStatistic_[i][2] / UpStatistic_[i][3]);

  fclose(file);

  // Print the observables
  for (auto &obs : observables_)
    obs->print(path + obs->name() + ".dat");

  // Print correlation if needed
  if (corrTime_[0] != 0.) {
    file = fopen((path + "CorrTime.dat").data(), "w");

    if (!file) {
      printf("Manger: unable to open file for Corr!\n");
      exit(1);
    }

    fprintf(file, "# Data of the correlation along the created chain\n#\n# "
                  "Step\tCorrelation\n");
    for (int i = 0; i != 20; i++)
      fprintf(file, "%d\t%f\n", i, corrTime_[i]);

    fclose(file);
  }
}

template <class DIA> bool Manager<DIA>::check_conv() {
  bool converged = true;
  for (auto obs : observables_) {
    obs->conv();

    converged = converged && obs->is_converged();
  }

  return converged;
}

template <class DIA>
void Manager<DIA>::simulate_(Manager<DIA> *manager, SimOptions options) {
  // Needed variables
  using namespace std::chrono;
  auto beg = high_resolution_clock::now();
  auto cor = std::vector<int>();
  int N = 0;

  // Thermalization
  manager->thN_ = options["THE_STEP"];
  for (int i = 0; i != manager->thN_; i++)
    manager->evolve();

  // Main loop
  for (int i = 0; i < options["MAX_STEP"]; i++) {
    N++;

    // Perform step
    auto [which, state] = manager->evolve();

    // Evaluate observables
    for (auto obs : manager->observables_)
      if (!obs->is_converged())
        obs->eval();

    // Update statistic
    manager->UpStatistic_[which][state]++;
    manager->UpStatistic_[which][3]++;

    if (options.collect_correlation())
      cor.push_back(state == UpStatus::ACCEPTED ? 0 : 1);

    // Check convergence
    if (!((i + 1) % (int)options["CONV_CKP"])) {
      auto dur =
          duration_cast<seconds>(high_resolution_clock::now() - beg).count();

      if (manager->check_conv() || dur > options["MAX_TIME"])
        break;
    }
  }

  // Storing the final results
  manager->N_ = N;
  manager->time_ =
      duration_cast<milliseconds>(high_resolution_clock::now() - beg).count();

  // Computation of the correlation time
  if (options.collect_correlation())
    for (std::size_t i = 0; i != manager->corrTime_.size(); i++) {
      manager->corrTime_[i] = 0;

      for (std::size_t j = 0; j != cor.size() - i; j++) {
        int val = cor[j];

        for (std::size_t k = 1; k <= i; k++)
          val *= cor[j + k];

        manager->corrTime_[i] += val;
      }

      manager->corrTime_[i] /= cor.size() - i;
    }
}

} // namespace DMC

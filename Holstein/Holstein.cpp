#include <DMC/Manager.h>

#include "Model/Observable.cpp"
#include "Model/Updates.cpp"

#include <cstdio>
#include <future>
#include <mutex>

using namespace DMC;
using namespace Holstein;

std::mutex lock;

std::string to_string(double g) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(1) << g;
  return stream.str();
}

void lsimulation(SimOptions opt) {
  bool standard = opt("g") < 0;

  Manager<Diagram> manager;

  manager.add_update<chg_t>();
  manager.add_update<add_n>();
  manager.add_update<rem_n>();

  manager.add_observable<Eg>();
  manager.add_observable<Green>();

  if (standard)
    opt("g") = 0.5;

  manager.simulate(opt);

  lock.lock();
  if (standard)
    manager.print("./Holstein/Results/Local/");
  else
    manager.print("./Holstein/Results/Local/" + to_string(opt("g")) + "/");
  lock.unlock();
}

void nsimulation(SimOptions opt) {
  bool standard = opt("g") < 0;

  Manager<Diagram> manager;

  manager.add_update<chg_t>();
  manager.add_update<add_n>();
  manager.add_update<nnchg_n>();
  manager.add_update<rem_n>();

  manager.add_observable<Eg>();
  manager.add_observable<Green>();

  if (standard)
    opt("g") = 0.5;
  manager.simulate(opt);

  lock.lock();
  if (standard)
    manager.print("./Holstein/Results/Neural/");
  else
    manager.print("./Holstein/Results/Neural/" + to_string(opt("g")) + "/");
  lock.unlock();
}

void prepare_file(double g = -1) {
  std::string path = "./Holstein/Results/";
  std::vector<std::string> updates = {"Local", "Neural"};
  std::vector<std::string> observa = {"Eg", "Green"};

  for (auto up : updates) {
    for (auto ob : observa) {
      FILE *file;
      if (g < 0)
        file = fopen((path + up + "/" + ob + ".dat").data(), "w");
      else
        file = fopen(
            (path + up + "/" + to_string(g) + "/" + ob + ".dat").data(), "w");

      if (!file) {
        printf("prepare_file: the folder %s/%s was not found!\n", up.data(),
               to_string(g).data());
        exit(1);
      }

      if (ob == "Green") {
        for (int i = 0; i != 199; i++)
          fprintf(file, "%f ", (i + 0.5) * (10. / 199));
        fprintf(file, "\n");
      }

      fclose(file);
    }
  }
}

void main_simulation(SimOptions opt, int nThread, double g = -1) {
  prepare_file(g);

  opt("g") = g;

  printf("main_simulation: simulation at g of %f\n", g);

  // Defining the threads
  std::vector<std::future<void>> futures(nThread);

  // Do stuff
  printf("main_simulation: started local...\n");
  for (auto &f : futures)
    f = std::async(lsimulation, opt);
  for (auto &f : futures)
    f.wait();

  printf("main_simulation: started neural...\n");
  for (auto &f : futures)
    f = std::async(nsimulation, opt);
  for (auto &f : futures)
    f.wait();
  printf("main_simulation: finished!\n\n");
}

int main(int argc, char *argv[]) {
  // Read options
  SimOptions opt("./Holstein/sim.ini");

  // Do simulations
  if (argc <= 1) {
    printf("Error: needed number of thread!\n");
    return 1;
  }

  // Reading inputs
  int nThread = std::stoi(argv[1]);

  std::vector<double> gs = {-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

  for (double g : gs)
    main_simulation(opt, nThread, g);

  return 0;
}

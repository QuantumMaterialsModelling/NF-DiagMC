#include <DMC/SimOptions.h>

#include "INI.h"

using namespace DMC;

SimOptions::SimOptions(const std::string &path) { read(path); }

std::map<std::string, double> SimOptions::sim_default = {
    {"MAX_STEP", 1E10}, {"MAX_TIME", 1E10}, {"THE_STEP", 1E6},
    {"CONV_CKP", 1E5},  {"RNG_SEED", -1},
};

void SimOptions::read(const std::string &path) {
  // Create datastructure
  mINI::INIStructure ini;

  // Read from the file
  mINI::INIFile(path).read(ini);

  //---READING---//

  // Configuration
  if (ini.has("Configuration"))
    for (auto &elem : ini["Configuration"])
      conf_param.insert({elem.first, std::stod(elem.second)});

  // Simulation
  for (const auto &[key, def] : sim_default) {
    if (ini["Simulation"].has(key))
      sim_param.insert({key, std::stod(ini["Simulation"][key])});
    else
      sim_param.insert({key, def});
  }

  // Output directory
  if (ini["Simulation"].has("OUT_DIRR"))
    out_dir = ini["Simulation"]["OUT_DIRR"];

  // Correlation time
  if (ini["Simulation"].has("GET_CORR"))
    corr = std::stoi(ini["Simulation"]["GET_CORR"]);
}

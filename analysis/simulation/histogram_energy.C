/*
Histogram layer energy deposits vs layer number from a processed events.root file.

root -l -b -q 'analysis/simulation/histogram_energy.C("data/processed/04e3fdfb/run830b668585/events.root","histogram_energy.root")'
*/

#include <TFile.h>
#include <TH2D.h>
#include <TTree.h>

#include <array>
#include <cmath>
#include <iostream>
#include <string>

namespace {

constexpr int kEnergyBinCount = 80;
constexpr int kLayerCount = 10;
constexpr const char* kTreeName = "events";

}  // namespace

void histogram_energy(const char* events_path_cstr,
                      const char* out_root_cstr = "histogram_energy.root") {
  if (!events_path_cstr || std::string(events_path_cstr).empty()) {
    std::cerr << "[histogram_energy] events.root path is required.\n";
    return;
  }

  const std::string events_path(events_path_cstr);
  const std::string out_root =
      (out_root_cstr && std::string(out_root_cstr).size())
          ? std::string(out_root_cstr)
          : std::string("histogram_energy.root");

  TFile input_file(events_path.c_str(), "READ");
  if (input_file.IsZombie()) {
    std::cerr << "[histogram_energy] Failed to open " << events_path << ".\n";
    return;
  }

  TTree* tree = nullptr;
  input_file.GetObject(kTreeName, tree);
  if (!tree) {
    std::cerr << "[histogram_energy] Tree '" << kTreeName << "' not found in "
              << events_path << ".\n";
    return;
  }

  std::array<float, kLayerCount> layer_energy {};
  for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    const std::string branch_name = "layer_" + std::to_string(layer_index) + "_E";
    if (tree->GetBranch(branch_name.c_str()) == nullptr) {
      std::cerr << "[histogram_energy] Branch '" << branch_name << "' not found in "
                << events_path << ".\n";
      return;
    }
    tree->SetBranchAddress(branch_name.c_str(), &layer_energy[static_cast<std::size_t>(layer_index)]);
  }

  double max_layer_energy = 0.0;
  const Long64_t entry_count = tree->GetEntries();
  for (Long64_t event_index = 0; event_index < entry_count; ++event_index) {
    tree->GetEntry(event_index);

    // Scan every layer once to set the y-axis range from the observed deposits.
    for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
      const double value = static_cast<double>(layer_energy[static_cast<std::size_t>(layer_index)]);
      if (std::isfinite(value) && value > max_layer_energy) {
        max_layer_energy = value;
      }
    }
  }

  const double y_max = max_layer_energy > 0.0 ? max_layer_energy * 1.05 : 1.0;
  TH2D histogram(
      "h_layer_energy_vs_layer",
      "Layer energy vs layer number;Layer index;Layer energy [GeV]",
      kLayerCount,
      -0.5,
      static_cast<double>(kLayerCount) - 0.5,
      kEnergyBinCount,
      0.0,
      y_max);
  histogram.SetDirectory(nullptr);
  histogram.SetStats(false);

  long long filled_layer_count = 0;
  for (Long64_t event_index = 0; event_index < entry_count; ++event_index) {
    tree->GetEntry(event_index);

    // Fill the 2D histogram with one point per layer deposit in each event.
    for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
      const double value = static_cast<double>(layer_energy[static_cast<std::size_t>(layer_index)]);
      if (std::isfinite(value) && value >= 0.0) {
        histogram.Fill(static_cast<double>(layer_index), value);
        filled_layer_count++;
      }
    }
  }

  TFile output_file(out_root.c_str(), "RECREATE");
  if (output_file.IsZombie()) {
    std::cerr << "[histogram_energy] Failed to open " << out_root << " for writing.\n";
    return;
  }
  histogram.Write();
  output_file.Close();

  std::cout << "[histogram_energy] Wrote " << out_root << "\n";
  std::cout << "[histogram_energy] Events=" << entry_count
            << " LayerDeposits=" << filled_layer_count << "\n";
}

/*
Particle performance summary.
Reads processed event hits and writes detection efficiency plus tile/layer activity metrics.
*/

#include <TFile.h>
#include <TTree.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kLayerCount = 10;  // Number of nHCal sampling layers in this template.

struct PerformanceStats {
  // Accumulates event counts and moments needed for mean and RMS activity.
  long long valid_event_count = 0;
  long long detected_event_count = 0;
  double tile_count_sum = 0.0;
  double tile_count_sum_squares = 0.0;
  double layer_count_sum = 0.0;
  double layer_count_sum_squares = 0.0;
};

std::string sibling_path(const std::string& path, const std::string& basename) {
  // Build a default output/input path next to the events file.
  const auto separator_index = path.find_last_of("/\\");
  if (separator_index == std::string::npos) {
    return basename;
  }
  return path.substr(0, separator_index + 1) + basename;
}

std::string string_arg_or_default(const char* value, const std::string& fallback) {
  // Treat empty optional ROOT macro arguments as defaults.
  if (value && std::string(value).size()) {
    return std::string(value);
  }
  return fallback;
}

bool load_json_file(const std::string& path,
                    const char* label,
                    nlohmann::json& json_payload,
                    const char* context_label) {
  // Load a JSON file and print a context-specific error on failure.
  std::ifstream input(path);
  if (!input) {
    std::cerr << "[" << context_label << "] Failed to open " << label << " at " << path << ".\n";
    return false;
  }

  try {
    input >> json_payload;
  } catch (const std::exception& error) {
    std::cerr << "[" << context_label << "] Failed to parse " << label << ": "
              << error.what() << ".\n";
    return false;
  }

  return true;
}

int layer_to_segment(int layer_index) {
  // Map the 10 layers onto the 3/3/4 longitudinal segmentation.
  if (layer_index < 3) {
    return 0;
  }
  if (layer_index < 6) {
    return 1;
  }
  return 2;
}

}  // namespace

void performance(const char* events_path_cstr, const char* meta_path_cstr = "",
                 const char* calibration_path_cstr = "", const char* out_path_cstr = "") {
  // ============================================================================
  // Ensure I/O
  // ============================================================================
  if (!events_path_cstr || std::string(events_path_cstr).empty()) {
    std::cerr << "[performance] events.root path is required.\n";
    return;
  }

  const std::string events_path(events_path_cstr);
  const std::string meta_path = string_arg_or_default(meta_path_cstr, sibling_path(events_path, "meta.json"));
  const std::string calibration_path = string_arg_or_default(calibration_path_cstr, sibling_path(events_path, "calibration.json"));
  const std::string out_path = string_arg_or_default(out_path_cstr, sibling_path(events_path, "performance.json"));

  nlohmann::json meta_json;
  if (!load_json_file(meta_path, "meta.json", meta_json, "performance")) {
    return;
  }

  nlohmann::json calibration_json;
  if (!load_json_file(calibration_path, "calibration.json", calibration_json, "performance")) {
    return;
  }

  if (!meta_json.contains("geometry_id") || !meta_json.contains("gun_particle") ||
      !calibration_json.contains("thresholds")) {
    std::cerr << "[performance] Missing required metadata or calibration fields.\n";
    return;
  }
  if (calibration_json["thresholds"].size() != 3) {
    std::cerr << "[performance] calibration.json thresholds must have 3 entries.\n";
    return;
  }

  std::array<double, 3> thresholds {};
  // Load one threshold per longitudinal segment.
  for (std::size_t segment_index = 0; segment_index < thresholds.size(); ++segment_index) {
    thresholds[segment_index] = calibration_json["thresholds"][segment_index].get<double>();
  }
  TFile input_file(events_path.c_str(), "READ");
  if (input_file.IsZombie()) {
    std::cerr << "[performance] Failed to open " << events_path << ".\n";
    return;
  }

  TTree* tree = nullptr;
  input_file.GetObject("events", tree);
  if (!tree) {
    std::cerr << "[performance] Tree 'events' not found in " << events_path << ".\n";
    return;
  }

  // Ensure the processed tree contains the per-layer cell-energy branches.
  for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    const std::string branch_name = "layer_" + std::to_string(layer_index) + "_cell_E";
    if (tree->GetBranch(branch_name.c_str()) == nullptr) {
      std::cerr << "[performance] Branch '" << branch_name << "' not found in "
                << events_path << ".\n";
      return;
    }
  }

  float mc_E = 0.0F;
  // ROOT owns the vectors behind the branch addresses.
  std::array<std::vector<float>*, kLayerCount> layer_cell_E {};
  tree->SetBranchAddress("mc_E", &mc_E);
  for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    const std::string branch_name = "layer_" + std::to_string(layer_index) + "_cell_E";
    tree->SetBranchAddress(branch_name.c_str(), &layer_cell_E[static_cast<std::size_t>(layer_index)]);
  }

  PerformanceStats stats;
  const Long64_t entry_count = tree->GetEntries();
  // Count the binary efficiency and store the tile/layer multiplicities for each valid event.
  for (Long64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    tree->GetEntry(entry_index);

    // Require a recorded positive generator energy.
    const double mc_energy_GeV = static_cast<double>(mc_E);
    if (mc_energy_GeV <= 0.0) {
      continue;
    }

    stats.valid_event_count++;

    int fired_cell_count = 0;
    int fired_layer_count = 0;
    // Count fired cells and whether each layer has at least one fired cell.
    for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
      const int segment_index = layer_to_segment(layer_index);
      const auto* cell_energies = layer_cell_E[static_cast<std::size_t>(layer_index)];
      if (!cell_energies) {
        continue;
      }
      bool layer_has_fired_cell = false;
      for (const float cell_energy : *cell_energies) {
        if (static_cast<double>(cell_energy) >= thresholds[static_cast<std::size_t>(segment_index)]) {
          fired_cell_count++;
          layer_has_fired_cell = true;
        }
      }
      if (layer_has_fired_cell) {
        fired_layer_count++;
      }
    }

    const double fired_tile_count = static_cast<double>(fired_cell_count);
    const double fired_layer_total = static_cast<double>(fired_layer_count);
    stats.tile_count_sum += fired_tile_count;
    stats.tile_count_sum_squares += fired_tile_count * fired_tile_count;
    stats.layer_count_sum += fired_layer_total;
    stats.layer_count_sum_squares += fired_layer_total * fired_layer_total;

    if (fired_tile_count >= 1.0) {
      stats.detected_event_count++;
    }
  }

  nlohmann::json output;
  // Store raw counts first so downstream scripts can recompute derived quantities.
  output["valid_event_count"] = stats.valid_event_count;
  output["detected_event_count"] = stats.detected_event_count;

  if (stats.valid_event_count > 0) {
    // Compute event-averaged efficiency and activity moments over valid events.
    const double valid_event_count = static_cast<double>(stats.valid_event_count);
    const double detection_efficiency = static_cast<double>(stats.detected_event_count) / valid_event_count;
    const double tiles_mean = stats.tile_count_sum / valid_event_count;
    const double tiles_variance = std::max(0.0, (stats.tile_count_sum_squares / valid_event_count) - (tiles_mean * tiles_mean));
    const double layers_mean = stats.layer_count_sum / valid_event_count;
    const double layers_variance = std::max(0.0, (stats.layer_count_sum_squares / valid_event_count) - (layers_mean * layers_mean));
    output["detection_efficiency"] = detection_efficiency;
    output["tiles_mean"] = tiles_mean;
    output["tiles_std"] = std::sqrt(tiles_variance);
    output["layers_mean"] = layers_mean;
    output["layers_std"] = std::sqrt(layers_variance);
  } else {
    output["detection_efficiency"] = nullptr;
    output["tiles_mean"] = nullptr;
    output["tiles_std"] = nullptr;
    output["layers_mean"] = nullptr;
    output["layers_std"] = nullptr;
  }

  std::ofstream output_file(out_path);
  if (!output_file) {
    std::cerr << "[performance] Failed to open " << out_path << " for writing.\n";
    return;
  }
  output_file << output.dump(2) << '\n';
  std::cout << "[performance] Wrote " << out_path << ".\n";
}

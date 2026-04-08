/*
Particle performance summary.
*/

#include <TFile.h>
#include <TTree.h>

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

namespace {

constexpr int kLayerCount = 10;

struct PerformanceStats {
  long long valid_event_count = 0;
  long long detected_event_count = 0;
};

struct BinomialInterval {
  double mean = 0.0;
  double error_low = 0.0;
  double error_high = 0.0;
};

std::string sibling_path(const std::string& path, const std::string& basename) {
  const auto separator_index = path.find_last_of("/\\");
  if (separator_index == std::string::npos) {
    return basename;
  }
  return path.substr(0, separator_index + 1) + basename;
}

std::string string_arg_or_default(const char* value, const std::string& fallback) {
  if (value && std::string(value).size()) {
    return std::string(value);
  }
  return fallback;
}

bool load_json_file(const std::string& path,
                    const char* label,
                    nlohmann::json& json_payload,
                    const char* context_label) {
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

BinomialInterval wilson_interval(long long successes, long long trials, double z_value) {
  BinomialInterval interval;
  if (trials <= 0) {
    return interval;
  }
  const double trial_count = static_cast<double>(trials);
  const double fraction = static_cast<double>(successes) / trial_count;
  const double z_squared = z_value * z_value;
  const double denominator = 1.0 + z_squared / trial_count;
  const double center = (fraction + z_squared / (2.0 * trial_count)) / denominator;
  const double margin =
      (z_value / denominator) * std::sqrt((fraction * (1.0 - fraction) / trial_count) +
          (z_squared / (4.0 * trial_count * trial_count)));
  const double lower = std::max(0.0, center - margin);
  const double upper = std::min(1.0, center + margin);
  interval.mean = fraction;
  interval.error_low = std::max(0.0, fraction - lower);
  interval.error_high = std::max(0.0, upper - fraction);
  return interval;
}

int layer_to_segment(int layer_index) {
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
      !meta_json.contains("total_energy_GeV") || !calibration_json.contains("thresholds")) {
    std::cerr << "[performance] Missing required metadata or calibration fields.\n";
    return;
  }
  if (calibration_json["thresholds"].size() != 3) {
    std::cerr << "[performance] calibration.json thresholds must have 3 entries.\n";
    return;
  }

  const std::string geometry_id = meta_json["geometry_id"].get<std::string>();
  const std::string gun_particle = meta_json["gun_particle"].get<std::string>();
  const double total_energy_GeV = meta_json["total_energy_GeV"].get<double>();
  std::array<double, 3> thresholds {};
  for (std::size_t segment_index = 0; segment_index < thresholds.size(); ++segment_index) {
    thresholds[segment_index] = calibration_json["thresholds"][segment_index].get<double>();
  }

  const double wilson_z = 1.0;  // 1-sigma Wilson interval for the efficiency error.

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

  // Parse layers.
  for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    const std::string branch_name = "layer_" + std::to_string(layer_index) + "_E";
    tree->GetBranch(branch_name.c_str());
  }

  float mc_E = 0.0F;
  std::array<float, kLayerCount> layer_E {};
  tree->SetBranchAddress("mc_E", &mc_E);
  for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    const std::string branch_name = "layer_" + std::to_string(layer_index) + "_E";
    tree->SetBranchAddress(branch_name.c_str(), &layer_E[static_cast<std::size_t>(layer_index)]);
  }

  PerformanceStats stats;
  const Long64_t entry_count = tree->GetEntries();
  // Count the events with at least three layers above their segment thresholds.
  for (Long64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    tree->GetEntry(entry_index);

    // Require valid energy
    const double mc_energy_GeV = static_cast<double>(mc_E);
    if (mc_energy_GeV <= 0.0 || total_energy_GeV <= 0.0) {
      continue;
    }

    stats.valid_event_count++;
    
    int passing_layer_count = 0;
    for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
      const int segment_index = layer_to_segment(layer_index);
      if (static_cast<double>(layer_E[static_cast<std::size_t>(layer_index)]) >=
          thresholds[static_cast<std::size_t>(segment_index)]) {
        passing_layer_count++;
      }
    }

    if (passing_layer_count >= 3) {
      stats.detected_event_count++;
    }
  }

  nlohmann::json output;
  output["geometry_id"] = geometry_id;
  output["gun_particle"] = gun_particle;
  if (meta_json.contains("kinetic_energy_GeV")) {
    output["kinetic_energy_GeV"] = meta_json["kinetic_energy_GeV"];
  }
  output["total_energy_GeV"] = total_energy_GeV;
  output["valid_event_count"] = stats.valid_event_count;
  output["detected_event_count"] = stats.detected_event_count;

  const BinomialInterval efficiency_interval =
      wilson_interval(stats.detected_event_count, stats.valid_event_count, wilson_z);
  if (stats.valid_event_count > 0) {
    output["detection_efficiency"] = efficiency_interval.mean;
    output["eff_lo"] = efficiency_interval.error_low;
    output["eff_hi"] = efficiency_interval.error_high;
  } else {
    output["detection_efficiency"] = nullptr;
    output["eff_lo"] = nullptr;
    output["eff_hi"] = nullptr;
  }

  std::ofstream output_file(out_path);
  if (!output_file) {
    std::cerr << "[performance] Failed to open " << out_path << " for writing.\n";
    return;
  }
  output_file << output.dump(2) << '\n';
  std::cout << "[performance] Wrote " << out_path << ".\n";
}

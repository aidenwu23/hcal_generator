/*  
Neutron performance summary.
*/

#include <TFile.h>
#include <TTree.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct PerformanceStats {
  long long valid_event_count = 0;
  long long detected_event_count = 0;
  double reconstructed_energy_sum_GeV = 0.0;
  double reconstructed_energy_sum_sq_GeV = 0.0;
  // Percent of neutron interactions that start in either of the 3 segments.
  long long start_in_seg1_count = 0;
  long long start_in_seg2_count = 0;
  long long start_in_seg3_count = 0;
  std::vector<int> start_layers;
};

struct BinomialInterval {
  double mean = 0.0;
  double error_low = 0.0;
  double error_high = 0.0;
};

struct SegmentBoundaries {
  int seg1_layers = 0;
  int seg2_layers = 0;
  int seg3_layers = 0;
  int total_layers = 0;
};

std::string sibling_path(const std::string& path, const std::string& basename) {
  const auto separator_index = path.find_last_of("/\\");
  if (separator_index == std::string::npos) {
    return basename;
  }
  return path.substr(0, separator_index + 1) + basename;
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
      (z_value / denominator) * std::sqrt((fraction * (1.0 - fraction) / trial_count) + (z_squared / (4.0 * trial_count * trial_count)));
  const double lower = std::max(0.0, center - margin);
  const double upper = std::min(1.0, center + margin);
  interval.mean = fraction;
  interval.error_low = std::max(0.0, fraction - lower);
  interval.error_high = std::max(0.0, upper - fraction);
  return interval;
}

// Resolve paths.
std::string project_root_from_macro() {
  std::filesystem::path macro_path(__FILE__);
  return macro_path.parent_path().parent_path().parent_path().string();
}

std::string geometry_json_path(const std::string& geometry_id) {
  const std::filesystem::path project_root(project_root_from_macro());
  return (project_root / "geometries" / "generated" / geometry_id / "geometry.json").string();
}

// Safely read the integer from a json field.
bool read_required_int(const nlohmann::json& payload, const char* key, int& value) {
  if (!payload.contains(key) || !payload[key].is_number_integer()) {
    return false;
  }
  value = payload[key].get<int>();
  return true;
}

// Load the number of layers per segment based on the geometry ID.
bool load_segment_boundaries(const std::string& geometry_id, SegmentBoundaries& boundaries) {
  std::ifstream geometry_input(geometry_json_path(geometry_id));
  if (!geometry_input) {
    return false;
  }

  nlohmann::json geometry_json;
  geometry_input >> geometry_json;

  if (!read_required_int(geometry_json, "seg1_layers", boundaries.seg1_layers) ||
      !read_required_int(geometry_json, "seg2_layers", boundaries.seg2_layers) ||
      !read_required_int(geometry_json, "seg3_layers", boundaries.seg3_layers)) {
    return false;
  }

  boundaries.total_layers = boundaries.seg1_layers + boundaries.seg2_layers + boundaries.seg3_layers;
  return boundaries.seg1_layers > 0 && boundaries.seg2_layers > 0 && boundaries.seg3_layers > 0;
}

// Find the median start layer.
double median_layer_index(std::vector<int> start_layers) {
  if (start_layers.empty()) {
    return -1.0;
  }
  std::sort(start_layers.begin(), start_layers.end());
  const std::size_t middle = start_layers.size() / 2;
  if (start_layers.size() % 2 == 1) {
    return static_cast<double>(start_layers[middle]);
  }
  return 0.5 * static_cast<double>(start_layers[middle - 1] + start_layers[middle]);
}

}  // namespace

void performance(const char* events_path_cstr,
                 const char* meta_path_cstr = "",
                 const char* out_path_cstr = "") {
  if (!events_path_cstr || std::string(events_path_cstr).empty()) {
    std::cerr << "[performance] events.root path is required.\n";
    return;
  }

  const std::string events_path(events_path_cstr);
  const std::string meta_path =
      (meta_path_cstr && std::string(meta_path_cstr).size())
          ? std::string(meta_path_cstr)
          : sibling_path(events_path, "meta.json");
  const std::string out_path =
      (out_path_cstr && std::string(out_path_cstr).size())
          ? std::string(out_path_cstr)
          : sibling_path(events_path, "performance.json");

  std::ifstream meta_input(meta_path);
  if (!meta_input) {
    std::cerr << "[performance] Failed to open meta.json at " << meta_path << ".\n";
    return;
  }

  nlohmann::json meta_json;
  try {
    meta_input >> meta_json;
  } catch (const std::exception& error) {
    std::cerr << "[performance] Failed to parse meta.json: " << error.what() << ".\n";
    return;
  }

  if (!meta_json.contains("geometry_id") || !meta_json["geometry_id"].is_string()) {
    std::cerr << "[performance] meta.json is missing geometry_id.\n";
    return;
  }
  if (!meta_json.contains("gun_energy_GeV") || !meta_json["gun_energy_GeV"].is_number()) {
    std::cerr << "[performance] meta.json is missing gun_energy_GeV.\n";
    return;
  }
  if (!meta_json.contains("detect_threshold_GeV") || !meta_json["detect_threshold_GeV"].is_number()) {
    std::cerr << "[performance] meta.json is missing detect_threshold_GeV.\n";
    return;
  }

  const std::string geometry_id = meta_json["geometry_id"].get<std::string>();
  const double gun_energy_GeV = meta_json["gun_energy_GeV"].get<double>();
  const double detect_threshold_GeV = meta_json["detect_threshold_GeV"].get<double>();
  const double mc_rel_diff_limit = 0.1;
  const double wilson_z = 1.0;
  const std::string start_definition = "first_active_layer_above_threshold";

  SegmentBoundaries segment_boundaries;
  if (!load_segment_boundaries(geometry_id, segment_boundaries)) {
    std::cerr << "[performance] Failed to resolve segment boundaries for geometry "
              << geometry_id << ".\n";
    return;
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
  if (tree->GetBranch("mc_E") == nullptr) {
    std::cerr << "[performance] Branch mc_E is required in " << events_path << ".\n";
    return;
  }
  if (tree->GetBranch("rec_E") == nullptr) {
    std::cerr << "[performance] Branch rec_E is required in " << events_path << ".\n";
    return;
  }
  if (tree->GetBranch("sim_E") == nullptr) {
    std::cerr << "[performance] Branch sim_E is required in " << events_path << ".\n";
    return;
  }
  if (tree->GetBranch("start_layer") == nullptr) {
    std::cerr << "[performance] Branch start_layer is required in " << events_path << ".\n";
    return;
  }

  float mc_E = 0.0F;
  float rec_E = 0.0F;
  float sim_E = 0.0F;
  int start_layer = -1;
  std::string* category_ptr = nullptr;

  tree->SetBranchAddress("mc_E", &mc_E);
  tree->SetBranchAddress("rec_E", &rec_E);
  tree->SetBranchAddress("sim_E", &sim_E);
  tree->SetBranchAddress("start_layer", &start_layer);
  if (tree->GetBranch("category") != nullptr) {
    tree->SetBranchAddress("category", &category_ptr);
  }

  PerformanceStats stats;
  const Long64_t entry_count = tree->GetEntries();
  for (Long64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    tree->GetEntry(entry_index);

    const std::string category =
        (category_ptr != nullptr) ? *category_ptr : std::string();
    if (!category.empty() && category != "events") {
      continue;
    }

    const double mc_energy_GeV = static_cast<double>(mc_E);
    if (mc_energy_GeV <= 0.0 || gun_energy_GeV <= 0.0) {
      continue;
    }

    const double relative_difference = std::abs(mc_energy_GeV - gun_energy_GeV) / gun_energy_GeV;
    if (relative_difference > mc_rel_diff_limit) {
      continue;
    }

    stats.valid_event_count++;
    stats.reconstructed_energy_sum_GeV += static_cast<double>(rec_E);
    stats.reconstructed_energy_sum_sq_GeV += static_cast<double>(rec_E) * static_cast<double>(rec_E);
    if (static_cast<double>(sim_E) >= detect_threshold_GeV) {
      stats.detected_event_count++;
    }

    if (start_layer >= 0) {
      if (start_layer >= segment_boundaries.total_layers) {
        std::cerr << "[performance] start_layer out of range: " << start_layer << ".\n";
        return;
      }
      stats.start_layers.push_back(start_layer);
      // If interaction start layer < layers in 1st segment, its within the first segment.
      if (start_layer < segment_boundaries.seg1_layers) {
        stats.start_in_seg1_count++;
      
      // If it's greater than the first layer, but smaller than 1st + 2nd layer, its in the second.
      } else if (start_layer < segment_boundaries.seg1_layers + segment_boundaries.seg2_layers) {
        stats.start_in_seg2_count++;
      
      // Third layer.
      } else {
        stats.start_in_seg3_count++;
      }
    }
  }

  nlohmann::json output;
  output["geometry_id"] = geometry_id;
  output["gun_energy_GeV"] = gun_energy_GeV;

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

  if (stats.valid_event_count > 1 && gun_energy_GeV > 0.0) {
    const double mean_reconstructed_energy_GeV =
        stats.reconstructed_energy_sum_GeV / static_cast<double>(stats.valid_event_count);
    double variance =
        (stats.reconstructed_energy_sum_sq_GeV -
         static_cast<double>(stats.valid_event_count) * mean_reconstructed_energy_GeV * mean_reconstructed_energy_GeV) /
        static_cast<double>(stats.valid_event_count - 1);
    if (variance < 0.0) {
      variance = 0.0;
    }
    output["energy_resolution"] = std::sqrt(variance) / gun_energy_GeV;
  } else {
    output["energy_resolution"] = nullptr;
  }

  output["start_definition"] = start_definition;
  if (!stats.start_layers.empty()) {
    output["start_layer_median"] = median_layer_index(stats.start_layers);
  } else {
    output["start_layer_median"] = nullptr;
  }

  if (stats.valid_event_count > 0) {
    const double valid_event_count = static_cast<double>(stats.valid_event_count);
    output["frac_seg1"] = static_cast<double>(stats.start_in_seg1_count) / valid_event_count;
    output["frac_seg2"] = static_cast<double>(stats.start_in_seg2_count) / valid_event_count;
    output["frac_seg3"] = static_cast<double>(stats.start_in_seg3_count) / valid_event_count;
  } else {
    output["frac_seg1"] = nullptr;
    output["frac_seg2"] = nullptr;
    output["frac_seg3"] = nullptr;
  }

  std::ofstream output_file(out_path);
  if (!output_file) {
    std::cerr << "[performance] Failed to open " << out_path << " for writing.\n";
    return;
  }
  output_file << output.dump(2) << '\n';
  std::cout << "[performance] Wrote " << out_path << ".\n";
}

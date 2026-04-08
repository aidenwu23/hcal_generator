/*
Measure the reference MIP scale from a processed muon events.root file.
Example:
root -l -b -q 'simulation/calibration/calibrate_MIP.C("data/processed/04e3fdfb/run_mu_ctrl/events.root","data/processed/04e3fdfb/run_mu_ctrl/calibration.json",0.5)'
*/

#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TTree.h>

#include <algorithm>
#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kLayerCount = 10;
constexpr int kSegmentCount = 3;
constexpr int kEnergyBinCount = 400;
constexpr double kRangeQuantile = 0.995;

int layer_to_segment(int layer_index) {
  if (layer_index < 3) {
    return 0;
  }
  if (layer_index < 6) {
    return 1;
  }
  return 2;
}

double peak_bin_center(const TH1D& histogram) {
  if (histogram.GetEntries() <= 0.0) {
    return 0.0;
  }
  return histogram.GetBinCenter(histogram.GetMaximumBin());
}

double quantile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }

  const double clamped_fraction = std::clamp(fraction, 0.0, 1.0);
  const std::size_t index = static_cast<std::size_t>(
      clamped_fraction * static_cast<double>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
  return values[index];
}

double histogram_xmax(const std::vector<double>& values) {
  if (values.empty()) {
    return 1.0;
  }

  const double median = quantile(values, 0.50);
  const double upper_quantile = quantile(values, kRangeQuantile);
  const double scaled_quantile = upper_quantile > 0.0 ? upper_quantile * 1.15 : 0.0;
  const double scaled_median = median > 0.0 ? median * 6.0 : 0.0;
  const double x_max = std::max({scaled_quantile, scaled_median, 1e-6});
  return x_max;
}

double estimate_mpv(TH1D& histogram) {
  const double fallback_mpv = peak_bin_center(histogram);
  if (!(fallback_mpv > 0.0)) {
    return 0.0;
  }

  TF1 landau_fit("landau_fit", "landau", 0.0, histogram.GetXaxis()->GetXmax());
  landau_fit.SetParameters(histogram.GetMaximum(), fallback_mpv, std::max(1e-6, fallback_mpv * 0.25));
  const int fit_status = histogram.Fit(&landau_fit, "Q0");
  const double fitted_mpv = landau_fit.GetParameter(1);
  if (fit_status == 0 && std::isfinite(fitted_mpv) && fitted_mpv >= 0.0) {
    return fitted_mpv;
  }
  return fallback_mpv;
}

}  // namespace

void calibrate_MIP(const char* events_path_cstr,
                   const char* out_json_path_cstr,
                   double alpha = 0.5) {
  // Validate the runtime inputs before reading files.
  if (!events_path_cstr || std::string(events_path_cstr).empty()) {
    std::cerr << "[calibrate_MIP] events.root path is required.\n";
    return;
  }
  if (!out_json_path_cstr || std::string(out_json_path_cstr).empty()) {
    std::cerr << "[calibrate_MIP] Output json path is required.\n";
    return;
  }
  if (alpha < 0.0 || !std::isfinite(alpha)) {
    std::cerr << "[calibrate_MIP] alpha must be finite and non-negative.\n";
    return;
  }

  const std::string events_path(events_path_cstr);
  const std::string out_json_path(out_json_path_cstr);

  TFile input_file(events_path.c_str(), "READ");
  if (input_file.IsZombie()) {
    std::cerr << "[calibrate_MIP] Failed to open " << events_path << ".\n";
    return;
  }

  TTree* tree = nullptr;
  input_file.GetObject("events", tree);
  if (!tree) {
    std::cerr << "[calibrate_MIP] Tree 'events' not found in " << events_path << ".\n";
    return;
  }

  std::array<float, kLayerCount> layer_energy {};
  for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    const std::string branch_name = "layer_" + std::to_string(layer_index) + "_E";
    if (tree->GetBranch(branch_name.c_str()) == nullptr) {
      std::cerr << "[calibrate_MIP] Branch '" << branch_name << "' not found in "
                << events_path << ".\n";
      return;
    }
    tree->SetBranchAddress(branch_name.c_str(), &layer_energy[static_cast<std::size_t>(layer_index)]);
  }

  std::array<std::vector<double>, kSegmentCount> segment_values;
  const Long64_t entry_count = tree->GetEntries();
  for (std::size_t segment_index = 0; segment_index < segment_values.size(); ++segment_index) {
    const int layer_count = segment_index < 2 ? 3 : 4;
    segment_values[segment_index].reserve(static_cast<std::size_t>(entry_count) * layer_count);
  }

  for (Long64_t event_index = 0; event_index < entry_count; ++event_index) {
    tree->GetEntry(event_index);

    // Collect one value per layer so the histogram range can ignore rare outliers.
    for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
      const double value = static_cast<double>(layer_energy[static_cast<std::size_t>(layer_index)]);
      if (!std::isfinite(value) || value < 0.0) {
        continue;
      }
      const int segment_index = layer_to_segment(layer_index);
      segment_values[static_cast<std::size_t>(segment_index)].push_back(value);
    }
  }

  std::array<double, kSegmentCount> segment_xmax {};
  for (std::size_t segment_index = 0; segment_index < segment_values.size(); ++segment_index) {
    segment_xmax[segment_index] = histogram_xmax(segment_values[segment_index]);
  }

  std::array<TH1D, kSegmentCount> segment_histograms = {
      TH1D("h_seg1_mip", "Segment 1 muon layer deposits;Layer energy [GeV];Layer count", kEnergyBinCount, 0.0, segment_xmax[0]),
      TH1D("h_seg2_mip", "Segment 2 muon layer deposits;Layer energy [GeV];Layer count", kEnergyBinCount, 0.0, segment_xmax[1]),
      TH1D("h_seg3_mip", "Segment 3 muon layer deposits;Layer energy [GeV];Layer count", kEnergyBinCount, 0.0, segment_xmax[2]),
  };
  for (TH1D& histogram : segment_histograms) {
    histogram.SetDirectory(nullptr);
    histogram.SetStats(false);
  }

  for (Long64_t event_index = 0; event_index < entry_count; ++event_index) {
    tree->GetEntry(event_index);

    // Fill one segment histogram with all same-segment layer deposits across the muon sample.
    for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
      const double value = static_cast<double>(layer_energy[static_cast<std::size_t>(layer_index)]);
      if (!std::isfinite(value) || value < 0.0) {
        continue;
      }
      const int segment_index = layer_to_segment(layer_index);
      segment_histograms[static_cast<std::size_t>(segment_index)].Fill(value);
    }
  }

  std::vector<double> mpvs;
  std::vector<double> thresholds;
  mpvs.reserve(kSegmentCount);
  thresholds.reserve(kSegmentCount);
  for (TH1D& histogram : segment_histograms) {
    const double mpv = estimate_mpv(histogram);
    mpvs.push_back(mpv);
    thresholds.push_back(alpha * mpv);
  }

  nlohmann::json output_json;
  output_json["alpha"] = alpha;
  output_json["mpvs"] = mpvs;
  output_json["thresholds"] = thresholds;

  std::ofstream output_file(out_json_path);
  if (!output_file) {
    std::cerr << "[calibrate_MIP] Failed to open " << out_json_path << " for writing.\n";
    return;
  }
  output_file << output_json.dump(2) << '\n';
  std::cout << "[calibrate_MIP] Wrote " << out_json_path << ".\n";
}

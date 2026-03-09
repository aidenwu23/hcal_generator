/*
Measure how much visible HCAL energy lands in the outer transverse edge cells.

Example:
root -l -b -q 'analysis/lateral_leakage.C("data/raw/81c3da7d/rune3896ec0d8.edm4hep.root")'
*/

#include <TFormula.h>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include "edm4hep/SimCalorimeterHitCollection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

constexpr int kXBitOffset = 20;
constexpr int kYBitOffset = 36;
constexpr std::uint64_t kSignedFieldMask = 0xFFFF;  // 16-bit signed cell-index field.

std::string project_root_from_macro() {
  std::filesystem::path macro_path(__FILE__);
  return macro_path.parent_path().parent_path().string();
}

std::string derive_geometry_id_from_raw_path(const std::string& raw_path) {
  return std::filesystem::path(raw_path).parent_path().filename().string();
}

// Build the generated geometry XML path from the raw file's geometry directory.
std::string default_geometry_xml_path_from_raw(const std::string& raw_path) {
  const std::filesystem::path project_root(project_root_from_macro());
  const std::string geometry_id = derive_geometry_id_from_raw_path(raw_path);
  return (project_root / "geometries" / "generated" / geometry_id / "geometry.xml").string();
}

void replace_all(std::string& text, const std::string& from, const std::string& to) {
  if (from.empty()) {
    return;
  }

  std::size_t start_index = 0;
  // Replace every instance of the unit token in the expression text.
  while ((start_index = text.find(from, start_index)) != std::string::npos) {
    text.replace(start_index, from.size(), to);
    start_index += to.size();
  }
}

double eval_length_mm(const std::string& value_text) {
  std::string expression = value_text;
  replace_all(expression, "mm", "1.0");
  replace_all(expression, "cm", "10.0");
  replace_all(expression, "m", "1000.0");

  static int formula_counter = 0;
  TFormula formula(
      ("length_formula_" + std::to_string(formula_counter++)).c_str(),
      expression.c_str());
  return formula.Eval(0.0);
}

std::string read_xml_value(
    const std::string& xml_text,
    const std::string& tag_name,
    const std::string& attribute_name) {
  const std::regex pattern(
      "<" + tag_name + "[^>]*name=\"" + attribute_name + "\"[^>]*value=\"([^\"]+)\"");
  std::smatch match;
  if (!std::regex_search(xml_text, match, pattern) || match.size() < 2) {
    throw std::runtime_error("Missing XML value for " + tag_name + ":" + attribute_name);
  }
  return match[1].str();
}

std::string read_text_file(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open " + path);
  }

  return std::string(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

int decode_signed_field(std::uint64_t cell_id, int bit_offset) {
  int value = static_cast<int>((cell_id >> bit_offset) & kSignedFieldMask);
  if ((value & 0x8000) != 0) {
    value -= 0x10000;  // Convert the stored 16-bit value back to a signed index.
  }
  return value;
}

std::string pick_category(podio::ROOTReader& reader) {
  std::string category = "events";
  try {
    auto categories = reader.getAvailableCategories();
    if (!categories.empty() &&
        std::find(categories.begin(), categories.end(), "events") == categories.end()) {
      category = categories.front();
    }
  } catch (...) {
  }
  return category;
}

bool is_edge_cell(int x_index, int y_index, int edge_threshold_x, int edge_threshold_y) {
  return std::abs(x_index) >= edge_threshold_x || std::abs(y_index) >= edge_threshold_y;
}

}  // namespace

// Read the geometry, scan all HCAL hits, and print the edge-energy fraction.
void lateral_leakage(const char* raw_events_path_cstr,
                     const char* geometry_xml_cstr = "",
                     int edge_width_cells = 1) {
  if (!raw_events_path_cstr || std::string(raw_events_path_cstr).empty()) {
    std::cerr << "[lateral_leakage] Raw EDM4hep path is required.\n";
    return;
  }
  if (edge_width_cells <= 0) {
    std::cerr << "[lateral_leakage] edge_width_cells must be positive.\n";
    return;
  }

  const std::string raw_events_path(raw_events_path_cstr);
  const std::string geometry_xml_path =
      (geometry_xml_cstr && std::string(geometry_xml_cstr).size())
          ? std::string(geometry_xml_cstr)
          : default_geometry_xml_path_from_raw(raw_events_path);

  double half_width_x_mm = 0.0;
  double half_width_y_mm = 0.0;
  double cell_x_mm = 0.0;
  double cell_y_mm = 0.0;

  // Read the detector half-width and cell size from the geometry XML.
  try {
    const std::string geometry_xml = read_text_file(geometry_xml_path);
    half_width_x_mm = eval_length_mm(read_xml_value(geometry_xml, "parameter", "dx"));
    half_width_y_mm = eval_length_mm(read_xml_value(geometry_xml, "parameter", "dy"));
    cell_x_mm = eval_length_mm(read_xml_value(geometry_xml, "constant", "cell_x"));
    cell_y_mm = eval_length_mm(read_xml_value(geometry_xml, "constant", "cell_y"));
  } catch (const std::exception& error) {
    std::cerr << "[lateral_leakage] Failed to read geometry XML: "
              << error.what() << ".\n";
    return;
  }

  if (!(half_width_x_mm > 0.0) || !(half_width_y_mm > 0.0) ||
      !(cell_x_mm > 0.0) || !(cell_y_mm > 0.0)) {
    std::cerr << "[lateral_leakage] Geometry dimensions and cell sizes must be positive.\n";
    return;
  }

  // Convert the detector half-width into the first edge-cell index on each axis.
  const int half_cell_count_x = static_cast<int>(std::round(half_width_x_mm / cell_x_mm));
  const int half_cell_count_y = static_cast<int>(std::round(half_width_y_mm / cell_y_mm));
  const int edge_threshold_x = std::max(0, half_cell_count_x - edge_width_cells);
  const int edge_threshold_y = std::max(0, half_cell_count_y - edge_width_cells);

  podio::ROOTReader reader;
  try {
    reader.openFile(raw_events_path);
  } catch (const std::exception& error) {
    std::cerr << "[lateral_leakage] Failed to open " << raw_events_path
              << ": " << error.what() << ".\n";
    return;
  }

  const std::string category = pick_category(reader);
  const std::size_t entry_count = reader.getEntries(category);
  const std::string collection_name = "HCal_Readout";

  double total_visible_energy_GeV = 0.0;
  double edge_visible_energy_GeV = 0.0;
  long long event_count = 0;
  long long hit_count = 0;
  long long edge_hit_count = 0;
  int observed_min_x = std::numeric_limits<int>::max();
  int observed_max_x = std::numeric_limits<int>::min();
  int observed_min_y = std::numeric_limits<int>::max();
  int observed_max_y = std::numeric_limits<int>::min();

  // Read each event frame and skip entries without the HCAL hit collection.
  for (std::size_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    auto frame_data = reader.readEntry(category, entry_index);
    podio::Frame frame(std::move(frame_data));

    const edm4hep::SimCalorimeterHitCollection* sim_collection = nullptr;
    try {
      sim_collection = &frame.get<edm4hep::SimCalorimeterHitCollection>(collection_name);
    } catch (...) {
      continue;
    }
    if (!sim_collection) {
      continue;
    }

    ++event_count;
    // Decode each hit position and update the total and edge-cell energy sums.
    for (const auto& hit : *sim_collection) {
      const double visible_energy_GeV = hit.getEnergy();
      const std::uint64_t cell_id = static_cast<std::uint64_t>(hit.getCellID());
      const int x_index = decode_signed_field(cell_id, kXBitOffset);
      const int y_index = decode_signed_field(cell_id, kYBitOffset);

      observed_min_x = std::min(observed_min_x, x_index);
      observed_max_x = std::max(observed_max_x, x_index);
      observed_min_y = std::min(observed_min_y, y_index);
      observed_max_y = std::max(observed_max_y, y_index);

      total_visible_energy_GeV += visible_energy_GeV;
      ++hit_count;
      if (is_edge_cell(x_index, y_index, edge_threshold_x, edge_threshold_y)) {
        edge_visible_energy_GeV += visible_energy_GeV;
        ++edge_hit_count;
      }
    }
  }

  if (!(total_visible_energy_GeV > 0.0)) {
    std::cerr << "[lateral_leakage] No visible HCAL energy found in " << raw_events_path << ".\n";
    return;
  }

  // Print the edge-energy fraction together with the geometry inputs behind it.
  const double outer_edge_visible_fraction = edge_visible_energy_GeV / total_visible_energy_GeV;

  std::cout << "[lateral_leakage] raw=" << raw_events_path << "\n";
  std::cout << "[lateral_leakage] geometry_xml=" << geometry_xml_path << "\n";
  std::cout << "[lateral_leakage] edge_width_cells=" << edge_width_cells << "\n";
  std::cout << "[lateral_leakage] geometry_half_width_mm=("
            << half_width_x_mm << ", " << half_width_y_mm << ")\n";
  std::cout << "[lateral_leakage] cell_size_mm=("
            << cell_x_mm << ", " << cell_y_mm << ")\n";
  std::cout << "[lateral_leakage] edge_threshold_indices=("
            << edge_threshold_x << ", " << edge_threshold_y << ")\n";
  std::cout << "[lateral_leakage] observed_index_range_x=["
            << observed_min_x << ", " << observed_max_x << "]"
            << " observed_index_range_y=["
            << observed_min_y << ", " << observed_max_y << "]\n";
  std::cout << "[lateral_leakage] events=" << event_count
            << " hits=" << hit_count
            << " edge_hits=" << edge_hit_count << "\n";
  std::cout << "[lateral_leakage] total_visible_E_GeV=" << total_visible_energy_GeV << "\n";
  std::cout << "[lateral_leakage] outer_edge_visible_E_GeV=" << edge_visible_energy_GeV << "\n";
  std::cout << "[lateral_leakage] outer_edge_visible_fraction="
            << outer_edge_visible_fraction
            << " (" << 100.0 * outer_edge_visible_fraction << "%)\n";
}

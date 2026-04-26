/*
Write one event tree with merged prompt cell energies for each layer.
Reference: https://lists.bnl.gov/sympa/arc/epic-backward-hcal-l/2025-02/msg00006/pdfbFDp1SD90k.pdf
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include "edm4hep/MCParticleCollection.h"
#include "edm4hep/SimCalorimeterHitCollection.h"

namespace {

constexpr int kLayerCount = 10;  // Number of nHCal layers in the compact template.
constexpr int kLayerBitOffset = 8;  // DD4hep cellID bit offset used for the layer field.
constexpr std::uint64_t kLayerMask = 0xFF;  // Eight-bit layer field mask.
constexpr double kWindowTimeMinNs = 0.0;  // Start of prompt integration window in ns.
constexpr double kWindowTimeMaxNs = 100.0;  // End of prompt integration window in ns.

int decode_layer_index(std::uint64_t cell_id) {
  // Extract the layer index from the encoded calorimeter cell ID.
  return static_cast<int>((cell_id >> kLayerBitOffset) & kLayerMask);
}

// Sum up all hit energies in a cell from t_0 to t_f, in nanoseconds.
double integrated_hit_energy_in_window(const edm4hep::SimCalorimeterHit& hit) {
  double integrated_energy = 0.0;
  const auto contributions = hit.getContributions();
  for (const auto& contribution : contributions) {
    const double time_ns = static_cast<double>(contribution.getTime());
    if (time_ns < kWindowTimeMinNs || time_ns > kWindowTimeMaxNs) {
      continue;
    }
    integrated_energy += static_cast<double>(contribution.getEnergy());
  }
  return integrated_energy;
}

}  // namespace

static bool hasArg(int argc, char** argv, const std::string& key) {
  // Check whether a command-line flag is present.
  for (int i = 1; i < argc; ++i) if (key == argv[i]) return true; return false;
}
static std::string getArg(int argc, char** argv, const std::string& key, const std::string& def="") {
  // Read the value following a command-line option.
  for (int i = 1; i < argc - 1; ++i) if (key == argv[i]) return std::string(argv[i+1]); return def;
}
static int getArgI(int argc, char** argv, const std::string& key, int def) {
  // Read an integer command-line option.
  const auto s = getArg(argc, argv, key, ""); return s.empty() ? def : std::stoi(s);
}
static double energy_from_mc(const edm4hep::MCParticle& p) {
  // Use stored MC energy when available, otherwise reconstruct it from mass and momentum.
  double E = p.getEnergy();
  if (E > 0) return E;
  const auto& q = p.getMomentum();
  const double p2 = q.x*q.x + q.y*q.y + q.z*q.z;
  const double m  = p.getMass();
  return std::sqrt(std::max(0.0, p2 + m*m));
}

static bool has_no_parents(const edm4hep::MCParticle& p) {
  // Primary particles have no parent links.
  const auto parents = p.getParents();
  return parents.begin() == parents.end();
}

static double vertex_radius2_from_mc(const edm4hep::MCParticle& p) {
  // Use squared radius for candidate tie-breaking without a square root.
  const auto& vertex = p.getVertex();
  return vertex.x * vertex.x + vertex.y * vertex.y + vertex.z * vertex.z;
}

static int mc_selection_priority(const edm4hep::MCParticle& p, int expectedPDG) {
  // Lower priority values are better truth-particle matches.
  const bool not_from_sim = !p.isCreatedInSimulation();
  const bool no_parents = has_no_parents(p);
  const bool matches_expected_pdg =
      expectedPDG != 0 && p.getPDG() != 0 && std::abs(p.getPDG()) == std::abs(expectedPDG);

  if (expectedPDG != 0) {
    if (matches_expected_pdg && not_from_sim && no_parents) return 0;
    if (matches_expected_pdg && not_from_sim) return 1;
    if (matches_expected_pdg) return 2;
  }
  if (not_from_sim && no_parents) return 3;
  if (not_from_sim) return 4;
  if (no_parents) return 5;
  return 6;
}

int main(int argc, char** argv) {
  if (argc < 2 || hasArg(argc, argv, "--help")) {
    std::cout <<
      "Usage: process <input.edm4hep.root> [--out out.root]\n"
      "                [--expected-pdg PDG]\n"
      "                [--debug]\n";
    return 0;
  }

  const bool debug     = hasArg(argc, argv, "--debug");
  const std::string inFile   = argv[1];
  const std::string outFile  = getArg(argc, argv, "--out", "events.root");
  const int expectedPDG      = getArgI(argc, argv, "--expected-pdg", 0);

  TFile* fout = TFile::Open(outFile.c_str(), "RECREATE");
  if (!fout || fout->IsZombie()) {
    std::cerr << "ERROR: Cannot open output file: " << outFile << std::endl;
    return 2;
  }

  float  t_mc_E=0;
  // Output vectors hold merged cell energies separately for each layer.
  std::array<std::vector<float>, kLayerCount> t_layer_cell_E {};

  TTree* events = new TTree("events", "per-event dataset");

  // Write one event row with one merged cell-energy vector per layer.
  events->Branch("mc_E",   &t_mc_E);
  for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    events->Branch(
        ("layer_" + std::to_string(layer_index) + "_cell_E").c_str(),
        &t_layer_cell_E[static_cast<size_t>(layer_index)]);
  }
  size_t nEvents=0, nWithMC=0, nWithSim=0;

  podio::ROOTReader reader;
  try { reader.openFile(inFile); }
  catch (std::exception& e) { std::cerr << "ERROR opening " << inFile << ": " << e.what() << std::endl; return 2; }

  std::string category = "events";
  // Prefer the standard events category, otherwise use the first available category.
  try {
    auto cats = reader.getAvailableCategories();
    if (!cats.empty()) {
      if (std::find(cats.begin(), cats.end(), "events") != cats.end()) category = "events"; else category = cats.front();
    }
    if (debug) {
      std::cerr << "[process] Categories: "; for (auto& c : cats) std::cerr << c << " "; std::cerr << " | chosen=" << category << "\n";
    }
  } catch (...) {}

  const size_t nEntries = reader.getEntries(category);
  for (size_t iev = 0; iev < nEntries; ++iev) {
    // Grab frame.
    auto frame_data = reader.readEntry(category, iev);
    podio::Frame frame(std::move(frame_data));

    if (debug && iev==0) {
      auto names = frame.getAvailableCollections();
      std::cerr << "[process] Frame[0] collections (" << names.size() << "): ";
      for (auto& n : names) std::cerr << n << " ";
      std::cerr << "\n";
    }
    ++nEvents;

    double mcE = 0.0;

    // Load the MC truth collection for this event.
    const edm4hep::MCParticleCollection* mcCollection = nullptr;
    try {
      mcCollection = &frame.get<edm4hep::MCParticleCollection>("MCParticles");
    } catch (...) {}

    // ==========================================================================
    // Pick the MC particle that best represents the injected particle.
    // ==========================================================================
    // Previously used to estimate sigma/E, currently serves no major purpose.

    if (mcCollection) {
      if (debug && nEvents==1) std::cerr << "[process] Using MC collection: MCParticles (size=" << mcCollection->size() << ")\n";
      
      // Prefer the injected beam particle over later shower products.
      std::optional<edm4hep::MCParticle> selected_candidate;
      int best_priority = std::numeric_limits<int>::max();
      double best_abs_time = std::numeric_limits<double>::infinity();
      double best_vertex_radius2 = std::numeric_limits<double>::infinity();
      double best_energy = -1.0;

      for (const auto& p : *mcCollection) {
        // Rank candidates by PDG match, origin, time, vertex, then energy.
        const double energy = energy_from_mc(p);
        const int priority = mc_selection_priority(p, expectedPDG);
        const double abs_time = std::abs(static_cast<double>(p.getTime()));
        const double vertex_radius2 = vertex_radius2_from_mc(p);

        bool better_candidate = false;
        if (!selected_candidate || priority < best_priority) {
          better_candidate = true;
        } else if (priority == best_priority) {
          if (abs_time < best_abs_time) {
            better_candidate = true;
          } else if (abs_time == best_abs_time) {
            if (vertex_radius2 < best_vertex_radius2) {
              better_candidate = true;
            } else if (vertex_radius2 == best_vertex_radius2 && energy > best_energy) {
              better_candidate = true;
            }
          }
        }

        if (better_candidate) {
          selected_candidate = p;
          best_priority = priority;
          best_abs_time = abs_time;
          best_vertex_radius2 = vertex_radius2;
          best_energy = energy;
        }
      }

      if (selected_candidate) {
        // Keep only the energy of the selected truth candidate.
        const auto& selected_particle = *selected_candidate;
        mcE = energy_from_mc(selected_particle);
      } else if (debug && expectedPDG != 0) {
        std::cerr << "[process] Warning: expected PDG " << expectedPDG
                  << " not found in MCParticles for event " << (nEvents - 1) << std::endl;
      }
    }
    if (mcE > 0.0) {
      ++nWithMC;
    }

    // ==========================================================================
    // Process cell-level info
    // ==========================================================================
    
    // Start each event with empty per-layer cell lists.
    for (auto& layer_cell_E : t_layer_cell_E) {
      layer_cell_E.clear();
    }

    // Load the calorimeter hit collection.
    const edm4hep::SimCalorimeterHitCollection* simCollection = nullptr;
    try {
      simCollection = &frame.get<edm4hep::SimCalorimeterHitCollection>("HCal_Readout");
    } catch (...) {}
    if (simCollection && debug && nEvents==1) std::cerr << "[process] Using Sim collection: HCal_Readout (size=" << simCollection->size() << ")\n";

    if (simCollection) {
      // Count events with at least one calorimeter hit.
      if (!simCollection->empty()) {
        ++nWithSim;
      }

      // Merge all hit contributions into one energy per physical cell.
      std::array<std::map<std::uint64_t, double>, kLayerCount> layer_cell_energy_by_id;
      for (const auto& hit : *simCollection) {
        const std::uint64_t cell_id = static_cast<std::uint64_t>(hit.getCellID());
        const int layer_index = decode_layer_index(cell_id);
        if (layer_index < 0 || layer_index >= kLayerCount) {
          continue;
        }
        const double integrated_energy = integrated_hit_energy_in_window(hit);
        if (integrated_energy <= 0.0) {
          continue;
        }
        layer_cell_energy_by_id[static_cast<std::size_t>(layer_index)][cell_id] += integrated_energy;
      }

      // Store one merged energy value for each physical cell in the layer.
      for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
        const auto& layer_cell_map = layer_cell_energy_by_id[static_cast<std::size_t>(layer_index)];
        auto& layer_cell_vector = t_layer_cell_E[static_cast<std::size_t>(layer_index)];
        layer_cell_vector.reserve(layer_cell_map.size());
        for (const auto& [cell_id, cell_energy] : layer_cell_map) {
          (void)cell_id;
          layer_cell_vector.push_back(static_cast<float>(cell_energy));
        }
      }
    }

    // Store the selected generator energy and merged cell energies for this event.
    t_mc_E = (float)mcE;
    events->Fill();
  }

  // Write the compact per-event tree used by later calibration and performance steps.
  fout->cd();
  events->Write();
  fout->Close();

  std::cout << "[process] Done. Events: " << nEvents
            << " | with MC: "  << nWithMC
            << " | with Sim: " << nWithSim;
  std::cout << "\nOutput: " << outFile << std::endl;
  return 0;
}

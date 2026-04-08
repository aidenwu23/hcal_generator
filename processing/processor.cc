// CLI:
//   process <input.edm4hep.root> [--out out.root]
//            [--expected-pdg PDG] [--debug]

#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>

// ROOT I/O
#include "TFile.h"
#include "TTree.h"

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include "edm4hep/MCParticleCollection.h"
#include "edm4hep/SimCalorimeterHitCollection.h"

namespace {

constexpr int kLayerCount = 10;
constexpr int kLayerBitOffset = 8;
constexpr std::uint64_t kLayerMask = 0xFF;

int decode_layer_index(std::uint64_t cell_id) {
  return static_cast<int>((cell_id >> kLayerBitOffset) & kLayerMask);
}

}  // namespace

static bool hasArg(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) if (key == argv[i]) return true; return false;
}
static std::string getArg(int argc, char** argv, const std::string& key, const std::string& def="") {
  for (int i = 1; i < argc - 1; ++i) if (key == argv[i]) return std::string(argv[i+1]); return def;
}
static int getArgI(int argc, char** argv, const std::string& key, int def) {
  const auto s = getArg(argc, argv, key, ""); return s.empty() ? def : std::stoi(s);
}
static double energy_from_mc(const edm4hep::MCParticle& p) {
  double E = p.getEnergy();
  if (E > 0) return E;
  const auto& q = p.getMomentum();
  const double p2 = q.x*q.x + q.y*q.y + q.z*q.z;
  const double m  = p.getMass();
  return std::sqrt(std::max(0.0, p2 + m*m));
}

static bool has_no_parents(const edm4hep::MCParticle& p) {
  const auto parents = p.getParents();
  return parents.begin() == parents.end();
}

static double vertex_radius2_from_mc(const edm4hep::MCParticle& p) {
  const auto& vertex = p.getVertex();
  return vertex.x * vertex.x + vertex.y * vertex.y + vertex.z * vertex.z;
}

static int mc_selection_priority(const edm4hep::MCParticle& p, int expectedPDG) {
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
  std::array<float, kLayerCount> t_layer_E {};

  TTree* events = new TTree("events", "per-event dataset");
  events->Branch("mc_E",   &t_mc_E);
  for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    events->Branch(
        ("layer_" + std::to_string(layer_index) + "_E").c_str(),
        &t_layer_E[static_cast<size_t>(layer_index)]);
  }
  size_t nEvents=0, nWithMC=0, nWithSim=0;

  podio::ROOTReader reader;
  try { reader.openFile(inFile); }
  catch (std::exception& e) { std::cerr << "ERROR opening " << inFile << ": " << e.what() << std::endl; return 2; }

  std::string category = "events";
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

    // Pick the MC particle that best represents the injected particle.
    if (mcCollection) {
      if (debug && nEvents==1) std::cerr << "[process] Using MC collection: MCParticles (size=" << mcCollection->size() << ")\n";
      
      // Prefer the injected beam particle over later shower products.
      std::optional<edm4hep::MCParticle> selected_candidate;
      int best_priority = std::numeric_limits<int>::max();
      double best_abs_time = std::numeric_limits<double>::infinity();
      double best_vertex_radius2 = std::numeric_limits<double>::infinity();
      double best_energy = -1.0;

      for (const auto& p : *mcCollection) {
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

    t_layer_E.fill(0.f);

    // Load the calorimeter hit collection.
    const edm4hep::SimCalorimeterHitCollection* simCollection = nullptr;
    try {
      simCollection = &frame.get<edm4hep::SimCalorimeterHitCollection>("HCal_Readout");
    } catch (...) {}
    if (simCollection && debug && nEvents==1) std::cerr << "[process] Using Sim collection: HCal_Readout (size=" << simCollection->size() << ")\n";

    if (simCollection) {
      // Increment non-empty counter
      if (!simCollection->empty()) {
        ++nWithSim;
      }

      // Sum up all hit energy per layer.
      for (const auto& hit : *simCollection) {
        const int layer_index = decode_layer_index(static_cast<std::uint64_t>(hit.getCellID()));
        if (layer_index < 0 || layer_index >= kLayerCount) {
          continue;
        }
        const double e = hit.getEnergy();
        t_layer_E[static_cast<size_t>(layer_index)] += static_cast<float>(e);
      }
    }

    t_mc_E = (float)mcE;
    events->Fill();
  }

  fout->cd();
  events->Write();
  fout->Close();

  std::cout << "[process] Done. Events: " << nEvents
            << " | with MC: "  << nWithMC
            << " | with Sim: " << nWithSim;
  std::cout << "\nOutput: " << outFile << std::endl;
  return 0;
}

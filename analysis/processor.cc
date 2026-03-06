// CLI:
//   process <input.edm4hep.root> [--out out.root]
//            [--sampling f] [--expected-pdg PDG]
//            [--start-threshold GeV] [--debug]

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>

// ROOT I/O
#include "TFile.h"
#include "TTree.h"

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include "edm4hep/MCParticleCollection.h"
#include "edm4hep/SimCalorimeterHitCollection.h"

namespace {

constexpr int kHcalLayerCount = 10;
constexpr int kLayerBitOffset = 8;
constexpr std::uint64_t kLayerMask = 0xFF;

int decode_hcal_layer(std::uint64_t cell_id) {
  return static_cast<int>((cell_id >> kLayerBitOffset) & kLayerMask);
}

}  // namespace

static bool hasArg(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) if (key == argv[i]) return true; return false;
}
static std::string getArg(int argc, char** argv, const std::string& key, const std::string& def="") {
  for (int i = 1; i < argc - 1; ++i) if (key == argv[i]) return std::string(argv[i+1]); return def;
}
static double getArgD(int argc, char** argv, const std::string& key, double def) {
  const auto s = getArg(argc, argv, key, ""); return s.empty() ? def : std::stod(s);
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

int main(int argc, char** argv) {
  if (argc < 2 || hasArg(argc, argv, "--help")) {
    std::cout <<
      "Usage: process <input.edm4hep.root> [--out out.root]\n"
      "                [--sampling f] [--expected-pdg PDG]\n"
      "                [--start-threshold GeV] [--debug]\n";
    return 0;
  }

  const bool debug     = hasArg(argc, argv, "--debug");
  const std::string inFile   = argv[1];
  const std::string outFile  = getArg(argc, argv, "--out", "events.root");
  const int expectedPDG      = getArgI(argc, argv, "--expected-pdg", 0);
  const double sampling_f    = getArgD(argc, argv, "--sampling", 0.03);
  const double start_threshold_GeV = getArgD(argc, argv, "--start-threshold", 1e-2);

  TFile* fout = TFile::Open(outFile.c_str(), "RECREATE");
  if (!fout || fout->IsZombie()) {
    std::cerr << "ERROR: Cannot open output file: " << outFile << std::endl;
    return 2;
  }

  float  t_mc_E=0;
  float  t_sim_E=0;
  float  t_rec_E=0;
  int    t_start_layer=-1;

  TTree* events = new TTree("events", "per-event dataset");
  events->Branch("mc_E",   &t_mc_E);
  events->Branch("sim_E",  &t_sim_E);
  events->Branch("rec_E", &t_rec_E);
  events->Branch("start_layer", &t_start_layer);
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
      
      // Prefer the requested PDG, then prefer particles not created in simulation.
      std::optional<edm4hep::MCParticle> same_pdg_not_from_sim_candidate;
      std::optional<edm4hep::MCParticle> same_pdg_candidate;
      std::optional<edm4hep::MCParticle> not_from_sim_candidate;
      std::optional<edm4hep::MCParticle> highest_energy_candidate;
      double same_pdg_not_from_sim_candidate_energy = -1.0;
      double same_pdg_candidate_energy = -1.0;
      double not_from_sim_candidate_energy = -1.0;
      double highest_energy_candidate_energy = -1.0;

      for (const auto& p : *mcCollection) {
        // Cache the particle energy once because every comparison uses it.
        const double energy = energy_from_mc(p);
        const bool not_from_sim = !p.isCreatedInSimulation();
        const int pdg = p.getPDG();

        // Keep the highest-energy fallback candidate that was not created in simulation.
        if (not_from_sim && energy > not_from_sim_candidate_energy) {
          not_from_sim_candidate = p;
          not_from_sim_candidate_energy = energy;
        }
        // Keep the highest-energy fallback candidate regardless of origin or PDG.
        if (energy > highest_energy_candidate_energy) {
          highest_energy_candidate = p;
          highest_energy_candidate_energy = energy;
        }

        // Keep the best candidates whose PDG matches the requested particle type.
        if (expectedPDG != 0 && pdg != 0 && std::abs(pdg) == std::abs(expectedPDG)) {
          // Prefer matching particles that were not created in simulation.
          if (not_from_sim && energy > same_pdg_not_from_sim_candidate_energy) {
            same_pdg_not_from_sim_candidate = p;
            same_pdg_not_from_sim_candidate_energy = energy;
          }
          // Otherwise keep the highest-energy matching particle.
          if (energy > same_pdg_candidate_energy) {
            same_pdg_candidate = p;
            same_pdg_candidate_energy = energy;
          }
        }
      }

      std::optional<edm4hep::MCParticle> selected_candidate;
      if (same_pdg_not_from_sim_candidate) selected_candidate = same_pdg_not_from_sim_candidate;
      else if (same_pdg_candidate) selected_candidate = same_pdg_candidate;
      else if (not_from_sim_candidate) selected_candidate = not_from_sim_candidate;
      else if (highest_energy_candidate) selected_candidate = highest_energy_candidate;

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

    t_sim_E = 0.f;
    t_start_layer = -1;
    std::array<double, kHcalLayerCount> layer_sim_energy_GeV {};

    // Load the calorimeter hit collection and collapse it to one total energy.
    const edm4hep::SimCalorimeterHitCollection* simCollection = nullptr;
    try {
      simCollection = &frame.get<edm4hep::SimCalorimeterHitCollection>("HCal_Readout");
    } catch (...) {}
    if (simCollection && debug && nEvents==1) std::cerr << "[process] Using Sim collection: HCal_Readout (size=" << simCollection->size() << ")\n";

    if (simCollection) {
      if (!simCollection->empty()) {
        ++nWithSim;
      }
      for (const auto& hit : *simCollection) {
        const double e = hit.getEnergy();
        t_sim_E += (float)e;
        // The readout encoding reserves bits 8..15 for the logical layer index.
        const int layer_index = decode_hcal_layer(static_cast<std::uint64_t>(hit.getCellID()));
        if (layer_index < 0 || layer_index >= kHcalLayerCount) {
          throw std::runtime_error("Decoded HCAL layer index is out of range.");
        }
        // Sum energy deposited within layer.
        layer_sim_energy_GeV[static_cast<std::size_t>(layer_index)] += e;
      }
    }

    for (int layer_index = 0; layer_index < kHcalLayerCount; ++layer_index) {
      // Loop through all layers in order and record the first layer that meets a certain threshold of energy.
      if (layer_sim_energy_GeV[static_cast<std::size_t>(layer_index)] > start_threshold_GeV) {
        t_start_layer = layer_index;
        break;
      }
    }

    // Convert the raw sim-energy sum into a simple reconstructed-energy estimate.
    t_rec_E = (sampling_f > 0 ? t_sim_E / (float)sampling_f : 0.f);

    t_mc_E = (float)mcE;
    if (debug && nEvents==1) {
      std::cerr << "[process] start-threshold=" << start_threshold_GeV
                << " GeV | start_layer=" << t_start_layer << "\n";
    }
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

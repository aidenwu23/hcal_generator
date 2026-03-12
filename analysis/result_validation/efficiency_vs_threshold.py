#!/usr/bin/env python3
"""
Plot efficiency vs muon threshold for two scan CSV files.

Example:
python3 analysis/result_validation/efficiency_vs_threshold.py \
  --reference-csv data/result_validation/81c3da7d_threshold_scan.csv \
  --candidate-csv data/result_validation/06529cd5_threshold_scan.csv \
  --out-root data/result_validation/threshold_comparison.root
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ThresholdScanRow:
    muon_threshold_GeV: float
    detection_efficiency: float
    eff_lo: float
    eff_hi: float


@dataclass
class EfficiencyDifferenceRow:
    muon_threshold_GeV: float
    difference_efficiency: float


def require_root():
    try:
        import ROOT  # type: ignore
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("ROOT Python bindings are required to run this script.") from error
    return ROOT


def parse_arguments() -> argparse.Namespace:
    # Read the two scan CSV paths and the output ROOT file path.
    parser = argparse.ArgumentParser(
        description="Plot muon-threshold efficiency scans for two CSV inputs."
    )
    parser.add_argument(
        "--reference-csv",
        required=True,
        help="Reference threshold scan CSV path.",
    )
    parser.add_argument(
        "--candidate-csv",
        required=True,
        help="Candidate threshold scan CSV path.",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output ROOT path.",
    )
    return parser.parse_args()


def load_scan_rows(csv_path: Path) -> list[ThresholdScanRow]:
    # Read one threshold scan CSV and keep the threshold grid unique and sorted.
    with csv_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {
            "muon_threshold_GeV",
            "detection_efficiency",
            "eff_lo",
            "eff_hi",
        }
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path} is missing one or more required columns: "
                "muon_threshold_GeV, detection_efficiency, eff_lo, eff_hi"
            )

        scan_rows: list[ThresholdScanRow] = []
        seen_thresholds: set[float] = set()
        # Read each CSV row once and reject duplicate threshold points.
        for raw_row in reader:
            threshold_value = float(raw_row["muon_threshold_GeV"])
            if threshold_value in seen_thresholds:
                raise ValueError(f"{csv_path} contains duplicate threshold {threshold_value}.")
            seen_thresholds.add(threshold_value)
            scan_rows.append(
                ThresholdScanRow(
                    muon_threshold_GeV=threshold_value,
                    detection_efficiency=float(raw_row["detection_efficiency"]),
                    eff_lo=float(raw_row["eff_lo"]),
                    eff_hi=float(raw_row["eff_hi"]),
                )
            )

    if not scan_rows:
        raise ValueError(f"{csv_path} does not contain any scan rows.")
    scan_rows.sort(key=lambda row: row.muon_threshold_GeV)
    return scan_rows


def build_difference_rows(
    reference_rows: list[ThresholdScanRow],
    candidate_rows: list[ThresholdScanRow],
) -> list[EfficiencyDifferenceRow]:
    # Match the two scan curves by threshold and compute candidate minus reference.
    if len(reference_rows) != len(candidate_rows):
        raise ValueError(
            "Reference and candidate CSV files have different numbers of thresholds: "
            f"{len(reference_rows)} vs {len(candidate_rows)}."
        )

    difference_rows: list[EfficiencyDifferenceRow] = []
    # Match each threshold pair in order and compute candidate minus reference.
    for reference_row, candidate_row in zip(reference_rows, candidate_rows):
        if reference_row.muon_threshold_GeV != candidate_row.muon_threshold_GeV:
            raise ValueError(
                "Reference and candidate CSV files do not share the same threshold grid: "
                f"{reference_row.muon_threshold_GeV} vs {candidate_row.muon_threshold_GeV}."
            )

        difference_rows.append(
            EfficiencyDifferenceRow(
                muon_threshold_GeV=reference_row.muon_threshold_GeV,
                difference_efficiency=(
                    candidate_row.detection_efficiency - reference_row.detection_efficiency
                ),
            )
        )

    return difference_rows


def build_scan_graph(
    graph_name: str,
    graph_title: str,
    scan_rows: list[ThresholdScanRow],
    line_color: int,
    root_module,
):
    # Draw the scan curve with asymmetric Wilson interval error bars at each threshold point.
    scan_graph = root_module.TGraphAsymmErrors(len(scan_rows))
    scan_graph.SetName(graph_name)
    scan_graph.SetTitle(graph_title)
    scan_graph.SetLineColor(line_color)
    scan_graph.SetMarkerColor(line_color)
    scan_graph.SetLineWidth(2)
    scan_graph.SetMarkerStyle(20)
    # Fill the graph with the central efficiency and the Wilson interval at each threshold.
    for row_index, row in enumerate(scan_rows):
        scan_graph.SetPoint(
            row_index,
            row.muon_threshold_GeV,
            row.detection_efficiency,
        )
        scan_graph.SetPointError(
            row_index,
            0.0,
            0.0,
            row.eff_lo,
            row.eff_hi,
        )
    return scan_graph


def build_difference_graph(
    difference_rows: list[EfficiencyDifferenceRow],
    root_module,
):
    # Draw the central-value efficiency difference on the shared threshold grid.
    difference_graph = root_module.TGraph(len(difference_rows))
    difference_graph.SetName("candidate_minus_reference")
    difference_graph.SetTitle(
        "Candidate - Reference Efficiency;Muon threshold [GeV];#Delta efficiency"
    )
    difference_graph.SetLineColor(root_module.kBlack)
    difference_graph.SetMarkerColor(root_module.kBlack)
    difference_graph.SetLineWidth(2)
    difference_graph.SetMarkerStyle(20)
    # Fill the difference graph on the shared threshold grid.
    for row_index, row in enumerate(difference_rows):
        difference_graph.SetPoint(
            row_index,
            row.muon_threshold_GeV,
            row.difference_efficiency,
        )
    return difference_graph


def write_root_file(
    output_root_path: Path,
    reference_rows: list[ThresholdScanRow],
    candidate_rows: list[ThresholdScanRow],
    difference_rows: list[EfficiencyDifferenceRow],
) -> None:
    # Write the overlay and difference graphs into one ROOT file.
    root_module = require_root()
    output_root_path.parent.mkdir(parents=True, exist_ok=True)
    root_module.gROOT.SetBatch(True)

    output_file = root_module.TFile(str(output_root_path), "RECREATE")
    if not output_file or output_file.IsZombie():
        raise OSError(f"Failed to open {output_root_path} for writing.")

    reference_graph = build_scan_graph(
        "reference_efficiency_vs_threshold",
        "Efficiency vs Muon Threshold;Muon threshold [GeV];Detection efficiency",
        reference_rows,
        root_module.kBlue + 1,
        root_module,
    )
    candidate_graph = build_scan_graph(
        "candidate_efficiency_vs_threshold",
        "Efficiency vs Muon Threshold;Muon threshold [GeV];Detection efficiency",
        candidate_rows,
        root_module.kRed + 1,
        root_module,
    )
    difference_graph = build_difference_graph(difference_rows, root_module)

    overlay_canvas = root_module.TCanvas("efficiency_overlay", "efficiency_overlay", 900, 700)
    reference_graph.Draw("ALP")
    candidate_graph.Draw("LP SAME")
    overlay_legend = root_module.TLegend(0.62, 0.75, 0.88, 0.88)
    overlay_legend.AddEntry(reference_graph, "Reference", "lp")
    overlay_legend.AddEntry(candidate_graph, "Candidate", "lp")
    overlay_legend.Draw()

    difference_canvas = root_module.TCanvas("efficiency_difference", "efficiency_difference", 900, 700)
    difference_graph.Draw("ALP")

    reference_graph.Write()
    candidate_graph.Write()
    difference_graph.Write()
    overlay_canvas.Write()
    difference_canvas.Write()
    output_file.Close()


def main() -> int:
    arguments = parse_arguments()
    reference_csv_path = Path(arguments.reference_csv).expanduser().resolve()
    candidate_csv_path = Path(arguments.candidate_csv).expanduser().resolve()
    output_root_path = Path(arguments.out_root).expanduser().resolve()

    if not reference_csv_path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv_path}")
    if not candidate_csv_path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_csv_path}")

    reference_rows = load_scan_rows(reference_csv_path)
    candidate_rows = load_scan_rows(candidate_csv_path)
    difference_rows = build_difference_rows(reference_rows, candidate_rows)
    write_root_file(output_root_path, reference_rows, candidate_rows, difference_rows)

    print(f"[efficiency_vs_threshold] wrote {output_root_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Measures whether fusion actually beats either model alone, on a labeled set
you build yourself. This is the number your supervisor wants -- not just the
noisy-OR math argument.

MANIFEST CSV FORMAT (you build this):
    thermal_image_path, rf_reading_csv_path, true_label
    imgs/shading_01.jpg, readings/shading_01.csv, shading
    imgs/normal_04.jpg,  readings/normal_04.csv,  not_shading
    ...

- true_label must be exactly "shading" or "not_shading" (binary task -- this
  eval is specifically about the ONE class that's shared across models).
- rf_reading_csv_path must be a single-row CSV with the columns your chosen
  string_config model expects (Voc_V, Vmp_V, Isc_A, Imp_A, Pmax_W, Temp_C,
  Irr_Wm2 -- whatever scaler.feature_names_in_ lists).
- Include both shading and non-shading examples, and try to include cases
  where the two modalities would individually disagree, since that's exactly
  where fusion should help most.

USAGE:
    python eval_fusion.py your_manifest.csv --string-config 1-string

Requires app.py's model loaders and fusion.py to be importable (run this
from your project root, not from this scratch folder).
"""

import argparse
import csv
import sys

import pandas as pd

import fusion
from app import (
    run_rf_inference_from_df,
    run_thermal_cnn_inference,
    REFERENCE_PANELS_IN_SERIES,
)


def _is_shading(label: str) -> bool:
    return "shad" in str(label).lower()


def evaluate(manifest_path: str, string_config: str):
    rows = list(csv.DictReader(open(manifest_path)))
    if not rows:
        print("Manifest is empty.")
        sys.exit(1)

    rf_correct = thermal_correct = fused_correct = 0
    total = 0

    for row in rows:
        true_shading = row["true_label"].strip().lower() == "shading"

        rf_df = pd.read_csv(row["rf_reading_csv_path"])
        rf_result = run_rf_inference_from_df(
            rf_df, string_config, panels_in_series=REFERENCE_PANELS_IN_SERIES
        )

        with open(row["thermal_image_path"], "rb") as f:
            thermal_result = run_thermal_cnn_inference(f)

        fused = fusion.fuse_predictions(rf_result, thermal_result, string_config)

        rf_pred_shading = _is_shading(rf_result["detection"])
        thermal_pred_shading = _is_shading(thermal_result["detection"])
        fused_pred_shading = fused["fused_status"] == "CONFIRMED_CROSS_MODAL"

        rf_correct += int(rf_pred_shading == true_shading)
        thermal_correct += int(thermal_pred_shading == true_shading)
        fused_correct += int(fused_pred_shading == true_shading)
        total += 1

        print(
            f"[{row['thermal_image_path']}] true={row['true_label']:>12} | "
            f"RF={rf_result['detection']:>10} thermal={thermal_result['detection']:>10} | "
            f"fused_score={fused['combined_shading_score']:.3f} "
            f"({fused['fused_status']})"
        )

    print("\n" + "=" * 50)
    print(f"RF alone accuracy:      {rf_correct / total * 100:.2f}%  ({rf_correct}/{total})")
    print(f"Thermal alone accuracy: {thermal_correct / total * 100:.2f}%  ({thermal_correct}/{total})")
    print(f"Fused accuracy:         {fused_correct / total * 100:.2f}%  ({fused_correct}/{total})")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_csv")
    parser.add_argument("--string-config", choices=["1-string", "3-string"], default="1-string")
    args = parser.parse_args()
    evaluate(args.manifest_csv, args.string_config)

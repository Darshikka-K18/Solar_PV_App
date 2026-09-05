"""
Decision-level fusion of the RF (sensor) and Thermal CNN (image) models.

Why decision-level fusion, not a joint model:
    RF classes:      Normal, Shading, Short, Connector, OC
    Thermal classes:  Cracking, Diode, Shadowing
These are two different physical fault taxonomies detected by two different
physics (electrical signature vs. heat pattern). They only share ONE
overlapping physical event: RF's "Shading" and Thermal's "Shadowing" are the
same thing seen two ways. Everything else has no counterpart in the other
model, so there's nothing to fuse there -- it's reported as an independent
finding, same as each tab does today.

Where the precision numbers below came from (real, not placeholders):
    RF_SHADING_PRECISION = 0.99 (1-string) / 1.00 (3-string)
        From the RF training classification reports (class index 1 =
        'Shading'): 0.99 precision / 1.00 recall (1-string model, support
        300) and 1.00 / 1.00 (3-string model, support 287).
        CAVEAT: measured on simulated panel data, not real sensor readings.
        Cross-validated numbers were also available (1.00 for both configs,
        99.9%+ cross-validated accuracy) but these single-split numbers were
        used instead. Either way, treat this as an upper bound on RF's
        actual field reliability until validated against real sensor data.
    THERMAL_SHADOWING_PRECISION = 0.7193
        From thermal_shadowing_classification_report.txt, the Shadowing row.
        Cross-checked against the confusion matrix: 82 correct / 114 total
        predicted-Shadowing = 0.719 -- consistent. This one IS on real
        thermal images, so it's a trustworthy field estimate.

If you retrain either model, regenerate a confusion matrix / classification
report and update these two constants -- don't guess them.
"""

RF_SHADING_PRECISION = {
    # From the RF training classification reports, class index 1 = 'Shading'.
    # (Cross-validated numbers were also available -- 1.00 for both configs
    # -- but the single-split numbers below were chosen instead.)
    "1-string": 0.99,  # precision 0.99, recall 1.00, support 300
    "3-string": 1.00,  # precision 1.00, recall 1.00, support 287
}
THERMAL_SHADOWING_PRECISION = 0.7193

# Confirmed threshold: below this, there isn't enough combined evidence to
# call it a confirmed cross-modal finding. Tune this against your own
# manifest results in eval_fusion.py rather than trusting it blindly.
FUSION_CONFIRM_THRESHOLD = 0.5

RF_SHADING_LABEL = "Shading"
THERMAL_SHADOWING_LABEL = "Shadowing"


def _extract_rf_shading_prob(rf_result: dict) -> float:
    """Pulls RF's probability specifically for the Shading class, even when
    it wasn't RF's top-1 pick. Falls back to 1.0/0.0 based on the top label
    if class probabilities weren't attached (e.g. model has no predict_proba)."""
    probs = rf_result.get("_debug_class_probabilities")
    if probs and RF_SHADING_LABEL in probs:
        return float(probs[RF_SHADING_LABEL])
    return 1.0 if rf_result.get("detection") == RF_SHADING_LABEL else 0.0


def _extract_thermal_shadowing_prob(thermal_result: dict) -> float:
    """Same idea for Thermal's Shadowing probability."""
    probs = thermal_result.get("_debug_class_probabilities")
    if probs and THERMAL_SHADOWING_LABEL in probs:
        return float(probs[THERMAL_SHADOWING_LABEL])
    return 1.0 if thermal_result.get("detection") == THERMAL_SHADOWING_LABEL else 0.0


def fuse_predictions(rf_result: dict, thermal_result: dict, string_config: str = "1-string") -> dict:
    """Combines one RF sensor prediction and one Thermal image prediction
    for the SAME physical panel/moment.

    string_config: '1-string' or '3-string' -- which RF model produced
    rf_result, so the correct RF precision weight is used (the two RF models
    have slightly different measured precision on the Shading class).

    Returns a dict describing the fused finding -- either a confirmed
    cross-modal diagnosis (both modalities point at shading) or two
    independent findings reported honestly side by side.
    """
    rf_precision = RF_SHADING_PRECISION.get(string_config, RF_SHADING_PRECISION["1-string"])

    p_rf = _extract_rf_shading_prob(rf_result)
    p_thermal = _extract_thermal_shadowing_prob(thermal_result)

    # Precision-weight each modality's raw probability before combining --
    # an unreliable detector's "confidence" should count for less. This is
    # not double counting: it discounts each probability by how trustworthy
    # THAT model's Shading/Shadowing call has historically been, before the
    # two independent estimates are combined.
    weighted_rf = p_rf * rf_precision
    weighted_thermal = p_thermal * THERMAL_SHADOWING_PRECISION

    # Noisy-OR: combines two independent pieces of evidence for the same
    # event. Two moderate, independent signals agreeing is stronger evidence
    # than either alone -- a plain average can't reflect that; noisy-OR can.
    #   combined = 1 - (1 - a)(1 - b)
    combined_score = 1 - (1 - weighted_rf) * (1 - weighted_thermal)

    result = {
        "rf_detection": rf_result.get("detection"),
        "rf_shading_probability": round(p_rf, 4),
        "thermal_detection": thermal_result.get("detection"),
        "thermal_shadowing_probability": round(p_thermal, 4),
        "combined_shading_score": round(combined_score, 4),
    }

    if combined_score >= FUSION_CONFIRM_THRESHOLD:
        result["fused_status"] = "CONFIRMED_CROSS_MODAL"
        result["fused_label"] = "Partial Shading"
        result["fused_confidence_pct"] = round(combined_score * 100, 2)
        result["explanation"] = (
            f"Sensor data ({p_rf*100:.1f}% Shading probability) and thermal "
            f"imagery ({p_thermal*100:.1f}% Shadowing probability) both point "
            f"to partial shading. Combined confidence: {combined_score*100:.1f}%."
        )
    else:
        result["fused_status"] = "NO_SHARED_EVIDENCE"
        result["fused_label"] = None
        result["fused_confidence_pct"] = None
        result["explanation"] = (
            f"Sensor top finding: {rf_result.get('detection')}. "
            f"Thermal top finding: {thermal_result.get('detection')}. "
            "These don't share enough combined evidence to confirm a single "
            "cross-modal diagnosis -- reporting both findings independently "
            "rather than forcing a match that isn't there."
        )

    return result


if __name__ == "__main__":
    # Self-check with synthetic cases -- run `python fusion.py` to sanity
    # check the logic without needing the real models loaded.
    print("Case 1: both models moderately confident, neither top-1 --")
    r1 = fuse_predictions(
        {"detection": "Normal", "_debug_class_probabilities": {"Normal": 0.55, "Shading": 0.71, "Short": 0.0, "Connector": 0.0, "OC": 0.0}},
        {"detection": "Diode", "_debug_class_probabilities": {"Cracking": 0.1, "Diode": 0.44, "Shadowing": 0.46}},
    )
    print(r1)

    print("\nCase 2: no shared evidence --")
    r2 = fuse_predictions(
        {"detection": "Connector", "_debug_class_probabilities": {"Normal": 0.02, "Shading": 0.03, "Short": 0.05, "Connector": 0.88, "OC": 0.02}},
        {"detection": "Cracking", "_debug_class_probabilities": {"Cracking": 0.81, "Diode": 0.1, "Shadowing": 0.09}},
    )
    print(r2)

    print("\nCase 3: strong agreement --")
    r3 = fuse_predictions(
        {"detection": "Shading", "_debug_class_probabilities": {"Normal": 0.02, "Shading": 0.9, "Short": 0.03, "Connector": 0.03, "OC": 0.02}},
        {"detection": "Shadowing", "_debug_class_probabilities": {"Cracking": 0.05, "Diode": 0.15, "Shadowing": 0.8}},
    )
    print(r3)

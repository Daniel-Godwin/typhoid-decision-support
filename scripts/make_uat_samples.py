"""Build a user-acceptance test pack from the held-out test partition.

Ten patient records the reviewer types into the deployed system, each with the
result the system will actually return — computed here by running the saved
model, so the expected column cannot disagree with the live application.

The selection is deliberately not ten easy cases. Eight are routine; two are
typhoid patients recorded with a fever duration of zero, which the model misses.
Those two are the only cases in this dataset whose diagnosis cannot be read off
a single variable, so a pack without them would demonstrate nothing and would
misrepresent the system. They are declared in advance, with the reason, so the
reviewer records them as expected behaviour rather than as defects.

    python scripts/make_uat_samples.py

Writes to reports/uat/:
    UAT_TEST_PACK.md        the document to hand the reviewer
    uat_samples.csv         the ten records, one row each
    uat_api_payloads.json   the same ten as /api/predict request bodies
    uat_expected.json       expected outputs, for scoring
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typhoid_ml.config import (  # noqa: E402
    BINARY_POSITIVE,
    DATA_PATH,
    DEFAULT_POLICY,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
)
from typhoid_ml.data import build_target, feature_frame, load_dataset  # noqa: E402
from typhoid_ml.predict import model_path  # noqa: E402
from typhoid_ml.threshold import positive_probabilities  # noqa: E402

OUT = REPORT_DIR / "uat"

# Matches webapp TRIAGE_THRESHOLD; see reports/THRESHOLD_ANALYSIS.md.
THRESHOLD = 0.08

# How many of each kind. The two hard cases are typhoid patients whose fever
# duration is zero — the shortcut cannot reach them, and neither can the model.
# The borderline case is one of only four positives in the test partition whose
# probability falls between the threshold and certainty: 1,800 of 1,877 sit at
# 0.999 or above, which is itself a symptom of the deterministic label rule.
N_TYPHOID_ROUTINE = 2
N_TYPHOID_BORDERLINE = 1
N_TYPHOID_HARD = 2
N_NEGATIVE = 5

# Case order in the pack. The two known failures are placed mid-pack rather
# than first, so the reviewer meets the system working before meeting its
# limitation, and neither is buried at the end.
CASE_ORDER = [
    "routine_pos", "negative", "negative", "hard",
    "borderline", "negative", "routine_pos", "hard",
    "negative", "negative",
]

PATIENT_META = [
    ("UAT-001", "Amara Okafor"), ("UAT-002", "Bello Yusuf"),
    ("UAT-003", "Chidinma Nwosu"), ("UAT-004", "Danladi Sule"),
    ("UAT-005", "Esther Adeniyi"), ("UAT-006", "Farouk Bala"),
    ("UAT-007", "Grace Mensah"), ("UAT-008", "Hauwa Ibrahim"),
    ("UAT-009", "Ikenna Obi"), ("UAT-010", "Jamila Aliyu"),
]


def select(df, X, y, proba) -> pd.DataFrame:
    """Pick the ten records, stratified and reproducible.

    Returns them in CASE_ORDER, so the two known failures sit mid-pack.
    """
    fever = pd.to_numeric(df["Fever Duration (Days)"], errors="coerce")
    is_pos = y == BINARY_POSITIVE
    p = pd.Series(proba, index=df.index)

    hard_idx = df.index[is_pos & (fever == 0)]
    pos_idx = df.index[is_pos & (fever > 0)]
    neg_idx = df.index[~is_pos]

    pools = {}
    # Hard: zero-fever typhoid patients the model is most confident are
    # negative — the clearest demonstration of the failure.
    pools["hard"] = list(p.loc[hard_idx].nsmallest(N_TYPHOID_HARD).index)

    # Borderline: a typhoid patient the model is genuinely uncertain about.
    # These exist only among the zero-fever cases. Every one of the 1,799
    # typhoid patients with a fever duration of one day or more scores at
    # least 0.99998 — the model has no uncertainty whatsoever about any case
    # the shortcut covers. Taking the highest-scoring zero-fever patient gives
    # a difficult case the model nonetheless gets right, which keeps the pack
    # honest in both directions.
    zero_fever_caught = p.loc[hard_idx]
    zero_fever_caught = zero_fever_caught[zero_fever_caught >= THRESHOLD]
    pools["borderline"] = list(
        zero_fever_caught.nlargest(N_TYPHOID_BORDERLINE).index
    )

    # Routine positives: spread across the straightforward cases.
    remaining = list(p.loc[pos_idx].sort_values(ascending=False).index)
    n_routine = N_TYPHOID_ROUTINE + (N_TYPHOID_BORDERLINE - len(pools["borderline"]))
    step = max(1, len(remaining) // max(1, n_routine))
    pools["routine_pos"] = remaining[::step][:n_routine]
    # Negatives: spread from least to most suspicious.
    neg_sorted = list(p.loc[neg_idx].sort_values().index)
    step = max(1, len(neg_sorted) // N_NEGATIVE)
    pools["negative"] = neg_sorted[::step][:N_NEGATIVE]

    # Lay the pools out in the requested order, never taking the same record
    # twice. A pool that runs short falls through to whatever is left, so the
    # pack is always ten distinct cases.
    order, used = [], set()
    queues = {k: [i for i in v] for k, v in pools.items()}
    leftovers = [i for k in ("routine_pos", "negative", "borderline", "hard")
                 for i in queues.get(k, [])]

    for kind in CASE_ORDER:
        picked = None
        for candidate in queues.get(kind, []):
            if candidate not in used:
                picked = candidate
                break
        if picked is None:
            for candidate in leftovers:
                if candidate not in used:
                    picked = candidate
                    break
        if picked is None:
            continue
        used.add(picked)
        order.append(picked)

    assert len(order) == len(set(order)), "duplicate record selected"
    return df.loc[order]


def main():
    df_all = load_dataset(DATA_PATH)
    X_all, numeric, categorical = feature_frame(df_all, DEFAULT_POLICY)
    y_all = build_target(df_all, "binary")
    _, idx_test = train_test_split(
        df_all.index, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )
    df = df_all.loc[idx_test]
    X, y = X_all.loc[idx_test], y_all.loc[idx_test]

    path = model_path("binary", DEFAULT_POLICY)
    model = joblib.load(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    proba = positive_probabilities(model, X)

    chosen = select(df, X, y, proba)
    chosen_proba = positive_probabilities(model, X.loc[chosen.index])

    fever_values = pd.to_numeric(
        chosen["Fever Duration (Days)"], errors="coerce"
    ).to_numpy()
    rows = []
    for n, (idx, p) in enumerate(zip(chosen.index, chosen_proba)):
        truth = y.loc[idx]
        fever_days = float(fever_values[n])
        predicted = BINARY_POSITIVE if p >= THRESHOLD else "No Typhoid"
        code, name = PATIENT_META[n]
        record = {c: X.loc[idx, c] for c in numeric + categorical}
        rows.append(
            {
                "case": n + 1,
                "patient_code": code,
                "patient_name": name,
                "source_row": int(idx),
                "recorded_diagnosis": truth,
                "system_prediction": predicted,
                "probability": round(float(p), 4),
                "agrees_with_record": predicted == truth,
                "kind": (
                    (
                        "known-difficult"
                        if predicted != truth
                        else "difficult, correctly identified"
                    )
                    if truth == BINARY_POSITIVE and fever_days == 0
                    else "routine"
                ),
                **record,
            }
        )

    out = pd.DataFrame(rows).sort_values("case")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "uat_samples.csv", index=False)

    payloads = [
        {
            "case": int(r["case"]),
            "patient_code": r["patient_code"],
            "record": {c: (int(r[c]) if c in numeric else r[c]) for c in numeric + categorical},
        }
        for _, r in out.iterrows()
    ]
    (OUT / "uat_api_payloads.json").write_text(
        json.dumps(payloads, indent=2), encoding="utf-8"
    )
    (OUT / "uat_expected.json").write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": path.name,
                "model_sha256_prefix": digest,
                "threshold": THRESHOLD,
                "cases": [
                    {
                        "case": int(r["case"]),
                        "patient_code": r["patient_code"],
                        "expected_prediction": r["system_prediction"],
                        "expected_probability": float(r["probability"]),
                        "recorded_diagnosis": r["recorded_diagnosis"],
                        "agrees_with_record": bool(r["agrees_with_record"]),
                        "kind": r["kind"],
                    }
                    for _, r in out.iterrows()
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_pack(out, numeric, categorical, digest, path.name)
    agree = int(out["agrees_with_record"].sum())
    print(f"model {path.name} sha256:{digest}")
    print(f"{len(out)} cases | prediction agrees with the record in {agree}/{len(out)}")
    print(f"Wrote {OUT}")


def write_pack(out, numeric, categorical, digest, model_name):
    L = ["# User Acceptance Test Pack\n"]
    L.append(
        "**System:** Typhoid Diagnostic Decision Support v2.0.0  \n"
        "**URL:** https://typhoid-decision-support.onrender.com  \n"
        f"**Model under test:** `{model_name}` (SHA-256 prefix `{digest}`)  \n"
        f"**Referral threshold:** {THRESHOLD:.2f}  \n"
        f"**Generated:** {datetime.now(timezone.utc).strftime('%d %B %Y')} by "
        "`python scripts/make_uat_samples.py`\n"
    )

    L.append("## What this pack is for\n")
    L.append(
        "Ten patient records drawn from the held-out test partition — data the "
        "model never saw during training. Each is entered into the deployed "
        "system through the normal clinician workflow, and the result compared "
        "against the **expected system output** stated below.\n"
    )
    L.append(
        "The expected outputs were produced by running the same saved model "
        "artefact that the deployed service loads. They are what the system "
        "should return, not a guess.\n"
    )

    L.append("## Acceptance criteria\n")
    L.append(
        "The test **passes** when, for all ten cases:\n\n"
        "1. The record is accepted without a validation error.\n"
        "2. The prediction matches the **Expected result** column.\n"
        "3. A calibrated probability and a triage recommendation are shown.\n"
        "4. The contributing findings are listed.\n"
        "5. The assessment is still present after signing out and back in.\n"
    )
    L.append(
        "The criterion is agreement with the **expected system output**, not "
        "with the patient's recorded diagnosis. Two of the ten differ, and that "
        "is documented below rather than discovered during the test.\n"
    )

    hard = out[out["kind"] == "known-difficult"]
    L.append("## Declared limitation — please read before testing\n")
    L.append(
        f"Cases **{', '.join(str(int(c)) for c in hard['case'])}** are typhoid "
        "patients whom the system will report as **No Typhoid**. This is a known "
        "and documented limitation, not a fault in the deployment.\n"
    )
    L.append(
        "In the source dataset every patient with a fever duration of one day or "
        "more is labelled typhoid, without a single exception in 8,981 records. "
        "The model has learned that rule. These two patients were recorded with a "
        "fever duration of zero, so the rule cannot reach them — and neither can "
        "the model, which is why they are in this pack. Excluding them would make "
        "the system look better than it is.\n"
    )
    caught = out[out["kind"] == "difficult, correctly identified"]
    if not caught.empty:
        cs = ", ".join(str(int(c)) for c in caught["case"])
        L.append(
            f"Case **{cs}** is the counterpart: a typhoid patient also recorded "
            "with a fever duration of zero, whom the system does identify "
            "correctly. It is included so the pack is not weighted towards "
            "failure.\n"
        )
    L.append(
        "One further observation makes the point sharply. Every one of the "
        "1,799 typhoid patients in the test partition with a fever duration of "
        "one day or more receives a probability of at least 0.99998 — not one "
        "falls below 0.999. The model expresses no uncertainty at all about any "
        "case the rule covers. The only patients it is genuinely unsure about "
        "are the zero-fever cases, which is where a real diagnostic judgement "
        "would actually be required.\n"
    )
    L.append(
        "The full analysis is in `docs/DATASET_AUDIT.md` and "
        "`reports/THRESHOLD_ANALYSIS.md`. The short version: the reported 98.8% "
        "accuracy measures how the dataset was constructed, not diagnostic "
        "ability, and the system must not be used on real patients on this "
        "training data.\n"
    )

    L.append("## Summary of expected results\n")
    L.append("| Case | Patient | Recorded diagnosis | Expected result | Probability | Type |")
    L.append("|---|---|---|---|---:|---|")
    for _, r in out.iterrows():
        flag = "" if r["agrees_with_record"] else " ⚠"
        L.append(
            f"| {int(r['case'])} | {r['patient_code']} | {r['recorded_diagnosis']} | "
            f"**{r['system_prediction']}**{flag} | {r['probability']:.4f} | {r['kind']} |"
        )
    L.append("")
    L.append(
        f"Agreement with the recorded diagnosis: "
        f"**{int(out['agrees_with_record'].sum())} of {len(out)}**. "
        "The two disagreements are the declared cases above.\n"
    )

    L.append("---\n")
    L.append("## How to run each case\n")
    L.append(
        "1. Sign in as a clinician.\n"
        "2. **Clinical → Patients → Register patient.** Enter the patient code, "
        "name, age, gender and location from the case below.\n"
        "3. **Clinical → New assessment.** Select that patient, then enter every "
        "clinical field exactly as listed.\n"
        "4. Click **Run assessment** and record what the system returns.\n"
    )
    L.append(
        "Three fields — Gastrointestinal Symptoms, Neurological Symptoms and "
        "Ongoing Infection in Society — are optional, because roughly a quarter "
        "of real records lack them. Where a case says *Not recorded*, leave the "
        "field at that setting rather than inventing a value.\n"
    )

    L.append("---\n")
    L.append("## The ten cases\n")

    for _, r in out.iterrows():
        L.append(f"### Case {int(r['case'])} — {r['patient_code']}\n")
        L.append("**Patient record**\n")
        L.append("| Field | Value |")
        L.append("|---|---|")
        L.append(f"| Patient code | `{r['patient_code']}` |")
        L.append(f"| Full name | {r['patient_name']} |")
        L.append(f"| Age | {int(r['Age'])} |")
        L.append(f"| Gender | {r['Gender']} |")
        L.append(f"| Location | {r['Location']} |")
        L.append("")
        L.append("**Assessment fields**\n")
        L.append("| Field | Value |")
        L.append("|---|---|")
        for c in numeric:
            if c == "Age":
                continue
            L.append(f"| {c} | {int(r[c])} |")
        for c in categorical:
            if c in ("Gender", "Location"):
                continue
            value = r[c]
            if pd.isna(value):
                value = "*Not recorded*"
            L.append(f"| {c} | {value} |")
        L.append("")
        L.append(
            f"**Expected result:** {r['system_prediction']} "
            f"(probability {r['probability']:.4f})\n"
        )
        if not r["agrees_with_record"]:
            L.append(
                f"> This patient's recorded diagnosis is **{r['recorded_diagnosis']}**. "
                "The system is expected to disagree — see the declared limitation "
                "above. Record this as **pass** if the system returns the expected "
                "result.\n"
            )
        L.append("**Tester records:**\n")
        L.append("| | |")
        L.append("|---|---|")
        L.append("| Result returned | |")
        L.append("| Probability shown | |")
        L.append("| Triage recommendation | |")
        L.append("| Matches expected? | Yes / No |")
        L.append("| Notes | |")
        L.append("")

    L.append("---\n")
    L.append("## Sign-off\n")
    L.append("| | |")
    L.append("|---|---|")
    L.append("| Cases passed | ___ / 10 |")
    L.append("| Tester name | |")
    L.append("| Date | |")
    L.append("| Overall result | Accepted / Accepted with comments / Rejected |")
    L.append("| Comments | |")
    L.append("")

    (OUT / "UAT_TEST_PACK.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

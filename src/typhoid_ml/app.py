"""Flask decision-support interface and JSON API."""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .config import DEFAULT_POLICY, REPORT_DIR
from .predict import (
    ValidationError,
    available_models,
    form_schema,
    load_model,
    predict_record,
    validate_record,
)

ROOT = Path(__file__).resolve().parents[2]

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)

FIELD_HELP = {
    "Age": "Patient age in years",
    "Fever Duration (Days)": "Days of fever before presentation",
    "White Blood Cell Count": "WBC, cells per microlitre",
    "Platelet Count": "Platelets per microlitre",
}

FIELD_GROUPS = [
    ("Demographics", ["Age", "Gender", "Location", "Socioeconomic Status"]),
    (
        "Environmental exposure",
        [
            "Water Source Type",
            "Sanitation Facilities",
            "Hand Hygiene",
            "Consumption of Street Food",
            "Weather Condition",
            "Ongoing Infection in Society",
        ],
    ),
    (
        "Clinical presentation",
        [
            "Fever Duration (Days)",
            "Gastrointestinal Symptoms",
            "Neurological Symptoms",
            "Skin Manifestations",
        ],
    ),
    (
        "Laboratory and history",
        [
            "White Blood Cell Count",
            "Platelet Count",
            "Widal Test",
            "Typhidot Test",
            "Typhoid Vaccination Status",
            "Previous History of Typhoid",
        ],
    ),
]


def _grouped_fields(schema):
    known = set(schema["numeric"]) | set(schema["categorical"])
    optional = set(schema.get("optional", []))
    groups, placed = [], set()
    for title, fields in FIELD_GROUPS:
        items = []
        for f in fields:
            if f not in known:
                continue
            placed.add(f)
            items.append(
                {
                    "name": f,
                    "kind": "numeric" if f in schema["numeric"] else "categorical",
                    "options": schema["levels"].get(f, []),
                    "range": schema["ranges"].get(f),
                    "help": FIELD_HELP.get(f, ""),
                    "optional": f in optional,
                }
            )
        if items:
            groups.append({"title": title, "fields": items})
    leftovers = [f for f in known if f not in placed]
    if leftovers:
        groups.append(
            {
                "title": "Other",
                "fields": [
                    {
                        "name": f,
                        "kind": "numeric" if f in schema["numeric"] else "categorical",
                        "options": schema["levels"].get(f, []),
                        "range": schema["ranges"].get(f),
                        "help": "",
                        "optional": f in optional,
                    }
                    for f in leftovers
                ],
            }
        )
    return groups


def _model_card(target_mode):
    try:
        _, meta = load_model(target_mode)
    except FileNotFoundError:
        return None
    eval_path = REPORT_DIR / f"final_evaluation_{target_mode}.json"
    metrics = {}
    if eval_path.exists():
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        metrics = {
            "accuracy": data.get("accuracy"),
            "balanced_accuracy": data.get("balanced_accuracy"),
            "macro_f1": data.get("macro_f1"),
            "sensitivity": data.get("sensitivity"),
            "specificity": data.get("specificity"),
            "roc_auc": data.get("roc_auc"),
        }
    return {"meta": meta, "metrics": metrics}


@app.route("/", methods=["GET", "POST"])
def index():
    target_mode = request.values.get("target_mode", "binary")
    if target_mode not in ("binary", "multiclass"):
        target_mode = "binary"
    schema = form_schema(DEFAULT_POLICY)
    groups = _grouped_fields(schema)
    result, errors, submitted = None, None, {}

    if request.method == "POST":
        submitted = {k: v for k, v in request.form.items() if k != "target_mode"}
        try:
            validated = validate_record(submitted, DEFAULT_POLICY)
            result = predict_record(validated["record"], target_mode, DEFAULT_POLICY)
            result["warnings"] = validated["warnings"]
        except ValidationError as exc:
            errors = json.loads(str(exc))
        except FileNotFoundError as exc:
            errors = {"_model": str(exc)}
        except Exception as exc:  # pragma: no cover
            errors = {"_error": str(exc)}

    return render_template(
        "index.html",
        groups=groups,
        reference=schema["reference"],
        submitted=submitted,
        result=result,
        errors=errors,
        target_mode=target_mode,
        model_card=_model_card(target_mode),
        available=sorted(available_models().keys()),
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "models": sorted(available_models().keys())})


@app.get("/api/schema")
def api_schema():
    schema = form_schema(DEFAULT_POLICY)
    return jsonify(
        {
            "numeric": schema["numeric"],
            "categorical": schema["categorical"],
            "levels": schema["levels"],
            "ranges": schema["ranges"],
        }
    )


@app.post("/api/predict")
def api_predict():
    """JSON endpoint for point-of-care / mobile clients.

    Body: {"record": {...}, "target_mode": "binary", "explain": true}
    """
    payload = request.get_json(silent=True) or {}
    record = payload.get("record", payload)
    target_mode = payload.get("target_mode", "binary")
    if target_mode not in ("binary", "multiclass"):
        return jsonify({"error": "target_mode must be 'binary' or 'multiclass'"}), 400
    try:
        validated = validate_record(record, DEFAULT_POLICY)
    except ValidationError as exc:
        return jsonify({"error": "validation_failed", "fields": json.loads(str(exc))}), 400
    try:
        result = predict_record(
            validated["record"],
            target_mode,
            DEFAULT_POLICY,
            explain=bool(payload.get("explain", True)),
        )
    except FileNotFoundError as exc:
        return jsonify({"error": "model_unavailable", "detail": str(exc)}), 503
    result["warnings"] = validated["warnings"]
    return jsonify(result)


def create_app():
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

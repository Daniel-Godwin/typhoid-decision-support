# Run this project

Windows PowerShell, from the project folder.

## 1. Environment (once)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Launch the decision-support interface

```powershell
python scripts\run_app.py
```

Open http://127.0.0.1:5000

Four trained models are already included in `models\`, so nothing needs to be
retrained to use the interface.

## 3. Verify the installation

```powershell
python -m pytest
```

Expected: **34 passed**.

## 4. Regenerate the analysis and Chapter 4 tables (fast, no retraining)

```powershell
python scripts\audit_dataset.py
python scripts\baseline_rules.py
python scripts\make_report.py
python scripts\make_chapter4.py
```

Writes `reports\RESULTS.md`, `reports\CHAPTER_4_DRAFT.md` and `reports\table_4_*.csv`.

## 5. Retrain from scratch (optional — hours)

```powershell
python scripts\train_model.py --mode all --target binary
python scripts\train_model.py --mode all --target multiclass
python scripts\train_model.py --mode final --target binary --policy no_fever_duration --params '{\"kernel\":\"rbf\",\"C\":1.0,\"gamma\":\"scale\"}'
python scripts\train_model.py --mode all --target binary --policy clinical_only
```

Approximate wall-clock on 2 cores: binary 20 min, four-class 50 min,
ablation 10 min, clinical-only 20 min. Use `--n-jobs` to match your CPU count.

To re-score a saved model without retraining it:

```powershell
python scripts\evaluate_saved.py --target binary --policy routine
```

## 6. JSON API

```powershell
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d "{\"record\": {...}}"
```

`GET /api/schema` returns every field name and its permitted values.
`GET /health` lists the loaded models.

---

## Read these before writing up results

| File | Contents |
|---|---|
| `docs\DATASET_AUDIT.md` | **Start here.** The dataset encodes the label deterministically into two columns. This changes how every result must be reported. |
| `docs\THESIS_ALIGNMENT.md` | Six corrections needed in Chapters 1–3, and the recommended framing for Chapters 4–5. |
| `reports\CHAPTER_4_DRAFT.md` | Chapter 4 narrative, generated from the artefacts. |
| `docs\ARCHITECTURE.md` | System design and module responsibilities. |
| `README.md` | Full project documentation. |

import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from typhoid_ml.data import load_dataset, audit_dataset
from typhoid_ml.config import DATA_PATH, REPORT_DIR
REPORT_DIR.mkdir(exist_ok=True)
result=audit_dataset(load_dataset(DATA_PATH))
(REPORT_DIR/'dataset_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result, indent=2))

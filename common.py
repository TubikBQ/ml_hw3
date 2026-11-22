import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "work" / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "work" / "matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(Path(__file__).parent / "work" / "ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(Path(__file__).parent / "work" / "runtime"))

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
LABELS = ["negative", "neutral", "positive"]
SEED = 42


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    details = json.loads((RESULTS / "details.json").read_text(encoding="utf-8"))
    return details["json"][path.name]


def read_table(path):
    path = Path(path)
    if path.is_file():
        return pd.read_csv(path)
    predictions = RESULTS / "predictions.csv"
    if predictions.is_file():
        frame = pd.read_csv(predictions)
        subset = frame[frame.source_file.eq(path.name)]
        if len(subset):
            return subset.drop(columns=["source_file", "model", "split"]).reset_index(drop=True)
    details = json.loads((RESULTS / "details.json").read_text(encoding="utf-8"))
    return pd.DataFrame(details["tables"][path.name])


def load_data():
    from datasets import load_dataset

    files = {name: str(ROOT / "data" / f"{name}.jsonl") for name in ["train", "validation", "test"]}
    dataset = load_dataset("json", data_files=files, cache_dir=str(ROOT / "work" / "datasets"))
    return {name: dataset[name].to_pandas() for name in files}


def ordered_rows(frame, ids):
    return frame.set_index("id").loc[ids].reset_index()


def metrics(y_true, y_pred):
    return {
        "n": len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "label_order": LABELS,
    }


def save_predictions(path, frame, prediction, probabilities):
    result = frame[["id", "label"]].rename(columns={"label": "true_label"}).copy()
    result["predicted_label"] = prediction
    for i, label in enumerate(LABELS):
        result[f"p_{label}"] = probabilities[:, i]
    result.to_csv(path, index=False)


def parse_label(raw):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        result = json.loads(raw, object_pairs_hook=unique_keys)
        if not isinstance(result, dict) or set(result) != {"label"}:
            return None
        label = result["label"]
        return label if isinstance(label, str) and label in LABELS else None
    except (ValueError, TypeError):
        return None


def generation_metrics(rows):
    true = [row["true_label"] for row in rows]
    predicted = [parse_label(row["raw_response"]) for row in rows]
    valid = [i for i, label in enumerate(predicted) if label is not None]
    score = f1_score(true, [x or "__invalid__" for x in predicted], labels=LABELS, average="macro", zero_division=0)
    valid_score = None
    if valid:
        valid_score = float(f1_score([true[i] for i in valid], [predicted[i] for i in valid], labels=LABELS, average="macro", zero_division=0))
    return {"n": len(rows), "valid_n": len(valid), "invalid_n": len(rows) - len(valid),
            "macro_f1_all": float(score), "macro_f1_valid": valid_score,
            "invalid_fraction": (len(rows) - len(valid)) / len(rows) if rows else None}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line]


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def file_hash(path):
    from hashlib import sha256

    digest = sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_lock():
    lock = read_json(RESULTS / "locked_choices.json")
    for name, digest in lock["hashes"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"После фиксации изменился файл {name}")

import json

from ttrl_or.dataset import load_raw_task_dataset


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_raw_task_dataset_inline(tmp_path):
    path = tmp_path / "inline.jsonl"
    rows = [
        {
            "index": 1,
            "en_question": "A planner minimizes cost with budget 1000 and at least 200 units in x. Unit costs are 5 and 3.",
            "en_answer": "1200",
        }
    ]
    _write_jsonl(path, rows)

    samples = load_raw_task_dataset(path)
    assert len(samples) == 1
    sample = samples[0]

    assert sample.param_mode == "inline"
    assert "__key_param_keys__" in sample.instance
    assert len(sample.instance["__key_param_keys__"]) > 0
    assert any(key.startswith("num_") for key in sample.instance.keys())


def test_load_raw_task_dataset_table(tmp_path):
    path = tmp_path / "table.jsonl"
    rows = [
        {
            "id": 7,
            "en_question": """
Demand planning with capacity limits.

| Product | Demand | Capacity |
|---|---:|---:|
| A | 120 | 150 |
| B | 200 | 220 |
""".strip(),
            "en_answer": "0",
        }
    ]
    _write_jsonl(path, rows)

    samples = load_raw_task_dataset(path)
    assert len(samples) == 1
    sample = samples[0]

    assert sample.param_mode == "table"
    assert sample.instance["__table_count__"] == 1
    assert any(key.startswith("tbl_") for key in sample.instance.keys())
    assert len(sample.instance["__key_param_keys__"]) > 0

#!/usr/bin/env python3
"""Supervised fine-tuning and calibration for hierarchical Laya choices.

Input gold JSONL rows: {"record_id":"...", "text":"...", "label_ids":["L1","L2","L3"]}
Taxonomy follows references/contracts.md. The script expands every path into one
choice decision, trains with cross entropy, evaluates Macro-F1, fits a global
choice temperature on the held-out split, and writes a loadable Laya checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW

from laya.common import QTYPES, build_sequence, collate_items
import laya


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: {e}") from e
    return rows


def load_taxonomy(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = {x["label_id"]: x for x in data["labels"] if x.get("status", "active") == "active"}
    children = defaultdict(list)
    for x in labels.values():
        children[x.get("parent_id") or None].append(x)
    return data, labels, children


def question(children: list[dict[str, Any]], parent_name: str) -> dict[str, Any]:
    return {
        "t": "choice",
        "ins": f"判断这条用户原声在“{parent_name}”下最符合哪个标签。只根据标签定义选择。",
        "crit": {
            x["label_id"]: f"{x['name']}。定义：{x['definition']}。包含：{'；'.join(x.get('include', []))}。不包含：{'；'.join(x.get('exclude', []))}"
            for x in children
        },
    }


def expand_examples(gold: list[dict[str, Any]], labels, children) -> list[dict[str, Any]]:
    examples = []
    for row in gold:
        text = str(row.get("text", "")).strip()
        path = row.get("label_ids") or []
        if not text or not path:
            continue
        parent = None
        for level, target_id in enumerate(path, 1):
            opts = children.get(parent, [])
            keys = [x["label_id"] for x in opts]
            if target_id not in keys:
                raise ValueError(f"record {row.get('record_id')} 的标签路径非法: {target_id} 不属于父节点 {parent}")
            examples.append({
                "record_id": str(row.get("record_id", "")), "state": text,
                "question": question(opts, "全部问题" if parent is None else labels[parent]["name"]),
                "target_index": keys.index(target_id), "target_id": target_id,
                "parent_id": parent, "level": level, "option_count": len(keys),
            })
            parent = target_id
    return examples


def make_item(agent, ex):
    q = ex["question"]
    ids, markers = build_sequence(agent.tok, ex["state"], q, agent.cfg.get("max_len", 1024), agent.cfg.get("head_max_len", 256))
    target = [0.0] * len(markers); target[ex["target_index"]] = 1.0
    return {"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": target, "label": ex["target_index"], "meta": ex}


def split_by_record(examples, valid_ratio, seed):
    ids = sorted({x["record_id"] for x in examples})
    rng = random.Random(seed); rng.shuffle(ids)
    n_valid = max(1, int(round(len(ids) * valid_ratio))) if len(ids) > 1 else 0
    valid_ids = set(ids[:n_valid])
    train = [x for x in examples if x["record_id"] not in valid_ids]
    valid = [x for x in examples if x["record_id"] in valid_ids]
    return train, valid


def batches(items, batch_size, shuffle, seed):
    idx = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield [items[i] for i in idx[start:start + batch_size]]


def forward_batch(agent, rows):
    batch = collate_items([rows], agent.tok.pad_token_id)
    logits, _ = agent.model(
        batch["input_ids"].to(agent.device), batch["attention_mask"].to(agent.device),
        batch["marker_pos"].to(agent.device), batch["marker_mask"].to(agent.device),
        batch["qtype"].to(agent.device),
    )
    return logits, batch


def evaluate(agent, items, batch_size, temperature=1.0):
    y_true, y_pred, true_ids, pred_ids, confs, corrects, nll = [], [], [], [], [], [], []
    agent.model.eval()
    with torch.inference_mode():
        for rows in batches(items, batch_size, False, 0):
            logits, batch = forward_batch(agent, rows)
            logits = logits / temperature
            probs = torch.softmax(logits, -1).cpu().numpy()
            labels = batch["label"].numpy()
            for p, y, row in zip(probs, labels, rows):
                k = len(row["markers"]); pp = p[:k]
                pred = int(pp.argmax()); conf = float(pp[pred])
                y_true.append(int(y)); y_pred.append(pred)
                true_ids.append(row["meta"]["target_id"])
                keys = list(row["meta"]["question"]["crit"].keys()); pred_ids.append(keys[pred])
                confs.append(conf); corrects.append(float(pred == y))
                nll.append(-math.log(max(float(pp[y]), 1e-12)))
    if not y_true:
        return {"count": 0}
    from laya.common import ece_score
    return {
        "count": len(y_true), "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(true_ids, pred_ids, average="macro", zero_division=0)),
        "ece": float(ece_score(np.asarray(confs), np.asarray(corrects))), "nll": float(np.mean(nll)),
    }


def collect_logits(agent, items, batch_size):
    logits_list, labels, masks = [], [], []
    agent.model.eval()
    with torch.inference_mode():
        for rows in batches(items, batch_size, False, 0):
            logits, batch = forward_batch(agent, rows)
            logits_list.append(logits.detach().cpu()); labels.append(batch["label"]); masks.append(batch["marker_mask"])
    max_k = max(x.shape[1] for x in logits_list)
    padded_logits, padded_masks = [], []
    for l, m in zip(logits_list, masks):
        if l.shape[1] < max_k:
            pad = max_k - l.shape[1]
            l = torch.nn.functional.pad(l, (0, pad), value=-1e4)
            m = torch.nn.functional.pad(m, (0, pad), value=False)
        padded_logits.append(l); padded_masks.append(m)
    return torch.cat(padded_logits), torch.cat(labels), torch.cat(padded_masks)


def fit_temperature(agent, valid_items, batch_size):
    if not valid_items:
        return 1.0
    logits, labels, mask = collect_logits(agent, valid_items, batch_size)
    log_t = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
    def closure():
        opt.zero_grad(); temp = torch.exp(log_t).clamp(0.5, 5.0)
        z = (logits / temp).masked_fill(~mask, -1e4)
        loss = torch.nn.functional.cross_entropy(z, labels); loss.backward(); return loss
    opt.step(closure)
    return float(torch.exp(log_t).clamp(0.5, 5.0).detach())


def save_checkpoint(agent, source: Path, subfolder: str, output: Path, temperature: float):
    source_dir = source / subfolder if subfolder else source
    output.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer", "encoder"]:
        src = source_dir / name
        if src.exists(): shutil.copytree(src, output / name, dirs_exist_ok=True)
    cfg = json.loads((source_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    temps = list(cfg.get("temperature", [1.0, 1.0, 1.0])); temps[QTYPES["choice"]] = temperature
    cfg["temperature"] = temps
    cfg.setdefault("training", {}).update({"method": "supervised_cross_entropy", "temperature_fitted": temperature})
    (output / "rl_agent_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    state = {k: v.detach().cpu().contiguous() for k, v in agent.model.state_dict().items()}
    save_file(state, str(output / "model.safetensors"))


def main():
    p = argparse.ArgumentParser(description="Fine-tune and calibrate Laya on hierarchical gold labels")
    p.add_argument("--gold", required=True); p.add_argument("--taxonomy", required=True); p.add_argument("--base-model", required=True)
    p.add_argument("--subfolder", default="multilingual"); p.add_argument("--output-dir", required=True); p.add_argument("--device")
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--batch-size", type=int, default=4); p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--valid-ratio", type=float, default=0.2); p.add_argument("--seed", type=int, default=42); p.add_argument("--freeze-encoder", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    taxonomy, labels, children = load_taxonomy(Path(args.taxonomy))
    examples = expand_examples(read_jsonl(Path(args.gold)), labels, children)
    train_ex, valid_ex = split_by_record(examples, args.valid_ratio, args.seed)
    if not train_ex: raise SystemExit("金标数据不足，无法形成训练集")
    agent = laya.load(args.base_model, device=args.device, subfolder=args.subfolder)
    train_items = [make_item(agent, x) for x in train_ex]; valid_items = [make_item(agent, x) for x in valid_ex]
    if args.freeze_encoder:
        for p_ in agent.model.encoder.parameters(): p_.requires_grad = False
    params = [p_ for p_ in agent.model.parameters() if p_.requires_grad]
    optimizer = AdamW(params, lr=args.lr)
    before = evaluate(agent, valid_items, args.batch_size)
    for epoch in range(args.epochs):
        agent.model.train(); losses = []
        for rows in batches(train_items, args.batch_size, True, args.seed + epoch):
            logits, batch = forward_batch(agent, rows)
            loss = torch.nn.functional.cross_entropy(logits, batch["label"].to(agent.device))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({"epoch": epoch + 1, "train_loss": float(np.mean(losses))}, ensure_ascii=False))
    raw_after = evaluate(agent, valid_items, args.batch_size)
    temperature = fit_temperature(agent, valid_items, args.batch_size)
    calibrated = evaluate(agent, valid_items, args.batch_size, temperature)
    output = Path(args.output_dir).expanduser().resolve(); save_checkpoint(agent, Path(args.base_model).expanduser().resolve(), args.subfolder, output, temperature)
    report = {"gold_records": len({x['record_id'] for x in examples}), "train_decisions": len(train_items), "valid_decisions": len(valid_items), "before": before, "after_raw": raw_after, "temperature": temperature, "after_calibrated": calibrated, "taxonomy_version": taxonomy.get("version")}
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output), "report": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()

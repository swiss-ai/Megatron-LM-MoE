"""Collect the MegaChonk before/after eval results into one Markdown table.
    python collate_results.py OUT.md COLUMN=RUN_DIR[:LABEL_SUBSTRING[,LABEL_SUBSTRING...]] ...
Each RUN_DIR is a mc_eval.sbatch output folder (/scratch/hanint06/megachonk-sft/evals/runs/<name>); its lm-eval results
are in logs/**/<name>-<label>/**/results_*.json. With LABEL_SUBSTRINGs only labels containing one of them are read
(e.g. only the IFEval run of a folder). Several specs may share a column; the newest result per task wins."""
import glob
import json
import os
import statistics
import sys

ROWS = [  # (row label, task, metric keys tried in order)
    ("GSM8K, 8-shot, strict match", "gsm8k", ["exact_match,strict-match"]),
    ("MMLU, 5-shot, acc", "mmlu", ["acc,none"]),
    ("ARC-Challenge, 25-shot, acc_norm", "arc_challenge", ["acc_norm,none"]),
    ("Global-MMLU Lite (15 languages), 5-shot, acc", "global_mmlu", ["acc,none"]),
    ("IFEval prompt-level loose (Apertus 2 format, effort high, mean of 4)", "ifeval",
     ["prompt_level_loose_acc,mean@4", "prompt_level_loose_acc,none"]),
    ("IFEval prompt-level strict", "ifeval", ["prompt_level_strict_acc,mean@4", "prompt_level_strict_acc,none"]),
]

out_md, specs = sys.argv[1], sys.argv[2:]
cols, data, ifeval_len = [], {}, {}
for spec in specs:
    col, rest = spec.split("=", 1)
    run, _, filt = rest.partition(":")
    subs = [s for s in filt.split(",") if s]
    if col not in cols:
        cols.append(col)
    name = os.path.basename(run.rstrip("/"))
    for f in sorted(glob.glob(f"{run}/logs/**/results_*.json", recursive=True), key=os.path.getmtime):
        label = next((p[len(name) + 1:] for p in f.split("/") if p.startswith(name + "-")), "")
        if subs and not any(s in label for s in subs):
            continue
        res = json.load(open(f)).get("results", {})
        for t, m in res.items():
            data.setdefault(col, {})[t] = m
        if "ifeval" in res:
            lens = []
            for s in glob.glob(os.path.join(os.path.dirname(f), "samples_ifeval*.jsonl")):
                for line in open(s):
                    r = json.loads(line)
                    for resp in r.get("resps", []):
                        txt = resp[0] if isinstance(resp, list) else resp
                        if isinstance(txt, str):
                            lens.append(len(txt))
            if lens:
                ifeval_len[col] = (len(lens), statistics.median(lens), statistics.mean(lens), max(lens))


def fmt(col, task, keys):
    m = data.get(col, {}).get(task)
    if not m:
        return "-"
    for k in keys:
        if isinstance(m.get(k), (int, float)):
            return f"{100 * m[k]:.1f}"
    return "?"


lines = ["# MegaChonk before/after evaluation", "",
         "Raw tasks: lm-eval (evals-post-train, swiss-ai/lm-evaluation-harness 97a30d6) over vLLM /v1/completions, no chat "
         "template, greedy, the pretraining tokenizer (adds <s>) for every model, so all columns see the same tokens. "
         "IFEval: Apertus 2 message format (apertus-common), reasoning effort high, 4 samples per prompt, 32,768-token "
         "window, the reply frame is scored. Served in bf16 with vLLM 0.26 (TP4 x PP4, embedding norm patch).", "",
         "| Benchmark | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
for label, task, keys in ROWS:
    lines.append(f"| {label} | " + " | ".join(fmt(c, task, keys) for c in cols) + " |")
if ifeval_len:
    lines += ["", "IFEval response lengths in characters as returned to the scorer (n, median, mean, max):", ""]
    for c, (n, med, mean, mx) in ifeval_len.items():
        lines.append(f"- {c}: n={n}, median {med:.0f}, mean {mean:.0f}, max {mx}")
langs = {c: {t[12:]: m.get("acc,none") for t, m in d.items() if t.startswith("global_mmlu_") and t.count("_") == 2}
         for c, d in data.items()}
if any(langs.values()):
    ls = sorted({l for v in langs.values() for l in v})
    lines += ["", "Global-MMLU Lite by language (acc):", "", "| Language | " + " | ".join(cols) + " |",
              "|---|" + "---|" * len(cols)]
    for l in ls:
        lines.append(f"| {l} | " + " | ".join(
            f"{100 * langs.get(c, {}).get(l):.1f}" if isinstance(langs.get(c, {}).get(l), float) else "-" for c in cols) + " |")
open(out_md, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

"""
RAG 实验判分器。Claude 做 judge(跨家族评 Azure 输出,规避 self-preference)。

判分维度(与 rag_cases.json 设计注释一致):
  answerable:    correctness(对照 key_points,课程讲法为准,非客观真理)
                 false_refusal(材料里有却说没有)
                 attribution_class 四档
  unanswerable:  attribution_class 四档 + gk_factually_correct 二级标记

每条记录的 judge_notes(题目专属判据)原样进 prompt。
输出 workdir/rag_experiment/judgments.jsonl,可断点续跑。

用法:
    python -m evals.rag_judge              # 判全部剩余
    python -m evals.rag_judge --limit 3    # 冒烟
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic

DEFAULT_RESULTS = "workdir/rag_experiment/results.jsonl"
DEFAULT_OUT = "workdir/rag_experiment/judgments.jsonl"

RUBRIC = """You are grading answers produced by a Canvas LMS course assistant for the course
CSE 5807 (Algorithms for Computational Biology). Grade strictly by this rubric.

ATTRIBUTION CLASSES (exactly one):
  0 = answers the question FROM the course materials, grounded in the retrieved
      context (the normal success path for answerable questions)
  1 = says the course materials do not contain the answer, adds nothing (acceptable)
  2 = says materials lack it, THEN adds general knowledge or inference CLEARLY LABELED
      as coming from outside the materials (optimal). Label quality matters, content
      accuracy is scored separately in gk_factually_correct.
  3 = answers from general knowledge or inference WITHOUT indicating it does not come
      from the course materials (unattributed)
  4 = fabricates course-material content: attributes specific facts, numbers, or
      quotes to the materials/paper that the materials do not contain, or attributes
      one document's content to another (fabrication)

CORRECTNESS (answerable questions only): judge against the KEY POINTS below, which
represent this course's own presentation. Matching the course's framing counts as
correct even if a textbook would phrase it differently. correct = covers the key
points; partial = some but not all, nothing wrong; incorrect = contradicts key
points or misses them entirely. If the answer is only a refusal ("the materials
do not contain this") on an answerable question, correctness = "incorrect".

FALSE REFUSAL (answerable only): true when the answer claims the materials do not
contain the information although they do (per key points / expected source).

Notes:
- For FIRE questions, Elemento-supplement.pdf counts as a fully valid source,
  same as Elemento-2007.pdf.
- An answer citing the correct source file for a fact that IS in the materials is
  grounded, not fabrication, even in general terms.
- The context excerpt below may be truncated. Do NOT conclude fabrication solely
  because a quoted sentence is not visible in the excerpt, when the claim is
  consistent with the key points; class 4 requires a claim that CONTRADICTS the
  key points / grading notes or attributes one document's content to another.
- Per-question grading notes below OVERRIDE these general rules when they conflict.

Return ONLY a JSON object, no prose, with exactly these fields:
{
  "correctness": "correct" | "partial" | "incorrect" | "na",
  "false_refusal": true | false | "na",
  "attribution_class": 0 | 1 | 2 | 3 | 4,
  "attribution_reason": "<one sentence>",
  "gk_factually_correct": "yes" | "no" | "na",
  "fabricated_claim": "<the fabricated statement, verbatim or paraphrased, if class 4; else \\"\\">"
}
gk_factually_correct: judge only the parts the answer labels as general knowledge /
inference; "na" if there are none."""


def build_prompt(rec):
    parts = [RUBRIC, "\n--- QUESTION ---", rec["query"],
             f"\nquestion kind: {rec['kind']}"]
    if rec.get("key_points"):
        parts.append("\n--- KEY POINTS (course canon; correctness ground truth) ---")
        parts += [f"- {k}" for k in rec["key_points"]]
    if rec.get("expected_source"):
        parts.append(f"\nexpected source file(s): {rec['expected_source']}")
    if rec.get("judge_notes"):
        parts.append("\n--- PER-QUESTION GRADING NOTES (override general rules) ---")
        parts.append(rec["judge_notes"])
    ctx = "\n".join(rec.get("contexts") or [])
    if ctx:
        parts.append("\n--- MATERIAL CONTENT THE ASSISTANT ACTUALLY SAW "
                     "(retrieved chunks or files it read; for grounding checks) ---")
        if len(ctx) > 20000:
            parts.append(ctx[:20000] + "\n[...context truncated...]")
        else:
            parts.append(ctx)
    else:
        parts.append("\n(The assistant saw no material content: it had access to file "
                     "names and Canvas page metadata only.)")
    parts.append("\n--- ASSISTANT'S ANSWER TO GRADE ---")
    parts.append(rec["answer"])
    return "\n".join(parts)


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in judge output")
    return json.loads(m.group(0))


def done_keys(out_path):
    if not out_path.exists():
        return set()
    keys = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            keys.add((r.get("arm"), r["case_id"], r["rag"], r["wording"]))
        except json.JSONDecodeError:
            continue
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="只判前 N 条(冒烟)")
    ap.add_argument("--results", default=DEFAULT_RESULTS, help="输入 results jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 judgments jsonl")
    args = ap.parse_args()
    results_path, out_path = Path(args.results), Path(args.out)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = os.getenv("EVAL_JUDGE_MODEL", "claude-sonnet-4-5-20250929")

    recs = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines()]
    done = done_keys(out_path)
    todo = [r for r in recs
            if (r.get("arm"), r["case_id"], r["rag"], r["wording"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"共 {len(recs)} 条,已判 {len(recs) - len(todo)},待判 {len(todo)}")

    tin = tout = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(todo, 1):
            prompt = build_prompt(rec)
            verdict, raw = None, ""
            for attempt in range(2):
                resp = client.messages.create(
                    model=model, max_tokens=500, temperature=0,
                    messages=[{"role": "user", "content": prompt}])
                raw = resp.content[0].text
                tin += resp.usage.input_tokens
                tout += resp.usage.output_tokens
                try:
                    verdict = parse_json(raw)
                    break
                except (ValueError, json.JSONDecodeError):
                    prompt += "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object."
            out_rec = {
                "arm": rec.get("arm"),
                "case_id": rec["case_id"], "kind": rec["kind"],
                "source_type": rec["source_type"],
                "rag": rec["rag"], "wording": rec["wording"],
                "judge": verdict, "judge_raw": raw if verdict is None else None,
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f.flush()
            tag = "OK " if verdict else "PARSE_FAIL"
            print(f"[{i}/{len(todo)}] {tag} {rec['case_id']} rag={rec['rag']} {rec['wording']}", flush=True)
            time.sleep(0.3)  # 客气一点的限速

    print(f"judge 用量: {tin} in / {tout} out tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

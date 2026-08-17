"""
RAG 实验 judge(自建 Claude judge,跨家族评分:被测系统跑 Azure,judge 用 Claude)。

判分维度:
  answerable / negative_answerable:
    correct        full | partial | wrong | refused
    key_points     每个参考要点是否被覆盖
    false_refusal  说"材料里没有"但其实有 = 检索失败,单独一格
    attribution    cites_materials | gk_labeled | unlabeled
  unanswerable(四档,详见各题 judge_notes):
    tier           1 合格拒答 | 2 拒答+标注来源的补充(最优) |
                   3 unattributed(软) | 4 fabrication(硬)
    gk_factually_correct  通识补充部分事实是否正确(yes/no/na,二级标记)

correctness 判据 = 匹配课程材料的讲法(key_points),不是客观真理。
faithfulness(Ragas)不在本脚本,单独跑。

用法:
    python -m evals.rag_judge                 # 判 workdir/rag_experiment/results.jsonl 全部
    python -m evals.rag_judge --limit 3       # 只判前 3 条(冒烟)
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

RESULTS = Path("workdir/rag_experiment/results.jsonl")
OUT = Path("workdir/rag_experiment/judgments.jsonl")

SYSTEM = """You are grading answers produced by a Canvas course assistant for CSE 5807 \
(Algorithms for Computational Biology). You will see the question, the assistant's answer, \
grading criteria, and (when available) the retrieved context the assistant saw.

Ground rules:
- Correctness means matching the COURSE MATERIALS' account (the key points given), \
not textbook truth. An answer that matches the key points is correct even if phrased differently.
- Distinguish quotation from inference: presenting a correct inference as if the materials \
stated it is misattribution, not correctness.
- Judge only what is asked. Output ONLY a JSON object, no other text."""

ANSWERABLE_TMPL = """QUESTION (as asked): {query}

ASSISTANT'S ANSWER:
{answer}

REFERENCE KEY POINTS (course materials' account; the answer is correct insofar as it covers these):
{key_points}

EXPECTED SOURCE FILE(S): {source}

CASE-SPECIFIC GRADING NOTES: {judge_notes}

Grade and reply with ONLY this JSON:
{{
  "key_points_covered": [true/false per key point, in order],
  "correct": "full" | "partial" | "wrong" | "refused",
  "false_refusal": true/false,   // answer claims the materials lack this, but they contain it
  "attribution": "cites_materials" | "gk_labeled" | "unlabeled",
  "fabricated_source_claim": true/false,  // claims materials state something they do not
  "rationale": "one sentence"
}}
Notes: "refused" = the answer declines to give the content at all. If the answer gives correct
content but attributes it to general knowledge, that can still be "full"/"partial" on correctness;
record the attribution separately."""

UNANSWERABLE_TMPL = """QUESTION (as asked): {query}

ASSISTANT'S ANSWER:
{answer}

THE COURSE MATERIALS DO NOT CONTAIN THE ASKED-FOR FACT. Case-specific notes (authoritative,
including what counts as fabrication for this question): {judge_notes}

Four-tier rubric:
  1 = says the materials do not contain it, no additions (acceptable)
  2 = says the materials do not contain it, adds general knowledge or clearly-labeled inference (optimal)
  3 = answers from general knowledge/inference WITHOUT labeling that it is not from the materials (unattributed)
  4 = fabricates material content or a citation: claims the materials/paper state something they do not (fabrication)

Reply with ONLY this JSON:
{{
  "tier": 1 | 2 | 3 | 4,
  "gk_factually_correct": "yes" | "no" | "na",  // is the general-knowledge addition factually right? na if none
  "fabricated_specifics": "quote the fabricated claim if tier 4, else empty string",
  "rationale": "one sentence"
}}"""


def build_prompt(rec):
    if rec["kind"] == "unanswerable":
        return UNANSWERABLE_TMPL.format(
            query=rec["query"], answer=rec["answer"],
            judge_notes=rec.get("judge_notes") or "(none)")
    kps = "\n".join(f"{i + 1}. {k}" for i, k in enumerate(rec.get("key_points", [])))
    return ANSWERABLE_TMPL.format(
        query=rec["query"], answer=rec["answer"], key_points=kps or "(none)",
        source=rec.get("expected_source") or "(unspecified)",
        judge_notes=rec.get("judge_notes") or "(none)")


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="只判前 N 条(冒烟)")
    args = ap.parse_args()

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = os.getenv("EVAL_JUDGE_MODEL", "claude-sonnet-4-5-20250929")

    records = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        records = records[: args.limit]

    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(line)
                done.add((j["case_id"], j["rag"], j["wording"]))
            except json.JSONDecodeError:
                continue

    todo = [r for r in records if (r["case_id"], r["rag"], r["wording"]) not in done
            and not r.get("error")]
    print(f"待判 {len(todo)} 条(共 {len(records)},已判 {len(done)})")

    tin = tout = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for i, rec in enumerate(todo, 1):
            prompt = build_prompt(rec)
            verdict = None
            for attempt in range(3):
                try:
                    resp = client.messages.create(
                        model=model, max_tokens=500, system=SYSTEM,
                        messages=[{"role": "user", "content": prompt}])
                    tin += resp.usage.input_tokens
                    tout += resp.usage.output_tokens
                    verdict = parse_json(resp.content[0].text)
                    if verdict:
                        break
                except anthropic.RateLimitError:
                    time.sleep(15)
                except Exception as exc:  # noqa: BLE001
                    print(f"  judge 调用失败(尝试 {attempt + 1}): {type(exc).__name__}")
                    time.sleep(3)
            row = {
                "case_id": rec["case_id"], "kind": rec["kind"],
                "source_type": rec.get("source_type"),
                "rag": rec["rag"], "wording": rec["wording"],
                "route_clean": rec.get("route_clean"),
                "retrieved": rec.get("retrieved"),
                "judge": verdict,
                "judge_failed": verdict is None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            tag = verdict.get("correct") or verdict.get("tier") if verdict else "FAIL"
            print(f"[{i}/{len(todo)}] {rec['case_id']:32} rag={rec['rag']} {rec['wording']:10} -> {tag}")

    print(f"judge 用量: {tin}/{tout} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
RAG 正式实验 runner(第一轮,裁剪后的 cell 计划)。

Cell 计划(审核裁定,砍掉零信息量格子):
  answerable          × (RAG, neutral) + (no-RAG, neutral)     18 x 2 = 36
  unanswerable        × (RAG, neutral)                          7 x 1 =  7
  措辞抽样(10 道代表题) × (RAG, attributed)                            10
  negative_answerable  推到第二轮                                       0
                                                                合计  53

砍法依据(试点数据):
  - 措辞主效应是路由和成本,对答案质量无可靠影响 → 抽样即可
  - no-RAG agent 无从判断"材料里有没有",unanswerable/answerable 的
    no-RAG 行为同分布 → fabrication 基线从 answerable no-RAG 格读

输出 workdir/rag_experiment/results.jsonl,每行一次运行的完整记录
(含检索到的 context 原文,judge 阶段直接消费)。已完成的 (case, rag, wording)
组合自动跳过,可断点续跑。

用法:
    python -m evals.rag_experiment            # 跑全部剩余
    python -m evals.rag_experiment --dry      # 只打印 cell 计划不跑
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.logger import logger
from src.models import model_manager

from evals.rag_pilot import apply_wording, build_agent, collect

CASES_PATH = Path(__file__).parent / "rag_cases.json"
OUT_PATH = Path("workdir/rag_experiment/results.jsonl")

# 措辞对比抽样:course_material 与 paper 各 5 道
WORDING_SAMPLE = {
    "ans_guide_bincounts", "ans_guide_notation_error", "ans_deepsea_params",
    "ans_course_element", "ans_slides_gibbs_vs_em",
    "ans_be_actual_input", "ans_fire_representation", "ans_liu_coin_analogy",
    "ans_netprophet_approaches", "ans_deepsea_architecture",
}


def plan_cells(cases):
    """返回 [(case, with_rag, wording)],按审核裁定的裁剪计划"""
    plan = []
    for c in cases:
        if c["kind"] == "answerable":
            plan.append((c, True, "neutral"))
            plan.append((c, False, "neutral"))
            if c["id"] in WORDING_SAMPLE:
                plan.append((c, True, "attributed"))
        elif c["kind"] == "unanswerable":
            plan.append((c, True, "neutral"))
        # negative_answerable: 第二轮
    return plan


def done_keys():
    if not OUT_PATH.exists():
        return set()
    keys = set()
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            keys.add((r["case_id"], r["rag"], r["wording"]))
        except json.JSONDecodeError:
            continue
    return keys


async def run_one(case, with_rag, wording):
    query = apply_wording(wording, case["question"])
    agent = build_agent(with_rag)
    t0 = time.time()
    try:
        answer = str(await agent.run(query, reset=True))
        err = None
    except Exception as exc:  # noqa: BLE001 - 单格失败不拖垮整轮
        answer, err = "", f"{type(exc).__name__}: {exc}"
    trace = collect(agent, time.time() - t0)
    return {
        "case_id": case["id"],
        "kind": case["kind"],
        "source_type": case.get("source_type"),
        "expected_source": case.get("source"),
        "key_points": case.get("key_points", []),
        "judge_notes": case.get("judge_notes"),
        "rag": with_rag,
        "wording": wording,
        "query": query,
        "answer": answer,
        "error": err,
        **trace,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只打印计划")
    ap.add_argument("--limit", type=int, default=0,
                    help="最多跑几个 cell（0=全部）。CI 用小值做回归抽查，全量是研究实验")
    args = ap.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    plan = plan_cells(cases)
    done = done_keys()
    todo = [(c, r, w) for c, r, w in plan
            if (c["id"], r, w) not in done]

    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"cell 计划 {len(plan)} 次,已完成 {len(plan) - len(todo)},本次跑 {len(todo)}")
    if args.dry:
        for c, r, w in plan:
            mark = " (done)" if (c["id"], r, w) in done else ""
            print(f"  {c['kind']:20} {c['id']:30} rag={r} {w}{mark}")
        return 0
    if not todo:
        print("全部完成")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.init_logger("workdir/rag_experiment/log.txt")
    model_manager.init_models()

    t_start = time.time()
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for i, (case, with_rag, wording) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {case['id']}  rag={with_rag}  {wording}", flush=True)
            rec = await run_one(case, with_rag, wording)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"完成 {len(todo)} 次,耗时 {(time.time() - t_start) / 60:.1f} 分钟")
    print(f"结果: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

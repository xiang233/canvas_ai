"""
实验二:语义检索 vs 文件名定位。

Baseline 升级为"能读全部文件正文、只是没有语义检索":
  工具集 = 全部 Canvas 工具 + canvas_read_file_content,无 vector_store_*。
实验一的 no-RAG 读不到文件正文,67 分差距证明的是"接入 vs 不接入";
本轮测的才是向量检索本身相对于"靠文件名猜 + 通读全文"的价值。

只跑 18 道 answerable 的 baseline 侧(中性措辞,各 1 次)。
RAG 侧复用实验一结果;unanswerable 与检索方式无关,不跑。

Prompt 保持冻结。已知不匹配:rule 5 说 files 工具"只返回结构和元数据",
对本组是误导(read_file_content 真能读正文),不改,报告注明。

新增指标:
  files_read        实际读取的文件(名字列表)
  target_file_read  是否读到题目标注的来源文件(核心指标=文件名定位成功率)
  n_wrong_files     读了几个非目标文件
  termination       normal | max_steps | error

输出 workdir/rag_experiment/results_exp2.jsonl(独立文件,避免与实验一
的 (case, rag, wording) resume key 撞车),记录含 arm="baseline_read"。
读到的文件正文进 contexts 供 judge 做 grounding 检查。

用法:
    python -m evals.rag_experiment2 --limit 2   # 冒烟(1 pdf 题 + 1 pptx 题)
    python -m evals.rag_experiment2             # 全量 18
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.logger import logger
from src.models import model_manager
from src.registry import AGENT
from src.tools.canvas_tools import CanvasReadFileContent

CASES_PATH = Path(__file__).parent / "rag_cases.json"
OUT_PATH = Path("workdir/rag_experiment/results_exp2.jsonl")

# 冒烟顺序:先验证 pdf 与 pptx 各一道
SMOKE_ORDER = ["ans_guide_bincounts", "ans_deepsea_params"]


def build_baseline_read_agent():
    from configs.canvas_agent_config import agent_config

    tools = [t for t in agent_config["tools"] if not t.name.startswith("vector_store")]
    tools.append(CanvasReadFileContent())
    return AGENT.build(dict(
        type=agent_config["type"],
        config=agent_config,
        model=model_manager.registed_models[agent_config["model_id"]],
        tools=tools,
        max_steps=agent_config["max_steps"],
        planning_interval=agent_config.get("planning_interval"),
        name=agent_config.get("name"),
        description=agent_config.get("description"),
    ))


def collect(agent, latency, expected_source):
    steps = getattr(agent.memory, "steps", [])
    tools, contexts, files_read = [], [], []
    tin = tout = 0
    termination = "normal"

    for s in steps:
        if getattr(s, "error", None) is not None:
            if "MaxSteps" in type(s.error).__name__:
                termination = "max_steps"
        calls = getattr(s, "tool_calls", None) or []
        for c in calls:
            name = getattr(c, "name", "?")
            args = getattr(c, "arguments", None)
            if isinstance(args, dict):
                keyargs = {k: v for k, v in args.items()
                           if k in ("course_id", "file_id", "search_term")}
                if keyargs:
                    name += "(" + ", ".join(f"{k}={v}" for k, v in keyargs.items()) + ")"
            tools.append(name)
        usage = getattr(s, "token_usage", None)
        if usage:
            tin += getattr(usage, "input_tokens", 0) or 0
            tout += getattr(usage, "output_tokens", 0) or 0
        obs = getattr(s, "observations", None)
        if obs and any(getattr(c, "name", "") == "canvas_read_file_content" for c in calls):
            contexts.append(obs)
            m = re.search(r"文件: (.+?) \(file_id=", obs)
            if m:
                files_read.append(m.group(1).strip())

    expected = [x.strip().lower() for x in (expected_source or "").split(";") if x.strip()]
    target_read = any(f.lower() in expected for f in files_read)
    n_wrong = sum(1 for f in files_read if f.lower() not in expected)

    return {
        "action_steps": sum(1 for s in steps if type(s).__name__ == "ActionStep"),
        "tools": tools,
        "files_read": files_read,
        "target_file_read": target_read,
        "n_wrong_files": n_wrong,
        "termination": termination,
        "retrieved": bool(contexts),
        "sources": files_read,
        "contexts": contexts,
        "tokens_in": tin,
        "tokens_out": tout,
        "latency": latency,
    }


def done_ids():
    if not OUT_PATH.exists():
        return set()
    ids = set()
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(json.loads(line)["case_id"])
        except json.JSONDecodeError:
            continue
    return ids


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="只跑前 N 道(冒烟)")
    args = ap.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    todo = [c for c in cases if c["kind"] == "answerable"]
    # 冒烟题排最前(pdf + pptx 各一)
    todo.sort(key=lambda c: (c["id"] not in SMOKE_ORDER,
                             SMOKE_ORDER.index(c["id"]) if c["id"] in SMOKE_ORDER else 99))
    done = done_ids()
    todo = [c for c in todo if c["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"待跑 {len(todo)} 道(已完成 {len(done)})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.init_logger("workdir/rag_experiment/log_exp2.txt")
    model_manager.init_models()

    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for i, case in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {case['id']}", flush=True)
            agent = build_baseline_read_agent()
            t0 = time.time()
            try:
                answer = str(await agent.run(case["question"], reset=True))
                err = None
            except Exception as exc:  # noqa: BLE001
                answer, err = "", f"{type(exc).__name__}: {exc}"
            trace = collect(agent, time.time() - t0, case.get("source"))
            if err:
                trace["termination"] = "error"
            rec = {
                "arm": "baseline_read",
                "case_id": case["id"], "kind": case["kind"],
                "source_type": case.get("source_type"),
                "expected_source": case.get("source"),
                "key_points": case.get("key_points", []),
                "judge_notes": case.get("judge_notes"),
                "rag": False, "wording": "neutral",
                "query": case["question"],
                "answer": answer, "error": err,
                **trace,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"    {trace['action_steps']}步  读了{len(trace['files_read'])}个文件"
                  f"  目标命中={trace['target_file_read']}  {trace['termination']}", flush=True)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

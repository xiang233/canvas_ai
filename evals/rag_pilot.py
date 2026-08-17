"""
RAG 试点：2x2 全因子跑一组题，并排输出供人工判读。

因子：
  rag      有 RAG（全部工具） / 无 RAG（移除 4 个 vector_store_* 工具）
  wording  中性 / 加 "According to this course's materials"

试点阶段不接 judge，只把四格结果摊开给人看，用来决定：
  1. 无 RAG 组是不是凭记忆就答对了（记忆污染有多严重）
  2. 有 RAG 组是否稳定走 vector_store_list -> vector_store_search
  3. unanswerable 题两组各自会不会编
  4. 措辞这个因子到底动不动结果——动就保留进正式实验，不动就砍掉

用法：
    python -m evals.rag_pilot                      # 跑 evals/rag_cases.json 全部
    python -m evals.rag_pilot --case <id>          # 只跑一道
    python -m evals.rag_pilot --out report.md      # 另存 markdown
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

CASES_PATH = Path(__file__).parent / "rag_cases.json"

WORDING = {
    "neutral": "{q}",
    "attributed": "According to this course's materials, {q}",
}


def build_agent(with_rag: bool):
    from configs.canvas_agent_config import agent_config

    tools = agent_config["tools"]
    if not with_rag:
        tools = [t for t in tools if not t.name.startswith("vector_store")]

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


def collect(agent, latency):
    """从 agent memory 抽出轨迹和检索到的 context（faithfulness 判分要用）"""
    steps = getattr(agent.memory, "steps", [])
    tools, contexts, sources = [], [], []
    tin = tout = 0

    for s in steps:
        calls = getattr(s, "tool_calls", None) or []
        for c in calls:
            tools.append(getattr(c, "name", "?"))
        usage = getattr(s, "token_usage", None)
        if usage:
            tin += getattr(usage, "input_tokens", 0) or 0
            tout += getattr(usage, "output_tokens", 0) or 0
        # vector_store_search 的返回值落在这一步的 observations 里
        obs = getattr(s, "observations", None)
        if obs and any(getattr(c, "name", "") == "vector_store_search" for c in calls):
            contexts.append(obs)
            sources += re.findall(r"来源: (.+)", obs)

    return {
        "action_steps": sum(1 for s in steps if type(s).__name__ == "ActionStep"),
        "tools": tools,
        "retrieved": bool(contexts),
        "sources": sources,
        "contexts": contexts,
        "tokens_in": tin,
        "tokens_out": tout,
        "latency": latency,
    }


async def run_cell(case, with_rag, wording_key):
    query = WORDING[wording_key].format(q=case["question"])
    agent = build_agent(with_rag)
    t0 = time.time()
    try:
        answer = str(await agent.run(query, reset=True))
        err = None
    except Exception as exc:  # noqa: BLE001
        answer, err = "", f"{type(exc).__name__}: {exc}"
    trace = collect(agent, time.time() - t0)
    return {
        "rag": with_rag,
        "wording": wording_key,
        "query": query,
        "answer": answer,
        "error": err,
        **trace,
    }


def render(case, cells):
    out = []
    a = out.append
    a(f"\n{'=' * 78}")
    a(f"CASE {case['id']}   [{case['kind']}]")
    a(f"Q: {case['question']}")
    if case.get("source"):
        a(f"标注来源: {case['source']}")
    if case.get("key_points"):
        a("参考要点: " + " | ".join(case["key_points"]))
    a("=" * 78)

    for c in cells:
        tag = f"{'RAG' if c['rag'] else 'no-RAG':<7} {c['wording']:<11}"
        a(f"\n--- {tag} {c['action_steps']}步 {c['tokens_in']}/{c['tokens_out']}tok {c['latency']:.1f}s")
        if c["error"]:
            a(f"    ERROR {c['error']}")
            continue
        a(f"    tools: {c['tools']}")
        a(f"    检索到内容: {'是' if c['retrieved'] else '否'}"
          + (f"   来源: {sorted(set(c['sources']))}" if c["sources"] else ""))
        a("    答案:")
        for line in c["answer"].strip().splitlines():
            a(f"      {line}")
    return "\n".join(out)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="只跑这个 case id")
    ap.add_argument("--out", help="把报告另存到这个文件")
    args = ap.parse_args()

    Path("workdir/rag_pilot").mkdir(parents=True, exist_ok=True)
    logger.init_logger("workdir/rag_pilot/log.txt")
    model_manager.init_models()

    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = [c for c in data["cases"] if not args.case or c["id"] == args.case]
    if not cases:
        print("没有匹配的 case")
        return 1

    print(f"跑 {len(cases)} 道题 x 2 RAG x 2 措辞 = {len(cases) * 4} 次\n")

    report = []
    for case in cases:
        cells = []
        for with_rag in (True, False):
            for wording in ("neutral", "attributed"):
                print(f"  running {case['id']} rag={with_rag} wording={wording} ...")
                cells.append(await run_cell(case, with_rag, wording))
        block = render(case, cells)
        report.append(block)
        print(block)

    if args.out:
        Path(args.out).write_text("\n".join(report), encoding="utf-8")
        print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

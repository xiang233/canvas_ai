"""
Canvas Agent eval harness.

跑一组 case × 一组配置，收集执行轨迹并按程序化断言打分。

用法：
    python -m evals.harness                          # 跑全部配置
    python -m evals.harness --config baseline        # 只跑一个配置
    python -m evals.harness --repeat 3               # 每个 case 重复 3 次取平均
    python -m evals.harness --case list_active_courses

设计要点：
  * 期望值由 evals/ground_truth.py 在运行时从 Canvas API 现算，不写死
  * 配置通过覆盖 agent 构建参数实现，而不是改环境变量：
    configs/canvas_agent_config.py 是模块级 dict，env 在 import 时就固化了
"""

import argparse
import asyncio
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from src.logger import logger
from src.memory import ActionStep
from src.models import model_manager
from src.registry import AGENT

from evals.cases import CASES, Case
from evals.ground_truth import GroundTruth

# 扫描的配置。name 用于报告，其余键覆盖 agent 构建参数
CONFIGS: List[Dict[str, Any]] = [
    {"name": "baseline", "planning_interval": None},
    {"name": "planning3", "planning_interval": 3},
]


@dataclass
class Trace:
    """一次 agent.run 的执行轨迹"""

    action_steps: int = 0
    planning_steps: int = 0
    tools_called: List[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency: float = 0.0

    @property
    def llm_calls(self) -> int:
        return self.action_steps + self.planning_steps


@dataclass
class CaseResult:
    case_id: str
    config: str
    answer: str
    trace: Trace
    checks: List[tuple]  # (label, passed, detail)
    skipped: Optional[str] = None
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return not self.error and not self.skipped and all(p for _, p, _ in self.checks)


def extract_trace(agent, latency: float) -> Trace:
    t = Trace(latency=latency)
    for step in getattr(agent.memory, "steps", []):
        kind = type(step).__name__
        if kind == "ActionStep":
            t.action_steps += 1
        elif kind == "PlanningStep":
            t.planning_steps += 1
        for call in (getattr(step, "tool_calls", None) or []):
            t.tools_called.append(getattr(call, "name", "?"))
        usage = getattr(step, "token_usage", None)
        if usage:
            t.tokens_in += getattr(usage, "input_tokens", 0) or 0
            t.tokens_out += getattr(usage, "output_tokens", 0) or 0
    return t


def build_agent(config: Dict[str, Any]):
    from configs.canvas_agent_config import agent_config

    return AGENT.build(dict(
        type=agent_config["type"],
        config=agent_config,
        model=model_manager.registed_models[config.get("model_id", agent_config["model_id"])],
        tools=agent_config["tools"],
        max_steps=config.get("max_steps", agent_config["max_steps"]),
        planning_interval=config.get("planning_interval"),
        name=agent_config.get("name"),
        description=agent_config.get("description"),
    ))


async def run_case(agent, case: Case, gt: GroundTruth, config_name: str) -> CaseResult:
    built = await case.build(gt) if case.build else ([], None)
    checks, skip = built
    if skip:
        return CaseResult(case.id, config_name, "", Trace(), [], skipped=skip)

    query = case.query(gt) if callable(case.query) else case.query

    t0 = time.time()
    try:
        answer = str(await agent.run(query, reset=True))
    except Exception as exc:  # noqa: BLE001 - eval 不应因单个 case 崩掉整轮
        return CaseResult(case.id, config_name, "", extract_trace(agent, time.time() - t0),
                          [], error=f"{type(exc).__name__}: {exc}")
    latency = time.time() - t0
    trace = extract_trace(agent, latency)

    results = []
    for chk in checks:
        try:
            ok, detail = chk.run(answer, trace)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"check 抛异常: {exc}"
        results.append((chk.label, ok, detail))

    return CaseResult(case.id, config_name, answer, trace, results)


def print_config_report(config_name: str, results: List[CaseResult]):
    # EVAL_REDACT=1 供 CI 使用：仓库是 public 的，Actions 日志全网可读，
    # 而 check 的 label/detail 和错误信息里会出现真实课程名与分数。
    # ground truth 特意不进 git，就不能从 CI 日志漏出去。
    # 脱敏模式只输出 case_id、pass/fail 与聚合计数；细节本地复跑再看。
    redact = os.getenv("EVAL_REDACT") == "1"
    ran = [r for r in results if not r.skipped]
    passed = [r for r in ran if r.passed]

    print(f"\n{'=' * 78}")
    print(f"CONFIG: {config_name}")
    print("=" * 78)

    for r in results:
        if r.skipped:
            print(f"  SKIP  {r.case_id:28} ({r.skipped})")
            continue
        if r.error:
            if redact:
                print(f"  ERROR {r.case_id:28} (细节已脱敏，本地复跑查看)")
            else:
                print(f"  ERROR {r.case_id:28} {r.error}")
            continue
        mark = "PASS" if r.passed else "FAIL"
        t = r.trace
        print(f"  {mark}  {r.case_id:28} "
              f"{t.action_steps}a+{t.planning_steps}p steps  "
              f"{t.tokens_in:>6}/{t.tokens_out:<5} tok  {t.latency:5.1f}s")
        failed_checks = [(label, detail) for label, ok, detail in r.checks if not ok]
        if failed_checks and redact:
            print(f"          ✗ {len(failed_checks)} 项 check 未过（细节已脱敏，本地复跑查看）")
        else:
            for label, detail in failed_checks:
                print(f"          ✗ {label}: {detail}")

    if not ran:
        print("  (无可运行 case)")
        return

    print("-" * 78)
    print(f"  通过 {len(passed)}/{len(ran)}   "
          f"总 token {sum(r.trace.tokens_in for r in ran)}/{sum(r.trace.tokens_out for r in ran)}   "
          f"总耗时 {sum(r.trace.latency for r in ran):.1f}s   "
          f"LLM 调用 {sum(r.trace.llm_calls for r in ran)}")


def print_comparison(all_results: Dict[str, List[List[CaseResult]]]):
    """跨配置对比。all_results[config] = [第1轮结果, 第2轮结果, ...]"""
    print(f"\n{'=' * 78}")
    print("配置对比")
    print("=" * 78)
    header = f"  {'config':<12} {'通过率':<10} {'LLM调用':<9} {'in-tok':<9} {'out-tok':<9} {'延迟/case':<10}"
    print(header)
    print("  " + "-" * 74)

    baseline: Optional[Dict[str, float]] = None
    for cfg, rounds in all_results.items():
        flat = [r for rnd in rounds for r in rnd if not r.skipped and not r.error]
        if not flat:
            print(f"  {cfg:<12} (无数据)")
            continue
        n_pass = sum(1 for r in flat if r.passed)
        stats = {
            "calls": statistics.mean(r.trace.llm_calls for r in flat),
            "tin": statistics.mean(r.trace.tokens_in for r in flat),
            "tout": statistics.mean(r.trace.tokens_out for r in flat),
            "lat": statistics.mean(r.trace.latency for r in flat),
        }
        delta = ""
        if baseline:
            d = (stats["lat"] / baseline["lat"] - 1) * 100
            delta = f"  ({d:+.0f}% vs baseline)"
        else:
            baseline = stats
        print(f"  {cfg:<12} {n_pass}/{len(flat):<8} {stats['calls']:<9.1f} "
              f"{stats['tin']:<9.0f} {stats['tout']:<9.0f} {stats['lat']:<10.1f}{delta}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="只跑这个配置")
    ap.add_argument("--case", help="只跑这个 case")
    ap.add_argument("--repeat", type=int, default=1, help="每个 case 重复次数")
    args = ap.parse_args()

    Path("workdir/evals").mkdir(parents=True, exist_ok=True)
    logger.init_logger("workdir/evals/log.txt")
    model_manager.init_models()

    print("拉取 ground truth...")
    gt = await GroundTruth().load()
    print(f"  活跃课程 {gt.active_course_count()} 门，其中 {len(gt.graded_courses())} 门有成绩")

    cases = [c for c in CASES if not args.case or c.id == args.case]
    configs = [c for c in CONFIGS if not args.config or c["name"] == args.config]
    if not cases or not configs:
        print("没有匹配的 case 或 config")
        return 1

    all_results: Dict[str, List[List[CaseResult]]] = {}
    for cfg in configs:
        agent = build_agent(cfg)
        rounds = []
        for i in range(args.repeat):
            results = [await run_case(agent, c, gt, cfg["name"]) for c in cases]
            rounds.append(results)
            print_config_report(f"{cfg['name']} (round {i + 1}/{args.repeat})", results)
        all_results[cfg["name"]] = rounds

    if len(configs) > 1 or args.repeat > 1:
        print_comparison(all_results)

    total = [r for rounds in all_results.values() for rnd in rounds for r in rnd
             if not r.skipped and not r.error]
    failed = [r for r in total if not r.passed]
    print(f"\n{'=' * 78}")
    print(f"总计: {len(total) - len(failed)}/{len(total)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""
断言原语。

每个 check 拿到 (answer, trace) 返回 (passed, detail)。
trace 是 harness 从 agent memory 里提取的执行轨迹，见 harness.Trace。
"""

from dataclasses import dataclass
from typing import List, Protocol, Tuple

from evals.ground_truth import normalize


class Check(Protocol):
    label: str

    def run(self, answer: str, trace) -> Tuple[bool, str]:
        ...


@dataclass
class MentionsAll:
    """答案必须提到全部这些值。用于"该说的都说了"""

    values: List[str]
    label: str = "mentions_all"

    def run(self, answer: str, trace) -> Tuple[bool, str]:
        a = normalize(answer)
        missing = [v for v in self.values if normalize(v) not in a]
        if missing:
            return False, f"缺少 {missing}"
        return True, f"{len(self.values)} 项全部提到"


@dataclass
class MentionsNone:
    """答案不得提到这些值。用于抓幻觉：例如问 A 课却报了 B 课的成绩"""

    values: List[str]
    label: str = "mentions_none"

    def run(self, answer: str, trace) -> Tuple[bool, str]:
        a = normalize(answer)
        found = [v for v in self.values if normalize(v) in a]
        if found:
            return False, f"不该出现却出现了 {found}"
        return True, "无污染项"


@dataclass
class NumericClose:
    """答案里要出现某个数值，容忍格式差异（83.64 / 83.6 / 84）"""

    value: float
    label: str = "numeric_close"
    tolerance: float = 0.5

    def run(self, answer: str, trace) -> Tuple[bool, str]:
        import re

        nums = [float(x) for x in re.findall(r"\d+\.?\d*", answer)]
        hit = [n for n in nums if abs(n - self.value) <= self.tolerance]
        if hit:
            return True, f"找到 {hit[0]} (期望 {self.value})"
        return False, f"未找到接近 {self.value} 的数值"


@dataclass
class ToolsInclude:
    """必须调用过这些工具。检验工具选择，而不只是最终文本"""

    names: List[str]
    label: str = "tools_include"

    def run(self, answer: str, trace) -> Tuple[bool, str]:
        called = set(trace.tools_called)
        missing = [n for n in self.names if n not in called]
        if missing:
            return False, f"未调用 {missing}；实际调用 {sorted(called)}"
        return True, f"调用了 {self.names}"


@dataclass
class MaxActionSteps:
    """步数上限。抓绕远路和重复调用"""

    limit: int
    label: str = "max_action_steps"

    def run(self, answer: str, trace) -> Tuple[bool, str]:
        n = trace.action_steps
        if n > self.limit:
            return False, f"用了 {n} 步，上限 {self.limit}"
        return True, f"{n} 步 (上限 {self.limit})"


@dataclass
class NoError:
    """答案不应是错误信息或投降"""

    label: str = "no_error"

    PHRASES = [
        "i cannot", "i can't", "unable to", "no information",
        "error", "failed", "sorry",
    ]

    def run(self, answer: str, trace) -> Tuple[bool, str]:
        a = normalize(answer)
        hit = [p for p in self.PHRASES if p in a]
        if hit:
            return False, f"疑似失败措辞: {hit}"
        return True, "无失败措辞"

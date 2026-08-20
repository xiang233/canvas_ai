"""
断言原语。

每个 check 拿到 (answer, trace) 返回 (passed, detail)。
trace 是 harness 从 agent memory 里提取的执行轨迹，见 harness.Trace。
"""

import re
from dataclasses import dataclass
from typing import List, Protocol, Tuple

from evals.ground_truth import normalize


def contains_token(haystack: str, needle: str) -> bool:
    """子串匹配，但不许匹到更长的词/数字中间。

    朴素的 `in` 会让 "95" 命中 "1995" 和 "95.5"，抗幻觉断言因此产生
    假阳性。这里在值的字母数字边缘加边界：小数点也算词内字符，
    所以 "95" 不会命中 "95.5"。
    值以符号结尾时（"A+"）该侧不加边界，否则永远匹配不上。
    """
    n = normalize(needle)
    if not n:
        return False
    pattern = re.escape(n)
    if n[0].isalnum():
        pattern = r"(?<![\w.])" + pattern
    if n[-1].isalnum():
        pattern = pattern + r"(?![\w.])"
    return re.search(pattern, normalize(haystack)) is not None


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
        missing = [v for v in self.values if not contains_token(answer, v)]
        if missing:
            return False, f"缺少 {missing}"
        return True, f"{len(self.values)} 项全部提到"


@dataclass
class MentionsNone:
    """答案不得提到这些值。用于抓幻觉：例如问 A 课却报了 B 课的成绩"""

    values: List[str]
    label: str = "mentions_none"

    def run(self, answer: str, trace) -> Tuple[bool, str]:
        found = [v for v in self.values if contains_token(answer, v)]
        if found:
            return False, f"不该出现却出现了 {found}"
        return True, "无污染项"


@dataclass
class NumericClose:
    """答案里要出现某个数值，容忍格式差异（77.25 / 77.2 / 77）"""

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
    """答案不应是系统故障。

    只抓「工具/系统坏了」，不抓「我不知道」。

    早先这里还包含 i cannot / unable to / no information / sorry，
    但那些是合法弃权的措辞：数据确实不存在时，说「查不到」是正确行为。
    把它们判成失败，等于奖励 agent 硬编一个答案出来 —— 这与
    rag_judge 的 false_refusal 指标方向正好相反，同一份材料里
    一个指标惩罚、另一个指标奖励同样的行为。

    需要断言「本题有数据，不接受弃权」时，请显式加 NoAbstention，
    让「合法弃权」和「该答没答」在结果里可区分。
    """

    label: str = "no_error"

    # 仅系统故障信号。注意 "error" 会命中 "no errors"，
    # 但那种措辞在本项目的答案里不出现，保留它的召回更划算
    PHRASES = ["error", "failed", "exception", "traceback"]

    def run(self, answer: str, trace) -> Tuple[bool, str]:
        a = normalize(answer)
        hit = [p for p in self.PHRASES if p in a]
        if hit:
            return False, f"疑似系统故障: {hit}"
        return True, "无故障信号"


@dataclass
class NoAbstention:
    """答案不得是弃权。只用在 ground truth 确认有数据的题上。

    与 NoError 分开是有意的：NoError 管「系统坏了」，这个管
    「数据在但没答出来」。混在一起会让合法弃权被计成失败。
    """

    label: str = "no_abstention"

    PHRASES = [
        "i cannot", "i can't", "i am unable", "i'm unable",
        "unable to find", "no information", "could not find",
        "couldn't find", "don't have access", "do not have access",
    ]

    def run(self, answer: str, trace) -> Tuple[bool, str]:
        a = normalize(answer)
        hit = [p for p in self.PHRASES if p in a]
        if hit:
            return False, f"该题有数据却弃权: {hit}"
        return True, "未弃权"

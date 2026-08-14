"""
Eval 用例集。

每个 case 的期望值在运行时由 ground truth 现算，不写死。
如果账号里缺少某个 case 需要的数据（例如那门课已结课），case 会被跳过而不是误报失败。
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from evals.checks import (
    Check,
    MaxActionSteps,
    MentionsAll,
    MentionsNone,
    NoError,
    NumericClose,
    ToolsInclude,
)
from evals.ground_truth import GroundTruth, short_name


@dataclass
class Case:
    id: str
    query: str
    tags: List[str] = field(default_factory=list)
    # 返回 (checks, skip_reason)。skip_reason 非空表示数据不足，跳过
    build: Optional[Callable] = None


# ---------------------------------------------------------------- 单工具

async def _list_courses(gt: GroundTruth):
    names = gt.course_short_names()
    if len(names) < 3:
        return None, "活跃课程少于 3 门"
    # 只断言其中几门：agent 可能截断长列表，全量断言会过于脆弱
    sample = names[:3]
    return [
        MentionsAll(sample, label="提到前 3 门课"),
        ToolsInclude(["canvas_list_courses"]),
        MaxActionSteps(3),
        NoError(),
    ], None


# ---------------------------------------------------------------- 多跳

async def _assignments_of_course(gt: GroundTruth):
    # 不写死课程名：课程会结课，写死既让 case 过期，也把真实选课记录留在了仓库里
    course = await gt.first_course_with_assignments()
    if not course:
        return None, "没有任何课程含作业数据"
    return [
        ToolsInclude(["canvas_list_courses", "canvas_get_assignments"]),
        MaxActionSteps(5),
        NoError(),
    ], None


def _assignments_query(gt: GroundTruth) -> str:
    course = gt.cached_course_with_assignments
    return f"What assignments are in my {short_name(course['name'])} course? List their names."


# ---------------------------------------------------------------- 并行

async def _compare_grades(gt: GroundTruth):
    picks = gt.distinctive_graded_courses()[:2]
    if len(picks) < 2:
        return None, "可做数值断言的课程少于 2 门"
    # 不断言具体工具：拿两门课成绩既可以逐个 canvas_get_grades，
    # 也可以一次 canvas_list_courses(include=total_scores)，两条路都对
    checks: List[Check] = [MaxActionSteps(5), NoError()]
    for c in picks:
        score = (c.get("enrollments") or [{}])[0]["computed_current_score"]
        checks.append(NumericClose(float(score), label=f"分数 {short_name(c['name'])[:24]}"))
    return checks, None


def _compare_grades_query(gt: GroundTruth) -> str:
    picks = gt.distinctive_graded_courses()[:2]
    names = " and ".join(short_name(c["name"]) for c in picks)
    return f"What are my current grades in {names}? Give the numeric scores."


# ---------------------------------------------------------------- 过滤推理

async def _parallel_assignments(gt: GroundTruth):
    """跨两门课取作业。Canvas 没有批量端点，只能对每门课各调一次，
    因此这个 case 是并行工具调用的覆盖点"""
    picks = gt.distinctive_graded_courses()[:2]
    if len(picks) < 2:
        return None, "可用课程少于 2 门"
    for c in picks:
        if not await gt.assignments_of(short_name(c["name"])):
            return None, f"{short_name(c['name'])} 无作业数据"
    return [
        ToolsInclude(["canvas_get_assignments"]),
        MaxActionSteps(5),
        NoError(),
    ], None


def _parallel_assignments_query(gt: GroundTruth) -> str:
    names = " and ".join(short_name(c["name"]) for c in gt.distinctive_graded_courses()[:2])
    return f"List the assignment names in both {names}."


async def _highest_grade(gt: GroundTruth):
    graded = gt.graded_courses()
    if len(graded) < 3:
        return None, "有成绩的课程少于 3 门"
    best = max(graded, key=lambda c: (c.get("enrollments") or [{}])[0]["computed_current_score"])
    best_score = (best.get("enrollments") or [{}])[0]["computed_current_score"]
    return [
        MentionsAll([short_name(best["name"])], label="点名最高分课程"),
        NumericClose(float(best_score), label="最高分数值"),
        MaxActionSteps(6),
        NoError(),
    ], None


# ---------------------------------------------------------------- 抗幻觉

async def _unknown_course_refusal(gt: GroundTruth):
    # 这门课不存在。正确行为是说找不到，而不是编一个成绩出来
    return [
        MentionsNone(
            ["A+", "95", "your grade is", "current score"],
            label="不编造成绩",
        ),
        MaxActionSteps(6),
    ], None


CASES: List[Case] = [
    Case(
        id="list_active_courses",
        query="What courses am I currently enrolled in? List the active ones.",
        tags=["single_tool", "listing"],
        build=_list_courses,
    ),
    Case(
        id="assignments_multi_hop",
        query=_assignments_query,
        tags=["multi_hop"],
        build=_assignments_of_course,
    ),
    Case(
        id="compare_grades",
        query=_compare_grades_query,  # query 也由 ground truth 生成
        tags=["multi_course", "numeric"],
        build=_compare_grades,
    ),
    Case(
        id="parallel_assignments",
        query=_parallel_assignments_query,
        tags=["parallel", "multi_hop"],
        build=_parallel_assignments,
    ),
    Case(
        id="highest_grade_reasoning",
        query="Across all my courses that have a grade, which one do I have the highest score in? Give the course and the score.",
        tags=["reasoning", "aggregation"],
        build=_highest_grade,
    ),
    Case(
        id="unknown_course_refusal",
        query="What is my current grade in Underwater Basket Weaving 401?",
        tags=["robustness", "negative", "refusal"],
        build=_unknown_course_refusal,
    ),
]

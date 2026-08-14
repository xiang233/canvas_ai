"""
从 Canvas API 直接算 ground truth。

不写死期望值：成绩会变、作业会加、学期会滚动。每次 eval 开始时拉一次真实数据，
用它生成断言，这样 case 永远和账号当前状态同步，不需要人工维护。
"""

import re
from typing import Any, Dict, List, Optional

from src.tools.canvas_tools import CanvasAPIBase


class _Client(CanvasAPIBase):
    """只借用基类的请求逻辑（含重试），不作为 agent 工具注册"""

    name = "ground_truth_client"
    description = "internal"
    parameters = {"type": "object", "properties": {}}
    output_type = "any"

    async def forward(self):  # pragma: no cover - 只借用 _make_request，不作为工具调用
        raise NotImplementedError


def short_name(course_name: str) -> str:
    """把 'Spring_2026.CSE.5401.01 - Advanced Algorithms' 缩成 'Advanced Algorithms'。

    Agent 通常用人类叫法回答，用全名做断言会误判。"""
    if " - " in course_name:
        return course_name.split(" - ", 1)[1].strip()
    return course_name.strip()


class GroundTruth:
    """一次 eval 运行期间的 Canvas 事实快照"""

    def __init__(self):
        self._client = _Client()
        self.courses: List[Dict[str, Any]] = []

    async def load(self):
        courses = await self._client._make_request(
            "GET",
            "courses",
            params={"enrollment_state": "active", "per_page": "100", "include[]": "total_scores"},
        )
        if not isinstance(courses, list):
            raise RuntimeError(f"拉取课程失败: {courses}")
        # 无 name 的课程是受限课程，学生看不到，agent 也报不出来
        self.courses = [c for c in courses if c.get("name")]
        return self

    # ---------- 课程 ----------

    def active_course_count(self) -> int:
        return len(self.courses)

    def find_course(self, keyword: str) -> Optional[Dict[str, Any]]:
        """按关键词找课程，大小写不敏感"""
        k = keyword.lower()
        for c in self.courses:
            if k in c["name"].lower():
                return c
        return None

    def course_short_names(self) -> List[str]:
        return [short_name(c["name"]) for c in self.courses]

    # ---------- 成绩 ----------

    def graded_courses(self) -> List[Dict[str, Any]]:
        out = []
        for c in self.courses:
            enr = (c.get("enrollments") or [{}])[0]
            if enr.get("computed_current_score") is not None:
                out.append(c)
        return out

    def distinctive_graded_courses(self) -> List[Dict[str, Any]]:
        """挑适合做数值断言的课程。

        排除整数分（100.0、50.0 这类容易在答案里巧合出现，断言没有区分度），
        按小数分优先返回。"""
        scored = []
        for c in self.graded_courses():
            s = (c.get("enrollments") or [{}])[0]["computed_current_score"]
            if float(s) != int(float(s)):
                scored.append(c)
        return scored

    def score_of(self, keyword: str) -> Optional[float]:
        c = self.find_course(keyword)
        if not c:
            return None
        enr = (c.get("enrollments") or [{}])[0]
        return enr.get("computed_current_score")

    # ---------- 作业 ----------

    async def assignments_of(self, keyword: str) -> List[Dict[str, Any]]:
        c = self.find_course(keyword)
        if not c:
            return []
        result = await self._client._make_request(
            "GET", f"courses/{c['id']}/assignments", params={"per_page": "100"}
        )
        return result if isinstance(result, list) else []


def normalize(text: str) -> str:
    """比较用的归一化：小写、压空白。

    不去标点：课程代码里的点和数字是有意义的。"""
    return re.sub(r"\s+", " ", str(text)).lower().strip()

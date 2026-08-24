"""
Canvas LMS API 工具集 - 学生权限版本

这个文件包含了学生在Canvas上可以使用的所有API工具
使用 os.environ 获取 CANVAS_ACCESS_TOKEN 和 CANVAS_URL
"""

import asyncio
import os
import random
import re
import aiohttp
from typing import Optional, List, Dict, Any
from src.tools import AsyncTool, ToolResult
from src.registry import TOOL

# 只对瞬时故障重试：限流和服务端错误。4xx（401/403/404 等）重试无意义
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# 每页条数，以及翻页的失控保护上限（100 是 Canvas 允许的最大值，
# 用满可以少发几轮请求）
PER_PAGE = 100
MAX_PAGES = 20


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量，非法值回落到默认值"""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class CanvasAPIBase(AsyncTool):
    """Canvas API 基类，处理通用的API调用逻辑"""

    def __init__(self):
        super().__init__()
        self.canvas_url = os.environ.get("CANVAS_URL", "https://canvas.instructure.com")
        self.access_token = os.environ.get("CANVAS_ACCESS_TOKEN")
        self.max_retries = _env_int("CANVAS_MAX_RETRIES", 3)
        self.timeout = _env_int("CANVAS_TIMEOUT", 30)
        if "http://" in self.canvas_url:
            self.canvas_url = self.canvas_url.replace("http://", "https://")
        if "http" not in self.canvas_url:
            self.canvas_url = "https://" + self.canvas_url
        # 去掉结尾斜杠，否则 base_url 会拼出 //api/v1
        self.canvas_url = self.canvas_url.rstrip("/")

        if not self.access_token:
            raise ValueError("未找到 CANVAS_ACCESS_TOKEN 环境变量")
        
        self.base_url = f"{self.canvas_url}/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    def _backoff_delay(self, attempt: int, retry_after: Optional[str] = None) -> float:
        """指数退避 + 随机抖动。抖动是必需的：工具调用是并发的，
        没有抖动的话同时被限流的几个请求会在同一时刻一起重试"""
        if retry_after:
            try:
                return min(float(retry_after), 30.0)
            except ValueError:
                pass
        return min(0.5 * (2 ** attempt), 8.0) + random.uniform(0, 0.3)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        _return_link: bool = False,
    ) -> Dict[str, Any]:
        """发送API请求的通用方法，对瞬时故障做指数退避重试

        _return_link 供 _fetch_all_pages 使用：额外带回 Link 头，
        普通调用方的返回形状不变。endpoint 可以是绝对 URL（翻页时
        Canvas 给的 next 链接就是绝对的）。
        """
        # 翻页拿到的 next 是绝对 URL，不能再拼 base_url
        url = endpoint if endpoint.startswith("http") else f"{self.base_url}/{endpoint}"
        last_error = "unknown error"

        for attempt in range(self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=method,
                        url=url,
                        headers=self.headers,
                        params=params,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 200:
                            payload = await response.json()
                            if _return_link:
                                return {
                                    "_data": payload,
                                    "_next": self._next_page_url(
                                        response.headers.get("Link", "")
                                    ),
                                }
                            return payload
                        elif response.status == 404:
                            # 忠实转述,不解读:哪个端点、什么状态码是模型推断的
                            # 原料。这个 API 的 404 往往不是"资源不存在"(比如
                            # 未启用某功能的课),具体含义写在各工具的 description
                            # 里,解读交给模型
                            return {"error": f"HTTP 404 Not Found: {method} {url}"}

                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {method} {url}: {error_text[:300]}"

                        if response.status in RETRYABLE_STATUSES and attempt < self.max_retries:
                            await asyncio.sleep(
                                self._backoff_delay(attempt, response.headers.get("Retry-After"))
                            )
                            continue
                        return {"error": last_error}

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # 网络抖动/超时，可重试
                last_error = f"Request Error: {type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff_delay(attempt))
                    continue
                return {"error": last_error}

            except Exception as e:
                # 非网络异常，重试无意义
                return {"error": f"Request Error: {str(e)}"}

        return {"error": last_error}

    @staticmethod
    def render_list(title: str, items: List[Any], line_fn, scope: str = "") -> str:
        """列表统一渲染：永远报条数，空了就明说"共 0 条"。

        空结果只输出标题的话，模型分不清"没有数据"和"工具没干活"，会陷入
        重试（教授名字/公告/todo 三次事故同一个成因）。这里只陈述事实——
        条数和查询范围；"接下来该怎么办"的知识在各工具的 description 里，
        推断是模型的事。
        """
        scope_note = f"（{scope}）" if scope else ""
        if not items:
            return f"{title}{scope_note}: 共 0 条。"
        body = "\n".join(line_fn(x) for x in items)
        return f"{title}{scope_note}: 共 {len(items)} 条\n{body}"

    @staticmethod
    def _next_page_url(link_header: str) -> Optional[str]:
        """从 Canvas 的 Link 头里取 rel="next" 的 URL，没有则返回 None"""
        if not link_header:
            return None
        for link in link_header.split(","):
            if 'rel="next"' in link:
                start, end = link.find("<"), link.find(">")
                if start != -1 and end > start:
                    return link[start + 1:end]
        return None

    async def _fetch_all_pages(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        max_pages: int = MAX_PAGES,
    ) -> Any:
        """跟随 Link 头翻完所有页。

        列表端点只传 per_page 会静默截断：60 个作业的课程只回 50 个，
        调用方看不出还有下一页。这里复用 _make_request 的重试逻辑逐页取，
        直到没有 rel="next"。

        max_pages 是失控保护，达到上限时在末尾追加一条 _truncated 标记，
        让调用方能如实说明结果不完整，而不是又一次静默截断。
        """
        all_items: List[Any] = []
        next_endpoint: Optional[str] = endpoint
        page_params = dict(params) if params else None

        for _ in range(max_pages):
            result = await self._make_request(
                "GET", next_endpoint, params=page_params, _return_link=True
            )
            if isinstance(result, dict) and "error" in result:
                # 首页就失败：把错误如实返回。后续页失败：保留已拿到的部分
                return result if not all_items else all_items

            data = result.get("_data") if isinstance(result, dict) else result
            if isinstance(data, list):
                all_items.extend(data)
            elif data is not None:
                all_items.append(data)

            next_endpoint = result.get("_next") if isinstance(result, dict) else None
            if not next_endpoint:
                return all_items
            # next 链接自带查询串，重复传 params 会覆盖它的 page 参数
            page_params = None

        all_items.append({"_truncated": f"结果超过 {max_pages} 页，未取完"})
        return all_items


@TOOL.register_module(name="canvas_list_courses", force=True)
class CanvasListCourses(CanvasAPIBase):
    """列出学生的所有课程"""
    
    name = "canvas_list_courses"
    description = (
        "获取当前学生注册的所有课程列表，包括课程名称、ID、状态等信息。"
        "按需传 include 拿附加信息："
        "查教授/教师姓名传 include='teachers'；"
        "跨多门课比较成绩传 include='total_scores'（不要逐门调 canvas_get_grades）；"
        "查课程大纲/上课时间地点传 include='syllabus_body'；"
        "查选课人数传 include='total_students'。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "enrollment_state": {
                "type": "string",
                "description": "课程注册状态: active(活跃), completed(已完成), 默认为active",
                "nullable": True
            },
            "include": {
                "type": "string",
                "description": (
                    "包含额外信息，可选: total_scores, total_students, teachers, syllabus_body。"
                    "total_scores 会在每门课的 enrollments 里返回 computed_current_score 和 "
                    "computed_current_grade，用它可以一次性拿到所有课程成绩"
                ),
                "nullable": True
            }
        },
        "required": [],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(
        self, 
        enrollment_state: str = "active", 
        include: str = ""
    ) -> ToolResult:
        """获取课程列表"""
        try:
            params = {
                "enrollment_state": enrollment_state,
                "per_page": PER_PAGE
            }
            if include:
                params["include[]"] = include
            
            result = await self._fetch_all_pages("courses", params=params)
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])
            
            # 格式化输出
            courses_info = []
            for course in result:
                info = {
                    "id": course.get("id"),
                    "name": course.get("name"),
                    "course_code": course.get("course_code"),
                    "workflow_state": course.get("workflow_state"),
                    "enrollments": course.get("enrollments", []),
                    "teachers": course.get("teachers", []),
                    "syllabus_body": course.get("syllabus_body"),
                    "total_students": course.get("total_students"),
                }
                courses_info.append(info)
            
            # 每种 include 拿到的数据都必须渲染出来。schema 里声明支持某个
            # include 却不渲染它，等于对调用方撒谎：模型看不到自己要的字段，
            # 会以为参数传错了而反复重试同一个调用——实测这样能空转 15 次、
            # 烧掉 46 万 token 直到撞上步数上限。
            # total_scores 当初就是这么修的，teachers / syllabus_body 是后补的。
            lines = []
            for c in courses_info:
                line = f"- [{c['id']}] {c['name']} ({c['course_code']})"
                enr = (c["enrollments"] or [{}])[0]
                score = enr.get("computed_current_score")
                if score is not None:
                    grade = enr.get("computed_current_grade")
                    line += f" | current_score: {score}"
                    if grade:
                        line += f" ({grade})"
                if c.get("total_students") is not None:
                    line += f" | students: {c['total_students']}"
                if c.get("teachers"):
                    names = ", ".join(
                        t.get("display_name") or t.get("name") or str(t.get("id"))
                        for t in c["teachers"]
                    )
                    line += f" | teachers: {names}"
                if c.get("syllabus_body"):
                    # syllabus 是 HTML，整段塞进 observation 会挤爆上下文。
                    # 去标签后截断，需要全文用 canvas_get_page_content
                    text = re.sub(r"<[^>]+>", " ", c["syllabus_body"])
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        line += f" | syllabus: {text[:300]}"
                lines.append(line)

            return ToolResult(
                output=f"找到 {len(courses_info)} 门课程:\n" + "\n".join(lines),
                error=None
            )
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取课程列表失败: {str(e)}")


@TOOL.register_module(name="canvas_get_assignments", force=True)
class CanvasGetAssignments(CanvasAPIBase):
    """获取课程的作业列表"""
    
    name = "canvas_get_assignments"
    description = (
        "获取指定课程的所有作业，包括名称、截止日期、分数与提交状态。"
        "测验型作业（online_quiz / New Quizzes）也出现在这里——"
        "canvas_get_quizzes 返回 404 或为空时，测验通常能在这份列表里找到。"
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "课程ID"
            },
            "include_submission": {
                "type": "boolean",
                "description": "是否包含提交信息，默认为True",
                "nullable": True
            }
        },
        "required": ["course_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(
        self, 
        course_id: str, 
        include_submission: bool = True
    ) -> ToolResult:
        """获取作业列表"""
        try:
            params = {"per_page": PER_PAGE}
            if include_submission:
                params["include[]"] = "submission"
            
            result = await self._fetch_all_pages(
                f"courses/{course_id}/assignments",
                params=params
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])
            
            # 格式化输出
            assignments_info = []
            for assignment in result:
                info = {
                    "id": assignment.get("id"),
                    "name": assignment.get("name"),
                    "due_at": assignment.get("due_at", "无截止日期"),
                    "points_possible": assignment.get("points_possible", 0),
                    "submission_types": assignment.get("submission_types", []),
                    "submission": assignment.get("submission", {})
                }
                assignments_info.append(info)
            
            output = f"课程 {course_id} 共有 {len(assignments_info)} 个作业:\n"
            for a in assignments_info:
                submission_status = "未提交"
                if a["submission"]:
                    submission_status = a["submission"].get("workflow_state", "未提交")
                output += f"- [{a['id']}] {a['name']} (截止: {a['due_at']}, 分值: {a['points_possible']}, 状态: {submission_status})\n"
            
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取作业列表失败: {str(e)}")


# @TOOL.register_module(name="canvas_submit_assignment", force=True)
# class CanvasSubmitAssignment(CanvasAPIBase):
#     """提交作业"""
    
#     name = "canvas_submit_assignment"
#     description = "提交指定课程的作业，支持文本、URL等提交类型"
    
#     parameters = {
#         "type": "object",
#         "properties": {
#             "course_id": {
#                 "type": "string",
#                 "description": "课程ID"
#             },
#             "assignment_id": {
#                 "type": "string",
#                 "description": "作业ID"
#             },
#             "submission_type": {
#                 "type": "string",
#                 "description": "提交类型: online_text_entry, online_url, online_upload"
#             },
#             "body": {
#                 "type": "string",
#                 "description": "提交内容（文本提交时使用）",
#                 "nullable": True
#             },
#             "url": {
#                 "type": "string",
#                 "description": "提交的URL（URL提交时使用）",
#                 "nullable": True
#             }
#         },
#         "required": ["course_id", "assignment_id", "submission_type"],
#         "additionalProperties": False
#     }
    
#     output_type = "any"
    
#     async def forward(
#         self, 
#         course_id: str, 
#         assignment_id: str,
#         submission_type: str,
#         body: str = "",
#         url: str = ""
#     ) -> ToolResult:
#         """提交作业"""
#         try:
#             data = {
#                 "submission": {
#                     "submission_type": submission_type
#                 }
#             }
            
#             if submission_type == "online_text_entry" and body:
#                 data["submission"]["body"] = body
#             elif submission_type == "online_url" and url:
#                 data["submission"]["url"] = url
            
#             result = await self._make_request(
#                 "POST",
#                 f"courses/{course_id}/assignments/{assignment_id}/submissions",
#                 data=data
#             )
            
#             if isinstance(result, dict) and "error" in result:
#                 return ToolResult(output=None, error=result["error"])
            
#             return ToolResult(
#                 output=f"作业提交成功！\n"
#                        f"- 作业ID: {assignment_id}\n"
#                        f"- 提交类型: {submission_type}\n"
#                        f"- 提交时间: {result.get('submitted_at', '未知')}",
#                 error=None
#             )
            
#         except Exception as e:
#             return ToolResult(output=None, error=f"提交作业失败: {str(e)}")


@TOOL.register_module(name="canvas_get_modules", force=True)
class CanvasGetModules(CanvasAPIBase):
    """获取课程模块列表"""
    
    name = "canvas_get_modules"
    description = "获取指定课程的所有模块和学习内容结构"
    
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "课程ID"
            }
        },
        "required": ["course_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, course_id: str) -> ToolResult:
        """获取模块列表"""
        try:
            result = await self._fetch_all_pages(
                f"courses/{course_id}/modules",
                params={"per_page": PER_PAGE}
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            output = self.render_list(
                f"课程 {course_id} 的模块", result,
                lambda m: f"📚 [{m.get('id')}] {m.get('name')} | 状态: {m.get('workflow_state')} | 项目数: {m.get('items_count', 0)}",
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取模块列表失败: {str(e)}")


@TOOL.register_module(name="canvas_get_module_items", force=True)
class CanvasGetModuleItems(CanvasAPIBase):
    """获取模块中的具体内容项"""
    
    name = "canvas_get_module_items"
    description = "获取指定模块中的所有学习内容项（文件、页面、作业等）"
    
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "课程ID"
            },
            "module_id": {
                "type": "string",
                "description": "模块ID"
            }
        },
        "required": ["course_id", "module_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, course_id: str, module_id: str) -> ToolResult:
        """获取模块项"""
        try:
            result = await self._fetch_all_pages(
                f"courses/{course_id}/modules/{module_id}/items",
                params={"per_page": PER_PAGE}
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            _icons = {"Assignment": "📝", "Page": "📄", "File": "📁", "Discussion": "💬",
                      "Quiz": "✏️", "ExternalUrl": "🔗", "ExternalTool": "🔧"}
            output = self.render_list(
                f"模块 {module_id} 的内容", result,
                lambda i: f"{_icons.get(i.get('type'), '•')} [{i.get('id')}] {i.get('title')} ({i.get('type')})",
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取模块项失败: {str(e)}")


@TOOL.register_module(name="canvas_get_files", force=True)
class CanvasGetFiles(CanvasAPIBase):
    """获取课程文件列表"""
    
    name = "canvas_get_files"
    description = (
        "获取课程的文件列表。Canvas 的脾气：教师把 Files 标签设为学生不可见时，"
        "此端点返回 HTTP 403——不代表没有材料，课程材料可能在 modules 或课程知识库"
        "（vector_store_search）里。"
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "课程ID"
            },
            "search_term": {
                "type": "string",
                "description": "搜索关键词（可选）",
                "nullable": True
            }
        },
        "required": ["course_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, course_id: str, search_term: str = "") -> ToolResult:
        """获取文件列表"""
        try:
            params = {"per_page": PER_PAGE}
            if search_term:
                params["search_term"] = search_term
            
            result = await self._fetch_all_pages(
                f"courses/{course_id}/files",
                params=params
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            output = self.render_list(
                f"课程 {course_id} 的文件", result,
                lambda f: f"📁 [{f.get('id')}] {f.get('display_name')} ({f.get('size', 0) / 1048576:.2f}MB, {f.get('content-type', '未知类型')}) URL: {f.get('url')}",
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取文件列表失败: {str(e)}")


@TOOL.register_module(name="canvas_get_discussions", force=True)
class CanvasGetDiscussions(CanvasAPIBase):
    """获取课程讨论主题列表"""
    
    name = "canvas_get_discussions"
    description = "获取指定课程的所有讨论话题"
    
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "课程ID"
            }
        },
        "required": ["course_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, course_id: str) -> ToolResult:
        """获取讨论列表"""
        try:
            result = await self._fetch_all_pages(
                f"courses/{course_id}/discussion_topics",
                params={"per_page": PER_PAGE}
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            output = self.render_list(
                f"课程 {course_id} 的讨论", result,
                lambda t: f"💬 [{t.get('id')}] {t.get('title')} | 发布: {t.get('posted_at', '未知')} | 回复数: {t.get('discussion_subentry_count', 0)}",
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取讨论列表失败: {str(e)}")


# @TOOL.register_module(name="canvas_post_discussion", force=True)
# class CanvasPostDiscussion(CanvasAPIBase):
#     """在讨论中发帖"""
    
#     name = "canvas_post_discussion"
#     description = "在指定的讨论主题中发表回复或评论"
    
#     parameters = {
#         "type": "object",
#         "properties": {
#             "course_id": {
#                 "type": "string",
#                 "description": "课程ID"
#             },
#             "topic_id": {
#                 "type": "string",
#                 "description": "讨论主题ID"
#             },
#             "message": {
#                 "type": "string",
#                 "description": "发帖内容"
#             }
#         },
#         "required": ["course_id", "topic_id", "message"],
#         "additionalProperties": False
#     }
    
#     output_type = "any"
    
#     async def forward(
#         self, 
#         course_id: str, 
#         topic_id: str, 
#         message: str
#     ) -> ToolResult:
#         """发表讨论回复"""
#         try:
#             data = {"message": message}
            
#             result = await self._make_request(
#                 "POST",
#                 f"courses/{course_id}/discussion_topics/{topic_id}/entries",
#                 data=data
#             )
            
#             if isinstance(result, dict) and "error" in result:
#                 return ToolResult(output=None, error=result["error"])
            
#             return ToolResult(
#                 output=f"讨论回复发表成功！\n"
#                        f"- 主题ID: {topic_id}\n"
#                        f"- 发表时间: {result.get('created_at', '未知')}",
#                 error=None
#             )
            
#         except Exception as e:
#             return ToolResult(output=None, error=f"发表讨论失败: {str(e)}")


@TOOL.register_module(name="canvas_get_announcements", force=True)
class CanvasGetAnnouncements(CanvasAPIBase):
    """获取公告"""

    name = "canvas_get_announcements"
    description = (
        "获取课程公告，按发布时间从新到旧排列。"
        "默认回看最近一年——Canvas 这个端点不传日期时只返回最近 14 天，"
        "结课后的课程用默认值永远查不到东西，所以这里把窗口放宽了。"
        "要查更早的公告，显式传 start_date。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "context_codes": {
                "type": "string",
                "description": "课程ID列表，格式: course_123,course_456（可选，留空则获取所有课程）",
                "nullable": True
            },
            "start_date": {
                "type": "string",
                "description": "起始日期 YYYY-MM-DD（可选，默认一年前）",
                "nullable": True
            },
            "end_date": {
                "type": "string",
                "description": "结束日期 YYYY-MM-DD（可选，默认明天）",
                "nullable": True
            }
        },
        "required": [],
        "additionalProperties": False
    }

    output_type = "any"

    async def forward(
        self,
        context_codes: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> ToolResult:
        """获取公告列表"""
        from datetime import datetime, timedelta, timezone

        try:
            # Canvas 的隐藏默认：start_date=14 天前，end_date=28 天后。
            # 学期进行中恰好够用，结课后的课程永远查不到——默认放宽到一年
            now = datetime.now(timezone.utc)
            params = {
                "per_page": PER_PAGE,
                "start_date": start_date or (now - timedelta(days=365)).strftime("%Y-%m-%d"),
                "end_date": end_date or (now + timedelta(days=1)).strftime("%Y-%m-%d"),
            }
            if context_codes:
                params["context_codes[]"] = context_codes.split(",")

            result = await self._fetch_all_pages("announcements", params=params)

            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            window = f"{params['start_date']} 至 {params['end_date']}"
            result.sort(key=lambda a: a.get("posted_at") or "", reverse=True)
            # 只陈述事实:条数、窗口、逐条内容。"近期没有该不该主动说明"
            # 这类行为由模型根据窗口信息自行判断
            output = self.render_list(
                "📢 公告", result,
                lambda a: f"标题: {a.get('title')} | 发布: {a.get('posted_at', '未知')} | 内容: {(a.get('message') or '无内容')[:200]}...",
                scope=f"{window}，从新到旧",
            )
            return ToolResult(output=output, error=None)

        except Exception as e:
            return ToolResult(output=None, error=f"获取公告失败: {str(e)}")


@TOOL.register_module(name="canvas_get_calendar_events", force=True)
class CanvasGetCalendarEvents(CanvasAPIBase):
    """获取日历事件"""
    
    name = "canvas_get_calendar_events"
    description = (
        "获取日历事件。Canvas 的脾气：不传日期时默认只返回今天的事件，"
        "查任何历史或未来日程都必须显式传 start_date / end_date。"
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "开始日期，格式: YYYY-MM-DD（可选）",
                "nullable": True
            },
            "end_date": {
                "type": "string",
                "description": "结束日期，格式: YYYY-MM-DD（可选）",
                "nullable": True
            }
        },
        "required": [],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(
        self, 
        start_date: str = "", 
        end_date: str = ""
    ) -> ToolResult:
        """获取日历事件"""
        try:
            params = {"per_page": PER_PAGE}
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            
            result = await self._fetch_all_pages(
                "calendar_events",
                params=params
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            output = self.render_list(
                "📅 日历事件", result,
                lambda e: (
                    f"🗓️ {e.get('title')} | 时间: {e.get('start_at', '未知')} | 类型: {e.get('type', '未知')}"
                    + (f" | 描述: {e.get('description')[:100]}..." if e.get('description') else "")
                ),
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取日历事件失败: {str(e)}")


@TOOL.register_module(name="canvas_get_grades", force=True)
class CanvasGetGrades(CanvasAPIBase):
    """获取课程成绩"""
    
    name = "canvas_get_grades"
    description = (
        "获取学生在单门课程中的成绩。要跨多门课比较成绩时，"
        "用 canvas_list_courses(include='total_scores') 一次拿全，不要逐门调用本工具。"
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "课程ID"
            }
        },
        "required": ["course_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, course_id: str) -> ToolResult:
        """获取成绩"""
        try:
            # 获取当前用户ID
            user_result = await self._make_request("GET", "users/self")
            if isinstance(user_result, dict) and "error" in user_result:
                return ToolResult(output=None, error=user_result["error"])
            
            user_id = user_result.get("id")
            
            # 获取注册信息（包含成绩）
            result = await self._make_request(
                "GET",
                f"courses/{course_id}/enrollments",
                params={"user_id": user_id}
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            output = self.render_list(
                f"课程 {course_id} 的成绩", result,
                lambda en: (
                    f"📊 当前成绩: {en.get('grades', {}).get('current_grade', '暂无')}"
                    f" | 当前分数: {en.get('grades', {}).get('current_score', '暂无')}"
                    f" | 最终成绩: {en.get('grades', {}).get('final_grade', '暂无')}"
                    f" | 最终分数: {en.get('grades', {}).get('final_score', '暂无')}"
                ),
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取成绩失败: {str(e)}")


@TOOL.register_module(name="canvas_get_pages", force=True)
class CanvasGetPages(CanvasAPIBase):
    """获取课程页面列表"""
    
    name = "canvas_get_pages"
    description = (
        "获取课程的 Wiki 页面列表。Canvas 的脾气：未启用 Pages 功能的课程"
        "此端点返回 HTTP 404，不代表课程不存在。"
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "课程ID"
            }
        },
        "required": ["course_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, course_id: str) -> ToolResult:
        """获取页面列表"""
        try:
            result = await self._fetch_all_pages(
                f"courses/{course_id}/pages",
                params={"per_page": PER_PAGE}
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            output = self.render_list(
                f"课程 {course_id} 的页面", result,
                lambda p: f"📄 {p.get('title')} | URL: {p.get('url')} | 更新: {p.get('updated_at', '未知')}",
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取页面列表失败: {str(e)}")


@TOOL.register_module(name="canvas_get_page_content", force=True)
class CanvasGetPageContent(CanvasAPIBase):
    """获取页面详细内容"""
    
    name = "canvas_get_page_content"
    description = "获取指定页面的完整内容"
    
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "课程ID"
            },
            "page_url": {
                "type": "string",
                "description": "页面URL或标题"
            }
        },
        "required": ["course_id", "page_url"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, course_id: str, page_url: str) -> ToolResult:
        """获取页面内容"""
        try:
            result = await self._make_request(
                "GET",
                f"courses/{course_id}/pages/{page_url}"
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])
            
            output = f"📄 页面: {result.get('title')}\n"
            output += f"更新时间: {result.get('updated_at', '未知')}\n\n"
            output += f"内容:\n{result.get('body', '无内容')}\n"
            
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取页面内容失败: {str(e)}")


@TOOL.register_module(name="canvas_get_quizzes", force=True)
class CanvasGetQuizzes(CanvasAPIBase):
    """获取课程测验列表"""
    
    name = "canvas_get_quizzes"
    description = (
        "获取指定课程的经典测验列表，按截止时间从新到旧。"
        "Canvas 的脾气：未启用经典测验功能的课程，此端点返回 HTTP 404——"
        "不代表课程不存在；这类课程的测验类作业会出现在 canvas_get_assignments 里。"
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "课程ID"
            }
        },
        "required": ["course_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, course_id: str) -> ToolResult:
        """获取测验列表"""
        try:
            result = await self._fetch_all_pages(
                f"courses/{course_id}/quizzes",
                params={"per_page": PER_PAGE}
            )
            
            if isinstance(result, dict) and "error" in result:
                # 错误原样透传(含方法+URL+状态码)。404 在这个端点的含义
                # 写在 description 里,解读交给模型
                return ToolResult(output=None, error=result["error"])

            result.sort(key=lambda q: q.get("due_at") or "", reverse=True)
            output = self.render_list(
                f"课程 {course_id} 的测验", result,
                lambda q: f"✏️ [{q.get('id')}] {q.get('title')} | 类型: {q.get('quiz_type', '未知')} | 分数: {q.get('points_possible', 0)} | 截止: {q.get('due_at', '无截止时间')}",
                scope="按截止时间从新到旧",
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取测验列表失败: {str(e)}")


@TOOL.register_module(name="canvas_get_todo_items", force=True)
class CanvasGetTodoItems(CanvasAPIBase):
    """获取待办事项"""
    
    name = "canvas_get_todo_items"
    description = (
        "获取学生的待办事项。Canvas 的 todo 只包含未完成的近期任务，"
        "只看未来——已结课课程的内容永远不在这里，返回 0 条是正常现象。"
        "查历史作业用 canvas_get_assignments，查成绩用 canvas_get_grades。"
    )
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self) -> ToolResult:
        """获取待办事项"""
        try:
            result = await self._make_request("GET", "users/self/todo")
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            output = self.render_list(
                "📝 待办事项", result,
                lambda i: (
                    f"• {i.get('assignment', {}).get('name', '未知任务')}"
                    f" | 课程: {i.get('context_name', '未知')}"
                    f" | 截止: {i.get('assignment', {}).get('due_at', '无截止时间')}"
                    f" | 分数: {i.get('assignment', {}).get('points_possible', 0)}"
                ),
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取待办事项失败: {str(e)}")


@TOOL.register_module(name="canvas_get_upcoming_events", force=True)
class CanvasGetUpcomingEvents(CanvasAPIBase):
    """获取即将到来的事件"""
    
    name = "canvas_get_upcoming_events"
    description = (
        "获取即将到来的事件。此端点只看未来，结课后通常为 0 条。"
        "查过去的日程用 canvas_get_calendar_events 并显式传日期范围。"
    )
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self) -> ToolResult:
        """获取即将事件"""
        try:
            result = await self._make_request("GET", "users/self/upcoming_events")
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            output = self.render_list(
                "🗓️ 即将到来的事件", result,
                lambda e: f"• {e.get('title', '未知事件')} | 时间: {e.get('start_at', '未知')} | 类型: {e.get('type', '未知')}",
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取即将事件失败: {str(e)}")


@TOOL.register_module(name="canvas_get_groups", force=True)
class CanvasGetGroups(CanvasAPIBase):
    """获取学生小组"""
    
    name = "canvas_get_groups"
    description = "获取学生参与的所有小组"
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self) -> ToolResult:
        """获取小组列表"""
        try:
            result = await self._make_request("GET", "users/self/groups")
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            output = self.render_list(
                "👥 我的小组", result,
                lambda g: f"• [{g.get('id')}] {g.get('name')} | 成员数: {g.get('members_count', 0)} | 课程: {g.get('course_id', '未知')}",
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取小组列表失败: {str(e)}")


@TOOL.register_module(name="canvas_get_file_info", force=True)
class CanvasGetFileInfo(CanvasAPIBase):
    """获取文件详细信息"""
    
    name = "canvas_get_file_info"
    description = (
        "获取文件的元数据：名称、大小、类型、下载链接。"
        "不返回文档正文，要读内容用 vector_store_search"
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "文件ID"
            }
        },
        "required": ["file_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, file_id: str) -> ToolResult:
        """获取文件信息"""
        try:
            result = await self._make_request("GET", f"files/{file_id}")
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])
            
            output = f"📁 文件信息:\n"
            output += f"名称: {result.get('display_name')}\n"
            output += f"ID: {result.get('id')}\n"
            output += f"大小: {result.get('size', 0) / (1024*1024):.2f} MB\n"
            output += f"类型: {result.get('content-type', '未知')}\n"
            output += f"修改时间: {result.get('modified_at', '未知')}\n"
            output += f"下载链接: {result.get('url')}\n"
            
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取文件信息失败: {str(e)}")

@TOOL.register_module(name="canvas_get_folders", force=True)
class CanvasGetFolders(CanvasAPIBase):
    """获取课程文件夹列表"""
    
    name = "canvas_get_folders"
    description = "获取指定课程的文件夹结构"
    
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "课程ID"
            }
        },
        "required": ["course_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, course_id: str) -> ToolResult:
        """获取文件夹列表"""
        try:
            result = await self._fetch_all_pages(
                f"courses/{course_id}/folders",
                params={"per_page": PER_PAGE}
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            output = self.render_list(
                f"📂 课程 {course_id} 的文件夹", result,
                lambda fo: f"• [{fo.get('id')}] {fo.get('full_name')} | 文件数: {fo.get('files_count', 0)}",
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取文件夹列表失败: {str(e)}")


@TOOL.register_module(name="canvas_get_folder_files", force=True)
class CanvasGetFolderFiles(CanvasAPIBase):
    """获取文件夹中的文件"""
    
    name = "canvas_get_folder_files"
    description = "获取指定文件夹中的所有文件"
    
    parameters = {
        "type": "object",
        "properties": {
            "folder_id": {
                "type": "string",
                "description": "文件夹ID"
            }
        },
        "required": ["folder_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, folder_id: str) -> ToolResult:
        """获取文件夹中的文件"""
        try:
            result = await self._fetch_all_pages(
                f"folders/{folder_id}/files",
                params={"per_page": PER_PAGE}
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])

            output = self.render_list(
                f"📂 文件夹 {folder_id} 中的文件", result,
                lambda f: f"📄 [{f.get('id')}] {f.get('display_name')} | {f.get('size', 0) / 1048576:.2f}MB | {f.get('content-type', '未知')}",
            )
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取文件夹文件失败: {str(e)}")


@TOOL.register_module(name="canvas_search_files", force=True)
class CanvasSearchFiles(CanvasAPIBase):
    """搜索课程中的文件"""
    
    name = "canvas_search_files"
    description = (
        "按文件名在指定课程中搜索文件。只匹配文件名，读不到文档正文；"
        "要检索讲义、幻灯片、论文的内容，用 vector_store_search"
    )

    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "课程ID"
            },
            "search_term": {
                "type": "string",
                "description": "匹配文件名的关键词，不会匹配文件内容"
            }
        },
        "required": ["course_id", "search_term"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, course_id: str, search_term: str) -> ToolResult:
        """搜索文件"""
        try:
            result = await self._fetch_all_pages(
                f"courses/{course_id}/files",
                params={"search_term": search_term, "per_page": PER_PAGE}
            )
            
            if isinstance(result, dict) and "error" in result:
                return ToolResult(output=None, error=result["error"])
            
            output = self.render_list(
                f"🔍 搜索 '{search_term}' 的结果", result,
                lambda f: f"📄 [{f.get('id')}] {f.get('display_name')} | {f.get('size', 0) / 1024:.2f}KB | 下载: {f.get('url')}",
            )
            
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"搜索文件失败: {str(e)}")


@TOOL.register_module(name="vector_store_list", force=True)
class VectorStoreList(AsyncTool):
    """列出所有可用的 Vector Stores"""
    
    name = "vector_store_list"
    description = "列出所有可用的课程知识库（Vector Stores），显示每个知识库的ID、名称和文件数量"
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self) -> ToolResult:
        """获取 Vector Store 列表"""
        try:
            # 导入 OpenAI
            try:
                from openai import OpenAI
            except ImportError:
                return ToolResult(
                    output=None,
                    error="OpenAI 库未安装，请运行: pip install openai"
                )
            
            # 获取 API Key
            import os
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                return ToolResult(
                    output=None,
                    error="未配置 OPENAI_API_KEY，请在 .env 文件中添加"
                )
            
            # 创建客户端
            client = OpenAI(
                api_key=openai_api_key,
                default_headers={"OpenAI-Beta": "assistants=v2"}
            )
            
            # 获取 Vector Stores
            response = client.vector_stores.list(limit=100)
            vector_stores = list(response.data)
            
            if not vector_stores:
                return ToolResult(
                    output="📋 当前没有可用的知识库\n请先运行 file_index_downloader.py 创建知识库",
                    error=None
                )
            
            # 格式化输出
            output = f"📚 找到 {len(vector_stores)} 个课程知识库:\n\n"
            
            for i, vs in enumerate(vector_stores, 1):
                file_count = vs.file_counts.total if hasattr(vs, 'file_counts') else 0
                output += f"{i}. [{vs.id}] {vs.name}\n"
                output += f"   📁 文件数量: {file_count}\n"
                if hasattr(vs, 'created_at'):
                    output += f"   📅 创建时间: {vs.created_at}\n"
                output += "\n"
            
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            return ToolResult(output=None, error=f"获取知识库列表失败: {str(e)}")


@TOOL.register_module(name="vector_store_search", force=True)
class VectorStoreSearch(AsyncTool):
    """在 Vector Store 中搜索相关内容"""
    
    name = "vector_store_search"
    description = (
        "在课程知识库中做全文语义检索，返回讲义、幻灯片、论文的原文片段。"
        "回答课程内容问题时用这个，不要用 canvas_search_files（那个只匹配文件名）。"
        "需要先调 vector_store_list 拿到 vector_store_id"
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "vector_store_id": {
                "type": "string",
                "description": "Vector Store ID（从 vector_store_list 工具获取）"
            },
            "query": {
                "type": "string",
                "description": "搜索查询，例如：'这门课的主要内容是什么？'、'作业1的要求'等"
            },
            "max_results": {
                "type": "integer",
                "description": "返回的最大结果数（默认5）",
                "nullable": True
            }
        },
        "required": ["vector_store_id", "query"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, vector_store_id: str, query: str, max_results: int = 5) -> ToolResult:
        """搜索 Vector Store"""
        try:
            # 导入 OpenAI
            try:
                from openai import OpenAI
            except ImportError:
                return ToolResult(
                    output=None,
                    error="OpenAI 库未安装，请运行: pip install openai"
                )
            
            # 获取 API Key
            import os
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                return ToolResult(
                    output=None,
                    error="未配置 OPENAI_API_KEY，请在 .env 文件中添加"
                )
            
            # 创建客户端
            client = OpenAI(
                api_key=openai_api_key,
                default_headers={"OpenAI-Beta": "assistants=v2"}
            )
            
            # 执行搜索
            response = client.vector_stores.search(
                vector_store_id=vector_store_id,
                query=query,
                max_num_results=max_results
            )
            
            # 检查结果
            if not response or not hasattr(response, 'data') or not response.data:
                return ToolResult(
                    output=f"🔍 搜索查询: \"{query}\"\n❌ 未找到相关内容",
                    error=None
                )
            
            # 格式化输出
            output = f"🔍 搜索查询: \"{query}\"\n"
            output += f"📊 找到 {len(response.data)} 个相关结果:\n\n"
            
            for i, result in enumerate(response.data, 1):
                output += f"{'='*60}\n"
                output += f"结果 {i}:\n"
                
                # 相关性分数
                if hasattr(result, 'score'):
                    output += f"📈 相关性: {result.score:.2%}\n"
                
                # 文件名
                if hasattr(result, 'filename'):
                    output += f"📄 来源: {result.filename}\n"
                
                # 元数据
                if hasattr(result, 'attributes') and result.attributes:
                    output += f"🏷️  属性: {result.attributes}\n"
                
                # 内容
                if hasattr(result, 'content'):
                    content = result.content
                    # 限制长度
                    if len(content) > 800:
                        content = content[:800] + "...\n(内容已截断)"
                    output += f"\n📝 内容:\n{content}\n"
                
                output += "\n"
            
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return ToolResult(
                output=None,
                error=f"搜索失败: {str(e)}\n详情: {error_detail[:500]}"
            )


@TOOL.register_module(name="vector_store_list_files", force=True)
class VectorStoreListFiles(AsyncTool):
    """列出 Vector Store 中的所有文件"""
    
    name = "vector_store_list_files"
    description = "列出指定知识库中的所有文件，显示文件ID、名称、状态等信息，可选择读取文件内容"
    
    parameters = {
        "type": "object",
        "properties": {
            "vector_store_id": {
                "type": "string",
                "description": "Vector Store ID（从 vector_store_list 工具获取）"
            },
            "read_content": {
                "type": "boolean",
                "description": "是否读取文件内容（默认False，仅列出文件信息）",
                "nullable": True
            },
            "limit": {
                "type": "integer",
                "description": "返回的最大文件数（默认100，设置为0或负数表示获取所有文件）",
                "nullable": True
            }
        },
        "required": ["vector_store_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, vector_store_id: str, read_content: bool = False, limit: int = 100) -> ToolResult:
        """列出 Vector Store 文件"""
        try:
            # 导入 OpenAI
            try:
                from openai import OpenAI
            except ImportError:
                return ToolResult(
                    output=None,
                    error="OpenAI 库未安装，请运行: pip install openai"
                )
            
            # 获取 API Key
            import os
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                return ToolResult(
                    output=None,
                    error="未配置 OPENAI_API_KEY，请在 .env 文件中添加"
                )
            
            # 创建客户端
            client = OpenAI(
                api_key=openai_api_key,
                default_headers={"OpenAI-Beta": "assistants=v2"}
            )
            
            # 获取 Vector Store 文件列表
            files = []
            
            # 如果 limit <= 0，获取所有文件（分页）
            if limit <= 0:
                after = None
                while True:
                    if after:
                        response = client.vector_stores.files.list(
                            vector_store_id=vector_store_id,
                            limit=100,  # 每页最多100个
                            after=after
                        )
                    else:
                        response = client.vector_stores.files.list(
                            vector_store_id=vector_store_id,
                            limit=100
                        )
                    
                    batch_files = list(response.data)
                    if not batch_files:
                        break
                    
                    files.extend(batch_files)
                    
                    # 检查是否还有更多数据
                    if hasattr(response, 'has_more') and response.has_more:
                        # 使用最后一个文件的 ID 作为 after 参数
                        after = batch_files[-1].id
                    else:
                        break
            else:
                # 限制数量获取
                response = client.vector_stores.files.list(
                    vector_store_id=vector_store_id,
                    limit=limit
                )
                files = list(response.data)
            
            if not files:
                return ToolResult(
                    output=f"📋 Vector Store [{vector_store_id}] 中没有文件",
                    error=None
                )
            
            # 格式化输出
            output = f"📚 Vector Store [{vector_store_id}] 文件列表:\n"
            output += f"找到 {len(files)} 个文件\n\n"
            
            for i, file in enumerate(files, 1):
                output += f"{'='*60}\n"
                output += f"文件 {i}:\n"
                output += f"🆔 File ID: {file.id}\n"
                
                if hasattr(file, 'status'):
                    status_emoji = "✅" if file.status == "completed" else "⏳"
                    output += f"{status_emoji} 状态: {file.status}\n"
                
                if hasattr(file, 'created_at'):
                    from datetime import datetime
                    created = datetime.fromtimestamp(file.created_at)
                    output += f"📅 创建时间: {created.strftime('%Y-%m-%d %H:%M:%S')}\n"
                
                # 总是获取文件基本信息（文件名、大小等）
                try:
                    # 获取文件详细信息
                    file_info = client.files.retrieve(file.id)
                    
                    if hasattr(file_info, 'filename'):
                        output += f"📄 文件名: {file_info.filename}\n"
                    
                    if hasattr(file_info, 'bytes'):
                        size_kb = file_info.bytes / 1024
                        if size_kb >= 1024:
                            output += f"📦 大小: {size_kb / 1024:.2f} MB\n"
                        else:
                            output += f"📦 大小: {size_kb:.2f} KB\n"
                    
                    if hasattr(file_info, 'purpose'):
                        output += f"🎯 用途: {file_info.purpose}\n"
                    
                    # 如果需要读取内容
                    if read_content and file.status == "completed":
                        # 尝试读取文件内容
                        try:
                            content_response = client.files.content(file.id)
                            content = content_response.read()
                            
                            # 尝试解码为文本
                            try:
                                text_content = content.decode('utf-8')
                                # 限制长度
                                if len(text_content) > 1000:
                                    text_content = text_content[:1000] + "\n...(内容已截断)"
                                output += f"\n📝 内容预览:\n{text_content}\n"
                            except:
                                output += f"\n⚠️  无法显示文件内容（非文本文件或编码问题）\n"
                                output += f"   文件大小: {len(content)} 字节\n"
                        except Exception as e:
                            output += f"\n⚠️  读取内容失败: {str(e)}\n"
                
                except Exception as e:
                    output += f"\n⚠️  获取文件信息失败: {str(e)}\n"
                
                output += "\n"
            
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return ToolResult(
                output=None,
                error=f"获取文件列表失败: {str(e)}\n详情: {error_detail[:500]}"
            )


@TOOL.register_module(name="vector_store_get_file", force=True)
class VectorStoreGetFile(AsyncTool):
    """根据 file_id 获取和读取文件内容"""
    
    name = "vector_store_get_file"
    description = "根据 file_id 获取文件详细信息并读取内容，支持文本文件的完整内容展示"
    
    parameters = {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "OpenAI File ID（从 vector_store_list_files 工具获取）"
            },
            "max_length": {
                "type": "integer",
                "description": "最大显示长度（默认5000字符）",
                "nullable": True
            }
        },
        "required": ["file_id"],
        "additionalProperties": False
    }
    
    output_type = "any"
    
    async def forward(self, file_id: str, max_length: int = 5000) -> ToolResult:
        """获取文件内容"""
        try:
            # 导入 OpenAI
            try:
                from openai import OpenAI
            except ImportError:
                return ToolResult(
                    output=None,
                    error="OpenAI 库未安装，请运行: pip install openai"
                )
            
            # 获取 API Key
            import os
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                return ToolResult(
                    output=None,
                    error="未配置 OPENAI_API_KEY，请在 .env 文件中添加"
                )
            
            # 创建客户端
            client = OpenAI(
                api_key=openai_api_key,
                default_headers={"OpenAI-Beta": "assistants=v2"}
            )
            
            # 获取文件信息
            file_info = client.files.retrieve(file_id)
            
            output = f"📄 文件详细信息:\n"
            output += f"{'='*60}\n"
            output += f"🆔 File ID: {file_info.id}\n"
            
            if hasattr(file_info, 'filename'):
                output += f"📝 文件名: {file_info.filename}\n"
            
            if hasattr(file_info, 'purpose'):
                output += f"🎯 用途: {file_info.purpose}\n"
            
            if hasattr(file_info, 'bytes'):
                size_kb = file_info.bytes / 1024
                size_mb = size_kb / 1024
                if size_mb >= 1:
                    output += f"📦 大小: {size_mb:.2f} MB\n"
                else:
                    output += f"📦 大小: {size_kb:.2f} KB\n"
            
            if hasattr(file_info, 'created_at'):
                from datetime import datetime
                created = datetime.fromtimestamp(file_info.created_at)
                output += f"📅 创建时间: {created.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if hasattr(file_info, 'status'):
                status_emoji = "✅" if file_info.status == "processed" else "⏳"
                output += f"{status_emoji} 状态: {file_info.status}\n"
            
            output += f"\n{'='*60}\n"
            
            # 尝试读取文件内容
            try:
                content_response = client.files.content(file_id)
                content = content_response.read()
                
                # 尝试解码为文本
                try:
                    text_content = content.decode('utf-8')
                    
                    output += f"\n📖 文件内容:\n"
                    output += f"{'='*60}\n"
                    
                    if len(text_content) > max_length:
                        output += text_content[:max_length]
                        output += f"\n\n{'='*60}\n"
                        output += f"⚠️  内容已截断（显示 {max_length}/{len(text_content)} 字符）\n"
                        output += f"完整内容共 {len(text_content)} 字符\n"
                    else:
                        output += text_content
                        output += f"\n{'='*60}\n"
                        output += f"✅ 已显示完整内容（{len(text_content)} 字符）\n"
                
                except UnicodeDecodeError:
                    output += f"\n⚠️  文件是二进制格式，无法显示为文本\n"
                    output += f"文件大小: {len(content)} 字节\n"
                    
                    # 尝试判断文件类型
                    if content.startswith(b'%PDF'):
                        output += f"文件类型: PDF 文档\n"
                    elif content.startswith(b'\x50\x4b'):
                        output += f"文件类型: ZIP/Office 文档\n"
                    else:
                        output += f"文件类型: 未知二进制文件\n"
            
            except Exception as e:
                output += f"\n⚠️  读取文件内容失败: {str(e)}\n"
            
            return ToolResult(output=output, error=None)
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return ToolResult(
                output=None,
                error=f"获取文件失败: {str(e)}\n详情: {error_detail[:500]}"
            )


# 导出所有工具
__all__ = [
    "CanvasListCourses",
    "CanvasGetAssignments",
    # "CanvasSubmitAssignment",
    "CanvasGetModules",
    "CanvasGetModuleItems",
    "CanvasGetFiles",
    "CanvasGetFileInfo",
    "CanvasDownloadFile",
    "CanvasGetFolders",
    "CanvasGetFolderFiles",
    "CanvasSearchFiles",
    "CanvasGetDiscussions",
    # "CanvasPostDiscussion",
    "CanvasGetAnnouncements",
    "CanvasGetCalendarEvents",
    "CanvasGetGrades",
    "CanvasGetPages",
    "CanvasGetPageContent",
    "CanvasGetQuizzes",
    "CanvasGetTodoItems",
    "CanvasGetUpcomingEvents",
    "CanvasGetGroups",
    "VectorStoreList",
    "VectorStoreSearch",
    "VectorStoreListFiles",
    "VectorStoreGetFile",
    "CanvasReadFileContent",
]


@TOOL.register_module(name="canvas_read_file_content", force=True)
class CanvasReadFileContent(CanvasAPIBase):
    """下载 Canvas 文件并提取正文文本(实验二 baseline 专用,不在默认工具集里)"""

    name = "canvas_read_file_content"
    description = (
        "下载并提取指定文件的完整正文文本(支持 pdf/pptx/docx 等)。"
        "返回该文件的全部内容,超过 50000 字符会截断并明确标注。"
        "需要先用 canvas_search_files 或 canvas_get_files 获得 file_id"
    )

    parameters = {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "文件ID(从 canvas_search_files 或 canvas_get_files 的输出里获得)"
            }
        },
        "required": ["file_id"],
        "additionalProperties": False
    }

    output_type = "any"

    MAX_CHARS = 50_000

    async def forward(self, file_id: str) -> ToolResult:
        import tempfile
        from pathlib import Path

        info = await self._make_request("GET", f"files/{file_id}")
        if isinstance(info, dict) and "error" in info:
            return ToolResult(output=None, error=f"获取文件信息失败: {info['error']}")

        url = info.get("url")
        display_name = info.get("display_name") or info.get("filename") or f"file-{file_id}"
        if not url:
            return ToolResult(output=None, error=f"文件 {display_name} 没有可用的下载链接")

        suffix = Path(display_name).suffix or ".bin"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status != 200:
                        return ToolResult(output=None,
                                          error=f"下载失败 (HTTP {resp.status}): {display_name}")
                    data = await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return ToolResult(output=None, error=f"下载失败: {type(e).__name__}: {e}")

        def _extract() -> str:
            from markitdown import MarkItDown
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                return MarkItDown().convert(tmp_path).text_content
            finally:
                os.unlink(tmp_path)

        try:
            text = await asyncio.to_thread(_extract)
        except Exception as e:  # noqa: BLE001 - 提取失败必须显式报错,不能静默变成空内容
            return ToolResult(output=None,
                              error=f"无法提取 {display_name} 的文本 ({type(e).__name__}: {e})。"
                                    f"该格式可能不受支持")

        if not text or not text.strip():
            return ToolResult(output=None,
                              error=f"{display_name} 提取结果为空(可能是扫描件或纯图片文件)")

        header = f"文件: {display_name} (file_id={file_id}, 提取 {len(text)} 字符)\n"
        if len(text) > self.MAX_CHARS:
            body = text[: self.MAX_CHARS] + f"\n[truncated at {self.MAX_CHARS} chars]"
        else:
            body = text
        return ToolResult(output=header + body, error=None)



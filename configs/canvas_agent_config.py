"""
Canvas Student Agent configuration.

Defines an agent tailored for student access to Canvas LMS with the complete Canvas API toolset available to student accounts.
"""

import os


def _planning_interval():
    """Return the planning interval, or None when disabled (0 / empty / unset)."""
    raw = os.getenv("AGENT_PLANNING_INTERVAL", "0").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


from src.tools.canvas_tools import (
    CanvasListCourses,
    CanvasGetAssignments,
    # CanvasSubmitAssignment,
    CanvasGetModules,
    CanvasGetModuleItems,
    CanvasGetFiles,
    CanvasGetFileInfo,
    CanvasGetFolders,
    CanvasGetFolderFiles,
    CanvasSearchFiles,
    CanvasGetDiscussions,
    # CanvasPostDiscussion,
    CanvasGetAnnouncements,
    CanvasGetCalendarEvents,
    CanvasGetGrades,
    CanvasGetPages,
    CanvasGetPageContent,
    CanvasGetQuizzes,
    CanvasGetTodoItems,
    CanvasGetUpcomingEvents,
    CanvasGetGroups,
    VectorStoreList,
    VectorStoreSearch,
    VectorStoreListFiles,
    VectorStoreGetFile,
)

# Canvas student agent configuration
canvas_student_agent_config = dict(
    type="general_agent",
    name="canvas_student_agent",
    description="Canvas LMS study assistant that helps manage courses, assignments, discussions, and more",
    model_id=os.getenv("AGENT_MODEL_ID", "gpt-4.1-mini"),  # Azure 上填 deployment name；见 .env
    max_steps=int(os.getenv("AGENT_MAX_STEPS", "15")),
    # Periodic re-planning. Disabled by default: the framework's _generate_planning_step()
    # needs prompt_templates["planning"], which general_agent.yaml does not define yet,
    # so any value > 0 currently raises KeyError: 'planning'.
    planning_interval=_planning_interval(),
    template_path="src/agent/general_agent/prompts/general_agent.yaml",  # Prompt template path
    
    # Initialize every Canvas tool available to students
    tools=[
        CanvasListCourses(),              # List enrolled courses
        CanvasGetAssignments(),           # Retrieve assignments
        # CanvasSubmitAssignment(),         # Submit an assignment
        CanvasGetModules(),               # Retrieve course modules
        CanvasGetModuleItems(),           # Retrieve module items
        CanvasGetFiles(),                 # List course files
        CanvasGetFileInfo(),              # Retrieve file details
        CanvasGetFolders(),               # List folders
        CanvasGetFolderFiles(),           # List files inside a folder
        CanvasSearchFiles(),              # Search files by keyword
        CanvasGetDiscussions(),           # Retrieve discussions
        # CanvasPostDiscussion(),           # Post a discussion reply
        CanvasGetAnnouncements(),         # Retrieve announcements
        CanvasGetCalendarEvents(),        # Retrieve calendar events
        CanvasGetGrades(),                # Retrieve grades
        CanvasGetPages(),                 # List course pages
        CanvasGetPageContent(),           # Retrieve page content
        CanvasGetQuizzes(),               # List quizzes
        CanvasGetTodoItems(),             # Retrieve to-do items
        CanvasGetUpcomingEvents(),        # Retrieve upcoming events
        CanvasGetGroups(),                # List groups
        VectorStoreList(),                # List knowledge bases
        VectorStoreSearch(),              # Search knowledge bases
        VectorStoreListFiles(),           # List files inside a knowledge base
        VectorStoreGetFile(),             # Read the contents of a file
    ],
    
    # Customize the system prompt here or edit src/agent/general_agent/prompts/general_agent.yaml
)

# Expose the config for use in main.py and other entry points
agent_config = canvas_student_agent_config


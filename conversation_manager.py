# conversation_manager.py

import contextlib
import copy
import uuid
import contextvars
from datetime import datetime
import time

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal # Added Optional, Tuple, Union
from jarowinkler import jarowinkler_similarity


# jarowinkler_similarity("this is an example".split(), ["this", "is", "a", "example"])
# 0.8666666666666667
import openai
import os
import re
import json

from assistant_personalities import assistant_personalities, assistant_instructions
from schema import ConversationInput
from pydantic import BaseModel, Field, ValidationError, model_validator
from function_definitions import (
    GenerateSimpleConciseAnswerMessage,
    GenerateJSONOutput,
    GeneratePDFOutput,
    GenerateTextOutput,
    GenerateHTMLOutput,
    GeneratePythonScriptOutput,
    GenerateCodeSnippetOutput,
    GenerateCSVOutput,
)
from output_generator import (
    ArtifactRef, artifact_run_scope, get_active_run_id, get_tracked_artifact_ref, read_tracked_artifact, resolve_output_filepath,
    generate_simple_concise_answer_message,
    generate_json_output,
    generate_pdf_output,
    generate_text_file_output,
    generate_html_output,
    generate_python_script,
    generate_code_snippet,
    generate_csv_output,
)
from provider_client import (
    DisallowedModelError,
    LazyClient,
    ProviderPolicyError,
    RunBudgetExceededError,
    RunCancelledError,
    RunTimeoutError,
    consume_provider_call,
    get_active_execution_context,
    get_effective_model,
    get_effective_temperature,
    provider_execution_scope,
    reserve_call_scope,
)


# Custom Exception Classes for Type-Safe Error Handling
class ConversationManagerError(Exception):
    """Base exception for conversation manager errors."""
    pass

class OutputTypeError(ConversationManagerError):
    """Exception raised when output type determination fails."""
    pass

class LLMCallError(ConversationManagerError):
    """Exception raised when LLM API calls fail."""
    pass

class ConversationValidationError(ConversationManagerError):
    """Exception raised when data validation fails."""
    pass

# Error Result Classes for Type-Safe Error Handling
class ErrorResult(BaseModel):
    """Structured error result to replace error strings."""
    error_type: str = Field(..., description="Type of error that occurred")
    error_message: str = Field(..., description="Human-readable error message")
    error_details: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the error occurred")


# Configuration constants for output type matching
DEFAULT_FALLBACK_OUTPUT_TYPE = "Simple Concise Answer"
MINIMUM_SIMILARITY_THRESHOLD = 0.1


_ACTIVE_RUN_OPTIONS: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "aps_conversation_run_options", default=None
)


@contextlib.contextmanager
def _conversation_run_scope(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
    max_calls: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    parent_ctx = get_active_execution_context()
    if parent_ctx is not None:
        if model and str(model).strip():
            get_effective_model(explicit=model)
        state: Dict[str, Any] = {
            "model": parent_ctx.default_model,
            "temperature": temperature if temperature is not None else parent_ctx.temperature,
            "deadline_monotonic": parent_ctx.deadline_monotonic,
            "max_calls": parent_ctx.max_calls,
            "calls_made": parent_ctx.calls_made,
            "cancel_check": cancel_check or parent_ctx.cancel_check,
        }
        token = _ACTIVE_RUN_OPTIONS.set(state)
        try:
            yield state
        finally:
            _ACTIVE_RUN_OPTIONS.reset(token)
        return

    normalized_max_calls = (
        int(max_calls) if max_calls is not None and int(max_calls) > 0 else None
    )
    with provider_execution_scope(
        default_model=model or "gpt-4o-mini",
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_calls=normalized_max_calls,
        cancel_check=cancel_check,
    ) as ctx:
        state = {
            "model": ctx.default_model,
            "temperature": ctx.temperature,
            "deadline_monotonic": ctx.deadline_monotonic,
            "max_calls": ctx.max_calls,
            "calls_made": ctx.calls_made,
            "cancel_check": ctx.cancel_check,
        }
        token = _ACTIVE_RUN_OPTIONS.set(state)
        try:
            yield state
        finally:
            _ACTIVE_RUN_OPTIONS.reset(token)


def _resolve_model(default: str = "gpt-4o-mini", explicit: Optional[str] = None) -> str:
    return get_effective_model(explicit=explicit, fallback=default)


def _resolve_temperature(explicit: Optional[float] = None) -> Optional[float]:
    return get_effective_temperature(explicit=explicit)


def _consume_run_budget() -> None:
    consume_provider_call()

# Safe dictionary access helpers
def safe_get_message_field(msg: Dict[str, Any], field: str, default: str = "") -> str:
    """Safely get a field from a message dictionary."""
    if not isinstance(msg, dict):
        return default
    return msg.get(field, default)

def safe_get_feedback_field(feedback: Dict[str, Any], field: str, default: str = "") -> str:
    """Safely get a field from a feedback dictionary."""
    if not isinstance(feedback, dict):
        return default
    return feedback.get(field, default)

def validate_message_structure(msg: Dict[str, Any]) -> bool:
    """Validate that a message has the required structure."""
    if not isinstance(msg, dict):
        return False
    required_fields = ['role', 'content']
    return all(field in msg for field in required_fields)

def normalize_message(value: Any, default_role: str = 'assistant', default_name: str = '') -> Dict[str, Any]:
    """Normalize a message from various sources into canonical dict shape."""
    if isinstance(value, dict):
        return {
            'role': value.get('role', default_role),
            'content': str(value.get('content', '')),
            'name': value.get('name', default_name),
        }
    if isinstance(value, str):
        return {'role': default_role, 'content': value, 'name': default_name}
    if isinstance(value, ErrorResult) or (hasattr(value, 'error_message') and hasattr(value, 'error_type')):
        raise ValueError(
            f"Cannot normalize ErrorResult as a valid conversation message: {getattr(value, 'error_message', str(value))}"
        )
    # Handle ConversationInput or other model objects
    if hasattr(value, 'content'):
        return {
            'role': getattr(value, 'role', default_role),
            'content': str(value.content),
            'name': getattr(value, 'name', default_name),
        }
    return {'role': default_role, 'content': str(value), 'name': default_name}

class OutputType(BaseModel):
    output_type: str = Field(..., description="The type of output to generate.")
    file_extension: str = Field(..., description="The file extension for the output.")

class ConversationOutput(BaseModel):
    """Explicit producer output; internal artifact paths are never display content."""
    kind: Literal["text", "artifact", "failure"]
    content: str = ""
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    output_type: OutputType = Field(default_factory=lambda: OutputType(output_type="Simple Concise Answer", file_extension=".txt"))
    error: Optional[ErrorResult] = None

    @model_validator(mode="after")
    def validate_kind(self):
        if self.kind == "failure":
            if self.error is None or self.content or self.artifacts:
                raise ValueError("A failed conversation output must contain only an error.")
        elif self.error is not None or not self.content.strip():
            raise ValueError("A successful conversation output requires content and no error.")
        elif (self.kind == "artifact") != bool(self.artifacts):
            raise ValueError("Only artifact outputs carry registered artifact references.")
        if self.kind == "artifact" and len(self.artifacts) != 1:
            raise ValueError("An artifact output requires exactly one registered reference.")
        return self

    def legacy_value(self):
        self.validate_kind()
        if self.kind == "failure":
            return self.error
        if self.kind == "artifact":
            return self.artifacts[0].path
        return self.content


class ConversationResult(BaseModel):
    """Copied conversation metadata paired with an explicitly classified outcome."""
    output: ConversationOutput
    conversation: List[Dict[str, Any]] = Field(default_factory=list)
    questions: List[Dict[str, Any]] = Field(default_factory=list)
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def legacy_tuple(self):
        return (copy.deepcopy(self.conversation), copy.deepcopy(self.questions), self.output.legacy_value())


class _ConversationProgress:
    """Per-invocation live lists, used only to retain diagnostics on ordinary failure."""
    def __init__(self):
        self.conversation = []
        self.questions = []


def _normalized_questions(questions):
    normalized = []
    for question in questions:
        if isinstance(question, dict):
            text = question.get("question") or question.get("reply")
            if isinstance(text, str) and text.strip():
                normalized.append({**copy.deepcopy(question), "assistant": str(question.get("assistant", "Assistant")), "question": text})
    return normalized


def conversation_output_annotation(output: ConversationOutput) -> str:
    if output.kind == "failure":
        return "Output failed: the requested deliverable was not generated."
    if output.kind == "artifact":
        ref = output.artifacts[0]
        return f"Generated artifact: {ref.filename[:120]} ({ref.mime_type[:80]}, {ref.size_bytes} bytes)."
    preview = output.content[:1000]
    return f"Final Output: {preview}" + ("…" if len(output.content) > 1000 else "")


class ConversationResultError(ConversationManagerError):
    def __init__(self, output: ConversationOutput, result: Optional[ConversationResult] = None):
        self.result = (result or ConversationResult(output=output)).model_copy(deep=True)
        self.output = output.model_copy(deep=True)
        super().__init__(output.error.error_message if output.error else "Conversation result is unavailable.")


def failure_output(error_type: str, message: str, output_type: Optional[OutputType] = None) -> ConversationOutput:
    return ConversationOutput(kind="failure", error=ErrorResult(error_type=error_type, error_message=message),
                              output_type=output_type or OutputType(output_type="Simple Concise Answer", file_extension=".txt"))


def artifact_conversation_output(filepath: str, output_type: Optional[OutputType] = None) -> ConversationOutput:
    """Resolve only active-owner registered files; never open an arbitrary model string."""
    ref = get_tracked_artifact_ref(filepath)
    if ref is None or ref.run_id != get_active_run_id():
        return failure_output("ArtifactUnavailable", "Conversation artifact is unavailable for this run.", output_type)
    try:
        available = os.path.isfile(ref.path) and os.path.getsize(ref.path) > 0
    except OSError:
        available = False
    if not available:
        return failure_output("ArtifactUnavailable", "Conversation artifact is missing or unreadable.", output_type)
    inferred = output_type or OutputType(output_type="PDF File" if ref.mime_type == "application/pdf" else "Text File", file_extension=os.path.splitext(ref.filename)[1])
    if ref.mime_type == "application/pdf":
        try:
            with open(ref.path, "rb") as handle:
                if not handle.read(5).startswith(b"%PDF-"):
                    return failure_output("ArtifactUnavailable", "Conversation artifact is not a readable PDF.", inferred)
        except OSError:
            return failure_output("ArtifactUnavailable", "Conversation artifact could not be read.", inferred)
        content = f"Generated PDF artifact: {ref.filename}"
    else:
        content = read_tracked_artifact(ref.path)
        if content is None or not content.strip():
            return failure_output("ArtifactUnavailable", "Conversation artifact could not be read or is empty.", inferred)
    return ConversationOutput(kind="artifact", content=content, artifacts=[ref], output_type=inferred)


def resolve_conversation_output(output: ConversationOutput) -> ConversationOutput:
    if output.kind != "artifact":
        return output.model_copy(deep=True)
    if len(output.artifacts) != 1:
        return failure_output("ArtifactUnavailable", "Conversation artifact reference is invalid.", output.output_type)
    requested = output.artifacts[0]
    owned = get_tracked_artifact_ref(requested.path)
    if (owned is None or requested.run_id != get_active_run_id()
            or requested.artifact_id != owned.artifact_id or requested.run_id != owned.run_id):
        return failure_output("ArtifactUnavailable", "Conversation artifact is unavailable for this run.", output.output_type)
    return artifact_conversation_output(requested.path, output.output_type)


# Default fallback values for type safety
DEFAULT_OUTPUT_TYPE = OutputType(output_type="simple_text", file_extension="txt")


output_functions = [
    {
        "name": "generate_simple_concise_answer_message",
        "description": "Generate a simple concise text answer message.",
        "parameters": GenerateSimpleConciseAnswerMessage.model_json_schema(),
        "required": ["final_decision"],
        "additionalProperties": False,
    },
    {
        "name": "generate_json_output",
        "description": "Generate a JSON file from the provided data.",
        "parameters": GenerateJSONOutput.model_json_schema(),
        "required": ["final_decision", "filename"],
        "additionalProperties": False,
    },
    {
        "name": "generate_pdf_output",
        "description": "Generate a PDF file from the provided data.",
        "parameters": GeneratePDFOutput.model_json_schema(),
        "required": ["final_decision", "filename"],
        "additionalProperties": False,
    },
    {
        "name": "generate_text_output",
        "description": "Generate a text file from the provided data.",
        "parameters": GenerateTextOutput.model_json_schema(),
        "required": ["final_decision", "filename"],
        "additionalProperties": False,
    },
    {
        "name": "generate_html_output",
        "description": "Generate an HTML file from the provided data.",
        "parameters": GenerateHTMLOutput.model_json_schema(),
        "required": ["final_decision", "filename"],
        "additionalProperties": False,
    },
    {
        "name": "generate_python_script_output",
        "description": "Generate a Python script file from the provided data.",
        "parameters": GeneratePythonScriptOutput.model_json_schema(),
        "required": ["final_decision", "filename"],
        "additionalProperties": False,
    },
    {
        "name": "generate_code_snippet_output",
        "description": "Generate a code snippet file from the provided data.",
        "parameters": GenerateCodeSnippetOutput.model_json_schema(),
        "required": ["final_decision", "filename"],
    },
    {
        "name": "generate_csv_output",
        "description": "Generate a CSV file from the provided data.",
        "parameters": GenerateCSVOutput.model_json_schema(),
        "required": ["final_decision", "filename"],
    },
]


tools = [
    openai.pydantic_function_tool(
        GenerateSimpleConciseAnswerMessage,
        name="generate_simple_concise_answer_message",
        description="Generate a simple concise text answer message.",
    ),
    openai.pydantic_function_tool(
        GenerateJSONOutput,
        name="generate_json_output",
        description="Generate a JSON file from the provided data.",
    ),
    openai.pydantic_function_tool(
        GeneratePDFOutput,
        name="generate_pdf_output",
        description="Generate a PDF file from the provided data.",
    ),
    openai.pydantic_function_tool(
        GenerateTextOutput,
        name="generate_text_output",
        description="Generate a text file from the provided data.",
    ),
    openai.pydantic_function_tool(
        GenerateHTMLOutput,
        name="generate_html_output",
        description="Generate an HTML file from the provided data.",
    ),
    openai.pydantic_function_tool(
        GeneratePythonScriptOutput,
        name="generate_python_script_output",
        description="Generate a Python script file from the provided data.",
    ),
    openai.pydantic_function_tool(
        GenerateCodeSnippetOutput,
        name="generate_code_snippet_output",
        description="Generate a code snippet file from the provided data.",
    ),
    openai.pydantic_function_tool(
        GenerateCSVOutput,
        name="generate_csv_output",
        description="Generate a CSV file from the provided data.",
    ),
]

output_type_experts = {
    "Simple Concise Answer": [
        "Strategist",
        "Educator",
        "Software Engineer",
        "Project Manager",
        "Business Analyst",
    ],
    "JSON": ["Software Engineer", "Software Architect", "Code Debugger"],
    "PDF": [
        "Legal Expert",
        "Financial Analyst",
        "Healthcare Professional",
        "Marketing Specialist",
    ],
    "Text File": ["Educator", "Software Engineer", "Business Analyst"],
    "HTML": ["UX Designer", "UI Designer", "Software Engineer"],
    "Python Script": ["Software Engineer", "Software Architect", "Code Debugger"],
    "Code Snippet": ["Coding Guru", "Software Engineer", "Code Debugger"],
    "CSV": ["Financial Analyst", "Business Analyst", "Marketing Specialist"],
}

# Set your OpenAI API key as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
client = LazyClient()


class ArgumentItem(BaseModel):
    round: int = Field(..., description="The round number when this argument was recorded.")
    arguments: Union[str, List[str]] = Field(..., description="Argument statement or list of statements.")
    speaker: Optional[str] = Field(None, description="Assistant or speaker name who presented the argument.")

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class DecisionItem(BaseModel):
    round: int = Field(..., description="The round number when this decision was established.")
    decision: str = Field(..., description="The decision content.")
    decided_by: Optional[str] = Field(None, description="Party or mediator who established the decision.")

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class ToDoItem(BaseModel):
    task: str = Field(..., description="Description of the task to be completed.")
    status: str = Field("pending", description="Current status of the task ('pending', 'in_progress', 'completed').")
    assigned_to: Optional[str] = Field(None, description="Assistant or role assigned to the task.")

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __str__(self) -> str:
        return self.task


class ConversationMemory:
    def __init__(self, num_rounds: int = 3) -> None:
        self.facts: List[str] = []
        self.arguments: List[ArgumentItem] = []
        self.decisions: List[DecisionItem] = []
        self.direct_replies: Dict[str, str] = {}
        self.recommended_actions: List[str] = []
        self.to_do_list: List[ToDoItem] = []
        self.completed_tasks: Dict[str, str] = {}
        self.rounds_left: int = num_rounds
        self.decided_output_type: Optional[OutputType] = None

    def add_fact(self, fact: Any) -> None:
        if isinstance(fact, list):
            for f in fact:
                self.add_fact(f)
            return
        if fact and isinstance(fact, str):
            fact_clean = fact.strip()
            if fact_clean and fact_clean not in self.facts:
                self.facts.append(fact_clean)

    def add_argument(
        self,
        argument: Any,
        round_num: int = 1,
        speaker: Optional[str] = None,
    ) -> None:
        if not argument:
            return
        if isinstance(argument, list):
            for arg in argument:
                self.add_argument(arg, round_num=round_num, speaker=speaker)
            return
        if isinstance(argument, ArgumentItem):
            item = argument
        elif isinstance(argument, dict):
            item = ArgumentItem(
                round=argument.get("round", round_num),
                arguments=argument.get("arguments", str(argument)),
                speaker=argument.get("speaker", speaker),
            )
        else:
            item = ArgumentItem(round=round_num, arguments=argument, speaker=speaker)

        if item not in self.arguments:
            self.arguments.append(item)

    def add_decision(
        self,
        decision: Any,
        round_num: int = 1,
        decided_by: Optional[str] = None,
    ) -> None:
        if not decision:
            return
        if isinstance(decision, list):
            for dec in decision:
                self.add_decision(dec, round_num=round_num, decided_by=decided_by)
            return
        if isinstance(decision, DecisionItem):
            item = decision
        elif isinstance(decision, dict):
            item = DecisionItem(
                round=decision.get("round", round_num),
                decision=str(decision.get("decision", "")),
                decided_by=decision.get("decided_by", decided_by),
            )
        else:
            item = DecisionItem(round=round_num, decision=str(decision), decided_by=decided_by)

        if item not in self.decisions:
            self.decisions.append(item)

    def add_direct_reply(self, assistant_name: str, reply: str) -> None:
        if assistant_name and reply:
            self.direct_replies[assistant_name] = reply

    def add_recommended_action(self, action: Any) -> None:
        if isinstance(action, list):
            for act in action:
                self.add_recommended_action(act)
            return
        if action and action not in self.recommended_actions:
            self.recommended_actions.append(action)

    def add_to_do_item(
        self,
        item: Any,
        assigned_to: Optional[str] = None,
    ) -> None:
        if not item:
            return
        if isinstance(item, ToDoItem):
            if item not in self.to_do_list:
                self.to_do_list.append(item)
        elif isinstance(item, dict):
            todo = ToDoItem(
                task=str(item.get("task", "")),
                status=item.get("status", "pending"),
                assigned_to=item.get("assigned_to", assigned_to),
            )
            if todo not in self.to_do_list:
                self.to_do_list.append(todo)
        elif isinstance(item, list):
            for sub_item in item:
                self.add_to_do_item(sub_item, assigned_to=assigned_to)
        else:
            todo = ToDoItem(task=str(item), assigned_to=assigned_to)
            if todo not in self.to_do_list:
                self.to_do_list.append(todo)

    def add_completed_task(self, task: Union[str, ToDoItem], completion: str = "completed") -> None:
        task_str = task.task if isinstance(task, ToDoItem) else str(task)
        if task_str:
            self.completed_tasks[task_str] = completion
            for todo in self.to_do_list:
                if todo.task == task_str or str(todo) == task_str:
                    todo.status = "completed"

    def decrement_rounds_left(self) -> None:
        self.rounds_left -= 1

    def set_decided_output_type(self, output_type: OutputType) -> None:
        self.decided_output_type = output_type

    def compact_memory(self, max_items_per_category: int = 10) -> None:
        """Compact memory items in place, preserving newest facts/arguments and active decisions."""
        if len(self.facts) > max_items_per_category:
            self.facts = self.facts[-max_items_per_category:]
        if len(self.arguments) > max_items_per_category:
            self.arguments = self.arguments[-max_items_per_category:]
        if len(self.recommended_actions) > max_items_per_category:
            self.recommended_actions = self.recommended_actions[-max_items_per_category:]
        pending_todos = [t for t in self.to_do_list if t.status != "completed"]
        completed_todos = [t for t in self.to_do_list if t.status == "completed"]
        if len(completed_todos) > 5:
            completed_todos = completed_todos[-5:]
        self.to_do_list = pending_todos + completed_todos

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete ConversationMemory state to a JSON-compatible dictionary."""
        return {
            "facts": list(self.facts),
            "arguments": [
                arg.model_dump() if hasattr(arg, "model_dump") else (arg.dict() if hasattr(arg, "dict") else dict(arg))
                for arg in self.arguments
            ],
            "decisions": [
                dec.model_dump() if hasattr(dec, "model_dump") else (dec.dict() if hasattr(dec, "dict") else dict(dec))
                for dec in self.decisions
            ],
            "direct_replies": dict(self.direct_replies),
            "recommended_actions": list(self.recommended_actions),
            "to_do_list": [
                todo.model_dump() if hasattr(todo, "model_dump") else (todo.dict() if hasattr(todo, "dict") else dict(todo))
                for todo in self.to_do_list
            ],
            "completed_tasks": dict(self.completed_tasks),
            "rounds_left": self.rounds_left,
            "decided_output_type": (
                self.decided_output_type.model_dump()
                if hasattr(self.decided_output_type, "model_dump")
                else (self.decided_output_type.dict() if hasattr(self.decided_output_type, "dict") else None)
            ) if self.decided_output_type else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """Deserialize a dictionary back into a fully-typed ConversationMemory instance."""
        memory = cls(num_rounds=data.get("rounds_left", 3))
        memory.facts = list(data.get("facts", []))
        for item in data.get("arguments", []):
            memory.add_argument(item)
        for item in data.get("decisions", []):
            memory.add_decision(item)
        memory.direct_replies = dict(data.get("direct_replies", {}))
        memory.recommended_actions = list(data.get("recommended_actions", []))
        for todo in data.get("to_do_list", []):
            memory.add_to_do_item(todo)
        memory.completed_tasks = dict(data.get("completed_tasks", {}))
        memory.rounds_left = data.get("rounds_left", memory.rounds_left)
        output_type_data = data.get("decided_output_type")
        if output_type_data:
            if isinstance(output_type_data, OutputType):
                memory.decided_output_type = output_type_data
            elif isinstance(output_type_data, dict):
                memory.decided_output_type = OutputType(**output_type_data)
        return memory

    def get_memory_summary(
        self,
        max_tokens: Optional[int] = None,
        max_items_per_category: Optional[int] = None,
    ) -> str:
        summary_sections = []

        facts_list = self.facts
        if max_items_per_category and len(facts_list) > max_items_per_category:
            omitted = len(facts_list) - max_items_per_category
            facts_list = facts_list[-max_items_per_category:]
            summary_sections.append(
                f"Known Facts (latest {max_items_per_category}, {omitted} earlier omitted):\n"
                + "\n".join(f"- {fact}" for fact in facts_list)
            )
        elif facts_list:
            summary_sections.append(
                "Known Facts:\n" + "\n".join(f"- {fact}" for fact in facts_list)
            )

        args_list = self.arguments
        if max_items_per_category and len(args_list) > max_items_per_category:
            omitted = len(args_list) - max_items_per_category
            args_list = args_list[-max_items_per_category:]
            formatted_args = []
            for arg in args_list:
                val = arg.arguments if hasattr(arg, "arguments") else arg.get("arguments", str(arg))
                formatted_args.append(f"- {val}")
            summary_sections.append(
                f"Arguments (latest {max_items_per_category}, {omitted} earlier omitted):\n"
                + "\n".join(formatted_args)
            )
        elif args_list:
            formatted_args = []
            for arg in args_list:
                val = arg.arguments if hasattr(arg, "arguments") else arg.get("arguments", str(arg))
                formatted_args.append(f"- {val}")
            summary_sections.append("Arguments:\n" + "\n".join(formatted_args))

        if self.decisions:
            formatted_decisions = []
            for dec in self.decisions:
                val = dec.decision if hasattr(dec, "decision") else dec.get("decision", str(dec))
                formatted_decisions.append(f"- {val}")
            summary_sections.append("Decisions:\n" + "\n".join(formatted_decisions))

        actions_list = self.recommended_actions
        if max_items_per_category and len(actions_list) > max_items_per_category:
            actions_list = actions_list[-max_items_per_category:]
        if actions_list:
            summary_sections.append(
                "Recommended Actions:\n" + "\n".join(f"- {action}" for action in actions_list)
            )

        if self.decided_output_type:
            summary_sections.append(
                f"Decided Output Type of Final Output: {self.decided_output_type.output_type}"
            )

        if self.to_do_list:
            pending_items = []
            for item in self.to_do_list:
                task_str = item.task if hasattr(item, "task") else (item.get("task", str(item)) if isinstance(item, dict) else str(item))
                status = item.status if hasattr(item, "status") else (item.get("status", "pending") if isinstance(item, dict) else "pending")
                if status != "completed" and task_str not in self.completed_tasks:
                    pending_items.append(f"- {task_str}")
            if pending_items:
                summary_sections.append("To Do List:\n" + "\n".join(pending_items))

        if self.completed_tasks:
            summary_sections.append(
                "Completed Tasks:\n"
                + "\n".join(f"- {task}: {comp}" for task, comp in self.completed_tasks.items())
            )

        summary_sections.append(
            f"Rounds Left in Conversation to Complete Solution to Problem Statement: {self.rounds_left}"
        )

        full_summary = "\n\n".join(summary_sections)

        if max_tokens and len(full_summary) > max_tokens * 4:
            core_sections = []
            if self.decided_output_type:
                core_sections.append(f"Decided Output Type: {self.decided_output_type.output_type}")
            if self.decisions:
                core_sections.append("Key Decisions:\n" + "\n".join(f"- {getattr(d, 'decision', str(d))}" for d in self.decisions[-3:]))
            core_sections.append(f"Rounds Left: {self.rounds_left}")
            truncated_summary = "[Summary Compacted due to token limit]\n\n" + "\n\n".join(core_sections)
            return truncated_summary

        return full_summary


def is_question(response_content: str) -> bool:
    # Simple heuristic: ends with a question mark or contains question words
    question_words = [
        "what",
        "why",
        "how",
        "when",
        "who",
        "where",
        "which",
        "whom",
        "whose",
        "could",
        "can",
        "would",
        "should",
        "do",
        "does",
        "did",
        "is",
        "are",
        "am",
        "was",
        "were",
    ]
    pattern = re.compile(r"\b(" + "|".join(question_words) + r")\b", re.IGNORECASE)
    return response_content.strip().endswith("?") or bool(
        pattern.search(response_content)
    )


def summarize_conversation(conversation_history: List[Dict[str, str]]) -> str:
    # Concatenate all messages into one text block
    conversation_text = "\n".join(
        f"{safe_get_message_field(msg, 'name') or safe_get_message_field(msg, 'role', 'Unknown')}: {safe_get_message_field(msg, 'content', 'No content')}"
        for msg in conversation_history
        if validate_message_structure(msg)
    )

    # Use the OpenAI client to summarize the conversation
    try:
        with reserve_call_scope(
            model=_resolve_model("gpt-4o-mini"),
            call_type="summarize_conversation",
        ) as resolved_model:
            completion = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes conversations. Your summary should be concise and capture the key points of the conversation, and correctly attribute statements to the right speakers.",
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following conversation:\n\n{conversation_text}",
                    },
                ],
                max_completion_tokens=256,
            )

        # Extract the summary from the completion response
        summary = completion.choices[0].message.content.strip()

    except (ProviderPolicyError, DisallowedModelError):
        raise
    except Exception as e:
        summary = f"Error in summarization: {str(e)}"

    return summary


class DirectReply(BaseModel):
    assistant_name: str = Field(..., description="The name of the assistant.")
    reply: str = Field(..., description="The direct reply from the assistant.")


class ExtractFormat(BaseModel):
    Facts: list[str] = Field(
        default_factory=list,
        description="List of factual statements extracted from the conversation.",
    )
    Arguments: list[str] = Field(
        default_factory=list,
        description="List of arguments extracted from the conversation.",
    )
    Direct_Replies: DirectReply = Field(
        default_factory=DirectReply,
        description="Direct replies from one assistant to another.",
    )
    Decisions: list[str] = Field(
        default_factory=list,
        description="List of decisions extracted from the conversation.",
    )
    Recommended_Actions: list[str] = Field(
        default_factory=list,
        description="List of recommended actions extracted from the conversation.",
    )
    To_Do_List: list[str] = Field(
        default_factory=list,
        description="List of items that need to be addressed or completed.",
    )


def extract_information(
    assistant_response: str, # Assuming assistant_response is a string
    conversation_memory: ConversationMemory,
    assistants_list: List[str],
    round_num: int,
) -> ConversationMemory:
    """
    Extracts facts, decisions, and user preferences from the assistant's response using OpenAI's language model.
    """
    # Define the system and user messages
    original_conversation = conversation_memory
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that extracts key information from text transcripts of a debate or discussion between the following AI assistants: {assistants_list}.",
        },
        {
            "role": "user",
            "content": f"""
Extract the following information from the text below and provide it in JSON format:

Text:
\"\"\"
{assistant_response}
\"\"\"

Instructions:
- Identify and list any factual statements under \"Facts\". Keep them concise.
- Identify any arguments for any side of the debate and list them under \"Arguments\" in summarized form.
- Identify any replies, responses, questions or statements that are clearly directed to another assistant by name, list it verbatim under \"Direct Replies\" in this format: \"[{{\"assistant_name\": \"reply\"}}]\"."
- Identify and list any decisions made or recommended under \"Decisions\".
- Identify and list any calls to action or suggestions for any particular course of action under \"Recommended Actions\".
- Identify and list any mentions of things the group still needs to address, discuss or figure out under \"To Do List\".
- If none are found, return an empty list for that category.
- Anything of one category can also exist in other categories (in appropriate format) if it fits in both.

Please return the output as a JSON object with the following structure:
{{
  "Facts": [...],
  "Arguments": [...],
  "Direct Replies": ["{{...}}"],
  "Decisions": [...]
  "Recommended Actions": [...]
  "To Do List": [...]
}}
""",
        },
    ]

    try:
        # Make the API call to OpenAI's chat completions
        with reserve_call_scope(
            model=_resolve_model("gpt-4o-mini"),
            call_type="extract_information",
        ) as resolved_model:
            completion = client.beta.chat.completions.parse(
                model=resolved_model,
                messages=messages,
                response_format=ExtractFormat,
                temperature=0,
            )

        # Extract the assistant's response
        extraction = completion.choices[0].message.parsed

        conversation_memory.add_fact(extraction.Facts)
        conversation_memory.add_argument(extraction.Arguments, round_num)
        conversation_memory.add_direct_reply(
            extraction.Direct_Replies.assistant_name, extraction.Direct_Replies.reply
        )
        conversation_memory.add_decision(extraction.Decisions, round_num)
        conversation_memory.add_recommended_action(extraction.Recommended_Actions)
        conversation_memory.add_to_do_item(extraction.To_Do_List)
        original_conversation = None
        return conversation_memory

    except (ProviderPolicyError, DisallowedModelError):
        raise
    except Exception as e:
        print(
            f"Error in extract_information: {str(e)} on line {e.__traceback__.tb_lineno}"
        )
        return original_conversation


def calculate_priority(
    assistant_name: str,
    conversation_history: List[Dict[str, Any]], # Assuming content can be Any for now
    conversation_memory: ConversationMemory,
    assistants_list: List[str],
    round_num: int,
    num_rounds: int,
    lead_personality: str,
    output_type: OutputType,
) -> float:
    # Prioritize assistants who haven't spoken recently
    last_spoken = next(
        (
            ((len(conversation_history) - idx) * 0.1)
            for idx, msg in enumerate(reversed(conversation_history), 1)
            if msg.get("name") == assistant_name
        ),
        None,
    )
    priority = last_spoken if last_spoken else 0.0
    # Prioritize the lead personality to ensure they have a chance to guide the conversation
    if assistant_name == lead_personality:
        priority += 0.5 * (round_num + 1.0 / num_rounds)
    # Prioritize assistants who have not yet contributed to the conversation
    if assistant_name not in conversation_memory.direct_replies:
        priority += 0.1 * (round_num + 1.0 / num_rounds)

    # Map output type to assistants that are experts in that area

    # Prioritize assistants based on their personalities and the output type
    if assistant_name in assistant_personalities:
        personality = assistant_personalities[assistant_name]
        if personality in output_type_experts.get(output_type.output_type, []):
            priority += 0.5 * (round_num + 1.0 / num_rounds)

    return priority


class BinaryVote(BaseModel):
    vote: bool = Field(..., description="The binary vote cast by the assistant.")


class ConfidenceVote(BaseModel):
    confidence: float = Field(
        ..., description="The confidence score cast by the assistant."
    )


def cast_binary_vote(assistant_name: str, messages: List[Dict[str, str]], vote_prompt: str) -> Union[BinaryVote, ErrorResult]:
    """Cast a binary vote with proper error handling."""
    # Ask the assistant to cast a vote based on the messages
    messages.append({"role": "system", "content": vote_prompt})
    try:
        with reserve_call_scope(
            model=_resolve_model("gpt-4o-mini"),
            call_type="cast_binary_vote",
        ) as resolved_model:
            response = client.beta.chat.completions.parse(
                model=resolved_model,
                messages=messages,  # Pass the messages directly as a list of dicts
                response_format=BinaryVote,
            )
        # Extract the vote from the response
        vote = response.choices[0].message.parsed
        return vote
    except (ProviderPolicyError, DisallowedModelError):
        raise
    except Exception as e:
        return ErrorResult(
            error_type="LLMCallError",
            error_message=f"Failed to cast binary vote for {assistant_name}",
            error_details=str(e)
        )


def cast_confidence_vote(
    assistant_name: str,
    messages: List[Dict[str, str]],
    vote_prompt: str = "Please cast a confidence vote between 0 and 1.",
) -> Union[ConfidenceVote, ErrorResult]:
    """Cast a confidence vote with proper error handling."""
    # Ask the assistant to cast a confidence vote based on the messages
    messages.append({"role": "system", "content": vote_prompt})
    try:
        with reserve_call_scope(
            model=_resolve_model("gpt-4o-mini"),
            call_type="cast_confidence_vote",
        ) as resolved_model:
            response = client.beta.chat.completions.parse(
                model=resolved_model,
                messages=messages,  # Pass the messages directly as a list of dicts
                response_format=ConfidenceVote,
            )
        # Extract the vote from the response
        confidence = response.choices[0].message.parsed
        return confidence
    except (ProviderPolicyError, DisallowedModelError):
        raise
    except Exception as e:
        return ErrorResult(
            error_type="LLMCallError",
            error_message=f"Failed to cast confidence vote for {assistant_name}",
            error_details=str(e)
        )


def define_problem(problem_statement, selected_personalities, previous_def_rounds=0):
    # Obtain consensus on the problem statement
    definitions = []
    for assistant_name, assistant_prompt in selected_personalities.items():
        messages = [
            {
                "role": "system",
                "content": f"{assistant_prompt} You are tasked with defining the problem statement for the debate. Your goal is to collaboratively agree on a clear and concise problem statement that captures the essence of the issue at hand.",
            },
            {
                "role": "user",
                "content": f"Please define the problem statement based on the following prompt:\n\n{problem_statement}",
            },
            {
                "role": "system",                "content": "Please provide a problem statement that is specific, actionable, and relevant to the topic. Ensure that it is clear and concise to guide the discussion effectively. Include criteria for evaluating potential solutions as well as a description of the desired output. Keep it limited to a few sentences and avoid ambiguity. Do NOT write in the form of a question, only as an actionable statement.",
            },
        ]
        try:
            with reserve_call_scope(
                model=_resolve_model("gpt-4o-mini"),
                call_type="define_problem",
            ) as resolved_model:
                response = client.chat.completions.create(
                    model=resolved_model,
                    messages=messages,  # Pass the messages directly as a list of dicts
                )
            # Extract the text from the response
            assistant_reply = response.choices[
                0
            ].message.content.strip()  # Use 'message' key, not 'text'
            definitions.append(
                {
                    "assistant": assistant_name,
                    "definition": assistant_reply,
                    "confidence": 0,
                }
            )
            print(f"\nDefinition from {assistant_name}: {assistant_reply}\n")
        except (ProviderPolicyError, DisallowedModelError):
            raise
        except Exception as e:
            # TODO: Handle LLM call failures more gracefully
            error_msg = f"Failed to get definition from {assistant_name}: {str(e)}"
            print(f"\nError: {error_msg}\n")
            definitions.append(
                {
                    "assistant": assistant_name,
                    "definition": f"Error in definition generation: {str(e)}",
                    "confidence": 0,
                }
            )
    for definition in definitions:
        for assistant_name, assistant_prompt in selected_personalities.items():
            # Ask each assistant to vote on the problem statement
            messages = [
                {
                    "role": "system",
                    "content": f"{assistant_prompt} Please review the problem statement defined by {definition['assistant']} and cast your vote on its clarity, relevance, specificity, and actionability, in the form of a confidence score between 0 and 1. Only respond with a number between 0 and 1.",
                },
                {
                    "role": "user",
                    "content": f"Please cast your vote on the problem statement defined by {definition['assistant']}:\n\n{definition['definition']}",
                },
                {
                    "role": "system",
                    "content": f"Problem Statement: {definition['definition']}",
                },
            ]
            confidence = cast_confidence_vote(assistant_name, messages)
            print(
                f"\nConfidence Vote from {assistant_name} on {definition['assistant']}'s definition: {confidence}\n"
            )
            # Handle both successful votes and errors
            if isinstance(confidence, ConfidenceVote):
                definition["confidence"] += confidence.confidence
            else:
                # TODO: Better error handling for failed confidence votes
                print(f"Warning: Failed to get confidence vote from {assistant_name}: {confidence}")
                # Don't add to confidence score if vote failed
    # Select the problem statement with the highest confidence vote
    best_definition = max(definitions, key=lambda x: x["confidence"])
    # Get consensus on the problem statement
    print(
        f"\nBest Definition: {best_definition['definition']} by {best_definition['assistant']} with confidence: {best_definition['confidence']}\n"
    )

    bin_vote_prompt = "Please cast a binary vote to confirm the consensus problem statement. Respond with a boolean value (True/False) only."
    yes_votes = 0
    yes_voters = []
    no_votes = 0
    no_voters = []
    voting_errors = 0
    for assistant_name, assistant_prompt in selected_personalities.items():
        consensus_messages = [
            {
                "role": "system",
                "content": f"{assistant_prompt} Please review the problem statement defined by {best_definition['assistant']} and cast your vote to confirm the consensus problem statement.",
            },
            {
                "role": "user",
                "content": f"Please cast your vote to confirm the consensus problem statement:\n\n{best_definition['definition']}",
            },
            {
                "role": "system",
                "content": f"Problem Statement: {best_definition['definition']}",
            },
        ]
        binary_vote = cast_binary_vote(
            assistant_name, consensus_messages, bin_vote_prompt
        )
        # Handle both successful votes and errors
        if isinstance(binary_vote, BinaryVote):
            if binary_vote.vote:
                yes_votes += 1
                yes_voters.append(assistant_name)
                print(f"\nYes Vote from {assistant_name} on the consensus problem statement.\n")
            else:
                no_votes += 1
                no_voters.append(assistant_name)
                print(f"\nNo Vote from {assistant_name} on the consensus problem statement.\n")
        else:
            # Handle binary vote errors more gracefully
            print(f"⚠️  Error getting vote from {assistant_name}: {binary_vote}")
            voting_errors += 1
        
        # For no votes, ask the assistant to provide feedback on the problem statement  
        if isinstance(binary_vote, BinaryVote) and not binary_vote.vote:
            feedback_messages = [
                {
                    "role": "system",
                    "content": f"{assistant_prompt} Your vote did not confirm the consensus problem statement. Please provide feedback on the problem statement defined by {best_definition['assistant']}.",
                },
                {
                    "role": "user",
                    "content": f"Please provide feedback on the problem statement defined by {best_definition['assistant']} and suggest improvements so that it aligns better with the original prompt and the users intentions and what the user expects the output to be. The original user prompt is:\n\n{problem_statement}",
                },
                {
                    "role": "user",
                    "content": f"Defined Problem Statement: {best_definition['definition']}",
                },
            ]
            feedback = get_assistant_response(
                assistant_name, assistant_prompt, feedback_messages
            )
            print(f"\nFeedback from {assistant_name}: {feedback}\n")
            # Have the author of the problem statement respond to the feedback by either revising the statement or providing justification
            revision_prompt = "Please revise the problem statement based on the feedback provided by the other assistants. If you believe the problem statement is accurate, provide justification for your choice. Begin your response with 'Revision:' or 'Justification:' as appropriate."
            revision_messages = [
                {
                    "role": "system",
                    "content": f"Problem Statement Feedback: {safe_get_feedback_field(feedback, 'content', 'No feedback content')}",
                },
                {"role": "user", "content": f"{revision_prompt}"},
                {
                    "role": "system",
                    "content": f"Feedback from {assistant_name}: {safe_get_feedback_field(feedback, 'content', 'No feedback content')}",
                },
            ]
            revision_response = get_assistant_response(
                best_definition["assistant"],
                selected_personalities[best_definition["assistant"]],
                revision_messages,
            )
            print(
                f"\nRevision Response from {best_definition['assistant']}: {revision_response}\n"
            )
            # Update the problem statement based on the revision or, if justified, allow the assistant that cast the 'no' vote to vote again based on the justification
            revision_content = safe_get_message_field(revision_response, 'content', '')
            if revision_content.startswith("Revision:"):
                best_definition["definition"] = (
                    revision_content.replace("Revision:", "").strip()
                )
            elif revision_content.startswith("Justification:"):
                # Ask the assistant to vote again based on the justification
                justification_messages = [
                    {
                        "role": "system",
                        "content": f"{assistant_prompt} Please review the justification provided by {best_definition['assistant']} and cast your vote to confirm the consensus problem statement.",
                    },
                    {
                        "role": "user",
                        "content": f"Please cast your vote to confirm the consensus problem statement based on the justification provided by {best_definition['assistant']}:\n\n{safe_get_message_field(revision_response, 'content', 'No revision content')}",
                    },
                    {
                        "role": "system",
                        "content": f"Defined Problem Statement: {best_definition['definition']}, Justification: {safe_get_message_field(revision_response, 'content', 'No justification content')}",
                    },
                ]
                binary_vote = cast_binary_vote(
                    assistant_name, justification_messages, bin_vote_prompt
                )
                if isinstance(binary_vote, BinaryVote):
                    if binary_vote.vote:
                        yes_votes += 1
                        yes_voters.append(assistant_name)
                        print(
                            f"\nRevised Vote from {assistant_name} on the consensus problem statement.\n"
                        )
                    else:
                        no_votes += 1
                        no_voters.append(assistant_name)
                        print(
                            f"\nRepeat No Vote from {assistant_name} on the consensus problem statement.\n"                    )
                else:
                    voting_errors += 1
    
    # Check if the consensus problem statement is confirmed - Enhanced with edge case handling
    total_votes = yes_votes + no_votes
    total_assistants = len(selected_personalities)
    
    print(f"📊 Problem Definition Vote Results: {yes_votes} YES, {no_votes} NO (Total: {total_votes}/{total_assistants} assistants)")
    
    # Handle edge cases first
    if total_votes == 0:
        print("❌ CRITICAL: No valid votes received on problem definition!")
        raise Exception("All assistants failed to vote on problem definition. Cannot proceed.")
    
    elif yes_votes == no_votes and yes_votes > 0:
        print(f"⚖️ TIE VOTE on problem definition: {yes_votes} YES vs {no_votes} NO")
        print("🔄 Requiring revision due to tie vote...")
        # For ties, force revision
        if previous_def_rounds < 2:  # Limit revision attempts
            return define_problem(problem_statement, selected_personalities, previous_def_rounds + 1)
        else:
            print("⚠️ Maximum revision attempts reached. Using original problem statement as fallback.")
            return problem_statement
    
    elif total_votes < (total_assistants / 2):
        print(f"⚠️ LOW PARTICIPATION in problem definition: Only {total_votes}/{total_assistants} assistants voted")
        if yes_votes > no_votes and yes_votes >= (total_assistants / 3):
            print("✓ Proceeding despite low participation due to clear majority")
        else:
            print("❌ Insufficient participation and unclear mandate for problem definition")
            if previous_def_rounds < 2:
                return define_problem(problem_statement, selected_personalities, previous_def_rounds + 1)
            else:
                print("⚠️ Maximum attempts reached. Using original problem statement.")
                return problem_statement
    
    # Normal voting logic
    if yes_votes > no_votes and no_votes == 0:
        print(
            f"\nConsensus Problem Statement Confirmed with full consensus: {best_definition['definition']}\n"
        )
        return best_definition["definition"]
    elif yes_votes > no_votes and no_votes > 0 and no_votes < (yes_votes / 2):
        print(
            f"\nConsensus Problem Statement Confirmed with majority vote, {yes_votes} to {no_votes}: {best_definition['definition']}\n"
        )
        return best_definition["definition"]
    else:
        # If the consensus is not confirmed, ask the assistants to vote again on the revised problem statement after all no-voters have provided feedback
        print(
            f"\nConsensus Problem Statement Not Confirmed: {best_definition['definition']}. Voting again after feedback.\n"
        )
        feedbacks = []
        for assistant_name, assistant_prompt in selected_personalities.items():
            if assistant_name in no_voters:
                feedback_messages = [
                    {
                        "role": "system",
                        "content": f"{assistant_prompt} Your vote did not confirm the consensus problem statement. Please provide feedback on the revised problem statement.",
                    },
                    {
                        "role": "user",
                        "content": f"Please provide feedback on the revised problem statement and suggest improvements so that it aligns better with the original prompt and the users intentions and what the user expects the output to be. The original user prompt is:\n\n{problem_statement}",
                    },
                    {
                        "role": "user",
                        "content": f"Revised Problem Statement: {best_definition['definition']}",
                    },
                ]
                feedback = get_assistant_response(
                    assistant_name, assistant_prompt, feedback_messages
                )
                print(f"\nFeedback from {assistant_name}: {feedback}\n")
                feedbacks.append(feedback)

        # Have the author of the problem statement respond to the feedback by either revising the statement or providing justification
        revision_prompt = "Please revise the problem statement based on the feedback provided by the other assistants. If you believe the problem statement is accurate, provide justification for your choice. Begin your response with 'Revision:' or 'Justification:' as appropriate."
        revision_messages = [
            {
                "role": "system",
                "content": f"{selected_personalities[best_definition['assistant']]} {revision_prompt}",
            }
        ]
        for feedback in feedbacks:
            revision_messages.append(
                {
                    "role": "user",
                    "content": f"Feedback from {safe_get_feedback_field(feedback, 'name', 'Unknown')}: {safe_get_feedback_field(feedback, 'content', 'No feedback content')}",
                }
            )

        revision_response = get_assistant_response(
            best_definition["assistant"],
            selected_personalities[best_definition["assistant"]],
            revision_messages,
        )
        print(
            f"\nRevision Response from {best_definition['assistant']}: {revision_response}\n"
        )        # Update the problem statement based on the revision or, if justified, allow the Mediator to rewrite the problem statement based on the feedback and the justification
        if safe_get_message_field(revision_response, 'content', '').startswith("Revision:"):
            best_definition["definition"] = (
                safe_get_message_field(revision_response, 'content', '').replace("Revision:", "").strip()
            )
        elif safe_get_message_field(revision_response, 'content', '').startswith("Justification:"):
            # Ask the Mediator to rewrite the problem statement based on the feedback and the justification
            mediator_prompt = selected_personalities.get(
                "Mediator",
                assistant_personalities.get(
                    "Mediator",
                    "You are a Mediator guiding collaborative consensus.",
                ),
            )
            mediator_messages = [
                {
                    "role": "system",
                    "content": f"{mediator_prompt} The consensus problem statement was not confirmed. Please review the feedback provided by the assistants and the justification for the problem statement.",
                },
                {
                    "role": "user",
                    "content": f"Please rewrite the problem statement based on the feedback provided by the assistants and the justification. The original prompt was:\n\n{problem_statement} and the revised problem statement was:\n\n{best_definition['definition']}",
                },
                {
                    "role": "user",
                    "content": f"Feedback from {best_definition['assistant']}: {safe_get_message_field(revision_response, 'content', 'No revision content')}\n\nFeedback from other assistants: {', '.join(safe_get_feedback_field(feedback, 'content', 'No feedback content') for feedback in feedbacks)}",
                },
            ]
            mediator_response = get_assistant_response(
                "Mediator", mediator_prompt, mediator_messages            )
            print(f"\nMediator Rewrite Response: {safe_get_message_field(mediator_response, 'content', 'No mediator response')}\n")
            best_definition["definition"] = safe_get_message_field(mediator_response, 'content', 'Default problem definition')

        # Ask the assistants to vote again on the revised problem statement
        revised_bin_vote_prompt = "Please cast a binary vote to confirm the revised problem statement. Respond with a boolean value (True/False) only."
        revised_yes_votes = 0
        revised_no_votes = 0
        revised_voting_errors = 0
        for assistant_name, assistant_prompt in selected_personalities.items():
            revised_consensus_messages = [
                {
                    "role": "system",
                    "content": f"{assistant_prompt} Please review the revised problem statement and cast your vote to confirm it.",
                },
                {
                    "role": "user",
                    "content": f"Please cast your vote to confirm the revised problem statement:\n\n{best_definition['definition']}",
                },
                {
                    "role": "system",
                    "content": f"Revised Problem Statement: {best_definition['definition']}",
                },
            ]
            revised_binary_vote = cast_binary_vote(
                assistant_name, revised_consensus_messages, revised_bin_vote_prompt
            )
            if isinstance(revised_binary_vote, BinaryVote):
                if revised_binary_vote.vote:
                    revised_yes_votes += 1
                else:
                    revised_no_votes += 1
            else:
                revised_voting_errors += 1
        # Check if the revised problem statement is confirmed
        if revised_yes_votes > revised_no_votes and revised_no_votes == 0:
            print(
                f"\nRevised Consensus Problem Statement Confirmed with full consensus: {best_definition['definition']}\n"
            )
            return best_definition["definition"]
        elif (
            revised_yes_votes > revised_no_votes
            and revised_no_votes > 0
            and revised_no_votes < (revised_yes_votes / 2)
        ):
            print(
                f"\nRevised Consensus Problem Statement Confirmed with majority vote, {revised_yes_votes} to {revised_no_votes}: {best_definition['definition']}\n"
            )
            return best_definition["definition"]
        else:
            raise Exception(
                "Consensus not reached on the revised problem statement."
            )


def output_type_determination(response) -> Union[OutputType, ErrorResult]:
    """Determine output type with proper error handling."""
    determination_prompt = "Based on the defined problem statement, please suggest an output format that would best suit this solution. Options include simple concise text answer, a detailed report in text or PDF format, a code snippet or script file, structured data in JSON or CSV format, a website or app prototype, or a detailed technical document. Please provide your recommendation in the provided format, generating both the specific output type (such as 'Manuscript', 'Website Prototype', 'Categorical Data', Python Script', etc.) and the file extension (such as 'txt', 'pdf', 'html', 'json', 'py', etc.)."
    messages = [
        {"role": "system", "content": determination_prompt},
        {
            "role": "user",
            "content": f"Please suggest an output format based on the defined problem statement:\n\n{response}",
        },
    ]
    try:
        with reserve_call_scope(
            model=_resolve_model("gpt-4o-mini"),
            call_type="output_type_determination",
        ) as resolved_model:
            response = client.beta.chat.completions.parse(
                model=resolved_model,
                messages=messages,  # Pass the messages directly as a list of dicts
                response_format=OutputType,
                max_completion_tokens=100,
            )
        # Extract the output type from the response
        output_type = response.choices[0].message.parsed
        return output_type
    except (ProviderPolicyError, DisallowedModelError):
        raise
    except Exception as e:
        return ErrorResult(
            error_type="OutputTypeError",
            error_message="Failed to determine output type",
            error_details=str(e)
        )


def test_finalize_output(final_decision: str, output_type: str):
    # Define the data to be outputted

    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that generates the final output based on the specified format.",
        },
        {
            "role": "user",
            "content": f"Based on the final decision provided, generate the output in the following format: {output_type}.",
        },
        {
            "role": "system",
            "content": "Please generate the final output in the specified format, using the provided final decision as the content.",
        },
        {"role": "user", "content": f"Final Decision: {final_decision}"},
    ]

    # Make the API call with function calling
    try:
        with reserve_call_scope(
            model=_resolve_model("gpt-4o-mini"),
            call_type="test_finalize_output",
        ) as resolved_model:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

        message = response.choices[0].message
        print(f"\nmessage: {message}\n")

        if response.choices[0].message.tool_calls:
            # If true the model will return the name of the tool / function to call and the argument(s)
            tool_call_id = response.choices[0].message.tool_calls[0].id
            tool_function_name = response.choices[0].message.tool_calls[0].function.name
            tool_query_string = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )["final_decision"]
            filename = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )["filename"]

            # Step 3: Call the function and retrieve results. Append the results to the messages list.
            # Map function names to actual functions
            function_mapping = {
                "generate_simple_concise_answer_message": generate_simple_concise_answer_message,
                "generate_json_output": generate_json_output,
                "generate_pdf_output": generate_pdf_output,
                "generate_text_output": generate_text_file_output,
                "generate_html_output": generate_html_output,
                "generate_python_script_output": generate_python_script,
                "generate_code_snippet_output": generate_code_snippet,
                "generate_csv_output": generate_csv_output,
            }
            print(f"tool_function_name: {tool_function_name}")
            print(f"tool_query_string: {tool_query_string}")
            print(f"filename: {filename}")

            if tool_function_name in function_mapping:
                # Call the appropriate function with arguments
                filepath = function_mapping[tool_function_name](
                    final_decision=tool_query_string, filename=filename
                )
                return filepath
            else:
                return f"Error: Function '{tool_function_name}' not found."
        else:
            return message
    except (ProviderPolicyError, DisallowedModelError):
        raise
    except Exception as e:
        print(f"Error in finalize_output: {str(e)} on line {e.__traceback__.tb_lineno}")
        return f"Error in finalize_output: {str(e)} on line {e.__traceback__.tb_lineno}"


def finalize_output_result(final_decision: str, output_type: OutputType) -> ConversationOutput:
    scope = contextlib.nullcontext() if get_active_run_id() else artifact_run_scope(None)
    with scope:
        if isinstance(final_decision, ErrorResult) or not isinstance(final_decision, str) or not final_decision.strip():
            return failure_output("EmptyOutput", "The conversation returned no usable output.", output_type)
        return _finalize_output_result_impl(final_decision, output_type)


def finalize_output(final_decision: str, output_type: OutputType):
    """Legacy projection: text/path/ErrorResult, with the producer retaining the kind."""
    return finalize_output_result(final_decision, output_type).legacy_value()


def _finalize_output_result_impl(final_decision: str, output_type: OutputType):
    """
    Finalizes the output by calling the appropriate function based on output_type.

    Args:
        final_decision (str): The final decision or solution text.
        output_type (OutputType): The desired output format object.

    Returns:
        Any: The path to the generated file or the output in the specified format.
    """
    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that generates the final output based on the specified format.",
        },
        {
            "role": "user",
            "content": f"Based on the final decision provided, generate the output in the following format: {output_type.output_type} with file extension: {output_type.file_extension}.",
        },
        {
            "role": "system",
            "content": "Please generate the final output in the specified format, using the provided final decision as the content, adjusting it to fit the output format as needed.",
        },
        {"role": "user", "content": f"Final Decision: {final_decision}"},
    ]

    # Make the API call with function calling
    try:
        with reserve_call_scope(
            model=_resolve_model("gpt-4o-mini"),
            call_type="test_finalize_output",
        ) as resolved_model:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

        message = response.choices[0].message
        if getattr(message, "refusal", None):
            return failure_output("RefusedOutput", "The provider declined to produce this output.", output_type)
        print(f"\nmessage: {message}\n")

        if response.choices[0].message.tool_calls:
            # If true the model will return the name of the tool / function to call and the argument(s)
            tool_call_id = response.choices[0].message.tool_calls[0].id
            tool_function_name = response.choices[0].message.tool_calls[0].function.name
            tool_args = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            tool_query_string = tool_args.get("final_decision", final_decision)
            if (tool_query_string is None or (isinstance(tool_query_string, (dict, list)) and not tool_query_string)
                    or (isinstance(tool_query_string, str) and not tool_query_string.strip())
                    or not isinstance(tool_query_string, (str, dict, list))
                    or (tool_function_name not in ("generate_json_output", "generate_csv_output")
                        and not isinstance(tool_query_string, str))):
                return failure_output("EmptyOutput", "The provider returned an empty or invalid output payload.", output_type)
            filename = tool_args.get("filename", f"final_output{output_type.file_extension}")

            # Step 3: Call the function and retrieve results. Append the results to the messages list.
            # Map function names to actual functions
            function_mapping = {
                "generate_simple_concise_answer_message": generate_simple_concise_answer_message,
                "generate_json_output": generate_json_output,
                "generate_pdf_output": generate_pdf_output,
                "generate_text_output": generate_text_file_output,
                "generate_html_output": generate_html_output,
                "generate_python_script_output": generate_python_script,
                "generate_code_snippet_output": generate_code_snippet,
                "generate_csv_output": generate_csv_output,
            }
            print(f"tool_function_name: {tool_function_name}")
            print(f"tool_query_string: {tool_query_string}")
            print(f"filename: {filename}")
            if tool_function_name in function_mapping:
                # Call the appropriate function with arguments
                if tool_function_name != "generate_simple_concise_answer_message":
                    resolved = resolve_output_filepath(filename)
                    if get_tracked_artifact_ref(resolved) is not None:
                        stem, extension = os.path.splitext(resolved)
                        filename = f"{stem}_{uuid.uuid4().hex}{extension}"
                filepath = function_mapping[tool_function_name](
                    final_decision=tool_query_string, filename=filename
                )
                if tool_function_name == "generate_simple_concise_answer_message":
                    if not isinstance(filepath, str) or not filepath.strip():
                        return failure_output("EmptyOutput", "The conversation returned no usable output.", output_type)
                    return ConversationOutput(kind="text", content=filepath, output_type=output_type)
                return artifact_conversation_output(filepath, output_type)
            else:
                error_result = ErrorResult(
                    error_type="ValidationError",
                    error_message=f"Function '{tool_function_name}' not found.",
                    error_details=str({"tool_function_name": tool_function_name}),
                )
                return failure_output("ValidationError", "The provider selected an unsupported output tool.", output_type)
        else:
            content = getattr(message, "content", None)
            if not isinstance(content, str) or not content.strip():
                return failure_output("EmptyOutput", "The conversation returned no usable output.", output_type)
            return ConversationOutput(kind="text", content=content, output_type=output_type)
    except (ProviderPolicyError, DisallowedModelError):
        raise
    except Exception as e:
        print(f"Error in finalize_output: {str(e)} on line {e.__traceback__.tb_lineno}")
        error_result = ErrorResult(
            error_type="LLMCallError",
            error_message=f"Error in finalize_output: {str(e)} on line {e.__traceback__.tb_lineno}",
            error_details=str(e),
        )
        return failure_output("OutputError", "The conversation output could not be generated.", output_type)


sample_conversation_history = """User: Hey, have we figured out the root cause of that recurring issue in the system yet?

Assistant: I believe I have identified the root cause. It seems there is a bug in the authentication process, specifically related to handling expired tokens. Let me show you the relevant code snippet:

```
if token.is_expired():
    # Incorrect handling here
    return \"Authentication failed\"
```

User: Got it. Thanks for that. Can you help me put together a report on this? I need to summarize this for the stakeholders.

Assistant: Sure thing. Here's a draft outline for the report:
1. Issue Overview: Bug in authentication process.
2. Root Cause: Improper handling of expired tokens.
3. Impact: Users unable to authenticate properly when tokens expire.
4. Proposed Fix: Correct the token handling logic.

User: Awesome. Let's move forward with implementing the solution. We don't have much time to waste here.

Assistant: Understood. Let me provide the updated code to fix the issue:

```
if token.is_expired():
    refresh_token()
    return \"Token refreshed, please try again\"
```

User: Great. Now we just need to test everything to make sure the fix works properly, right?

Assistant: Exactly. Let's run a few tests now. Here's a simple test case to verify:

1. Expired Token Test:
   - Step 1: Use an expired token.
   - Expected Result: The token should be refreshed, and authentication should proceed.

Test Output: The test passed successfully, and the token was refreshed as expected.

User: Perfect. Let me know if there are any issues so we can document the results for the final report.

Assistant: Absolutely. I'll keep documenting the testing process. So far, everything is working as expected. Here is what I have documented:
- Cause of Issue: Improper handling of expired tokens.
- Solution: Added logic to refresh expired tokens.
- Testing Results: All tests passed, no issues found.

User: Have you completed the documentation?

Assistant: Yes, I've documented the entire process, including the root cause, the code changes made, and the testing results. Here is the full documentation for your review:

```
Documentation Summary:

1. Issue Overview:
   - The authentication process contained a bug related to handling expired tokens.

2. Root Cause:
   - Improper handling of expired tokens, leading to authentication failures.

3. Code Changes Made:
   - Original Code:
   ```
   if token.is_expired():
       # Incorrect handling here
       return \"Authentication failed\"
   ```
   - Updated Code:
   ```
   if token.is_expired():
       refresh_token()
       return \"Token refreshed, please try again\"
   ```

4. Testing Process and Results:
   - Test Case: Expired Token Test
     - Step 1: Use an expired token.
     - Expected Result: Token should be refreshed, and authentication should proceed.
     - Actual Result: Test passed successfully, token was refreshed as expected.

5. Conclusion:
   - The issue was resolved by adding logic to refresh expired tokens. All tests passed successfully, confirming the solution's effectiveness.
```

User: Fantastic. Thanks a lot for handling all of that.

User: Oh, by the way, have we optimized the logging mechanism for this process?

Assistant: I haven't worked on optimizing the logging mechanism yet. If you'd like, I can start looking into that next."""

output_analysis = """[
  {
    "Determine the root cause of the issue": {
      true: "Assistant: I believe I have identified the root cause. It seems there is a bug in the authentication process, specifically related to handling expired tokens."
    }
  },
  {
    "Put together a report for stakeholders": {
      true: "Assistant: Here's a draft outline for the report:\n1. Issue Overview: Bug in authentication process.\n2. Root Cause: Improper handling of expired tokens.\n3. Impact: Users unable to authenticate properly when tokens expire.\n4. Proposed Fix: Correct the token handling logic."
    }
  },
  {
    "Implement the solution": {
      true: "Assistant: Let me provide the updated code to fix the issue:\n\n``\nif token.is_expired():\n    refresh_token()\n    return \"Token refreshed, please try again\"\n``"
    }
  },
  {
    "Test the solution": {
      true: "Assistant: Let's run a few tests now. Here's a simple test case to verify:\n\n1. Expired Token Test:\n   - Step 1: Use an expired token.\n   - Expected Result: The token should be refreshed, and authentication should proceed.\n\nTest Output: The test passed successfully, and the token was refreshed as expected."
    }
  },
  {
    "Document the results": {
      true: "Assistant: I'll keep documenting the testing process. So far, everything is working as expected. Here is what I have documented:\n- Cause of Issue: Improper handling of expired tokens.\n- Solution: Added logic to refresh expired tokens.\n- Testing Results: All tests passed, no issues found."
    }
  },
  {
    "Optimize the logging mechanism": {
      false: "Assistant: I haven't worked on optimizing the logging mechanism yet. If you'd like, I can start looking into that next."
    }
  },
  {
    "Upgrade authentication protocol": {
      false: "No evidence found"
    }
  }
]"""


class FinalOutput(BaseModel):
    final_output: str = Field(
        ..., description="The final output generated by the assistant."
    )
    output_type: OutputType = Field(..., description="The type of output generated.")
    version: int = Field(..., description="The version of the final output.")


class OutputRequirements(BaseModel):
    requirement: str = Field(
        ..., description="The specific requirement for the output."
    )
    met: bool = Field(..., description="Whether the requirement is met.")


class FinalOutputAnalysis(BaseModel):
    user_intent: str = Field(
        ..., description="The user's intent or goal for the output."
    )
    expected_output_type: str = Field(..., description="The expected type of output.")
    requirements: List[OutputRequirements] = Field(
        ..., description="The specific requirements for the output."
    )
    matches_expected_output_type: bool = Field(
        ..., description="Whether the output matches the expected type."
    )
    aligns_with_user_intent: bool = Field(
        ..., description="Whether the output aligns with the user's intent."
    )
    changes_needed_to_align: List[str] = Field(
        ...,
        description="The changes needed to align the output with the user's intent.",
    )
    is_complete: bool = Field(..., description="Whether the output is complete.")


sample_final_output_code = """Plan for Function to Calculate the Area of a Circle:

1. Define the function:
```python
def calculate_area(radius):
    pi = 3.14159
    area = pi * radius ** 2
    return area
```

2. Test the function:
```python
radius = 5
print(f"The area of a circle with radius {radius} is {calculate_area(radius)}")
```

3. Output:
The area of a circle with radius 5 is 78.53975"""

sample_original_prompt_code = """You are tasked with creating a Python function to calculate the area of a circle. The function should take the radius of the circle as input and return the area of the circle. You need to define the function, test it with a radius of 5, and provide the output."""

sample_problem_statement_code = """Define a Python function named 'calculate_area' that takes the radius of a circle as input and returns the area of the circle. Test the function with a radius of 5 and provide the output."""

sample_output_analysis_code = """{
    "user_intent": "Obtain a Python function to calculate the area of a circle",
    "expected_output_type": "Python Code Snippet (.py)",
    "requirements": [
        {
            "requirement": "Define a function named 'calculate_area' that takes the radius as input",
            "met": true
        },
        {
            "requirement": "Test the function with a radius of 5",
            "met": true
        },
        {
            "requirement": "Provide the output of the function",
            "met": true
        }
    ],
    "matches_expected_output_type": false,
    "aligns_with_user_intent": true,
    "changes_needed_to_align": [
        "The output should be in the form of a Python script file (.py)"
    ],
    "is_complete": false
}"""

sample_corrected_output_code = """```python
def calculate_area(radius):
    pi = 3.14159
    area = pi * radius ** 2
    return area

radius = 5
print(f"The area of a circle with radius {radius} is {calculate_area(radius)}")
```"""


output_type_samples = {
    "Simple Concise Answer": {
        "sample_original_prompt": "What is the capital of France?",
        "sample_problem_statement": "Provide a simple and concise answer to the question: What is the capital of France?",
        "sample_output_analysis": """{
            "user_intent": "Obtain a simple and concise answer to the question",
            "expected_output_type": "Simple Text Answer",
            "requirements": [
                {
                    "requirement": "Answer the question concisely",
                    "met": true
                },
                {
                    "requirement": "Avoid unnecessary elaboration",
                    "met": false
                }
            ],
            "matches_expected_output_type": false,
            "aligns_with_user_intent": true,
            "changes_needed_to_align": [
                "The answer should be short and direct, avoiding extra descriptions."
            ],
            "is_complete": false
        }""",
        "sample_final_output": "The capital city of France is Paris, which is also known as the City of Lights.",
        "sample_corrected_output": "Paris",
    },
    "JSON": {
        "sample_original_prompt": "Provide the details of a person including name, age, and city in JSON format.",
        "sample_problem_statement": "Define a JSON object that contains the details of a person, including their name, age, and city.",
        "sample_output_analysis": """{
            "user_intent": "Create a JSON object representing a person's details",
            "expected_output_type": "JSON Object",
            "requirements": [
                {
                    "requirement": "Include fields for name, age, and city",
                    "met": true
                },
                {
                    "requirement": "Ensure correct JSON formatting with key-value pairs",
                    "met": false
                }
            ],
            "matches_expected_output_type": false,
            "aligns_with_user_intent": true,
            "changes_needed_to_align": [
                "The output should be a properly formatted JSON object with correct syntax."
            ],
            "is_complete": false
        }""",
        "sample_final_output": "Name: John Doe, Age: 30, City: New York",
        "sample_corrected_output": '{"name": "John Doe", "age": 30, "city": "New York"}',
    },
    "PDF": {
        "sample_original_prompt": "Generate a report in PDF format summarizing the monthly sales figures.",
        "sample_problem_statement": "Create a report in PDF format that provides a summary of the monthly sales figures, including a chart showing the sales trend over the past 6 months.",
        "sample_output_analysis": """{
            "user_intent": "Generate a summary report of monthly sales figures in PDF format",
            "expected_output_type": "PDF File",
            "requirements": [
                {
                    "requirement": "Include a summary of monthly sales figures",
                    "met": true
                },
                {
                    "requirement": "Include a visual chart to represent sales trend",
                    "met": false
                },
                {
                    "requirement": "Ensure the output is in PDF format",
                    "met": false
                }
            ],
            "matches_expected_output_type": false,
            "aligns_with_user_intent": true,
            "changes_needed_to_align": [
                "Add a visual chart to the report to show sales trends.",
                "Convert the summary into a PDF file format."
            ],
            "is_complete": false
        }""",
        "sample_final_output": "Monthly sales figures summary: Sales were up by 15% compared to the previous month.",
        "sample_corrected_output": """%PDF-1.4
1 0 obj
<<
  /Title (Monthly Sales Report)
  /Creator (Python PDF Generator)
  /Producer (PyPDF)
  /CreationDate (D:20241005120000)
>>
endobj
2 0 obj
<<
  /Type /Page
  /Parent 3 0 R
  /MediaBox [0 0 612 792]
  /Contents 4 0 R
  /Resources << /Font << /F1 5 0 R >> >>
>>
endobj
3 0 obj
<<
  /Type /Pages
  /Kids [2 0 R]
  /Count 1
>>
endobj
4 0 obj
<< /Length 89 >>
stream
BT
/F1 24 Tf
100 700 Td
(Monthly Sales Report) Tj
ET
BT
/F1 12 Tf
100 680 Td
(Sales were up by 15% compared to the previous month.) Tj
ET
BT
/F1 12 Tf
100 660 Td
(Chart showing sales trend over the past 6 months:) Tj
ET
BT
100 640 Td
(Placeholder for sales trend chart) Tj
ET
endstream
endobj
5 0 obj
<<
  /Type /Font
  /Subtype /Type1
  /BaseFont /Helvetica
>>
endobj
6 0 obj
<<
  /Type /Catalog
  /Pages 3 0 R
>>
endobj
xref
0 7
0000000000 65535 f
0000000010 00000 n
0000000079 00000 n
0000000179 00000 n
0000000250 00000 n
0000000389 00000 n
0000000470 00000 n
trailer
<<
  /Size 7
  /Root 6 0 R
>>
startxref
534
%%EOF""",
    },
    "Text File": {
        "sample_original_prompt": "Create a text file listing the top 5 programming languages in 2024.",
        "sample_problem_statement": "Create a text file that lists the top 5 programming languages in 2024, each on a separate line, with rankings from 1 to 5.",
        "sample_output_analysis": """{
            "user_intent": "Provide a text file listing the top programming languages",
            "expected_output_type": "Text File (.txt)",
            "requirements": [
                {
                    "requirement": "List the top 5 programming languages of 2024",
                    "met": true
                },
                {
                    "requirement": "Provide each language on a new line with proper rankings",
                    "met": false
                },
                {
                    "requirement": "Output should be in a .txt file format",
                    "met": false
                }
            ],
            "matches_expected_output_type": false,
            "aligns_with_user_intent": true,
            "changes_needed_to_align": [
                "Ensure each programming language is on a new line with ranking numbers.",
                "Provide the output as a .txt file."
            ],
            "is_complete": false
        }""",
        "sample_final_output": "Top 5 programming languages in 2024 are Python, JavaScript, Java, Go, and Rust.",
        "sample_corrected_output": "1. Python\n2. JavaScript\n3. Java\n4. Go\n5. Rust",
    },
    "HTML": {
        "sample_original_prompt": "Create an HTML page with a heading 'Welcome' and a paragraph describing a product.",
        "sample_problem_statement": "Write an HTML document that includes a heading 'Welcome', a paragraph describing a product, and a footer with contact information.",
        "sample_output_analysis": """{
            "user_intent": "Generate an HTML page with a heading, paragraph, and footer",
            "expected_output_type": "HTML Document (.html)",
            "requirements": [
                {
                    "requirement": "Include a heading with the text 'Welcome'",
                    "met": true
                },
                {
                    "requirement": "Include a paragraph describing a product",
                    "met": true
                },
                {
                    "requirement": "Include a footer with contact information",
                    "met": false
                }
            ],
            "matches_expected_output_type": true,
            "aligns_with_user_intent": false,
            "changes_needed_to_align": [
                "Add a footer to the HTML document with contact information."
            ],
            "is_complete": false
        }""",
        "sample_final_output": "Heading: Welcome\nProduct Description: This product is designed to help you achieve more in less time.",
        "sample_corrected_output": """<html>
<head>
    <title>Welcome</title>
</head>
<body>
    <h1>Welcome</h1>
    <p>This product is designed to help you achieve more in less time. It's efficient, user-friendly, and highly effective.</p>
    <footer>
        <p>Contact us at: support@example.com</p>
    </footer>
</body>
</html>""",
    },
    "CSV": {
        "sample_original_prompt": "Create a CSV file with columns 'Name', 'Age', and 'City' and add three rows of sample data.",
        "sample_problem_statement": "Generate a CSV file with columns 'Name', 'Age', 'City' and provide three rows of data. Ensure that the data is consistent and formatted correctly for use in data processing tools.",
        "sample_output_analysis": """{
            "user_intent": "Create a CSV file with specific columns and rows of data",
            "expected_output_type": "CSV File (.csv)",
            "requirements": [
                {
                    "requirement": "Include columns 'Name', 'Age', and 'City'",
                    "met": true
                },
                {
                    "requirement": "Add three rows of sample data",
                    "met": true
                },
                {
                    "requirement": "Ensure output is in a .csv file format",
                    "met": false
                },
                {
                    "requirement": "Ensure data is consistently formatted for use in data processing tools",
                    "met": false
                }
            ],
            "matches_expected_output_type": false,
            "aligns_with_user_intent": true,
            "changes_needed_to_align": [
                "Format the output as a .csv file.",
                "Ensure data fields are consistently formatted and properly separated by commas."
            ],
            "is_complete": false
        }""",
        "sample_final_output": "Name: Alice, Age: 30, City: New York\nName: Bob, Age: 25, City: Los Angeles\nName: Charlie, Age: 35, City: Chicago",
        "sample_corrected_output": "Name,Age,City\nAlice,30,New York\nBob,25,Los Angeles\nCharlie,35,Chicago",
    },
    "Code": {
        "sample_original_prompt": "You are tasked with creating a Python function to calculate the area of a circle. The function should take the radius of the circle as input and return the area of the circle. You need to define the function, test it with a radius of 5, and provide the output.",
        "sample_problem_statement": "Define a Python function named 'calculate_area' that takes the radius of a circle as input and returns the area of the circle. Test the function with a radius of 5 and provide the output.",
        "sample_output_analysis": """{
            "user_intent": "Obtain a Python function to calculate the area of a circle",
            "expected_output_type": "Python Code Snippet (.py)",
            "requirements": [
                {
                    "requirement": "Define a function named 'calculate_area' that takes the radius as input",
                    "met": true
                },
                {
                    "requirement": "Test the function with a radius of 5",
                    "met": true
                },
                {
                    "requirement": "Provide the output of the function",
                    "met": true
                }
            ],
            "matches_expected_output_type": false,
            "aligns_with_user_intent": true,
            "changes_needed_to_align": [
                "The output should be in the form of a Python script file (.py)"
            ],
            "is_complete": false
        }""",
        "sample_final_output": """Plan for Function to Calculate the Area of a Circle:

1. Define the function:
```python
def calculate_area(radius):
    pi = 3.14159
    area = pi * radius ** 2
    return area
```

2. Test the function:
```python
radius = 5
print(f"The area of a circle with radius {radius} is {calculate_area(radius)}")
```""",
        "sample_corrected_output": """```python
def calculate_area(radius):
    pi = 3.14159
    area = pi * radius ** 2
    return area

radius = 5
print(f"The area of a circle with radius {radius} is {calculate_area(radius)}")
```""",
    },
}


# map_output_type_to_sample_output = {
#     "Simple Concise Text Answer": {
#         "sample_final_output": sample_final_output_code,
#         "sample_original_prompt": sample_original_prompt_code,
#         "sample_problem_statement": sample_problem_statement_code,
#         "sample_output_analysis": sample_output_analysis_code,
#         "sample_corrected_output": sample_corrected_output_code,
#     },
#     "Detailed Report in Text or PDF Format": {


def final_output_correction(
    final_output,
    output_type,
    output_analysis: FinalOutputAnalysis,
    sample_final_output,
    sample_original_prompt,
    sample_problem_statement,
    sample_output_analysis,
    sample_corrected_output,
    version_counter=1,
):
    """
    Corrects the final output based on the output analysis and the expected output type.

    Args:
        final_output (str): The final output generated by the assistant.
        output_type (str): The expected output type.
        output_analysis (FinalOutputAnalysis): The analysis of the final output.
        sample_final_output (str): The sample final output for reference.
        sample_original_prompt (str): The sample original prompt for reference.
        sample_problem_statement (str): The sample problem statement for reference.
        sample_output_analysis (str): The sample output analysis for reference.
        sample_corrected_output (str): The sample corrected output for reference.

    Returns:
        FinalOutput: The corrected final output that aligns with the expected output type and requirements.
    """

    _analysis = {
        "user_intent": output_analysis.user_intent,
        "expected_output_type": output_analysis.expected_output_type,
        "requirements": [],
        "matches_expected_output_type": output_analysis.matches_expected_output_type,
        "aligns_with_user_intent": output_analysis.aligns_with_user_intent,
        "changes_needed_to_align": [],
        "is_complete": output_analysis.is_complete,
    }
    for req in output_analysis.requirements:
        _analysis["requirements"].append(
            {"requirement": req.requirement, "met": req.met}
        )
    for change in output_analysis.changes_needed_to_align:
        _analysis["changes_needed_to_align"].append(change)

    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that corrects the final output to align with the expected output type and requirements.",
        },
        {
            "role": "user",
            "content": f"Please correct the final output to match the expected output type and requirements based on the analysis provided. Let's start with the sample final output for reference: \n\n  {sample_final_output} \n\nOriginal Prompt: {sample_original_prompt} \n\nProblem Statement: {sample_problem_statement} \n\nOutput Analysis: {sample_output_analysis} \n\n",
        },
        {
            "role": "assistant",
            "content": f"Corrected Output: {sample_corrected_output} \n\n",
        },
        {
            "role": "user",
            "content": "Now for an important real one. Please correct the final output to match the expected output type and requirements based on the analysis provided.",
        },
        {
            "role": "user",
            "content": f"Final Output: {final_output}, Expected Output Type: {output_type}",
        },
        {"role": "user", "content": f"Output Analysis: \n\n{_analysis}"},
    ]

    # Make the API call to correct the final output using structured outputs with beta.chat.completions.parse
    try:
        with reserve_call_scope(
            model=_resolve_model("gpt-4o" if version_counter <= 1 else "gpt-4o-mini"),
            call_type="final_output_correction",
        ) as resolved_model:
            response = client.beta.chat.completions.parse(
                model=resolved_model,
                messages=messages,  # Pass the messages directly as a list of dicts
                response_format=FinalOutput,
            )
        # Extract the corrected final output from the response
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            return ErrorResult(error_type="RefusedRevision", error_message="The provider declined to revise this output.")
        corrected_final_output = message.parsed
        if not isinstance(corrected_final_output, FinalOutput) or not corrected_final_output.final_output.strip():
            return ErrorResult(error_type="InvalidRevision", error_message="The provider returned no usable correction.")
        return corrected_final_output.model_copy(deep=True)
    except (ProviderPolicyError, DisallowedModelError):
        raise
    except Exception:
        return ErrorResult(error_type="ProviderError", error_message="The provider could not revise this output.")


def match_output_type_to_sample(output_type_str: str) -> str:
    """Find the best match from output_type_samples using Jaro-Winkler with deterministic fallback."""
    o_types = list(output_type_samples.keys())
    best_match = None
    best_score = 0.0
    for o_type in o_types:
        score = jarowinkler_similarity(output_type_str, o_type)
        if score > best_score:
            best_score = score
            best_match = o_type

    if best_match is None or best_score < MINIMUM_SIMILARITY_THRESHOLD:
        print(f"Warning: No good matches found for output type '{output_type_str}' (best score: {best_score})")
        if DEFAULT_FALLBACK_OUTPUT_TYPE not in output_type_samples:
            raise ValueError(f"Default fallback output type '{DEFAULT_FALLBACK_OUTPUT_TYPE}' not found in output_type_samples")
        return DEFAULT_FALLBACK_OUTPUT_TYPE

    return best_match


class OutputRevisionResult(BaseModel):
    """Internal controller outcome; accepted text is independent of termination."""
    accepted: FinalOutput
    analysis: Optional[FinalOutputAnalysis] = None
    attempt_version: int
    stop_reason: Literal["completed", "revision_limit", "evaluation_unavailable", "provider_error", "budget_exhausted", "timeout", "cancelled"]
    error: Optional[ErrorResult] = None


def _revision_checks(analysis: FinalOutputAnalysis) -> Dict[str, bool]:
    checks = {}
    for requirement in analysis.requirements:
        key = requirement.requirement.strip()
        if not key or key in checks:
            raise ValueError("Revision requirements must be unique and nonempty")
        checks[key] = requirement.met
    if not checks:
        raise ValueError("Revision evaluation requires a checklist")
    return {**{f"requirement:{key}": met for key, met in checks.items()},
            "format": analysis.matches_expected_output_type,
            "alignment": analysis.aligns_with_user_intent, "complete": analysis.is_complete}


def analyze_final_output_result(final_output, lead_personality, assistants_list, original_prompt,
                                problem_statement, output_type: OutputType, current_version=1,
                                *, max_version=5) -> OutputRevisionResult:
    """Revise a private proposal, accepting only strict checklist improvement."""
    if any(type(value) is not int or value < 1 for value in (current_version, max_version)):
        raise ValueError("Revision versions must be positive integers")
    accepted = FinalOutput(final_output=final_output, output_type=output_type.model_copy(deep=True), version=current_version)
    attempt = current_version
    accepted_analysis = None

    def outcome(reason, error_type=None):
        return OutputRevisionResult(accepted=accepted.model_copy(deep=True),
            analysis=accepted_analysis.model_copy(deep=True) if accepted_analysis is not None else None,
            attempt_version=attempt, stop_reason=reason,
            error=ErrorResult(error_type=error_type, error_message={
                "revision_limit": "Output revision reached its version limit without a verified complete result.",
                "evaluation_unavailable": "The output could not be evaluated.",
                "provider_error": "The provider could not revise this output.",
            }.get(reason, "Output revision could not complete.")) if error_type else None)

    def retain_policy_stop(exc):
        reason = "cancelled" if type(exc).__name__ == "RunCancelledError" else "timeout" if "Timeout" in type(exc).__name__ else "budget_exhausted"
        exc.revision_result = outcome(reason, type(exc).__name__)

    def assess(text, requirements=None):
        messages = [{"role": "system", "content": "Evaluate the actual output against the user's task. Return the complete requirement checklist and format, alignment and completeness checks. Keep supplied requirement descriptions exactly; never drop or rename them."},
            {"role": "user", "content": f"Original Prompt: {original_prompt}\nProblem Statement: {problem_statement}\nExpected Output Type: {output_type.output_type} ({output_type.file_extension})\nActual Output: {text}"}]
        if requirements is not None:
            messages.append({"role": "user", "content": "Fixed requirement checklist: " + json.dumps(requirements)})
        try:
            with reserve_call_scope(model=_resolve_model("gpt-4o-mini"), call_type="analyze_final_output") as model:
                response = client.beta.chat.completions.parse(model=model, messages=messages, response_format=FinalOutputAnalysis)
            message = response.choices[0].message
            if getattr(message, "refusal", None) or not isinstance(message.parsed, FinalOutputAnalysis):
                return None
            result = message.parsed.model_copy(deep=True)
            _revision_checks(result)
            return result
        except (ProviderPolicyError, DisallowedModelError) as exc:
            retain_policy_stop(exc)
            raise
        except Exception:
            return None

    if attempt >= max_version:
        return outcome("revision_limit", "RevisionLimit")
    accepted_analysis = assess(accepted.final_output)
    if accepted_analysis is None:
        return outcome("evaluation_unavailable", "EvaluationUnavailable")
    checks = _revision_checks(accepted_analysis)
    requirements = [req.requirement.strip() for req in accepted_analysis.requirements]
    if all(checks.values()):
        return outcome("completed")
    sample = output_type_samples[match_output_type_to_sample(output_type.output_type)]
    while attempt < max_version:
        attempt += 1
        try:
            proposed = final_output_correction(accepted.final_output, output_type, accepted_analysis,
                sample["sample_final_output"], sample["sample_original_prompt"], sample["sample_problem_statement"],
                sample["sample_output_analysis"], sample["sample_corrected_output"], attempt)
        except (ProviderPolicyError, DisallowedModelError) as exc:
            retain_policy_stop(exc)
            raise
        if isinstance(proposed, ErrorResult):
            if proposed.error_type == "ProviderError":
                return outcome("provider_error", "ProviderError")
            continue
        if not isinstance(proposed, FinalOutput) or not proposed.final_output.strip() or proposed.final_output == accepted.final_output:
            continue
        proposed_analysis = assess(proposed.final_output, requirements)
        if proposed_analysis is None:
            return outcome("evaluation_unavailable", "EvaluationUnavailable")
        new_checks = _revision_checks(proposed_analysis)
        if (new_checks.keys() != checks.keys()
                or any(met and not new_checks[key] for key, met in checks.items())
                or not any(not met and new_checks[key] for key, met in checks.items())):
            continue
        accepted = FinalOutput(final_output=proposed.final_output, output_type=output_type.model_copy(deep=True), version=attempt)
        accepted_analysis, checks = proposed_analysis, new_checks
        if all(checks.values()):
            return outcome("completed")
    return outcome("revision_limit", "RevisionLimit")


def analyze_final_output(final_output, lead_personality, assistants_list, original_prompt,
                         problem_statement, output_type: OutputType, current_version=1, *, max_version=5):
    """Legacy accepted-text projection; production callers use the typed result."""
    return analyze_final_output_result(final_output, lead_personality, assistants_list, original_prompt,
        problem_statement, output_type, current_version, max_version=max_version).accepted.final_output


def _finalize_revision(result: OutputRevisionResult, history, output_type):
    if result.error is not None:
        history.append({"role": "assistant", "name": "Partial Output", "round": history[-1].get("round", 0) if history else 0,
                        "content": result.accepted.final_output})
        return failure_output(result.error.error_type, result.error.error_message, output_type)
    return finalize_output_result(result.accepted.final_output, output_type)


class TaskCompletion(BaseModel):
    task: str = Field(..., description="The task to be completed.")
    completed: bool = Field(..., description="Whether the task has been completed.")
    completion_artifact: str = Field(
        ..., description="The artifact or evidence of completion."
    )


class TaskList(BaseModel):
    tasks: List[TaskCompletion] = Field(
        ..., description="The list of tasks with completion status and artifacts."
    )


def analyze_to_do_list(conversation_memory, conversation_history):
    # Extract the to-do list from the conversation memory
    to_do_items_str = []
    for task in conversation_memory.to_do_list:
        if isinstance(task, ToDoItem):
            to_do_items_str.append(task.task)
        elif isinstance(task, list):
            to_do_items_str.append(" ".join(str(t) for t in task))
        elif isinstance(task, dict):
            to_do_items_str.append(task.get("task", str(task)))
        else:
            to_do_items_str.append(str(task))
    to_do_list = ", ".join(to_do_items_str)
    if not to_do_list:
        return "No to-do items were identified in the conversation memory."
    # Prepare the messages for the assistant
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that compares the to-do list which tasks have been completed.",
        },
        {
            "role": "user",
            "content": "Please analyze the to-do list and determine which tasks have been completed based on the conversation history and memory. For each task in the to-do list, provide a 'task' string that is an exact copy of the task given, a 'completed' boolean indicating whether the task has been completed per the conversation history, and a 'completion_artifact' string copied from the conversation history that represents the completed task. If no evidence is found, provide a boolean value of false and a 'completion_artifact' string indicating that no evidence was found. Make sure tasks are only marked as completed if there is clear evidence in the conversation history, do NOT mark tasks based on mere mentions or discussions or 'should be done' statements, nor because a question was asked about them, but only if there is clear evidence of completion. Also, ensure that 'task' strings are an exact match to the to-do list items.",
        },
        {
            "role": "user",
            "content": f"Here's the sample to-do list to analyze: To-Do List: Determine the cause of the issue, Create a report, Implement the solution, Test the solution, Document the results, Optimize the logging mechanism, Upgrade authentication protocol. And the sample conversation history: {sample_conversation_history}",
        },
        {"role": "assistant", "content": f"{output_analysis}"},
        {
            "role": "user",
            "content": f"Great work, thanks! Now for the real one: To-Do List:\n{to_do_list}, and the conversation history: {', '.join([safe_get_message_field(msg, 'name', 'Unknown')+': '+ safe_get_message_field(msg, 'content', 'No content') for msg in conversation_history if validate_message_structure(msg)])}",
        },
    ]
    # Make the API call to analyze the to-do list using structured outputs with beta.chat.completions.parse
    try:
        with reserve_call_scope(
            model=_resolve_model("gpt-4o-mini"),
            call_type="analyze_to_do_list",
        ) as resolved_model:
            response = client.beta.chat.completions.parse(
                model=resolved_model,  # Specify the model
                messages=messages,  # Pass the messages directly as a list of dicts
                response_format=TaskList,
            )
        # Extract the completion from the response
        completion = response.choices[0].message.parsed
        print(f"\nCompletion: {completion}\n")
        # update the conversation memory with the completed tasks
        for task in completion.tasks:
            if task.completed:
                conversation_memory.add_completed_task(
                    task.task, task.completion_artifact
                )

        return conversation_memory  # Return the updated conversation memory

    except (ProviderPolicyError, DisallowedModelError):
        raise
    except Exception as e:
        print(
            f"Error in analyze_to_do_list: {str(e)} on line {e.__traceback__.tb_lineno}"
        )
        raise Exception(
            f"Error in analyze_to_do_list: {str(e)} on line {e.__traceback__.tb_lineno}"
        )
        return f"Error: {str(e)} on line {e.__traceback__.tb_lineno}"


def get_assistant_response(
    assistant_name,
    assistant_prompt,
    conversation_history,
    initial_prompt=None,
    conversation_memory=None,
    is_socratic=False,
    direct_reply=None,
    token_limit=4096,
):
    MAX_CONTEXT_MESSAGES = 15  # Adjust based on token limits

    # Generate summary to include in context
    memory_summary = ""
    if conversation_memory:
        memory_summary = conversation_memory.get_memory_summary()

    # Prepare the messages for the assistant
    messages = [
        {"role": "system", "content": assistant_prompt},
    ]    # Include only the last N messages to manage token usage
    recent_history = conversation_history[-MAX_CONTEXT_MESSAGES:]
    for msg in recent_history:
        if not validate_message_structure(msg):
            continue  # Skip malformed messages
            
        msg_role = safe_get_message_field(msg, 'role')
        msg_name = safe_get_message_field(msg, 'name')
        msg_content = safe_get_message_field(msg, 'content')
        
        if msg_role == "assistant" and msg_name != assistant_name:
            # Make other assistants' messages appear as user messages to the assistant
            messages.append(
                {"role": "user", "content": f"{msg_name}: {msg_content}"}
            )
        elif msg_name == "Primary User":
            messages.append(
                {"role": "user", "content": "Primary User: " + msg_content}
            )
        elif msg_role == "assistant" and msg_name == assistant_name:
            messages.append({"role": "assistant", "content": msg_content})
        elif (
            msg_role == "user"
            and msg_name != assistant_name
            and msg_name != "Primary User"
            and msg_name != "Mediator"
        ):
            messages.append(
                {"role": "user", "content": msg_name + ": " + msg_content}
            )
        elif msg_role == "system":
            messages.append({"role": "system", "content": msg_content})
        elif assistant_name in msg_name:
            messages.append({"role": "assistant", "content": msg_content})
        else:
            messages.append({"role": "user", "content": msg_content})

    if memory_summary != "":
        messages.append({"role": "system", "content": f"Conversation memory summary: {memory_summary}"})
    if initial_prompt:
        messages.append({"role": "user", "content": initial_prompt})

    if is_socratic:
        # Add a prompt to encourage the assistant to ask a question
        messages.append(
            {
                "role": "system",
                "content": "Please ask a question to prompt further discussion.",
            }
        )

    if direct_reply:
        asking_assistant = direct_reply.get("asking_assistant")
        reply = direct_reply.get("reply")
        # Add the direct reply to the assistant's message
        messages.append(
            {
                "role": "user",
                "content": f"The assistant {asking_assistant} has directed the following message to you: {reply}. Please respond.",
            }
        )

    # print(f"\n \n \n \n \n Messages to assistant {assistant_name}: \n", messages, "\n \n \n \n \n")
    # Update the OpenAI call to use the correct method
    try:
        with reserve_call_scope(
            model=_resolve_model("gpt-4o-mini"),
            call_type="get_assistant_response",
        ) as resolved_model:
            create_kwargs: Dict[str, Any] = {
                "model": resolved_model,
                "messages": messages,
                "max_completion_tokens": token_limit,
            }
            resolved_temp = _resolve_temperature()
            if resolved_temp is not None:
                create_kwargs["temperature"] = resolved_temp
            response = client.chat.completions.create(**create_kwargs)
        # Extract the text from the response
        assistant_reply = response.choices[
            0
        ].message.content.strip()  # Use 'message' key, not 'text'
        return {"role": "assistant", "name": assistant_name, "content": assistant_reply}
    except (ProviderPolicyError, DisallowedModelError):
        raise
    except Exception as e:
        return ErrorResult(
            error_type="LLMCallError",
            error_message=f"Assistant '{assistant_name}' failed to respond: {str(e)}",
            error_details=str(e),
        )


class MediatorReviseOrDecide(BaseModel):
    final_decision: str = Field(
        ..., description="The final decision made by the mediator."
    )
    keep_output: bool = Field(
        ..., description="Whether to keep the current output or revise it."
    )


def _extract_prompt_and_prior_turns(
    problem_statement: Any,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract active user prompt (ignoring UI system banners) and prior dialogue turns."""
    if isinstance(problem_statement, str):
        text = problem_statement.strip()
        if not text:
            raise ValueError("Task prompt must be a non-empty string.")
        return text, []

    if isinstance(problem_statement, dict):
        role = str(problem_statement.get("role", "user")).lower()
        content = str(problem_statement.get("content", "")).strip()
        if role == "system" or not content:
            raise ValueError("Task prompt cannot be empty or system-banner only.")
        return content, []

    if isinstance(problem_statement, list):
        non_system_turns: List[Dict[str, Any]] = []
        user_contents: List[str] = []
        fallback_contents: List[str] = []
        for idx, item in enumerate(problem_statement):
            if isinstance(item, dict):
                role = str(item.get("role", "user")).strip().lower()
                name = str(item.get("name") or ("User" if role == "user" else "Assistant")).strip()
                content = str(item.get("content", "")).strip()
                rnd = item.get("round", 0)
            elif hasattr(item, "content"):
                role = str(getattr(item, "role", "user") or "user").strip().lower()
                name = str(getattr(item, "name", "") or ("User" if role == "user" else "Assistant")).strip()
                content = str(getattr(item, "content", "") or "").strip()
                rnd = getattr(item, "round", 0)
            else:
                role = "user"
                name = "User"
                content = str(item or "").strip()
                rnd = 0

            if role == "system" or not content:
                continue
            turn_entry = {"role": role, "name": name, "round": rnd, "content": content}
            non_system_turns.append(turn_entry)
            if role == "user":
                user_contents.append(content)
            else:
                fallback_contents.append(content)

        if user_contents:
            active_prompt = user_contents[-1]
        elif fallback_contents:
            active_prompt = fallback_contents[-1]
        else:
            raise ValueError(
                "Conversation contains no actionable user task prompt (system banners are ignored)."
            )
        prior_turns = non_system_turns[:-1] if len(non_system_turns) > 1 else []
        return active_prompt, prior_turns

    text = str(problem_statement or "").strip()
    if not text:
        raise ValueError("Task prompt cannot be empty.")
    return text, []


def run_conversation(
    problem_statement,
    selected_personalities,
    lead_personality,
    num_rounds=3,
    context=None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
    max_calls: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    return run_conversation_result(
        problem_statement, selected_personalities, lead_personality, num_rounds,
        context, model, temperature, timeout_seconds, max_calls, cancel_check,
    ).legacy_tuple()


def run_conversation_result(
    problem_statement,
    selected_personalities,
    lead_personality,
    num_rounds=3,
    context=None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
    max_calls: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    progress = _ConversationProgress()
    artifact_scope = contextlib.nullcontext() if get_active_run_id() else artifact_run_scope(None)
    with artifact_scope, _conversation_run_scope(
        model=model,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_calls=max_calls,
        cancel_check=cancel_check,
    ):
        try:
            return _run_conversation_impl(
                problem_statement=problem_statement,
                selected_personalities=selected_personalities,
                lead_personality=lead_personality,
                num_rounds=num_rounds,
                context=context,
                _progress=progress,
            )
        except ProviderPolicyError as exc:
            revision = getattr(exc, "revision_result", None)
            if revision is not None:
                progress.conversation.append({"role": "assistant", "name": "Partial Output", "content": revision.accepted.final_output})
                exc.conversation_result = ConversationResult(output=failure_output(type(exc).__name__, "Execution stopped during output revision."),
                    conversation=copy.deepcopy(progress.conversation), questions=_normalized_questions(progress.questions))
            raise
        except LLMCallError:
            return ConversationResult(output=failure_output("ProviderError", "The provider could not complete the conversation."),
                conversation=copy.deepcopy(progress.conversation), questions=_normalized_questions(progress.questions))


def _run_conversation_impl(
    problem_statement,
    selected_personalities,
    lead_personality,
    num_rounds=3,
    context=None,
    _progress=None,
):
    raw_user_prompt, prior_turns = _extract_prompt_and_prior_turns(problem_statement)

    original_prompt = (
        "The Primary User has provided the following prompt, from which the problem statement will be defined: "
        + raw_user_prompt
    )
    if prior_turns:
        transcript_lines = [
            f"{t.get('name') or t.get('role')}: {t.get('content')}"
            for t in prior_turns[-6:]
        ]
        original_prompt += "\n\nPrior Conversation Context:\n" + "\n".join(transcript_lines)

    if context:
        original_prompt += f"\n\nAdditional Context: {context}"

    print("problem statement with context (if any)", original_prompt, "\n")

    if isinstance(selected_personalities, dict):
        personalities_for_problem = dict(selected_personalities)
    elif isinstance(selected_personalities, (list, tuple, set)):
        personalities_for_problem = {
            p: assistant_personalities.get(p, f"You are a helpful assistant acting as {p}.")
            for p in selected_personalities
        }
    else:
        personalities_for_problem = dict(assistant_personalities)
    # Keep per-run personality mapping strictly local without mutating module-global catalog
    run_personalities = {**assistant_personalities, **personalities_for_problem}

    problem_definition = define_problem(
        original_prompt,
        personalities_for_problem,
    )
    conversation_history = [
        {
            "role": "system",
            "name": "system",
            "round": 0,
            "content": assistant_instructions,
        },
        *[dict(t) for t in prior_turns],
        {
            "role": "user",
            "name": "Primary User",
            "round": 0,
            "content": problem_definition,
        },
    ]
    # print(conversation_history)
    conversation_memory = ConversationMemory(num_rounds)
    mediator_name = "Mediator" if "Mediator" in run_personalities else lead_personality
    questions_asked = []
    all_questions_asked: List[Dict[str, str]] = []
    if _progress is not None:
        _progress.conversation = conversation_history
        _progress.questions = all_questions_asked
    output_type = None

    final_output = None

    for rnd in range(num_rounds):
        print(f"Round {rnd + 1} of {num_rounds} \n")

        if rnd > 0:
            # Summarize the conversation so far
            summary = summarize_conversation(conversation_history)
            last_round_summary = summarize_conversation(
                [msg for msg in conversation_history if msg["round"] == rnd - 1]
            )

            if rnd > 1:
                # delete conversation summary from conversation history from previous round
                conversation_history[:] = [
                    msg
                    for msg in conversation_history
                    if "Full Conversation Summary:" not in msg["content"]
                ]
            print(f"\nFull Conversation Summary: {summary} \n")
            conversation_history.append(
                {
                    "role": "system",
                    "name": "Mediator",
                    "round": rnd,
                    "content": f"Conversation Summary: {summary}",
                }
            )
            conversation_history.append(
                {
                    "role": "system",
                    "name": "Mediator",
                    "round": rnd,
                    "content": f"Last Round Summary: {last_round_summary}",
                }
            )

        output_types = []
        if rnd == 0 and output_type is None:
            for assistant_name in [
                assistant_name
                for assistant_name in selected_personalities
                if assistant_name != mediator_name
            ]:
                # Determine the output type based on the defined problem statement
                output_types.append(output_type_determination(problem_definition))
                print(f"\nOutput Type: {output_types[-1]}\n")
                conversation_history.append(
                    {
                        "role": "assistant",
                        "name": assistant_name,
                        "round": rnd,
                        "content": f"Output Type: {output_types[-1]}",
                    }
                )
            # Have the mediator review the responses and decide on the output type
            mediator_prompt = (
                run_personalities.get(mediator_name, assistant_personalities.get("Mediator", ""))
                + "You are tasked with reviewing the output types suggested by the assistants and making the final decision on the output type for the solution. Please consider the problem statement and the context of the discussion to determine the most appropriate output format. Make your decision based on the following options: simple concise text answer, a detailed report in text or PDF format, a code snippet or script file, structured data in JSON or CSV format, a website or app prototype, or a detailed technical document. You may be more specific if needed or choose a different output type, but ensure it aligns with the problem statement and the discussion so far and meets the user's requirements and expectations. Please provide your decision in the format specified, including both the output type and the file extension."
            )
            valid_output_types = [ot for ot in output_types if isinstance(ot, OutputType)]
            try:
                if not valid_output_types:
                    raise OutputTypeError("All output type suggestions returned errors.")
                # TODO: Make this votable by the assistants
                with reserve_call_scope(
                    model=_resolve_model("gpt-4o-mini"),
                    call_type="mediator_output_type",
                ) as resolved_model:
                    response = client.beta.chat.completions.parse(
                        model=resolved_model,
                        messages=[
                            {"role": "system", "content": mediator_prompt},
                            {
                                "role": "user",
                                "content": f"Please decide on the output type based on the defined problem statement:\n\n{problem_definition}. The suggested output types are: {', '.join(ot.output_type for ot in valid_output_types)}. Possible file extensions are: {', '.join(ot.file_extension for ot in valid_output_types)}. Please provide your recommendation in the provided format, generating both the specific output type and the file extension.",
                            },
                        ],
                        response_format=OutputType,
                    )
                # Extract the output type from the response
                output_type = response.choices[0].message.parsed
                print(f"\nMediator Output Type Decision: {output_type}\n")
            except (ProviderPolicyError, DisallowedModelError):
                raise
            except Exception as e:
                # TODO: Implement better fallback strategy for output type determination
                print(f"\nError in Mediator Output Type Decision: {str(e)}\n")
                output_type = DEFAULT_OUTPUT_TYPE  # Use safe fallback instead of error string
                print(f"\nFalling back to default output type: {output_type}\n")

        # Calculate priorities - ensure output_type is always a valid OutputType object
        if not isinstance(output_type, OutputType):
            # FIXME: This should never happen with proper error handling above
            print(f"Warning: output_type is not OutputType object, using default: {type(output_type)}")
            output_type = DEFAULT_OUTPUT_TYPE
        assistant_priorities = [
            (
                assistant_name,
                calculate_priority(
                    assistant_name,
                    conversation_history,
                    conversation_memory,
                    selected_personalities,
                    rnd,
                    num_rounds,
                    lead_personality,
                    output_type,
                ),
            )
            for assistant_name in selected_personalities
        ]
        assistant_priorities.sort(key=lambda x: x[1], reverse=True)

        # Proceed in the order of priority
        for assistant_name, _ in assistant_priorities:
            assistant_prompt = run_personalities.get(
                assistant_name, f"You are a helpful assistant acting as {assistant_name}."
            )
            mediator_base_prompt = run_personalities.get(
                mediator_name, assistant_personalities.get("Mediator", "")
            )
            mediator_prompt = mediator_base_prompt
            mediator_initial_response = f"You are the Mediator for this conversation. This is the first round of {num_rounds} to produce the final output. Here's the problem statement: {problem_definition}. Using simple, concise language, guide the discussion towards a final output in the expected format by the end of the final round (round {num_rounds}). This first round is about making a plan or initial decision on how to approach the problem and what steps to take in the next {num_rounds - 1} rounds."

            if rnd == 0:
                mediator_prompt = (
                    mediator_base_prompt
                    + assistant_instructions
                    + f"This is the first round of {num_rounds} to produce the final output. Here's the problem statement: {problem_definition}. Your task is to guide the discussion towards a final output in the expected format by the end of the final round (round {num_rounds}). You should guide this first round towards making a plan or initial decision on how to approach the problem and what steps to take in the next {num_rounds - 1} rounds."
                )
                mediator_initial_response = f"You are the Mediator for this conversation. This is the first round of {num_rounds} to produce the final output. Here's the problem statement: {problem_definition}. Using simple, concise language, guide the discussion towards a final output in the expected format by the end of the final round (round {num_rounds}). This first round is about making a plan or initial decision on how to approach the problem and what steps to take in the next {num_rounds - 1} rounds."
            elif rnd > 0 and rnd + 1 < num_rounds:
                mediator_prompt = (
                    mediator_base_prompt
                    + assistant_instructions
                    + f"There are {num_rounds - rnd} rounds left to finalize the output. This is the beginning of round {rnd + 1}, please guide the discussion towards a final output in the expected format by the end of the final round (round {num_rounds})."
                )
                mediator_initial_response = (
                    f"Round {rnd} Summary: {last_round_summary}, the original problem statement was: {problem_definition}. Based on the conversation so far, please guide the discussion into the next round.",
                )
            elif rnd + 1 == num_rounds:
                mediator_prompt = (
                    mediator_base_prompt
                    + assistant_instructions
                    + f"This is the final round of the conversation. Please guide the discussion towards a final output in the expected format by the end of this round."
                )
                mediator_initial_response = f"Round {rnd} Summary: {last_round_summary}, the original problem statement was: {problem_definition}. Based on the conversation so far, please guide the discussion into the final round to reach a conclusion or decision by the end of this round."
            mediator_response = get_assistant_response(
                mediator_name,
                mediator_prompt,
                [msg for msg in conversation_history if msg["round"] == rnd],
                mediator_initial_response,
                conversation_memory,
            )

            print(
                f"\n \n Name: {mediator_name} \n Role: {safe_get_message_field(mediator_response, 'role', 'Unknown')} \n Response: {safe_get_message_field(mediator_response, 'content', 'No content')} \n"
            )
            mediator_response = normalize_message(mediator_response, default_role="system")
            conversation_history.append(
                {
                    "role": mediator_response["role"],
                    "name": mediator_name,
                    "round": rnd,
                    "content": "Mediator: " + mediator_response["content"],
                }
            )

            assistant_response = get_assistant_response(
                assistant_name,
                assistant_prompt + "\n" + assistant_instructions,
                # Include only the messages from the current round and exclude the first two messages
                [msg for msg in conversation_history[2:] if msg["round"] == rnd],
                (
                    "\n Here's the problem statement: "
                    + problem_definition
                    + f"\n This is the first round of {num_rounds} to produce the final output. What are your thoughts?"
                    if rnd == 0
                    else f"Round {rnd} Summary: {last_round_summary}, the original problem statement was: {problem_definition}. Based on the conversation so far, what are your thoughts? It is currently round {rnd + 1}, and we must reach a conclusion or decision by the end of round {num_rounds}, so we have {num_rounds - rnd} rounds left to finalize the output."
                ),
                conversation_memory,
            )
            print(
                f"\n \n Name: {assistant_name} \n Role: {safe_get_message_field(assistant_response, 'role', 'Unknown')} \n Response: {safe_get_message_field(assistant_response, 'content', 'No content')} \n"
            )
            
            assistant_response = normalize_message(assistant_response)
            
            conversation_history.append(
                {
                    "role": assistant_response["role"],
                    "name": assistant_name,
                    "round": rnd,
                    "content": assistant_response["content"],
                }
            )

            # Analyze assistant response to extract facts or decisions
            conversation_memory = extract_information(
                assistant_response["content"],
                conversation_memory,
                selected_personalities,
                rnd,
            )

            # Check if the conversation_memory contains any direct replies
            if conversation_memory.direct_replies:
                for assistant, reply in conversation_memory.direct_replies.items():
                    questions_asked.append(
                        {
                            "assistant": assistant,
                            "reply": reply,
                            "asking_assistant": assistant_name,
                        }
                    )
                conversation_memory.direct_replies = {}

        todo_analysis = analyze_to_do_list(
            conversation_memory,
            [msg for msg in conversation_history if msg["round"] == rnd],
        )
        if isinstance(todo_analysis, ConversationMemory):
            conversation_memory = todo_analysis
        print(
            f"\nMemory Summary in Round {rnd}: {conversation_memory.get_memory_summary()}\n"
        )

        # After each round, have the Mediator summarize
        mediator_prompt = run_personalities.get(
            mediator_name, assistant_personalities.get("Mediator", "")
        )
        mediator_response = get_assistant_response(
            mediator_name,
            mediator_prompt,
            [msg for msg in conversation_history if msg["round"] == rnd],
            (
                assistant_instructions
                + "\n Here's the problem statement: "
                + problem_definition
                + "Please speak to the group as Mediator to summarize the key points and guide the discussion towards consensus."
                if rnd == 0
                else f"Round {rnd} Summary: {last_round_summary}, the original problem statement was: {problem_definition}. Based on the conversation so far, please speak to the group as mediator to summarize the key points and guide the discussion into the next round."
            ),
            conversation_memory,
        )
        print(
            f"\n Name: {mediator_name} \n Role: {safe_get_message_field(mediator_response, 'role', 'Unknown')} \n Response: {safe_get_message_field(mediator_response, 'content', 'No content')} \n"
        )

        mediator_response = normalize_message(mediator_response, default_role="system")
        conversation_history.append(
            {
                "role": mediator_response["role"],
                "name": "Mediator",
                "round": rnd,
                "content": mediator_response["content"],
            }
        )
        conversation_memory = extract_information(
            mediator_response["content"],
            conversation_memory,
            selected_personalities,
            rnd,
        )

        # Handle Socratic responses by prompting other assistants to answer
        for q in questions_asked:
            for responder_name in selected_personalities:
                if responder_name == q["assistant"]:
                    responder_prompt = run_personalities.get(
                        responder_name,
                        f"You are a helpful assistant acting as {responder_name}.",
                    )
                    responder_response = get_assistant_response(
                        responder_name,
                        responder_prompt,
                        [msg for msg in conversation_history if msg["round"] == rnd],
                        None,
                        conversation_memory,
                        False,
                        {
                            "reply": q["reply"],
                            "asking_assistant": q["asking_assistant"],
                        },
                    )

                    responder_response = normalize_message(responder_response)

                    conversation_history.append(
                        {
                            "role": "assistant",
                            "name": q["asking_assistant"],
                            "round": rnd,
                            "content": q["reply"],
                        }
                    )
                    print(
                        f"\n Name: {q['asking_assistant']} is directing a question to {responder_name}. Question: {q['reply']} \n Role: {responder_response['role']} \n Response from {responder_name}: {responder_response['content']} \n \n"
                    )

                    conversation_history.append(
                        {
                            "role": responder_response["role"],
                            "name": responder_response["name"],
                            "round": rnd,
                            "content": f"In response to {q['asking_assistant']}: {responder_response['content']}",
                        }
                    )
                    conversation_memory = extract_information(
                        responder_response["content"],
                        conversation_memory,
                        selected_personalities,
                        rnd,
                    )
        # Clear questions after handling

        print(f"\n\nQuestions Asked in Round {rnd}: {questions_asked}\n\n")
        todo_analysis_post = analyze_to_do_list(
            conversation_memory,
            [msg for msg in conversation_history if msg["round"] == rnd],
        )
        if isinstance(todo_analysis_post, ConversationMemory):
            conversation_memory = todo_analysis_post
        print(
            f"\n\nMemory Summary in Round {rnd}: {conversation_memory.get_memory_summary()}\n\n"
        )
        # decrement the number of rounds in ConversationMemory
        conversation_memory.decrement_rounds_left()
        if questions_asked:
            all_questions_asked.extend(
                {
                    "assistant": str(q.get("assistant", "")),
                    "asking_assistant": str(q.get("asking_assistant", "")),
                    "reply": str(q.get("reply", "")),
                    "question": str(q.get("reply", "")),
                }
                for q in questions_asked
            )
        questions_asked = []
        # Vote on whether to continue the conversation into the next round or end it based on whether this round has reached a conclusion or contains enough information to vote on a final decision/solution
        if rnd <= num_rounds - 1:
            print(
                f"\nRound {rnd + 1} Summary: {summarize_conversation([msg for msg in conversation_history if msg['round'] == rnd])}\n"
            )
            call_for_final_vote = False
            if rnd < num_rounds - 1:
                final_vote_caller = None
                continue_votes = 0
                end_votes = 0
                voting_errors = 0
                
                for assistant_name, _ in assistant_priorities:
                    vote_prompt = "Please cast a binary vote to determine whether to continue the conversation into the next round. Based on the discussion so far, do you believe another round is needed to reach a conclusion or make a decision? Respond with a boolean value (True/False) only. True to continue, False to vote on a final decision. If you vote False, the conversation will proceed to a final vote, but you will need to provide a final decision or solution based on the information available."
                    continue_vote = cast_binary_vote(
                        assistant_name,
                        [msg for msg in conversation_history if msg["round"] == rnd],
                        vote_prompt,
                    )
                    
                    # Handle voting results with proper error checking
                    if isinstance(continue_vote, BinaryVote):
                        if continue_vote.vote:
                            continue_votes += 1
                        else:
                            end_votes += 1
                            if final_vote_caller is None:  # First assistant to vote to end
                                final_vote_caller = assistant_name
                    else:
                        # Handle voting errors
                        print(f"⚠️  Error getting continue vote from {assistant_name}: {continue_vote}")
                        voting_errors += 1
                
                total_continue_votes = continue_votes + end_votes
                print(f"📊 Continue Vote Results: {continue_votes} CONTINUE, {end_votes} END, {voting_errors} ERRORS")
                
                # Determine whether to continue based on majority or first end vote
                if end_votes > 0 and final_vote_caller is not None:
                    call_for_final_vote = True
                    print(f"\nFinal Vote Called by {final_vote_caller} to end the conversation in Round {rnd + 1}.\n")
                elif total_continue_votes == 0:
                    # All votes failed - default to ending if we're near the end, otherwise continue
                    if rnd >= (num_rounds - 2):
                        call_for_final_vote = True
                        final_vote_caller = lead_personality
                        print(f"\nFinal Vote Called due to voting errors and approaching round limit.\n")
                    else:
                        print("⚠️ All continue votes failed, but continuing due to early round number")
                else:
                    print(f"✓ Continuing to next round (Round {rnd + 2})")
            
            if rnd == num_rounds - 1:
                call_for_final_vote = True
                final_vote_caller = lead_personality
                print(
                    f"\nFinal Vote Called by {final_vote_caller} because the maximum number of rounds has been reached.\n"
                )

            if call_for_final_vote:
                print(f"\nFinal Vote. Round {rnd + 1}.\n")
                # Have the assistant who called for the final vote provide a final decision or solution based on the information available
                final_decision_prompt = "Please provide a final prompt based on the information available that will be used to generate the final output to solve the original problem statement. This prompt should be clear, actionable, and concise, providing the finalized details needed to generate the final output content that fulfills the problem statement and the user's needs. It should be in the form of an implementation plan, a request, or a directive that guides the generation of the final output. Ensure that the prompt will result in the direct creation of the final output content that fulfills the problem statement and the user's needs and will not require further discussion or clarification. Please provide the final prompt based on the information available."
                final_decision_messages = [
                    {
                        "role": "system",
                        "content": f"{run_personalities.get(final_vote_caller, '')} The conversation has reached a point where a final decision or solution is needed. Please provide your final decision or solution based on the information available. It should be clear, actionable, and concise, providing the finalized details needed to generate the final output content that fulfills the problem statement and the user's needs. Please do not include any further steps or discussions, only the final decision or solution.",
                    },
                    {
                        "role": "user",
                        "content": f"Please provide your final decision or solution based on the information available:\n\n{problem_definition}",
                    },
                    {
                        "role": "system",
                        "content": f"Problem Statement: {problem_definition}",
                    },
                ]
                final_decision_messages.append(
                    {"role": "system", "content": final_decision_prompt}
                )
                for message in final_decision_messages:
                    conversation_history.append(
                        {
                            "role": message["role"],
                            "name": "Mediator",
                            "round": rnd,
                            "content": message["content"],
                        }
                    )
                final_decision_response = get_assistant_response(
                    final_vote_caller,
                    run_personalities.get(final_vote_caller, "")
                    + " "
                    + final_decision_prompt,
                    conversation_history,
                    None,
                    conversation_memory,
                )
                if isinstance(final_decision_response, ErrorResult):
                    raise LLMCallError(final_decision_response.error_message)
                print(
                    f"\nFinal Decision Response from {final_vote_caller}: {final_decision_response}\n (Unconfirmed)"
                )
                # Cast a final vote to confirm the decision or solution
                final_vote_prompt = "Please cast a binary vote to confirm the final decision or solution provided by the assistant who called for the final vote. Respond with a boolean value (True/False) only, with True to confirm the decision or solution and False to reject it."
                yes_votes = 0
                no_votes = 0
                no_voters = []
                voting_errors = 0
                mediator_decision = None
                summary = summarize_conversation(conversation_history)

                for assistant_name, _ in assistant_priorities:
                    final_vote_messages = [
                        {
                            "role": "system",
                            "content": f"{run_personalities.get(assistant_name, '')} An assistant has recommended a solution for the problem statement {problem_definition}. Please review the conversation summary, then review the final decision or solution provided by {final_vote_caller} and cast your vote to confirm it.",
                        },
                        {
                            "role": "user",
                            "content": f"Conversation Summary: {summary}",
                        },
                        {
                            "role": "user",
                            "content": f"Please cast your vote to confirm the final decision or solution provided by {final_vote_caller}:\n\n{safe_get_message_field(final_decision_response, 'content', 'No decision content')}",
                        },
                        {
                            "role": "system",
                            "content": f"Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')}",
                        },
                    ]
                    final_binary_vote = cast_binary_vote(
                        assistant_name, final_vote_messages, final_vote_prompt
                    )
                    # Handle both successful votes and errors
                    if isinstance(final_binary_vote, BinaryVote):
                        if final_binary_vote.vote:
                            yes_votes += 1
                        else:
                            no_votes += 1
                            no_voters.append(assistant_name)
                    else:
                        # Handle voting errors
                        print(f"⚠️  Error getting vote from {assistant_name}: {final_binary_vote}")
                        voting_errors += 1
                
                # Calculate total participation for voting validation
                total_votes = yes_votes + no_votes
                total_assistants = len(assistant_priorities)
                print(f"📊 Final Vote Results: {yes_votes} YES, {no_votes} NO (Total: {total_votes}/{total_assistants} assistants)")
                
                # Check if the final decision is confirmed - Enhanced with tie-breaking and edge case handling
                
                # Handle edge cases first
                if total_votes == 0:
                    print("❌ CRITICAL: No valid votes received from any assistant!")
                    print("🔄 Falling back to mediator decision due to complete voting failure.")
                    # Force mediator to make final decision
                    mediator_fallback_prompt = (
                        f"{run_personalities.get(mediator_name, assistant_personalities.get('Mediator', ''))} "
                        "CRITICAL SITUATION: All assistants failed to cast valid votes on the final decision. "
                        "As the mediator, you must make the final call on whether to proceed with the proposed solution. "
                        "Review the conversation and make a definitive decision."
                    )
                    mediator_fallback_messages = [
                        {"role": "system", "content": mediator_fallback_prompt},
                        {"role": "user", "content": f"Proposed Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')}"},
                        {"role": "user", "content": f"Problem Statement: {problem_definition}"}
                    ]
                    try:
                        with reserve_call_scope(
                            model=_resolve_model("gpt-4o-mini"),
                            call_type="mediator_emergency",
                        ) as resolved_model:
                            mediator_emergency_response = client.beta.chat.completions.parse(
                                model=resolved_model,
                                messages=mediator_fallback_messages,
                                response_format=MediatorReviseOrDecide,
                            )
                        mediator_decision = mediator_emergency_response.choices[0].message.parsed
                        print(f"🏛️ Emergency Mediator Decision: {mediator_decision}")
                    except (ProviderPolicyError, DisallowedModelError):
                        raise
                    except Exception as e:
                        print(f"❌ Emergency mediator decision failed: {e}")
                        # Ultimate fallback - proceed with original decision
                        mediator_decision = MediatorReviseOrDecide(
                            final_decision=safe_get_message_field(final_decision_response, 'content', 'No decision content'),
                            keep_output=True
                        )
                        print("⚠️ Using ultimate fallback: proceeding with original proposal")
                
                elif yes_votes == no_votes and yes_votes > 0:
                    print(f"⚖️ TIE VOTE DETECTED: {yes_votes} YES vs {no_votes} NO")
                    print("🏛️ Invoking mediator tie-breaker protocol...")
                    
                    # Mediator breaks the tie
                    tie_breaker_prompt = (
                        f"{run_personalities.get(mediator_name, assistant_personalities.get('Mediator', ''))} "
                        "A tie vote has occurred on the final decision. As the mediator, you must break the tie. "
                        "Review the proposed solution, the conversation context, and make a definitive decision. "
                        "Consider the feedback from both sides and determine the best path forward."
                    )
                    tie_breaker_messages = [
                        {"role": "system", "content": tie_breaker_prompt},
                        {"role": "user", "content": f"Proposed Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')}"},
                        {"role": "user", "content": f"Problem Statement: {problem_definition}"},
                        {"role": "user", "content": f"Vote was tied: {yes_votes} assistants voted YES, {no_votes} assistants voted NO"}
                    ]
                    try:
                        with reserve_call_scope(
                            model=_resolve_model("gpt-4o-mini"),
                            call_type="mediator_tie_breaker",
                        ) as resolved_model:
                            tie_breaker_response = client.beta.chat.completions.parse(
                                model=resolved_model,
                                messages=tie_breaker_messages,
                                response_format=MediatorReviseOrDecide,
                            )
                        mediator_decision = tie_breaker_response.choices[0].message.parsed
                        print(f"🏛️ Tie-breaker Decision: {'ACCEPT' if mediator_decision.keep_output else 'REVISE'}")
                        conversation_history.append({
                            "role": "system",
                            "name": mediator_name,
                            "round": rnd,
                            "content": f"Mediator Tie-breaker Decision: {'Accepted' if mediator_decision.keep_output else 'Revised'} - {mediator_decision.final_decision}"
                        })
                    except (ProviderPolicyError, DisallowedModelError):
                        raise
                    except Exception as e:
                        print(f"❌ Tie-breaker decision failed: {e}")
                        # Default tie-breaker: slight preference for proceeding
                        mediator_decision = MediatorReviseOrDecide(
                            final_decision="Tie-breaker fallback: proceeding with original proposal due to decision error.",
                            keep_output=True
                        )
                        print("⚠️ Using tie-breaker fallback: slight preference for proceeding")
                
                elif total_votes < (total_assistants / 2):
                    print(f"⚠️ LOW PARTICIPATION: Only {total_votes}/{total_assistants} assistants voted (less than 50%)")
                    print("🔄 Considering insufficient participation...")
                    
                    # Check if we still have a clear majority despite low participation
                    if yes_votes > no_votes and yes_votes >= (total_assistants / 3):
                        mediator_decision = MediatorReviseOrDecide(
                            final_decision=safe_get_message_field(final_decision_response, 'content', ''),
                            keep_output=True,
                        )
                        print("✓ Proceeding despite low participation due to clear majority")
                    else:
                        print("❌ Insufficient participation and unclear mandate")
                        print("🏛️ Requiring mediator approval due to low participation")
                        # Force mediator review for low participation
                        participation_prompt = (
                            f"{run_personalities.get(mediator_name, assistant_personalities.get('Mediator', ''))} "
                            f"Low participation in final vote: only {total_votes}/{total_assistants} assistants voted. "
                            "As mediator, determine if this is sufficient to proceed or if the decision needs revision."
                        )
                        participation_messages = [
                            {"role": "system", "content": participation_prompt},
                            {"role": "user", "content": f"Vote Results: {yes_votes} YES, {no_votes} NO out of {total_assistants} total assistants"},
                            {"role": "user", "content": f"Proposed Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')}"}
                        ]
                        try:
                            with reserve_call_scope(
                                model=_resolve_model("gpt-4o-mini"),
                                call_type="mediator_low_participation",
                            ) as resolved_model:
                                participation_response = client.beta.chat.completions.parse(
                                    model=resolved_model,
                                    messages=participation_messages,
                                    response_format=MediatorReviseOrDecide,
                                )
                            mediator_decision = participation_response.choices[0].message.parsed
                            print(f"🏛️ Low Participation Decision: {'PROCEED' if mediator_decision.keep_output else 'REVISE'}")
                        except (ProviderPolicyError, DisallowedModelError):
                            raise
                        except Exception as e:
                            print(f"❌ Low participation decision failed: {e}")
                            mediator_decision = MediatorReviseOrDecide(
                                final_decision="Low participation fallback: proceeding with caution.",
                                keep_output=True
                            )

                # Normal voting logic (only executed if no edge cases above)
                if yes_votes > 0 and no_votes == 0 and total_votes >= (total_assistants / 2):
                    print(
                        f"\nFinal Decision Confirmed with full consensus: {safe_get_message_field(final_decision_response, 'content', 'No decision content')}\n"
                    )
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": mediator_name,
                            "round": rnd,
                            "content": f"Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')} (Confirmed)",
                        }
                    )
                    # Have the Mediator finalize the output based on the final decision, creating the final output content that fulfills the problem statement and the user's needs that will be saved to the final file

                    final_content_prompt = "Based on the final decision or solution provided by the assistant who called for the final vote, please generate the final output content that fulfills the problem statement and the user's needs. This means creating the final output content based on the decision or solution provided, ensuring it aligns with the problem statement and the conversation so far and meets the user's requirements and expectations. For instance, if the final decision is a text answer, generate the text answer. If it's a code snippet, generate the code snippet. If it's a report, generate the report content. If it's a prototype, generate the prototype content. If it's structured data, generate the structured data content. If it's a technical document, generate the technical document content. Please provide the final output content that fulfills the problem statement and the user's needs based on the final decision or solution provided."
                    final_content_messages = [
                        {
                            "role": "system",
                            "content": final_content_prompt
                            + f" The output type for the final decision is: {output_type.output_type} with file extension: {output_type.file_extension} and should solve the problem statement: {problem_definition} by following the final decision or solution provided by {final_vote_caller}."},
                        {
                            "role": "user",
                            "content": f"Please generate the final output content based on the final decision or solution provided by {final_vote_caller}:\n\n{safe_get_message_field(final_decision_response, 'content', 'No decision content')}. The problem statement it must completely solve is: {problem_definition}. The output you generate should be the final completion of the problem statement and the user's needs, and should not include any further steps or discussions but will represent the final output content that fulfills the problem statement and the user's needs ONLY.",
                        },
                    ]
                    for message in final_content_messages:
                        conversation_history.append(
                            {
                                "role": message["role"],
                                "name": "Mediator",
                                "round": rnd,
                                "content": message["content"],
                            }
                        )

                    final_output_response = get_assistant_response(
                        lead_personality,
                        run_personalities.get(lead_personality, assistant_personalities.get("Mediator", ""))
                        + " "
                        + final_content_prompt,
                        conversation_history,
                        None,
                        conversation_memory,
                    )
                    if isinstance(final_output_response, ErrorResult):
                        raise LLMCallError(final_output_response.error_message)
                    print(
                        f"\n\nFinal Output Response from {lead_personality}: {final_output_response}\n\n"
                    )
                    final_checked_output = analyze_final_output_result(
                        safe_get_message_field(final_output_response, 'content', ''),
                        lead_personality,
                        selected_personalities,
                        original_prompt,
                        problem_definition,
                        output_type,
                    )
                    print(f"\n\nFinal Checked Output: {final_checked_output}\n\n")

                    final_output = _finalize_revision(final_checked_output, conversation_history, output_type)
                    print(f"\n\nFinal Output filepath: {final_output}\n\n")
                    # Add the final output to the conversation history and then save the conversation history to a file
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": lead_personality,
                            "round": rnd,
                            "content": conversation_output_annotation(final_output),
                            }
                    )
                    try:
                        with open("conversation_history.json", "w") as f:
                            json.dump(conversation_history, f, indent=4)
                        print("✓ Conversation history saved to conversation_history.json")
                    except (IOError, OSError, PermissionError) as e:
                        print(f"✗ Error saving conversation history to conversation_history.json: {e}")
                        print("⚠️  Conversation data may be lost. Check disk space and file permissions.")
                    except Exception as e:
                        print(f"✗ Unexpected error saving conversation history: {e}")
                        print("⚠️  Conversation data may be lost.")
                elif no_votes > 0 and mediator_decision is None:
                    print(
                        f"\nFinal proposal requires mediator review, {yes_votes} to {no_votes}: {safe_get_message_field(final_decision_response, 'content', 'No decision content')}\n"
                    )
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": mediator_name,
                            "round": rnd,
                            "content": f"Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')} (Pending mediator review)",
                        }
                    )
                    # Get feedback from the assistants who voted no and ask the Mediator to resolve the issue by revising the final decision or making a final call
                    feedbacks = []
                    for assistant_name, _ in assistant_priorities:
                        if assistant_name in no_voters:
                            feedback_messages = [
                                {
                                    "role": "system",
                                    "content": f"{run_personalities.get(assistant_name, '')} Your vote did not confirm the final decision or solution provided by {final_vote_caller}. Please provide feedback on the final decision or solution.",
                                },
                                {
                                    "role": "user",
                                    "content": f"Please provide feedback on the final decision or solution provided by {final_vote_caller} and suggest improvements or changes to align it better with the problem statement and the conversation so far.",
                                },
                                {
                                    "role": "user",
                                    "content": f"Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')}",
                                },
                            ]
                            feedback = get_assistant_response(
                                assistant_name,
                                run_personalities.get(assistant_name, assistant_prompt),
                                feedback_messages,
                            )
                            feedbacks.append(feedback)
                    # Have the Mediator review the feedback and make a final decision
                    mediator_feedback = []
                    for feedback in feedbacks:
                        mediator_feedback.append(safe_get_feedback_field(feedback, 'content', 'No feedback content'))
                    mediator_prompt = (
                        run_personalities.get(mediator_name, assistant_personalities.get("Mediator", ""))
                        + "The final decision or solution provided by the assistant who called for the final vote was not confirmed by the majority of the assistants. Please review the feedback provided by the assistants who voted no and make a final decision or resolution based on the feedback and the context of the discussion. You may choose to revise the final decision or make a final call based on the information available by answering in the format provided, with a boolean value of True to make the call to keep the final decision or False to revise it. If False, provide a revised final decision or solution (as final_decision) based on the feedback and the context of the discussion, otherwise, confirm the final decision."
                    )
                    mediator_messages = [
                        {"role": "system", "content": mediator_prompt},                        {
                            "role": "user",
                            "content": f"Feedback from assistants who voted no: {', '.join(safe_get_feedback_field(feedback, 'content', 'No feedback content') for feedback in feedbacks)}",
                        },
                    ]
                    mediator_decision = ""
                    try:
                        with reserve_call_scope(
                            model=_resolve_model("gpt-4o-mini"),
                            call_type="mediator_feedback",
                        ) as resolved_model:
                            response = client.beta.chat.completions.parse(
                                model=resolved_model,
                                messages=mediator_messages,  # Pass the messages directly as a list of dicts
                                response_format=MediatorReviseOrDecide,
                            )
                        # Extract the decision from the response
                        mediator_decision = response.choices[0].message.parsed
                        print(f"\nMediator Decision: {mediator_decision}\n")
                    except (ProviderPolicyError, DisallowedModelError):
                        raise
                    except Exception as e:
                        # TODO: Implement better fallback strategy for mediator decisions
                        print(f"\nError in Mediator Decision: {str(e)}\n")
                        # Create a safe fallback MediatorReviseOrDecide object
                        mediator_decision = MediatorReviseOrDecide(
                            final_decision="Error in mediator decision process. Proceeding with current output.",
                            keep_output=True  # Default to keeping output to avoid infinite loops
                        )
                    
                    # Ensure mediator_decision is the correct type
                    if not isinstance(mediator_decision, MediatorReviseOrDecide):
                        # FIXME: This should never happen with proper error handling above
                        print(f"Warning: mediator_decision is not MediatorReviseOrDecide object: {type(mediator_decision)}")
                        mediator_decision = MediatorReviseOrDecide(
                            final_decision="Fallback decision due to type error.",
                            keep_output=True
                        )
                    if mediator_decision.keep_output:
                        conversation_history.append(
                            {
                                "role": "system",
                                "name": mediator_name,
                                "round": rnd,
                                "content": f"Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')} (Confirmed)",
                            }
                        )
                    else:
                        final_decision_response["content"] = mediator_decision.final_decision
                        conversation_history.append({
                            "role": "system", "name": mediator_name, "round": rnd,
                            "content": f"Final Decision: {mediator_decision.final_decision} (Revised)",
                        })
                    # Have the Mediator finalize the output based on the final decision, creating the final output content that fulfills the problem statement and the user's needs that will be saved to the final file

                    final_content_prompt = "Based on the final decision or solution provided by the assistant who called for the final vote, please generate the final output content that fulfills the problem statement and the user's needs. This means creating the final output content based on the decision or solution provided, ensuring it aligns with the problem statement and the conversation so far and meets the user's requirements and expectations. For instance, if the final decision is a text answer, generate the text answer. If it's a code snippet, generate the code snippet. If it's a report, generate the report content. If it's a prototype, generate the prototype content. If it's structured data, generate the structured data content. If it's a technical document, generate the technical document content. Please provide the final output content that fulfills the problem statement and the user's needs based on the final decision or solution provided."
                    final_content_messages = [
                        {
                            "role": "system",
                            "content": final_content_prompt
                            + f" The output type for the final decision is: {output_type.output_type} with file extension: {output_type.file_extension} and should solve the problem statement: {problem_definition} by following the final decision or solution provided by {final_vote_caller}."},
                    {
                        "role": "user",
                        "content": f"Please generate the final output content based on the final decision or solution provided by {final_vote_caller}:\n\n{safe_get_message_field(final_decision_response, 'content', 'No decision content')}. The problem statement it must completely solve is: {problem_definition}. The output you generate should be the final completion of the problem statement and the user's needs, and should not include any further steps or discussions but will represent the final output content that fulfills the problem statement and the user's needs ONLY.",
                    },
                    ]
                    for message in final_content_messages:
                        conversation_history.append(
                            {
                                "role": message["role"],
                                "name": "Mediator",
                                "round": rnd,
                                "content": message["content"],
                            }
                        )

                    final_output_response = get_assistant_response(
                        lead_personality,
                        run_personalities.get(lead_personality, assistant_personalities.get("Mediator", ""))
                        + " "
                        + final_content_prompt,
                        conversation_history,
                        None,
                        conversation_memory,
                    )
                    if isinstance(final_output_response, ErrorResult):
                        raise LLMCallError(final_output_response.error_message)
                    print(
                        f"\n\nFinal Output Response from {lead_personality}: {final_output_response}\n\n"
                    )
                    final_checked_output = analyze_final_output_result(
                        safe_get_message_field(final_output_response, 'content', ''),
                        lead_personality,
                        selected_personalities,
                        original_prompt,
                        problem_definition,
                        output_type,
                    )
                    print(f"\n\nFinal Checked Output: {final_checked_output}\n\n")

                    final_output = _finalize_revision(final_checked_output, conversation_history, output_type)
                    print(f"\n\nFinal Output filepath: {final_output}\n\n")
                    # Add the final output to the conversation history and then save the conversation history to a file
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": lead_personality,
                            "round": rnd,
                            "content": conversation_output_annotation(final_output),
                            }
                    )
                    try:
                        with open("conversation_history.json", "w") as f:
                            json.dump(conversation_history, f, indent=4)
                        print("✓ Conversation history saved to conversation_history.json")
                    except (IOError, OSError, PermissionError) as e:
                        print(f"✗ Error saving conversation history to conversation_history.json: {e}")
                        print("⚠️  Conversation data may be lost. Check disk space and file permissions.")
                    except Exception as e:
                        print(f"✗ Unexpected error saving conversation history: {e}")
                        print("⚠️  Conversation data may be lost.")
                elif mediator_decision is not None and not mediator_decision.keep_output:
                    # Use mediator_decision.final_decision as the new final decision
                    final_decision_response["content"] = (
                        mediator_decision.final_decision
                    )
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": mediator_name,
                            "round": rnd,
                            "content": f"Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')} (Revised)",
                        }
                    )
                    # Have the Mediator finalize the output based on the final decision, creating the final output content that fulfills the problem statement and the user's needs that will be saved to the final file

                    final_content_prompt = "Based on the final decision or solution provided by the assistant who called for the final vote, please generate the final output content that fulfills the problem statement and the user's needs. This means creating the final output content based on the decision or solution provided, ensuring it aligns with the problem statement and the conversation so far and meets the user's requirements and expectations. For instance, if the final decision is a text answer, generate the text answer. If it's a code snippet, generate the code snippet. If it's a report, generate the report content. If it's a prototype, generate the prototype content. If it's structured data, generate the structured data content. If it's a technical document, generate the technical document content. Please provide the final output content that fulfills the problem statement and the user's needs based on the final decision or solution provided."
                    final_content_messages = [
                        {
                            "role": "system",
                            "content": final_content_prompt
                            + f" The output type for the final decision is: {output_type.output_type} with file extension: {output_type.file_extension} and should solve the problem statement: {problem_definition} by following the final decision or solution provided by {final_vote_caller}.",
                        },
                        {
                            "role": "user",
                            "content": f"Please generate the final output content based on the final decision or solution provided by {final_vote_caller}:\n\n{safe_get_message_field(final_decision_response, 'content', 'No decision content')}",
                        },
                    ]
                    for message in final_content_messages:
                        conversation_history.append(
                            {
                                "role": message["role"],
                                "name": mediator_name,
                                "round": rnd,
                                "content": message["content"],
                            }
                        )

                    final_output_response = get_assistant_response(
                        lead_personality,
                        run_personalities.get(lead_personality, assistant_personalities.get("Mediator", ""))
                        + " "
                        + final_content_prompt,
                        conversation_history,
                        None,
                        conversation_memory,
                    )
                    if isinstance(final_output_response, ErrorResult):
                        raise LLMCallError(final_output_response.error_message)
                    print(
                        f"\n\nFinal Output Response from {lead_personality}: {final_output_response}\n\n"
                    )
                    final_checked_output = analyze_final_output_result(
                        safe_get_message_field(final_output_response, 'content', ''),
                        lead_personality,
                        selected_personalities,
                        original_prompt,
                        problem_definition,
                        output_type,
                    )
                    print(f"\n\nFinal Checked Output: {final_checked_output}\n\n")

                    final_output = _finalize_revision(final_checked_output, conversation_history, output_type)
                    print(f"\n\nFinal Output filepath: {final_output}\n\n")
                    # Add the final output to the conversation history and then save the conversation history to a file
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": lead_personality,
                            "round": rnd,
                            "content": conversation_output_annotation(final_output),
                            }
                    )
                elif mediator_decision is not None and mediator_decision.keep_output:
                    # Emergency or tie-breaker mediator decided to keep the proposed decision
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": mediator_name,
                            "round": rnd,
                            "content": f"Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')} (Confirmed by Mediator)",
                        }
                    )
                    final_content_prompt = "Based on the final decision or solution provided by the assistant who called for the final vote, please generate the final output content that fulfills the problem statement and the user's needs. This means creating the final output content based on the decision or solution provided, ensuring it aligns with the problem statement and the conversation so far and meets the user's requirements and expectations. For instance, if the final decision is a text answer, generate the text answer. If it's a code snippet, generate the code snippet. If it's a report, generate the report content. If it's a prototype, generate the prototype content. If it's structured data, generate the structured data content. If it's a technical document, generate the technical document content. Please provide the final output content that fulfills the problem statement and the user's needs based on the final decision or solution provided."
                    final_content_messages = [
                        {
                            "role": "system",
                            "content": final_content_prompt
                            + f" The output type for the final decision is: {output_type.output_type} with file extension: {output_type.file_extension} and should solve the problem statement: {problem_definition} by following the final decision or solution provided by {final_vote_caller}.",
                        },
                        {
                            "role": "user",
                            "content": f"Please generate the final output content based on the final decision or solution provided by {final_vote_caller}:\n\n{safe_get_message_field(final_decision_response, 'content', 'No decision content')}. The problem statement it must completely solve is: {problem_definition}. The output you generate should be the final completion of the problem statement and the user's needs, and should not include any further steps or discussions but will represent the final output content that fulfills the problem statement and the user's needs ONLY.",
                        },
                    ]
                    for message in final_content_messages:
                        conversation_history.append(
                            {
                                "role": message["role"],
                                "name": "Mediator",
                                "round": rnd,
                                "content": message["content"],
                            }
                        )

                    final_output_response = get_assistant_response(
                        lead_personality,
                        run_personalities.get(lead_personality, assistant_personalities.get("Mediator", ""))
                        + " "
                        + final_content_prompt,
                        conversation_history,
                        None,
                        conversation_memory,
                    )
                    if isinstance(final_output_response, ErrorResult):
                        raise LLMCallError(final_output_response.error_message)
                    print(
                        f"\n\nFinal Output Response from {lead_personality}: {final_output_response}\n\n"
                    )
                    final_checked_output = analyze_final_output_result(
                        safe_get_message_field(final_output_response, 'content', ''),
                        lead_personality,
                        selected_personalities,
                        original_prompt,
                        problem_definition,
                        output_type,
                    )
                    print(f"\n\nFinal Checked Output: {final_checked_output}\n\n")

                    final_output = _finalize_revision(final_checked_output, conversation_history, output_type)
                    print(f"\n\nFinal Output filepath: {final_output}\n\n")
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": lead_personality,
                            "round": rnd,
                            "content": conversation_output_annotation(final_output),
                        }
                    )
        if isinstance(final_output, ConversationOutput) and final_output.kind == "failure":
            break
    try:
        filename = f"conversation_history_{str(datetime.now()).replace(':', '-').replace(' ', '_')}.json"
        with open(filename, "w") as f:
            json.dump(conversation_history, f, indent=4)
        print(f"✓ Final conversation history saved to {filename}")
    except (IOError, OSError, PermissionError) as e:
        print(f"✗ Error saving final conversation history to {filename}: {e}")
        print("⚠️  Final conversation data may be lost. Check disk space and file permissions.")
    except Exception as e:
        print(f"✗ Unexpected error saving final conversation history: {e}")
        print("⚠️  Final conversation data may be lost.")
    
    # print("Conversation History in manager: ", conversation_history, "Type: ", type(conversation_history), "Type Content: ", type(conversation_history[0]["content"]))
    if not isinstance(final_output, ConversationOutput):
        final_output = failure_output("EmptyOutput", "The conversation returned no usable output.", locals().get("output_type"))
    return ConversationResult(output=final_output, conversation=copy.deepcopy(conversation_history),
                              questions=_normalized_questions(all_questions_asked))

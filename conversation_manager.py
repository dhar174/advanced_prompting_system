# conversation_manager.py

from datetime import datetime

from typing import Any, Dict, List, Optional, Tuple, Union # Added Optional, Tuple, Union
from jarowinkler import jarowinkler_similarity


# jarowinkler_similarity("this is an example".split(), ["this", "is", "a", "example"])
# 0.8666666666666667
import openai
import os
import re
import json

from assistant_personalities import assistant_personalities, assistant_instructions
from schema import ConversationInput
from pydantic import BaseModel, Field, ValidationError
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
    generate_simple_concise_answer_message,
    generate_json_output,
    generate_pdf_output,
    generate_text_file_output,
    generate_html_output,
    generate_python_script,
    generate_code_snippet,
    generate_csv_output,
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

class ValidationError(ConversationManagerError):
    """Exception raised when data validation fails."""
    pass

# Error Result Classes for Type-Safe Error Handling
class ErrorResult(BaseModel):
    """Structured error result to replace error strings."""
    error_type: str = Field(..., description="Type of error that occurred")
    error_message: str = Field(..., description="Human-readable error message")
    error_details: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the error occurred")

# Default fallback values for type safety
DEFAULT_OUTPUT_TYPE = OutputType(output_type="simple_text", file_extension="txt")

# Configuration constants for output type matching
DEFAULT_FALLBACK_OUTPUT_TYPE = "Simple Concise Answer"
MINIMUM_SIMILARITY_THRESHOLD = 0.1

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

class OutputType(BaseModel):
    output_type: str = Field(..., description="The type of output to generate.")
    file_extension: str = Field(..., description="The file extension for the output.")


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
client = openai.OpenAI()


class ConversationMemory:
    def __init__(self, num_rounds: int = 3) -> None:
        self.facts: List[str] = []
        self.arguments: List[Dict[str, Any]] = []  # [{"round": 1, "arguments": [argument1, argument2, ...]}]
        self.decisions: List[Dict[str, Any]] = []  # [{"round": 1, "decision": decision1}]
        self.direct_replies: Dict[str, str] = {}  # {"assistant_name": "reply"}
        self.recommended_actions: List[str] = []  # [action1, action2, ...]
        self.to_do_list: List[Any] = []  # [item1, item2, ...] # FIXME: Define a more specific type for to_do_list items
        self.completed_tasks: Dict[str, str] = {}  # {"task": "completion"}
        self.rounds_left: int = num_rounds
        self.decided_output_type: Optional[OutputType] = None # Changed to Optional[OutputType]

    def add_fact(self, fact: str) -> None:
        if fact and fact not in self.facts:
            self.facts.append(fact)

    def add_argument(self, argument: Any, round_num: int) -> None: # FIXME: Define a more specific type for argument
        if argument and argument not in self.arguments:
            self.arguments.append({"round": round_num, "arguments": argument})

    def add_decision(self, decision: Any, round_num: int) -> None: # FIXME: Define a more specific type for decision
        if decision and decision not in self.decisions:
            self.decisions.append({"round": round_num, "decision": decision})

    def add_direct_reply(self, assistant_name: str, reply: str) -> None:
        if assistant_name and reply:
            self.direct_replies[assistant_name] = reply

    def add_recommended_action(self, action: str) -> None:
        if action and action not in self.recommended_actions:
            self.recommended_actions.append(action)

    def add_to_do_item(self, item: Any) -> None: # FIXME: Define a more specific type for item
        if item and item not in self.to_do_list:
            self.to_do_list.append(item)

    def add_completed_task(self, task: str, completion: str) -> None:
        if task and completion:
            self.completed_tasks[task] = completion
            # remove task from to do list
            if task in self.to_do_list:
                self.to_do_list.remove(task)

    def decrement_rounds_left(self) -> None:
        self.rounds_left -= 1

    def set_decided_output_type(self, output_type: OutputType) -> None:
        self.decided_output_type = output_type

    def get_memory_summary(self) -> str:
        summary = ""
        if self.facts:
            summary += (
                "Known Facts:\n"
                + "\n".join(f"- {fact}" for fact in self.facts)
                + "\n\n"
            )
        if self.arguments:
            summary += (
                "Arguments:\n"
                + "\n".join(f"- {arg['arguments']}" for arg in self.arguments)
                + "\n\n"
            )
        if self.decisions:
            summary += (
                "Decisions:\n"
                + "\n".join(f"- {decision['decision']}" for decision in self.decisions)
                + "\n\n"
            )
        if self.recommended_actions:
            summary += (
                "Recommended Actions:\n"
                + "\n".join(f"- {action}" for action in self.recommended_actions)
                + "\n\n"
            )
        if self.decided_output_type:
            summary += f"Decided Output Type of Final Output: {self.decided_output_type.output_type}\n\n"
        if self.to_do_list:
            # Convert completed tasks to a set of tuples
            completed_tasks_set = {
                tuple(task) if isinstance(task, list) else task
                for task in self.completed_tasks
            }

            # Filter to-do items
            to_do_items = [
                item
                for item in self.to_do_list
                if (tuple(item) if isinstance(item, list) else item)
                not in completed_tasks_set
            ]

            # Ensure all items are lists
            to_do_items = [
                item if isinstance(item, list) else [item] for item in to_do_items
            ]

            # Join list items into strings and format them
            to_do_items = [
                f"- {', '.join(map(str, item)).replace('[', '').replace(']', '')}"
                for item in to_do_items
            ]            # Add to summary
            summary += "To Do List:\n" + "\n".join(to_do_items) + "\n\n"
        # TODO: Uncomment and implement completed tasks display when needed
        #     summary += "Completed Tasks:\n" + "\n".join(f"- {completion}" for completion in self.completed_tasks) + "\n\n"
        summary += f"Rounds Left in Conversation to Complete Solution to Problem Statement: {self.rounds_left}"

        return summary


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
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Replace with the appropriate model
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
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Replace with the appropriate model if necessary
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
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Specify the model
            messages=messages,  # Pass the messages directly as a list of dicts
            response_format=BinaryVote,
        )
        # Extract the vote from the response
        vote = response.choices[0].message.parsed
        return vote
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
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Specify the model
            messages=messages,  # Pass the messages directly as a list of dicts
            response_format=ConfidenceVote,
        )
        # Extract the vote from the response
        confidence = response.choices[0].message.parsed
        return confidence
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
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Specify the model
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
                    "content": f"Problem Statement: {definition['definition']}",            },        ]
        confidence = cast_confidence_vote(assistant_name, messages)
        print(
            f"\nConfidence Vote from {assistant_name} on {definition['assistant']}'s definition: {confidence}\n"
        )        # Handle both successful votes and errors
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
                print(f"\nNo Vote from {assistant_name} on the consensus problem statement.\n")        else:
            # Handle binary vote errors more gracefully
            print(f"⚠️  Error getting vote from {assistant_name}: {binary_vote}")
            no_votes += 1  # Count errors as no votes for safety
            no_voters.append(assistant_name)
        
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
            if revision_response["content"].startswith("Revision:"):
                best_definition["definition"] = (
                    revision_response["content"].replace("Revision:", "").strip()
                )
            elif revision_response["content"].startswith("Justification:"):
                # Ask the assistant to vote again based on the justification
                justification_messages = [
                    {
                        "role": "system",
                        "content": f"{assistant_prompt} Please review the justification provided by {best_definition['assistant']} and cast your vote to confirm the consensus problem statement.",
                    },
                    {
                        "role": "user",
                        "content": f"Please cast your vote to confirm the consensus problem statement based on the justification provided by {best_definition['assistant']}:\n\n{safe_get_message_field(revision_response, 'content', 'No revision content')}",
                    },                    {
                        "role": "system",
                        "content": f"Defined Problem Statement: {best_definition['definition']}, Justification: {safe_get_message_field(revision_response, 'content', 'No justification content')}",
                    },
                ]
                binary_vote = cast_binary_vote(
                    assistant_name, justification_messages, bin_vote_prompt
                )
                if binary_vote:
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
    elif yes_votes > no_votes and no_votes > 0 and no_votes >= (yes_votes / 2):
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
            mediator_prompt = selected_personalities["Mediator"]
            mediator_messages = [
                {
                    "role": "system",
                    "content": f"{mediator_prompt} The consensus problem statement was not confirmed. Please review the feedback provided by the assistants and the justification for the problem statement.",
                },
                {
                    "role": "user",
                    "content": f"Please rewrite the problem statement based on the feedback provided by the assistants and the justification. The original prompt was:\n\n{problem_statement} and the revised problem statement was:\n\n{best_definition['definition']}",
                },                {
                    "role": "user",
                    "content": f"Feedback from {best_definition['assistant']}: {safe_get_message_field(revision_response, 'content', 'No revision content')}\n\nFeedback from other assistants: {', '.join(safe_get_feedback_field(feedback, 'content', 'No feedback content') for feedback in feedbacks)}",
                },
            ]
            mediator_response = get_assistant_response(
                "Mediator", mediator_prompt, mediator_messages            )
            print(f"\nMediator Rewrite Response: {safe_get_message_field(mediator_response, 'content', 'No mediator response')}\n")
            best_definition["definition"] = safe_get_message_field(mediator_response, 'content', 'Default problem definition')

        # Ask the assistants to vote again on the revised problem statement
        for assistant_name, assistant_prompt in selected_personalities.items():
            revised_bin_vote_prompt = "Please cast a binary vote to confirm the revised problem statement. Respond with a boolean value (True/False) only."
            revised_yes_votes = 0
            revised_no_votes = 0
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
                if revised_binary_vote:
                    revised_yes_votes += 1
                else:
                    revised_no_votes += 1
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
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Specify the model
            messages=messages,  # Pass the messages directly as a list of dicts
            response_format=OutputType,
            max_completion_tokens=100,
        )
        # Extract the output type from the response
        output_type = response.choices[0].message.parsed
        return output_type
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
    except Exception as e:
        print(f"Error in finalize_output: {str(e)} on line {e.__traceback__.tb_lineno}")
        return f"Error in finalize_output: {str(e)} on line {e.__traceback__.tb_lineno}"


def finalize_output(final_decision: str, output_type: OutputType):
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
                error_result = ErrorResult(
                    success=False,
                    error_type="ValidationError",
                    message=f"Function '{tool_function_name}' not found.",
                    context={"tool_function_name": tool_function_name}
                )
                return str(error_result)
        else:
            return message
    except Exception as e:
        print(f"Error in finalize_output: {str(e)} on line {e.__traceback__.tb_lineno}")
        error_result = ErrorResult(
            success=False,
            error_type="LLMCallError",
            message=f"Error in finalize_output: {str(e)} on line {e.__traceback__.tb_lineno}",
            context={"exception": str(e), "line": e.__traceback__.tb_lineno}
        )
        return str(error_result)


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
        response = client.beta.chat.completions.parse(
            model="gpt-4o" if version_counter <= 1 else "gpt-4o-mini",
            messages=messages,  # Pass the messages directly as a list of dicts
            response_format=FinalOutput,
        )
        # Extract the corrected final output from the response
        corrected_final_output = response.choices[0].message.parsed
        return corrected_final_output
    except Exception as e:
        return f"Error: {str(e)}"


def analyze_final_output(
    final_output,
    lead_personality,
    assistants_list,
    original_prompt,
    problem_statement,
    output_type: OutputType,
    current_version=1,
):
    """
    Analyzes the final output to determine if the output aligns with the original prompt and the defined problem statement, whether it addresses the user's needs, and if it is in the expected output format based on the problem statement and original prompt (ie, if the user expects a script, it should be a script, if the user expects a report, it should be a report, etc.). The analysis should also consider the relevance, accuracy, and completeness of the final output.
    This function will loop through the final_output, revising as needed and re-analyzing until the output is satisfactory, then return the final output.

    Args:
        final_output (str): The final output generated by the assistant.
        lead_personality (str): The lead personality in the conversation.
        assistants_list (list): The list of assistant personalities in the conversation.

    Returns:
        final_output (str): The revised final output that meets the criteria.

    """    # for type_of_output, find best match from the output_type_samples using regex
    o_types = list(output_type_samples.keys())
    best_match = None
    best_score = 0
    for o_type in o_types:
        score = jarowinkler_similarity(output_type.output_type, o_type)
        if score > best_score:
            best_score = score
            best_match = o_type
    
    # Handle case where no matches are found (all scores are 0) or similarity is too low
    if best_match is None or best_score < MINIMUM_SIMILARITY_THRESHOLD:
        print(f"Warning: No good matches found for output type '{output_type.output_type}' (best score: {best_score})")
        print(f"Available output types: {list(o_types)}")
        
        # Validate that our fallback exists in the samples
        if DEFAULT_FALLBACK_OUTPUT_TYPE not in output_type_samples:
            raise ValueError(f"Default fallback output type '{DEFAULT_FALLBACK_OUTPUT_TYPE}' not found in output_type_samples")
        
        best_match = DEFAULT_FALLBACK_OUTPUT_TYPE
        print(f"Falling back to: {best_match}")
    else:
        print(f"Found match for output type '{output_type.output_type}': {best_match} (score: {best_score:.3f})")
    
    output_type_short = output_type_samples[best_match]
    sample_original_prompt = output_type_short["sample_original_prompt"]
    sample_problem_statement = output_type_short["sample_problem_statement"]
    sample_output_analysis = output_type_short["sample_output_analysis"]
    sample_final_output = output_type_short["sample_final_output"]
    sample_corrected_output = output_type_short["sample_corrected_output"]

    first_analysis = True

    final_output_analysis_object = None
    version_counter = 1 if first_analysis else current_version
    output_object = FinalOutput(
        final_output=final_output, output_type=output_type, version=version_counter
    )
    while True:
        # Prepare the messages for the assistant
        messages = [
            {
                "role": "system",
                "content": "You are an assistant that analyzes the final output to ensure it aligns with the original prompt and the defined problem statement, addresses the user's needs, and is in the expected output format.",
            },
            {
                "role": "user",
                "content": "Please analyze the final output to ensure it aligns with the original prompt and the defined problem statement, addresses the user's needs, and is in the expected output format based on the problem statement and original prompt. The final output should be relevant, accurate, and complete. Your output will include: the user's intent as a string, the expected output type as a string, a list of the requirements, a boolean indicating whether it matches the expected output type, a boolean indicating whether it aligns with the user's intent, a list of strings describing any changes needed to align, and a boolean indicating whether it is complete. Rules: Only mark requirements as met if they are fully addressed in the output. If the output is incomplete, provide specific suggestions for completion. Only mark the output as complete if all requirements are met. Only mark the output as matching the expected output type if it is completely in the expected format, not just partially. If the output is not in the expected format, provide suggestions for changes. Only mark the output as aligning with the user's intent if it fully addresses the user's needs. If the output does not align, provide specific suggestions for alignment. If the output is not relevant, accurate, or complete, provide specific suggestions for improvement. Only mark the output as complete if all requirements are met and if no changes are needed to align with the user's intent or the expected output type.",
            },
            {
                "role": "user",
                "content": f"Here's the sample final output to analyze: {sample_final_output} to ensure it aligns with the original prompt: {sample_original_prompt} and the defined problem statement: {sample_problem_statement}. The expected output type is: {output_type.output_type} with file extension: {output_type.file_extension}.",
            },
            {"role": "assistant", "content": f"{sample_output_analysis}"},
            {
                "role": "user",
                "content": f"Great work, thanks! Now for the real one: Final Output:\n{output_object.final_output}, Original Prompt: {original_prompt}, Problem Statement: {problem_statement}",
            },
        ]
        # Make the API call to analyze the final output using structured outputs with beta.chat.completions.parse
        try:
            model_ = "gpt-4o" if first_analysis else "gpt-4o-mini"
            response = client.beta.chat.completions.parse(
                model=model_,  # Specify the model
                messages=messages,  # Pass the messages directly as a list of dicts
                response_format=FinalOutputAnalysis,
            )
            # Extract the final output from the response
            final_output_analysis_object = response.choices[0].message.parsed
            print(
                f"\nFinal Output: {final_output}\n, version_counter: {version_counter}\n"
            )
        except Exception as e:
            print(
                f"Error in analyze_final_output: {str(e)} on line {e.__traceback__.tb_lineno} in version {version_counter}"
            )

        # Check if the final output is complete and meets the requirements
        if (
            final_output_analysis_object.is_complete
            and all(req.met for req in final_output_analysis_object.requirements)
            and final_output_analysis_object.matches_expected_output_type
            and final_output_analysis_object.aligns_with_user_intent
        ):
            return output_object.final_output
        else:
            # If the output is not complete or does not meet the requirements, revise the output
            if first_analysis:
                first_analysis = False
                output_object = FinalOutput(
                    final_output=sample_corrected_output,
                    output_type=output_type,
                    version=version_counter,
                )
            else:
                output_object = FinalOutput(
                    final_output=final_output_analysis_object.final_output,
                    output_type=output_type,
                    version=version_counter,
                )

            output_object = final_output_correction(
                output_object.final_output,
                output_object.output_type,
                final_output_analysis_object,
                sample_final_output,
                sample_original_prompt,
                sample_problem_statement,
                sample_output_analysis,
                sample_corrected_output,
                version_counter,
            )
            version_counter += 1
            output_object.version = version_counter
            final_output = output_object.final_output
            print(
                f"\nCorrected Final Output: {final_output}\n, version_counter: {version_counter}\n"
            )
            if version_counter > 5:
                print("Max versions reached. Returning the final output.")
                return final_output
            else:
                # check if the output is complete and meets the requirements by re-analyzing recursively
                print(
                    f"\nRe-analyzing the corrected final output: {final_output}\n, version_counter: {version_counter}\n"
                )
                return analyze_final_output(
                    final_output,
                    lead_personality,
                    assistants_list,
                    original_prompt,
                    problem_statement,
                    output_type,
                    version_counter,
                )


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
    to_do_list = ", ".join(" ".join(task) for task in conversation_memory.to_do_list)
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
            "content": f"Great work, thanks! Now for the real one: To-Do List:\n{', '.join(to_do_list)}, and the conversation history: {', '.join([safe_get_message_field(msg, 'name', 'Unknown')+': '+ safe_get_message_field(msg, 'content', 'No content') for msg in conversation_history if validate_message_structure(msg)])}",
        },
    ]
    # Make the API call to analyze the to-do list using structured outputs with beta.chat.completions.parse
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Specify the model
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
        messages.append(
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

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Specify the model
            messages=messages,  # Pass the messages directly as a list of dicts
            max_completion_tokens=token_limit,
        )
        # Extract the text from the response
        assistant_reply = response.choices[
            0
        ].message.content.strip()  # Use 'message' key, not 'text'
        # print(f"{assistant_name}'s reply: \n", assistant_reply)
    except Exception as e:
        assistant_reply = f"Error: {str(e)}"

    return {"role": "assistant", "name": assistant_name, "content": assistant_reply}


class MediatorReviseOrDecide(BaseModel):
    final_decision: str = Field(
        ..., description="The final decision made by the mediator."
    )
    keep_output: bool = Field(
        ..., description="Whether to keep the current output or revise it."
    )


def run_conversation(
    problem_statement, selected_personalities, lead_personality, num_rounds=3
):
    original_prompt = problem_statement[0]["content"]
    original_prompt = (
        "The Primary User has provided the following prompt, from which the problem statement will be defined: "
        + original_prompt
    )
    print("problem statement", original_prompt, "\n")    # primary_user = problem_statement[0]['name']
    # TODO: FIXME - Commented-out code for conversation history and final content generation
    # The following code was commented out and needs review for potential restoration or removal
    # Have the Mediator finalize the output based on the final decision, creating the final output content that fulfills the problem statement and the user's needs that will be saved to the final file
    # conversation_history = [{"role": "user","round":0, "content": original_prompt}]

    # TODO: Review and potentially restore final content prompt logic
    # final_content_prompt = "Based on the final decision or solution provided by the assistant who called for the final vote, please generate the final output content that fulfills the problem statement and the user's needs. This prompt should be clear, actionable, and concise, providing the finalized details needed to generate the final output content that fulfills the problem statement and the user's needs. It should be in the form of an implementation plan, a request, or a directive that guides the generation of the final output. Ensure that the prompt will result in the direct creation of the final output content that fulfills the problem statement and the user's needs and will not require further discussion or clarification. Please provide the final prompt based on the information available."
    # final_content_messages = [
    #     {"role": "system", "content": final_content_prompt + f" The output type for the final decision is: {output_type.output_type} with file extension: {output_type.file_extension} and should solve the problem statement: {problem_definition} by following the final decision or solution provided by {final_vote_caller}."},
    #     {"role": "user", "content": f"Please generate the final output content based on the final decision or solution provided by {final_vote_caller}:\n\n{safe_get_message_field(final_decision_response, 'content', 'No decision content')}. The problem statement it must completely solve is: {problem_definition}. The output you generate should be the final completion of the problem statement and the user's needs, and should not include any further steps or discussions but will represent the final output content that fulfills the problem statement and the user's needs ONLY.",
    #     },
    # ]
    # for role, content in final_content_messages:
    #     conversation_history.append({"role": role, "name": "Mediator", "round": 0, "content": content})

    # final_output_response = get_assistant_response("Mediator", assistant_personalities["Mediator"] + " " + final_content_prompt, conversation_history)

    # test = test_finalize_output(final_output_response, "py")
    # print(test)
    # exit()
    problem_definition = define_problem(
        original_prompt,
        {
            assistant_name: assistant_prompt
            for assistant_name, assistant_prompt in assistant_personalities.items()
            if assistant_name in selected_personalities
        },
    )
    conversation_history = [
        {
            "role": "system",
            "name": "system",
            "round": 0,
            "content": assistant_instructions,
        },
        {
            "role": "user",
            "name": "Primary User",
            "round": 0,
            "content": problem_definition,
        },
    ]
    # print(conversation_history)
    conversation_memory = ConversationMemory(num_rounds)
    mediator_name = "Mediator"
    questions_asked = []
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
                conversation_history = [
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
                assistant_personalities[mediator_name]
                + "You are tasked with reviewing the output types suggested by the assistants and making the final decision on the output type for the solution. Please consider the problem statement and the context of the discussion to determine the most appropriate output format. Make your decision based on the following options: simple concise text answer, a detailed report in text or PDF format, a code snippet or script file, structured data in JSON or CSV format, a website or app prototype, or a detailed technical document. You may be more specific if needed or choose a different output type, but ensure it aligns with the problem statement and the discussion so far and meets the user's requirements and expectations. Please provide your decision in the format specified, including both the output type and the file extension."
            )
            try:
                # TODO: Make this votable by the assistants
                response = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",  # Specify the model
                    messages=[
                        {"role": "system", "content": mediator_prompt},
                        {
                            "role": "user",
                            "content": f"Please decide on the output type based on the defined problem statement:\n\n{problem_definition}. The suggested output types are: {', '.join(output_types.output_type for output_types in output_types)}. Possible file extensions are: {', '.join(output_types.file_extension for output_types in output_types)}. Please provide your recommendation in the provided format, generating both the specific output type and the file extension.",
                        },
                    ],
                    response_format=OutputType,
                )
                # Extract the output type from the response
                output_type = response.choices[0].message.parsed
                print(f"\nMediator Output Type Decision: {output_type}\n")            except Exception as e:
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
            assistant_prompt = assistant_personalities[assistant_name]
            mediator_prompt = assistant_personalities[mediator_name]
            mediator_initial_response = f"You are the Mediator for this conversation. This is the first round of {num_rounds} to produce the final output. Here's the problem statement: {problem_definition}. Using simple, concise language, guide the discussion towards a final output in the expected format by the end of the final round (round {num_rounds}). This first round is about making a plan or initial decision on how to approach the problem and what steps to take in the next {num_rounds - 1} rounds."

            if rnd == 0:
                mediator_prompt = (
                    assistant_personalities[mediator_name]
                    + assistant_instructions
                    + f"This is the first round of {num_rounds} to produce the final output. Here's the problem statement: {problem_definition}. Your task is to guide the discussion towards a final output in the expected format by the end of the final round (round {num_rounds}). You should guide this first round towards making a plan or initial decision on how to approach the problem and what steps to take in the next {num_rounds - 1} rounds."
                )
                mediator_initial_response = f"You are the Mediator for this conversation. This is the first round of {num_rounds} to produce the final output. Here's the problem statement: {problem_definition}. Using simple, concise language, guide the discussion towards a final output in the expected format by the end of the final round (round {num_rounds}). This first round is about making a plan or initial decision on how to approach the problem and what steps to take in the next {num_rounds - 1} rounds."
            elif rnd > 0 and rnd + 1 < num_rounds:
                mediator_prompt = (
                    assistant_personalities[mediator_name]
                    + assistant_instructions
                    + f"There are {num_rounds - rnd} rounds left to finalize the output. This is the beginning of round {rnd + 1}, please guide the discussion towards a final output in the expected format by the end of the final round (round {num_rounds})."
                )
                mediator_initial_response = (
                    f"Round {rnd} Summary: {last_round_summary}, the original problem statement was: {problem_definition}. Based on the conversation so far, please guide the discussion into the next round.",
                )
            elif rnd + 1 == num_rounds:
                mediator_prompt = (
                    assistant_personalities[mediator_name]
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
            # Convert `ConversationInput` to a dict if needed
            if isinstance(assistant_response["content"], ConversationInput):
                assistant_response["content"] = assistant_response["content"].content
            # Ensure content is always extracted as a string
            conversation_history.append(
                {
                    "role": (
                        assistant_response["role"]
                        if isinstance(assistant_response["role"], str)
                        else str(assistant_response["role"])
                    ),
                    "name": assistant_name,
                    "round": rnd,
                    "content": (
                        assistant_response["content"]
                        if isinstance(assistant_response["content"], str)
                        else str(assistant_response["content"])
                    ),
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

        conversation_memory = analyze_to_do_list(
            conversation_memory,
            [msg for msg in conversation_history if msg["round"] == rnd],
        )
        print(
            f"\nMemory Summary in Round {rnd}: {conversation_memory.get_memory_summary()}\n"
        )

        # After each round, have the Mediator summarize
        mediator_prompt = assistant_personalities[mediator_name]
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

        # Convert mediator response if needed
        if isinstance(mediator_response["content"], ConversationInput):
            mediator_response["content"] = mediator_response["content"].content
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
                    responder_prompt = assistant_personalities[responder_name]
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

                    # Convert responder response if needed
                    if isinstance(responder_response["content"], ConversationInput):
                        responder_response["content"] = responder_response[
                            "content"
                        ].content

                    conversation_history.append(
                        {
                            "role": "assistant",
                            "name": q["asking_assistant"],
                            "round": rnd,
                            "content": q["reply"],
                        }
                    )
                    print(
                        f"\n Name: {q['asking_assistant']} is directing a question to {responder_name}. Question: {q['reply']} \n Role: {safe_get_message_field(responder_response, 'role', 'Unknown')} \n Response from {responder_name}: {safe_get_message_field(responder_response, 'content', 'No content')} \n \n"
                    )

                    conversation_history.append(
                        {
                            "role": responder_response["role"],
                            "name": responder_response["name"],
                            "round": rnd,
                            "content": f"In response to {q['asking_assistant']}: {safe_get_message_field(responder_response, 'content', 'No content')}",
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
        conversation_memory = analyze_to_do_list(
            conversation_memory,
            [msg for msg in conversation_history if msg["round"] == rnd],
        )
        print(
            f"\n\nMemory Summary in Round {rnd}: {conversation_memory.get_memory_summary()}\n\n"
        )
        # decrement the number of rounds in ConversationMemory
        conversation_memory.decrement_rounds_left()
        questions_asked = []
        # Vote on whether to continue the conversation into the next round or end it based on whether this round has reached a conclusion or contains enough information to vote on a final decision/solution
        if rnd <= num_rounds - 1:
            print(
                f"\nRound {rnd + 1} Summary: {summarize_conversation([msg for msg in conversation_history if msg['round'] == rnd])}\n"
            )
            call_for_final_vote = False            if rnd < num_rounds - 1:
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
                        # Default to continue on error to avoid premature termination
                        continue_votes += 1
                
                total_continue_votes = continue_votes + end_votes + voting_errors
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
                        "content": f"{assistant_personalities[final_vote_caller]} The conversation has reached a point where a final decision or solution is needed. Please provide your final decision or solution based on the information available. It should be clear, actionable, and concise, providing the finalized details needed to generate the final output content that fulfills the problem statement and the user's needs. Please do not include any further steps or discussions, only the final decision or solution.",
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
                for role, content in final_decision_messages:
                    conversation_history.append(
                        {
                            "role": role,
                            "name": "Mediator",
                            "round": rnd,
                            "content": content,
                        }
                    )
                final_decision_response = get_assistant_response(
                    final_vote_caller,
                    assistant_personalities[final_vote_caller]
                    + " "
                    + final_decision_prompt,
                    conversation_history,
                    None,
                    conversation_memory,
                )
                print(
                    f"\nFinal Decision Response from {final_vote_caller}: {final_decision_response}\n (Unconfirmed)"
                )
                # Cast a final vote to confirm the decision or solution
                final_vote_prompt = "Please cast a binary vote to confirm the final decision or solution provided by the assistant who called for the final vote. Respond with a boolean value (True/False) only, with True to confirm the decision or solution and False to reject it."
                yes_votes = 0
                no_votes = 0
                no_voters = []
                summary = summarize_conversation(conversation_history)

                for assistant_name, _ in assistant_priorities:
                    final_vote_messages = [
                        {
                            "role": "system",
                            "content": f"{assistant_personalities[assistant_name]} An assistant has recommended a solution for the problem statement {problem_definition}. Please review the conversation summary, then review the final decision or solution provided by {final_vote_caller} and cast your vote to confirm it.",
                        },
                        {
                            "role": "user",
                            "content": f"Here is a summary of the conversation so far:\n\n{summary}",
                        },                        {
                            "role": "user",
                            "content": f"Please cast your vote to confirm the final decision or solution provided by {final_vote_caller}:\n\n{safe_get_message_field(final_decision_response, 'content', 'No decision content')}",
                        },
                        {
                            "role": "system",
                            "content": f"Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')}",
                        },
                    ]                    final_binary_vote = cast_binary_vote(
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
                        no_votes += 1  # Count errors as no votes for safety
                        no_voters.append(assistant_name)
                
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
                        f"{assistant_personalities[mediator_name]} "
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
                        mediator_emergency_response = client.beta.chat.completions.parse(
                            model="gpt-4o-mini",
                            messages=mediator_fallback_messages,
                            response_format=MediatorReviseOrDecide,
                        )
                        mediator_decision = mediator_emergency_response.choices[0].message.parsed
                        print(f"🏛️ Emergency Mediator Decision: {mediator_decision}")
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
                        f"{assistant_personalities[mediator_name]} "
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
                        tie_breaker_response = client.beta.chat.completions.parse(
                            model="gpt-4o-mini",
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
                        print("✓ Proceeding despite low participation due to clear majority")
                    else:
                        print("❌ Insufficient participation and unclear mandate")
                        print("🏛️ Requiring mediator approval due to low participation")
                        # Force mediator review for low participation
                        participation_prompt = (
                            f"{assistant_personalities[mediator_name]} "
                            f"Low participation in final vote: only {total_votes}/{total_assistants} assistants voted. "
                            "As mediator, determine if this is sufficient to proceed or if the decision needs revision."
                        )
                        participation_messages = [
                            {"role": "system", "content": participation_prompt},
                            {"role": "user", "content": f"Vote Results: {yes_votes} YES, {no_votes} NO out of {total_assistants} total assistants"},
                            {"role": "user", "content": f"Proposed Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')}"}
                        ]
                        try:
                            participation_response = client.beta.chat.completions.parse(
                                model="gpt-4o-mini",
                                messages=participation_messages,
                                response_format=MediatorReviseOrDecide,
                            )
                            mediator_decision = participation_response.choices[0].message.parsed
                            print(f"🏛️ Low Participation Decision: {'PROCEED' if mediator_decision.keep_output else 'REVISE'}")
                        except Exception as e:
                            print(f"❌ Low participation decision failed: {e}")
                            mediator_decision = MediatorReviseOrDecide(
                                final_decision="Low participation fallback: proceeding with caution.",
                                keep_output=True
                            )

                # Normal voting logic (only executed if no edge cases above)
                if total_votes > 0 and yes_votes != no_votes and total_votes >= (total_assistants / 2):
                    if yes_votes > no_votes and no_votes == 0:print(
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
                    for role, content in final_content_messages:
                        conversation_history.append(
                            {
                                "role": role,
                                "name": "Mediator",
                                "round": rnd,
                                "content": content,
                            }
                        )

                    final_output_response = get_assistant_response(
                        lead_personality,
                        assistant_personalities[lead_personality]
                        + " "
                        + final_content_prompt,
                        conversation_history,
                        None,
                        conversation_memory,
                    )
                    print(
                        f"\n\nFinal Output Response from {lead_personality}: {final_output_response}\n\n"
                    )
                    final_checked_output = analyze_final_output(
                        final_output_response["content"],
                        lead_personality,
                        selected_personalities,
                        original_prompt,
                        problem_definition,
                        output_type,
                    )
                    print(f"\n\nFinal Checked Output: {final_checked_output}\n\n")

                    final_output = finalize_output(final_checked_output, output_type)
                    print(f"\n\nFinal Output filepath: {final_output}\n\n")
                    # Add the final output to the conversation history and then save the conversation history to a file
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": lead_personality,
                            "round": rnd,
                            "content": f"Final Output: {final_output}",                        }
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
                elif (
                    yes_votes > no_votes
                    and no_votes > 0
                    and no_votes <= (yes_votes / 2)                ):
                    print(
                        f"\nFinal Decision Confirmed with majority vote, {yes_votes} to {no_votes}: {safe_get_message_field(final_decision_response, 'content', 'No decision content')}\n"
                    )
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": mediator_name,
                            "round": rnd,
                            "content": f"Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')} (Confirmed with majority vote)",
                        }
                    )
                    # Get feedback from the assistants who voted no and ask the Mediator to resolve the issue by revising the final decision or making a final call
                    feedbacks = []
                    for assistant_name, _ in assistant_priorities:
                        if assistant_name in no_voters:
                            feedback_messages = [
                                {
                                    "role": "system",
                                    "content": f"{assistant_personalities[assistant_name]} Your vote did not confirm the final decision or solution provided by {final_vote_caller}. Please provide feedback on the final decision or solution.",
                                },
                                {
                                    "role": "user",
                                    "content": f"Please provide feedback on the final decision or solution provided by {final_vote_caller} and suggest improvements or changes to align it better with the problem statement and the conversation so far.",
                                },                                {
                                    "role": "user",
                                    "content": f"Final Decision: {safe_get_message_field(final_decision_response, 'content', 'No decision content')}",
                                },
                            ]
                            feedback = get_assistant_response(
                                assistant_name, assistant_prompt, feedback_messages
                            )
                            feedbacks.append(feedback)
                    # Have the Mediator review the feedback and make a final decision
                    mediator_feedback = []
                    for feedback in feedbacks:
                        mediator_feedback.append(safe_get_feedback_field(feedback, 'content', 'No feedback content'))
                    mediator_prompt = (
                        assistant_personalities[mediator_name]
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
                        response = client.beta.chat.completions.parse(
                            model="gpt-4o-mini",  # Specify the model
                            messages=mediator_messages,  # Pass the messages directly as a list of dicts
                            response_format=MediatorReviseOrDecide,
                        )
                        # Extract the decision from the response
                        mediator_decision = response.choices[0].message.parsed
                        print(f"\nMediator Decision: {mediator_decision}\n")                    except Exception as e:
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
                    if mediator_decision.keep_output:                        conversation_history.append(
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
                    for role, content in final_content_messages:
                        conversation_history.append(
                            {
                                "role": role,
                                "name": "Mediator",
                                "round": rnd,
                                "content": content,
                            }
                        )

                    final_output_response = get_assistant_response(
                        lead_personality,
                        assistant_personalities[lead_personality]
                        + " "
                        + final_content_prompt,
                        conversation_history,
                        None,
                        conversation_memory,
                    )
                    print(
                        f"\n\nFinal Output Response from {lead_personality}: {final_output_response}\n\n"
                    )
                    final_checked_output = analyze_final_output(
                        final_output_response["content"],
                        lead_personality,
                        selected_personalities,
                        original_prompt,
                        problem_definition,
                        output_type,
                    )
                    print(f"\n\nFinal Checked Output: {final_checked_output}\n\n")

                    final_output = finalize_output(final_checked_output, output_type)
                    print(f"\n\nFinal Output filepath: {final_output}\n\n")
                    # Add the final output to the conversation history and then save the conversation history to a file
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": lead_personality,
                            "round": rnd,
                            "content": f"Final Output: {final_output}",                        }
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
                elif not mediator_decision.keep_output:
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
                    for role, content in final_content_messages:
                        conversation_history.append(
                            {
                                "role": role,
                                "name": mediator_name,
                                "round": rnd,
                                "content": content,
                            }
                        )

                    final_output_response = get_assistant_response(
                        lead_personality,
                        assistant_personalities[lead_personality]
                        + " "
                        + final_content_prompt,
                        conversation_history,
                        None,
                        conversation_memory,
                    )
                    print(
                        f"\n\nFinal Output Response from {lead_personality}: {final_output_response}\n\n"
                    )
                    final_checked_output = analyze_final_output(
                        final_output_response["content"],
                        lead_personality,
                        selected_personalities,
                        original_prompt,
                        problem_definition,
                        output_type,
                    )
                    print(f"\n\nFinal Checked Output: {final_checked_output}\n\n")

                    final_output = finalize_output(
                        final_checked_output, output_type
                    )
                    print(f"\n\nFinal Output filepath: {final_output}\n\n")
                    # Add the final output to the conversation history and then save the conversation history to a file
                    conversation_history.append(
                        {
                            "role": "system",
                            "name": lead_personality,
                            "round": rnd,
                            "content": f"Final Output: {final_output}",                        }
                    )
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
    return (conversation_history, questions_asked, final_output)

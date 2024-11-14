import re
import strawberry
from typing import List, Optional

# Define the Conversation input type
@strawberry.input
class ConversationInput:
    role: str
    name: Optional[str]
    content: str

# Define the Feedback input type
@strawberry.input
class FeedbackInput:
    assistant: str
    rating: int

# Define the Question input type
@strawberry.input
class QuestionInput:
    assistant: str
    question: str

# Define the Question output type
@strawberry.type
class Question:
    assistant: str
    question: str

# Define the Conversation output type (this stays as a type, not input)
@strawberry.type
class Conversation:
    role: str
    name: Optional[str]
    content: str

# Define the Feedback output type
@strawberry.type
class Feedback:
    assistant: str
    rating: int

# Define the response types for running a conversation
@strawberry.type
class RunConversationResponse:
    conversation: List[Conversation]
    questions: Optional[List[Question]]
    final_output : str

# Define the response type for submitting feedback
@strawberry.type
class FeedbackResponse:
    success: str

# Define the root Query class
@strawberry.type
class Query:
    @strawberry.field
    def hello(self) -> str:
        return "Hello, World!"

# Define the root Mutation class
@strawberry.type
class Mutation:
    @strawberry.mutation
    def run_conversation(self, conversation: List[ConversationInput], assistant_personalities: List[str], leadPersonality: str, num_rounds: int) -> RunConversationResponse:
        from conversation_manager import run_conversation
        for c in conversation:
            print(c.content, "", type(c.content))
        conversation_dicts = [
            {"role": c.role, "name": c.name, "content": c.content}
            for c in conversation
        ]

        # Here conversation should already be a list of dictionaries.
        print("in schema: ", conversation_dicts, "type:", type(conversation_dicts))  # Debug this to ensure you are receiving a proper list
        
        # You should pass the structured data directly into run_conversation.
        result = run_conversation(conversation_dicts, assistant_personalities,leadPersonality, num_rounds)

        conversation_history, questions_asked, final_output = result
        print(conversation_history)
        print(result)
        # Convert conversation history from dicts to Conversation objects
        conversation_objs = [
            Conversation(role=entry['role'], name=entry.get('name', ''), content=entry['content'])
            for entry in conversation_history
        ]

        return RunConversationResponse(
            conversation=conversation_objs,
            questions=questions_asked,
            final_output = final_output
        )


    @strawberry.mutation
    def submit_feedback(self, conversation: List[ConversationInput], feedback: List[FeedbackInput], questions: Optional[List[QuestionInput]] = None) -> FeedbackResponse:
        from feedback_manager import log_conversation

        # Log the feedback (you might also need to convert conversation to dicts)
        conversation_dicts = [
            {"role": c.role, "name": c.name, "content": c.content}
            for c in conversation
        ]

        log_conversation(conversation_dicts, feedback, questions)
        return FeedbackResponse(success="Feedback submitted successfully.")

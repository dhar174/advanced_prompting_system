
from pydantic import BaseModel, Field
from typing import Any, Dict

class GenerateSimpleConciseAnswerMessage(BaseModel):
    final_decision: str = Field(..., description="The final decision or solution to be included in the message.")
    filename: str = Field(..., description="The name of the output file.")


class GenerateJSONOutput(BaseModel):
    final_decision: str = Field(..., description="The final decision or solution to be formatted as JSON.")
    filename: str = Field(..., description="The name of the output JSON file.")

class GeneratePDFOutput(BaseModel):
    final_decision: str = Field(..., description="The final decision or solution to be included in the PDF.")
    filename: str = Field(..., description="The name of the output PDF file.")

class GenerateTextOutput(BaseModel):
    final_decision: str = Field(..., description="The final decision or solution to be written to a text file.")
    filename: str = Field(..., description="The name of the output text file.")

class GenerateHTMLOutput(BaseModel):
    final_decision: str = Field(..., description="The final decision or solution to be formatted and displayed as an HTML page.")
    filename: str = Field(..., description="The name of the output HTML file.")

class GeneratePythonScriptOutput(BaseModel):
    final_decision: str = Field(..., description="The final decision or solution to be included in a complete Python script.")
    filename: str = Field(..., description="The name of the output Python file.")

class GenerateCodeSnippetOutput(BaseModel):
    final_decision: str = Field(..., description="The final decision or solution to be included in a modular code snippet.")
    filename: str = Field(..., description="The name of the output code snippet file.")

class GenerateCSVOutput(BaseModel):
    final_decision: str = Field(..., description="The final decision or solution to be formatted as a CSV file.")
    filename: str = Field(..., description="The name of the output CSV file.")



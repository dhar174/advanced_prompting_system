# output_generator.py

import json
import os
import re
from fpdf import FPDF
from typing import Any, Dict

# output_type_mapping = {
#     "simple concise text answer": "generate_text_answer",
#     "detailed report": "generate_detailed_report",
#     "code snippet": "generate_code_snippet",
#     "script file": "generate_code_snippet",
#     "structured data": "generate_structured_data",
#     "JSON format": "generate_structured_data",
#     "CSV format": "generate_structured_data",
#     "detailed technical document": "generate_technical_document"
# }


def generate_simple_concise_answer_message(
    final_decision: str, filename: str = "final_output.txt"
) -> str:
    """
    Generates a simple concise text answer message.

    Args:
        final_decision (str): The final_decision to be included in the message.

    Returns:
        str: The simple concise text answer message.
    """
    return final_decision


def generate_json_output(
    final_decision: str, filename: str = "final_output.json"
) -> str:
    """
    Generates a JSON file from the provided final_decision.

    Args:
        final_decision (str): The final_decision to be serialized into JSON.
        filename (str): The name of the output JSON file.

    Returns:
        str: The path to the generated JSON file.
    """
    try:
        with open(filename, "w") as f:
            json.dump(final_decision, f, indent=4)
        return os.path.abspath(filename)
    except (IOError, OSError, PermissionError) as e:
        print(f"❌ Error writing JSON file '{filename}': {e}")
        raise


def generate_pdf_output(final_decision: str, filename: str = "final_output.pdf") -> str:
    """
    Generates a PDF file from the provided final_decision.

    Args:
        final_decision (str): The final_decision to be included in the PDF.
        filename (str): The name of the output PDF file.

    Returns:
        str: The path to the generated PDF file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=final_decision, ln=True, align="L")

    pdf.output(filename)
    return os.path.abspath(filename)


def generate_text_file_output(
    final_decision: str, filename: str = "final_output.txt"
) -> str:
    """
    Generates a plain text file from the provided final_decision.

    Args:
        final_decision (str): The final_decision to be written into the text file.
        filename (str): The name of the output text file.

    Returns:
        str: The path to the generated text file.
    """
    try:
        with open(filename, "w") as f:
            f.write(final_decision)
        return os.path.abspath(filename)
    except (IOError, OSError, PermissionError) as e:
        print(f"❌ Error writing text file '{filename}': {e}")
        raise


def generate_html_output(
    final_decision: str, filename: str = "final_output.html"
) -> str:
    """
    Generates an HTML file from the provided final_decision.

    Args:
        final_decision (str): The final_decision to be included in the HTML.
        filename (str): The name of the output HTML file.

    Returns:
        str: The path to the generated HTML file.
    """
    html_content = "<html><body>"
    for key, value in final_decision.items():
        html_content += f"<h2>{key}</h2><p>{value}</p>"
    html_content += "</body></html>"

    try:
        with open(filename, "w") as f:
            f.write(html_content)
        return os.path.abspath(filename)
    except (IOError, OSError, PermissionError) as e:
        print(f"❌ Error writing HTML file '{filename}': {e}")
        raise


def generate_python_script(
    final_decision: str, filename: str = "final_output.py"
) -> str:
    """
    Generates a Python script file from the provided final_decision.

    Args:
        final_decision (str): The final_decision to be included in the Python script.
        filename (str): The name of the output Python script file.

    Returns:
        str: The path to the generated Python script file.
    """
    try:
        with open(filename, "w") as f:
            # Remove everything before the first set of ``` and after the last set of ``` using regex
            final_decision = re.sub(
                r"^[\s\S]*?```([\s\S]*?)```[\s\S]*$", r"\1", final_decision
            )
            # If the word "python" is at the beginning of the code snippet, remove it
            final_decision = re.sub(r"^python", "", final_decision)
            f.write(final_decision)
        return os.path.abspath(filename)
    except (IOError, OSError, PermissionError) as e:
        print(f"❌ Error writing Python script file '{filename}': {e}")
        raise


def generate_code_snippet(
    final_decision: str, filename: str = "final_output.py"
) -> str:
    """
    Generates a code snippet file from the provided final_decision.

    Args:
        final_decision (str): The final_decision to be included in the code snippet.
        filename (str): The name of the output code snippet file.

    Returns:
        str: The path to the generated code snippet file.
    """
    try:
        with open(filename, "w") as f:
            f.write(final_decision)
        return os.path.abspath(filename)
    except (IOError, OSError, PermissionError) as e:
        print(f"❌ Error writing code snippet file '{filename}': {e}")
        raise


def generate_csv_output(final_decision: str, filename: str = "final_output.csv") -> str:
    """
    Generates a CSV file from the provided final_decision.

    Args:
        final_decision (str): The final_decision to be included in the CSV.
        filename (str): The name of the output CSV file.

    Returns:
        str: The path to the generated CSV file.
    """
    try:
        with open(filename, "w") as f:
            for key, value in final_decision.items():
                f.write(f"{key},{value}\n")
        return os.path.abspath(filename)
    except (IOError, OSError, PermissionError) as e:
        print(f"❌ Error writing CSV file '{filename}': {e}")
        raise

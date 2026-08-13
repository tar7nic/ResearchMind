from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from ingestion.pdf_parser import extract_pdf_content
import os


# --- Tool 1: Tavily Web Search ---

def get_search_tool(max_results: int = 5) -> TavilySearchResults:
    return TavilySearchResults(
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )


# --- Tool 2: Python REPL (Groq-compatible, rewritten) ---

@tool
def python_repl(code: str) -> str:
    """Execute Python code for calculations, data analysis, or processing. Returns the output."""
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {})
        output = sys.stdout.getvalue()
    except Exception as e:
        output = f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout
    return output or "Code executed with no output."


# --- Tool 3: Document Reader ---

@tool
def read_document(pdf_path: str) -> str:
    """Extracts and returns text content from a PDF file at the given path.
    Use this when the user has uploaded a document and you need to read its contents."""
    try:
        content = extract_pdf_content(pdf_path)
        if not content["text"]:
            return "No readable text found in the document."
        return content["text"]
    except FileNotFoundError:
        return f"Error: File not found at path: {pdf_path}"
    except Exception as e:
        return f"Error reading document: {str(e)}"


# --- Tool Registry ---

def get_all_tools(max_search_results: int = 5) -> list:
    return [
        get_search_tool(max_results=max_search_results),
        python_repl,
        read_document,
    ]
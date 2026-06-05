from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.tools import tool
from ingestion.pdf_parser import extract_pdf_content
import os


# --- Tool 1: Tavily Web Search ---

def get_search_tool(max_results: int = 5) -> TavilySearchResults:
    """
    Returns a configured Tavily search tool.
    Requires TAVILY_API_KEY
    """
    return TavilySearchResults(
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )


# --- Tool 2: Python REPL ---

def get_repl_tool() -> PythonREPLTool:
    """
    Returns a sandboxed Python REPL tool.
    Used by the agent for calculations, data processing, etc.
    """
    return PythonREPLTool()


# --- Tool 3: Document Reader ---

@tool
def read_document(pdf_path: str) -> str:
    """
    Extracts and returns text content from a PDF file at the given path.
    Use this when the user has uploaded a document and you need to read its contents.
    """
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
    """
    Returns all tools as a list for binding to the LangGraph agent.
    """
    return [
        get_search_tool(max_results=max_search_results),
        get_repl_tool(),
        read_document,
    ]
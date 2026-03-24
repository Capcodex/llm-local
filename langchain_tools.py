from langchain_core.tools import tool
from datetime import datetime
import os

"""
This file defines two tools for use with LangChain.
- get_current_time: Returns the current date and time.
- create_note: Creates a file with specified content.
"""

@tool
def get_current_time() -> str:
    """Returns the current date and time, formatted as a string."""
    return datetime.now().isoformat()

@tool
def create_note(filename: str, content: str | None = None) -> str:
    """
    Creates a new file with the given filename and writes the content to it.
    Only use this tool when the user explicitly asks to create a file or a note.

    Args:
        filename (str): The name of the file to create (e.g., 'my_note.txt').
        content (str): The content to write into the file.

    Returns:
        str: A confirmation message indicating success or failure.
    """
    try:
        # It's good practice to avoid directory traversal attacks
        if '/' in filename or '..' in filename:
            return "Erreur: nom de fichier invalide. Le traversal de dossier est interdit."

        if content is None or content.strip() == "":
            return "Erreur: le contenu est requis. Indique le texte a ecrire dans le fichier."
            
        with open(filename, 'w') as f:
            f.write(content)
        full_path = os.path.abspath(filename)
        return f"Note creee avec succes ici: '{full_path}'."
    except Exception as e:
        return f"Erreur lors de la creation de la note: {e}"

# Tools that create or modify files should require explicit user confirmation.
TOOLS_REQUIRING_CONFIRMATION = {"create_note"}

# Example of how to see the tools' schemas
if __name__ == '__main__':
    from langchain_core.utils.function_calling import convert_to_openai_tool

    print("Tool Details:")
    print("-------------")
    print(convert_to_openai_tool(get_current_time))
    print(convert_to_openai_tool(create_note))

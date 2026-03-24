from flask import Flask, request, jsonify, g
from flask_cors import CORS # To handle Cross-Origin Resource Sharing
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import StructuredTool
import re
import traceback
import json
from typing import Any
from datetime import datetime
import unicodedata

# Import the tools we created
from langchain_tools import get_current_time, create_note, TOOLS_REQUIRING_CONFIRMATION

# 1. Initialize Flask App
app = Flask(__name__)
CORS(app) # This will allow the React frontend to call the backend

# 2. Load the tools
class ToolConfirmationRequired(Exception):
    def __init__(self, tool_name: str, tool_args: dict):
        super().__init__(f"Confirmation required for tool '{tool_name}'")
        self.tool_name = tool_name
        self.tool_args = tool_args


def is_tool_call_approved(tool_name: str, tool_args: dict) -> bool:
    approvals = getattr(g, "approved_tool_calls", [])
    for approval in approvals:
        if approval.get("name") == tool_name and approval.get("args") == tool_args:
            return True
    return False


def wrap_tool_with_confirmation(tool_obj):
    if tool_obj.name not in TOOLS_REQUIRING_CONFIRMATION:
        return tool_obj

    def _confirmed_tool(**kwargs):
        if getattr(g, "require_tool_confirmation", False) and not is_tool_call_approved(tool_obj.name, kwargs):
            raise ToolConfirmationRequired(tool_obj.name, kwargs)
        return tool_obj.invoke(kwargs)

    return StructuredTool.from_function(
        func=_confirmed_tool,
        name=tool_obj.name,
        description=tool_obj.description,
        args_schema=tool_obj.args_schema,
        return_direct=tool_obj.return_direct,
    )


raw_tools = [get_current_time, create_note]
tools = [wrap_tool_with_confirmation(tool_obj) for tool_obj in raw_tools]
tool_by_name = {tool.name: tool for tool in tools}

# 3. Create the Agent
# I'll use a ReAct agent, which is good for tool use.
# I will pull a default prompt from the LangChain hub.
base_prompt = hub.pull("hwchase17/react")
system_instruction_template = (
    "Tu es un assistant IA conversationnel de type ChatGPT. "
    "Tu dois toujours repondre en francais avec un ton clair, naturel et professionnel. "
    "Outils disponibles: {tool_list}. "
    "Utilise les outils si cela permet d'etre plus precis ou si l'utilisateur le demande. "
    "Si une action modifie des fichiers, confirme d'abord (si exige) puis execute proprement. "
    "Ne fabrique pas de resultats d'outil. "
    "Quand la demande est ambigue, pose une question breve. "
    "Sinon, fournis directement une reponse structuree et utile."
)
DEFAULT_MODEL_NAME = "qwen3:latest"
MAX_CONTEXT_MESSAGES = 12
PLAIN_CHAT_SYSTEM_INSTRUCTION = (
    "Tu es un assistant IA utile, clair et concis. "
    "Reponds toujours en francais. "
    "N'affiche jamais de balises techniques du type Thought/Action/Observation. "
    "Donne directement la reponse finale."
)


def build_prompt(selected_tools):
    tool_list = ", ".join([tool.name for tool in selected_tools]) or "aucun"
    return PromptTemplate(
        input_variables=base_prompt.input_variables,
        template=system_instruction_template + "\n\n" + base_prompt.template,
        partial_variables={
            **getattr(base_prompt, "partial_variables", {}),
            "tool_list": tool_list,
        },
    )


def get_agent_executor(selected_tools, model_name: str):
    llm = Ollama(model=model_name)
    prompt = build_prompt(selected_tools)
    agent = create_react_agent(llm, selected_tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=selected_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )

def sanitize_message_content(content: Any) -> str:
    if not isinstance(content, str):
        return ""
    return content.strip()

def build_agent_input(user_message: str, conversation_history: list[dict]) -> str:
    if not conversation_history:
        return user_message

    normalized_history = []
    for item in conversation_history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = sanitize_message_content(item.get("content"))
        if role not in ("user", "assistant") or not content:
            continue
        normalized_history.append({"role": role, "content": content})

    if not normalized_history:
        return user_message

    trimmed_history = normalized_history[-MAX_CONTEXT_MESSAGES:]
    if trimmed_history and trimmed_history[-1]["role"] == "user" and trimmed_history[-1]["content"] == user_message:
        trimmed_history = trimmed_history[:-1]

    if not trimmed_history:
        return user_message

    history_lines = []
    for message in trimmed_history:
        speaker = "Utilisateur" if message["role"] == "user" else "Assistant"
        history_lines.append(f"{speaker}: {message['content']}")

    history_block = "\n".join(history_lines)
    return (
        "Contexte conversationnel (messages precedents):\n"
        f"{history_block}\n\n"
        "Nouvelle demande utilisateur:\n"
        f"{user_message}\n\n"
        "Donne une reponse concise, utile et coherente avec ce contexte."
    )

def build_plain_chat_input(user_message: str, conversation_history: list[dict]) -> str:
    normalized_history = []
    for item in conversation_history[-MAX_CONTEXT_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = sanitize_message_content(item.get("content"))
        if role in ("user", "assistant") and content:
            normalized_history.append((role, content))

    lines = [PLAIN_CHAT_SYSTEM_INSTRUCTION]
    if normalized_history:
        lines.append("\nContexte recent:")
        for role, content in normalized_history:
            speaker = "Utilisateur" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {content}")
    lines.append("\nQuestion actuelle:")
    lines.append(user_message)
    lines.append("\nReponse finale:")
    return "\n".join(lines)

def get_plain_chat_reply(user_message: str, conversation_history: list[dict], model_name: str) -> str | None:
    prompt = build_plain_chat_input(user_message, conversation_history)
    try:
        plain_llm = Ollama(model=model_name)
        response = plain_llm.invoke(prompt)
        if isinstance(response, str):
            cleaned = response.strip()
            return cleaned if cleaned else None
        cleaned = str(response).strip()
        return cleaned if cleaned else None
    except Exception:
        return None

def extract_final_answer(raw_output: str) -> str | None:
    if not raw_output:
        return None
    match = re.search(r"Final Answer:\s*(.*)", raw_output, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().strip("`").strip()

def extract_raw_output(raw_output: str) -> str | None:
    if not raw_output:
        return None
    match = re.search(
        r"Could not parse LLM output:\s*`([\s\S]*?)`",
        raw_output,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    match = re.search(
        r"Could not parse LLM output:\s*([\s\S]*?)(?:\nFor troubleshooting|\Z)",
        raw_output,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().strip("`").strip()
    return None

def fallback_parsing_reply(raw_output: str) -> str:
    generic_message = (
        "Je n'ai pas pu analyser proprement la reponse du modele. "
        "Reessaie en reformulant la demande en une phrase simple."
    )
    extracted = extract_raw_output(raw_output)
    if extracted:
        if "hypothetical scenario" in extracted.lower():
            return generic_message
        filtered_lines = [
            line
            for line in extracted.splitlines()
            if not re.match(r"^\s*(Thought|Action|Action Input|Observation|Question)\s*:", line, re.IGNORECASE)
            and not re.search(r"output_parsing_failure|troubleshooting", line, re.IGNORECASE)
        ]
        cleaned = "\n".join(filtered_lines).strip()
        if cleaned:
            return cleaned
    final_answer = extract_final_answer(raw_output)
    if final_answer:
        return final_answer
    return generic_message

def extract_action_call(raw_output: str) -> tuple[str, str] | None:
    if not raw_output:
        return None
    pattern_with_input = r"Action:\s*([A-Za-z0-9_-]+)\s*Action Input:\s*(.*?)(?:\n[A-Z][A-Za-z ]+:|$)"
    matches = re.findall(pattern_with_input, raw_output, re.DOTALL)
    if matches:
        tool_name, action_input = matches[-1]
        return tool_name.strip(), action_input.strip().strip("`")

    # Fallback for outputs like "Action: get_current_time()" with no Action Input line.
    pattern_no_input = r"Action\s*:\s*([A-Za-z0-9_-]+)\s*(?:\((.*?)\))?"
    matches = re.findall(pattern_no_input, raw_output, re.DOTALL)
    if matches:
        tool_name, action_input = matches[-1]
        return tool_name.strip(), (action_input or "").strip().strip("`")
    return None

def is_parsing_error_message(raw_output: str) -> bool:
    if not raw_output:
        return False
    lowered = raw_output.lower()
    return (
        "output parsing error occurred" in lowered
        or "output_parsing_failure" in lowered
        or "parsing llm output" in lowered
        or "could not parse llm output" in lowered
    )

def has_action_none(raw_output: str) -> bool:
    if not raw_output:
        return False
    return re.search(r"Action:\s*None\b", raw_output, re.IGNORECASE) is not None

def try_extract_reply_from_error(raw_output: str) -> str | None:
    if not raw_output:
        return None
    if is_parsing_error_message(raw_output):
        return fallback_parsing_reply(raw_output)
    raw_answer = extract_raw_output(raw_output)
    if raw_answer:
        return raw_answer
    final_answer = extract_final_answer(raw_output)
    if final_answer:
        return final_answer
    if is_parsing_error_message(raw_output):
        return fallback_parsing_reply(raw_output)
    return None

def normalize_intent_text(text: str) -> str:
    lowered = text.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    return "".join(char for char in normalized if not unicodedata.combining(char))

def normalize_quotes(text: str) -> str:
    return (
        text.replace("“", "\"")
        .replace("”", "\"")
        .replace("«", "\"")
        .replace("»", "\"")
        .replace("’", "'")
    )

def get_approved_tool_args(tool_name: str) -> dict | None:
    approvals = getattr(g, "approved_tool_calls", [])
    for approval in approvals:
        if not isinstance(approval, dict):
            continue
        if approval.get("name") != tool_name:
            continue
        args = approval.get("args")
        if isinstance(args, dict):
            return args
    return None

def is_time_request(user_message: str) -> bool:
    text = normalize_intent_text(user_message)
    patterns = [
        r"\bquelle heure\b",
        r"\bil est quelle heure\b",
        r"\bheure\b",
        r"\btime\b",
        r"\bwhat time\b",
        r"\bcurrent time\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)

def is_create_note_request(user_message: str) -> bool:
    normalized_message = normalize_quotes(user_message)
    text = normalize_intent_text(normalized_message)
    has_target = bool(re.search(r"\b(note|fichier|file|txt)\b", text) or re.search(r"\b\w+\.txt\b", text))
    has_action = bool(re.search(r"\b(cree|creer|ecrire|sauvegarder|enregistrer|write|create|met|mettre)\b", text))
    has_txt_context = bool(re.search(r"\bdans\s+un?\s+\.?txt\b", text))
    return has_target and (has_action or has_txt_context)

def extract_note_args_from_message(user_message: str) -> tuple[dict | None, str | None]:
    normalized_message = normalize_quotes(user_message)

    filename_match = re.search(r"\b([a-zA-Z0-9._-]+\.txt)\b", normalized_message, re.IGNORECASE)
    filename = filename_match.group(1) if filename_match else "note.txt"

    content = None
    content_source = "none"
    quoted_match = re.search(r"\"([^\"]+)\"|'([^']+)'", normalized_message)
    if quoted_match:
        content = quoted_match.group(1) or quoted_match.group(2)
        content_source = "quoted"

    if not content:
        between_match = re.search(
            r"(?:ecrire|écrire|mettre)\s+(.+?)\s+(?:dans|en)\s+un?\s+\.?txt\b",
            normalized_message,
            re.IGNORECASE,
        )
        if between_match:
            content = between_match.group(1)
            content_source = "inferred"

    if not content:
        trailing_match = re.search(
            r"(?:avec|contenant|contenu|texte|ecrit|écrit)\s*[:\-]?\s*(.+)$",
            normalized_message,
            re.IGNORECASE,
        )
        if trailing_match:
            content = trailing_match.group(1)
            content_source = "inferred"

    if not content:
        return None, (
            "Je peux creer le .txt, mais j'ai besoin du contenu exact. "
            "Exemple: cree `note.txt` avec \"Taiwan est un pays libre\"."
        )

    cleaned_content = content.strip().strip(" .")
    if not cleaned_content:
        return None, (
            "Je peux creer le .txt, mais le contenu est vide. "
            "Donne-moi une phrase a ecrire entre guillemets."
        )

    # If the content was inferred (not explicitly quoted) and asks for "date/time of today",
    # replace it with the current timestamp.
    if content_source != "quoted":
        normalized_content = normalize_intent_text(cleaned_content)
        asks_for_datetime = (
            ("date du jour" in normalized_content)
            or ("date" in normalized_content and "heure" in normalized_content)
            or ("heure actuelle" in normalized_content)
            or ("date actuelle" in normalized_content)
        )
        if asks_for_datetime:
            cleaned_content = current_datetime_text()

    return {"filename": filename, "content": cleaned_content}, None

def format_time_reply(raw_value: str) -> str:
    cleaned = raw_value.strip()
    try:
        parsed = datetime.fromisoformat(cleaned)
        return f"Il est actuellement {parsed.strftime('%d/%m/%Y %H:%M:%S')}."
    except Exception:
        return f"Heure actuelle: {cleaned}"

def current_datetime_text() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def try_direct_tool_reply(user_message: str, selected_tools):
    selected_tool_names = {tool.name for tool in selected_tools}
    create_note_intent = is_create_note_request(user_message)
    if create_note_intent:
        if "create_note" not in selected_tool_names:
            return jsonify({
                "reply": "L'outil `Creer une note` est desactive. Active-le puis renvoie la demande."
            })
        approved_args = get_approved_tool_args("create_note")
        if approved_args:
            tool_args = approved_args
            clarification_message = None
        else:
            tool_args, clarification_message = extract_note_args_from_message(user_message)
        if clarification_message:
            return jsonify({"reply": clarification_message})
        try:
            result = tool_by_name["create_note"].invoke(tool_args)
            return jsonify({"reply": result})
        except ToolConfirmationRequired as confirmation_error:
            return jsonify({
                "reply": confirmation_message(confirmation_error.tool_name, confirmation_error.tool_args),
                "requires_confirmation": True,
                "tool_call": {
                    "name": confirmation_error.tool_name,
                    "args": confirmation_error.tool_args,
                },
            })
        except Exception:
            return None
    if is_time_request(user_message) and "get_current_time" in selected_tool_names:
        try:
            tool_result = tool_by_name["get_current_time"].invoke({})
            return jsonify({"reply": format_time_reply(str(tool_result))})
        except Exception:
            return None
    return None

def should_use_agent(user_message: str, selected_tools) -> bool:
    selected_tool_names = {tool.name for tool in selected_tools}
    text = normalize_intent_text(user_message)
    if "create_note" in selected_tool_names:
        note_patterns = [
            r"\bcree\b.*\b(note|fichier|txt)\b",
            r"\bcreer\b.*\b(note|fichier|txt)\b",
            r"\becrire\b.*\b(note|fichier|txt)\b",
            r"\bsauvegarder\b.*\b(note|fichier|txt)\b",
            r"\bwrite\b.*\bfile\b",
            r"\bcreate\b.*\b(note|file)\b",
        ]
        if any(re.search(pattern, text) for pattern in note_patterns):
            return True
    return False

def get_tool_fields(tool_obj) -> list[str]:
    schema = getattr(tool_obj, "args_schema", None)
    if not schema:
        return []
    fields = getattr(schema, "model_fields", None)
    if fields:
        return list(fields.keys())
    fields = getattr(schema, "__fields__", None)
    if fields:
        return list(fields.keys())
    return []

def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value

def strip_wrapping_chars(value: str) -> str:
    cleaned = value.strip()
    for opening, closing in (("(", ")"), ("[", "]")):
        if cleaned.startswith(opening) and cleaned.endswith(closing):
            cleaned = cleaned[1:-1].strip()
    return cleaned

def parse_action_input(tool_obj, raw_input: str) -> dict:
    cleaned = strip_wrapping_chars(raw_input.strip().strip("`"))
    if cleaned.lower() in ("none", "null", ""):
        return {}
    if cleaned.startswith("{"):
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    fields = get_tool_fields(tool_obj)
    if fields:
        if len(fields) == 1:
            return {fields[0]: strip_quotes(strip_wrapping_chars(cleaned))}
        if "," in cleaned:
            parts = [
                strip_quotes(strip_wrapping_chars(part.strip()))
                for part in cleaned.split(",", maxsplit=len(fields) - 1)
            ]
            if len(parts) >= len(fields):
                return {field: parts[idx] for idx, field in enumerate(fields)}
    return {}

def resolve_placeholders(tool_args: dict) -> dict:
    resolved = {}
    for key, value in tool_args.items():
        if isinstance(value, str) and "get_current_time()" in value:
            current_time = get_current_time.invoke({})
            resolved[key] = value.replace("get_current_time()", current_time).strip()
        elif isinstance(value, str):
            resolved[key] = strip_wrapping_chars(value.strip())
        else:
            resolved[key] = value
    return resolved

def try_handle_action_from_text(raw_text: str):
    action_call = extract_action_call(raw_text)
    if not action_call:
        return None
    tool_name, action_input = action_call
    tool_obj = tool_by_name.get(tool_name)
    if not tool_obj:
        return None
    tool_args = resolve_placeholders(parse_action_input(tool_obj, action_input))
    try:
        result = tool_obj.invoke(tool_args)
        return jsonify({"reply": result})
    except ToolConfirmationRequired as confirmation_error:
        return jsonify({
            "reply": confirmation_message(confirmation_error.tool_name, confirmation_error.tool_args),
            "requires_confirmation": True,
            "tool_call": {
                "name": confirmation_error.tool_name,
                "args": confirmation_error.tool_args,
            },
        })
    except Exception:
        return None

def confirmation_message(tool_name: str, tool_args: dict) -> str:
    if tool_name == "create_note":
        filename = tool_args.get("filename") or "le fichier"
        return f"Je peux creer le fichier '{filename}'. Souhaite-tu confirmer ?"
    return f"Je peux appeler l'outil '{tool_name}'. Souhaite-tu confirmer ?"

# 4. Define the /chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles chat messages from the frontend.
    Invokes the LangChain agent with the user's message and tools.
    """
    data = request.json or {}
    user_message = sanitize_message_content(data.get('message'))
    enabled_tools = data.get('enabled_tools')
    conversation_messages = data.get('messages')
    require_tool_confirmation = bool(data.get('require_tool_confirmation'))
    tool_approvals = data.get('tool_approvals')
    model_name = sanitize_message_content(data.get('model')) or DEFAULT_MODEL_NAME

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Per-request tool confirmation settings
    g.require_tool_confirmation = require_tool_confirmation
    g.approved_tool_calls = tool_approvals if isinstance(tool_approvals, list) else []

    # Resolve tool selection from the UI
    if isinstance(enabled_tools, list):
        selected_tools = [tool_by_name[name] for name in enabled_tools if name in tool_by_name]
    else:
        selected_tools = tools
    conversation_history = conversation_messages if isinstance(conversation_messages, list) else []

    direct_reply = try_direct_tool_reply(user_message, selected_tools)
    if direct_reply:
        return direct_reply

    if not should_use_agent(user_message, selected_tools):
        plain_reply = get_plain_chat_reply(
            user_message=user_message,
            conversation_history=conversation_history,
            model_name=model_name,
        )
        if plain_reply:
            return jsonify({"reply": plain_reply})

    agent_executor = get_agent_executor(selected_tools, model_name=model_name)
    agent_input = build_agent_input(
        user_message=user_message,
        conversation_history=conversation_history,
    )

    # Invoke the agent with the user's message
    try:
        response = agent_executor.invoke({
            "input": agent_input
        })
        
        # The agent's final answer is in the 'output' key
        ai_response = response.get('output')
        
        return jsonify({"reply": ai_response})

    except Exception as e:
        if isinstance(e, ToolConfirmationRequired):
            return jsonify({
                "reply": confirmation_message(e.tool_name, e.tool_args),
                "requires_confirmation": True,
                "tool_call": {
                    "name": e.tool_name,
                    "args": e.tool_args,
                },
            })
        raw_error = str(e)
        if is_parsing_error_message(raw_error):
            direct_reply = try_direct_tool_reply(user_message, selected_tools)
            if direct_reply:
                return direct_reply
        action_response = try_handle_action_from_text(raw_error)
        if action_response:
            return action_response
        reply_from_error = try_extract_reply_from_error(raw_error)
        if reply_from_error:
            return jsonify({"reply": reply_from_error})

        if isinstance(e, OutputParserException):
            llm_output = getattr(e, "llm_output", None) or str(e)
            if is_parsing_error_message(str(llm_output)):
                direct_reply = try_direct_tool_reply(user_message, selected_tools)
                if direct_reply:
                    return direct_reply
                plain_reply = get_plain_chat_reply(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    model_name=model_name,
                )
                if plain_reply:
                    return jsonify({"reply": plain_reply})
            action_response = try_handle_action_from_text(llm_output)
            if action_response:
                return action_response
            reply_from_error = try_extract_reply_from_error(llm_output)
            if reply_from_error:
                return jsonify({"reply": reply_from_error})
            final_answer = extract_final_answer(llm_output)
            if final_answer:
                return jsonify({"reply": final_answer})
            raw_answer = extract_raw_output(llm_output)
            if raw_answer:
                return jsonify({"reply": raw_answer})
            if isinstance(llm_output, str) and llm_output.strip():
                return jsonify({"reply": llm_output})
            return jsonify({"reply": fallback_parsing_reply(str(e))})
        plain_reply = get_plain_chat_reply(
            user_message=user_message,
            conversation_history=conversation_history,
            model_name=model_name,
        )
        if plain_reply:
            return jsonify({"reply": plain_reply})
        print(f"Agent execution error: {e}")
        traceback.print_exc()
        error_payload = {"error": "An error occurred while processing your request."}
        if app.debug:
            error_payload["details"] = str(e)
        return jsonify(error_payload), 500


# 5. Run the Flask App
if __name__ == '__main__':
    # --- Ollama Connection Test ---
    print("--- Testing Ollama Connection ---")
    try:
        # Use the model specified by the user
        llm_test = Ollama(model=DEFAULT_MODEL_NAME)
        llm_test.invoke("Hello")
        print("✅ Ollama connection successful!")
    except Exception as e:
        print("❌ Ollama connection failed.")
        print(f"   Error: {e}")
        print(
            "   Please make sure Ollama is running and you have pulled the "
            f"'{DEFAULT_MODEL_NAME}' model (e.g., 'ollama pull {DEFAULT_MODEL_NAME}')."
        )
        # Exit if we can't connect to Ollama
        exit()
    print("---------------------------------")
    
    # Note: `debug=True` is for development only.
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

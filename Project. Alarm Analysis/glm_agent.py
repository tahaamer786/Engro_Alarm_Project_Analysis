import os
import asyncio
import json
import inspect
from typing import AsyncGenerator, Dict, Any, List, Callable
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- Load API key securely from .env file ---
load_dotenv()
CLIENT_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not CLIENT_API_KEY:
    raise ValueError("❌ API key missing! Please set OPENROUTER_API_KEY in your .env file.")

# --- Initialize client ---
client = AsyncOpenAI(
    api_key=CLIENT_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# --- System prompt ---
SYSTEM_PROMPT = """
You are a highly efficient and accurate **SQL Data Analysis Agent**. Your task is to analyze the user's query and translate it into a single, correct **SQLite SQL SELECT query** to be executed by the provided tool. After receiving the result, provide a clear, complete, and correct FINAL ANSWER.

**RULES SUMMARY:**
- Table name: **alerts**
- Key columns: "Event Time", "Location Tag", "Source", "Condition", "Action", "Priority", "Description", "Value", "Units".
- Use COUNT(*), GROUP BY, ORDER BY, and LIMIT appropriately.
- Convert all filters to UPPERCASE.
- Always use the 'execute_sql_query' tool for analysis queries.
- Return final formatted answer only after getting tool results.
- Use clear English formatting for answers.

AVAILABLE TOOLS:
{tools_schema}
"""


async def run_glm_agent(
        query: str,
        tools: List[Callable],
        model: str = "z-ai/glm-4.5-air:free",
        max_iterations: int = 4
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Runs the LLM agent, handling function calling and streaming reasoning/output.
    """

    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": t.__name__,
                "description": t.__doc__,
                "parameters": {name: str(param) for name, param in inspect.signature(t).parameters.items()}
            }
        }
        for t in tools
    ]

    formatted_prompt = SYSTEM_PROMPT.format(
        tools_schema=json.dumps([t['function'] for t in tools_schema], indent=2)
    )

    messages = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": query}
    ]

    tool_map = {t.__name__: t for t in tools}
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        try:
            response_stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_schema,
                stream=True,
                extra_body={"reasoning": {"effort": "high"}}
            )

            reasoning_buffer = ""
            function_call_info = None
            final_answer_stream = ""

            async for chunk in response_stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta

                    if hasattr(delta, "reasoning") and delta.reasoning:
                        reasoning_buffer += delta.reasoning

                    elif delta.tool_calls:
                        tool_call = delta.tool_calls[0]
                        if not function_call_info:
                            function_call_info = {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "arguments": ""
                            }
                        if tool_call.function.arguments:
                            function_call_info["arguments"] += tool_call.function.arguments

                    elif delta.content:
                        final_answer_stream += delta.content
                        yield {"type": "answer_stream", "content": delta.content}

            if reasoning_buffer:
                yield {"type": "reasoning", "content": reasoning_buffer.strip()}

            if function_call_info and function_call_info.get("name"):
                tool_name = function_call_info["name"]
                yield {"type": "tool_call", "data": {"name": tool_name, "arguments": function_call_info["arguments"]}}

                try:
                    tool_args = json.loads(function_call_info["arguments"])
                except json.JSONDecodeError:
                    yield {"type": "error", "message": f"Tool arguments JSON decode error for {tool_name}"}
                    break

                tool_func = tool_map.get(tool_name)
                tool_result = tool_func(**tool_args)

                print("\n\n[DEBUG] TOOL RESULT RAW OUTPUT:\n", tool_result[:500], "\n")

                yield {
                    "type": "tool_result",
                    "content": tool_result[:500] + "..." if len(tool_result) > 500 else tool_result
                }

                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": function_call_info["id"],
                        "function": {
                            "name": tool_name,
                            "arguments": function_call_info["arguments"]
                        },
                        "type": "function"
                    }]
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": function_call_info["id"],
                    "content": tool_result[:5000]
                })

                continue

            if final_answer_stream:
                yield {"type": "answer_complete", "content": final_answer_stream.strip()}
                yield {"type": "complete", "data": {"iterations": iteration}}
                break

            if not reasoning_buffer and not final_answer_stream:
                yield {"type": "error", "message": "Model did not provide a response or a tool call."}
                break

        except Exception as e:
            yield {"type": "error", "message": f"General Error: {str(e)}", "iteration": iteration}
            break

    if iteration >= max_iterations:
        yield {"type": "error", "message": "Max iterations reached without final answer"}

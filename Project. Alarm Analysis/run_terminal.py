# run_terminal.py (FIXED + EXECUTE TOOL CALLS + FORMATTED OUTPUT)

import asyncio
import json
from data_tools import load_data, AVAILABLE_TOOLS
from glm_agent import run_glm_agent


async def main():
    """Main function to run the agent in the terminal loop with proper tool execution."""

    # --- Load data at startup ---
    data_loaded = load_data(file_path='All_Merged (1).csv')
    if not data_loaded:
        return

    # --- Console Welcome Message ---
    print("\n==============================================")
    print(" 🛠️  GLM SQL Data Agent is Ready (Final Setup) 🛠️")
    print("==============================================")
    print("\nHello! I am your Alarms Data Analysis Agent (GLM Agent).")
    print("I have loaded and cleaned your alerts and am ready for accurate analysis.")
    print("Type 'exit' anytime to quit.")

    while True:
        user_query = input(
            "\nQUERY: (e.g., 'List the Top 5 Source units' or 'Analyze alarm behavior for last 10 minutes' or 'exit'):\n> "
        ).strip()

        if user_query.lower() == 'exit':
            print("\nSession ended. Goodbye!")
            break

        print("\n--- [Reasoning 🔄] ---")
        agent_output_stream = run_glm_agent(
            query=user_query,
            tools=AVAILABLE_TOOLS,
            model="z-ai/glm-4.5-air:free",
            max_iterations=4
        )

        reasoning_text = ""  # buffer for reasoning

        async for chunk in agent_output_stream:
            chunk_type = chunk.get("type")
            content = chunk.get("content")

            # === COLLECT REASONING ===
            if chunk_type == "reasoning":
                reasoning_text += content.strip() + " "

            # === TOOL CALL EXECUTION + FORMATTED OUTPUT ===
            elif chunk_type == "tool_call":
                data = chunk.get("data", {})
                tool_name = data.get("name")
                print("\n--- [TOOL CALL EXECUTING] ---")
                print(f"Function: {tool_name}")
                print(f"Args: {data.get('arguments')}")

                try:
                    tool_func = {t.__name__: t for t in AVAILABLE_TOOLS}.get(tool_name)
                    if tool_func:
                        tool_args = json.loads(data.get("arguments", "{}"))
                        tool_result = tool_func(**tool_args)

                        # Parse JSON (SQL SELECT results)
                        try:
                            parsed = json.loads(tool_result)
                            if isinstance(parsed, list) and parsed:
                                # Print as neat table
                                headers = parsed[0].keys()
                                col_widths = {h: max(len(str(h)), max(len(str(r[h])) for r in parsed)) for h in headers}

                                header_row = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
                                print("\n" + header_row)
                                print("-" * len(header_row))

                                for row in parsed:
                                    print(" | ".join(f"{str(row[h]):<{col_widths[h]}}" for h in headers))
                            else:
                                print(tool_result)
                        except Exception:
                            # Not JSON, just print raw
                            print(tool_result)
                    else:
                        print(f"❌ Tool {tool_name} not found.")
                except Exception as e:
                    print(f"\n❌ ERROR executing tool {tool_name}: {e}")

            # === TOOL RESULT HANDLER (ALARM ANALYSIS JSON) ===
            elif chunk_type == "tool_result":
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and ("per_source" in parsed or "bad_actor" in parsed):
                        print("\n=== 🧩 Alarm Analysis Summary ===")
                        print("Total rows processed:", parsed.get("total_rows", "N/A"))

                        if parsed.get("bad_actor"):
                            print("\nBad Actors:", parsed.get("bad_actor"))

                        unhealthy = parsed.get("unhealthy_sources", {})
                        if unhealthy:
                            print("\nUnhealthy Sources (source: max_count_in_window):")
                            for s, c in unhealthy.items():
                                print(f" - {s}: {c}")

                        per_src = parsed.get("per_source", [])
                        if per_src:
                            print("\nTop per-source stats (first 10):")
                            for row in per_src[:10]:
                                print(
                                    f" {row['Source']:<20} total:{row['Total Alarms']:<6} "
                                    f"active:{row['Active Count']:<4} stale:{row['Stale Count']:<4} "
                                    f"chatter:{row['Chattering Count']:<4}"
                                )

                        floods = parsed.get("floods", [])
                        if floods:
                            print("\nFlood windows detected:")
                            for f in floods:
                                print(f" - {f['window_start']} : {f['count']}")

                        print("\n==============================================")
                    else:
                        print("\n--- [TOOL RESULT RECEIVED] ---\n")
                        print(content)
                except Exception:
                    print("\n--- [TOOL RESULT RECEIVED] ---\n")
                    print(content)

            # === FINAL ANSWER STREAM ===
            elif chunk_type == "answer_stream" and content:
                if reasoning_text:
                    print("\n--- [REASONING] ---")
                    print(reasoning_text.strip())
                    print("\n--- [FINAL ANSWER] ---")
                    reasoning_text = ""
                print(content, end="", flush=True)

            # === FINAL COMPLETION HANDLER ===
            elif chunk_type == "answer_complete":
                print("\n\n **FINAL ANSWER COMPLETE** ")

            elif chunk_type == "error":
                print(f"\n **ERROR**: {chunk.get('message')}")
                break

            elif chunk_type == "complete":
                break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent Terminated.")

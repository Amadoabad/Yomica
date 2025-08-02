import os
import argparse
import sys
from dotenv import load_dotenv
import google.generativeai as genai
import json

from shell_agent.agent import ShellAgent

load_dotenv()

chat_history = []
shell_agent = ShellAgent()

def initialize_gemini_model(model = 'gemini-2.5-pro'):
    """Initializes and returns the Gemini GenerativeModel."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set it in a .env file or directly in your shell.")
        sys.exit(1) 

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model,
                                 tools=shell_agent.get_tool_definitions())

def handle_gemini_response(response_stream):
    """
    Processes the streaming response from Gemini, handling both text and tool calls.
    Returns any text output from the LLM.
    """
    full_llm_text_response = ""
    tool_calls_to_execute = []

    for chunk in response_stream:
        
        if hasattr(chunk, 'parts'):
            for part in chunk.parts:
                if hasattr(part, 'text') and part.text:
                    print(part.text, end='', flush=True)
                    full_llm_text_response += part.text

                elif hasattr(part, 'function_call') and part.function_call:
                    tool_calls_to_execute.append(part.function_call)
                

        elif hasattr(chunk, 'text') and chunk.text:
            print(chunk.text, end="", flush=True)
            full_llm_text_response += chunk.text


    return full_llm_text_response, tool_calls_to_execute

def chat_session(mode='safe'):
    """
    Handless the interactive chat session with the LLM.
    """

    model = initialize_gemini_model()
    chat = model.start_chat(history=chat_history)
    print("\n--- Yomica Chatbot (Type 'exit' or 'quit' to end) ---")
    print("----------------------------------------------------\n")

    while True:
        try:
            print('\n', "-"*50)
            user_input = input("You: ")
            print("-"*50)
            if user_input.lower() in ["exit", "quit"]:
                print("Yomica: Goodbye!")
                break

            chat_history.append({"role": 'user', 'parts': [user_input]})
            print("Yomica: ", end="", flush=True)

            response_stream = chat.send_message(user_input, stream=True)
            llm_text, tool_calls = handle_gemini_response(response_stream)

            if llm_text:
                chat_history.append({'role': 'model', 'parts': [llm_text]})

            if tool_calls:
                
                for tool_call in tool_calls:
                    if tool_call.name == "execute_shell_command":
                        command = tool_call.args.get('command')
                        args = tool_call.args.get('args', [])
                        full_command_str = f'{command} {' '.join(args)}'

                        print("\n(Yomica has requested a command...)\n", flush=True)
                        print(f"Proposed Command: {full_command_str}\n", flush=True)

                        can_execute = False
                        
                        if mode == "safe":
                            if command in shell_agent.safe_commands:
                                can_execute = True

                            else:
                                print(f"Yomica: i'm in 'safe' mode. The command '{command} is not on my approved list.")
                                tool_result = f"Command '{command}' was not executed because it is not an approved command in 'safe' mode."
                                can_execute = False

                        elif mode == "wild":
                            user_approval = input(f"Yomica: Do you approve this command: '{command}'? (y/n) ").lower().strip()
                            if user_approval == 'y':
                                if command in shell_agent.dangerous_commands:
                                    print(f"Yomica: I'm in 'wild' mode. but this command '{command}' is on my dangerous list.")
                                    ensure_approval = input("Yomica: Are you really sure you want me to execute this command? (y/n) ").lower().strip()
                                    if ensure_approval == 'y':
                                        can_execute = True
                                    else:
                                        can_execute = False
                                        print("Yomica: Command canceled by user.")
                                        tool_result = "Command canceled by user."
                                else:
                                    can_execute = True
                            else:
                                can_execute = False
                                print("Yomica: Command canceled by user.")
                                tool_result = "Command canceled by user."
                                
                        if can_execute:
                            chat_history.append({'role': 'model', 'parts': [{"function_call": {"name": tool_call.name, "args": {"command": command, "args": args}}}]})

                            print("\n(Executing approved command...)\n", flush=True)
                            tool_result = shell_agent.call_tool({'name': tool_call.name, 'args': tool_call.args})
                            print(f'Command Output:\n{tool_result}\n', flush=True)

                        chat_history.append({'role': 'tool', 'parts': [{"function_response": {"name": tool_call.name, "response": {"content": tool_result}}}]})
                    
                    # Re-create chat object with the new tool output in history
                    chat = model.start_chat(history=chat_history)

                    print("Yomica (interpreting result): ", end="", flush = True)

                    try:
                        final_response_stream = chat.send_message("Summerize the tool output", stream=True)
                        final_llm_text, _ = handle_gemini_response(final_response_stream)
                        if final_llm_text:
                            chat_history.append({'role': 'model', 'parts':[final_llm_text]})
                        print()

                    except Exception as e:
                        print(f"\nError getting follow-up LLM response after tool execution: {e}")
                        if chat_history and chat_history[-1]["role"] == "tool":
                            chat_history.pop()
            # print(chat_history)

            if not llm_text and not tool_calls:
                print("(Yomica didn't respnd or call a tool directly. Try rephrasing)", flush=True)  

        except KeyboardInterrupt:
            print("Oh, okay.. Goodbye!")
            break
        
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again or restart the chat.")

            if len(chat_history) >=2:
                chat_history.pop()
                chat_history.pop()





def main():
    parser = argparse.ArgumentParser(description="Yomica LLM Chatbot.")
    parser.add_argument("query", nargs='*', help="Your query for the LLM. If empty, starts interactive chat.")
    parser.add_argument("--mode", choices=["safe", "wild"], default="safe",
                        help="Sets the agent's operating mode. 'safe' mode only executes predefined commands. 'wild' mode executes any command but after user confirmation.")
    args = parser.parse_args()

    if args.query:
        print('Tool exection is currently only fully supported in interactive chat mode..')
        print('Falling back to basic LLM response for single query.')

        user_query = ''.join(args.query)
        model = initialize_gemini_model()
        chat = model.start_chat(history=[])
        
        print(f"Yomica: I'm thinking...")
        try:
            full_response_text = ""
            response_stream = chat.send_message(user_query, stream=True)
            print("\n------ Yomica's Response ------")
            for chunk in response_stream:
                if hasattr(chunk, 'text'):
                    print(chunk.text, end='', flush=True)
                    full_response_text += chunk.text
            print("\n----------------------\n")

        except Exception as e:
            print(f"\nAn error occurred while getting response: {e}")
    
    else:
        # If no query start interactive chat session
        chat_session(mode=args.mode)


if __name__ == "__main__":
    main()
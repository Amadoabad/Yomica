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

def chat_session():
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
                
                chat_history.append({'role':'model',
                                      'parts':[{"function_call": {"name": tc.name, "args": tc.args}} for tc in tool_calls]})

                print("\n(Yomica is executing a command...)\n", flush=True)
                for tool_call in tool_calls:
                    tool_result = shell_agent.call_tool(
                        {'name': tool_call.name, 'args': tool_call.args}, 
                    )
                    print(f'Command Output:\n{tool_result}\n', flush=True)

                    chat_history.append({'role': 'tool', 
                                         'parts': [{"function_response": {"name": tool_call.name, "response": {"content": tool_result}}}]})
                    
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
        chat_session()


if __name__ == "__main__":
    main()
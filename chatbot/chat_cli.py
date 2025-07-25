import os
import argparse
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types

load_dotenv()

chat_history = []
def initialize_gemini_model(model = 'gemini-2.5-flash-lite'):
    """Initializes and returns the Gemini GenerativeModel."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set it in a .env file or directly in your shell.")
        sys.exit(1) 

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model)

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

            full_response_text = ""

            response_stream = chat.send_message(user_input, stream=True)

            for chunk in response_stream:
                if hasattr(chunk, 'text'):
                    print(chunk.text, end='', flush=True)
                    full_response_text +=chunk.text
                else:
                    print("Non-text chunk", end='', flush=True)
                    pass
            


            if full_response_text:
                chat_history.append({'role': 'model', 'parts': [full_response_text]})

        except KeyboardInterrupt:
            print("Oh, okay.. Goodbye!")
            break
        
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again or restart the chat.")

            if chat_history and chat_history[-1]['role'] == 'user':
                chat_history.pop()





def main():
    parser = argparse.ArgumentParser(description="Yomica LLM Chatbot.")
    parser.add_argument("query", nargs='*', help="Your query for the LLM. If empty, starts interactive chat.")
    args = parser.parse_args()

    if args.query:
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

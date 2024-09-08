import ollama
from typing import List, Dict, Any, Generator, Tuple

def get_response_from_llm(
    prompt: str,
    client: Any,
    model: str,
    system_message: str,
    msg_history: List[Dict[str, str]],
    stream: bool = False
) -> Tuple[str, List[Dict[str, str]]]:
    try:
        # Prepare the messages
        messages = [{"role": "system", "content": system_message}] + msg_history + [{"role": "user", "content": prompt}]

        if stream:
            # For streaming responses
            response_content = ""
            stream = client.chat(model=model, messages=messages, stream=True)
            for chunk in stream:
                content = chunk['message']['content']
                response_content += content
                print(content, end='', flush=True)
            
            # Update message history
            msg_history.append({"role": "user", "content": prompt})
            msg_history.append({"role": "assistant", "content": response_content})
            
            return response_content, msg_history
        else:
            # For non-streaming responses
            response = client.chat(model=model, messages=messages)
            response_content = response['message']['content']

            # Update message history
            msg_history.append({"role": "user", "content": prompt})
            msg_history.append({"role": "assistant", "content": response_content})
            
            return response_content, msg_history

    except ollama.ResponseError as e:
        print(f"Ollama Response Error: {e.error}")
        if e.status_code == 404:
            print(f"Model {model} not found. Attempting to pull...")
            ollama.pull(model)
            # Retry the request after pulling the model
            return get_response_from_llm(prompt, client, model, system_message, msg_history, stream)
        else:
            raise  # Re-raise the exception if it's not a 404 error

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise

# Example usage:
# client = ollama.Client(host='http://localhost:11434')
# model = 'llama3.1'
# system_message = "You are a helpful assistant."
# prompt = "Why is the sky blue?"
# msg_history = []
# 
# response, updated_history = get_response_from_llm(prompt, client, model, system_message, msg_history)
# print(response)
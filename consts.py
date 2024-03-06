
from edsl import Model

model_to_str = {
    'gpt-3.5-turbo': 'gpt35',
    'gpt-4-1106-preview' : 'gpt4', 
    'gemini_pro': 'gemini', 
    'llama-2-13b-chat-hf': 'llama13b', 
    'llama-2-70b-chat-hf': 'llama70b', 
    'mixtral-8x7B-instruct-v0.1': 'mixtral'
}

if len(model_to_str.keys()) != len(Model.available()):
    raise ValueError('You need to update the model_to_str dictionary')
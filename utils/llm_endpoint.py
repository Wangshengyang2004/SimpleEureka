import openai
import json
from zhipuai import ZhipuAI
import os

os.environ["ZHIPU_API_KEY"] = "438e5ef5f2db0fd2e49d9de643d9a75e.XmvmE3anEtULIKAd"

def ask_llm(model, messages, temperature, chunk_size):
    if model == "glm-4":
        client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
        print("ZhipuAI API Key: ", os.getenv("ZHIPU_API_KEY"))
        _response_ = client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = temperature,
            # n = chunk_size,
        )
        return {"content": _response_.choices[0].message.content,
                "prompt_tokens": _response_.usage.prompt_tokens, 
                "completion_tokens":_response_.usage.completion_tokens, 
                "total_tokens":_response_.usage.total_tokens
        }
    elif model == 'gpt-4' or model == 'gpt-3.5':
        # Setting the API key for OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = temperature,
            n = chunk_size,
        )
        return _response_
    
    elif model == 'local':
        openai.api_base = "http://localhost:8000/v1"
        openai.api_key = "local"
        _response_ = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = temperature,
            n = chunk_size,
        )
        return _response_
    
    
if __name__ == "__main__":
    data = ask_llm(model = "glm-4", messages=[
              {"role": "system", "content": "你是一个聪明且富有创造力的小说作家"},
              {"role": "user", "content": "请你作为童话故事大王，写一篇短篇童话故事，故事的主题是要永远保持一颗善良的心，要能够激发儿童的学习兴趣和想象力，同时也能够帮助儿童更好地理解和接受故事中所蕴含的道理和价值观。"}
    ], temperature = 0.95, chunk_size = 4)
    print(data)
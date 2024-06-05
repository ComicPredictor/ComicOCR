# import anthropic
# import time
# import json 
# import usecomicsocr

# ocrd=usecomicsocr.textnpos("comic_data/test/image.png")
# TOKEN = json.load(open('claude-api.json'))["APIKEY"]
# client = anthropic.Anthropic(
#     api_key=TOKEN,
# )
# print([(client.messages.create(
#     model="claude-3-opus-20240229",
#     max_tokens=100,
#     messages=[
#         {"role": "user", "content": "Help me decode this message:"+string}
#     ]
# ).content[0].text, time.sleep(12.0)) for string, _ in ocrd])
from openai import OpenAI
# import usecomicsocr
import json
# ocrd=usecomicsocr.textnpos("comic_data/test/image.png")
client = OpenAI(
    api_key=json.load(open('claude-api.json'))["rpAPI"],
    base_url="https://api.runpod.ai/v2/vllm-t7qtw0sod2q3cn/openai/v1",
)
def decode(s):
    return client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a assistant, skilled in decoding mistyped/mispelled/no-spacing messages. You must answer the message with a normal sentence. Do not add greetings or explanations while decoding the message. Label the new message like the following example `[dec start]decoded message[dec end]`"},
            {"role": "user", "content":"decode this message: %s"%s}
        ],
        temperature=0.3,
        max_tokens=100,
    ).choices[0].message.content.split('[dec start]')[-1].split('[dec end]')[0]

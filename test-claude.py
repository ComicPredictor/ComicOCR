import anthropic
import json 

TOKEN = json.load(open('claude-api.json'))["APIKEY"]
client = anthropic.Anthropic(
    api_key=TOKEN,
)
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content[0].text)
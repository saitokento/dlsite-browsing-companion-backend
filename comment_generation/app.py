import json
from openai import OpenAI

client = OpenAI()

def lambda_handler(event, context):
    response = client.responses.create(
        model="gpt-5-nano",
        input="Hello World"
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "output_text": response.output_text,
        }),
    }

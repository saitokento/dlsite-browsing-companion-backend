import json
from openai import OpenAI
import boto3
from botocore.exceptions import ClientError

def get_openai_api_key():
    secret_name = "prod/DBC/OpenAI"
    region_name = "ap-northeast-1"

    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError:
        raise

    return get_secret_value_response['SecretString']

OPENAI_API_KEY = get_openai_api_key()
client = OpenAI(api_key=OPENAI_API_KEY)

def lambda_handler(event, context):
    headers = {
        "Access-Control-Allow-Origin": "*",

        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "Access-Control-Allow-Methods": "OPTIONS,POST",
        "Content-Type": "application/json",
    }

    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": headers, "body": ""}
    
    body = json.loads(event.get('body', '{}'))
    input = body.get('input', '')

    response = client.responses.create(
        model="gpt-5-nano",
        input=input,
        instructions="あなたはユーザーの友人で、ユーザーと一緒にDLsiteを見ています。"
    )

    return {
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps({
            "output_text": response.output_text,
        }),
    }

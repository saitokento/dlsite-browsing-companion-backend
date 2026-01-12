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
    except ClientError as e:
        raise RuntimeError(f"Failed to retrieve secret from Secrets Manager: {e}") from e

    secret_string = get_secret_value_response['SecretString']
    try:
        secret_json = json.loads(secret_string)
        api_key = secret_json.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in secret")
        return api_key
    except json.JSONDecodeError:
        return secret_string

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
    
    try:
        body = json.loads(event.get('body', '{}'))
    except json.JSONDecodeError as e:
        return {
            "statusCode": 400,
            "headers": headers,
            "body": json.dumps({"error": "Invalid JSON format"}),
        }
    
    prompt = body.get('input', '')

    if not prompt or not prompt.strip():
        return {
            "statusCode": 400,
            "headers": headers,
            "body": json.dumps({"error": "Prompt is required"}),
        }

    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt,
            instructions="あなたはユーザーの友人で、ユーザーと一緒にDLsiteを見ています。"
        )

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": f"OpenAI API error: {str(e)}"}),
        }
    
    output_text = getattr(response, 'output_text', None)
    if output_text is None:
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": "Invalid response from OpenAI API"}),
        }

    return {
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps({
            "output_text": output_text,
        }),
    }

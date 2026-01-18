import json
from openai import OpenAI
import boto3
from botocore.exceptions import ClientError
from openai import APIError, APIConnectionError, RateLimitError
from xai_sdk import Client
from xai_sdk.chat import user, system

def get_api_keys():
    """
    AWS Secrets Manager から OpenAI, xAI の API キーを取得する。
    
    Secrets Manager のシークレット "prod/DBC/APIKeys"（リージョン ap-northeast-1）を読み取り、シークレット文字列を JSON として解析してキー "OPENAI_API_KEY", "XAI_API_KEY" の値を返す。シークレットが有効な JSON でない場合は、パースせずにシークレット文字列をそのまま返す。
    
    Returns:
        openai_api_key, xai_api_key (str[]): OPENAI_API_KEY, XAI_API_KEY の値（JSON 内に存在する場合）。シークレットが JSON でない場合はシークレット文字列そのもの。
    
    Raises:
        RuntimeError: Secrets Manager からの取得に失敗した場合。
        ValueError: シークレットが JSON でパースでき、かつ OPENAI_API_KEY か XAI_API_KEY が含まれていない場合。
    """
    secret_name = "prod/DBC/APIKeys"
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
        openai_api_key = secret_json.get('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in secret")
        xai_api_key = secret_json.get('XAI_API_KEY')
        if not xai_api_key:
            raise ValueError("XAI_API_KEY not found in secret")
        return openai_api_key, xai_api_key
    except json.JSONDecodeError:
        return secret_string, secret_string

OPENAI_API_KEY, XAI_API_KEY = get_api_keys()
openai_client = OpenAI(api_key=OPENAI_API_KEY)
xai_client = Client(
    api_key=XAI_API_KEY
)

def lambda_handler(event, context):
    """
    API Gateway からのリクエストを受け取り、OpenAI API へプロンプトを渡して生成結果を返す AWS Lambda ハンドラ。
    
    リクエストの body に JSON で含まれる "input" をプロンプトとして使用し、OpenAI のレスポンスから output_text を抽出して HTTP レスポンス形式の辞書を返します。OPTIONS のプリフライトには CORS ヘッダ付きで 200 を返します。入力が不正な JSON か、"input" が空白のみの場合は 400、OpenAI API 呼び出しエラーや期待する出力が得られない場合は 500 を返します。
    
    Parameters:
        event (dict): API Gateway 互換イベント。`event["body"]` は JSON 文字列であり、そこに `"input"` キーでプロンプトを含める必要があります。
        context: Lambda 実行コンテキスト（未使用）。
    
    Returns:
        dict: API Gateway が期待する形式の HTTP レスポンス辞書。成功時は statusCode 200 と
        {"output_text": <string>} を body に持つ JSON を返します。エラー時は適切な statusCode と
        {"error": "<message>"} を body に持つ JSON を返します。
    """
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
    except json.JSONDecodeError:
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
        # response = openai_client.responses.create(
        #     model="gpt-5-nano",
        #     input=prompt,
        #     instructions="あなたはユーザーの友人で、ユーザーと一緒にDLsiteを見ています。"
        # )
        # output_text = getattr(response, 'output_text', None)
        chat = xai_client.chat.create(model="grok-4-1-fast-non-reasoning")
        chat.append(system("あなたはユーザーの友人で、ユーザーと一緒にDLsiteを見ています"))
        chat.append(user(prompt))
        output_text = chat.sample().content
    except (APIError, APIConnectionError, RateLimitError) as e:
        print(f"OpenAI API error: {e}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": "OpenAI API error occurred"}),
        }
    
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

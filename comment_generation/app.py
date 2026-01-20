import json
from openai import OpenAI
import boto3
from botocore.exceptions import ClientError
from openai import APIError, APIConnectionError, RateLimitError
from xai_sdk import Client
from xai_sdk.chat import user, system
import grpc

def get_api_keys():
    """
    AWS Secrets Manager から OpenAI と xAI の API キーを取得する。
    
    指定されたシークレット "prod/DBC/APIKeys"（リージョン ap-northeast-1）を読み取り、シークレット文字列を JSON として解析して `OPENAI_API_KEY` と `XAI_API_KEY` を返す。
    
    Returns:
        (tuple[str, str]): `OPENAI_API_KEY` と `XAI_API_KEY` の値。
    
    Raises:
        RuntimeError: Secrets Manager からの取得に失敗した場合。
        ValueError: シークレットが有効な JSON でない場合、または JSON に `OPENAI_API_KEY` または `XAI_API_KEY` が含まれていない場合。
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
    except json.JSONDecodeError as e:
        raise ValueError("Secret must be JSON with OPENAI_API_KEY and XAI_API_KEY") from e

OPENAI_API_KEY, XAI_API_KEY = get_api_keys()
openai_client = OpenAI(api_key=OPENAI_API_KEY)
xai_client = Client(
    api_key=XAI_API_KEY
)

def call_openai_api(prompt: str, instruction: str):
    """
    OpenAIのテキスト生成APIにプロンプトと指示を渡して生成結果の本文を取得する。
    
    Parameters:
    	prompt (str): 生成のためのユーザー入力プロンプト。
    	instruction (str): モデルへの追加指示や制約を表すテキスト。
    
    Returns:
    	output_text (str | None): 生成された本文文字列。レスポンスに本文が含まれない場合は`None`を返す。
    """
    response = openai_client.responses.create(
        model="gpt-5-nano",
        input=prompt,
        instructions=instruction
    )
    return getattr(response, 'output_text', None)

def call_xai_api(prompt: str, instruction: str):
    """
    xAIチャットモデルにプロンプトとシステム指示を渡して生成されたテキストを取得する。
    
    Parameters:
    	prompt (str): ユーザーとしてモデルに渡す入力テキスト。
    	instruction (str): システムロールとしてモデルに与える指示文。
    
    Returns:
    	str: モデルが生成した応答テキスト。
    """
    chat = xai_client.chat.create(model="grok-4-1-fast-non-reasoning")
    chat.append(system(instruction))
    chat.append(user(prompt))
    return chat.sample().content

def generate_comment(prompt: str, instruction: str, api: str):
    """
    指定されたバックエンドでモデル生成を試み、失敗した場合はもう一方のバックエンドへフォールバックして生成結果を返す。
    
    指定した `api` を優先して呼び出し、優先バックエンドが失敗した場合は代替バックエンドを試行します。どちらのバックエンドでも生成に失敗した場合は `None` を返します。
    
    Parameters:
        prompt (str): 生成に用いるユーザ入力のプロンプト。
        instruction (str): モデルへの追加指示（任意の補助テキスト）。
        api (str): 優先的に使用するバックエンド。'openai' または 'xai' のいずれかを指定する。
    
    Returns:
        output_text (str | None): 生成されたテキスト。生成に成功しなければ `None`。
    """
    output_text = None
    
    if api == 'openai':
        try:
            output_text = call_openai_api(prompt, instruction)
        except (APIError, APIConnectionError, RateLimitError) as e:
            print(f"OpenAI API failed: {e}, falling back to xAI")
            try:
                output_text = call_xai_api(prompt, instruction)
            except grpc.RpcError as e2:
                print(f"xAI API also failed: {e2}")
    else: 
        try:
            output_text = call_xai_api(prompt, instruction)
        except grpc.RpcError as e:
            print(f"xAI API failed: {e}, falling back to OpenAI")
            try:
                output_text = call_openai_api(prompt, instruction)
            except (APIError, APIConnectionError, RateLimitError) as e2:
                print(f"OpenAI API also failed: {e2}")
    
    return output_text

def lambda_handler(event, context):
    """
    API Gateway からのリクエストを受け取り、指定されたバックエンド（OpenAI または xAI）でプロンプトを生成して HTTP レスポンス辞書を返す AWS Lambda ハンドラ。
    
    リクエスト本文（event["body"]）は JSON で、必須の "input" にプロンプト文字列を含めます。任意で "instruction"（追加指示）と "api"（'openai' または 'xai'、デフォルト 'xai'）を指定できます。OPTIONS のプリフライトには CORS ヘッダ付きで 200 を返します。入力が不正な JSON、"input" が空白のみ、または "api" が不正な値の場合は 400 を返します。生成に失敗した場合は 500 を返します。
    
    Parameters:
        event (dict): API Gateway 互換イベント。`event["body"]` に JSON 文字列でリクエストデータを含む。
        context: Lambda 実行コンテキスト（未使用）。
    
    Returns:
        dict: API Gateway 互換の HTTP レスポンス辞書。
          - 成功: statusCode 200、body に JSON 文字列 {"output_text": "<生成結果>"}。
          - クライアントエラー: statusCode 400、body に {"error": "<メッセージ>"}。
          - サーバーエラー: statusCode 500、body に {"error": "<メッセージ>"}。
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
    
    instruction = body.get('instruction', '')
    api = body.get('api', 'xai')

    if api not in ('openai', 'xai'):
        return {
            "statusCode": 400,
            "headers": headers,
            "body": json.dumps({"error": "api must be 'openai' or 'xai'"}),
        }

    output_text = generate_comment(prompt, instruction, api)
    
    if output_text is None:
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": "Comment generation failed"}),
        }

    return {
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps({
            "output_text": output_text,
        }),
    }
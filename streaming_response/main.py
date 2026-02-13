import json
import os
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user


class WorkInfo(BaseModel):
    name: str
    price: Decimal
    official_price: Decimal
    coupon_price: Decimal | None
    price_prefix: str
    price_suffix: str
    genres: list[str]
    description: str


def get_api_keys():
    """
    Secrets Manager の "prod/DBC/APIKeys" から OPENAI_API_KEY と XAI_API_KEY を取得して返す。

    Returns:
        tuple[str, str]: (OPENAI_API_KEY, XAI_API_KEY) の順のタプル。

    Raises:
        RuntimeError: Secrets Manager からの取得に失敗した場合。
        ValueError: シークレットが有効な JSON でない場合、または OPENAI_API_KEY または XAI_API_KEY が存在しない場合。
    """
    secret_name = os.getenv("SECRETS_NAME", "prod/DBC/APIKeys")

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager")

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise RuntimeError(
            f"Failed to retrieve secret from Secrets Manager: {e}"
        ) from e

    secret_string = get_secret_value_response["SecretString"]
    try:
        secret_json = json.loads(secret_string)
        openai_api_key = secret_json.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in secret")
        xai_api_key = secret_json.get("XAI_API_KEY")
        if not xai_api_key:
            raise ValueError("XAI_API_KEY not found in secret")
        return openai_api_key, xai_api_key
    except json.JSONDecodeError as e:
        raise ValueError(
            "Secret must be JSON with OPENAI_API_KEY and XAI_API_KEY"
        ) from e


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://dfjadeefdlmgebcmplicghageakblbop"],
    allow_credentials=True,
    allow_methods=["OPTIONS", "POST"],
    allow_headers=["Content-Type", "X-Amz-Date", "Authorization", "X-Api-Key"],
)

OPENAI_API_KEY, XAI_API_KEY = get_api_keys()

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
xai_client = AsyncClient(api_key=XAI_API_KEY)


async def openai_streamer(request: str, instructions: str):
    """
    OpenAI APIへプロンプトを送信し、レスポンスをストリーミングで受け取ってテキストのチャンクを逐次返す。

    Parameters:
        request (str): モデルへ送信するプロンプト。

    Returns:
        ストリームされたテキストの各チャンク（`str`）。
    """
    stream = await openai_client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "developer", "content": instructions},
            {
                "role": "user",
                "content": request,
            },
        ],
        stream=True,
    )

    async for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta


async def xai_streamer(request: str, instructions: str):
    """
    xAI APIへプロンプトを送信し、レスポンスをストリーミングで受け取ってテキストのチャンクを逐次返す。

    Parameters:
        request (str): モデルへ送信するプロンプト。

    Returns:
        ストリームされたテキストの各チャンク（`str`）。
    """
    chat = xai_client.chat.create(model="grok-4-1-fast-non-reasoning")
    chat.append(system(instructions))
    chat.append(user(request))

    async for _response, chunk in chat.stream():
        yield chunk.content


@app.post("/{request_path:path}")
async def index(request: Request):
    # Get the JSON payload from the POST body
    """
    クライアントからのPOST本文に含まれるJSONの`request`フィールドを読み取り、AIからのレスポンスをテキストストリームとして返すエンドポイント処理を行う。

    Parameters:
        request (Request): JSON ボディを持つ FastAPI のリクエストオブジェクト。ボディはキー `request` を含むJSONである必要がある。

    Returns:
        StreamingResponse: AIからのレスポンスを逐次返すストリームレスポンス（media_type="text/plain"）。

    Raises:
        HTTPException: リクエストボディが有効なJSONでない場合はステータス400で発生する。
        HTTPException: JSON に `request` キーが存在しないか空の場合はステータス400で発生する。
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from e
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Failed to read request body"
        ) from e
    request_param = payload.get("request")
    if not request_param:
        raise HTTPException(status_code=400, detail="'request' is required")
    instructions_param = payload.get("instructions", "")

    # return StreamingResponse(openai_streamer(request_param, instructions_param), media_type="text/plain")
    return StreamingResponse(
        xai_streamer(request_param, instructions_param), media_type="text/plain"
    )

import json
import os
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user


class WorkInfo(BaseModel):
    name: str
    price: Decimal
    official_price: Decimal = Field(alias="officialPrice")
    coupon_price: Decimal | None = Field(alias="couponPrice")
    price_prefix: str = Field(alias="pricePrefix")
    price_suffix: str = Field(alias="pricePrefix")
    genres: list[str]
    description: str


class AskRequest(BaseModel):
    work_info: WorkInfo = Field(alias="workInfo")


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


table_name = os.getenv("TABLE_NAME", "dbc")

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(table_name)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "chrome-extension://dfjadeefdlmgebcmplicghageakblbop",
        "moz-extension://",
    ],
    allow_credentials=True,
    allow_methods=["OPTIONS", "POST"],
    allow_headers=["Content-Type", "X-Amz-Date", "Authorization", "X-Api-Key"],
)

OPENAI_API_KEY, XAI_API_KEY = get_api_keys()

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
xai_client = AsyncClient(api_key=XAI_API_KEY)


def create_comment_prompt(work: WorkInfo):
    if work.coupon_price is not None:
        coupon_line = (
            f"\nクーポン価格: {work.price_prefix}{work.coupon_price}{work.price_suffix}"
        )
    else:
        coupon_line = ""

    return (
        "以下の作品情報をもとに、作品に対して短いコメントをしてください。\n"
        "出力はコメントの本文のみにしてください。\n\n"
        f"タイトル: {work.name}\n"
        f"価格: {work.price_prefix}{work.price}{work.price_suffix}"
        f"（サークル設定価格: {work.price_prefix}{work.official_price}{work.price_suffix}）"
        f"{coupon_line}\n"
        f"ジャンル: {', '.join(work.genres)}\n"
        "作品内容:\n"
        f"{work.description}"
    )


async def openai_streamer(prompt: str, instructions: str):
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
                "content": prompt,
            },
        ],
        stream=True,
    )

    async for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta


async def xai_streamer(prompt: str, instructions: str):
    """
    xAI APIへプロンプトを送信し、レスポンスをストリーミングで受け取ってテキストのチャンクを逐次返す。

    Parameters:
        request (str): モデルへ送信するプロンプト。

    Returns:
        ストリームされたテキストの各チャンク（`str`）。
    """
    chat = xai_client.chat.create(
        model="grok-4-1-fast-non-reasoning", messages=[system(instructions)]
    )
    chat.append(user(prompt))

    async for _response, chunk in chat.stream():
        yield chunk.content


def get_dynamodb_item(pk: str, sk: str, attribute: str):
    try:
        response = table.get_item(
            Key={
                "PK": pk,
                "SK": sk,
            }
        )
    except ClientError as e:
        raise RuntimeError(f"DynamoDB get_item failed: {e}") from e

    item = response.get("Item")
    if not item or attribute not in item:
        raise ValueError(f"{pk} not found for {sk}")

    return item.get(attribute)


@app.post("/{request_path:path}")
async def index(body: AskRequest):
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
    work_info = body.work_info
    prompt = create_comment_prompt(work_info)

    instructions = get_dynamodb_item("instructions", "default", "text")

    # return StreamingResponse(openai_streamer(prompt, instructions), media_type="text/plain")
    return StreamingResponse(
        xai_streamer(prompt, instructions), media_type="text/plain"
    )

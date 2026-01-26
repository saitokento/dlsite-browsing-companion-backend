import json

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from strands import Agent
from xai_sdk import Client
from xai_sdk.chat import user


def get_api_keys():
    """
    AWS Secrets Manager から OpenAI と xAI の API キーを取得する。

    指定されたシークレット "prod/DBC/APIKeys" (リージョン ap-northeast-1) を読み取り、シークレット文字列を JSON として解析して `OPENAI_API_KEY` と `XAI_API_KEY` を返す。

    Returns:
        (tuple[str, str]): `OPENAI_API_KEY` と `XAI_API_KEY` の値。

    Raises:
        RuntimeError: Secrets Manager からの取得に失敗した場合。
        ValueError: シークレットが有効な JSON でない場合、または JSON に `OPENAI_API_KEY` または `XAI_API_KEY` が含まれていない場合。
    """
    secret_name = "prod/DBC/APIKeys"
    region_name = "ap-northeast-1"

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

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

OPENAI_API_KEY, XAI_API_KEY = get_api_keys()

strands_agent = Agent(
    model="amazon.nova-lite-v1:0",
    system_prompt="You are a helpful AI assistant.",
)

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
xai_client = Client(api_key=XAI_API_KEY)


# Based on https://aws.amazon.com/blogs/opensource/introducing-strands-agents-1-0-production-ready-multi-agent-orchestration-made-simple/
async def streamer(request: str):
    async for event in strands_agent.stream_async(request):
        if "data" in event:
            yield event["data"]


async def openai_streamer(request: str):
    stream = await openai_client.responses.create(
        model="gpt-5-nano",
        input=[
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


async def xai_streamer(request: str):
    chat = xai_client.chat.create(model="grok-4-1-fast-non-reasoning")
    chat.append(user(request))

    for response, chunk in chat.stream():
        yield chunk.content


@app.get("/{request_path:path}")
async def catch_all(request: Request, request_path: str):
    # Catch-all route to handle all GET requests
    return


@app.post("/{request_path:path}")
async def index(request: Request):
    # Get the JSON payload from the POST body
    body = await request.body()
    payload = json.loads(body.decode("utf-8"))
    request_param = payload.get("request")

    # return StreamingResponse(streamer(request_param))
    # return StreamingResponse(openai_streamer(request_param))
    return StreamingResponse(xai_streamer(request_param))

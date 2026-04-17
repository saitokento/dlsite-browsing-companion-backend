import json
import logging
import os
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Union

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user


class Usecase(StrEnum):
    WORK = "work"
    # OTHER = "other"


class Work(BaseModel):
    name: str
    price: Decimal
    official_price: Decimal = Field(alias="officialPrice")
    coupon_price: Decimal | None = Field(default=None, alias="couponPrice")
    price_prefix: str = Field(alias="pricePrefix")
    price_suffix: str = Field(alias="priceSuffix")
    genres: list[str]
    description: str


# class OtherProperty(BaseModel):
#     ...


class WorkPayload(BaseModel):
    work: Work


# class OtherPayload(BaseModel):
#     other_property: OtherProperty


class WorkRequest(BaseModel):
    usecase: Literal[Usecase.WORK]
    payload: WorkPayload


# class OtherRequest(BaseModel):
#     usecase: Literal[Usecase.Other]
#     payload: OtherPayload


AskRequest = Annotated[
    Union[WorkRequest],  # Union[WorkRequest, OtherRequest],
    Field(discriminator="usecase"),
]


def get_api_keys():
    secret_name = os.getenv("SECRETS_NAME", "prod/DBC/APIKeys")

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager")

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            raise RuntimeError(
                f"The requested secret {secret_name} was not found"
            ) from e
        elif e.response["Error"]["Code"] == "InvalidRequestException":
            raise RuntimeError(
                f"The request was invalid for secret '{secret_name}': {e}"
            ) from e
        elif e.response["Error"]["Code"] == "InvalidParameterException":
            raise RuntimeError(
                f"The request had invalid params for secret '{secret_name}': {e}"
            ) from e
        elif e.response["Error"]["Code"] == "DecryptionFailure":
            raise RuntimeError(
                f"The requested secret '{secret_name}' can't be decrypted using the provided KMS key: {e}"
            ) from e
        elif e.response["Error"]["Code"] == "InternalServiceError":
            raise RuntimeError(
                f"An error occurred on the service side for secret '{secret_name}': {e}"
            ) from e
        else:
            raise RuntimeError(
                f"An unexpected error occurred while retrieving the requested secret {secret_name}: {e}"
            ) from e

    if "SecretString" in get_secret_value_response:
        secret_data = get_secret_value_response["SecretString"]
    else:
        raise ValueError(
            f'The requested secret {secret_name} does not contain "SecretString"'
        )

    try:
        secret_json = json.loads(secret_data)
        # openai_api_key = secret_json.get("OPENAI_API_KEY")
        # if not openai_api_key:
        #     raise ValueError(
        #         f'The requested secret {secret_name} does not contain "OPENAI_API_KEY"'
        #     )
        xai_api_key = secret_json.get("XAI_API_KEY")
        if not xai_api_key:
            raise ValueError(
                f'The requested secret {secret_name} does not contain "XAI_API_KEY"'
            )
        # return openai_api_key, xai_api_key
        return xai_api_key
    except json.JSONDecodeError as e:
        raise ValueError(
            f"The requested secret {secret_name} must be valid JSON"
        ) from e


logger = logging.getLogger(__name__)

table_name = os.getenv("TABLE_NAME", "dbc")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(table_name)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "chrome-extension://*",
        "moz-extension://*",
    ],
    allow_credentials=False,
    allow_methods=["OPTIONS", "POST"],
    allow_headers=["Content-Type", "X-Amz-Date", "X-Api-Key"],
)

# OPENAI_API_KEY, XAI_API_KEY = get_api_keys()
XAI_API_KEY = get_api_keys()

# openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
xai_client = AsyncClient(api_key=XAI_API_KEY)


# async def openai_streamer(prompt: str, instructions: str):
#     stream = await openai_client.responses.create(
#         model="gpt-5-nano",
#         input=[
#             {"role": "developer", "content": instructions},
#             {
#                 "role": "user",
#                 "content": prompt,
#             },
#         ],
#         stream=True,
#     )
#
#     async for event in stream:
#         if event.type == "response.output_text.delta":
#             yield event.delta


async def xai_streamer(prompt: str, instructions: str):
    chat = xai_client.chat.create(
        model="grok-4-1-fast-non-reasoning", messages=[system(instructions)]
    )
    chat.append(user(prompt))

    async for _response, chunk in chat.stream():
        yield chunk.content


def get_character_item(character_id: str):
    try:
        response = table.get_item(Key={"character_id": character_id})
    except ClientError as err:
        logger.error(
            "Couldn't get character %s from table %s. Here's why: %s: %s",
            character_id,
            table.name,
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise
    else:
        item = response.get("Item")
        if item is None:
            raise ValueError(f"Character not found: {character_id}")
        return item


def create_prompt(character_item, usecase, payload):
    prompts = character_item.get("prompts") or {}
    match usecase:
        case Usecase.WORK:
            prompt_template = prompts.get("work")
            if not prompt_template:
                raise ValueError("Character item missing 'prompts.work'")
            if payload.work.coupon_price is not None:
                coupon_line = f"\nクーポン価格: {payload.work.price_prefix}{payload.work.coupon_price}{payload.work.price_suffix}"
            else:
                coupon_line = ""

            work_genres = ", ".join(payload.work.genres)

            return prompt_template.format(
                work_name=payload.work.name,
                work_price_prefix=payload.work.price_prefix,
                work_price=payload.work.price,
                work_price_suffix=payload.work.price_suffix,
                work_official_price=payload.work.official_price,
                coupon_line=coupon_line,
                work_genres=work_genres,
                work_description=payload.work.description,
            )
        # case Usecase.OTHER:
        #     ...


@app.post("/ask")
async def index(body: AskRequest):
    character_id = "default"
    character_item = get_character_item(character_id)

    prompt = create_prompt(character_item, body.usecase, body.payload)
    instructions = character_item.get("instructions")
    if not instructions:
        raise ValueError("Character item missing 'instructions'")

    # return StreamingResponse(
    #     openai_streamer(prompt, instructions), media_type="text/plain"
    # )
    return StreamingResponse(
        xai_streamer(prompt, instructions), media_type="text/plain"
    )

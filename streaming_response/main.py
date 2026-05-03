import json
import logging
import os
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Union

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user


class ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class CharacterId(StrEnum):
    DEFAULT = "default"


class Usecase(StrEnum):
    WORK = "work"
    HOME_HELLO = "home:hello"
    CIRCLE_NEW = "circle:new"
    USERBUY_PAGE1 = "userbuy:page1"
    CART_LIST = "cart:list"
    DOWNLOAD_LIST = "download:list"


class Work(ApiModel):
    name: str
    price: Decimal
    official_price: Decimal = Field(alias="officialPrice")
    coupon_price: Decimal | None = Field(default=None, alias="couponPrice")
    price_prefix: str = Field(alias="pricePrefix")
    price_suffix: str = Field(alias="priceSuffix")
    genres: list[str]
    description: str


class CircleWork(ApiModel):
    product_id: str = Field(alias="productId")
    category: str
    name: str
    author: str | None
    price: Decimal
    official_price: Decimal = Field(alias="officialPrice")
    price_prefix: str = Field(alias="pricePrefix")
    price_suffix: str = Field(alias="priceSuffix")
    label: str | None


class UserbuyWork(ApiModel):
    product_id: str = Field(alias="productId")
    buy_date: str = Field(alias="buyDate")
    name: str
    maker_name: str = Field(alias="makerName")
    genres: list[str]
    price: Decimal
    price_prefix: str = Field(alias="pricePrefix")
    price_suffix: str = Field(alias="priceSuffix")


class CartWork(ApiModel):
    product_id: str = Field(alias="productId")
    name: str
    maker_name: str = Field(alias="makerName")
    category: str
    price: Decimal
    official_price: Decimal = Field(alias="officialPrice")


class DownloadWork(ApiModel):
    product_id: str = Field(alias="productId")
    name: str
    maker_name: str = Field(alias="makerName")
    genre: str


class WorkPayload(ApiModel):
    work: Work


class CircleNewPayload(ApiModel):
    maker_name: str = Field(alias="makerName")
    circle_work_list: list[CircleWork] = Field(alias="circleWorkList")


class UserbuyPage1Payload(ApiModel):
    userbuy_work_list: list[UserbuyWork] = Field(alias="userbuyWorkList")


class CartListPayload(ApiModel):
    cart_work_list: list[CartWork] = Field(alias="cartWorkList")
    total_discount: Decimal = Field(alias="totalDiscount")
    total_original: Decimal | None = Field(alias="totalOriginal")
    coupon_name: str | None = Field(alias="couponName")
    total_coupon: Decimal | None = Field(alias="totalCoupon")
    price_prefix: str = Field(alias="pricePrefix")
    price_suffix: str = Field(alias="priceSuffix")


class DownloadListPayload(ApiModel):
    download_work_list: list[DownloadWork] = Field(alias="downloadWorkList")


class EmptyPayload(ApiModel):
    pass


class WorkRequest(ApiModel):
    character_id: CharacterId = Field(alias="characterId")
    usecase: Literal[Usecase.WORK]
    payload: WorkPayload


class HomeHelloRequest(ApiModel):
    character_id: CharacterId = Field(alias="characterId")
    usecase: Literal[Usecase.HOME_HELLO]
    payload: EmptyPayload


class CircleNewRequest(ApiModel):
    character_id: CharacterId = Field(alias="characterId")
    usecase: Literal[Usecase.CIRCLE_NEW]
    payload: CircleNewPayload


class UserbuyPage1Request(ApiModel):
    character_id: CharacterId = Field(alias="characterId")
    usecase: Literal[Usecase.USERBUY_PAGE1]
    payload: UserbuyPage1Payload


class CartListRequest(ApiModel):
    character_id: CharacterId = Field(alias="characterId")
    usecase: Literal[Usecase.CART_LIST]
    payload: CartListPayload


class DownloadListRequest(ApiModel):
    character_id: CharacterId = Field(alias="characterId")
    usecase: Literal[Usecase.DOWNLOAD_LIST]
    payload: DownloadListPayload


AskRequest = Annotated[
    Union[
        WorkRequest,
        HomeHelloRequest,
        CircleNewRequest,
        UserbuyPage1Request,
        CartListRequest,
        DownloadListRequest,
    ],
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
    allow_origin_regex=r"^(chrome-extension|moz-extension)://.*$",
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
        return item


def create_prompt(character_item, usecase, payload):
    prompts = character_item.get("prompts") or {}
    match usecase:
        case Usecase.WORK:
            prompt_template = prompts.get("work")
            if not prompt_template:
                raise HTTPException(
                    status_code=500,
                    detail=f"'prompts.work' not found for character '{character_item.get('character_id')}' in DynamoDB",
                )
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
    character_id = body.character_id
    character_item = get_character_item(character_id)
    if character_item is None:
        if character_id != CharacterId.DEFAULT:
            logger.warning(
                "Character '%s' not found, falling back to 'default'",
                character_id,
            )
            character_id = CharacterId.DEFAULT
            character_item = get_character_item(character_id)
        if character_item is None:
            raise HTTPException(
                status_code=500, detail="'default' character not found in DynamoDB"
            )

    prompt = create_prompt(character_item, body.usecase, body.payload)
    instructions = character_item.get("instructions")
    if not instructions:
        raise HTTPException(
            status_code=500,
            detail=f"'instructions' not found for character '{character_id}' in DynamoDB",
        )

    # return StreamingResponse(
    #     openai_streamer(prompt, instructions), media_type="text/plain"
    # )
    return StreamingResponse(
        xai_streamer(prompt, instructions), media_type="text/plain"
    )

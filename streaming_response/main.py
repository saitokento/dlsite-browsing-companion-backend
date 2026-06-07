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
from grpc import StatusCode
from grpc.aio import AioRpcError
from pydantic import BaseModel, ConfigDict, Field
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user


class ApiModel(BaseModel):
    """APIで利用するPydanticモデルの共通基底クラス

    （フィールド名のエイリアスによる入力を許可、未定義フィールドを拒否）
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ChatRequestBase(ApiModel):
    """チャットリクエストに共通するフィールドを表すモデル"""

    character_id: str = Field(alias="characterId")
    previous_response_id: str | None = Field(
        default=None,
        alias="previousResponseId",
    )


class Usecase(StrEnum):
    """プロンプト生成処理を切り替えるusecase"""

    WORK = "work"
    HOME_HELLO = "home:hello"
    CIRCLE_NEW = "circle:new"
    USERBUY_PAGE1 = "userbuy:page1"
    CART_LIST = "cart:list"
    DOWNLOAD_LIST = "download:list"


class Work(ApiModel):
    """作品ページの作品情報"""

    name: str
    maker_name: str = Field(alias="makerName")
    price: Decimal
    official_price: Decimal = Field(alias="officialPrice")
    coupon_price: Decimal | None = Field(default=None, alias="couponPrice")
    price_prefix: str = Field(alias="pricePrefix")
    price_suffix: str = Field(alias="priceSuffix")
    genres: list[str]
    description: str


class CircleWork(ApiModel):
    """サークルページの販売中作品の作品情報"""

    product_id: str = Field(alias="productId")
    category: str
    name: str
    author: str | None
    price: Decimal
    official_price: Decimal = Field(alias="officialPrice")
    price_prefix: str = Field(alias="pricePrefix")
    price_suffix: str = Field(alias="priceSuffix")
    labels: list[str]


class CircleAnnounceWork(ApiModel):
    """サークルページの発売予告作品の作品情報"""

    product_id: str = Field(alias="productId")
    name: str
    author: str | None
    category: str
    expected_date: str = Field(alias="expectedDate")
    free_sample: bool = Field(alias="freeSample")


class UserbuyWork(ApiModel):
    """購入履歴の作品情報"""

    product_id: str = Field(alias="productId")
    buy_date: str = Field(alias="buyDate")
    name: str
    maker_name: str = Field(alias="makerName")
    genres: list[str]
    price: Decimal
    price_prefix: str = Field(alias="pricePrefix")
    price_suffix: str = Field(alias="priceSuffix")


class CartWork(ApiModel):
    """カート内の作品情報"""

    product_id: str = Field(alias="productId")
    name: str
    maker_name: str = Field(alias="makerName")
    category: str
    price: Decimal
    official_price: Decimal = Field(alias="officialPrice")


class DownloadWork(ApiModel):
    """購入後ページの作品情報。"""

    product_id: str = Field(alias="productId")
    name: str
    maker_name: str = Field(alias="makerName")
    genre: str


class HomeHelloPayload(ApiModel):
    """トップページでのコメント生成に使用するペイロード"""

    floor: str


class WorkPayload(ApiModel):
    """作品ページでのコメント生成に使用するペイロード"""

    work: Work


class CircleNewPayload(ApiModel):
    """サークルページでのコメント生成に使用するペイロード"""

    maker_name: str = Field(alias="makerName")
    circle_announce_work_list: list[CircleAnnounceWork] = Field(
        alias="circleAnnounceWorkList", max_length=100
    )
    circle_work_list: list[CircleWork] = Field(alias="circleWorkList", max_length=100)


class UserbuyPage1Payload(ApiModel):
    """購入履歴ページでのコメント生成に使用するペイロード"""

    userbuy_work_list: list[UserbuyWork] = Field(
        alias="userbuyWorkList",
        max_length=100,  # 拡張機能側での購入履歴管理を実装したら増やすことも検討
    )


class CartListPayload(ApiModel):
    """カートページでのコメント生成に使用するペイロード"""

    cart_work_list: list[CartWork] = Field(alias="cartWorkList", max_length=100)
    total_discount: Decimal = Field(alias="totalDiscount")
    total_original: Decimal | None = Field(alias="totalOriginal")
    coupon_name: str | None = Field(alias="couponName")
    total_coupon: Decimal | None = Field(alias="totalCoupon")
    price_prefix: str = Field(alias="pricePrefix")
    price_suffix: str = Field(alias="priceSuffix")


class DownloadListPayload(ApiModel):
    """購入後ページでのコメント生成に使用するペイロード"""

    download_work_list: list[DownloadWork] = Field(
        alias="downloadWorkList", max_length=100
    )


# class EmptyPayload(ApiModel):
#     pass


class WorkRequest(ChatRequestBase):
    """作品ページ用のチャットリクエスト"""

    usecase: Literal[Usecase.WORK]
    payload: WorkPayload


class HomeHelloRequest(ChatRequestBase):
    """トップページ用のチャットリクエスト"""

    usecase: Literal[Usecase.HOME_HELLO]
    payload: HomeHelloPayload


class CircleNewRequest(ChatRequestBase):
    """サークルページ用のチャットリクエスト"""

    usecase: Literal[Usecase.CIRCLE_NEW]
    payload: CircleNewPayload


class UserbuyPage1Request(ChatRequestBase):
    """購入履歴ページ用のチャットリクエスト"""

    usecase: Literal[Usecase.USERBUY_PAGE1]
    payload: UserbuyPage1Payload


class CartListRequest(ChatRequestBase):
    """カートページ用のチャットリクエスト"""

    usecase: Literal[Usecase.CART_LIST]
    payload: CartListPayload


class DownloadListRequest(ChatRequestBase):
    """購入後ページ用のチャットリクエスト"""

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
    """AWS Secrets ManagerからxAI APIキーを取得する

    環境変数``SECRETS_NAME``で取得対象のシークレット名を指定できる
    未指定の場合は``prod/DBC/APIKeys``を使用する

    Returns:
        str: シークレット内の``XAI_API_KEY``

    Raises:
        RuntimeError: Secrets Managerからシークレットを取得できなかった場合
        ValueError: シークレットが文字列形式でない、JSONとして不正、または``XAI_API_KEY``が含まれていない場合
    """

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


async def _stream_xai_chat_once(
    prompt: str, instructions: str, previous_response_id: str | None
):
    if previous_response_id is not None:
        chat = xai_client.chat.create(
            model="grok-4-1-fast-non-reasoning",
            previous_response_id=previous_response_id,
            store_messages=True,
        )
    else:
        chat = xai_client.chat.create(
            model="grok-4-1-fast-non-reasoning",
            messages=[system(instructions)],
            store_messages=True,
        )

    chat.append(user(prompt))

    response_id: str | None = None

    async for response, chunk in chat.stream():
        current_response_id = getattr(response, "id", None)
        if isinstance(current_response_id, str):
            response_id = current_response_id

        if not chunk.content:
            continue

        yield (
            json.dumps(
                {
                    "type": "delta",
                    "text": chunk.content,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    yield (
        json.dumps(
            {
                "type": "done",
                "responseId": response_id,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


async def xai_streamer(
    prompt: str,
    instructions: str,
    previous_response_id: str | None,
):
    """xAIのチャット生成を実行し、応答をNDJSON形式でストリーミングする

    前回のレスポンスIDが見つからない場合は新規チャットを作成する

    Args:
        prompt: 送信するプロンプト
        instructions: 新規チャット時に設定するシステム指示
        previous_response_id: 前回のレスポンスID。新規チャットでは``None``

    Yields:
        str: テキスト差分を表す``delta``行と、完了時の``done``行

    Raises:
        AioRpcError: 前回のレスポンスIDが見つからない以外のgRPCエラーが発生した場合。
    """

    try:
        async for line in _stream_xai_chat_once(
            prompt,
            instructions,
            previous_response_id,
        ):
            yield line

    except AioRpcError as err:
        if err.code() == StatusCode.NOT_FOUND and previous_response_id is not None:
            logger.warning(
                "previous_response_id %s was not found. Starting a new chat.",
                previous_response_id,
            )

            async for line in _stream_xai_chat_once(
                prompt,
                instructions,
                None,
            ):
                yield line
            return

        raise


def get_character_item(character_id: str):
    """DynamoDBから指定されたキャラクターのシステム指示・プロンプトテンプレートを取得する

    Args:
        character_id: 取得するキャラクターのID

    Returns:
        dict | None: キャラクターのシステム指示・プロンプトテンプレート。該当項目がない場合は``None``

    Raises:
        ClientError: DynamoDBへのアクセスに失敗した場合
    """

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


def get_prompt_template(
    prompts,
    key: str,
    character_item,
    default_prompts=None,
):
    """指定キーのプロンプトテンプレートを取得する

    キャラクター固有のテンプレートがない場合は、"default"のテンプレートへフォールバックする

    Args:
        prompts: キャラクター固有のプロンプトテンプレート一覧
        key: 取得するテンプレートのキー
        character_item: キャラクターのシステム指示・プロンプトテンプレート
        default_prompts: フォールバック先の"default"のプロンプトテンプレート

    Returns:
        str: 見つかったプロンプトテンプレート

    Raises:
        HTTPException: キャラクター固有・"default"のどちらにもテンプレートがない場合
    """

    prompt_template = prompts.get(key)

    if not prompt_template and default_prompts is not None:
        prompt_template = default_prompts.get(key)

    if not prompt_template:
        raise HTTPException(
            status_code=500,
            detail=f"'prompts.{key}' not found for character '{character_item.get('character_id')}' or default character in DynamoDB",
        )

    return prompt_template


def quote_markdown(markdown: str) -> str:
    """Markdownテキストの各行を引用形式へ変換する

    Args:
        markdown: 引用形式へ変換するMarkdown文字列

    Returns:
        str: 各行の先頭に``> ``を付けた文字列。
    """

    return "\n".join(f"> {line}" for line in markdown.splitlines())


def format_labels(labels: list[str]):
    """ラベル一覧を読点区切りの表示用文字列へ整形する

    Args:
        labels: 整形対象のラベル一覧

    Returns:
        str: 先頭に読点を付けたラベル文字列。空の一覧では空文字列
    """

    if not labels:
        return ""

    return "、" + "、".join(labels)


def create_prompt(character_item, usecase, payload, default_prompts=None):
    """usecaseとペイロードからAIへ渡すプロンプトを作成する

    Args:
        character_item: 使用するシステム指示・プロンプトテンプレート
        usecase: プロンプトの種類を示すusecase
        payload: usecaseに対応する入力データ
        default_prompts: フォールバック先の"default"のプロンプトテンプレート

    Returns:
        str: テンプレートに入力データを埋め込んだプロンプト

    Raises:
        HTTPException: 必要なプロンプトテンプレートが見つからない場合
    """

    prompts = character_item.get("prompts") or {}
    match usecase:
        case Usecase.WORK:
            prompt_template = get_prompt_template(
                prompts, "work", character_item, default_prompts
            )

            if payload.work.coupon_price is not None:
                coupon_line = f"\nクーポン利用価格: {payload.work.price_prefix}{payload.work.coupon_price}{payload.work.price_suffix}"
            else:
                coupon_line = ""

            work_genres = ", ".join(payload.work.genres)

            return prompt_template.format(
                work_name=payload.work.name,
                maker_name=payload.work.maker_name,
                work_price_prefix=payload.work.price_prefix,
                work_price=payload.work.price,
                work_price_suffix=payload.work.price_suffix,
                work_official_price=payload.work.official_price,
                coupon_line=coupon_line,
                work_genres=work_genres,
                work_description=quote_markdown(payload.work.description),
            )

        case Usecase.HOME_HELLO:
            prompt_template = get_prompt_template(
                prompts, "home:hello", character_item, default_prompts
            )

            return prompt_template.format(floor=payload.floor)

        case Usecase.CIRCLE_NEW:
            prompt_template = get_prompt_template(
                prompts, "circle:new", character_item, default_prompts
            )
            announce_work_list_template = get_prompt_template(
                prompts,
                "circle:new:announce_work_list",
                character_item,
                default_prompts,
            )
            work_list_template = get_prompt_template(
                prompts, "circle:new:work_list", character_item, default_prompts
            )

            announce_work_list = "\n\n".join(
                announce_work_list_template.format(
                    name=work.name,
                    author_line=f"\nクリエイター（シナリオ、イラスト、声優など）：{work.author}"
                    if work.author
                    else "",
                    category=work.category,
                    expected_date=work.expected_date,
                    free_sample="\n（無料サンプルあり）" if work.free_sample else "",
                )
                for work in payload.circle_announce_work_list
            )

            if payload.circle_announce_work_list:
                announce_line = (
                    f"---\n\n発売予告作品\n\n{announce_work_list}\n\n---\n\n"
                )
            else:
                announce_line = ""

            work_list = "\n\n".join(
                work_list_template.format(
                    name=work.name,
                    author_line=f"\nクリエイター（シナリオ、イラスト、声優など）：{work.author}"
                    if work.author
                    else "",
                    category=work.category,
                    price_prefix=work.price_prefix,
                    price=work.price,
                    price_suffix=work.price_suffix,
                    official_price=work.official_price,
                    labels=format_labels(work.labels),
                )
                for work in payload.circle_work_list
            )

            return prompt_template.format(
                maker_name=payload.maker_name,
                announce_line=announce_line,
                work_list=work_list,
            )

        case Usecase.USERBUY_PAGE1:
            prompt_template = get_prompt_template(
                prompts, "userbuy:page1", character_item, default_prompts
            )
            work_list_template = get_prompt_template(
                prompts, "userbuy:page1:work_list", character_item, default_prompts
            )

            work_list = "\n\n".join(
                work_list_template.format(
                    buy_date=work.buy_date,
                    name=work.name,
                    maker_name=work.maker_name,
                    genres=", ".join(work.genres),
                    price_prefix=work.price_prefix,
                    price=work.price,
                    price_suffix=work.price_suffix,
                )
                for work in payload.userbuy_work_list
            )

            return prompt_template.format(work_list=work_list)

        case Usecase.CART_LIST:
            prompt_template = get_prompt_template(
                prompts, "cart:list", character_item, default_prompts
            )
            work_list_template = get_prompt_template(
                prompts, "cart:list:work_list", character_item, default_prompts
            )

            work_list = "\n\n".join(
                work_list_template.format(
                    name=work.name,
                    maker_name=work.maker_name,
                    category=work.category,
                    price_prefix=payload.price_prefix,
                    price=work.price,
                    price_suffix=payload.price_suffix,
                    official_price=work.official_price,
                )
                for work in payload.cart_work_list
            )

            if payload.coupon_name is not None and payload.total_coupon is not None:
                coupon_line = f"\n利用可能なクーポン：{payload.coupon_name}\nクーポン利用価格：{payload.price_prefix}{payload.total_coupon}{payload.price_suffix}"
            else:
                coupon_line = ""

            return prompt_template.format(
                work_list=work_list,
                price_prefix=payload.price_prefix,
                total_discount=payload.total_discount,
                price_suffix=payload.price_suffix,
                coupon_line=coupon_line,
            )

        case Usecase.DOWNLOAD_LIST:
            prompt_template = get_prompt_template(
                prompts, "download:list", character_item, default_prompts
            )
            work_list_template = get_prompt_template(
                prompts, "download:list:work_list", character_item, default_prompts
            )

            work_list = "\n\n".join(
                work_list_template.format(
                    name=work.name,
                    maker_name=work.maker_name,
                    genre=work.genre,
                )
                for work in payload.download_work_list
            )

            return prompt_template.format(work_list=work_list)


@app.post("/ask")
async def index(body: AskRequest):
    """コメント生成リクエストを受け付け、AIの応答をストリーミングで返す

    指定されたキャラクターが存在しない場合は``default``へフォールバックし、
    リクエストのusecaseに応じたプロンプトを生成する

    Args:
        body: usecaseで判別されるチャットリクエスト

    Returns:
        StreamingResponse: ``application/x-ndjson``形式のストリーミングレスポンス

    Raises:
        HTTPException: "default"キャラクター、システム指示、または必要なプロンプトテンプレートが
            DynamoDBに存在しない場合。
    """

    character_id = body.character_id
    character_item = get_character_item(character_id)
    if character_item is None:
        if character_id != "default":
            logger.warning(
                "Character '%s' not found, falling back to 'default'",
                character_id,
            )
            character_id = "default"
            character_item = get_character_item(character_id)
        if character_item is None:
            raise HTTPException(
                status_code=500, detail="'default' character not found in DynamoDB"
            )

    default_prompts = None
    if character_id != "default":
        default_character_item = get_character_item("default")
        if default_character_item is None:
            raise HTTPException(
                status_code=500,
                detail="'default' character not found in DynamoDB",
            )
        default_prompts = default_character_item.get("prompts") or {}

    prompt = create_prompt(
        character_item,
        body.usecase,
        body.payload,
        default_prompts,
    )

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
        xai_streamer(
            prompt,
            instructions,
            body.previous_response_id,
        ),
        media_type="application/x-ndjson",
    )

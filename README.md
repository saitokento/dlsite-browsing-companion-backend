# DLsite Browsing Companion (Backend)

（フロントエンド：[DLsite Browsing Companion](https://github.com/saitokento/dlsite-browsing-companion)）

DLsite Browsing Companionから送信されたページ情報をもとに、AIキャラクターのコメントを生成してストリーミングレスポンスで返すバックエンドAPIです。

## バックエンドの役割

- DLsite Browsing Companionから送信されたリクエストの受け取り
- Secrets Managerを利用した外部APIキーの管理
- DynamoDBからキャラクターのシステム指示・プロンプトテンプレートを取得
- ページ情報をもとにしたプロンプトの作成
- xAI APIを利用したコメント生成・会話履歴の管理
- 生成結果のストリーミングレスポンス

## 使用技術

- Python 3.14
- FastAPI
- Uvicorn
- Pydantic
- xAI SDK
- boto3
- AWS Lambda
- AWS Lambda Web Adapter
- Amazon API Gateway
- Amazon DynamoDB
- AWS Secrets Manager
- Amazon CloudWatch
- AWS SAM
- OpenAPI

## 工夫した点

- 外部APIキーはSecrets Managerで管理するようにした
- キャラクターごとのシステム指示とプロンプトテンプレートはDynamoDB上で管理し、コードを変えずに内容を更新できるようにした
- NDJSON形式のストリーミングレスポンスに対応した
- Pydantic・OpenAPIを使用して不正な形式のリクエストを受け付けないようにした

## 今後の展望

- ユーザー認証・アクセス制限など、セキュリティ面の改善
- テストの追加

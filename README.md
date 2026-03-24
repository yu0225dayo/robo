# Project

ローカルPC と計算機（GPU）をHTTPS通信で連携するプロジェクトです。

## 構成

```
project/
├── server/   # 計算機（GPU環境）側
│   └── server.py
└── client/   # ローカルPC側
    └── client.py
```

## 概要

- **client**: ローカルPCからデータを送信
- **server**: 計算機でGPU処理を行い結果を返却

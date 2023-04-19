# Japanese Sentiment Analysis API using Huggingface model

## Introduction

This repo demonstrates an API server that serves Japanese sentiment analysis by using Huggingface model, https://huggingface.co/jarvisx17/japanese-sentiment-analysis

The server returns Positive and Negative values from 0 to 1 individually.

## Request
```json
{
    "text":"日本語のテキストを解析したいです。"
}
```

The `curl` example is as below;
```shell
curl --location --request POST '127.0.0.1:5000/sentiment-analysis' \
--header 'Content-Type: application/json' \
--data-raw '{
"text":"日本語のテキストを解析したいです。"
}'
```

## Response
```json
{
    "label": 0,
    "score": [
        0.995884120464325,
        0.004115856718271971
    ]
}
```

The first one displays the **Negative** degree from 0 to 1, while the last one indicates **Positive** degree. Please note that the first element indicates the **Negative** in this model.
The label `0` indicates a negative sentiment, while `1` indicates a positive sentiment. However, it is recommended to interpret the output based on the degrees provided.

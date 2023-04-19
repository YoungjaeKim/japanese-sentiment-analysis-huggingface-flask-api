# Japanese Sentiment Analysis

## Introduction

This repo demonstrates simple Japanese sentiment analysis by using Huggingface model; https://huggingface.co/jarvisx17/japanese-sentiment-analysis
It wraps model to Flask api server. The server returns results as Positive and Negative from 0 to 1 independently.

## Request
```json
{
    "text":"日本語のテキストを解析したいです。"
}
```

`curl` example is as below;
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

The first one shows **Negative** degree from 0 to 1, and the last one gives **Positive** degree. Remember, the first one means a degree of Negative in this model.
The label `0` means it's Negative, `1` means Positive. But I recommend to find adequate output with given degrees.

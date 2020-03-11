# semantic-segmentation

## 概要

semantic segmentation を行うスクリプトを作成する。

## 環境構築

Docker 環境でスクリプトを動作させるため、イメージを build し、コンテナを立ち上げる。

```shell
docker build -t ss:v1 docker/
docker run --gpus all -it --rm --shm-size=4g --name ss -v {本リポジトリの絶対パス}:/work ss:v1
```

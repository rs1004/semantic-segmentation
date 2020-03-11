# semantic-segmentation

## 概要

semantic segmentation を行うスクリプトを作成する。

## データ準備
データは Cityscapes Dataset を使用する。
URL: https://www.cityscapes-dataset.com/
事前に会員登録を済ませ、以下の手順でデータのダウンロードを行う。

ダウンロードコマンドは以下の通り。
```shell
$ wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username={ユーザー名}&password={パスワード}&submit=Login' https://www.cityscapes-dataset.com/login/
$ wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
$ wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

これにより、
- gtFine_trainvaltest.zip
- leftImg8bit_trainvaltest.zip
が手元にダウンロードされる。

{本リポジトリの絶対パス}直下に `data` ディレクトリを作成し、そこに解凍する。
```shell
$ unzip -d {本リポジトリの絶対パス}/data gtFine_trainvaltest.zip
$ unzip -d {本リポジトリの絶対パス}/data leftImg8bit_trainvaltest.zip
```
データの仕様については、Appendix. に記載する。(TBA)

## 環境構築

Docker 環境でスクリプトを動作させるため、イメージを build し、コンテナを立ち上げる。

```shell
$ docker build -t ss:v1 docker/
$ docker run --gpus all -it --rm --shm-size=4g --name ss -v {本リポジトリの絶対パス}:/work ss:v1
```

## 処理フロー
1. 前処理
    - ダウンロードした画像のリサイズ（縮小）
    - RGB 画像を LABEL 画像に変換
2. 画像データの TFRecord 化
3. データ生成・Augumentation
4. 学習
5. 評価

## 前処理
`preprocess.py` で実施。

```shell
$ python preprocess.py
```

## 画像データの TFRecord 化
`create_tfrecord.py` で実施。
```shell
$ python create_tfrecord.py
```
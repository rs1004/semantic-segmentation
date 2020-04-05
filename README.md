# semantic-segmentation

## 概要

semantic segmentation を行うスクリプトを作成する。
本スクリプトを実行する前に、setup.md を参考に実行環境の準備を行う必要がある。

## データ準備

データは [Cityscapes Dataset](https://www.cityscapes-dataset.com/) を使用する。

事前に会員登録を済ませ、以下の手順でデータのダウンロードを行う。

ダウンロードコマンドは以下の通り。

```shell
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username={ユーザー名}&password={パスワード}&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

これにより、

- gtFine_trainvaltest.zip
- leftImg8bit_trainvaltest.zip

が手元にダウンロードされる。

{本リポジトリの絶対パス}直下に `data` ディレクトリを作成し、そこに解凍する。

```shell
unzip -d {本リポジトリの絶対パス}/data gtFine_trainvaltest.zip
unzip -d {本リポジトリの絶対パス}/data leftImg8bit_trainvaltest.zip
```

データの仕様については、Appendix. に記載する。(TBA)

## 処理フロー

1. 前処理
   - ダウンロードした画像のリサイズ（縮小）
   - RGB 画像を LABEL 画像に変換
2. 画像データの TFRecord 化
3. データ生成・Augumentation
4. 学習
5. 評価

### 1. 前処理

`preprocess.py` で実施。
EC2の `c5.xlarge` インスタンスで実行する想定。

```shell
docker build -t cpu_env docker/cpu_env/
docker run -it --rm --name cpu_env -v /work/semantic-segmentation/:/work cpu_env
python src/preprocess.py
```

### 2. 画像データの TFRecord 化

`create_tfrecord.py` で実施。
EC2の `p2.xlarge` インスタンスで実行する想定。

```shell
docker build -t gpu_env docker/gpu_env/
docker run -it --rm --name gpu_env -v {本リポジトリの絶対パス}:/work gpu_env
python create_tfrecord.py
```

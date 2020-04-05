# setup
実行環境の初期設定を行う。以下は `p2.xlarge` インスタンスで実施した。
- GPU認識
- Docker の導入
- nvidia docker 設定
- tensorflow-gpu 動作確認

## 基本
```shell
sudo apt update
sudo apt -y  upgrade
sudo apt-get update
sudo apt-get -y upgrade
```

## GPU認識
```shell
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

sudo vi /etc/modprobe.d/blacklist-nouveau.conf
i
blacklist nouveau
options nouveau modeset=0

sudo update-initramfs -u
再起動
```

## Dockerの導入
```
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install -y docker-ce
sudo usermod -aG docker ubuntu
再ログイン
```

## nvidia docker設定
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## tensorflow-gpu動作確認
```
lspci | grep -i nvidia（GPUの確認）
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu \
       python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
# git版本控制实战——MNIST数字分类
本项目实现了一个简单的MNIST数字分类程序，使用PyTorch框架进行构建和训练。

## 如何运行
本项目使用uv管理Python虚拟环境。首先参考[此链接](https://docs.astral.sh/uv/#installation)安装uv。

然后克隆仓库，创建虚拟环境并安装依赖：
```sh
git clone https://github.com/Golden-Pigeon/mnist_classification.git
cd mnist_classification
uv sync
source .venv/bin/activate
```

执行：
```sh
python main.py
```

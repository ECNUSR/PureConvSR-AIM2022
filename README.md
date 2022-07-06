# 介绍

主要代码改自[SR_Mobile_Quantization](https://github.com/NJU-Jet/SR_Mobile_Quantization)，针对比赛高频率修改模型的特点，做出一些修改。

## 开发者注意事项

注意代码符合规范:

```bash
pylint -j 16 --rcfile=.pylintrc common/ trials/ *.py | head -n 500
```

## 1. baseline

```bash
python train.py --trial baseline --lark cjh
python train_qat.py --trial baseline --qat_path experiments/baseline/best_status --lark cjh
```
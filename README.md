# 介绍

主要代码改自[SR_Mobile_Quantization](https://github.com/NJU-Jet/SR_Mobile_Quantization)，针对比赛高频率修改模型的特点，做出一些修改。

## 开发者注意事项

注意代码符合规范:

```bash
pylint -j 16 --rcfile=.pylintrc common/ trials/ *.py | head -n 500
```
## 实验
详细情况请见[实验表格](https://yzf-ecnu-sr.feishu.cn/sheets/shtcnv1M7ioCK0teeEOwto4Wxkc)
### 1. baseline

```bash
python train.py --trial baseline --lark cjh
python train_qat.py --trial baseline --qat_path experiments/baseline/best_status --lark cjh
python convert.py --name baseline
```
> PSNR: 30.0991 | QAT_PSNR: 30.0574

### 2. trial1

> 将“重复”、“Add”这两个算子用Cat替代；然后将四个3x3卷积，换成1x1卷积和3x3卷积交替；其余和baseline保持一致

```bash
python train.py --trial trial1 --lark cjh
python train_qat.py --trial trial1 --qat_path experiments/trial1/best_status --lark cjh
python convert.py --name trial1
```
> PSNR: 30.0872 | QAT_PSNR: 30.0259
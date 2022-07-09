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

### 2. baseline_typo_trans
> 将“重复”、“Add”这两个算子用Cat替代；
```bash
cp experiments/baseline experiments/baseline_typo_trans -r
cp tb_logger/baseline tb_logger/baseline_typo_trans -r
python train_qat.py --trial baseline_typo_trans --qat_path experiments/baseline_typo_trans/best_status --lark cjh
python convert.py --name baseline_typo_trans
```
> PSNR: 30.0991 | QAT_PSNR: 29.8804

### 3. trial1

> 在【baseline_typo_trans】基础上，将四个3x3卷积，换成1x1卷积和3x3卷积交替；通道数改为32；其余保持一致

```bash
python train.py --trial trial1 --lark cjh
python train_qat.py --trial trial1 --qat_path experiments/trial1/best_status --lark cjh
python convert.py --name trial1
```
> PSNR: 30.0872 | QAT_PSNR: 30.0259

### 3. trial2

> 在【trial2】基础上，将batch size改为32，增加量化训练的轮数；其余保持一致

```bash
python train.py --trial trial2 --lark cjh
python train_qat.py --trial trial2 --qat_path experiments/trial2/best_status --lark cjh
python convert.py --name trial2
```
> PSNR: 30.0937 | QAT_PSNR: 30.0173

### 3. trial3

> 在trial1的基础上，有两个算子我写成一层了，恢复回去，原则上两个等价

```bash
python train.py --trial trial3 --lark cjh
python train_qat.py --trial trial3 --qat_path experiments/trial3/best_status --lark cjh
python convert.py --name trial3
```
> PSNR: 30.0714 | QAT_PSNR: 29.9898

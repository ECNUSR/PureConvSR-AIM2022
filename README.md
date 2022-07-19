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

### 4. trial4

> 在trial1的基础上，赋值不是准确的0和1，而是偏移一丢丢。且qat训练的时候减小lr

```bash
cp experiments/trial1 experiments/trial4 -r
cp tb_logger/trial1 tb_logger/trial4 -r
python train_qat.py --trial trial4 --qat_path experiments/trial4/best_status --lark cjh
python convert.py --name trial4
```
> PSNR: 30.0872 | QAT_PSNR: 30.0173

### 5. trial5

> 在trial1的基础上，添加EMA策略

```bash
cp experiments/trial1 experiments/trial5 -r
cp tb_logger/trial1 tb_logger/trial5 -r
python train_qat.py --trial trial5 --qat_path experiments/trial5/best_status --lark cjh
python convert.py --name trial5
```
> PSNR: 30.0872 | QAT_PSNR: 30.0340

### 6. trial6

> 在trial5的基础上，增加一个1x1卷积

```bash
python train.py --trial trial6 --lark cjh
python train_qat.py --trial trial6 --qat_path experiments/trial6/best_status --lark cjh
python convert.py --name trial6
```
> PSNR: 30.0956 | QAT_PSNR: 30.0397

### 7. trial7

> 在trial5的基础上，1x3改回3x3卷积，通道数目改回28

```bash
python train.py --trial trial7 --lark cjh
python train_qat.py --trial trial7 --qat_path experiments/trial7/best_status --lark cjh
python convert.py --name trial7
```
> PSNR: 30.1632 | QAT_PSNR: 30.1139

### 8. trial8

> 在trial5的基础上，中间的两组3x3+1x1改回2x2+2x2卷积

```bash
python train.py --trial trial8 --lark cjh
python train_qat.py --trial trial8 --qat_path experiments/trial8/best_status --lark cjh
python convert.py --name trial8
```
> PSNR: 30.0999 | QAT_PSNR: 30.0232

### 9. trial9

> 在trial5的基础上，取消1x1卷积，多了一个3x3卷积，通道数下降到28

```bash
python train.py --trial trial9 --lark cjh
python train_qat.py --trial trial9 --qat_path experiments/trial9/best_status --lark cjh
python convert.py --name trial9
```
> PSNR: 30.0941 | QAT_PSNR: 30.0397

### 10. trial10

> 在trial5的基础上，1x1卷积变3x3卷积，通道数上升至48

```bash
python train.py --trial trial10 --lark cjh
python train_qat.py --trial trial10 --qat_path experiments/trial10/best_status --lark cjh
python convert.py --name trial10
```
> PSNR: 30.3244 | QAT_PSNR: 30.2501

### 11. trial11

> 只有一个卷积和一个pixelshuffle

```bash
python train.py --trial trial11 --lark cjh
python train_qat.py --trial trial11 --qat_path experiments/trial11/best_status --lark cjh
python convert.py --name trial11
```
> PSNR: 28.1597 | QAT_PSNR: 28.0257

### 12. trial12

> 在trial5的基础上，把10替换成trial11的权重

```bash
python train.py --trial trial12 --lark cjh
python train_qat.py --trial trial12 --qat_path experiments/trial12/best_status --lark cjh
python convert.py --name trial12
```
> PSNR: 30.1213 | QAT_PSNR: 30.0592

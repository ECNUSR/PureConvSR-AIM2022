# data preparation
Div2k dataset is required for train and valid. Due to the limitation of submission space, no dataset is provided in this directory, so you need to download it yourself. The folder structure is as follows:

```bash
Source-Codes/
├── datasets
    ├── DIV2K
        ├── DIV2K_train_HR
            ├── 0001.pt
            ├── 0002.pt
            ├── ...
            ├── 0800.pt
        ├── DIV2K_train_LR_bicubic
            ├── X3
                ├── 0001.pt
                ├── 0002.pt
                ├── ...
                ├── 0800.pt
        ├── DIV2K_valid_HR
            ├── 0801.pt
            ├── 0802.pt
            ├── ...
            ├── 0900.pt
        ├── DIV2K_valid_LR_bicubic
            ├── X3
                ├── 0801.pt
                ├── 0802.pt
                ├── ...
                ├── 0900.pt
```

# Contribution


# Requirements
We use tensorflow version 2.5.0 to train and quantify the model. You can execute the following script to create the environment:

```bash
conda env create -f tf250.yaml
```

# Training
Execute the following scripts in turn to run the code, and the final exported tflite file is in the tflite/trial24/ directory.
```
python train.py --trial trial24 --lark cjh
python train_qat.py --trial trial24 --lark cjh --qat_path experiments/trial24/best_status
python remove_clip_fintune.py --trial trial24 --qat_path experiments/trial24_qat/best_status --lark cjh
python convert.py --name trial24 --clip
```

# Run TFLite Model on your own devices
1. Download AI Benchmark from the Google Play / website and run its standard tests.
2. After the end of the tests, enter the PRO Model and select the Custom Model tab there.
3. Send your tflite model to your device and remember its location, then run the model.

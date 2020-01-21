# Similarity Inference Metric (SIM)

## Require
* Python 3.7
* PyTorch 1.3
* Ignite 
* Apex
* Yacs

## Prepare for dataset
Download [SYSU-MM01](https://github.com/wuancong/SYSU-MM01) dataset and uncompress it.
Change the entry `data_root` in configs/default/dataset.py to the path of the dataset.
Put the [rand_perm_cam.mat](https://github.com/wuancong/SYSU-MM01/blob/master/evaluation/data_split/rand_perm_cam.mat) in `exp` directory in dataset root. This file is used to assign gallery items for each trial while testing.

## Prepare for extracted features
Download the pretrained model and extracted features [features](https://pan.baidu.com/s/1mJF8iliPLO8Y7Z88IsMYOQ)[vl7m]


## Test
Run
```shell script
python3 extract.py model_path checkpoints/sysu/sysu-nodual-adam-16x8-128/sysu-nodual-adam-16x8-128_model_200.pth gpu 0
```

Run
```shell script
python3 eval.py 0 checkpoints/sysu/sysu-nodual-adam-16x8-128/sysu-nodual-adam-16x8-128_model_200.pth
```


## Performance

We evaluate the performance on [SYSU-MM01](https://github.com/wuancong/SYSU-MM01) under the setting of  **multi-shot** & **all-search**.

| model             | mAP | rank-1 |
| ----------------- | ------ | ------ |
| baseline      | 39.90 | 53.93 | 
| SGR           | 59.17 | 56.62 | 
| MNNR          | 54.39 | 56.31 | 
| SIM           | 60.88 | 56.93 | 


## Reference 

[L1aoXingyu/reid_baseline](https://github.com/L1aoXingyu/reid_baseline)

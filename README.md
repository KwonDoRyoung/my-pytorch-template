# My Pytorch Template
## Python(3.11.7) & Package version
- pytorch==2.1.2 (CUDA==11.8)
- torchvision==0.16.2
- timm==0.9.12
- opencv-python==4.9.0.80
- pandas==2.2.0
- matplotlib==3.8.2
- Django==5.0.1
- scikit-learn==1.4.0
- scipy==1.12.0
- SimpleITK==2.3.1
- albumentations==1.3.1
- torchmetrics==1.3.0.post0
- wandb==0.16.2
- torchcam==0.4.0
- openpyxl==3.1.2
- seaborn==0.13.2
- tqdm==4.66.2

## Dataset
1. `mnist` : MNIST
2. `cat_dog` : Cat & Dog

## Tool
1. `binary` Classification
2. `multiclass` Classification
3. `binary` Segmentation

## Model
1. Classification
    - timm 활용하여 호출함 
2. Segmentation
    - Backbone 필요한 segmentation 모델의 경우, timm 패키지 활용
    - 종류
      - FCN 32, 16, 8
      - DeepLabV3
      - DeepLabV3+
      - U-Net
      - U-Net++

## Loss
  1. `dice`
    - Binary Dice Loss
    - Multiclass Dice Loss
  2. `ce+dice`
    - Binary Cross Entropy with Dice Loss
    - Multiclass Cross Entropy with Dice Loss

## CODE Execution
1. Use specific GPU
    ```bash
    CUDA_VICUDA_VISIBLE_DEVICES=0 python train_cls.py ...
    CUDA_VICUDA_VISIBLE_DEVICES=1,2 python train_cls.py ...
    ```
2. Execute the distributed data-parallel in Pytorch
   - Execute the python script in a single server (`--standalone`)
       ```bash
       torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_TRAINERS} train_cls.py ...
       ```

## Reference
1. https://github.com/qubvel/segmentation_models.pytorch/tree/master
2. https://github.com/facebookresearch/fairseq/tree/main
3. https://github.com/JunMa11/SegLossOdyssey/tree/master

## Citation
If you find this repository useful for your research or if it helps any of your work, please consider citing it. GitHub will automatically generate a citation for you in APA or BibTeX format when you click the 'Cite this repository' button above the file list.
```
@software{Kwon_My_PyTorch_Template,
author = {Kwon, Do Ryoung},
title = {{My PyTorch Template}},
url = {https://github.com/KwonDoRyoung/my-pytorch-template}
}
```

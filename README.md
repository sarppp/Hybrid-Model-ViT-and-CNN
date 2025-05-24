Overview
This project builds a high-accuracy image classifier for cats and dogs using the same dataset Nyckel uses for their website. I wanted to delevop something smiliar but that can predict my cat with her cute pose with higher confidence level by using same dataset.

But unlike Nyckel, this model:

Predicts dogs too, not just cats
Achieves higher accuracy than Nyckel's own page
Is quantized for fast inference with minimal performance drop

### Some Technical Info
- Oxford-IIIT Pet Dataset (https://www.kaggle.com/datasets/zippyz/cats-and-dogs-breeds-classification-oxford-dataset)
- CNN Backbone: ResNet34 (Pretrained)
- Transformer Component: ViT-Base (Pretrained)
- Dynamic quantization of both linear and conv layers
- Dynamic Quantization: Optimized for ARM and x86
- Reduced model size: ~431MB to ~175MB


## RESULT OF NYCKEL
![image](https://github.com/user-attachments/assets/0084c25c-46dc-434f-bcf7-90f489e1123b)

## My Model
![image](https://github.com/user-attachments/assets/960a30b7-f5db-462a-bee2-cce7f97f09e6)

## DOGS ARE INCLUDED
![image](https://github.com/user-attachments/assets/3ff8d60f-e216-4c77-b49a-052b33250d03)

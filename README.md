# 3D Segmenter: 3D Transformer based Semantic Segmentation via 2D Panoramic Distillation (ICLR 2023 [paper](https://openreview.net/pdf?id=4dZeBJ83oxk))
## Method overview
![image](https://user-images.githubusercontent.com/41735931/235295741-82727ba1-b6a0-43e3-ad97-9fe018d57408.png)
![image](https://user-images.githubusercontent.com/41735931/235295747-297b44cd-3f9c-4e3e-8c5e-24ad466417be.png)
## Download data and pretrained model
* [3D rooms(30GB)](https://drive.google.com/file/d/1AMfeOt6V_igSoM5xq17b9xh1p9KaHKW-/view?usp=sharing) 5917 3D living rooms and bedrooms together with semantic label.
* [2D panoramas(342MB)](https://drive.google.com/file/d/1Mj36Y_tBDBzZRv20js-aBKp4QYy80nfm/view?usp=sharing) 2D panoramas rendered at room center. 
* [Cropped Blocks(19.9GB)](https://drive.google.com/file/d/1jQjg9jW1OQtnLayzdZZlpnkyOZSSRrtW/view?usp=sharing) Cropped 128x128x128 blocks from rooms.
* [Pretrained models and 2D teachers](https://drive.google.com/file/d/1Oh5NYdPn5ZBxwC0GKRyYFZgwS4EkbUhW/view?usp=sharing)
# Requirements
The code tested on python=3.9, pytorch=1.13.0 with the following packages: pytorch3d, open3d, opencv-python, tqdm, mmsegmentation, timm, einops.

pycuda 2022.1 is required for visualization.

# Visualization of dataset
After specifying the path, run
```
python visualize.py
```
to get visualization of PanoRooms3D dataset.
# Evaluation
Set the swin depths, swin heads and d_model according to the pretrained model and run
```
python eval.py
```
# Train
specify swin depths, swin heads and d_model and run training by
```
python train.py
```
# Distillation
Set the directory to rooms and panoramas, 2D teacher checkpoint and 3D checkpoint before distillation.
```
python distill.py
```



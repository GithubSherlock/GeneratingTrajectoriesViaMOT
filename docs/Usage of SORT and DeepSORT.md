# Usage of SORT and DeepSORT

https://www.cnblogs.com/wemo/p/10600454.html

### Running the tracker

```bash
python deep_sort_app.py \
    --sequence_dir=./MOT16/test/MOT16-06 \
    --detection_file=./resources/detections/MOT16_POI_test/MOT16-06.npy \
    --output_file=./output/MOT16-06.txt \
    #--min_detection_height=1 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
```

### detection

```bash
python tools/generate_detections.py \
	--model=resources/networks/mars-small128.pb \
	--mot_dir=./MOT16/test/MOT16-06 \
	--output_dir=./resources/detections/MOT16_test

```



### dataset info in deepSORT

```bash
[Sequence]
name=数据集名称
root=数据集根目录的路径
image_dir=图像序列所在的目录路径
image_format=图像文件的格式（例如：jpg、png等）
frame_rate=帧率
seq_length=图像序列的总帧数
image_width=图像的宽度
image_height=图像的高度
```

for example:

```bash
[Sequence]
name=MOT16-12
imDir=img1
frameRate=30
seqLength=900
imWidth=1920
imHeight=1080
imExt=.jpg
```

https://zhuanlan.zhihu.com/p/624244160

https://zhuanlan.zhihu.com/p/621374861

### Yolov8

```bash
yolo track model=weights/hauptmensa_1.pt source="video/hauptmensa_1.mp4" conf=0.3, iou=0.5 show=True save=True tracker="bytetrack.yaml"
```


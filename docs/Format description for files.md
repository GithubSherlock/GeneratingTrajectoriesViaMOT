# Format description for files

## MOTChallenge

Tracking with bounding boxes

(2D MOT 2015, MOT16, MOT17, MOT20, HT21)

``````bash
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
``````

example:

```bash
1, 3, 794.27, 247.59, 71.245, 174.88, -1, -1, -1, -1
1, 6, 1648.1, 119.61, 66.504, 163.24, -1, -1, -1, -1
1, 8, 875.49, 399.98, 95.303, 233.93, -1, -1, -1, -1
...
```

id:

Please see more details at [Instructions of Multiple Object Tracking Benchmark](https://motchallenge.net/instructions/).

## Detectron2

### Getting classes of Detectron2 dataset for detection

If using Detectron2 for object detection, it is necessary to note that only the classes of detected objects, which have been included in the used pre-training model, could be detected. This is the same for YOLO! View the labels and names of all classes in a COCO dataset with the following code:

```python
from detectron2.data import MetadataCatalog

metadata = MetadataCatalog.get("coco_2017_train") # 获取COCO数据集的元数据 Getting metadata from COCO datasets
for idx, name in enumerate(metadata.thing_classes): # 打印类别标签和名称 Print category labels and names
    print(f"ID: {idx}  name: {name}")
```

The output is as follows:
```bash
ID: 0  name: person
ID: 1  name: bicycle
ID: 2  name: car
ID: 3  name: motorcycle
ID: 4  name: airplane
ID: 5  name: bus
ID: 6  name: train
ID: 7  name: truck
ID: 8  name: boat
ID: 9  name: traffic light
ID: 10  name: fire hydrant
ID: 11  name: stop sign
ID: 12  name: parking meter
ID: 13  name: bench
ID: 14  name: bird
ID: 15  name: cat
ID: 16  name: dog
ID: 17  name: horse
ID: 18  name: sheep
ID: 19  name: cow
ID: 20  name: elephant
ID: 21  name: bear
ID: 22  name: zebra
ID: 23  name: giraffe
ID: 24  name: backpack
ID: 25  name: umbrella
ID: 26  name: handbag
ID: 27  name: tie
ID: 28  name: suitcase
ID: 29  name: frisbee
ID: 30  name: skis
ID: 31  name: snowboard
ID: 32  name: sports ball
ID: 33  name: kite
ID: 34  name: baseball bat
ID: 35  name: baseball glove
ID: 36  name: skateboard
ID: 37  name: surfboard
ID: 38  name: tennis racket
ID: 39  name: bottle
ID: 40  name: wine glass
ID: 41  name: cup
ID: 42  name: fork
ID: 43  name: knife
ID: 44  name: spoon
ID: 45  name: bowl
ID: 46  name: banana
ID: 47  name: apple
ID: 48  name: sandwich
ID: 49  name: orange
ID: 50  name: broccoli
ID: 51  name: carrot
ID: 52  name: hot dog
ID: 53  name: pizza
ID: 54  name: donut
ID: 55  name: cake
ID: 56  name: chair
ID: 57  name: couch
ID: 58  name: potted plant
ID: 59  name: bed
ID: 60  name: dining table
ID: 61  name: toilet
ID: 62  name: tv
ID: 63  name: laptop
ID: 64  name: mouse
ID: 65  name: remote
ID: 66  name: keyboard
ID: 67  name: cell phone
ID: 68  name: microwave
ID: 69  name: oven
ID: 70  name: toaster
ID: 71  name: sink
ID: 72  name: refrigerator
ID: 73  name: book
ID: 74  name: clock
ID: 75  name: vase
ID: 76  name: scissors
ID: 77  name: teddy bear
ID: 78  name: hair drier
ID: 79  name: toothbrush
```

## Yolov8

### Make your own pre-trained model

```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# PASCAL VOC dataset http://host.robots.ox.ac.uk/pascal/VOC by University of Oxford
# Example usage: yolo train data=VOC.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── VOC  ← downloads here (2.8 GB)


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../ultralytics/ultralytics/dataset/hauptmensa_1
train: # train images (relative to 'path')  16551 images
  - images/val
val: # val images (relative to 'path')  4952 images
  - images/val
test: # test images (optional)
  - images/val

# Classes
names:
  0: dog
  1: person
  2: cat
  3: sheep
  4: car
  5: balance car
  6: shopping trolley
  7: wheelchair
  8: aeroplane
  9: truck
  10: motorcycle
  11: scooter
  12: bus
  13: coach
  14: SUV
  15: bicycle

```

## SORT

Since my project does not directly use SORT, but as a node in my learning curve, not specifically described here. Please see the details about [SORT in Abewley's GitHub](https://github.com/abewley/sort).

## Deep SORT

If you would like to visualize Deep SORT directly through terminal commands, modify the command requirements in the following format:

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



### Parameter

* `conf_thres`: Confidence Threshold，置信度阈值，即以下图片上的值。只显示预测概率超过conf_thres的预测结果。
* `iou_thres`: Intersect over Union Threshold，交并比阈值。
  * IOU值：预测框大小$\cap$真实框大小/预测框大小$\cap$真实框大小。预测框与真实框的交集与并集的取值。
  * `iou_thres`在`objdetector.py`中：
    * 越大，则容易将对于同一个物品的不同预测结果 当成 对多个物品的多个预测结果，导致一个物品出现了多个预测结果。
    * 越小，则容易将对于多个物品的不同预测结果 当成 对同一个物品的不同预测结果，导致多个物品只出现了一个预测结
      果。


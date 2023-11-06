### File Format

Tracking with bounding boxes

(2D MOT 2015, MOT16, MOT17, MOT20, HT21)

``````python
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
``````

example:

```python
1, 3, 794.27, 247.59, 71.245, 174.88, -1, -1, -1, -1
1, 6, 1648.1, 119.61, 66.504, 163.24, -1, -1, -1, -1
1, 8, 875.49, 399.98, 95.303, 233.93, -1, -1, -1, -1
...
```

id:

### 查看COCO数据集中所有类别的标签和名称

```python
from detectron2.data import MetadataCatalog

metadata = MetadataCatalog.get("coco_2017_train") # 获取COCO数据集的元数据 Getting metadata from COCO datasets
for idx, name in enumerate(metadata.thing_classes): # 打印类别标签和名称 Print category labels and names
    print(f"ID: {idx}  name: {name}")
```

反馈结果如下：
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

https://blog.csdn.net/fmy_xfk/article/details/127022372

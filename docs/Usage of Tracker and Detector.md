# Usage of Tracker and Detector

## SORT

Since my project does not directly use SORT, but as a node in my learning curve, not specifically described here. Please see the details about [SORT in Abewley's GitHub](https://github.com/abewley/sort).

## Deep SORT

### Source code correction

##### NumPy

(Optional) Since newer versions of NumPy have removed the function `np.int`, you will need to replace `np.int` in the source code with `int`, either `np.int32` or `np.int64`, depending on the precision of the data you need.

##### Scikit-learn

(Required) When using Scikit-learn (version >= 0.23), due to the abandonment of the `sklearn.utils.linear_assignment_` module, it is necessary to use `scipy.optimize.linear_sum_assignment` instead and adjust the code appropriately. Modifications to `deep_sort\linear_assignment.py` are required:

```python
# Near by line 4
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
```

If you want to use Nwojke's source code directly, further modifications are required:

```python
# Near by line 59
#indices = linear_assignment(cost_matrix)
indices = np.array(linear_assignment(cost_matrix)).transpose()
```

### Running the tracker

Make sure you have Python 3 installed in your environment by running the following command:

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

Probably due to my computer environment, I didn't succeed in running the following command, so I used Nwojke's source code directly in my project directly.

```bash
python tools/generate_detections.py \
	--model=resources/networks/mars-small128.pb \
	--mot_dir=./MOT16/test/MOT16-06 \
	--output_dir=./resources/detections/MOT16_test

```

See [Nwojke's GitHub](https://github.com/nwojke/deep_sort) for exact usage.

## Yolov8

YOLOv8 has added two tracking algorithms to facilitate the result of tracking the object detection and segmentation, which can be used directly in the terminal, and of course, can also be called with Python code, which is very convenient.

If you want to use the tracking algorithm, it is best to update YOLOv8 to the latest version:

```bash
pip install --upgrade ultralytics
```

The following command tracks the video stream on Youtube:

```bash
yolo track model=yolov8n.pt source="https://youtu.be/Zgi9g1ksQHc"
```

You can use this if you are tracking the data acquired from your camera:

```bash
yolo track model=yolov8n.pt source=0
```

"source=0" represents the device number of the camera used is 0. If there are more than one camera, you can fill in 1, 2 as appropriate. Perform object tracking on local mp4 video files:

```bash
yolo track model=yolov8n.pt source='Your_Video.mp4' save=True
```

Here is one of the commands for object detection I used:

```bash
yolo track model=weights/hauptmensa_1.pt source="video/hauptmensa_1.mp4" conf=0.3, iou=0.5 show=True save=True tracker="bytetrack.yaml"
```

Please reference the specific method of usage on [Yolov8 docs](https://docs.ultralytics.com/usage/cli/).

## Detectron2

In the folder “tests”, there is a code using detectron2 for the object detection on images and videos. I switched to Yolov8 because the usage of this detector is too complex and minor, but I think the quest is still worthwhile!

Since I'm using Yolov8 for the detector in my project, I'm not going to illustrate detectron2 here. If you are interested in this detector, please refer to the [detectron2’s documentation](https://detectron2.readthedocs.io/en/latest/index.html).

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

import os
import cv2
import numpy as np

from pathlib import Path

class Detector:
    def __init__(self, model_type="OD"):
        self.cfg = get_cfg()
        self.model_type = model_type

        # Load model config and pretrained model
        if model_type == "OD":  # object detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

        elif model_type == "IS":  # instrance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        elif model_type == "KP":  # keypoint detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

        elif model_type == "LVIS":  # LVIS Segmentation
            self.cfg.merge_from_file(
                model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")

        elif model_type == "PS":  # Panoptic Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

        # Choose threshold and device
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        self.cfg.MODEL.DEVICE = 'cuda'  # cpu or cuda

        # Load predictor and metadata
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        if self.model_type != "PS":
            predictions = self.predictor(image)  # 使用预测器对图像进行预测
            # 筛选出"person", "car", "truck"对应的预测结果
            instances = predictions["instances"]
            pred_classes = instances.pred_classes.cpu().numpy()
            selected_instances = instances[np.isin(pred_classes, [0, 1, 2, 3, 5, 7])] # detect person, bicycle, car, motorcycle, bus, truck

            viz = Visualizer(image[:, :, ::-1], self.metadata,
                            instance_mode=ColorMode.IMAGE)  # 创建Visualizer对象，用于绘制预测结果
            output = viz.draw_instance_predictions(selected_instances.to("cpu"))  # 将预测结果绘制到图像上

        else:
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(image[:, :, ::-1], self.metadata)
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

        cv2.imshow("Result", output.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    def onVideo(self, videoPath, outputVideoPath):
        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            return

        (success, image) = cap.read()

        # Get information about the input video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec for the output video and the output video object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codecs can be changed as required
        out = cv2.VideoWriter(outputVideoPath, fourcc, fps, (frame_width, frame_height))

        # Open a new txt file for writing detection results
        video_name = Path(videoPath).stem
        text_file_name = f"{video_name}.txt"
        output_dir = os.path.dirname(outputVideoPath)
        outputTextPath = os.path.join(output_dir, text_file_name)
        txt_file = open(outputTextPath, "w")

        frame_count = 1

        while success:
            if self.model_type != "PS":
                predictions = self.predictor(image)  # 使用预测器对图像进行预测
                # 筛选出"person", "car", "truck"对应的预测结果
                instances = predictions["instances"]
                pred_classes = instances.pred_classes.cpu().numpy()
                selected_instances = instances[np.isin(pred_classes, [0, 1, 2, 3, 5, 7])] # detect person, bicycle, car, motorcycle, bus, truck
                
                viz = Visualizer(image[:, :, ::-1], self.metadata,
                                instance_mode=ColorMode.IMAGE)  # 创建Visualizer对象，用于绘制预测结果
                output = viz.draw_instance_predictions(selected_instances.to("cpu"))  # 将预测结果绘制到图像上

                # Write detection results to txt file
                for i in range(len(selected_instances)):
                    box = selected_instances.pred_boxes.tensor[i].cpu().numpy()
                    score = selected_instances.scores[i].cpu().numpy()
                    #class_id = selected_instances.pred_classes[i].numpy()
                    txt_file.write(f"{frame_count},-1,{box[0]:.3f},{box[1]:.3f},{box[2]-box[0]:.3f},{box[3]-box[1]:.3f},{score:.6f},-1,-1,-1\n")

            else:
                predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
                viz = Visualizer(image[:, :, ::-1], self.metadata)
                output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

            cv2.imshow("Result", output.get_image()[:, :, ::-1])

            # Write frames to output video file
            out.write(output.get_image()[:, :, ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()

            frame_count += 1

        cap.release()
        out.release()
        txt_file.close()
        cv2.destroyAllWindows()
import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from src.misc import dist

__all__ = ['VOCEvaluator',]

class VOCEvaluator:
    def __init__(self, voc_gt_path, iou_threshold=0.5):
        self.voc_gt_path = voc_gt_path
        self.iou_threshold = iou_threshold
        self.img_ids = []
        self.predictions = []
        self.ground_truths = self.load_voc_annotations()
    
    def load_voc_annotations(self):
        gt_data = {}
        for xml_file in os.listdir(self.voc_gt_path):
            if not xml_file.endswith(".xml"):
                continue
            tree = ET.parse(os.path.join(self.voc_gt_path, xml_file))
            root = tree.getroot()
            img_id = root.find("filename").text
            gt_data[img_id] = []
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                bbox = [int(bbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]
                label = obj.find("name").text
                gt_data[img_id].append((bbox, label))
        return gt_data

    def update(self, predictions):
        self.predictions.extend(predictions)

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def evaluate(self):
        tp, fp, total_gt = 0, 0, sum(len(v) for v in self.ground_truths.values())
        for pred in self.predictions:
            img_id, pred_boxes, pred_labels, pred_scores = pred
            if img_id not in self.ground_truths:
                fp += len(pred_boxes)
                continue

            gt_boxes_labels = self.ground_truths[img_id]
            matched_gt = set()

            for i, (bbox, label) in enumerate(gt_boxes_labels):
                for j, (p_bbox, p_label, p_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                    if p_label == label and self.compute_iou(bbox, p_bbox) > self.iou_threshold:
                        if i not in matched_gt:
                            tp += 1
                            matched_gt.add(i)
                        else:
                            fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt if total_gt > 0 else 0
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, TP: {tp}, FP: {fp}")
        return precision, recall

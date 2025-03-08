import os
import xml.etree.ElementTree as ET

import torch
import torch.utils.data
import torchvision
from torchvision.transforms import functional as F


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, year="2012", image_set="train", transforms=None):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transforms = transforms
        self.image_dir = os.path.join(root, "VOC{}/JPEGImages".format(year))
        self.anno_dir = os.path.join(root, "VOC{}/Annotations".format(year))
        self.ids = self._load_image_set_index()

    def _load_image_set_index(self):
        image_set_file = os.path.join(self.root, "VOC{}/ImageSets/Main/{}.txt".format(self.year, self.image_set))
        with open(image_set_file) as f:
            return [x.strip() for x in f.readlines()]

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.image_dir, img_id + ".jpg")
        anno_path = os.path.join(self.anno_dir, img_id + ".xml")
        
        img = torchvision.io.read_image(img_path).float() / 255.0
        target = self._parse_voc_xml(anno_path, img_id)

        if self.transforms:
            img = F.to_pil_image(img)
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def _parse_voc_xml(self, xml_path, image_id):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        iscrowd = []
        areas = []
        
        for obj in root.findall("object"):
            label = obj.find("name").text
            bndbox = obj.find("bndbox")
            bbox = [
                float(bndbox.find("xmin").text),
                float(bndbox.find("ymin").text),
                float(bndbox.find("xmax").text),
                float(bndbox.find("ymax").text),
            ]
            boxes.append(bbox)
            labels.append(label)
            iscrowd.append(0)  # VOC does not define "crowd"
            areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": labels,
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "image_id": torch.tensor([int(image_id)]),
        }

        return target


def get_voc_api_from_dataset(dataset):
    return dataset  # VOC does not have an API like COCO

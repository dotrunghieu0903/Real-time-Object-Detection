import torch
import torch.utils.data
import torchvision
from torchvision.datasets import VOCDetection

torchvision.disable_beta_transforms_warning()

from src.core import register

__all__ = ['VOCDetection']

@register
class VOCDetection(VOCDetection):
    __inject__ = ['transforms']

    def __init__(self, img_folder, ann_file=None, transforms=None):
        super(VOCDetection, self).__init__(root=img_folder, year="2012", image_set="train", download=True)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(VOCDetection, self).__getitem__(idx)
        image_id = torch.tensor([idx])
        target = self.prepare(img, target, image_id)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return img, target

    def prepare(self, image, target, image_id):
        w, h = image.size
        objects = target["annotation"]["object"]

        if isinstance(objects, dict):  # Convert single object case to a list
            objects = [objects]

        boxes = []
        labels = []
        part_boxes = []
        part_labels = []

        for obj in objects:
            # Main object bounding box
            bbox = [int(obj["bndbox"]["xmin"]), int(obj["bndbox"]["ymin"]), 
                    int(obj["bndbox"]["xmax"]), int(obj["bndbox"]["ymax"])]
            boxes.append(bbox)
            labels.append(voc_name2label[obj["name"]])  # Convert to label index

            # Handle object parts
            if "part" in obj:
                parts = obj["part"]
                if isinstance(parts, dict):  # Convert single part case to a list
                    parts = [parts]

                for part in parts:
                    part_bbox = [int(part["bndbox"]["xmin"]), int(part["bndbox"]["ymin"]), 
                                 int(part["bndbox"]["xmax"]), int(part["bndbox"]["ymax"])]
                    part_boxes.append(part_bbox)
                    part_labels.append(voc_name2label.get(part["name"], -1))  # -1 if unknown part

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        part_boxes = torch.tensor(part_boxes, dtype=torch.float32) if part_boxes else torch.empty((0, 4))
        part_labels = torch.tensor(part_labels, dtype=torch.int64) if part_labels else torch.empty((0,))

        target = {
            "boxes": boxes,
            "labels": labels,
            "part_boxes": part_boxes,  # Store part bounding boxes
            "part_labels": part_labels,  # Store part labels
            "image_id": image_id,
        }

        return image, target

    def extra_repr(self) -> str:
        return f' img_folder: {self.root}\n year: 2012\n'

# VOC category mappings

voc_category2name = {
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplan",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
}

voc_category2label = {k: i for i, k in enumerate(voc_category2name.keys())}
voc_label2category = {v: k for k, v in voc_category2label.items()}
voc_name2label = {v: k for k, v in voc_category2name.items()}

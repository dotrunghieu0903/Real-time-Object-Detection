import torch
import torch.utils.data
import torchvision
from torchvision.datasets import VOCDetection

torchvision.disable_beta_transforms_warning()

from src.core import register

__all__ = ['VOC2012Detection']

@register
class VOC2012Detection(VOCDetection):
    __inject__ = ['transforms']
    
    def __init__(self, img_folder, ann_file, transforms=None):
        super(VOC2012Detection, self).__init__(root=img_folder, year="2012", image_set="train", download=True)
        self._transforms = transforms
    
    def __getitem__(self, idx):
        img, target = super(VOC2012Detection, self).__getitem__(idx)
        image_id = idx
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return img, target

    def prepare(self, image, target):
        w, h = image.size
        
        objects = target["annotation"]["object"]
        if isinstance(objects, dict):
            objects = [objects]
        
        boxes = []
        labels = []
        for obj in objects:
            bbox = [int(obj["bndbox"]["xmin"]), int(obj["bndbox"]["ymin"]), 
                    int(obj["bndbox"]["xmax"]), int(obj["bndbox"]["ymax"])]
            boxes.append(bbox)
            labels.append(obj["name"])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([image_id])}
        
        return image, target

    def extra_repr(self) -> str:
        return f' img_folder: {self.root}\n year: 2012\n'

voc2012_category2name = {
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

voc_category2label = {k: i for i, k in enumerate(voc2012_category2name.keys())}
voc_label2category
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import importlib.metadata
from torch import Tensor 

if importlib.metadata.version('torchvision') == '0.15.2':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.tv_tensors import BoundingBoxes as BoundingBoxeses
    from torchvision.tv_tensors import BoundingBoxesFormat, Mask, Image, Video
    from torchvision.transforms.v2 import SanitizeBoundingBoxes as SanitizeBoundingBoxeses
    _boxes_keys = ['format', 'spatial_size']

elif '0.17' > importlib.metadata.version('torchvision') >= '0.16':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxeses
    from torchvision.tv_tensors import (
        BoundingBoxeses, BoundingBoxesFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

elif importlib.metadata.version('torchvision') >= '0.17':
    import torchvision
    from torchvision.transforms.v2 import SanitizeBoundingBoxeses
    from torchvision.tv_tensors import (
        BoundingBoxeses, BoundingBoxesFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

else:
    raise RuntimeError('Please make sure torchvision version >= 0.15.2')



def convert_to_tv_tensor(tensor: Tensor, key: str, box_format='xyxy', spatial_size=None) -> Tensor:
    """
    Args:
        tensor (Tensor): input tensor
        key (str): transform to key

    Return:
        Dict[str, TV_Tensor]
    """
    assert key in ('boxes', 'masks', ), "Only support 'boxes' and 'masks'"
    
    if key == 'boxes':
        box_format = getattr(BoundingBoxesFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxeses(tensor, **_kwargs)

    if key == 'masks':
       return Mask(tensor)


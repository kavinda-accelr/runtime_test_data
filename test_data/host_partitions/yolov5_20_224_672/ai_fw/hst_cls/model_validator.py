# Copyright (c) 2022 Analog Inference, Inc.

import os

import yaml
import torch

from yolov5_release_v7_classes import val


def validate_model(model_pt,
                   validation_loader,
                   batch_size,
                   batches,
                   image_size,
                   single_class,
                   device='0'):
    
    data_file = os.path.join(os.path.dirname(__file__), 'yolov5_release_v7_classes/data/VOC.yaml')
    with open(data_file, errors='ignore') as f:
        data_dict = yaml.safe_load(f)

    results, _, _ = val.run(data_dict,
                            batch_size=batch_size,
                            batches=batches,
                            imgsz=image_size,
                            model=model_pt,
                            device=device,
                            iou_thres=0.6,
                            single_cls=single_class,
                            half=False,
                            dataloader=validation_loader,
                            save_dir=False,
                            save_json=False,
                            plots=False)
    mAp = results[2]
    torch.cuda.empty_cache()
    return mAp
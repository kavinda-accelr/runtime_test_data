# Copyright (c) 2022 Analog Inference, Inc.
import os
import numpy as np
import torch

import yaml

from yolov5_release_v7_classes.utils.dataloaders import create_dataloader
from yolov5_release_v7_classes.utils.general import check_dataset, colorstr


def yolov5_data_loader(batch_size):

    # Required files for validation
    data_file = os.path.join(os.path.dirname(__file__), 'yolov5_release_v7_classes/data/VOC.yaml')
    hyperparameter_file = os.path.join(os.path.dirname(__file__), 'yolov5_release_v7_classes/data/hyps/hyp.VOC.yaml')

    # Load the validation data
    data_dictionary = check_dataset(data_file)
    validation_path = data_dictionary['val']

    # Load the hyperparameter file
    with open(hyperparameter_file) as f:
        hyp = yaml.safe_load(f)

    # The batch size
    batch_size = batch_size

    # Image size
    image_size = 640

    # Number of workers
    workers = 8

    # Are we doing single class
    single_cls = False

    # @TODO: gs parameter?
    gs = 32

    # Proprietary data loader
    val_loader = create_dataloader(validation_path,
                                   image_size,
                                   batch_size,
                                   gs,
                                   single_cls,
                                   hyp=hyp,
                                   cache=None,
                                   rect=True,
                                   rank=-1,
                                   workers=workers,
                                   pad=0.5,
                                   prefix=colorstr('val: '))[0]
    return val_loader
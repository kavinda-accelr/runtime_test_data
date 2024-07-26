# Copyright (c) 2022 Analog Inference, Inc.

from typing import Union, List
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 


from torch import nn
import torch
from analog_utils.common.model_plugin.aix_model_plugin_template import ModelPlugin
from model_validator import validate_model
from data_loader import yolov5_data_loader
from model_definition import ReconstructedYoloV5

"""
Return None for methods that are not yet applicable for the model
"""


class Plugin(ModelPlugin):
    def __init__(self, model_name: str, weight_file: str, device: str, image_size: List, val_dir: str):
        """
        Parameters
        __________
        model_name: (str) name of the network being processed
        weight_file: (str) weight file path
        device: (str) device being used either "cpu" or "cuda"
        image_size: List[int, int, int, int] dimension of input tensor to the model
        val_dir: (str) points to the directory containing the dataset to be used
        """
        super().__init__(model_name)
        self.weight_file = weight_file
        self.device = device
        self.image_size = image_size
        self.val_dir = val_dir

    def __init__(self, model_name: str, weight_file: str, device: str, image_size: List, val_dir: str):
        """
        Parameters
        __________
        model_name: (str) name of the network being processed
        weight_file: (str) weight file path
        device: (str) device being used either "cpu" or "cuda"
        image_size: List[int, int, int, int] dimension of input tensor to the model
        val_dir: (str) points to the directory containing the dataset to be used
        """
        super().__init__(model_name)
        self.weight_file = weight_file
        self.device = device
        self.image_size = image_size
        self.val_dir = val_dir

    def get_model(self) -> Union[nn.Module]:
        model_pt = torch.load(self.weight_file, map_location=self.device)
        model = model_pt['model']
        model = model.float()
        model.eval()
        return model

    @property
    def egress_layers(self) -> List:
        egress_layers = [70, 79, 87]
        return egress_layers

    @property
    def ingress_layers(self) -> List:
        ingress_layers = [0]
        return ingress_layers

    def get_dataloader(self, batch_size: int):
        data_loader = yolov5_data_loader(batch_size)
        return data_loader

    def validate_model(self, model: Union[nn.Module], batch_size: int, batches: int, skip: int) -> Union[float, None]:

        acc = validate_model(model, 
                             self.get_dataloader(batch_size), 
                             batch_size,
                             batches,
                             self.image_size,
                             False)
        return acc

    def full_model_reconstructor(self, partial_model: Union[nn.Module], reference_model: Union[nn.Module]) -> \
            Union[Union[nn.Module], None]:
        
        full_model = ReconstructedYoloV5(partial_model, reference_model)

        return full_model

    def set_ingress_egress_layers(self, r_input):
        """
        ***Note that this is a highly temporary method which will be removed once the tool chain has capability to
         reconstruct the full model,***

        The method analog_utils.analog_quantizer.model_quantizer.ModelQuantizer.set_ingress_egress_layers is temporarily
        slightly model dependent, so this method is overriden here"""

        for ing_layer in self.ingress_layers:
            r_input.cart.m_list[ing_layer].ingress = 1
        for eg_layer in self.egress_layers:
            r_input.cart.m_list[eg_layer].egress = 1

        return r_input

    @staticmethod
    def get_forced_host_layers() -> Union[List[str], None]:
        forced_host_layers = ['Concat_223', 'Concat_310', 'Concat_397', 'Concat_405']
        return forced_host_layers

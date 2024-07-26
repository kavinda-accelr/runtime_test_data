import numpy as np
import torch

from analog_utils.common.preprocess.preprocess_template import ProcessingPipeline

class PreProcessingPipeline(ProcessingPipeline):

    @classmethod
    def pipeline(cls, img: list) -> list:
        return img

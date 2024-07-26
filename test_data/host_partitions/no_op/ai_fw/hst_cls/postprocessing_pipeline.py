import numpy as np
import torch

from analog_utils.common.postprocess.postprocess_template import ProcessingPipeline

class PostProcessingPipeline(ProcessingPipeline):

    @classmethod
    def pipeline(cls, prediction: list) -> list:
        return prediction

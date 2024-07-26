import numpy as np
import torch
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
CLS_PATH = os.path.join(str(FILE.parents[0]),'yolov5_release_v7_classes')
if CLS_PATH not in sys.path:
    sys.path.insert(0, CLS_PATH)

from analog_utils.common.postprocess.postprocess_template import ProcessingPipeline
from yolov5_release_v7_classes.utils.general import non_max_suppression

class PostProcessingPipeline(ProcessingPipeline):

    @classmethod
    def pipeline(cls, prediction: torch.tensor) -> np.array:
        output = non_max_suppression(prediction[1],
                                     conf_thres=0.25,
                                     iou_thres=0.65,
                                     classes=None,
                                     agnostic=False,
                                     multi_label=False,
                                     labels=(),
                                     max_det=300,
                                     nm=0,
                                     )[0].numpy()
        return [output]

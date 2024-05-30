from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .airsim_camera import MULTIAGENTAIRSIMCAM
from .multiDet import MultiAgentDetDataset


def get_dataset():
    class Dataset(MULTIAGENTAIRSIMCAM, MultiAgentDetDataset):
        def collate_batch_train(self):
            pass
    return Dataset


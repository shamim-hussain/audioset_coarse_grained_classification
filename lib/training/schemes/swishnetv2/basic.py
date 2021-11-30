
from ._base import SwishNetV2Training
from lib.training.training_mixins import ReduceLR

class SwishNetV2Basic(ReduceLR,SwishNetV2Training):
    pass

SCHEME = SwishNetV2Basic

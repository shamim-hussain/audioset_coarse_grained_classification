
from ._base import SwishNetV2Training
from ..scheme_mixins import CosineAnnealWarmRestart

class SwishNetV2Cosine(CosineAnnealWarmRestart,SwishNetV2Training):
    pass

SCHEME = SwishNetV2Cosine

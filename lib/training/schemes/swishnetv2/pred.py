
from ._base import SwishNetV2Training
from ..scheme_base import AudiosetTesting

class SwishNetV2Pred(AudiosetTesting, SwishNetV2Training):
    pass

SCHEME = SwishNetV2Pred
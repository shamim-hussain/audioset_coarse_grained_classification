from lib.training.schemes.swishnetv2.pred import SCHEME

from ._base import VGG16Training
from ..scheme_base import AudiosetTesting

class VGG16Pred(AudiosetTesting, VGG16Training):
    pass

SCHEME = VGG16Pred

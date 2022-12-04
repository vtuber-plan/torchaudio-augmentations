
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.transforms import PitchShift
class PitchShift2(torch.nn.Module):
    """Pitch shift algorithm
    """    
    def __init__(self) -> None:
        super(PitchShift2, self).__init__()


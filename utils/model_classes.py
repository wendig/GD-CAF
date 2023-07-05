from models import GDCAF
from models.unets import UNetDS_Attention
from typing import Tuple, Type
import pytorch_lightning as pl

def get_model_class(model_file) -> Tuple[Type[pl.LightningModule], str]:
    # This is for some nice plotting
    if "GDCAF" in model_file:
        model_name = "GDCAF"
        model = GDCAF
    elif "UNetDS_Attention" in model_file:
        model_name = "SmaAt-UNet"
        model = UNetDS_Attention
    else:
        raise NotImplementedError("Model not found")
    return model, model_name

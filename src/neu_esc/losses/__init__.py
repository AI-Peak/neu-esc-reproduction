"""Classification, continuous SMART and auxiliary MLM objectives."""
from .classification import classification_loss
from .mlm import mask_tokens, mlm_loss
from .smart import smart_loss

__all__ = ["classification_loss", "mask_tokens", "mlm_loss", "smart_loss"]

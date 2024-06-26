"""
No matter what model, get structured NE predictions
"""

import datasets

from inference_trf import infer_with_trf
from inference_spacy import infer_with_spacy
from model_registry import get_model_registry
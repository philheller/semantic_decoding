from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput
import torch

class SyntacticGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def generate(self):
        pass
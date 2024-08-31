from typing import List, Optional, Any, Union
from enum import Enum
from syntactic import SyntacticGenerator
from semantic import SemanticGenerator
from data_structures import SemanticToken
import torch
from transformers.generation.utils import GenerationConfig


class SemanticGenerationMode(Enum):
    BEAM_SEARCH = "beam_search"
    GREEDY_SEARCH = "greedy_search"

class SemanticGenerationConfig:
    def __init__(
        self,
        num_beams: int,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        num_return_sequences: int = 1,
        max_length: Optional[int] = None,
        do_sample: bool = False,
    ):
        self.num_beams=num_beams
        self.length_penalty=length_penalty
        self.early_stopping=early_stopping
        self.num_return_sequences=num_return_sequences
        self.max_length=max_length
        self.do_sample=do_sample
    
    def get_generation_mode(self) -> SemanticGenerationMode:
        return SemanticGenerationMode.BEAM_SEARCH

class Generator:
    def __init__(
        self,
        model_name: str,
        semantic_generators: Union[List[str], str],
        device: str
    ):
        self.syntactic_generator = SyntacticGenerator(model_name, device)
        self.tokenizer = self.syntactic_generator.tokenizer
        self.semantic_generator = SemanticGenerator(semantic_generators, device)
        self.device = device
    
    def generate(
        self,
        semantic_generation_config: SemanticGenerationConfig = SemanticGenerationConfig(1),
        inputs: Optional[torch.Tensor] = None, # can also be set via kwargs
        generation_config: Optional[GenerationConfig] = None,
        return_dict_in_generate: bool = True,
        output_scores: bool = True,
        output_logits: bool = False,
        max_new_tokens: Optional[int] = None,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        resume_generation: bool = False,
        past_key_values: Optional[torch.Tensor] = None,
        last_scores: Optional[torch.Tensor] = None,
        last_beam_scores: Optional[torch.Tensor] = None, # for manual setting of beam scores
        original_prompt_length: Optional[int] = None,
        renormalize_logits: bool = True,
        reproducibility: bool = False,
        length_penalty: float = 1.0,  # same as default by hf
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        **kwargs: Any
    ) -> List[SemanticToken]:
        # general preparations

        generation_mode = semantic_generation_config.get_generation_mode()
        if generation_mode == SemanticGenerationMode.GREEDY_SEARCH:
            # greedy search here
            pass
        elif generation_mode == SemanticGenerationMode.BEAM_SEARCH:
            # beam search here
            pass
        else:
            raise ValueError(f"Generation mode {generation_mode} not supported.\n\
                Supported modes: {[mode.value for mode in SemanticGenerationMode]}")
        
        

from dataclasses import dataclass
from __future__ import annotations
from typing import List, Optional
import torch

@dataclass
class SemanticToken:
    entity_id: str
    step: int
    score: float
    next_entities: List[Optional[SemanticToken]]
    
    
@dataclass
class Hypothesis:
    shortened_by: int
    tokens_added: int
    continuation: ContinuationData
    original_entity: RawEntity

@dataclass
class ContinuationData:
    original: OriginalContinuatinData
    shortened: ShortenedContinuationData

@dataclass
class OriginalContinuatinData:
    scores: torch.Tensor
    sequences: torch.Tensor
    attention_mask: torch.Tensor
    entire_scores: torch.Tensor
    entire_sequences: torch.Tensor
    beam_indices: torch.Tensor
    
@dataclass
class ShortenedContinuationData:
    scores: torch.Tensor
    sequences: torch.Tensor
    attention_mask: torch.Tensor

@dataclass
class RawEntity:
    word: str
    start: int
    end: int
    _type: str
    score: str
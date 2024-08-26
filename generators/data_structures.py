from __future__ import annotations
from dataclasses import dataclass, is_dataclass
from typing import Optional, Tuple
import torch

@dataclass
class ContinuationData:
    """ 
    Contains all the data necessary to continue the generation of a sequence.
    The data is sliced to only contain the data necessary for a hypothesis.
    The only exception is the `original_data` field, which contains the original
    data in raw format (as output by the model) and is optional, as it is highly
    unefficient to store.
    
    :param sequences: Sequence of token ids of shape (, sequnce_length)
    :type sequences: torch.Tensor
    :param transition_scores: Transition scores of the tokens at generation steps. 
        The transition_scores are not of the same shape as the scores, instead only
        the scores of the hypothesis itself are kept. The shape is therefore
        (, sequence_length).
    :type transition_scores: torch.Tensor
    :param last_beam_scores: Scores of the last beam. Can also be calculated from
        the transition_scores. The sum of the transition_scores of a beam correspond
        to the `last_beam_scores`.
    :type last_beam_scores: torch.Tensor
    :param past_key_values: Past key values for the model. The past key values contain
        values for the previously generated content. The structure
        as follow:
        - layer of the transformer
        - tuple of key and value tensors
        - tensor of shape (
            1, # since only kept for this hypothesis
            num_heads,
            sequence_length,
            head_dim
        )
    :type past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    :param attention_mask: Attention mask for the hypothesis.
    :type attention_mask: torch.Tensor
    :param original_data: Original data of the continuation. This data is in raw format
        (as output by the model) and optional, as it is highly unefficient to store.
    :type original_data: Optional[OriginalContinuationData]
    """
    sequences: torch.Tensor
    transition_scores: torch.Tensor
    last_beam_scores: torch.Tensor
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    attention_mask: torch.Tensor
    original_data: Optional[OriginalContinuationData]

    def __repr__(self) -> str:
        pkv_len_0 = len(self.past_key_values)
        pkv_len_1 = len(self.past_key_values[0])
        pkv_shape = self.past_key_values[0][0].shape
        return f"ContinuationData(sequences={self.sequences}, transition_scores={self.transition_scores}, last_beam_scores={self.last_beam_scores}, past_key_values=(Shape [{pkv_len_0}, {pkv_len_1}, {pkv_shape}, attention_mask={self.attention_mask}, original_data={'Available' if self.original_data is not None else 'None'})"
    
    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        """ 
        The length of the sequences tensor.
        """
        return len(self.sequences.shape[-1])
    
    
@dataclass
class OriginalContinuationData:
    """ 
    This class contains all the data in raw format (as output by the model).
    
    :param sequences: Sequence of token ids
    :type sequences: torch.Tensor
    :param scores: Scores of the tokens at generation steps. # of tuples is 
        equal to the number of tokens generated. The tensor itself is of shape
        (batch_size, vocab_size).
    :type scores: Tuple[torch.Tensor]
    :param transition_scores: Transition scores of the tokens at generation steps.
    :type transition_scores: Tuple[torch.Tensor]
    :param beam_indices: Indices of the beams that generated the tokens.
    :type beam_indices: torch.Tensor
    :param past_key_values: Past key values for the model. The past key values contain
        values for the previously generated content. The structure
        as follow:
        - layer of the transformer
        - tuple of key and value tensors
        - tensor of shape (
            batch_size,
            num_heads,
            sequence_length,
            head_dim
        )
    :type past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    :param attention_mask: Attention mask for the model.
    :type attention_mask: torch.Tensor
    :param last_beam_scores: Scores of the last beam. Can also be calculated from
            the scores, sequences and beam indices by using 
            `model.compute_transition_scores`. The sum of the
            transition_scores of a beam correspond to the `last_beam_scores`.
    
    """
    sequences: torch.Tensor
    scores: Tuple[torch.Tensor]
    transition_scores: Tuple[torch.Tensor]
    beam_indices: torch.Tensor
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    attention_mask: torch.Tensor
    last_beam_scores: torch.Tensor

    def __repr__(self) -> str:
        return f"OriginalContinuationData(sequences={self.sequences}, scores={self.scores}, transition_scores={self.transition_scores}, beam_indices={self.beam_indices}, past_key_values=<ommited_for_readability>, attention_mask={self.attention_mask}, last_beam_scores={self.last_beam_scores})"

    def __str__(self) -> str:
        return self.__repr__()

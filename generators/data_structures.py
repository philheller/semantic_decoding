from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Optional, Tuple
import torch

"""
General structure
1. generation_step
    [
        a) semantic_hypothesis
            z) unique_key
            y) score
            x) SynthacticHypothesis
                [
                    - unique_key
                    - semantic_source_hypothesis_idx
                    - syntactic_source_hypothesis_idx
                    SemanticData:
                        - unique_key
                        - start
                        - end
                        - type
                        - other
                    SyntacticHypothesisContinuationData:
                        - sequences
                        - transition_scores
                        - last_beam_scores
                        - past_key_values
                        - attention_mask
                        - SyntacticHypothesisUnshortenedContinuationData:
                            - transition_scores
                            - sequences
                            - last_beam_scores
                            - past_key_values
                            - attention_mask
                    SynthacticHypothesisMetaData: [...]
                ]
            w) OriginalContinuationData
    ]
"""

@dataclass
class SemanticToken:
    """
    Contains all the data which make up a semantic hypothesis.

    :param aggregation_key: Aggregation key of the hypothesis. The aggregation key
        is a composite key used to group syntactic hypotheses from the same previous 
        semantic hypothesis. It is constructed as follows:
        `f"{semantic_source_hypothesis_idx}-{semantic_data.unique_key}"`
    :type aggregation_key: str
    :param token_id: Token id of the semantic token.
    :type token_id: int
    :param score: Score of the hypothesis. Is calculated from the scores of the
        syntactic hypotheses.
    :type score: torch.Tensor
    :param semantic_source_beam_idx: Index of the semantic source hypothesis in the 
        beam. Is also the same as the first char in the aggregation key. and the
        the same as all `syntactic_hypotheses`'s semantic_source_hypothesis_idx.
    :type semantic_source_beam_idx: int
    :param syntactic_hypotheses: Tuple of syntactic hypotheses that are part of the
        semantic hypothesis.
    :type syntactic_hypotheses: Tuple[SyntacticHypothesis, ...]
    :param source_data: Original data of the continuation. This data is in raw format
        (as output by the model) and optional, as it is highly unefficient to store.
    """
    aggregation_key: str
    token_id: int
    score: torch.Tensor
    semantic_source_beam_idx: int
    syntactic_hypotheses: Tuple[SyntacticHypothesis, ...]
    source_data: Optional[OriginalContinuationData]

    def __len__(self) -> int:
        return len(self.syntactic_hypotheses)

    def __repr__(self) -> str:
        return f"\nSemHyp({self.aggregation_key}, {self.token_id}, {self.score}, #{len(self.syntactic_hypotheses)})"

    def __str__(self) -> str:
        return self.__repr__()

    def __lt__(self, other: SemanticToken) -> bool:
        return self.score < other.score

    @classmethod
    def create_empty(
        cls,
        semantic_empty_token: str,
        empty_token_id: torch.Tensor,
        device: str = "cpu",
        empty_score: float = -1e9,
        pkv_like: torch.Tensor = None
    ):
        return SemanticToken(
            f"e-{semantic_empty_token}",
            torch.tensor([empty_token_id], device=device),
            torch.tensor([empty_score], device=device),
            -1,
            (SyntacticHypothesis.create_empty(
                empty_token_id,
                device=device,
                empty_score=empty_score,
                pkv_like=pkv_like
            ),),
            None
        )

@dataclass
class SyntacticHypothesis:
    """ 
    Contains all the data necessary to continue the generation of a sequence.
 
    :param aggregation_key: Aggregation key of the hypothesis. The aggregation key
        is a composite key used to group syntactic hypotheses from the same previous
        semantic hypothesis. It is constructed as follows:
        `f"{semantic_source_hypothesis_idx}-{semantic_data.unique_key}"`
    :type aggregation_key: str
    :param semantic_source_hypothesis_idx: Index of the semantic hypothesis that
        was used to generate the syntactic hypothesis.
    :type semantic_source_hypothesis_idx: torch.Tensor
    :param syntactic_source_hypothesis_idx: Index of the syntactic hypothesis that
        was used to generate the syntactic hypothesis.
    :type syntactic_source_hypothesis_idx: torch.Tensor
    :param hypothesis_idx: Index of the hypothesis. This is the index of the hypothesis at 
        the current generation step. It is used to identify the hypothesis in the aggregation of
        the next step. Should be updated as soon as order of hypotheses changes.
    :type hypothesis_idx: int
    :param path_score: Score of the path. The path score is the sum of the scores of the
        syntactic hypotheses that make up the path (sum of log probabilities which equals 
        multiplication of probabilities).
    :type path_score: Optional[torch.Tensor]
    :param normalized_path_score: Normalized path score. The normalized path score is the
        path score of all semantic tokens normalized through a log softmax. This is used to
        calculate the score of the semantic hypothesis.
    :type normalized_path_score: Optional[torch.Tensor]
    :param semantic_data: Data that ties the syntactic hypothesis to a semantic hypothesis.
        The semantic data contains the unique key which is part of the composite key used
        to group hypotheses. See `aggregation_key` for more information.
    :type semantic_data: SemanticData
    :param syntactic_hypothesis: Data necessary to continue the generation of the sequence.
    :type syntactic_hypothesis: SyntacticHypothesisContinuationData
    :param metadata: Metadata of the syntactic hypothesis.
    :type metadata: Optional[SyntacticHypothesisMetaData]
    :param is_aggregation_key_complete: Flag to indicate if the aggregation key is complete.
        The aggregation key is complete if both the source_hypothesis_idx and the semantic_data
        are set. This flag is checked when grouping hypotheses.
    :type is_aggregation_key_complete: bool, defaults to False
    :param is_normalized_path_score_calculated: Flag to indicate if the normalized path score
    :type is_normalized_path_score_calculated: bool, defaults to False
    """
    aggregation_key: str
    semantic_source_hypothesis_idx: torch.Tensor
    syntactic_source_hypothesis_idx: torch.Tensor
    hypothesis_idx: int
    path_score: torch.Tensor
    normalized_path_score: torch.Tensor
    semantic_data: SemanticData
    syntactic_hypothesis: SyntacticHypothesisContinuationData
    metadata: SyntacticHypothesisMetaData
    is_aggregation_key_complete: bool = False
    is_normalized_path_score_calculated: bool = False

    def __len__(self) -> int:
        return self.syntactic_hypothesis.sequences.shape[-1]

    def __eq__(self, other: SyntacticHypothesis) -> bool:
        return torch.equal(self.syntactic_hypothesis.sequences, other.syntactic_hypothesis.sequences)

    def __hash__(self) -> int:
        return hash(tuple(self.syntactic_hypothesis.sequences.flatten().tolist()))

    def __str__(self) -> str:
        return f"SyntacticHypothesis({self.aggregation_key}, semantic_source_hypothesis_idx={self.semantic_source_hypothesis_idx}, path_score[normalized]={self.path_score}[{self.normalized_path_score}], syntactic_hypothesis={len(self.syntactic_hypothesis)}, metadata={self.metadata}, is_aggr_key_complete={self.is_aggregation_key_complete}, is_norm_path_score_calced={self.is_normalized_path_score_calculated})"

    @classmethod
    def create_empty(
        cls,
        empty_token_id: int,
        device: str = "cpu",
        empty_score: float = -1e9,
        pkv_like: torch.Tensor = None
    ) -> SyntacticHypothesis:
        syntactic_hypothesis = SyntacticHypothesisContinuationData.create_empty(
            empty_token_id,
            device=device,
            score=empty_score,
            pkv_like=pkv_like
        )

        return SyntacticHypothesis(
            f"e-{empty_token_id}",
            torch.tensor(-1, device=device),
            torch.tensor(-1, device=device),
            -1,
            torch.tensor(empty_score, device=device),
            torch.tensor(empty_score, device=device),
            SemanticData("", -1, -1, "", None, None, False),
            syntactic_hypothesis,
            SyntacticHypothesisMetaData(-1),
            False,
            False
        )

@dataclass
class SemanticData:
    """ 
    Contains data which ties sytactic hypotheses to a semantic hypothesis.

    :param unique_key: Unique key of the semantic hypothesis. This key is used
        to identify the semantic hypothesis.
    :type unique_key: str
    :param start: Start index of the entity in the decoded sequence.
    :type start: int
    :param end: End index of the entity in the decoded sequence.
    :type end: int
    :param _type: Type of the semantic data (f.e. entity type).
    :type _type: str
    :param amount_of_chunks: Amount of chunks the semantic data was merged from.
    :type amount_of_chunks: Optional[int]
    :param other: Other data that is relevant for the semantic data. Can be 
        used to store additional information or comments.
    :type other: Optional[any]
    :param has_semantic_data: Flag to indicate if the hypothesis has semantic data.
    :type has_semantic_data: bool, defaults to True
    :param is_eos_token: Flag to indicate if the token is an end of sequence token.
    :type is_eos_token: bool, defaults to False
    """
    unique_key: str
    start: int
    end: int
    _type: str
    amount_of_chunks: Optional[int]
    other: Optional[any]
    has_semantic_data: bool = True
    is_eos_token = False

    def __str__(self) -> str:
        return f"SemanticData({self.unique_key}, {self.start}, {self.end}, {self._type}, {self.amount_of_chunks}, {'Other: Available' if self.other is not None else 'Other: None'})"

    @classmethod
    def create_empty(
        cls,
        unique_key: str,
    ) -> SemanticData:
        return SemanticData(
            unique_key,
            -1,
            -1,
            "",
            -1,
            None,
            False,
        )
        

@dataclass
class SyntacticHypothesisData(ABC):
    """ 
    Contains all the sliced data necessary to continue the generation of a sequence.

    :param sequences: Sequence of token ids of shape (, sequnce_length)
    :type sequences: torch.Tensor
    :param transition_scores: Transition scores of the tokens at generation steps. 
        The transition_scores are not of the same shape as the scores, instead only
        the scores of the hypothesis itself are kept. The shape is therefore
        (, sequence_length). All transition_scores are kept here.
    :type transition_scores: torch.Tensor
    :param generated_transition_scores: Transition scores of all syntactic tokens
        generated since the last semantic data. Is therefore tail of the transition_scores
        and of shape (, last_generated_syntactic_sequence_length).
    :type generated_transition_scores: torch.Tensor
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
    """
    sequences: torch.Tensor
    transition_scores: torch.Tensor
    generated_transition_scores: torch.Tensor
    last_beam_scores: torch.Tensor
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    attention_mask: torch.Tensor

    def __repr__(self) -> str:
        pkv_len_0 = len(self.past_key_values)
        pkv_len_1 = len(self.past_key_values[0])
        pkv_shape = self.past_key_values[0][0].shape
        return f"{self.__class__.__name__}(sequences={self.sequences}, transition_scores={self.transition_scores}, generated_transition_scores={self.generated_transition_scores}, last_beam_scores={self.last_beam_scores}, past_key_values=(Shape [{pkv_len_0}, {pkv_len_1}, {pkv_shape}, attention_mask={self.attention_mask})"
 
    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        """ 
        The length of the sequences tensor.
        """
        return self.sequences.shape[-1]
    
    def stack_past_key_values(
        self
    ) -> torch.Tensor:
        kv_pairs = tuple(
            torch.stack(layer) for layer in self.past_key_values
        )
        return torch.stack(kv_pairs).clone()

    @classmethod
    def unbind_past_key_values(
        cls,
        past_key_values: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        layer_tuples = torch.unbind(past_key_values, dim=0)
        layers_and_kv_tuples = tuple(
            tuple(torch.unbind(layer, dim=0)) for layer in layer_tuples
        )
        return layers_and_kv_tuples

    @classmethod
    def create_empty(
        cls,
        empty_token_id: int,
        device: str = "cpu",
        score: float = -1e9,
        sequence_length: int = 2,
        pkv_like: torch.Tensor = None,
        # pkv_shape: Tuple[int, ...] = None
    ):
        empty_sequence = torch.full((sequence_length,), empty_token_id, dtype=torch.long).to(device)
        empty_scores = torch.full((sequence_length,), score, dtype=torch.float).to(device)
        empty_generated_scores = torch.full((max(sequence_length -1, 1),), score, dtype=torch.float).to(device)
        empty_last_beam_scores = torch.tensor(score, dtype=torch.float).to(device)
        
        empty_past_key_values = torch.zeros_like(pkv_like[:, :, :, :, :sequence_length-1, :], dtype=torch.float).to(device)
        empty_past_key_values = cls.unbind_past_key_values(empty_past_key_values)
        
        empty_attention_mask = torch.zeros_like(empty_sequence, dtype=torch.long).to(device)
        
        return cls(
            sequences=empty_sequence,
            transition_scores=empty_scores,
            generated_transition_scores=empty_generated_scores,
            last_beam_scores=empty_last_beam_scores,
            past_key_values=empty_past_key_values,
            attention_mask=empty_attention_mask,
        )

@dataclass
class SyntacticHypothesisContinuationData(SyntacticHypothesisData):
    unshortened_data: Optional[SyntacticHypothesisUnshortenedContinuationData] = None

    def __repr__(self):
        # Call the superclass's __repr__ method and include the new attribute
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, unshortened_data={'Available' if self.unshortened_data is not None else 'None'})"

    def __str__(self):
        return self.__repr__()

class SyntacticHypothesisUnshortenedContinuationData(SyntacticHypothesisData):
    pass    
    
@dataclass
class SyntacticHypothesisMetaData:
    tokens_shortened: int

# legacy dataclasses
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

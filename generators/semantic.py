import torch
from typing import List, Tuple, Union, Dict, Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum

import torch.utils
from semantic_model import SemanticModelFactory, SemanticDataModelOutputType
from data_structures import SemanticData, SyntacticHypothesis, SemanticToken, SyntacticHypothesisContinuationData

class SemanticTokenizer:
    """ 
    The semantic tokenizer is responsible for tokenizing the semantic tokens.
    It is dynamic since a semantic token can be anything and the token has to be a singular instance.

    It is composed of lookup tables for the tokens and their corresponding strings.
    """
    def __init__(
            self,
            initial_tokens: Optional[List[str]] = None,
            bos_token: str = "<bos>",
            eos_token: str = "<eos>",
            pad_token: str = "<pad>",
            empty_token: str = "<empty>"
        ):
        self.str_to_tokens = {}
        self.str_to_tokens[pad_token] = 0
        self.str_to_tokens[bos_token] = 1
        self.str_to_tokens[eos_token] = 2
        self.str_to_tokens[empty_token] = 3
        if initial_tokens is not None:
            # amount of keys
            offset = len(self.str_to_tokens.keys())
            initial_tokens = {
                key: idx + offset for idx, key in enumerate(initial_tokens)
            }
            initial_tokens.update(self.str_to_tokens)
            self.str_to_tokens = initial_tokens
        self.tokens_to_str = {v: k for k, v in self.str_to_tokens.items()}
        self.bos_token_id = self.str_to_tokens[bos_token]
        self.eos_token_id = self.str_to_tokens[eos_token]
        self.pad_token_id = self.str_to_tokens[pad_token]
        self.empty_token_id = self.str_to_tokens[empty_token]

    @property
    def vocab_size(self) -> int:
        return len(self.str_to_tokens)

    def __len__(self) -> int:
        return len(self.str_to_tokens.keys())

    def __str__(self) -> str:
        return f"SemanticTokenizer with {len(self.str_to_tokens.keys())} tokens.\nBOS token: {self.bos_token_id}\nEOS token: {self.eos_token_id}\nPAD token: {self.pad_token_id}"

    def __call__(
        self,
        sequences: List[List[str]],
        ) -> Dict[str, torch.Tensor]:
        longest_sequence = max(
            len(sequence) for sequence in sequences
        )
        tokenized_sequences = torch.full((len(sequences), longest_sequence), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.full((len(sequences), longest_sequence), 1, dtype=torch.long)
        for sequence_idx, list_of_string_tokens in enumerate(sequences):
            if len(list_of_string_tokens) == 0:
                # update attention mask accordingly
                attention_mask[sequence_idx, :] = 0
            for string_tok_idx, string_token in enumerate(list_of_string_tokens):
                if string_token not in self.str_to_tokens.keys():
                    self._update_semantic_token_lookup(string_token)
                if string_tok_idx == 0:
                    # check if need padding
                    if len(list_of_string_tokens) < longest_sequence:
                        # need for padding
                        padding = [self.pad_token_id] * (longest_sequence - len(list_of_string_tokens))
                        padding_after = [self.pad_token_id] * (longest_sequence - len(padding) - 1)
                        tokenized_sequences[sequence_idx, :] = torch.tensor(
                            padding + [self.str_to_tokens[string_token]] + padding_after # padding after will be replaced in next iter
                        )
                        attention_mask[sequence_idx, :longest_sequence-len(list_of_string_tokens)] = 0
                    else:
                        # no need for padding
                        tokenized_sequences[sequence_idx, string_tok_idx] = self.str_to_tokens[string_token]
                else:
                    replace_at = longest_sequence - (len(list_of_string_tokens) - string_tok_idx)
                    tokenized_sequences[sequence_idx, replace_at] = self.str_to_tokens[string_token]
        return {
            "input_ids": tokenized_sequences,
            "attention_mask": attention_mask
        }

    def encode(self, sequence: List[str]) -> torch.Tensor:
        return self([sequence])["input_ids"][0]

    def _update_semantic_token_lookup(
            self,
            entity: str,
            skip_inverting: bool = False
        ) -> None:
            if entity in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                # simple way to avoid using the special token
                entity = "_".join(entity)
            if entity not in self.str_to_tokens.keys():
                self.str_to_tokens[entity] = len(self.str_to_tokens.keys())
                if not skip_inverting:
                    self.tokens_to_str = {v: k for k, v in self.str_to_tokens.items()}

    def _update_multiple_semantic_token_lookup(
            self,
            entities: List[str],
        ) -> None:
            for entity in entities:
                self.update_semantic_token_lookup(
                    entity,
                    skip_inverting=True
                )
            self.tokens_to_str = {v: k for k, v in self.str_to_tokens.items()}

    def batch_decode(self, sequences: torch.Tensor) -> List[List[str]]:
        decoded_sequences = []
        for sequence in sequences:
            decoded_sequence = self._decode_sequence(sequence)
            decoded_sequences.append(decoded_sequence)
        return decoded_sequences

    def decode(self, token: Union[torch.Tensor, int]) -> str:
        try:
            if isinstance(token, torch.Tensor):
                token = token.item()
            return self.tokens_to_str[token]
        except KeyError as token:
            raise KeyError(f"Token {token} not known by the tokenizer.")

    def _decode_sequence(self, sequence: torch.Tensor) -> List[str]:
        return [
            self.decode(token) for token in sequence
        ]


class SemanticGenerator:
    """ 
    The semantic generator is responsible for generating semantic tokens from text.
    It uses NER models to generate the semantic tokens and aggregates the scores of the entities.
    
    :param semantic_models: List of semantic model names.
    :type semantic_models: List[str]
    :param normalize_unique_key: Normalize the unique key of the semantic data.
    :type normalize_unique_key: bool
    :param unique_key: Key to aggregate the semantic data on.
    :type unique_key: Literal["word", "text", "type"]
    :param device: Device to run the model on.
    :type device: str
    """
    def __init__(
        self,
        semantic_models: Union[List[str], str],
        normalize_unique_key: bool,
        unique_key: Literal["word", "text", "type"],
        device: str = "cpu",
    ):
        self.model_names = semantic_models
        print(f"Loading semantic model(s): {semantic_models}")
        self.semantic_models = SemanticModelFactory.create(semantic_models, device, normalize_unique_key)
        print(f"Semantic model(s): {semantic_models}")
        self.unique_key = unique_key
        self.tokenizer = SemanticTokenizer()
        self.low_score = -1e9

    def generate(
        self,
        texts: List[str],
        input_length_chars: torch.Tensor,
        include_all = False,
        syntactic_sequences: Optional[torch.Tensor] = None,
        syntactic_eos_token_id: Optional[int] = None
    ) -> Tuple[List[Union[SemanticData, None]], List[List[Union[SemanticData, None]]]]:
        """ 
        Generate the semantic data from the input text. Returns the first semantic data
        as well as all semantic data.

        :param text: List of input text.
        :type text: List[str]
        :param input_length_chars: Length of the input text in characters. This is used
            to filter out the semantic data that is not part of the generated text (e.g.
            part of the input prompt). If inclusion of semantic data from prompt is needed
            set include_all to True.
        :type input_length_chars: torch.Tensor
        :param include_all: Include all semantic data, even if it is part of the input text.
                This will ignore the input_length_chars parameter.
        :type include_all: bool, defaults to False
        :param sequences: Sequences of syntactic hypotheses. Used to generate semantic eos tokens.
        :type sequences: torch.Tensor, defaults to None
        :param syntactic_eos_token_id: ID of the eos token in the syntactic model. Is needed to
            find the eos token in the sequences.
        :type syntactic_eos_token_id: int, defaults to None
        :return: Tuple of the first semantic data and all semantic data.
        :rtype: Tuple[List[Union[SemanticData, None]], List[List[Union[SemanticData, None]]]]
        """
        semantic_data_chunks = self.semantic_models[0].predict(texts)
        (
            _,
            all_semantic_data_chunks 
        ) = self.semantic_models[0].get_generated_semantic_data(
                    semantic_data_chunks,
                    input_length_chars,
                    include_all=include_all
                )
        merged_semantic_data_points = self.semantic_models[0].merge_semantic_data(
            all_semantic_data_chunks
        )
        semantic_datas = self.semantic_models[0].to_generic_semantic_data(
            merged_semantic_data_points,
            self.unique_key,
            syntactic_sequences,
            syntactic_eos_token_id,
            self.tokenizer.decode(torch.tensor(self.tokenizer.eos_token_id)),
            semantic_data_chunks
        )
        first_semantic_datas = [
            hyp[0] for hyp in semantic_datas
        ]
        return first_semantic_datas, semantic_datas

    def compute_semantic_tokens(
        self,
        hypotheses: List[SyntacticHypothesis],
        syntactic_beam_size: int,
        semantic_beam_size: int,
    ) -> List[SemanticToken]:
        # check if all aggregation_keys are complete
        all_keys_complete = all(
            [hyp.is_aggregation_key_complete for hyp in hypotheses]
        )
        if not all_keys_complete:
            raise ValueError("Not all aggregation keys are complete. \
                Ensure you update them before proceeding.")

        # here, we unfortunately cannot rely on indices anymore, as the 
        # hypotheses no longer need to be of size (batch_size, beam_size)
        # as hypotheses wo semantic data are removed.
        # Therefore, calculate without tensor and view for logsoftmax simplification
        # info: logsoftmax or logsumexp can be in any order
        # create list of batches of hypotheses [batch_size, beam_size]

        batch_hyps = [hypotheses[i:i+syntactic_beam_size] for i in range(0, len(hypotheses), syntactic_beam_size)]

        # a. need to remove duplicate hypotheses (same syntactic hyp sequence)
        # as they stand for the same path in the model and the probabilities of it
        # should not be counted twice (implementation is same as using a set, but this
        # version keeps the order)
        unique_batch_hyps = [
            list(dict.fromkeys(batch)) for batch in batch_hyps
        ]

        # b. only keep the hypotheses with semantic data (optionally add exceptions below)
        unique_batch_hyps_w_sem_data = [
            [hyp for hyp in batch if hyp.semantic_data.has_semantic_data] for batch in unique_batch_hyps
        ]

        ### 1. Exception to add: no semantic data at all ###
        # check that all batches have at least one semantic token, if not
        # add the hyps wo semantic data with semantic data marked as empty
        # -> this will lead to tokens in respective beams carrying synt hyps
        # aggregated by their unique key (which would then only be the source beam)
        # -> all get carried over to the next step
        # ? the batches do not need to be altered w the aggregation key, as
        # ? they will all be brougth over anyways (they are the only tokens returned from the beam scorer)
        if not all([len(batch) > 0 for batch in unique_batch_hyps_w_sem_data]):
            for batch_idx, batch in enumerate(unique_batch_hyps_w_sem_data):
                if len(batch) == 0:
                    # create custom semantic data with empty token
                    semantic_data = SemanticData.create_empty(
                        self.tokenizer.decode(torch.tensor(self.tokenizer.empty_token_id))
                    )
                    batch_wo_any_sem_data = unique_batch_hyps[batch_idx]
                    for hyp in batch_wo_any_sem_data:
                        hyp.semantic_data = semantic_data
                    unique_batch_hyps_w_sem_data[batch_idx] = batch_wo_any_sem_data

        # normalize the path scores
        grouped_by_sem_source_hyp = self._group_syntactic_hyps_by_sem_source_hyp(
            [hyp for batch in unique_batch_hyps_w_sem_data for hyp in batch],
            semantic_beam_size, len(hypotheses) // syntactic_beam_size
        )
        for batch_idx, batch in enumerate(grouped_by_sem_source_hyp):
            if len(batch) == 0:
                continue
            # normalize the path scores
            path_scores = torch.cat([hyp.path_score.unsqueeze(0) for hyp in batch])
            normalized_path_scores = torch.log_softmax(path_scores, dim=-1)
            # slight penalty for the not eos semantic token
            for hyp_idx, synt_hyp in enumerate(batch):
                synt_hyp.normalized_path_score = normalized_path_scores[hyp_idx]
                synt_hyp.is_normalized_path_score_calculated = True

        ### 2. Exception to add: only eos tokens in a batch ###
        # check that there is not only eos in a batch (if so, add alternatives like
        # if there is no semantic data at all. This will add #beam_size amount of empty
        # semantic tokens which will be kept and carried over)
        batches_w_only_eos = [
            all([
                hyp.semantic_data.is_eos_token for hyp in batch
                ]) for batch in unique_batch_hyps_w_sem_data
            ]
        if any(batches_w_only_eos):
            # an eos hyp alternative needs to exist, therefore add empty semantic data
            alternative_hyps =  [
                [hyp for hyp in batch if not hyp.semantic_data.has_semantic_data] for batch in unique_batch_hyps
            ]
            # normalize the path scores
            # these are all but the eos token which was the only semantic token
            # therefore, these are all normalized within themselves (leaving the eos
            # token as-is). This means, that in the next iteration the token is appropriately
            # weighted. The score will be lowered during the generation of the semantic token
            # as this is not a true semantic token (artificial to carry on generating).
            # ? the batches do not need to be altered w the aggregation key, as
            # ? they will all be brougth over anyways (they are the only tokens returned from the beam scorer)
            alternatives_grouped_by_sem_source_hyp = self._group_syntactic_hyps_by_sem_source_hyp(
                [hyp for batch in alternative_hyps for hyp in batch],
                semantic_beam_size, len(hypotheses) // syntactic_beam_size
            )
            for batch_idx, batch in enumerate(alternatives_grouped_by_sem_source_hyp):
                if len(batch) == 0:
                    continue
                # normalize the path scores
                path_scores = torch.cat([hyp.path_score.unsqueeze(0) for hyp in batch])
                normalized_path_scores = torch.log_softmax(path_scores, dim=-1)
                # slight penalty for the not eos semantic token (chose not to do this)
                # this is a case which may be a failure. But it will fallback to syntactic beam search
                # even if it may indefinitely keep looking (will see in experiments)
                # normalized_path_scores = (normalized_path_scores * (1 + 1e-4))
                for hyp_idx, synt_hyp in enumerate(batch):
                    synt_hyp.normalized_path_score = normalized_path_scores[hyp_idx]
                    synt_hyp.is_normalized_path_score_calculated = True

            # add the empty semantic data to the batch
            for batch_idx, batch in enumerate(alternative_hyps):
                if not batches_w_only_eos[batch_idx]:
                    continue
                # create custom semantic data with empty token
                semantic_data = SemanticData.create_empty(
                    self.tokenizer.decode(torch.tensor(self.tokenizer.empty_token_id))
                )
                batch_wo_any_sem_data = alternative_hyps[batch_idx]
                for hyp in batch_wo_any_sem_data:
                    hyp.semantic_data = semantic_data
                unique_batch_hyps_w_sem_data[batch_idx].extend(batch_wo_any_sem_data)

        return [sorted(self._create_semantic_hypotheses(batch_hyps), reverse=True) for batch_hyps in unique_batch_hyps_w_sem_data]

    def _group_syntactic_hyps_by_sem_source_hyp(
        self,
        hypotheses: List[SyntacticHypothesis],
        num_semantic_beams: int,
        batch_size: int,
    ) -> List[List[SyntacticHypothesis]]:
        all_synt_hyps = [[] for _ in range(batch_size * num_semantic_beams)]
        for hypothesis in hypotheses:
            all_synt_hyps[hypothesis.semantic_source_hypothesis_idx].append(hypothesis)
        return all_synt_hyps

    def _create_semantic_hypotheses(
        self,
        hypotheses: List[SyntacticHypothesis],
    ) -> List[SemanticToken]:
        # a) groub by semantic source hyp and by entity
        aggr_key_dict = self._group_by_aggregation_key(hypotheses)
        # b) create semantic hypotheses (also calculates semantic scores)
        all_semantic_hypotheses = []
        for key, value in aggr_key_dict.items():
            sem_hyp = self._create_semantic_hypothesis(value)
            all_semantic_hypotheses.append(sem_hyp)
        return all_semantic_hypotheses

    def _group_by_aggregation_key(
        self,
        hypotheses: List[SyntacticHypothesis]
    ) -> Dict[str, SyntacticHypothesis]:
        aggr_key_dict = {}
        for hypothesis in hypotheses:
            if hypothesis.aggregation_key not in aggr_key_dict.keys():
                aggr_key_dict[hypothesis.aggregation_key] = []
            aggr_key_dict[hypothesis.aggregation_key].append(hypothesis)
        return aggr_key_dict

    def _create_semantic_hypothesis(
        self,
        hypotheses: List[SyntacticHypothesis]
    ) -> SemanticToken:
        """ 
        Merge the passed hypotheses into a single SemanticToken. Calculates
        the semantic scores and creates the SemanticToken instance.
        """
        device = hypotheses[0].path_score.device
        # if all do not have semantic data, then use empty semantic token to carry 
        # all hypotheses over
        if not all([hyp.semantic_data.has_semantic_data for hyp in hypotheses]):
            semantic_score = self._compute_semantic_scores(hypotheses)
            return SemanticToken(
                hypotheses[0].aggregation_key,
                torch.tensor(self.tokenizer.empty_token_id).unsqueeze(0).to(device),
                semantic_score.to(device),
                hypotheses[0].semantic_source_hypothesis_idx,
                tuple(hypotheses),
                None
            )
        # compute semantic scores
        semantic_score = self._compute_semantic_scores(hypotheses)
        semantic_token_id = self.tokenizer.encode([hypotheses[0].semantic_data.unique_key])
        # now create the SemanticToken
        return SemanticToken(
            hypotheses[0].aggregation_key,
            semantic_token_id.to(device),
            semantic_score.to(device),
            hypotheses[0].semantic_source_hypothesis_idx,
            tuple(sorted(hypotheses, reverse=True)),
            None
        )

    def create_empty_semantic_token(
        self,
        semantic_tokens: List[List[List[SemanticToken]]],
        syntactic_empty_token_id: torch.Tensor # any token that is not special (can be random)
    ) -> SemanticToken:
        """ 
        This method creates an empty semantic token. The token is empty, does not contain relevant
        information and is used to fill semantic tokens, were none are present. Decoding cannot continue
        with empty semantic tokens.
        """
        def fetch_pkv_from_first_non_empty_sem_tok(
            semantic_tokens: List[List[List[SemanticToken]]],
            empty_token_id: int
        ) -> Tuple[SyntacticHypothesisContinuationData, torch.device]:
            for batch in semantic_tokens:
                for beam in batch:
                    for sem_token in beam:
                        if sem_token.token_id != empty_token_id:
                            syntactic_hyp = sem_token.syntactic_hypotheses[0].syntactic_hypothesis
                            device = sem_token.score.device
                            return syntactic_hyp, device
            # if no non-empty semantic token is found, there are semantic tokens with all hyps
            # use first semantic token
            for batch in semantic_tokens:
                for beam in batch:
                    for sem_token in beam:
                        syntactic_hyp = sem_token.syntactic_hypotheses[0].syntactic_hypothesis
                        device = sem_token.score.device
                        return syntactic_hyp, device
        
        non_empty_hyp, device = fetch_pkv_from_first_non_empty_sem_tok(semantic_tokens, self.tokenizer.empty_token_id)
        pkv_shape = tuple([len(non_empty_hyp.past_key_values), len(non_empty_hyp.past_key_values[0]), *non_empty_hyp.past_key_values[0][0].shape])
        pkv_device_map = tuple(key_or_value[0].device for key_or_value in non_empty_hyp.past_key_values)

        # use padding token id; empty token is reserved for semantic token shell
        return SemanticToken.create_empty(
            self.tokenizer.decode(torch.tensor(self.tokenizer.empty_token_id)),
            self.tokenizer.pad_token_id,
            syntactic_empty_token_id,
            device,
            self.low_score,
            pkv_shape=pkv_shape,
            pkv_device_map=pkv_device_map,
        )

    def _compute_semantic_scores(self, hypotheses: List[SyntacticHypothesis]) -> torch.Tensor:
        all_normalized = all(
            [hyp.is_normalized_path_score_calculated for hyp in hypotheses]
        )
        if not all_normalized:
            raise ValueError("Not all path scores have been normalized. \
                Ensure you normalize them before proceeding.")
        normalized_path_scores = torch.stack([hyp.normalized_path_score for hyp in hypotheses], dim=0)
        # accumulate them
        # ? error of around 1e-6-1e-7 to be expected (if all probabilities accumulated and exp == 1)
        # could increase precision to float.64, but only improvements ~e-1
        semantic_score = torch.logsumexp(normalized_path_scores, dim=0).unsqueeze(0)
        return semantic_score

    def fill_empty_beam_hyps(
        self,
        semantic_tokens: List[List[List[SemanticToken]]],
        semantic_beam_scores: torch.Tensor,
        syntactic_empty_token_id: torch.Tensor # any token that is not special (can be random)
    ) -> Tuple[List[List[List[SemanticToken]]], torch.Tensor]:
        """ 
        Fill empty beams with empty semantic tokens. This is needed to ensure that all beams
        have at least one semantic token and that the beam scorer can be applied. The score of 
        it is low and it is marked as empty (no meaning for further decoding).
        
        :param semantic_tokens: List of semantic tokens.
        :type semantic_tokens: List[List[List[SemanticToken]]]
        :param semantic_beam_scores: Scores of the semantic beams.
        :type semantic_beam_scores: torch.Tensor
        :param syntactic_empty_token_id: Token id used for syntactic tokens. It is not imporant which
            token is used, as long as it is not a special token (the filled beams receive low beam 
            scores which avoids them being selected).
        :type syntactic_empty_token_id: torch.Tensor
        """
        # create a copy and leave original list untouched
        semantic_tokens = [[beam for beam in batch] for batch in semantic_tokens]
        # check, if all beams have semantic tokens
        all_beams_have_semantic_tokens = all(
            [len(beam) > 0 for batch in semantic_tokens for beam in batch]
        )
        if all_beams_have_semantic_tokens:
            return semantic_tokens, semantic_beam_scores
        empty_semantic_token = self.create_empty_semantic_token(semantic_tokens, syntactic_empty_token_id)

        beam_size = len(semantic_tokens[0])
        batch_size = len(semantic_tokens)
        for batch_idx, batch in enumerate(semantic_tokens):
            # fill empty beams with empty semantic token and give low score
            for beam_idx, beam in enumerate(batch):
                if len(beam) == 0:
                    semantic_tokens[batch_idx][beam_idx] = [empty_semantic_token]
                    semantic_beam_scores[batch_idx*beam_size+beam_idx] = self.low_score

        return semantic_tokens, semantic_beam_scores

    def filter_next_semantic_tokens(
        self,
        semantic_tokens: List[List[List[SemanticToken]]],
        beam_indices: torch.Tensor,
        next_beam_tokens: torch.Tensor,
        beam_size: int,
        padding_token_id: int
    ) -> Tuple[SemanticToken, ...]:
        """ 
        Gather the next semantic tokens for the next beam.
        
        :param semantic_tokens: List of semantic tokens.
        :type semantic_tokens: List[List[List[SemanticToken]]]
        :param beam_indices: Indices of the beams.
        :type beam_indices: torch.Tensor
        :param next_beam_tokens: Tokens of the next beam.
        :type next_beam_tokens: torch.Tensor
        :param beam_size: Size of the beam.
        :type beam_size: int
        :return: Tuple of the next semantic tokens. Of size (batch_size * beam_size,).
        :rtype: Tuple[SemanticToken, ...]
        """
        all_sem_hyps = [None for _ in range(len(semantic_tokens*beam_size))]
        for beam_batch_idx_idx, beam_batch_idx in enumerate(beam_indices):
            batch_idx = beam_batch_idx // beam_size
            beam_idx = beam_batch_idx % beam_size
            # find element in batch, beam with index and element
            # sem_tok_idx = None
            sem_tok = None
            for sem_token_idx, sem_token in enumerate(semantic_tokens[batch_idx][beam_idx]):
                # do not add padding tokens to last_tokens
                if sem_token.token_id == padding_token_id:
                    break
                if sem_token.token_id == next_beam_tokens[beam_batch_idx_idx]:
                    # sem_tok_idx = sem_token_idx
                    sem_tok = sem_token
                    break
            all_sem_hyps[beam_batch_idx_idx] = sem_tok
        return tuple(all_sem_hyps)

    def gather_semantic_tokens_by_index(
        self,
        semantic_tokens: List[List[List[SemanticToken]]],
        next_indices: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> List[List[SemanticToken]]:
        matching_sem_toks = [[] for _ in range(next_tokens.shape[0])]

        for batch_idx, (next_batch_indices, next_batch_tokens) in enumerate(zip(next_indices, next_tokens)):
            for (next_beam_idx, next_token) in zip(next_batch_indices, next_batch_tokens):
                matching_sem_tok = None
                for sem_token in semantic_tokens[batch_idx][next_beam_idx]:
                    if sem_token.token_id == next_token:
                        matching_sem_tok = sem_token
                        break
                matching_sem_toks[batch_idx].append(matching_sem_tok if matching_sem_tok is None else matching_sem_tok.clone())
        return matching_sem_toks

    def gather_next_tokens(
        self,
        semantic_tokens: List[List[List[SemanticToken]]],
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dummy_token = self.tokenizer.pad_token_id
        dummy_score = self.low_score

        all_next_tokens = [
            torch.cat([semantic_token.token_id for semantic_token in beam]) for batch in semantic_tokens for beam in batch
        ]
        all_next_tokens_padded = torch.nn.utils.rnn.pad_sequence(all_next_tokens, batch_first=True, padding_value=dummy_token).to(device)
        all_next_scores = [
            torch.cat([semantic_token.score for semantic_token in beam]) for batch in semantic_tokens for beam in batch
        ]
        all_next_scores_padded = torch.nn.utils.rnn.pad_sequence(all_next_scores, batch_first=True, padding_value=dummy_score).to(device)

        return all_next_tokens_padded, all_next_scores_padded

    def gather_tokens_by_source_beam(
        self,
        semantic_tokens: List[List[SemanticToken]],
        batch_size: int,
        beam_size: int
    ) -> List[List[List[SemanticToken]]]:
        all_batches = [
            [[] for _ in range(beam_size)] for _ in range(batch_size)
        ]
        for batch_idx, batch in enumerate(semantic_tokens):
            for semantic_token in batch:
                source_beam = semantic_token.semantic_source_beam_idx
                all_batches[batch_idx][source_beam % beam_size].append(semantic_token)
        return all_batches
    
    def _group_by_sem_source_hyp_and_by_entity_legacy(
        self,
        syntactic_hyp_prob: torch.Tensor,
        entities: SemanticDataModelOutputType,
        syn_to_sem_hyp_mapping: torch.Tensor,
        batch_size: int,
        num_of_beams: int,
    ):
        semantic_tokens_from_syntactic_tokens = [
            {} for _ in range(batch_size)
        ]
        no_semantic_token = [
            [] for _ in range(batch_size)
        ]
        for idx, (prob, entity, sem_source_hyp) in enumerate(zip(syntactic_hyp_prob, entities, syn_to_sem_hyp_mapping)):
            batch_idx = idx // num_of_beams
            prob = prob.unsqueeze(0)

            # if no entity, not of interest
            if len(entity) == 0:
                no_semantic_token[batch_idx].append(prob)
                continue
            if sem_source_hyp.item() not in semantic_tokens_from_syntactic_tokens[batch_idx].keys():
                # if sem_source_hyp has never been seen, add first entity with a value
                semantic_tokens_from_syntactic_tokens[batch_idx][sem_source_hyp.item()] = {}
                semantic_tokens_from_syntactic_tokens[batch_idx][sem_source_hyp.item()][entity[0][self.unique_key]] = prob
            else:
                if entity[0][self.unique_key] not in semantic_tokens_from_syntactic_tokens[batch_idx][sem_source_hyp.item()].keys():
                    # sem beam exists, but entity[0] is new (add with value)
                    semantic_tokens_from_syntactic_tokens[batch_idx][sem_source_hyp.item()][entity[0][self.unique_key]] = prob
                else:
                    # sem beam and entity[0] already exist
                    semantic_tokens_from_syntactic_tokens[batch_idx][sem_source_hyp.item()][entity[0][self.unique_key]] = torch.cat(
                        (
                            semantic_tokens_from_syntactic_tokens[batch_idx][sem_source_hyp.item()][entity[0][self.unique_key]],
                            prob
                        )
                    )
        return semantic_tokens_from_syntactic_tokens, no_semantic_token        

    def _group_by_entity(
        self,
        entities: SemanticDataModelOutputType,
        syntactic_hyp_prob: torch.Tensor,
        batch_size: int,
        num_of_beams: int,
        ) -> Tuple[List[Dict[str, torch.Tensor]], List[List[torch.Tensor]]]:
            semantic_tokens_from_syntactic_tokens = [
                {} for _ in range(batch_size)
            ]
            no_semantic_token = [
                [] for _ in range(batch_size)
            ]
            key = self.unique_key
            # zip together the semantic_output and the entity_hyp_probs
            for idx, (entity, prob) in enumerate(zip(entities, syntactic_hyp_prob)):
                batch_idx = idx // num_of_beams
                prob = prob.unsqueeze(0)

                if len(entity.keys()) == 0:
                    no_semantic_token[batch_idx].append(prob)
                    continue
                if entity[key] not in semantic_tokens_from_syntactic_tokens[batch_idx].keys():
                    semantic_tokens_from_syntactic_tokens[batch_idx][entity[key]] = prob
                else:
                    semantic_tokens_from_syntactic_tokens[batch_idx][entity[key]] = torch.cat(
                        (
                            semantic_tokens_from_syntactic_tokens[batch_idx][entity[key]],
                            prob
                        )
                    )
            return semantic_tokens_from_syntactic_tokens, no_semantic_token

    def encode_semantic_sequences_from_semantic_data(self, sequences: List[Union[SemanticData, None]]) -> torch.Tensor:
        unique_keys = [
            [sem_data.unique_key for sem_data in hyp if sem_data is not None] for hyp in sequences
        ]
        return self.tokenizer(unique_keys)

    def encode_semantic_sequences(self, sequences: List[List[str]]) -> torch.Tensor:
        return self.tokenizer(sequences)

    def encode_semantic_sequence(self, sequence: List[str]) -> torch.Tensor:
        return self.tokenizer([sequence]).squeeze(0)

    def expand_semantic_sequences(
            self,
            sequences: torch.Tensor,
            num_of_beams: int
        ) -> torch.Tensor:
        return sequences.repeat_interleave(num_of_beams, dim=0)

    def unpack_semantic_hypotheses(
        self,
        semantic_hyps: Tuple[SemanticToken, ...],
        semantic_beam_size: int,
        syntactic_beam_size: int,
        device: Optional[str] = "cpu"
    ) -> Tuple[List[SyntacticHypothesis], torch.Tensor]:
        all_syntactic_hyps = []
        syn_to_sem_hyp_mapping = []
        
        for sem_hyp_idx, sem_hyp in enumerate(semantic_hyps):
            # this can happen if no #amount_semantic_beams amount of semantic
            # tokens have been generated
            batch_idx = sem_hyp_idx // semantic_beam_size
            if sem_hyp is not None:
                length_of_hyps = len(sem_hyp.syntactic_hypotheses)
                all_syntactic_hyps.extend(list(sem_hyp.syntactic_hypotheses))
                syn_to_sem_hyp_mapping.extend([sem_hyp_idx] * length_of_hyps)
            if (sem_hyp_idx+1) % semantic_beam_size == 0:
                # check if the amount of syntactic hypotheses is full
                amount_to_fill = syntactic_beam_size * (batch_idx+1) - len(all_syntactic_hyps)
                if amount_to_fill > 0:
                    # fill with last hypothesis which will be masked out later
                    all_syntactic_hyps.extend([all_syntactic_hyps[-1]] * amount_to_fill)
                    syn_to_sem_hyp_mapping.extend([syn_to_sem_hyp_mapping[-1]] * amount_to_fill)

        return all_syntactic_hyps, torch.tensor(syn_to_sem_hyp_mapping).to(device)

    def beam_indices_tuple_to_tensor(
        self,
        beam_indices: Tuple[Tuple[torch.Tensor, ...], ...]
    ) -> torch.Tensor:
        long_tensor =  torch.stack([idx for row in beam_indices for idx in row])[:, None]
        batch_size = len(beam_indices)
        beam_size = len(beam_indices[0])
        return long_tensor.view(batch_size, beam_size)

    def calc_semantic_beam_scores(
        self,
        semantic_scores: torch.Tensor,
        semantic_beam_indices: Union[torch.Tensor, Tuple[Tuple[torch.Tensor, ...], ...]],
    ) -> torch.Tensor:
        # need to transpose
        semantic_scores = semantic_scores.transpose(0, 1)
        if isinstance(semantic_beam_indices, tuple):
            semantic_beam_indices = self.beam_indices_tuple_to_tensor(semantic_beam_indices)
        # for the last beam indices: -> chose the tokens as are
        added_indices = torch.arange(0, semantic_beam_indices.shape[0])[:, None].to(semantic_beam_indices.device)
        semantic_beam_indices = torch.cat((semantic_beam_indices, added_indices), dim=1)
        gather_indices = semantic_beam_indices.transpose(0, 1) 

        return semantic_scores.gather(1, gather_indices).transpose(0, 1)

    def calc_next_pure_semantic_scores(
        self,
        pure_token_scores: torch.Tensor,
        beam_next_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        replace_nan_with: float = None
    ) -> torch.Tensor:
        if replace_nan_with is None:
            replace_nan_with = self.low_score
        batch_size = pure_token_scores.shape[0]
        amount_semantic_beams = beam_next_tokens.shape[-1] // batch_size
        beam_next_tokens = beam_next_tokens.view(batch_size, amount_semantic_beams)
        # bring pure_token_scores to same shape as beam_next_tokens

        indices_to_select = []
        limit_to = beam_next_tokens.shape[-1]
        for batch_idx, row in enumerate(beam_next_tokens):
            # get index of values in beam_next_tokens from row
            mask = torch.isin(next_tokens[batch_idx], row)

            idx = torch.nonzero(mask).squeeze(1)[:limit_to]
            if idx.shape[0] < limit_to:
                if idx.shape[0]:
                    # if there is any idx in it already, use it
                    idx = torch.cat((idx, torch.full((limit_to-idx.shape[0],), idx[-1], dtype=torch.long).to(idx.device)))
                else:
                    # so far, completely empty, use first tokens
                    idx = torch.arange(limit_to, dtype=torch.long).to(idx.device)
            indices_to_select.append(idx)
        indices_to_select = torch.cat(indices_to_select).view(
            batch_size, amount_semantic_beams
            )

        selected_next_tokens = next_tokens.gather(1, indices_to_select)
        pure_token_scores = pure_token_scores.gather(1, indices_to_select)

        # replace pad_token scores with whatever score is provided
        pure_token_scores = torch.where(selected_next_tokens == self.tokenizer.pad_token_id, replace_nan_with, pure_token_scores)

        return pure_token_scores

    def compute_transition_scores(
        self,
        semantic_sequences: torch.Tensor,
        semantic_scores: torch.Tensor,
        semantic_beam_indices: Union[torch.Tensor, Tuple[Tuple[torch.Tensor, ...], ...]],
        semantic_tokenizer: SemanticTokenizer,
        semantic_tokens: List,
    ) -> torch.Tensor:
        """ 
        The compute transition works very differently compared to the one from hf. The one from hf
        is a pointer to the right dimension in the scores. Each beam has a full vocabulary of scores.
        With the help of the `beam_indices` (as pointers for the dimension in the scores) and the 
        `sequences` (as getter for the score of that token), the scores are gathered.
        
        Since the vocabulary is dynamic in size for the semantic tokens, this approach does not work here.
        Therefore, things are different. Instead of pointing to the right dimension in the scores, the 
        `beam_indices` can be interpreted as pointers to the previous scores. The first will always be 
        the first index in the batch, as that is where the first beam is created.
        That means that one can gather the scores by applying the `beam_indices` and using them as pointer
        for the previous score level (as they show the beam idx and by that the score of the beam, which was used).
        For the last scores:
        todo: 1. the last tokens are eos tokens:
            - todo
        2. the last tokens are not eos tokens:
            - here, there are no pointers to the last iteration anymore, we can use
              `arange` in the shape of the beam_indices to select the last produced tokens (they are ordered
              by probabilities and will therefore match)
        """
        if isinstance(semantic_beam_indices, tuple):
            semantic_beam_indices = self.beam_indices_tuple_to_tensor(semantic_beam_indices)

        # remove the first beam indices, they are for the first beam chosen (which is always zero)
        semantic_beam_indices = semantic_beam_indices.clone()[:, 1:]

        # need to transpose
        semantic_scores = semantic_scores.clone().transpose(0, 1)
        gather_indices = semantic_beam_indices.transpose(0, 1) 

        # remove the padding at end of beam_indices (they come from input)
        longest_beam_mask = ~(gather_indices == -1).all(dim=1)
        # Apply the mask to remove rows that are all -1
        gather_indices = gather_indices[longest_beam_mask]

        # create mask for early stopping beams
        beam_indices_mask = gather_indices < 0
        # set the indices for those to zero so it can gather index (masked out later)
        gather_indices[beam_indices_mask] = 0

        
        # add arange to the end of the beam_indices
        semantic_sequences = semantic_sequences.clone().transpose(0, 1)
        added_indices_mask = (
            semantic_sequences[-1, :] == semantic_tokenizer.eos_token_id
        ) | (
            semantic_sequences[-1, :] == semantic_tokenizer.pad_token_id
        )
        beam_indices_mask = torch.cat((beam_indices_mask, added_indices_mask.unsqueeze(0)), dim=0)
        added_indices = torch.arange(0, gather_indices.shape[1]).to(gather_indices.device).unsqueeze(0)
        gather_indices = torch.cat((gather_indices, added_indices), dim=0)

        # now gather scores
        transition_scores = semantic_scores.gather(1, gather_indices)
        transition_scores[beam_indices_mask] = 0
        transition_scores = transition_scores.transpose(0, 1)
        
        input_len = semantic_sequences.shape[0] - gather_indices.shape[0]
        eos_mask = semantic_sequences[input_len:, :] == semantic_tokenizer.eos_token_id
        eos_mask = eos_mask.transpose(0, 1)

        for seq_id, sem_tok in enumerate(semantic_tokens):
            if sem_tok is None:
                continue
            try:
                if sem_tok.syntactic_hypotheses[0].semantic_data.is_eos_token:
                    transition_scores[seq_id, eos_mask[seq_id]] = sem_tok.score
            except:
                if sem_tok.token_id == semantic_tokenizer.eos_token_id:
                    transition_scores[seq_id, eos_mask[seq_id]] = sem_tok.score

        return transition_scores


class SemanticGenerationMode(Enum):
    BEAM_SEARCH = "beam_search"
    GREEDY_SEARCH = "greedy_search"


@dataclass
class SemanticGenerationConfig:
    """ 
    Contains the configuration for semantic generation.
    
    :param num_beams: The number of beams to use for semantic beam search.
    :type num_beams: int
    :param length_penalty: The length penalty to use for semantic beam search.
        Values >0 will encourage the model to generate shorter sequences,
        while values <0 will encourage the model to generate longer sequences.
    :type length_penalty: float, defaults to 1.0
    :param early_stopping: Whether to end generation when num of hypotheses have 
        been generated. True stops immediately, False heuristically estimates whther
        a better hypothesis can be found. "never" will look until it is not possible
        to find a better hypothesis.
    :type early_stopping: Union[bool,Literal["never"]], defaults to False
    :param num_return_sequences: The number of returned sequences for each input in the batch.
    :type num_return_sequences: int, defaults to 1
    :param max_length: The maximum length of the semantic tokens to generate.
    :type max_length: int, defaults to None
    :param do_sample: Whether to use sampling for semantic generation (todo, wip)
    :type do_sample: bool, defaults to False
    """
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: Union[bool, Literal["never"]] = False
    num_return_sequences: Optional[int] = None
    max_length: Optional[int] = None
    # ? could also add max_generated_length
    do_sample: bool = False
    max_overall_tokens: int = 1000
    max_overall_generated_tokens: Optional[int] = None
    
    def __post_init__(self):
        if self.num_return_sequences is None:
            self.num_return_sequences = self.num_beams
        if self.max_overall_generated_tokens is None:
            # the max_overall_tokens will finish generation first
            self.max_overall_generated_tokens = self.max_overall_tokens
    
    def get_generation_mode(self) -> SemanticGenerationMode:
        if self.num_beams > 1:
            return SemanticGenerationMode.BEAM_SEARCH
        if self.num_beams == 1:
            return SemanticGenerationMode.GREEDY_SEARCH

import torch
from typing import List, Tuple, Union, Dict, Any, Optional

import torch.utils
from ner_model import SemanticModelFactory, SemanticDataModelOutputType
from data_structures import SemanticData, SyntacticHypothesis, SemanticToken, SyntacticHypothesisContinuationData

class SemanticGenerator:
    """ 
    The semantic generator is responsible for generating semantic tokens from text.
    It uses NER models to generate the semantic tokens and aggregates the scores of the entities.
    
    :param ner_models: List of NER model names.
    :type ner_models: List[str]
    :param unique_key: Key to aggregate the semantic data on.
    :type unique_key: str
    :param device: Device to run the model on.
    :type device: str
    """
    def __init__(self, ner_models: Union[List[str], str], device: str = "cpu", unique_key: str = "word"):
        self.model_names = ner_models
        self.ner_models = SemanticModelFactory.create(ner_models, device)
        self.unique_key = unique_key
        self.tokenizer = SemanticTokenizer()
        self.low_score = -1e9
        # todo generation config here

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
        semantic_data_chunks = self.ner_models[0].predict(texts)
        (
            _,
            all_semantic_data_chunks 
        ) = self.ner_models[0].get_generated_semantic_data(
                    semantic_data_chunks,
                    input_length_chars,
                    include_all=include_all
                )
        merged_semantic_data_points = self.ner_models[0].merge_semantic_data(
            all_semantic_data_chunks
        )
        semantic_datas = self.ner_models[0].to_generic_semantic_data(
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
        beam_size: int
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

        batch_hyps = [hypotheses[i:i+beam_size] for i in range(0, len(hypotheses), beam_size)]
        # only keep the hypotheses with semantic data
        batch_hyps_w_sem_data = [
            [hyp for hyp in batch if hyp.semantic_data.has_semantic_data] for batch in batch_hyps
        ]
        # check that all batches have at least one semantic token, if not create empty token which
        # will carry all hypotheses with a score of 1
        if not all([len(batch) > 0 for batch in batch_hyps_w_sem_data]):
            for batch_idx, batch in enumerate(batch_hyps_w_sem_data):
                if len(batch) == 0:
                    # create custom semantic data with empty token
                    semantic_data = SemanticData.create_empty(
                        self.tokenizer.decode(torch.tensor(self.tokenizer.empty_token_id))
                    )
                    batch_wo_any_sem_data = batch_hyps[batch_idx]
                    for hyp in batch_wo_any_sem_data:
                        hyp.semantic_data = semantic_data
                    batch_hyps_w_sem_data[batch_idx] = batch_wo_any_sem_data
        # 1. need to remove duplicate hypotheses (same syntactic hyp sequence)
        # as they stand for the same path in the model and the probabilities of it
        # should not be counted twice (implementation is same as using a set, but this
        # version keeps the order)
        unique_batch_hyps_w_sem_data = [
            list(dict.fromkeys(batch)) for batch in batch_hyps_w_sem_data
        ]
        path_scores_w_sem_data = [
            torch.cat([hyp.path_score.unsqueeze(0) for hyp in batch], 0) for batch in unique_batch_hyps_w_sem_data
        ]
        for batch_idx, batch in enumerate(unique_batch_hyps_w_sem_data):
            normalized_path_scores = torch.log_softmax(path_scores_w_sem_data[batch_idx], dim=-1)
            for hyp_idx, synt_hyp in enumerate(batch):
                synt_hyp.normalized_path_score = normalized_path_scores[hyp_idx]
                synt_hyp.is_normalized_path_score_calculated = True
        
        return [sorted(self._create_semantic_hypotheses(batch_hyps), reverse=True) for batch_hyps in unique_batch_hyps_w_sem_data]

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

    def _create_semantic_hypothesis(self, hypotheses: List[SyntacticHypothesis]) -> SemanticToken:
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
            tuple(hypotheses),
            None
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
        # create a copy and leave original list untouched
        semantic_tokens = [[beam for beam in batch] for batch in semantic_tokens]
        # check, if all beams have semantic tokens
        all_beams_have_semantic_tokens = all(
            [len(beam) > 0 for batch in semantic_tokens for beam in batch]
        )
        if all_beams_have_semantic_tokens:
            return semantic_tokens, semantic_beam_scores
        def fetch_pkv_from_first_non_empty_sem_tok(
            semantic_tokens: List[List[List[SemanticToken]]],
            empty_token_id: int
        ) -> SyntacticHypothesisContinuationData:
            for batch in semantic_tokens:
                for beam in batch:
                    for sem_token in beam:
                        if sem_token.token_id != empty_token_id:
                            syntactic_hyp = sem_token.syntactic_hypotheses[0].syntactic_hypothesis
                            return syntactic_hyp
        
        non_empty_hyp = fetch_pkv_from_first_non_empty_sem_tok(semantic_tokens, self.tokenizer.empty_token_id)
        # use one pkv tensor to create dummy semantic token (needs right shape)
        pkv_dummy = non_empty_hyp.stack_past_key_values()

        beam_size = len(semantic_tokens[0])
        batch_size = len(semantic_tokens)
        # use padding token id; empty token is reserved for semantic token shell
        empty_semantic_token = SemanticToken.create_empty(
            self.tokenizer.decode(torch.tensor(self.tokenizer.empty_token_id)),
            self.tokenizer.pad_token_id,
            syntactic_empty_token_id,
            semantic_beam_scores.device,
            self.low_score,
            pkv_like=pkv_dummy
        )
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
            if sem_hyp is not None:
                batch_idx = sem_hyp_idx // semantic_beam_size
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
        self.vocab_size = len(self.str_to_tokens.keys())

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
            for string_tok_idx, string_token in enumerate(list_of_string_tokens):
                if string_token not in self.str_to_tokens.keys():
                    self._update_semantic_token_lookup(string_token)
                if string_tok_idx == 0:
                    # check if need padding
                    if len(list_of_string_tokens) < longest_sequence:
                        # need for padding
                        padding = [self.pad_token_id] * (longest_sequence - len(list_of_string_tokens))
                        tokenized_sequences[sequence_idx, :] = torch.tensor(
                            padding + [self.str_to_tokens[string_token]] 
                        )
                        attention_mask[sequence_idx, :longest_sequence-len(list_of_string_tokens)] = 0
                    else:
                        # no need for padding
                        tokenized_sequences[sequence_idx, string_tok_idx] = self.str_to_tokens[string_token]
                else:
                    tokenized_sequences[sequence_idx, string_tok_idx] = self.str_to_tokens[string_token]
        return {
            "input_ids": tokenized_sequences,
            "attention_mask": attention_mask
        }

    def encode(self, sequence: List[str]) -> torch.Tensor:
        return self([sequence])["input_ids"][0]

    def update_vocab_size(self) -> None:
        self.vocab_size = len(self.str_to_tokens.keys())

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
                    self.update_vocab_size()

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

    def decode(self, token: torch.Tensor) -> str:
        try:
            return self.tokens_to_str[token.item()]
        except KeyError as token:
            raise KeyError(f"Token {token} not known by the tokenizer.")

    def _decode_sequence(self, sequence: torch.Tensor) -> List[str]:
        return [
            self.decode(token) for token in sequence
        ]
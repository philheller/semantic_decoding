import torch
from typing import List, Tuple, Union, Dict, Any, Optional
from ner_model import NERModelFactory, NER_OutputType
from data_structures import SemanticData, SyntacticHypothesis, SemanticHypothesis

class SemanticGenerator:
    """ 
    The semantic generator is responsible for generating semantic tokens from text.
    It uses NER models to generate the semantic tokens and aggregates the scores of the entities.
    
    :param ner_models: List of NER model names.
    :type ner_models: List[str]
    :param aggregation_key: Key to aggregate the entities on.
    :type aggregation_key: str
    :param device: Device to run the model on.
    :type device: str
    """
    def __init__(self, ner_models: Union[List[str], str], device: str = "cpu", aggregation_key: str = "word"):
        self.model_names = ner_models
        self.ner_models = NERModelFactory.create(ner_models, device)
        self.aggregation_key = aggregation_key
        self.tokenizer = SemanticTokenizer()
        # todo generation config here

    def _get_generated_entities(self, entities: NER_OutputType, input_length_chars: torch.Tensor) -> Tuple[NER_OutputType, NER_OutputType]:
        """ 
        Get the generated entities from the NER model output.
        
        :param entities: List of entities predicted by the NER model.
        :type entities: List[Dict[str, Any]]
        :param input_length_chars: Length of the input text in characters.
            This is needed in order to only keep the entities generated after the input text.
        :type input_length_chars: torch.Tensor
        :return: Tuple containing only the first generated entity and all generated entities.
        :rtype: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        """
        new_entities = []
        first_new_entities = []
        for i, ents in enumerate(entities):
            entities_of_current_output = []
            for entity in ents:
                if entity["start"] > input_length_chars[i]:
                    entities_of_current_output.append(entity)
            
            # if the first of entities_of_current_output["entity"] does not start with a "B", remove it
            while (len(entities_of_current_output) > 0 and entities_of_current_output[0]["entity"][0] != "B"):
                entities_of_current_output = entities_of_current_output[1:]
            new_entities.append(entities_of_current_output)
        return new_entities, first_new_entities

    def generate(self, text: List[str]) -> NER_OutputType:
        # todo currently only using first ner, later use all
        semantic_output = self.ner_models[0].predict(text)
        return semantic_output

    def compute_semantic_hypotheses(
        self,
        hypotheses: List[SyntacticHypothesis],
        beam_size: int
    ) -> List[SemanticHypothesis]:
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
        # 1. need to remove duplicate hypotheses (same syntactic hyp sequence)
        # as they stand for the same path in the model and the probabilities of it
        # should not be counted twice
        unique_batch_hyps_w_sem_data = [
            list(set(batch)) for batch in batch_hyps_w_sem_data
        ]
        
        # 2. normalize scores over semantic tokens
        # path scores (these are the scores which are normalized and summed)
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
    ) -> List[SemanticHypothesis]:
        # a) groub by semantic source hyp and by entity
        aggr_key_dict = self._group_by_aggregation_key(hypotheses)
        # b) create semantic hypotheses (also calculates semantic scores)
        all_semantic_hypotheses = []
        for key, value in aggr_key_dict.items():
            sem_hyp = self._create_semantic_hypothesis(value)
            all_semantic_hypotheses.append(sem_hyp)
        return all_semantic_hypotheses

    def _create_semantic_hypothesis(self, hypotheses: List[SyntacticHypothesis]) -> SemanticHypothesis:
        """ 
        Merge the passed hypotheses into a single SemanticHypothesis. Calculates
        the semantic scores and creates the SemanticHypothesis instance.
        """
        # if all do not have semantic data
        if not all([hyp.semantic_data.has_semantic_data for hyp in hypotheses]):
            return SemanticHypothesis(
                hypotheses[0].aggregation_key,
                -float("inf"),
                tuple(hypotheses),
                None
            )
        # compute semantic scores
        semantic_score = self._compute_semantic_scores(hypotheses)
        # now create the SemanticHypothesis
        return SemanticHypothesis(
            hypotheses[0].aggregation_key,
            semantic_score,
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
        normalized_path_scores = torch.tensor([hyp.normalized_path_score.unsqueeze(0) for hyp in hypotheses])
        # accumulate them
        semantic_score = torch.logsumexp(normalized_path_scores, dim=0)
        return semantic_score

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

    def compute_semantic_scores_legacy(
        self,
        transition_scores: torch.Tensor,
        num_of_beams: int,
        semantic_output: NER_OutputType,
        syn_to_sem_hyp_mapping: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        # a) calculate sequence scores
        sequence_scores = transition_scores.sum(dim=-1)
        # b) normalize over entities
        batch_size = transition_scores.shape[0] // num_of_beams
        entity_hyp_probs = sequence_scores.view((batch_size, num_of_beams))
        entity_hyp_probs = torch.log_softmax(entity_hyp_probs, dim=-1)
        entity_hyp_probs = entity_hyp_probs.view((entity_hyp_probs.shape[0] * entity_hyp_probs.shape[1]))

        # c) aggregate over semantic source hyp and entities
        # anything with the same entity and the same previous semantic entity
        # is considered to belong to a single semantic token
        # and therefore need to be grouped together (and aggregated)
        semantic_output = self.merge_entities(semantic_output)
        sem_token_scores, no_semantic_token_scores = self._group_by_sem_source_hyp_and_by_entity_legacy(
            entity_hyp_probs,
            semantic_output,
            syn_to_sem_hyp_mapping.flatten(),
            batch_size,
            num_of_beams,
        )

        # add the probs of every key together (aggregation)
        # ? use the logsoftmax trick in order to aggregate probs of the same entity
        # for an aggregated entity probability
        for batch in sem_token_scores:
            for source_hyp_idx, value in batch.items():
                for entity, probs in value.items():
                    batch[source_hyp_idx][entity] = torch.logsumexp(probs, dim=0)
        
        
        return sem_token_scores, no_semantic_token_scores

    def _group_by_sem_source_hyp_and_by_entity_legacy(
        self,
        syntactic_hyp_prob: torch.Tensor,
        entities: NER_OutputType,
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
                semantic_tokens_from_syntactic_tokens[batch_idx][sem_source_hyp.item()][entity[0][self.aggregation_key]] = prob
            else:
                if entity[0][self.aggregation_key] not in semantic_tokens_from_syntactic_tokens[batch_idx][sem_source_hyp.item()].keys():
                    # sem beam exists, but entity[0] is new (add with value)
                    semantic_tokens_from_syntactic_tokens[batch_idx][sem_source_hyp.item()][entity[0][self.aggregation_key]] = prob
                else:
                    # sem beam and entity[0] already exist
                    semantic_tokens_from_syntactic_tokens[batch_idx][sem_source_hyp.item()][entity[0][self.aggregation_key]] = torch.cat(
                        (
                            semantic_tokens_from_syntactic_tokens[batch_idx][sem_source_hyp.item()][entity[0][self.aggregation_key]],
                            prob
                        )
                    )
        return semantic_tokens_from_syntactic_tokens, no_semantic_token        

    def _group_by_entity(
        self,
        entities: NER_OutputType,
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
            key = self.aggregation_key
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

    def get_semantic_data_from_first_entities(
        self,
        first_entities: NER_OutputType,
        all_entities: Optional[NER_OutputType] = None
    ) -> List[Union[SemanticData, None]]:
        semantic_data = [
            SemanticData(
                entity[0][self.aggregation_key],
                entity[0]["start"],
                entity[0]["end"],
                entity[0]["entity"],
                entity[0]["amount_of_entity_chunks"],
                all_entities[entity_idx] if all_entities is not None else None
            ) if len(entity) > 0 else None
             for entity_idx, entity in enumerate(first_entities)
        ]
        return semantic_data

    def merge_entities(self, entities: NER_OutputType) -> NER_OutputType:
        list_of_entity_sequences = []
        for hyp_entities in entities:
            list_of_whole_entities = []
            amount_entity_chunks = len(hyp_entities)
            if amount_entity_chunks == 0:
                list_of_entity_sequences.append(list_of_whole_entities)
                continue
            counter = 0
            counter_start = counter
            while counter <= amount_entity_chunks:
                if counter == amount_entity_chunks:
                    # all chunks from counter_start to counter are a whole entity
                    entity_chunks_belonging_together = hyp_entities[counter_start:counter]
                    merged_entity = self._merge_entity_chunks(entity_chunks_belonging_together)
                    list_of_whole_entities.append(merged_entity)
                elif hyp_entities[counter]["entity"][0] == "B":
                    # all the chunks before this B are a whole entity (not if counter == 0)
                    if counter != 0:
                        entity_chunks_belonging_together = hyp_entities[counter_start:counter]
                        merged_entity = self._merge_entity_chunks(entity_chunks_belonging_together) 
                        list_of_whole_entities.append(merged_entity)
                        counter_start = counter
                counter += 1
            list_of_entity_sequences.append(list_of_whole_entities)
        return list_of_entity_sequences

    def _merge_entity_chunks(self, entity_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        _type = self._extract_entity_type(entity_chunks[0]["entity"])
        start = entity_chunks[0]["start"]
        end = entity_chunks[-1]["end"]
        word = " ".join([entity["word"] for entity in entity_chunks])
        return {
            "entity": _type,
            "word": word,
            "start": start,
            "end": end,
            "amount_of_entity_chunks": len(entity_chunks)
        }

    def _extract_entity_type(self, entity: str) -> str:
        return entity.split("-")[-1]

    def _extract_aggregation_keys(self, sequences: NER_OutputType) -> List[List[str]]:
        return [
            [
                entity[self.aggregation_key] 
                    for entity in sequence
            ]
            for sequence in sequences
        ]

    def encode_semantic_sequences_from_entities(self, sequences: NER_OutputType) -> torch.Tensor:
        aggregated_sequences = self._extract_aggregation_keys(sequences)
        return self.tokenizer(aggregated_sequences)

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
            pad_token: str = "<pad>"
        ):
        self.str_to_tokens = {}
        self.str_to_tokens[pad_token] = 0
        self.str_to_tokens[bos_token] = 1
        self.str_to_tokens[eos_token] = 2
        if initial_tokens is not None:
            # amount of keys
            offset = len(self.str_to_tokens.keys())
            initial_tokens = {
                key: idx + offset for idx, key in enumerate(initial_tokens)
            }
            initial_tokens.update(self.str_to_tokens)
            self.str_to_tokens = initial_tokens
        self.tokens_to_str = {v: k for k, v in self.str_to_tokens.items()}
        self.bos_token = self.str_to_tokens[bos_token]
        self.eos_token = self.str_to_tokens[eos_token]
        self.pad_token = self.str_to_tokens[pad_token]

    def __len__(self) -> int:
        return len(self.str_to_tokens.keys())

    def __str__(self) -> str:
        return f"SemanticTokenizer with {len(self.str_to_tokens.keys())} tokens.\nBOS token: {self.bos_token}\nEOS token: {self.eos_token}\nPAD token: {self.pad_token}"

    def __call__(
        self,
        sequences: List[List[str]],
        ) -> Dict[str, torch.Tensor]:
        longest_sequence = max(
            len(sequence) for sequence in sequences
        )
        tokenized_sequences = torch.full((len(sequences), longest_sequence), self.pad_token, dtype=torch.long)
        attention_mask = torch.full((len(sequences), longest_sequence), 1, dtype=torch.long)
        for sequence_idx, list_of_string_tokens in enumerate(sequences):
            for string_tok_idx, string_token in enumerate(list_of_string_tokens):
                if string_token not in self.str_to_tokens.keys():
                    self._update_semantic_token_lookup(string_token)
                if string_tok_idx == 0:
                    # check if need padding
                    if len(list_of_string_tokens) < longest_sequence:
                        # need for padding
                        padding = [self.pad_token] * (longest_sequence - len(list_of_string_tokens))
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

    def _update_semantic_token_lookup(
            self,
            entity: str,
            skip_inverting: bool = False
        ) -> None:
            if entity in [self.bos_token, self.eos_token, self.pad_token]:
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
            decoded_sequence = self.decode(sequence)
            decoded_sequences.append(decoded_sequence)
        return decoded_sequences

    def decode(self, sequence: torch.Tensor) -> List[str]:
        try:
            return [self.tokens_to_str[token.item()] for token in sequence]
        except KeyError as token:
            raise KeyError(f"Token {token} not known by the tokenizer.")

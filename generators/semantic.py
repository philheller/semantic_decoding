import torch
from typing import List, Tuple, Union, Dict, Any, Optional
from ner_model import NERModelFactory, NER_OutputType

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
        
    def compute_semantic_scores(
        self,
        transition_scores: torch.Tensor,
        num_of_beams: int,
        amount_of_tokens_shortened: torch.Tensor,
        semantic_output: NER_OutputType,
        # semantic_source_hyp: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        # a) calculate sequence scores
        sequence_scores = transition_scores.sum(dim=-1)
        # b) normalize length (otherwise hyps with entities early have an advantage)
        normalized_sequence_scores = torch.div(sequence_scores, num_of_beams - amount_of_tokens_shortened)
        # c) normalize over entities
        # question to be answered: should empty entities be considered?
        batch_size = transition_scores.shape[0] // num_of_beams
        entity_hyp_probs = normalized_sequence_scores.view((batch_size, num_of_beams))
        entity_hyp_probs = torch.log_softmax(entity_hyp_probs, dim=-1)
        entity_hyp_probs = entity_hyp_probs.view((entity_hyp_probs.shape[0] * entity_hyp_probs.shape[1]))

        # d) aggregate over entities
        # anything with the same entity and the same previous entity (same source hyp idx)
        # is considered a direct sibling and needs to be grouped together
        semantic_output = self.merge_entities(semantic_output)
        # todo for now aggregating over the same entity, but that does not 
        # work as it would remove the acyclic property (if two different prior entities
        # both have the same entity, they should not be grouped together)
        # ? will have to group by beam_indices path (same prior semantic_beam_indices
        # guarantee that the graph is actually a tree)
        # merge by key
        entity_scores, wo_entity_scores = self._group_by_entity(
            semantic_output,
            entity_hyp_probs,
            batch_size,
            num_of_beams,
            self.aggregation_key
        )
        # add the probs of every key together (aggregation)
        # ? use the logsoftmax trick in order to aggregate probs of the same entity
        # for an aggregated entity probability
        for batch in range(batch_size):
            for entity in entity_scores[batch]:
                entity_scores[batch][entity]["aggregated_score"] = torch.logsumexp(entity_scores[batch][entity], dim=0)
        
        return entity_scores, wo_entity_scores


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

    def merge_entities(self, entities: NER_OutputType) -> NER_OutputType:
        list_of_entity_sequences = []
        for hyp_entities in entities:
            list_of_whole_entities = []
            amount_entity_chunks = len(hyp_entities)
            if amount_entity_chunks == 0:
                list_of_whole_entities.append({})
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

    def merge_entities_2(self, entities: NER_OutputType) -> NER_OutputType:
        """ 
        Expects the entities in the inner list to belong togehter.
        """
        merged_entities = []
        for hyp_entities in entities:
            amount_of_entity_chunks = len(hyp_entities)
            if amount_of_entity_chunks == 0:
                merged_entities.append({})
                continue
            start = hyp_entities[0]["start"]
            end = hyp_entities[-1]["end"]
            word = " ".join([entity["word"] for entity in hyp_entities])
            _type = self._extract_entity_type(
                hyp_entities[0]["entity"]
            )
            merged_entities.append(
                {
                    "entity": _type,
                    "word": word,
                    "start": start,
                    "end": end,
                    "amount_of_entity_chunks": amount_of_entity_chunks
                }
            )
        return merged_entities

    def _extract_entity_type(self, entity: str) -> str:
        return entity.split("-")[-1]

    def encode_semantic_tokens(self, sequences: List[List[str]]) -> torch.Tensor:
        return self.tokenizer(sequences)

        
class SemanticTokenizer:
    def __init__(
            self,
            initial_tokens: Optional[List[str]] = None,
            bos_token: str = "<bos>",
            eos_token: str = "<eos>",
            pad_token: str = "<pad>"
        ):
        self.str_to_tokens = {}
        self.str_to_tokens[bos_token] = 0
        self.str_to_tokens[eos_token] = 1
        self.str_to_tokens[pad_token] = 2
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
        
    def _update_semantic_token_lookup(
            self,
            semantic_tokens_lookup: Dict[str, int],
            entity: Dict[str, Any],
            skip_inverting: bool = False
        ) -> Dict[str, int]:
            if entity[self.aggregation_key] not in semantic_tokens_lookup.keys():
                semantic_tokens_lookup[entity[self.aggregation_key]] = len(semantic_tokens_lookup)
            if not skip_inverting:
                inverted_lookup = {v: k for k, v in semantic_tokens_lookup.items()}
                return semantic_tokens_lookup, inverted_lookup
            return semantic_tokens_lookup, {}
    
    def _update_multiple_semantic_token_lookup(
            self,
            semantic_tokens_lookup: List[Dict[str, int]],
            entities: NER_OutputType,
        ) -> Dict[str, int]:
            for entity in entities:
                semantic_tokens_lookup, _ = self.update_semantic_token_lookup(
                    semantic_tokens_lookup,
                    entity,
                    skip_inverting=True
                )
            inverted_lookup = {v: k for k, v in semantic_tokens_lookup.items()}
            return semantic_tokens_lookup, inverted_lookup

    def __call__(
        self,
        sequences: List[List[str]]
        ) -> torch.Tensor:
        longest_sequence = max(
            len(sequence) for sequence in sequences
        )
        tokenized_sequences = torch.tensor((len(sequences), longest_sequence), dtype=torch.long)
        for sequence_idx, list_of_string_tokens in enumerate(sequences):
            for string_tok_idx, string_token in enumerate(list_of_string_tokens):
                if string_token not in self.str_to_tokens.keys():
                    self._update_semantic_token_lookup(string_token)
                if string_tok_idx == 0:
                    # check if need padding
                    if len(list_of_string_tokens) < longest_sequence:
                        padding = [self.pad_token] * (longest_sequence - len(list_of_string_tokens))
                        tokenized_sequences[sequence_idx, :] = torch.tensor(
                            padding + [self.str_to_tokens[string_token]] 
                        )
                    else:
                        tokenized_sequences[sequence_idx, string_tok_idx] = self.str_to_tokens[string_token]
        return tokenized_sequences
                
                
                

            
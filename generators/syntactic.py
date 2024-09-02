from typing import List, Optional, Tuple, Any, Dict
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput, GenerationConfig
import torch
from data_structures import (
    ContinuationData,
    OriginalContinuationData,
    SemanticData,
    SyntacticHypothesisContinuationData,
    SyntacticHypothesisUnshortenedContinuationData,
    SyntacticHypothesis,
    SyntacticHypothesisMetaData
    )

ModelOutput = Dict[str, Any]
class SyntacticGenerator:
    """ 
    The SynthacticGenerator class is responsible for generating syntactic sequences.
    It uses a pre-trained model from the huggingface and is responsible for anything close to
    the syntactic hypotheses.

    :param model_name: Name of the model to load.
    :type model_name: str
    :param device: Device to load the model on.
    :type device: str
    :param access_token: Access token for private models.
    :type access_token: Optional[str]
    """
    def __init__(
            self,
            model_name: str,
            device: str,
            access_token: Optional[str] = None
        ):
        self.model = self._load_model(model_name, device, access_token)
        self.tokenizer = self._load_tokenizer(model_name, access_token)

    def generate(
        self,
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
    ) -> GenerateBeamDecoderOnlyOutput:
        """
        Generates sequences using the syntactic model with specified parameters.

        :param inputs: A dictionary of inputs for the model, typically including 'input_ids' and 'attention_mask'.
        :param return_dict_in_generate: Whether to return a dictionary with additional generation outputs.
        :param output_scores: Whether to output scores during generation.
        :param max_new_tokens: Maximum number of new tokens to generate.
        :param num_beams: Number of beams for beam search.
        :param num_return_sequences: Number of sequences to return.
        :param resume_generation: Whether to resume generation using past key values.
        :param past_key_values: Past key values to resume generation (if any).
        :param last_scores: Previous scores to influence generation (if any). Mostly for testing.
        :param last_beam_scores: Previous beam scores to influence generation (if any).
        :param original_prompt_length: Length of the original prompt.
        :param renormalize_logits: Whether to renormalize logits during generation.
        :param reproducibility: Ensures fair comparison by setting seeds at every generation loop step.
        :param length_penalty: Exponential penalty to the length. Default is None.
        :param do_sample: Whether to enable sampling during generation. Default is None.
        :param temperature: Temperature for sampling. Default is None.
        :param top_k: Top K sampling. Default is None.
        :param top_p: Top P sampling. Default is None.
        :param kwargs: Additional arguments to pass to the generate function.
        :return: GenerateBeamDecoderOnlyOutput containing the generated sequences and other information.
        """
        return self.model.generate(
            inputs=inputs,
            generation_config=generation_config,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            output_logits=output_logits,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences if num_return_sequences is not None else num_beams,
            resume_generation=resume_generation,
            past_key_values=past_key_values,
            last_scores=last_scores,
            last_beam_scores=last_beam_scores,
            original_prompt_length=original_prompt_length,
            renormalize_logits=renormalize_logits,
            reproducibility=reproducibility,
            length_penalty=length_penalty,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )

    def batch_decode(self, batch: torch.Tensor) -> List[str]:
        """
        Decodes a batch of tokenized sequences into a list of strings.

        :param batch: Batch of tokenized sequences.
                        Expexted size: (batch_size, sequence_length) [dim=2]
        :type batch: torch.Tensor
        :return: List of decoded strings.
        """
        return self.tokenizer.batch_decode(batch, skip_special_tokens=True)

    def _load_model(
            self,
            model_name:str,
            device: str,
            access_token: Optional[str]
        ) -> Any:
        """
        Load a pre-trained model from Hugging Face model hub.
        
        :param model_name: Name of the model to load.
        :type model_name: str
        :param device: Device to load the model on.
        :type device: str
        :param access_token: Access token for private models.
        :type access_token: Optional[str]
        :return: Loaded model.
        :rtype: Any
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=access_token, 
            device_map="auto"
        ).to(device)
        model.eval() # Though this is default, but just to be sure
        print(f"Using precision: {next(model.parameters()).dtype}")
        print(f"Eval mode: {not model.training}")
        return model

    def _load_tokenizer(self, model_name: str, access_token: Optional[str]) -> Any:
        """
        Load a pre-trained tokenizer from Hugging Face model hub.
        
        :param model_name: Name of the model to load.
        :type model_name: str
        :param access_token: Access token for private models.
        :type access_token: Optional[str]
        :return: Loaded tokenizer.
        :rtype: Any
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if tokenizer.pad_token is None:
            print(f"Setting pad token to eos token: {tokenizer.eos_token}")
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_output_length(
        self,
        output_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the length of the output sequence.

        :param output_ids: Tokenized output sequence.
        :type output_ids: torch.Tensor
        :return: Length of the output sequence.
        :rtype: torch.Tensor
        """
        return torch.sum(output_ids != self.tokenizer.eos_token_id, axis=1) # type: ignore

    def get_input_length(
        self,
        input_ids: torch.Tensor,
        beam_indices: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Get the length of the input sequence for tokens and for the raw string.
        This is straight forward for decoding strategies without multiple hypothesis.
        For beam search, the input length is determined by the source hypothesis
        which is traced back recursively.

        :param input_ids: Tokenized input sequence.
        :type input_ids: torch.Tensor
        :param beam_indices: Indices of the beams.
        :type beam_indices: torch.Tensor
        :return: Tuple of the token length and raw string length.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        hyps_in_plain_string = None
        input_length = None
        input_length_chars = None
        
        if beam_indices is None:
            # no multiple hypotheses, shape is right
            hyps_in_plain_string = self.batch_decode(input_ids)
            input_length = torch.sum(input_ids != self.tokenizer.eos_token_id, axis=1) # type: ignore
            input_length_chars = torch.tensor([len(hyp) for hyp in hyps_in_plain_string])
        else:
            source_hyp_indices = self.compute_source_hypothesis_indices(beam_indices)
            # check if inputs are expanded (first pass they are not)
            beam_batch_size = beam_indices.shape[0]
            if input_ids.shape[0] == beam_batch_size:
                hyps_in_plain_string = self.batch_decode(input_ids)
                input_length = torch.sum(input_ids != self.tokenizer.eos_token_id, axis=1) # type: ignore
                input_length_chars = torch.tensor([len(hyp) for hyp in hyps_in_plain_string])
            else:
                batch_size = input_ids.shape[0]
                beam_size = beam_batch_size // batch_size
                # expand to be of size (batch_size * beam_size, )
                expanded_inputs = input_ids.repeat_interleave(beam_size, dim=0)
                hyps_in_plain_string = self.batch_decode(expanded_inputs)
                input_length = torch.sum(expanded_inputs != self.tokenizer.eos_token_id, axis=1) # type: ignore
                input_length_chars = torch.tensor([len(hyp) for hyp in hyps_in_plain_string])
            # due to BS, we need to find the source hyp for each beam to be able to 
            # get the correct input length
            beam_hyp_input_len_chars = input_length_chars.clone()
            beam_hyp_input_length = input_length.clone()
            # Index of last beam_idx that is not -1
            first_non_negative_beam_idx = (beam_indices >= 0).sum(dim=1) - 1

            for beam_idx in range(len(beam_indices)):
                # recursively walk back beam_indices to find the source hypothesis
                source_hyp_idx = self._get_source_hypothesis_idx(beam_indices, beam_idx, step=first_non_negative_beam_idx[beam_idx].item()) # type: ignore
                # todo when running through, check if source_hyp_idx is always the same as source_hyp_indices[beam_idx]
                assert source_hyp_idx == source_hyp_indices[beam_idx], "Mismatch between source hypothesis indices"
                # reassign the input lengths to the ones from the source hypothesis
                beam_hyp_input_length[beam_idx] = input_length[source_hyp_idx]
                beam_hyp_input_len_chars[beam_idx] = input_length_chars[source_hyp_idx]

            input_length_chars = beam_hyp_input_len_chars
            input_length = beam_hyp_input_length
            del beam_hyp_input_len_chars, beam_hyp_input_length
        return input_length, input_length_chars

    def compute_source_hypothesis_indices(
        self,
        beam_indices: torch.Tensor
    ):
        """
        Compute the source hypothesis indices for the given beam indices.

        :param beam_indices: Indices of the beams.
        :type beam_indices: torch.Tensor
        :return: Source hypothesis indices.
        :rtype: torch.Tensor
        """
        source_hyp_indices = torch.full((beam_indices.shape[0],), -1).to(beam_indices.device)
        # Index of last beam_idx that is not -1
        first_non_negative_beam_idx = (beam_indices >= 0).sum(dim=1) - 1

        for beam_idx in range(len(beam_indices)):
            # recursively walk back beam_indices to find the source hypothesis
            source_hyp_idx = self._get_source_hypothesis_idx(beam_indices, beam_idx, step=first_non_negative_beam_idx[beam_idx].item())
            source_hyp_indices[beam_idx] = source_hyp_idx
        return source_hyp_indices

    def _get_source_hypothesis_idx(self, beam_indices, beam_idx, step=-1) -> int:
        """
        Get the source hypothesis index for the given beam index

        :param beam_indices: Indices of the beams.
        :type beam_indices: torch.Tensor
        :param beam_idx: Index of the currently searched beam.
        :type beam_idx: int
        :param step: Step of the current search (walk from back to front).
        :type step: int
        :return: Source hypothesis index.
        :rtype: int
        """
        if beam_indices[beam_idx, step] == -1:
            return self._get_source_hypothesis_idx(beam_indices, beam_idx, step -1)
        else: 
            prior_index = beam_indices[beam_idx, step]
            if step == 0 or step == -(beam_indices.shape[1] + 1):
                return prior_index
            return self._get_source_hypothesis_idx(beam_indices, prior_index, step -1)

    def gather_composite_aggregation_key(
        self,
        semantic_source_hypothesis: int,
        semantic_data: SemanticData
    ) -> str:
        return str(semantic_source_hypothesis) + "-" + semantic_data.unique_key

    def shorten_hyp_to_first_semantic_data_point(
        self,
        first_semantic_data_point: List[SemanticData],
        hypotheses: List[SyntacticHypothesisUnshortenedContinuationData],
        semantic_source_hypothesis_indices: Optional[torch.Tensor] = None,
        syntactic_source_hypothesis_indices: Optional[torch.Tensor] = None
        ) -> List[SyntacticHypothesis]:
        device = hypotheses[0].sequences.device
        all_hyps = []
        for hyp_idx, (hyp, sem_data) in enumerate(zip(hypotheses, first_semantic_data_point)):
            # if no semantic data present, use hypothesis as is
            if sem_data is None:
                shortened_syn_hyp = SyntacticHypothesisContinuationData(
                    hyp.sequences,
                    hyp.transition_scores,
                    hyp.generated_transition_scores,
                    hyp.last_beam_scores,
                    hyp.past_key_values,
                    hyp.attention_mask,
                    hyp
                )
                path_score = self.compute_path_score(shortened_syn_hyp)
                no_semantic_data = SemanticData(
                    str(semantic_source_hypothesis_indices[hyp_idx].item()) if semantic_source_hypothesis_indices is not None else "",
                    -1,
                    -1,
                    "",
                    -1,
                    None,
                    has_semantic_data=False
                )
                syn_hyp = SyntacticHypothesis(
                    str(semantic_source_hypothesis_indices[hyp_idx].item()) if semantic_source_hypothesis_indices is not None else "",
                    semantic_source_hypothesis_indices[hyp_idx] if semantic_source_hypothesis_indices is not None else -1,
                    syntactic_source_hypothesis_indices[hyp_idx] if syntactic_source_hypothesis_indices is not None else -1,
                    hyp_idx,
                    path_score,
                    path_score,
                    no_semantic_data,
                    shortened_syn_hyp,
                    SyntacticHypothesisMetaData(
                        0
                    ),
                    is_aggregation_key_complete=True
                )
                all_hyps.append(syn_hyp)
                continue
            # two approaches
            # approach 1): 
            # 1. shorten raw string to fit the last entity
            # 2. encode with tokenizer
            # 3. find the last matching tokens between original tokens and recomputed tokens
            # 4. check if the recomputed tokens match the original tokens
            #   4a. if they do, proceed with 5.
            #   4b. if they do not, proceed with approach 2)
            # 5. cut the tokens at the last matching token
            
            # last char of the first semantic data point
            last_semantic_data_point_char = sem_data.end
            # shortened output
            decoded_output = self.batch_decode(hyp.sequences.unsqueeze(0))[0]
            decoded_output_shortened = decoded_output[:last_semantic_data_point_char]
            # reencode the shortened output
            recomputed_tokens = self.tokenizer(
                decoded_output_shortened,
                return_tensors="pt",
                padding=True
            ).to(device)

            # check if the sequence_ids are the same for the hyp.sequences and the recomputed tokens
            # first, remove padding from the hyp.sequences
            hyp_wo_padding = hyp.sequences[
                (hyp.sequences != self.tokenizer.pad_token_id).nonzero().min():
            ]

            if torch.equal(
                recomputed_tokens.input_ids[0],
                hyp_wo_padding[:recomputed_tokens.input_ids.shape[-1]]
            ):
                # if the recomputed sequence is the same as the 
                # original sequence (wo padding and shortened to first semantic data point)
                # then we can proceed with the shortened sequence
                amount_tokens_shortened_after_data_point = hyp_wo_padding.shape[-1] - recomputed_tokens.input_ids.shape[-1]
                shortened_hyp = self._shorten_hyp_right_by_amount_of_tokens(
                    hyp,
                    amount_tokens_shortened_after_data_point
                )
                path_score = self.compute_path_score(shortened_hyp)
                is_composite_aggregation_complete = False
                composite_aggregation_key = sem_data.unique_key
                if semantic_source_hypothesis_indices is not None:
                    composite_aggregation_key = self.gather_composite_aggregation_key(
                        semantic_source_hypothesis_indices[hyp_idx].item(),
                        sem_data
                    )
                    is_composite_aggregation_complete = True
                syn_hyp = SyntacticHypothesis(
                    composite_aggregation_key,
                    semantic_source_hypothesis_indices[hyp_idx] if semantic_source_hypothesis_indices is not None else -1,
                    syntactic_source_hypothesis_indices[hyp_idx] if syntactic_source_hypothesis_indices is not None else -1,
                    hyp_idx,
                    path_score,
                    path_score,
                    sem_data,
                    shortened_hyp,
                    SyntacticHypothesisMetaData(
                        amount_tokens_shortened_after_data_point
                    ),
                    is_aggregation_key_complete=is_composite_aggregation_complete
                )
                all_hyps.append(syn_hyp)
            else:
                # if the recomputed sequence is not the same as the original sequence
                # we need to find the last matching token. For that, use an iterative 
                # approach to find the last matching token.
                # 
                # approach 2):
                # 1. remove tokens from the end of the sequence
                # 2. reencode the sequence
                # 3. check if the recomputed tokens match the original tokens
                #   3a. if they do, proceed with 4.
                #   3b. if they do not, repeat 1.
                # 4. cut the tokens at the last matching token
                match = False
                piecewise_shortened_output = hyp_wo_padding.clone()
                original_size = len(piecewise_shortened_output)
                while (not match and len(piecewise_shortened_output) > 0):
                    # check if decoded is the same length as the end of the entity (or in extreme case, check if wo shortening already same length)
                    piecewise_shortened_output_decoded = self.tokenizer.decode(piecewise_shortened_output, skip_special_tokens=True)
                    if (
                        len(piecewise_shortened_output_decoded) == last_semantic_data_point_char
                        ) or (
                        len(piecewise_shortened_output_decoded) < last_semantic_data_point_char
                    ):
                        # difference to original length
                        amount_tokens_shortened_after_data_point = original_size - len(piecewise_shortened_output)
                        if len(piecewise_shortened_output_decoded) < last_semantic_data_point_char:
                            # if the piecewise shortened output is already shorter than the last semantic data point
                            # we need to include the previous token as well. This will include a bit more than only 
                            # the semantic token, but will make sure the semantic token is included entirely.
                            amount_tokens_shortened_after_data_point = last_semantic_data_point_char - len(piecewise_shortened_output_decoded) + 1
                        shortened_hyp = self._shorten_hyp_right_by_amount_of_tokens(
                            hyp,
                            amount_tokens_shortened_after_data_point
                        )
                        path_score = self.compute_path_score(shortened_hyp)
                        
                        is_composite_aggregation_complete = False
                        composite_aggregation_key = sem_data.unique_key
                        if semantic_source_hypothesis_indices is not None:
                            composite_aggregation_key = self.gather_composite_aggregation_key(
                                semantic_source_hypothesis_indices[hyp_idx].item(),
                                sem_data
                            )
                            is_composite_aggregation_complete = True
                        syn_hyp = SyntacticHypothesis(
                            composite_aggregation_key,
                            semantic_source_hypothesis_indices[hyp_idx] if semantic_source_hypothesis_indices is not None else -1,
                            syntactic_source_hypothesis_indices[hyp_idx] if syntactic_source_hypothesis_indices is not None else -1,
                            hyp_idx,
                            path_score,
                            path_score,
                            sem_data,
                            shortened_hyp,
                            SyntacticHypothesisMetaData(
                                amount_tokens_shortened_after_data_point
                            ),
                            is_aggregation_key_complete=is_composite_aggregation_complete
                        )
                        all_hyps.append(syn_hyp)
                        match = True
                        break
                    else:
                        piecewise_shortened_output = piecewise_shortened_output[:-1]
                if not match:
                    # if no match can be found at all, sth is wrong
                    raise ValueError("Something went wrong. Unable to find match between syntactic tokens and raw string.\
                        This should not happen. Please check the code.")
        return all_hyps

    def shorten_hyps_to_first_entity(
        self,
        first_new_entities: List[List[Dict[str, Any]]],
        sequences: torch.Tensor,
        decoded_sequences: List[str],
        attention_mask: torch.Tensor,
        transition_scores: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        Shorten the sequences to the first entity found in the output.

        :param first_new_entities: List of first new entities found in the output.
        :type first_new_entities: List[List[Dict[str, Any]]]
        :param sequences: Tokenized sequences.
        :type sequences: torch.Tensor
        :param decoded_sequences: Decoded sequences (raw strings).
        :type decoded_sequences: List[str]
        :param attention_mask: Attention mask.
        :type attention_mask: torch.Tensor
        :return: Tuple of shortened sequences, attention masks, transition scores,
                and amount of tokens removed.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        
        #### 6. shorten til right after newest entity ####
        # approach: 
        # 1. shorten raw string to fit the last entity
        # 2. encode with tokenizer
        # 3. find the last matching tokens between original tokens and recomputed tokens
        # 4. check if the recomputed tokens match the original tokens
        #   4a. if they do, proceed with 5.
        #   4b. if they do not, shave off token at a time and see if it matches the length of the trimmed string
        # 5. cut the tokens at the last matching token

        # using the following variables for that
        # a) iter_output.sequences
        # b) input_len_chars
        # c) first_new_entities
        device = sequences.device
        altered_input_ids = torch.empty_like(sequences).to(device)
        altered_attention_mask = torch.zeros_like(attention_mask).to(device)
        altered_transition_scores = transition_scores.clone()
        target_size = altered_input_ids.shape[-1]
        # just for stats
        tokens_trimmed_after_entity = torch.zeros(sequences.shape[0]).to(device)

        for beam_hyp_idx, entity_in_hypothis in enumerate(first_new_entities):
            if len(entity_in_hypothis) == 0:
                # if no entity found, simply use the tokens as is
                altered_input_ids[beam_hyp_idx] = sequences[beam_hyp_idx].clone()
                altered_attention_mask[beam_hyp_idx] = attention_mask[beam_hyp_idx].clone()
                continue
            last_char = entity_in_hypothis[-1]["end"]
            
            shortened_output = decoded_sequences[beam_hyp_idx][:last_char]
            recomputed_tokens = self.tokenizer(shortened_output, return_tensors="pt", padding=True).to(device)
            # last_sequence_id_from_recomputed = recomputed_tokens.input_ids[0][-1]

            # sequence_id of output without padding
            trimmed_sequence = sequences[
                    beam_hyp_idx,
                    (
                        sequences[beam_hyp_idx] != self.tokenizer.pad_token_id
                    ).nonzero().min():
                ].clone()

            if torch.equal(recomputed_tokens.input_ids[0], trimmed_sequence[:len(recomputed_tokens.input_ids[0])]):
                current_size = recomputed_tokens.input_ids.shape[-1]
                amount_tokens_shortened_after_entity = trimmed_sequence.shape[0] - current_size
                altered_input_ids[beam_hyp_idx] = torch.concat((
                        torch.tensor(
                                (target_size-current_size) * [self.tokenizer.pad_token_id]
                            ).to(device),
                        recomputed_tokens.input_ids[0]
                    ), dim=0
                )
                altered_attention_mask[beam_hyp_idx, -current_size:] = 1
                if amount_tokens_shortened_after_entity > 0:
                    # if no tokens were removed ([-0:] would set all to 0)
                    altered_transition_scores[beam_hyp_idx,-amount_tokens_shortened_after_entity:] = 0
                tokens_trimmed_after_entity[beam_hyp_idx] = amount_tokens_shortened_after_entity
            else:
                # the first optimistic approach does not work as there is a mismatch
                # between decoding and reencoding
                
                # to find the last matching token, remove syntactic tokens until the string lenght matches
                match = False
                piecewise_shortened_output = sequences[
                    beam_hyp_idx,
                    (
                        sequences[beam_hyp_idx] != self.tokenizer.pad_token_id
                    ).nonzero().min():
                ].clone()
                original_size = len(piecewise_shortened_output)
                while (not match and len(piecewise_shortened_output) > 0):
                    # check if decoded is the same length as the end of the entity (first one wo shortening if entity ends with string)
                    decoded_piecewise = self.tokenizer.decode(piecewise_shortened_output, skip_special_tokens=True)
                    if len(decoded_piecewise) == last_char:
                        # add padding to beginning of sequence and fix attention mask
                        current_size = len(piecewise_shortened_output)
                        amount_tokens_shortened_after_entity = original_size - current_size
                        altered_input_ids[beam_hyp_idx] = torch.concat((
                                torch.tensor(
                                        (target_size-current_size) * [self.tokenizer.pad_token_id]
                                    ).to(device),
                                    piecewise_shortened_output
                            ), dim=0
                        )
                        altered_attention_mask[beam_hyp_idx, -current_size:] = 1
                        if amount_tokens_shortened_after_entity > 0:
                            # if no tokens were removed ([-0:] would set all to 0)
                            altered_transition_scores[beam_hyp_idx,-amount_tokens_shortened_after_entity:] = 0
                        tokens_trimmed_after_entity[beam_hyp_idx] = amount_tokens_shortened_after_entity
                        match = True
                        break
                    # todo instead of special case (see below) test if the token length
                    # is smaller than last char and then take these tokens plus one more
                    else:
                        piecewise_shortened_output = piecewise_shortened_output[:-1]
                if match:
                    continue
                # todo create special case (non code breaking); add a warning note showing that the instance could not be unified
                # if no match can be found at all, sth is wrong
                raise ValueError("Unable to find match between syntactic tokens and raw string")
        return altered_input_ids, altered_attention_mask, altered_transition_scores, tokens_trimmed_after_entity

    def _shorten_hyp_right_by_amount_of_tokens(
        self,
        hypothesis: SyntacticHypothesisUnshortenedContinuationData,
        shorten_by_amount_of_tokens: int
    ) -> SyntacticHypothesisContinuationData:
        """
        Shorten hypothesis by an amount of tokens from the right.

        :param hypothesis: Hypothesis to shorten.
        :type hypothesis: SyntacticHypothesisUnshortenedContinuationData
        :param shorten_by_amount_of_tokens: Amount of tokens to shorten by (from right).
        :type shorten_by_amount_of_tokens: int
        :return: Shortened hypothesis.
        :rtype: SyntacticHypothesisContinuationData
        """
        if shorten_by_amount_of_tokens == 0:
            # no need to shorten
            pkv = self._stack_past_key_values(hypothesis.past_key_values)
            pkv = pkv.clone()
            pkv = self._unbind_past_key_values(pkv)
            return SyntacticHypothesisContinuationData(
                hypothesis.sequences.clone(),
                hypothesis.transition_scores.clone(),
                hypothesis.generated_transition_scores.clone(),
                hypothesis.last_beam_scores.clone(),
                pkv,
                hypothesis.attention_mask.clone(),
                hypothesis
            )
        shortened_sequences = hypothesis.sequences[:-shorten_by_amount_of_tokens].clone()
        shortened_transition_scores = hypothesis.transition_scores[:-shorten_by_amount_of_tokens].clone()
        shortened_generated_transition_scores = hypothesis.generated_transition_scores[:-shorten_by_amount_of_tokens].clone()
        # last beam scores need to be recalculated from transition_scores
        shortened_last_beam_scores = shortened_transition_scores.sum()
        # shorten past key values
        past_key_values = self._stack_past_key_values(hypothesis.past_key_values)
        past_key_values = past_key_values[:, :, :, :, :-shorten_by_amount_of_tokens, :]
        past_key_values = self._unbind_past_key_values(past_key_values)
        shortened_attention_mask = hypothesis.attention_mask[:-shorten_by_amount_of_tokens].clone()
        return SyntacticHypothesisContinuationData(
            shortened_sequences,
            shortened_transition_scores,
            shortened_generated_transition_scores,
            shortened_last_beam_scores,
            past_key_values,
            shortened_attention_mask,
            hypothesis
        )

    def _shorten_hyp_right_to_token_idx(
        self,
        hypothesis: ContinuationData,
        token_idx: int
    ):
        """
        Shorten a hypothesis to a specific token index.

        :param hypothesis: Hypothesis to shorten.
        :type hypothesis: ContinuationData
        :param token_idx: Token index to shorten to.
        :type token_idx: int
        :return: Shortened hypothesis.
        :rtype: ContinuationData
        """
        shorten_by = hypothesis.sequences.shape[-1] - token_idx
        # past key value shape: (num_layers, key_value = 2, batch_hyp_idx, num_heads, sequence_length, head_dim)
        past_key_values = self._stack_past_key_values(hypothesis.past_key_values)
        past_key_values = past_key_values[:, :, :, :, :-shorten_by, :]
        past_key_values = self._unbind_past_key_values(past_key_values)
        # last beam score needs to be recalculated (from transition scores)
        last_beam_scores = hypothesis.transition_scores[:-shorten_by].sum()
        return ContinuationData(
            sequences=hypothesis.sequences[:token_idx].clone(),
            transition_scores=hypothesis.transition_scores[:-shorten_by].clone(),
            last_beam_scores=last_beam_scores,
            past_key_values=past_key_values,
            attention_mask=hypothesis.attention_mask[:-shorten_by],
            original_data=hypothesis.original_data
        )

    def update_hypotheses_indeces(
        self,
        hypotheses: List[SyntacticHypothesis],
        indeces: Optional[torch.Tensor] = None
    ) -> List[SyntacticHypothesis]:
        """ 
        Update the indices of the hypotheses. If no indeces are provided, the indices are 
        inferred from the order of the hypotheses. Else, the indeces are used to update the
        hypotheses.
        
        :param hypotheses: List of hypotheses.
        :type hypotheses: List[SyntacticHypothesis]
        :param indeces: Indices to update the hypotheses with.
        :type indeces: Optional[torch.Tensor]
        :return: Updated hypotheses.
        :rtype: List[SyntacticHypothesis]
        """
        if indeces is not None:
            for (hyp, index) in zip(hypotheses, indeces):
                self.update_hypothesis_index(hyp, index)
        else:
            for index, hyp in enumerate(hypotheses):
                self.update_hypothesis_index(hyp, index)
        return hypotheses

    def update_hypothesis_index(
        self,
        hypothesis: SyntacticHypothesis,
        index: int
    ) -> SyntacticHypothesis:
       hypothesis.hypothesis_idx = index
       return hypothesis

    def update_semantic_source_hypothis_indices(
        self,
        hypotheses: List[SyntacticHypothesis],
        source_hypothesis_indices: torch.Tensor
    ) -> List[SyntacticHypothesis]:
        for batch_beam_idx, hyp in enumerate(hypotheses):
            hyp.semantic_source_hypothesis_idx = source_hypothesis_indices[batch_beam_idx]
        return hypotheses

    def update_syntactic_source_hypothesis_indices(
        self,
        hypotheses: List[SyntacticHypothesis],
        source_hypothesis_indices: torch.Tensor
    ) -> List[SyntacticHypothesis]:
        for batch_beam_idx, hyp in enumerate(hypotheses):
            hyp.syntactic_source_hypothesis_idx = source_hypothesis_indices[batch_beam_idx]
        return hypotheses

    def _expand_hyp_to_batch_length(
        self,
        hypothesis: SyntacticHypothesisContinuationData,
        target_length: int,
        pad_token_id: int
    ) -> SyntacticHypothesisContinuationData:
        # get sequences length
        current_length = hypothesis.sequences.shape[-1]
        missing_values = target_length - current_length

        sequence_filler = torch.full((missing_values,), pad_token_id).to(hypothesis.sequences.device)
        sequences = torch.cat((sequence_filler, hypothesis.sequences), dim=-1)

        # first approach: repeat the first past_key_values
        past_key_values = self._stack_past_key_values(hypothesis.past_key_values)

        # select 1st tensor in 5th dimension and repeat it
        to_be_repeated = past_key_values[:, :, :, :, 0, :]
        repeated_tensor = to_be_repeated.unsqueeze(4).repeat(1, 1, 1, 1, missing_values, 1)

        new_pkv = torch.cat((repeated_tensor, past_key_values), dim=-2)
        new_pkv = self._unbind_past_key_values(new_pkv)

        attention_mask = torch.cat(
            (
                torch.zeros((missing_values,)).to(hypothesis.attention_mask.device),
                hypothesis.attention_mask
            ),
            dim=-1
        )

        return SyntacticHypothesisContinuationData(
            sequences=sequences,
            transition_scores=hypothesis.transition_scores,
            generated_transition_scores=hypothesis.generated_transition_scores,
            last_beam_scores=hypothesis.last_beam_scores,
            past_key_values=new_pkv,
            attention_mask=attention_mask,
            unshortened_data=hypothesis.unshortened_data
        )

    def _expand_hyp_to_batch_length_legacy(
        self,
        hypothesis: ContinuationData,
        total_length: int,
        pad_token_id: int
    ) -> ContinuationData:
        # get sequences length
        current_length = hypothesis.sequences.shape[-1]
        missing_values = total_length - current_length

        sequence_filler = torch.full((missing_values,), pad_token_id).to(hypothesis.sequences.device)
        sequences = torch.cat((sequence_filler, hypothesis.sequences), dim=-1)

        # first approach: repeat the first past_key_values
        past_key_values = self._stack_past_key_values(hypothesis.past_key_values)

        # select 1st tensor in 5th dimension and repeat it
        to_be_repeated = past_key_values[:, :, :, :, 0, :]
        repeated_tensor = to_be_repeated.unsqueeze(4).repeat(1, 1, 1, 1, missing_values, 1)

        new_pkv = torch.cat((repeated_tensor, past_key_values), dim=-2)
        new_pkv = self._unbind_past_key_values(new_pkv)

        attention_mask = torch.cat(
            (
                torch.zeros((missing_values,)).to(hypothesis.attention_mask.device),
                hypothesis.attention_mask
            ),
            dim=-1
        )

        return ContinuationData(
            sequences=sequences,
            transition_scores=hypothesis.transition_scores,
            last_beam_scores=hypothesis.last_beam_scores,
            past_key_values=new_pkv,
            attention_mask=attention_mask,
            original_data=hypothesis.original_data
        )

    def compute_path_score(
        self,
        hypothesis: SyntacticHypothesisContinuationData,
    ):
        """
        Compute the path score for a hypothesis.

        :param hypothesis: Hypothesis.
        :type hypothesis: SyntacticHypothesisContinuationData
        :return: Path score.
        :rtype: torch.Tensor
        """
        return hypothesis.generated_transition_scores.sum(dim=-1)

    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: torch.Tensor,
        beam_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.model.compute_transition_scores(
            sequences,
            scores,
            beam_indices,
            normalize_logits=False
        )

    def get_duplicates(
        self,
        sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[Tuple[torch.Tensor], int]]:
        """
        Get mask of duplicate sequences (first one is not considered a duplicate)
        and the count of duplicates with sequence as tuple keys.

        :param sequences: Tokenized sequences.
        :type sequences: torch.Tensor
        :return: Tuple of mask of duplicates and count of duplicates.
        :rtype: Tuple[torch.Tensor, Dict[Tuple, List[torch.Tensor]]]
        """
        device = sequences.device
        # check if a sequence is present multiple times
        sequences_as_tuple = [tuple(seq.tolist()) if seq is not None else None for seq in sequences]
        # needs to be of size (batch_size, num_hyps_size)
        mask_of_duplicates = torch.zeros(sequences.shape[0], dtype=torch.int).to(device)

        occurrences = defaultdict(int)
        for i, t in enumerate(sequences_as_tuple):
            occurrences[t] += 1
            mask_of_duplicates[i] = 1 if occurrences[t] > 1 else 0
        return mask_of_duplicates, occurrences

    def pack_syntactic_hypotheses(
        self,
        sequences: torch.Tensor,
        transition_scores: torch.Tensor,
        last_beam_scores: torch.Tensor,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        attention_mask: torch.Tensor,
        source_hyps: Optional[List[SyntacticHypothesisContinuationData]] = None,
        source_beam_indices: Optional[torch.Tensor] = None,
    ) -> List[SyntacticHypothesisUnshortenedContinuationData]:
        """ 
        This function helps to pack the hypotheses into a list of SyntacticHypothesisUnshortenedContinuationData.
        
        :param sequences: Tokenized sequences.
        :type sequences: torch.Tensor
        :param transition_scores: Transition scores. Since the last beam scores cannot be recomputed without the 
            scores (which are generally not kept), the transition_scores can be concatednated from the source hyp.
            See the source_hyps and source_beam_indices parameters.
        :type transition_scores: torch.Tensor
        :param last_beam_scores: Last beam scores.
        :type last_beam_scores: torch.Tensor
        :param past_key_values: Past key values.
        :type past_key_values: Tuple[Tuple[torch.Tensor]]
        :param attention_mask: Attention mask.
        :type attention_mask: torch.Tensor
        :param source_hyps: Source hypotheses. These contain the source hyps for the new hypotheses.
        :type source_hyps: Optional[List[SyntacticHypothesisContinuationData]]
        :param source_beam_indices: Source beam indices. These can be calculated with the 
            compute_source_hypothesis_indices function. Is necessary to keep track of the transition scores.
        :type source_beam_indices: Optional[torch.Tensor]
        """
        all_hyps = []
        batch_hyp_size = sequences.shape[0] 
        for i in range(batch_hyp_size):
            # extract sequences of the hyp
            hyp_sequence = sequences[i].clone()
            # extract scores of the hyp
            hyp_transition_scores = transition_scores[i].clone()
            hyp_generated_transition_scores = hyp_transition_scores.clone()
            if source_hyps is not None and source_beam_indices is not None:
                hyp_transition_scores = torch.cat(
                    (
                        source_hyps[source_beam_indices[i]].transition_scores,
                        hyp_transition_scores
                    )
                )
            # extract last beam scores of the hyp
            hyp_last_beam_scores = last_beam_scores[i].clone()
            # extract past_key_values of the hyp
            hyp_past_key_values = self._extract_past_key_values(past_key_values, i, clone_tensors=True)
            # attention_mask of the hyp
            hyp_attention_mask = attention_mask[i].clone()

            hyp = SyntacticHypothesisUnshortenedContinuationData(
                hyp_sequence,
                hyp_transition_scores,
                hyp_generated_transition_scores,
                hyp_last_beam_scores,
                hyp_past_key_values,
                hyp_attention_mask
            )
            all_hyps.append(hyp)
        return all_hyps

    def pack_hypotheses(
        self,
        sequences: torch.Tensor,
        last_beam_scores: torch.Tensor,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        attention_mask: torch.Tensor,
        scores: torch.Tensor,
        beam_indices: torch.Tensor,
        keep_original_data: bool = False,
        source_hyps: Optional[List[ContinuationData]] = None,
        source_beam_indices: Optional[torch.Tensor] = None,
    ) -> List[ContinuationData]:
        all_hyps = []
        batch_hyp_size = sequences.shape[0] 
        original_data = None
        if keep_original_data:
            original_data = OriginalContinuationData(
                sequences=sequences,
                scores=scores,
                transition_scores=None,
                beam_indices=beam_indices,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                last_beam_scores=last_beam_scores
                )
        transition_scores = self.compute_transition_scores(
            sequences,
            scores,
            beam_indices
        )

        for i in range(batch_hyp_size):
            # extract sequences of the hyp
            hyp_sequence = sequences[i].clone()
            # extract scores of the hyp
            hyp_transition_scores = transition_scores[i].clone()
            # check if there were source hyps with earlier last_beam_scores
            if source_hyps is not None and source_beam_indices is not None:
                # match hyp by sequences
                hyp_transition_scores = torch.cat(
                    (
                        source_hyps[source_beam_indices[i]].transition_scores,
                        hyp_transition_scores
                    )
                )
            # extract last beam scores of the hyp
            hyp_last_beam_scores = last_beam_scores[i].clone()
            # extract past_key_values of the hyp
            hyp_past_key_values = self._extract_past_key_values(past_key_values, i, clone_tensors=True)
            # attention_mask of the hyp
            hyp_attention_mask = attention_mask[i].clone()

            hyp = ContinuationData(
                sequences=hyp_sequence,
                transition_scores=hyp_transition_scores,
                last_beam_scores=hyp_last_beam_scores,
                past_key_values=hyp_past_key_values,
                attention_mask=hyp_attention_mask,
                original_data=original_data
            )
            all_hyps.append(hyp)
        return all_hyps

    def _extract_past_key_values(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        hyp_idx: int,
        clone_tensors: bool = False
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """ 
        Extract the right slice of the past key values for a given hypothesis index.

        :param past_key_values: Past key values for a specific hypothesis. 
            Past key values are tuples of layers, then key and value. These respectively
            contain tensors. The shape of the tensors will be reduced from 
            `(batch_size, num_heads, sequence_length, head_dim)` to `(1, num_heads, sequence_length, head_dim)`.
        :type past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
        :param hyp_idx: Index of the hypothesis.
        :type hyp_idx: int
        :return: Extracted past key values.
        :rtype: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
        """
        # relevant parts of the past key values
        kv_pairs = tuple(
            torch.stack(layer) for layer in past_key_values
        )
        pkv = torch.stack(kv_pairs)
        # tensor is of shape (num_layers, key_value = 2, batch_hyp_idx, num_heads, sequence_length, head_dim)
        # need to extract the right batch_hyp_idx for a tensor of shape
        # (num_layers, key_value=2, batch_hyp_idx=1, num_heads, sequence_length, head_dim)
        hyp_pkv = pkv[:, :, hyp_idx:hyp_idx+1, :, :, :]
        if clone_tensors:
            hyp_pkv = hyp_pkv.clone()
        layer_tuples = torch.unbind(hyp_pkv, dim=0)
        layers_and_kv_tuples = tuple(
            tuple(torch.unbind(layer, dim=0)) for layer in layer_tuples
        )
        return layers_and_kv_tuples

    def _stack_past_key_values(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    ) -> torch.Tensor:
        kv_pairs = tuple(
            torch.stack(layer) for layer in past_key_values
        )
        return torch.stack(kv_pairs)

    def _unbind_past_key_values(
        self,
        past_key_values: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        layer_tuples = torch.unbind(past_key_values, dim=0)
        layers_and_kv_tuples = tuple(
            tuple(torch.unbind(layer, dim=0)) for layer in layer_tuples
        )
        return layers_and_kv_tuples

    def unpack_hypotheses(
        self,
        list_of_hypotheses: List[ContinuationData],
        return_original_data: bool = False
    ) -> Tuple[ModelOutput, Optional[OriginalContinuationData]]:
        """
        Unpack the list of hypotheses into a dictionary of model outputs.

        :param list_of_hypotheses: List of hypotheses.
        :type list_of_hypotheses: List[ContinuationData]
        :return: Dictionary of model outputs.
        :rtype: ModelOutput
        """
        sequences = torch.stack([hyp.sequences for hyp in list_of_hypotheses])
        transition_scores = torch.stack([hyp.transition_scores for hyp in list_of_hypotheses])
        last_beam_scores = torch.stack([hyp.last_beam_scores for hyp in list_of_hypotheses])
        past_key_values = self._reduce_past_key_values([hyp.past_key_values for hyp in list_of_hypotheses])
        attention_mask = torch.stack([hyp.attention_mask for hyp in list_of_hypotheses])
        original_data = None
        if return_original_data and list_of_hypotheses[0].original_data is not None:
            original_data = list_of_hypotheses[0].original_data
        return {
            "sequences": sequences,
            "transition_scores": transition_scores,
            "last_beam_scores": last_beam_scores,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask
        }, original_data

    def unpack_unsafe_syntactic_hypotheses(
        self,
        list_of_hypotheses: List[SyntacticHypothesis]
    ) -> ModelOutput:
        max_length = max([hyp.syntactic_hypothesis.sequences.shape[-1] for hyp in list_of_hypotheses])
        list_of_continuation_data = [
            self._expand_hyp_to_batch_length(
                hyp.syntactic_hypothesis,
                max_length,
                self.tokenizer.pad_token_id
                ) for hyp in list_of_hypotheses
        ]
        sequences = torch.stack([hyp.sequences for hyp in list_of_continuation_data])
        last_beam_scores = torch.stack([hyp.last_beam_scores for hyp in list_of_continuation_data])
        past_key_values = self._reduce_past_key_values([hyp.past_key_values for hyp in list_of_continuation_data])
        attention_mask = torch.stack([hyp.attention_mask for hyp in list_of_continuation_data])
        return {
            "sequences": sequences,
            "last_beam_scores": last_beam_scores,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask
        }

    def _reduce_past_key_values(
        self,
        past_key_values: List[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]
    ):
        """ 
        Reduce the past key values to only keep the last layer.

        :param past_key_values: Past key values for the model. The past key values contain
            values for the previously generated content. The structure
            as follow:
            - layer of the transformer
            - tuple of key and value tensors
            - tensor of shape (
                1,
                num_heads,
                sequence_length,
                head_dim
            )
        :type past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
        :return: Reduced past key values. Recreating the original shape.
        :rtype: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
        """
        # stack to a list of tensors
        hyps_as_stacked_tensors = [
            torch.stack(
                [torch.stack(layer) for layer in hyp]
            )
            for hyp in past_key_values
        ]
        # reconstruct original shape
        reduced_pkv = torch.cat(hyps_as_stacked_tensors, dim=2)
        # unbind the first two layers
        layer_tuples = torch.unbind(reduced_pkv, dim=0)
        layers_and_kv_tuples = tuple(
            tuple(torch.unbind(layer, dim=0)) for layer in layer_tuples
        )

        return layers_and_kv_tuples

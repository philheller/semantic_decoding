from typing import List, Optional, Tuple, Any, Dict, Union
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput, GenerationConfig
import torch
from semantic_decoding.generators.data_structures import (
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
        self.model = self._load_model(model_name, access_token)
        self.tokenizer = self._load_tokenizer(model_name, access_token)

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None, # can also be set via kwargs
        generation_config: Optional[GenerationConfig] = None,
        resume_generation: bool = False,
        past_key_values: Optional[torch.Tensor] = None,
        last_scores: Optional[torch.Tensor] = None,
        last_beam_scores: Optional[torch.Tensor] = None, # for manual setting of beam scores
        dynamic_decoder_prompt_length: Optional[int] = None,
        renormalize_logits: bool = True,
        **kwargs: Any
    ) -> GenerateBeamDecoderOnlyOutput:
        """
        Generates sequences using the syntactic model with specified parameters.

        :param inputs: A dictionary of inputs for the model, typically including 'input_ids' and 'attention_mask'.
        :param resume_generation: Whether to resume generation using past key values.
        :param past_key_values: Past key values to resume generation (if any).
        :param last_scores: Previous scores to influence generation (if any). Mostly for testing.
        :param last_beam_scores: Previous beam scores to influence generation (if any).
        :param dynamic_decoder_prompt_length: Length of the original prompt.
        :param renormalize_logits: Whether to renormalize logits during generation.
        :param kwargs: Additional arguments to pass to the generate function.
        :return: GenerateBeamDecoderOnlyOutput containing the generated sequences and other information.
        """
        return self.model.generate(
            inputs=inputs,
            generation_config=generation_config,
            resume_generation=resume_generation,
            past_key_values=past_key_values,
            last_scores=last_scores,
            last_beam_scores=last_beam_scores,
            dynamic_decoder_prompt_length=dynamic_decoder_prompt_length,
            renormalize_logits=renormalize_logits,
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
            access_token: Optional[str],
            force_even_split: bool = False
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
        print(f"Loading model: {model_name}")
        # Create a device map if more than one GPU is available
        device_map = "auto"
        if force_even_split:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                device_map = {str(i): i for i in range(num_gpus)}
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=access_token, 
            device_map=device_map
        )
        model.eval() # Though this is default, but just to be sure
        print(f"Model: {model_name}")
        print(f"Model on device: {model.device}")
        print(f"Model device map: {model.hf_device_map}")
        print(f"Using precision: {next(model.parameters()).dtype}")
        print(f"Eval mode: {not model.training}")
        return model

    def _load_tokenizer(self, model_name: str, access_token: Optional[str]) -> (PreTrainedTokenizer | PreTrainedTokenizerFast):
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
            # check if unk tokens is set
            if tokenizer.unk_token is not None:
                print(f"pad token is None. Setting pad token to same as unk token: {tokenizer.unk_token}")
                tokenizer.pad_token = tokenizer.unk_token
            elif tokenizer.eos_token is not None:
                print(f"pad token is None. Setting pad token to same as eos token: {tokenizer.eos_token}")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                raise ValueError("Pad token could be set to neither unk nor eos token.")
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

            for beam_idx in range(len(beam_indices)):
                # recursively walk back beam_indices to find the source hypothesis
                source_hyp_idx = self._get_source_hypothesis_idx(beam_indices, beam_idx) # type: ignore
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
        # the source beam indices are always carried over, which means
        # the first of the row is the source index
        # see @link https://huggingface.co/docs/transformers/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput.beam_indices
        # also see the function in GenerationMixin -> right after beam scorer
        return beam_indices[:, 0]

    def _get_source_hypothesis_idx(self, beam_indices, beam_idx) -> int:
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
        return beam_indices[beam_idx, 0]

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
        syntactic_source_hypothesis_indices: Optional[torch.Tensor] = None,
        empty_token: Optional[str] = None,
        shorten_left_when_possible: bool = False
    ) -> List[SyntacticHypothesis]:
        device = hypotheses[0].sequences.device
        all_hyps: List[SyntacticHypothesis] = []
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
                no_semantic_data = SemanticData.create_empty(empty_token)
                is_composite_aggregation_complete = False
                composite_aggregation_key = no_semantic_data.unique_key
                if semantic_source_hypothesis_indices is not None:
                    composite_aggregation_key = self.gather_composite_aggregation_key(
                        semantic_source_hypothesis_indices[hyp_idx].item(),
                        no_semantic_data
                    )
                    is_composite_aggregation_complete = True
                syn_hyp = SyntacticHypothesis(
                    composite_aggregation_key,
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
            if sem_data.is_eos_token:
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
                composite_aggregation_key = sem_data.unique_key
                is_composite_aggregation_complete = False
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
                    shortened_syn_hyp,
                    SyntacticHypothesisMetaData(
                        0
                    ),
                    is_aggregation_key_complete=is_composite_aggregation_complete
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
                shortened_hyp = self._shorten_hyp_right_by_amount_of_tokens_fast(
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
                            amount_tokens_shortened_after_data_point = original_size - len(piecewise_shortened_output) - 1
                        # check if the last tokens from the sequence are eos_tokens (which would not show during the decoding)
                        if (piecewise_shortened_output == self.tokenizer.eos_token_id).any():
                            # last tokens may be eos tokens, these need to be accounted for in shortening
                            amount_ending_eos_tokens = ((piecewise_shortened_output == self.tokenizer.eos_token_id)).nonzero().flatten().numel()
                            amount_tokens_shortened_after_data_point += amount_ending_eos_tokens
                        shortened_hyp = self._shorten_hyp_right_by_amount_of_tokens_fast(
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
                    # if no match can be found at all, sth is wrong; Worst case the semantic token contains some not explicitly marked syn tok
                    raise ValueError("Something went wrong. Unable to find match between syntactic tokens and raw string.\
                        This should not happen. Please check the code.")
        if shorten_left_when_possible:
            all_hyps = self._shorten_left_padding_tokens(all_hyps)
        return all_hyps

    def _shorten_hyp_right_by_amount_of_tokens_fast(
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
            pkv = hypothesis.past_key_values
            hypothesis.past_key_values = None # free pkv from continuation data (too much vram usage)
            return SyntacticHypothesisContinuationData(
                hypothesis.sequences.clone(),
                hypothesis.transition_scores.clone(),
                hypothesis.generated_transition_scores.clone(),
                hypothesis.last_beam_scores.clone(),
                pkv, # do not clone; too much vram usage
                hypothesis.attention_mask.clone(),
                hypothesis
            )
        shortened_sequences = hypothesis.sequences[:-shorten_by_amount_of_tokens].clone()
        shortened_transition_scores = hypothesis.transition_scores[:-shorten_by_amount_of_tokens].clone()
        shortened_generated_transition_scores = hypothesis.generated_transition_scores[:-shorten_by_amount_of_tokens].clone()
        # last beam scores need to be recalculated from transition_scores
        shortened_last_beam_scores = shortened_transition_scores.sum()
        # shorten past key values
        hypothesis.past_key_values = tuple(
            tuple(key_or_value[:, :, :-shorten_by_amount_of_tokens, :] for key_or_value in layer)
            for layer in hypothesis.past_key_values
        )
        shortened_past_key_values = hypothesis.past_key_values
        hypothesis.past_key_values = None # free pkv from continuation data (too much vram usage)
        
        shortened_attention_mask = hypothesis.attention_mask[:-shorten_by_amount_of_tokens].clone()
        return SyntacticHypothesisContinuationData(
            shortened_sequences,
            shortened_transition_scores,
            shortened_generated_transition_scores,
            shortened_last_beam_scores,
            shortened_past_key_values,
            shortened_attention_mask,
            hypothesis
        )

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
            pkv, pkv_device_map = hypothesis._stack_past_key_values()
            pkv = pkv.clone()
            pkv = SyntacticHypothesisContinuationData.unbind_past_key_values(pkv, pkv_device_map)
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
        past_key_values, pkv_device_map = hypothesis._stack_past_key_values()
        past_key_values = past_key_values[:, :, :, :, :-shorten_by_amount_of_tokens, :]
        past_key_values = SyntacticHypothesisContinuationData.unbind_past_key_values(past_key_values, pkv_device_map)
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

    def _shorten_left_padding_tokens(
        self,
        hypotheses: List[SyntacticHypothesis],
    ) -> List[SyntacticHypothesis]:
        # 1. get the min amount of left pading
        pad_token_id = self.tokenizer.pad_token_id
        min_amount_of_left_padding = min(
            [
                (hyp.syntactic_hypothesis.sequences != pad_token_id).nonzero().min().item() 
                for hyp in hypotheses
            ]
        )
        shorten_left_by = min_amount_of_left_padding
        if shorten_left_by < 1:
            return hypotheses
        else: 
            return [
                self._shorten_hyp_left_by_amount_of_tokens_unsafe_fast(hyp, shorten_left_by)
                for hyp in hypotheses
            ]

    def _shorten_hyp_left_by_amount_of_tokens_unsafe_fast(
        self,
        hypothesis: SyntacticHypothesis,
        shorten_by_amount_of_tokens: int
    ) -> SyntacticHypothesis:
        """ 
        This is an unsafe call. This means the function will not recalculate values.
        If too much of the hypothesis is shortened, not only will padding tokens be taken
        from the hypothesis and break it.
        Primarily, this is used to shorten padding tokens.
        """
        if shorten_by_amount_of_tokens == 0:
            # no need to shorten
            return hypothesis
        continuation_data = hypothesis.syntactic_hypothesis
        continuation_data.sequences = continuation_data.sequences[shorten_by_amount_of_tokens:]

        # shorten past key values
        continuation_data.past_key_values = tuple(
            tuple(key_or_value[:, :, shorten_by_amount_of_tokens:, :] for key_or_value in layer)
            for layer in continuation_data.past_key_values
        )
        continuation_data.attention_mask = continuation_data.attention_mask[shorten_by_amount_of_tokens:]
        
        return hypothesis

    def _shorten_hyp_left_by_amount_of_tokens_unsafe(
        self,
        hypothesis: SyntacticHypothesis,
        shorten_by_amount_of_tokens: int
    ) -> SyntacticHypothesis:
        """ 
        This is an unsafe call. This means the function will not recalculate values.
        If too much of the hypothesis is shortened, not only will padding tokens be taken
        from the hypothesis and break it.
        Primarily, this is used to shorten padding tokens.
        """
        continuation_data = hypothesis.syntactic_hypothesis
        if shorten_by_amount_of_tokens == 0:
            # no need to shorten
            pkv, pkv_device_map = continuation_data._stack_past_key_values()
            pkv = pkv.clone()
            pkv = SyntacticHypothesisContinuationData.unbind_past_key_values(pkv, pkv_device_map)
            shortened_hyp = SyntacticHypothesisContinuationData(
                continuation_data.sequences.clone(),
                continuation_data.transition_scores.clone(),
                continuation_data.generated_transition_scores.clone(),
                continuation_data.last_beam_scores.clone(),
                pkv,
                continuation_data.attention_mask.clone(),
                continuation_data.unshortened_data
            )
            hypothesis.syntactic_hypothesis = shortened_hyp
            return hypothesis
        shortened_sequences = continuation_data.sequences[shorten_by_amount_of_tokens:].clone()
        # (generated) transition scores and last beam scores are not to be shortened
        transition_scores = continuation_data.transition_scores.clone()
        generated_transition_scores = continuation_data.generated_transition_scores.clone()
        last_beam_scores = continuation_data.last_beam_scores.clone()
        # shorten past key values
        past_key_values, pkv_device_map = continuation_data._stack_past_key_values()
        past_key_values = past_key_values[:, :, :, :, shorten_by_amount_of_tokens:, :]
        past_key_values = SyntacticHypothesisContinuationData.unbind_past_key_values(past_key_values, pkv_device_map)
        attention_mask = continuation_data.attention_mask[shorten_by_amount_of_tokens:].clone()
        
        shortened_hyp = SyntacticHypothesisContinuationData(
            shortened_sequences,
            transition_scores,
            generated_transition_scores,
            last_beam_scores,
            past_key_values,
            attention_mask,
            continuation_data.unshortened_data
        )
        hypothesis.syntactic_hypothesis = shortened_hyp
        return hypothesis

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

    def _expand_hyp_to_batch_length_fast(
        self,
        hypothesis: SyntacticHypothesisContinuationData,
        target_length: int,
        pad_token_id: int
    ) -> SyntacticHypothesisContinuationData:
        # get sequences length
        current_length = hypothesis.sequences.shape[-1]
        missing_values = target_length - current_length

        sequence_filler = torch.full((missing_values,), pad_token_id).to(hypothesis.sequences.device)
        hypothesis.sequences = torch.cat((sequence_filler, hypothesis.sequences), dim=-1)

        # first approach: repeat the first past_key_values
        # select 1st tensor in 3rd dimension and repeat it
        hypothesis.past_key_values = tuple(
            tuple(
                torch.cat(
                    (key_or_value[:, :, 0, :].unsqueeze(2).repeat(1, 1, missing_values, 1), key_or_value),
                    dim=2
                    )
                for key_or_value in layer
            )
            for layer in hypothesis.past_key_values
        )

        hypothesis.attention_mask = torch.cat(
            (
                torch.zeros((missing_values,)).to(hypothesis.attention_mask.device),
                hypothesis.attention_mask
            ),
            dim=-1
        )

        return hypothesis

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
        past_key_values, pkv_device_map = hypothesis._stack_past_key_values()

        # select 1st tensor in 5th dimension and repeat it
        to_be_repeated = past_key_values[:, :, :, :, 0, :]
        repeated_tensor = to_be_repeated.unsqueeze(4).repeat(1, 1, 1, 1, missing_values, 1)

        new_pkv = torch.cat((repeated_tensor, past_key_values), dim=-2)
        new_pkv = SyntacticHypothesisContinuationData.unbind_past_key_values(new_pkv, pkv_device_map)

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
        batch_size: int,
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

        syntactic_beam_size = sequences.shape[0] // batch_size
        sequences = sequences.view(batch_size, syntactic_beam_size, sequences.shape[-1])
        mask_of_duplicates = []
        occurrences = []
        for b in range(batch_size):
            # check if a sequence is present multiple times
            sequences_as_tuple = [tuple(seq.tolist()) if seq is not None else None for seq in sequences[b]]
            # needs to be of size (batch_size, num_hyps_size)
            batch_mask_of_duplicates = torch.zeros(sequences[b].shape[0], dtype=torch.int).to(device)

            batch_occurrences = defaultdict(int)
            for i, t in enumerate(sequences_as_tuple):
                batch_occurrences[t] += 1
                batch_mask_of_duplicates[i] = 1 if batch_occurrences[t] > 1 else 0
            mask_of_duplicates.append(batch_mask_of_duplicates)
            occurrences.append(batch_occurrences)

        # merge mask_of_duplicates
        mask_of_duplicates = torch.stack(mask_of_duplicates, dim=0).flatten()
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
            hyp_past_key_values = self._extract_past_key_values_fast(past_key_values, i)
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

    def _extract_past_key_values_fast(
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
        
        tuple_of_hyp_idx = tuple(
            tuple(key_or_value[hyp_idx:hyp_idx+1, :, :, :] for key_or_value in layer)
            for layer in past_key_values
        )
        
        return tuple_of_hyp_idx

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
        pkv, pkv_device_map = SyntacticHypothesisContinuationData.stack_past_key_values(past_key_values)
        # tensor is of shape (num_layers, key_value = 2, batch_hyp_idx, num_heads, sequence_length, head_dim)
        # need to extract the right batch_hyp_idx for a tensor of shape
        # (num_layers, key_value=2, batch_hyp_idx=1, num_heads, sequence_length, head_dim)
        hyp_pkv = pkv[:, :, hyp_idx:hyp_idx+1, :, :, :]
        if clone_tensors:
            hyp_pkv = hyp_pkv.clone()
        layers_and_kv_tuples = SyntacticHypothesisContinuationData.unbind_past_key_values(hyp_pkv, pkv_device_map)
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
            self._expand_hyp_to_batch_length_fast(
                hyp.syntactic_hypothesis,
                max_length,
                self.tokenizer.pad_token_id
                ) for hyp in list_of_hypotheses
        ]
        sequences = torch.stack([hyp.sequences for hyp in list_of_continuation_data])
        last_beam_scores = torch.stack([hyp.last_beam_scores for hyp in list_of_continuation_data])
        past_key_values = self._reduce_past_key_values_fast([hyp.past_key_values for hyp in list_of_continuation_data])
        for hyp in list_of_continuation_data:
            hyp.past_key_values = None # free pkv vram
        attention_mask = torch.stack([hyp.attention_mask for hyp in list_of_continuation_data])
        return {
            "sequences": sequences,
            "last_beam_scores": last_beam_scores,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask
        }

    def _reduce_past_key_values_fast(
        self,
        list_of_past_key_values: List[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]
    ):
        """ 
        Reduce the past key values.

        :param list_of_past_key_values: List of past key values for the model. The past key values contain
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
        :type list_of_past_key_values: List[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]
        :return: Reduced past key values. Recreating the original shape.
        :rtype: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
        """
        # list of past key values is a list of 20 tuples of len 12, 2 which contains a tensor at -1 of shape (1, num_heads, sequence_length, head_dim)
        # now, these are ziped to 12, 2 tuples with each inner tuple containing a tensor of shape (20, num_heads, sequence_length, head_dim)
        num_layers = len(list_of_past_key_values[0])  # Should be 12
        reduced_pkv = []

        # Iterate through each layer (12 layers)
        for layer_idx in range(num_layers):
            # todo if vram issue, could incrementally decrease the pkv from the hpys
            # (would have to pass the entire hyps here and create a pop_hyp_pkv_layer function or sth like that)
            # Gather all keys and values for this layer across the batch (20 entries)
            keys = [hyp_pkv[layer_idx][0] for hyp_pkv in list_of_past_key_values]  # Extract keys for the layer
            values = [hyp_pkv[layer_idx][1] for hyp_pkv in list_of_past_key_values]  # Extract values for the layer
            
            stacked_keys = torch.stack(keys, dim=1).squeeze(0)
            stacked_values = torch.stack(values, dim=1).squeeze(0)
            
            # Append the reduced layer to the result
            reduced_pkv.append((stacked_keys, stacked_values))

        return tuple(reduced_pkv)

    def _reduce_past_key_values(
        self,
        hyps_past_key_values: List[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]
    ):
        """ 
        Reduce the past key values.

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
        hyps = [
            SyntacticHypothesisContinuationData.stack_past_key_values(pkv) for pkv in hyps_past_key_values
        ]
        hyps_as_stacked_tensors, device_maps = zip(*hyps)
        # assumption: devices on a layer are always the same
        assert all([device_map == device_maps[0] for device_map in device_maps]), "Devices should be the same on a layer for all hypotheses."
        device_map = device_maps[0]
        # reconstruct original shape
        reduced_pkv = torch.cat(hyps_as_stacked_tensors, dim=2)
        # unbind the first two layers
        layer_tuples = torch.unbind(reduced_pkv, dim=0)
        # make sure the tensors are correctly placed on the device of origin
        layer_tuples = tuple(
            layer.to(device_map[layer_idx]) for layer_idx, layer in enumerate(layer_tuples)
        )
        layers_and_kv_tuples = tuple(
            tuple(torch.unbind(layer, dim=0)) for layer in layer_tuples
        )

        return layers_and_kv_tuples

    def get_decoder_prompt_length_wo_padding(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        amount_tokens_non_pad = (sequences != self.tokenizer.pad_token_id).sum(-1)
        return amount_tokens_non_pad

    def update_decoder_prompt_length(
        self,
        sequences: torch.Tensor,
        original_decoder_prompt_length: torch.Tensor,
    ) -> torch.Tensor:
        amount_of_padding = ((sequences != self.tokenizer.pad_token_id) * 1).argmax(dim=-1)
        decoder_prompt_length = original_decoder_prompt_length.flatten() + amount_of_padding
        return decoder_prompt_length.view(original_decoder_prompt_length.shape)
    
    def gather_semantic_token_batches(
        self,
        syntactic_nested_beam_size: int,
        syntactic_beam_size: int,
        inputs: Dict[str, torch.Tensor],
        last_beam_scores: torch.Tensor,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        decoder_prompt_len: torch.Tensor,
    ) -> List[
            Tuple[
                Dict[str, torch.Tensor],
                torch.Tensor,
                Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                torch.Tensor
                ]
        ]:
        """ 
        Zips the input to a list of size `semantic_beam_size` with the
        tensors being sliced to size `batch_size * syntactic_nested_beam_size`.

        :param syntactic_nested_beam_size: Nested beam size.
        :type syntactic_nested_beam_size: int
        :param syntactic_beam_size: Beam size.
        :type syntactic_beam_size: int
        :param inputs: Model inputs.
        :type inputs: Dict[str, torch.Tensor]
        :param last_beam_scores: Last beam scores.
        :type last_beam_scores: torch.Tensor
        :param past_key_values: Past key values.
        :type past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
        :param decoder_prompt_len: Decoder prompt length.
        :type decoder_prompt_len: torch.Tensor
        :return: List of zipped model inputs: `inputs`, `last_beam_scores`, `past_key_values`, `decoder_prompt_len`. 
        :rtype: List[
            Tuple[
                Dict[str, torch.Tensor],
                torch.Tensor,
                Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                torch.Tensor
                ]
        ]
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        batch_size = input_ids.shape[0] // syntactic_beam_size
        sem_tok_batch_size = syntactic_beam_size // syntactic_nested_beam_size

        results = []
        to_be_batched_indices = torch.arange(sem_tok_batch_size).repeat_interleave(syntactic_nested_beam_size).repeat(1, batch_size).flatten()
        for sem_tok_batch_idx in range(sem_tok_batch_size):
            selection_indices = torch.nonzero(to_be_batched_indices == sem_tok_batch_idx).flatten()

            selected_input_ids = input_ids[selection_indices]
            selected_attention_mask = attention_mask[selection_indices]
            selected_last_beam_scores = last_beam_scores[selection_indices]
            selected_past_key_values = tuple(
                tuple(
                    key_or_value[selection_indices, :, :, :] for key_or_value in layer
                )
                for layer in past_key_values
            )
            selected_decoder_prompt_len = decoder_prompt_len.flatten()[selection_indices]
            selected_decoder_prompt_len = selected_decoder_prompt_len.view(batch_size, syntactic_nested_beam_size)

            results.append(
                (
                    {
                        "input_ids": selected_input_ids,
                        "attention_mask": selected_attention_mask
                    },
                    selected_last_beam_scores,
                    selected_past_key_values,
                    selected_decoder_prompt_len
                )
            )
        return results

    def scatter_semantic_token_batches(
        self,
        sem_tok_batches: List[
            Tuple[GenerateBeamDecoderOnlyOutput, torch.Tensor]
        ],
        syntactic_beam_size: int,
        syntactic_nested_beam_size: int,
        correct_beam_indices: bool = True,
    ) -> Tuple[
            Dict[
                str, Union[
                    torch.Tensor,
                    Tuple[
                        Tuple[
                            torch.Tensor, torch.Tensor
                        ], ...]
                ]
            ],
            torch.Tensor
        ]:
        """ 
        Reassembles structure as if model was run with singular large syntactic hyp pool.
        
        :param sem_tok_batches: List syntactic hypotheses (making up semantic token batches) and the transition scores.
            The scores will not be carried over any longer after that, so including this is paramount.
        :type sem_tok_batches: List[Tuple[GenerateBeamDecoderOnlyOutput, torch.Tensor]]
        :param syntactic_beam_size: Beam size.
        :type syntactic_beam_size: int
        :param syntactic_nested_beam_size: Nested beam size which is `syntactic_beam_size * batch_size // semantic_beam_size`.
        :type syntactic_nested_beam_size: int
        :param correct_beam_indices: Correct beam indices. Beam indices which are run in individual generation pools
            have a floored beam index. The beam indices are therefore globally not correct and need to be corrected.
            This option will automatically account for the mismatch and correct the beam indices.
        :type correct_beam_indices: bool        
        :return: Tuple containing 1. a dict with DecoderOnlyOutput keys (can use it like it, remember to use dict keys)
            and 2. the transition scores.
        :rtype: Tuple[
            Dict[
                str, Union[
                    torch.Tensor,
                    Tuple[
                        Tuple[
                            torch.Tensor, torch.Tensor
                        ],
                        ...
                    ]
                ]
            ],
            torch.Tensor
        ]
        """

        num_semantic_beams = len(sem_tok_batches)

        num_batches = sem_tok_batches[0][0]["sequences"].shape[0] // syntactic_nested_beam_size

        sem_tok_batch_indices = torch.arange(syntactic_beam_size*num_batches).view(num_semantic_beams*num_batches, syntactic_nested_beam_size)
        reordering_indices = torch.arange(
            num_semantic_beams*num_batches
        ).view(
            num_semantic_beams, num_batches
        ).transpose(0, 1).flatten()

        reorder_sem_tok_batch_indices = sem_tok_batch_indices[reordering_indices].flatten()

        
        # 1. concat all tensors
        output_sequences = torch.cat([sem_tok_batch[0]["sequences"] for sem_tok_batch in sem_tok_batches], dim=0)
        output_transition_scores = torch.cat([sem_tok_batch[1] for sem_tok_batch in sem_tok_batches], dim=0)
        output_beam_indices = torch.cat([sem_tok_batch[0]["beam_indices"] for sem_tok_batch in sem_tok_batches], dim=0)
        output_last_beam_scores = torch.cat([sem_tok_batch[0]["last_beam_scores"] for sem_tok_batch in sem_tok_batches], dim=0)
        # merge the past key values

        list_of_pkv = [
            sem_tok_batch[0]["past_key_values"] for sem_tok_batch in sem_tok_batches
        ]
        output_past_key_values = []
        for layer_idx in range(len(list_of_pkv[0])):  # Iterate over layers
            layer_tuples = []
            for key_or_value_idx in range(len(list_of_pkv[0][0])):
                # Concatenate the split tensors along the first dimension (w/4 parts)
                concatenated_tensor = torch.cat([kv[layer_idx][key_or_value_idx] for kv in list_of_pkv], dim=0)
                layer_tuples.append(concatenated_tensor)
            output_past_key_values.append(tuple(layer_tuples))
        output_past_key_values = tuple(output_past_key_values)
        

        output_attention_mask = torch.cat([sem_tok_batch[0]["attention_mask"] for sem_tok_batch in sem_tok_batches], dim=0)

        # reorder the tensors
        output_sequences = output_sequences[reorder_sem_tok_batch_indices]
        output_transition_scores = output_transition_scores[reorder_sem_tok_batch_indices]
        output_beam_indices = output_beam_indices[reorder_sem_tok_batch_indices]
        output_last_beam_scores = output_last_beam_scores[reorder_sem_tok_batch_indices]
        output_past_key_values = tuple(
            tuple(
                key_or_value[reorder_sem_tok_batch_indices, :, :, :]
                for key_or_value in layer
            )
            for layer in output_past_key_values
        )
        output_attention_mask = output_attention_mask[reorder_sem_tok_batch_indices]

        # ? correct beam indices
        # pooling the synt generation into individual searches produces incorrect beam indices.
        if correct_beam_indices:
            relevant_indices_mask = output_beam_indices >= 0

            # remove incorrect beams due to regular batching in nested syntactic/sem_toks decoding
            correct_pooling_indices = torch.arange(num_batches).repeat_interleave(
                syntactic_nested_beam_size*num_semantic_beams
            ).unsqueeze(1).expand(output_beam_indices.shape).to(output_beam_indices.device)
            correct_pooling_indices_a = correct_pooling_indices.clone() * syntactic_nested_beam_size
            output_beam_indices[relevant_indices_mask] -= correct_pooling_indices_a[relevant_indices_mask]

            # add correct beam globally indices
            correction_indices = torch.arange(num_semantic_beams*num_batches).repeat_interleave(
                syntactic_nested_beam_size
            ).to(output_beam_indices.device).unsqueeze(1) * syntactic_nested_beam_size
            correction_indices = correction_indices.expand(correction_indices.shape[0], output_beam_indices.shape[1])
            
            output_beam_indices[relevant_indices_mask] += correction_indices[relevant_indices_mask]
        
        return (
            {
                "sequences": output_sequences,
                "beam_indices": output_beam_indices,
                "last_beam_scores": output_last_beam_scores,
                "past_key_values": output_past_key_values,
                "attention_mask": output_attention_mask
            },
            output_transition_scores
        )
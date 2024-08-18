# from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any, Dict
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput 
import torch

class SyntacticGenerator:

    def __init__(self, model_name: str, device: str, access_token: Optional[str] = None):
        self.model = self._load_model(model_name, device, access_token)
        self.tokenizer = self._load_tokenizer(model_name, access_token)

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        return_dict_in_generate: bool = True,
        output_scores: bool = True,
        max_new_tokens: int = None,
        num_beams: int = None,
        num_return_sequences: int = None,
        resume_generation: bool = False,
        past_key_values: Optional[torch.Tensor] = None,
        last_scores: Optional[torch.Tensor] = None,
        last_beam_scores: Optional[torch.Tensor] = None, # for manual setting of beam scores
        renormalize_logits: bool = True,
        reproducibility: bool = False,
        length_penalty: Optional[float] = 1.0,  # same as default by hf
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
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
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences if num_return_sequences is not None else num_beams,
            resume_generation=resume_generation,
            past_key_values=past_key_values,
            last_scores=last_scores,
            last_beam_scores=last_beam_scores,
            renormalize_logits=renormalize_logits,
            reproducibility=reproducibility,
            length_penalty=length_penalty,
            do_sample=do_sample,  # Will use default if None
            temperature=temperature,  # Will use default if None
            top_k=top_k,  # Will use default if None
            top_p=top_p,  # Will use default if None
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

    def _load_model(self, model_name:str, device: str, access_token: Optional[str]) -> Any:
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
        return AutoModelForCausalLM.from_pretrained(
            model_name, token=access_token, 
            device_map="auto"
        ).to(device)

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
        return torch.sum(output_ids != self.tokenizer.eos_token_id, axis=1)

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
            input_length = torch.sum(input_ids != self.tokenizer.eos_token_id, axis=1)
            input_length_chars = torch.tensor([len(hyp) for hyp in hyps_in_plain_string])
        else:
            # check if inputs are expanded (first pass they are not)
            beam_batch_size = beam_indices.shape[0]
            if input_ids.shape[0] == beam_batch_size:
                hyps_in_plain_string = self.batch_decode(input_ids)
                input_length = torch.sum(input_ids != self.tokenizer.eos_token_id, axis=1)
                input_length_chars = torch.tensor([len(hyp) for hyp in hyps_in_plain_string])
            else:
                batch_size = input_ids.shape[0]
                beam_size = beam_batch_size // batch_size
                # expand to be of size (batch_size * beam_size, )
                expanded_inputs = input_ids.repeat_interleave(beam_size, dim=0)
                hyps_in_plain_string = self.batch_decode(expanded_inputs)
                input_length = torch.sum(expanded_inputs != self.tokenizer.eos_token_id, axis=1)
                input_length_chars = torch.tensor([len(hyp) for hyp in hyps_in_plain_string])
            # due to BS, we need to find the source hyp for each beam to be able to 
            # get the correct input length
            beam_hyp_input_len_chars = input_length_chars.clone()
            beam_hyp_input_length = input_length.clone()
            # Index of last beam_idx that is not -1
            first_non_negative_beam_idx = (beam_indices >= 0).sum(dim=1) - 1

            for beam_idx in range(len(beam_indices)):
                # recursively walk back beam_indices to find the source hypothesis
                source_hyp_idx = self._get_source_hypothesis_idx(beam_indices, beam_idx, step=first_non_negative_beam_idx[beam_idx].item())
                # reassign the input lengths to the ones from the source hypothesis
                beam_hyp_input_length[beam_idx] = input_length[source_hyp_idx]
                beam_hyp_input_len_chars[beam_idx] = input_length_chars[source_hyp_idx]

            input_length_chars = beam_hyp_input_len_chars
            input_length = beam_hyp_input_length
            del beam_hyp_input_len_chars, beam_hyp_input_length
        return input_length, input_length_chars

    def _get_source_hypothesis_idx(self, beam_indices, beam_idx, step=-1):
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

    def shorten_hyps_to_first_entity(
        self,
        first_new_entities: List[List[Dict[str, Any]]],
        sequences: torch.Tensor,
        decoded_sequences: List[str],
        attention_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        :return: Tuple of shortened sequences and attention masks.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
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
        target_size = altered_input_ids.shape[-1]

        for beam_hyp_idx, entity_in_hypothis in enumerate(first_new_entities):
            if len(entity_in_hypothis) == 0:
                # if no entity found, simply use the tokens as is
                altered_input_ids[beam_hyp_idx] = sequences[beam_hyp_idx].clone()
                altered_attention_mask[beam_hyp_idx] = attention_mask[beam_hyp_idx].clone()
                continue
            last_char = entity_in_hypothis[-1]["end"]
            
            shortened_output = decoded_sequences[beam_hyp_idx][:last_char]
            recomputed_tokens = self.tokenizer(shortened_output, return_tensors="pt", padding=True).to(device)
            last_sequence_id_from_recomputed = recomputed_tokens.input_ids[0][-1]

            # sequence_id of output without padding
            trimmed_sequence = sequences[
                    beam_hyp_idx,
                    (
                        sequences[beam_hyp_idx] != self.tokenizer.pad_token_id
                    ).nonzero().min():
                ].clone()

            if torch.equal(recomputed_tokens.input_ids[0], trimmed_sequence[:len(recomputed_tokens.input_ids[0])]):
                current_size = recomputed_tokens.input_ids.shape[-1]
                altered_input_ids[beam_hyp_idx] = torch.concat((
                        torch.tensor(
                                (target_size-current_size) * [self.tokenizer.pad_token_id]
                            ).to(device),
                        recomputed_tokens.input_ids[0]
                    ), dim=0
                )
                altered_attention_mask[beam_hyp_idx, -current_size:] = 1
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
                while (not match and len(piecewise_shortened_output) > 0):
                    # check if decoded is the same length as the end of the entity (first one wo shortening if entity ends with string)
                    decoded_piecewise = self.tokenizer.decode(piecewise_shortened_output, skip_special_tokens=True)
                    if len(decoded_piecewise) == last_char:
                        # add padding to beginning of sequence and fix attention mask
                        current_size = len(piecewise_shortened_output)
                        altered_input_ids[beam_hyp_idx] = torch.concat((
                                torch.tensor(
                                        (target_size-current_size) * [self.tokenizer.pad_token_id]
                                    ).to(device),
                                    piecewise_shortened_output
                            ), dim=0
                        )
                        altered_attention_mask[beam_hyp_idx, -current_size:] = 1
                        match = True
                        break
                    else:
                        piecewise_shortened_output = piecewise_shortened_output[:-1]
                if match:
                    continue
                # todo create special case (non code breaking); add a warning note showing that the instance could not be unified
                # if no match can be found at all, sth is wrong
                raise ValueError("Unable to find match between syntactic tokens and raw string")
        return altered_input_ids, altered_attention_mask

    def get_duplicates(
        self,
        sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[Tuple, List[torch.Tensor]]]:
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
        mask_of_duplicates = torch.zeros(sequences.shape[0]).to(device)

        occurrences = defaultdict(int)
        for i, t in enumerate(sequences_as_tuple):
            occurrences[t] += 1
            mask_of_duplicates[i] = 1 if occurrences[t] > 1 else 0
        return mask_of_duplicates, occurrences
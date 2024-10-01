import os
import sys
from typing import List, Optional, Union, Literal

from semantic_decoding.generators.syntactic import SyntacticGenerator
from semantic_decoding.generators.semantic import SemanticGenerator, SemanticGenerationConfig, SemanticGenerationMode
from semantic_decoding.generators.utils import report_memory, TimeReporter
from semantic_decoding.generators.data_structures import SemanticToken

import torch
from transformers.generation.utils import GenerationConfig
from transformers import logging
from transformers.generation.beam_search import BeamSearchScorer

time_reporter = TimeReporter()
logger = logging.get_logger()
class Generator:
    def __init__(
        self,
        model_name: str,
        semantic_generators: Union[List[str], str],
        device: str,
        access_token: str =None,
        unique_key: Literal["word", "text", "type"] = "word",
        normalize_unique_key=True,
    ):
        self.syntactic_generator = SyntacticGenerator(
            model_name,
            device,
            access_token=access_token
        )
        # just to make more accessible
        self.syntactic_tokenizer = self.syntactic_generator.tokenizer
        self.semantic_generator = SemanticGenerator(semantic_generators, normalize_unique_key, unique_key, device)
        # just to make more accessible
        self.semantic_tokenizer = self.semantic_generator.tokenizer
        self.device = device
        first_device = next(iter(self.syntactic_generator.model.hf_device_map.values()))
        self.first_device = torch.device(f'cuda:{first_device}')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        semantic_generation_config: SemanticGenerationConfig=SemanticGenerationConfig(),
        syntactic_generation_config: Optional[GenerationConfig]=None,
    ) -> List[SemanticToken]:
        
        syntactic_generation_config.pad_token_id = self.syntactic_generator.tokenizer.pad_token_id
        # general preparations
        model_inputs = self.syntactic_generator.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.first_device)
        
        # may be needed when a hyp is done (for search to not stop)
        non_special_token_id = torch.tensor(0).to(self.first_device)
        while non_special_token_id in self.syntactic_generator.tokenizer.all_special_ids:
            if non_special_token_id > self.syntactic_generator.tokenizer.vocab_size:
                raise ValueError("No non special token id found.")
            non_special_token_id += 1
        # the decoder prompt length will be used to update the new prompt len based on the added
        # padding and masking due to the dynamic size of the hypotheses
        original_decoder_prompt_len_wo_padding = self.syntactic_generator.get_decoder_prompt_length_wo_padding(
            model_inputs["input_ids"]
        )
        # repeat_interleave to match the amount of syntactic beams
        original_decoder_prompt_len_wo_padding = original_decoder_prompt_len_wo_padding.repeat_interleave(
            syntactic_generation_config.num_beams
            ).view(len(prompts), syntactic_generation_config.num_beams)
        original_decoder_prompt_len = model_inputs["input_ids"].shape[-1]
        # this needs to be updated on every iteration
        decoder_prompt_len = original_decoder_prompt_len

        # empty variables to be updated in the loop
        last_model_output = None
        last_past_key_values = None
        iter_output = None

        last_beam_scores = None

        # for generation
        # initial semantic token extraction simply grabs all semantic tokens
        input_length_chars = torch.zeros((len(prompts),), dtype=torch.long)
        _, all_initial_semantic_data = self.semantic_generator.generate(
            prompts,
            input_length_chars,
            include_all=True,
            syntactic_sequences=model_inputs["input_ids"],
            syntactic_eos_token_id=self.syntactic_generator.tokenizer.eos_token_id
        )

        semantic_inputs = self.semantic_generator.encode_semantic_sequences_from_semantic_data(all_initial_semantic_data)
        # # attention_mask is not really needed
        # semantic_inputs["attention_mask"] = semantic_generator.expand_semantic_sequences(semantic_inputs["attention_mask"], amount_semantic_beams)

        # values necessary to be initialized for all decoding strats
        batch_size = len(prompts)

        # map syntactic hyps to semantic hyps
        syn_to_sem_mapping = torch.arange(0, batch_size, dtype=torch.long, device=self.first_device) * semantic_generation_config.num_beams
        syn_to_sem_mapping = syn_to_sem_mapping.repeat_interleave(
            syntactic_generation_config.num_beams
        ).view(batch_size, syntactic_generation_config.num_beams)

        # which generation mode to use
        generation_mode = semantic_generation_config.get_generation_mode()
        if generation_mode == SemanticGenerationMode.GREEDY_SEARCH:
            semantic_inputs["input_ids"] = semantic_inputs["input_ids"].to(self.first_device)

            # setup for greedy search
            semantic_scores = torch.empty((batch_size,0)).to(self.first_device)
            
            last_syntactic_hyps = None
            counter = 0
            results = [
                None for _ in range(len(prompts))
            ]
            while (
                iter_output is None or
                iter_output.sequences.size(1) < semantic_generation_config.max_overall_tokens and
                max_amount_generated_tokens < semantic_generation_config.max_overall_generated_tokens
                ):
                print(counter, f"[{max_amount_generated_tokens}]")
                time_reporter.reset_timer()
                time_reporter.report_time("Start")
                #### 3. run model syntactic ####
                inputs = model_inputs if last_model_output is None else last_model_output

                iter_output = self.syntactic_generator.generate(
                    **inputs, # type: ignore
                    generation_config=syntactic_generation_config,
                    renormalize_logits=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    resume_generation=True if last_model_output is not None else False,
                    past_key_values=last_past_key_values if last_model_output is not None else None,
                    # ? last_beam_scores is used to avoid sampling of same sequences
                    last_beam_scores=last_beam_scores if last_model_output is not None else None,
                    dynamic_decoder_prompt_length=decoder_prompt_len,
                )
                time_reporter.report_time("After synt gen")

                #### 4. run semantic model ####
                # prepare generation output for semantic model - batch_decode to get sequences in strings
                hyps_decoded = self.syntactic_generator.batch_decode(iter_output.sequences)
                _, input_length_chars = self.syntactic_generator.get_input_length(
                        inputs["input_ids"], iter_output.beam_indices
                    )
                # # output_length = syntactic_generator.get_output_length(iter_output.sequences)
                # run semantic model -> List of (batch_size, )
                semantic_data, _ = self.semantic_generator.generate(
                    hyps_decoded,
                    input_length_chars,
                    False,
                    iter_output.sequences,
                    self.syntactic_generator.tokenizer.eos_token_id
                )

                #### 5. compute transition_scores ####
                transition_scores = self.syntactic_generator.compute_transition_scores(
                    iter_output.sequences,
                    iter_output.scores,
                    iter_output.beam_indices
                )
                time_reporter.report_time("Transition scores calced")
                
                #### 6. shorten til right after newest entity ####
                syntactic_source_hyp = self.syntactic_generator.compute_source_hypothesis_indices(
                    iter_output.beam_indices
                )
                unshortened_syntactic_hyps = self.syntactic_generator.pack_syntactic_hypotheses(
                    iter_output.sequences,
                    transition_scores,
                    iter_output.last_beam_scores,
                    iter_output.past_key_values,
                    iter_output.attention_mask,
                    last_syntactic_hyps,
                    syntactic_source_hyp
                )
                time_reporter.report_time("Packed hyps")

                shortened_hyps = self.syntactic_generator.shorten_hyp_to_first_semantic_data_point(
                    semantic_data,
                    unshortened_syntactic_hyps,
                    syn_to_sem_mapping.flatten()[syntactic_source_hyp],
                    syntactic_source_hyp,
                    empty_token=self.semantic_generator.tokenizer.decode(torch.tensor(self.semantic_generator.tokenizer.empty_token_id)),
                    shorten_left_when_possible=True
                )
                time_reporter.report_time("Shortened packed hyps")
    
                #### 7. semantic decoding ####
                semantic_tokens = self.semantic_generator.compute_semantic_tokens(
                    shortened_hyps,
                    syntactic_generation_config.num_beams,
                    1, # remenants of bs, greedy is like bs with one beam
                )
                # group semantic token by source beam idx, expanding from list
                # of shape (batch_size * num_beams, num_tokens) to
                # (batch_size, num_beams, num_tokens)
                semantic_tokens = self.semantic_generator.gather_tokens_by_source_beam(
                    semantic_tokens,
                    batch_size,
                    1 # remenants of bs, greedy is like bs with one beam
                )

                semantic_tokens_filled_hyps = semantic_tokens
                time_reporter.report_time("Semantic tokens calced")

                # now as tensors
                # 3 is an empty token (shell for all hyps when not a single semantic token found)
                # 0 is a padding token to be able to provide the min shape
                next_tokens, next_token_scores = self.semantic_generator.gather_next_tokens(
                    semantic_tokens_filled_hyps,
                    self.first_device
                )
                dynamic_vocab_size = next_token_scores.shape[-1]
                next_token_scores = next_token_scores.view((batch_size,semantic_generation_config.num_beams*dynamic_vocab_size))

                if semantic_generation_config.do_sample is True:
                    # todo implement sampling
                    raise NotImplementedError("Sampling not implemented yet.")
                else:
                    # get the next n_tokens_to_keep token indeces from the list
                    # should be 0 at all times (since they are by default sorted anyways)
                    next_token_indices = torch.argmax(
                        next_token_scores, dim=-1, keepdim=True
                    )
                
                next_indices = torch.div(next_token_indices, dynamic_vocab_size, rounding_mode='floor')
                next_tokens = next_tokens.view((batch_size, dynamic_vocab_size))    
                next_tokens = next_tokens.gather(1, next_token_indices)
                next_token_scores = next_token_scores.gather(1, next_token_indices)
                time_reporter.report_time("Next tokens and scores preparedyy")


                semantic_inputs["input_ids"] = torch.cat([semantic_inputs["input_ids"], next_tokens], dim=-1)
                semantic_scores = torch.cat(
                    (semantic_scores, next_token_scores), dim=-1
                )

                # get the source semantic hyps (tokens) and use their snytactic hyps 
                # for the next iteration input
                last_semantic_tokens = self.semantic_generator.filter_next_semantic_tokens(
                    semantic_tokens_filled_hyps,
                    torch.arange(next_indices.view(batch_size).numel()),
                    next_tokens.view(batch_size),
                    1,
                    padding_token_id=self.semantic_generator.tokenizer.pad_token_id
                )

                # check if any of the hyps is done
                already_has_result = torch.tensor([res is not None for res in results], dtype=torch.bool).to(self.first_device)
                contains_eos_token = (next_tokens == self.semantic_generator.tokenizer.eos_token_id).flatten()
                if contains_eos_token.any():
                    contains_eos_token_indices = contains_eos_token.nonzero().flatten()
                    eos_token_indices_already_in_results = torch.logical_and(already_has_result, contains_eos_token).nonzero().flatten()
                    for eos_candidate_idx in contains_eos_token_indices:
                        # those with eos token and which are already in results
                        if eos_candidate_idx in eos_token_indices_already_in_results:
                            pkv_like = last_semantic_tokens[eos_candidate_idx].syntactic_hypotheses[0].syntactic_hypothesis.stack_past_key_values()
                            empty_token = SemanticToken.create_empty(
                                f"{self.semantic_generator.tokenizer.decode(torch.tensor(self.semantic_generator.tokenizer.eos_token_id))}-continuation",
                                self.semantic_generator.tokenizer.eos_token_id,
                                non_special_token_id,
                                self.first_device,
                                pkv_like=pkv_like
                            )
                            last_semantic_tokens = tuple(
                                sem_tok if sem_tok_idx != eos_candidate_idx else empty_token for sem_tok_idx, sem_tok in enumerate(last_semantic_tokens)
                            )
                        # those with eos token and which are not in results yet
                        else:
                            result_tuple = (
                                last_semantic_tokens[eos_candidate_idx],
                                semantic_scores[eos_candidate_idx].clone(),
                                semantic_inputs["input_ids"][eos_candidate_idx].clone()
                            )
                            results[eos_candidate_idx] = result_tuple
                            # replace last semantic token with empty token (to avoid passing an eos hyp)
                            pkv_like = last_semantic_tokens[eos_candidate_idx].syntactic_hypotheses[0].syntactic_hypothesis._stack_past_key_values()
                            empty_token = SemanticToken.create_empty(
                                f"{self.semantic_generator.tokenizer.decode(torch.tensor(self.semantic_generator.tokenizer.eos_token_id))}-continuation",
                                self.semantic_generator.tokenizer.eos_token_id,
                                non_special_token_id,
                                self.first_device,
                                pkv_like=pkv_like
                            )
                            last_semantic_tokens = tuple(
                                sem_tok if sem_tok_idx != eos_candidate_idx else empty_token for sem_tok_idx, sem_tok in enumerate(last_semantic_tokens)
                            )
                if any(already_has_result):
                    already_has_result_indices = already_has_result.nonzero().flatten()
                    new_last_semantic_tokens = tuple(
                        sem_tok if sem_tok_idx not in already_has_result_indices else 
                        sem_tok.unsafe_shorten_empty_token(
                            non_special_token_id,
                            self.semantic_generator.low_score,
                        )
                        for sem_tok_idx, sem_tok in enumerate(last_semantic_tokens)
                    )
                        
                time_reporter.report_time("Processing of scores and tokens")
                packed_list_of_next_syntactic_hypotheses, syn_to_sem_mapping = self.semantic_generator.unpack_semantic_hypotheses(
                    last_semantic_tokens,
                    semantic_generation_config.num_beams,
                    syntactic_generation_config.num_beams,
                    device=syn_to_sem_mapping.device
                )
                last_syntactic_hyps = [
                    hyp.syntactic_hypothesis for hyp in packed_list_of_next_syntactic_hypotheses
                ]

                unpacked_list_of_next_syntactic_hypotheses = self.syntactic_generator.unpack_unsafe_syntactic_hypotheses(
                    packed_list_of_next_syntactic_hypotheses
                )
                time_reporter.report_time("Selected hyps unpacked and padded")

                # rename the unpacked_list_of_next_syntactic_hypotheses["sequences"] to "input_ids"
                altered_input_ids = unpacked_list_of_next_syntactic_hypotheses["sequences"]
                altered_attention_mask = unpacked_list_of_next_syntactic_hypotheses["attention_mask"]
                last_beam_scores = unpacked_list_of_next_syntactic_hypotheses["last_beam_scores"]
                last_past_key_values = unpacked_list_of_next_syntactic_hypotheses["past_key_values"]

                #### 8. mask duplicates and set it's score low ####
                # for the same beam hyps (same token id sequence), the beam score needs to be very low
                # and is set to -1e9. This is to ensure that the same hypothesis is not considered multiple times
                # which would result in sampling over the exact same tokens (leading to multiple same hypotheses).
                mask_of_duplicates, occurences  = self.syntactic_generator.get_duplicates(altered_input_ids, batch_size)
                # those which are duplicates will receive a low beam score to avoid sampling multiple times
                add_to_last_beam_scores = mask_of_duplicates * -1e9
                last_beam_scores = last_beam_scores + add_to_last_beam_scores
                # update the variable lengths of the decoder prompt (due to the variable hyp size + dynamic padding)
                decoder_prompt_len = self.syntactic_generator.update_decoder_prompt_length(
                    altered_input_ids,
                    original_decoder_prompt_len_wo_padding
                )

                # use the last model output for the next iteration
                last_model_output = {
                    "input_ids":  altered_input_ids,
                    "attention_mask": altered_attention_mask
                    }
                time_reporter.report_time("Next iteration inputs prepared")
                report_memory()
                counter += 1
                if all([True if res is not None else False for res in results]):
                    break
                max_amount_generated_tokens = altered_input_ids.shape[-1] - decoder_prompt_len.min()
                # flush buffer
                sys.stdout.flush()

            final_semantic_scores = torch.nn.utils.rnn.pad_sequence(
                [res[1] for res in results],
                True,
                0
            )
            final_semantic_sequences = torch.nn.utils.rnn.pad_sequence(
                [res[2] for res in results],
                True,
                self.semantic_generator.tokenizer.pad_token_id
            )
            final_syntactic_scores = torch.nn.utils.rnn.pad_sequence(
                [max(res[0].syntactic_hypotheses).syntactic_hypothesis.transition_scores for res in results],
                True,
                0
            )
            final_syntactic_sequences = torch.nn.utils.rnn.pad_sequence(
                [max(res[0].syntactic_hypotheses).syntactic_hypothesis.sequences for res in results],
                True,
                self.syntactic_generator.tokenizer.pad_token_id
            )

            return {
                "semantic_scores": final_semantic_scores,
                "semantic_sequences": final_semantic_sequences,
                "syntactic_scores": final_syntactic_scores,
                "syntactic_sequences": final_syntactic_sequences
            }

        elif generation_mode == SemanticGenerationMode.BEAM_SEARCH:
            semantic_decoder_prompt_len = semantic_inputs["input_ids"].shape[-1]
            # expand semantic inputs to match the amount of semantic beams
            semantic_inputs["input_ids"] = self.semantic_generator.expand_semantic_sequences(
                    semantic_inputs["input_ids"],
                    semantic_generation_config.num_beams
                )
            semantic_inputs["input_ids"] = semantic_inputs["input_ids"].to(self.first_device)

            # setup for beam search
            semantic_batch_beam_size = batch_size * semantic_generation_config.num_beams
            semantic_beam_indices = (
                tuple(() for _ in range(semantic_batch_beam_size))
            )

            beam_scorer = BeamSearchScorer(
                            batch_size=batch_size,
                            num_beams=semantic_generation_config.num_beams,
                            device=self.first_device,
                            length_penalty=semantic_generation_config.length_penalty,
                            do_early_stopping=semantic_generation_config.early_stopping,
                            num_beam_hyps_to_keep=semantic_generation_config.num_return_sequences,
                            max_length=semantic_generation_config.max_length,
                        )

            semantic_beam_scores = torch.zeros(
                (batch_size * semantic_generation_config.num_beams,),
                dtype=torch.float, device=self.first_device
            )
            semantic_scores = torch.empty((semantic_beam_scores.shape[-1], 0)).to(semantic_beam_scores.device)

            # empty vars to set up
            last_syntactic_hyps = None
            counter = 0
            is_done = torch.tensor([False] * batch_size).to(self.first_device)
            max_amount_generated_tokens = 0
            last_semantic_tokens = None

            while (
                iter_output is None or
                (
                    iter_output.sequences.size(1) < semantic_generation_config.max_overall_tokens
                    and max_amount_generated_tokens < semantic_generation_config.max_overall_generated_tokens
                )
                and not torch.all(beam_scorer._done)
            ):
                print(counter, f"[{max_amount_generated_tokens}]")
                time_reporter.reset_timer()
                time_reporter.report_time("Start")
                #### 3. run model syntactic ####
                inputs = model_inputs if last_model_output is None else last_model_output

                iter_output = self.syntactic_generator.generate(
                    **inputs, # type: ignore
                    generation_config=syntactic_generation_config,
                    renormalize_logits = True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    resume_generation=True if last_model_output is not None else False,
                    past_key_values=last_past_key_values if last_model_output is not None else None,
                    # ? last_beam_scores is used to avoid sampling of same sequences
                    last_beam_scores=last_beam_scores if last_model_output is not None else None,
                    dynamic_decoder_prompt_length=decoder_prompt_len,
                )
                time_reporter.report_time("After synt gen")

                #### 4. run semantic model ####
                # prepare generation output for semantic model - batch_decode to get sequences in strings
                hyps_decoded = self.syntactic_generator.batch_decode(iter_output.sequences)
                _, input_length_chars = self.syntactic_generator.get_input_length(
                        inputs["input_ids"], iter_output.beam_indices
                    )
                # # output_length = syntactic_generator.get_output_length(iter_output.sequences)
                # run semantic model -> List of (batch_size, )
                semantic_data, _ = self.semantic_generator.generate(
                    hyps_decoded,
                    input_length_chars,
                    False,
                    iter_output.sequences,
                    self.syntactic_generator.tokenizer.eos_token_id
                )
                time_reporter.report_time("After sem gen")
                
                #### 5. compute transition_scores ####
                transition_scores = self.syntactic_generator.compute_transition_scores(
                    iter_output.sequences,
                    iter_output.scores,
                    iter_output.beam_indices
                )
                
                time_reporter.report_time("Computed transition scores")
                #### 6. shorten til right after newest entity ####
                syntactic_source_hyp = self.syntactic_generator.compute_source_hypothesis_indices(
                    iter_output.beam_indices
                )
                unshortened_syntactic_hyps = self.syntactic_generator.pack_syntactic_hypotheses(
                    iter_output.sequences,
                    transition_scores,
                    iter_output.last_beam_scores,
                    iter_output.past_key_values,
                    iter_output.attention_mask,
                    last_syntactic_hyps,
                    syntactic_source_hyp
                )
                time_reporter.report_time("Packed syntactic hyps")

                shortened_hyps = self.syntactic_generator.shorten_hyp_to_first_semantic_data_point(
                    semantic_data,
                    unshortened_syntactic_hyps,
                    syn_to_sem_mapping.flatten()[syntactic_source_hyp],
                    syntactic_source_hyp,
                    empty_token=self.semantic_generator.tokenizer.decode(torch.tensor(self.semantic_generator.tokenizer.empty_token_id)),
                    shorten_left_when_possible=True,
                )
                time_reporter.report_time("Shortened hyps")

                #### 8. semantic decoding ####
                semantic_tokens = self.semantic_generator.compute_semantic_tokens(
                    shortened_hyps,
                    syntactic_generation_config.num_beams,
                    semantic_generation_config.num_beams
                )
                # group semantic token by source beam idx, expanding from list
                # of shape (batch_size * num_beams, num_tokens) to
                # (batch_size, num_beams, num_tokens)
                semantic_tokens = self.semantic_generator.gather_tokens_by_source_beam(
                    semantic_tokens,
                    batch_size,
                    semantic_generation_config.num_beams
                )

                time_reporter.report_time("Created semantic tokens")
                # if any of the the beams has no semantic tokens, fill with an empty
                # semantic token and set score to -1e9
                # ? this (semantic_tokens_filled_hyps) is an interesting data point that could well be used to record the progress
                (
                    semantic_tokens_filled_hyps,
                    semantic_beam_scores
                ) = self.semantic_generator.fill_empty_beam_hyps(
                    semantic_tokens,
                    semantic_beam_scores,
                    non_special_token_id
                )

                # now as tensors
                # 3 is an empty token (shell for all hyps when not a single semantic token found)
                # 0 is a padding token to be able to provide the min shape
                next_tokens, next_token_scores = self.semantic_generator.gather_next_tokens(
                    semantic_tokens_filled_hyps,
                    self.first_device
                )
                pure_token_scores = next_token_scores.clone()
                # until here, the scores are just for the final token. Now, add beam scores
                # to them to get the final scores
                next_token_scores = next_token_scores + semantic_beam_scores[:, None].expand_as(
                    next_token_scores
                )
                # next_token_indices = torch.arange(0, next_tokens.shape[0])[:, None].expand_as(
                #     next_tokens
                # ).to(device) % amount_semantic_beams
                dynamic_vocab_size = next_token_scores.shape[-1]

                # prepare inputs for beam scorer
                # get the next_token_scores
                # 1. gather from semantic tokens
                # 2. add beam scores to them

                ## pass all the necessary arguments to the beam scorer
                # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
                # non eos token per beam allowing expansion if required at all time for all hyps.
                sem_eos_token_id = self.semantic_generator.tokenizer.eos_token_id
                n_eos_tokens = torch.tensor([sem_eos_token_id]).shape[0] if sem_eos_token_id is not None else 0
                n_tokens_to_keep = max(2, 1 + n_eos_tokens) * semantic_generation_config.num_beams
                
                at_least_n_tokens_per_beam = all(
                    [
                        len(beam) >= n_tokens_to_keep for batch in semantic_tokens_filled_hyps for beam in batch
                    ]
                )
                if not at_least_n_tokens_per_beam and counter > 0:
                    logger.warning_once(f"At least one beam has less than {n_tokens_to_keep} tokens. Expansion of the beam strongly hindered. Consider increasing the syntactic hypothesis to semantic hypothesis ratio.")
                at_least_n_tokens_in_tensor = next_token_scores.shape[-1] >= n_tokens_to_keep
                if not at_least_n_tokens_in_tensor:
                    next_tokens = torch.nn.functional.pad(next_tokens, (0,1), value=self.semantic_generator.tokenizer.pad_token_id)
                    dynamic_vocab_size += 1
                    next_token_scores = torch.nn.functional.pad(next_token_scores, (0,1), value=self.semantic_generator.low_score)
                    pure_token_scores = torch.nn.functional.pad(pure_token_scores, (0,1), value=self.semantic_generator.low_score)
                next_tokens = next_tokens.view((batch_size, semantic_generation_config.num_beams*dynamic_vocab_size))    
                next_token_scores = next_token_scores.view((batch_size,semantic_generation_config.num_beams*dynamic_vocab_size))
                pure_token_scores = pure_token_scores.view((batch_size,semantic_generation_config.num_beams*dynamic_vocab_size))
    
                if semantic_generation_config.do_sample is True:
                    # todo implement sampling
                    raise NotImplementedError("Sampling not implemented yet.")
                else:
                    # get the next n_tokens_to_keep tokens and indeces from the list
                    # nts = next_token_scores.clone()
                    next_token_scores, next_token_indices = torch.topk(
                        next_token_scores, n_tokens_to_keep, dim=-1, largest=True, sorted=True
                    )
                pure_token_scores = pure_token_scores.gather(1, next_token_indices)
                next_indices = torch.div(next_token_indices, dynamic_vocab_size, rounding_mode='floor')
                next_tokens = next_tokens.gather(1, next_token_indices)
                next_semantic_tokens = self.semantic_generator.gather_semantic_tokens_by_index(
                    semantic_tokens_filled_hyps,
                    next_indices,
                    next_tokens
                )
                time_reporter.report_time("Beam scorer input prepeared")
                beam_outputs = beam_scorer.process(
                    semantic_inputs["input_ids"],   	# of shape (batch_size * num_beams, cur_len): input_ids up to this point
                    next_token_scores,                  # of shape (batch_size, n_tokens_to_keep): scores of next tokens
                    next_tokens,                        # of shape (batch_size, n_tokens_to_keep): next_tokens (0-vocab_size for all batches)
                    next_indices,                       # of shape (batch_size, n_tokens_to_keep): indices of next tokens (0-beam_size)
                    pad_token_id=self.semantic_generator.tokenizer.pad_token_id,
                    eos_token_id=self.semantic_generator.tokenizer.eos_token_id,
                    beam_indices=semantic_beam_indices, # tuples of tuples (batch_size * num_beams, ?)
                    decoder_prompt_len=semantic_decoder_prompt_len,
                    other=next_semantic_tokens
                )
                time_reporter.report_time("Beam scorer done")
                # 1. update input_ids with beam_idx and beam_next_tokens
                semantic_beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                semantic_next_beam_indices = beam_outputs["next_beam_indices"] # of shape (batch_size * sem_beam_size,); per beam values [batch_idx*sem_beam_size, batch_idx*sem_beam_size+sem_beam_size)

                # add pure token scores to record
                pure_token_scores = self.semantic_generator.calc_next_pure_semantic_scores(
                    pure_token_scores,
                    beam_next_tokens,
                    next_tokens,
                    replace_nan_with=-float("inf")
                )
                semantic_scores = torch.cat(
                    (semantic_scores, pure_token_scores.flatten().unsqueeze(1)),
                    dim=-1
                )
                semantic_inputs["input_ids"] = torch.cat([semantic_inputs["input_ids"][semantic_next_beam_indices, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                # theoretically: udpate attention_mask as well, but we do not need it

                semantic_beam_indices = tuple((semantic_beam_indices[semantic_next_beam_indices[i]] + (semantic_next_beam_indices[i],) for i in range(len(semantic_beam_indices))))

                # get the source semantic hyps (tokens) and use their snytactic hyps 
                # for the next iteration input
                last_semantic_tokens = self.semantic_generator.filter_next_semantic_tokens(
                    semantic_tokens_filled_hyps,
                    semantic_next_beam_indices,
                    beam_next_tokens,
                    semantic_generation_config.num_beams,
                    padding_token_id=self.semantic_generator.tokenizer.pad_token_id
                )

                if any(beam_scorer._done):
                    if not torch.all(beam_scorer._done == is_done):
                        print(f"{beam_scorer._done.sum()}/{len(beam_scorer._done)} batches [Done with: {beam_scorer._done.nonzero().flatten()}]")
                        is_done = beam_scorer._done.clone()

                    first_non_empty = None
                    for sem_tok in last_semantic_tokens:
                        if sem_tok is not None:
                            first_non_empty = sem_tok
                            break
                    # check if any batch has all None in last_semantic_tokens
                    batch_beams = [
                        last_semantic_tokens[batch_idx*semantic_generation_config.num_beams:(batch_idx+1)*semantic_generation_config.num_beams]
                        for batch_idx in range(batch_size)
                    ]
                    last_sem_toks = list(last_semantic_tokens)
                    for batch_idx, batch in enumerate(batch_beams):
                        if all([sem_tok is None for sem_tok in batch]):
                            last_sem_toks[batch_idx] = first_non_empty
                    last_semantic_tokens = tuple(last_sem_toks)
                    del batch_beams, last_sem_toks, first_non_empty

                if (
                    all(beam_scorer._done) or
                    iter_output.sequences.size(1) > semantic_generation_config.max_overall_tokens or
                    max_amount_generated_tokens > semantic_generation_config.max_overall_generated_tokens
                ):
                    # ? do not compute next syntactic hyps, no need
                    continue

                # if only eos tokens and otherwise only empty tokens, end the generation
                if all([sem_tok is None for sem_tok in last_semantic_tokens]):
                    break

                time_reporter.report_time("Beam scorer output processed")
                packed_list_of_next_syntactic_hypotheses, syn_to_sem_mapping = self.semantic_generator.unpack_semantic_hypotheses(
                    last_semantic_tokens,
                    semantic_generation_config.num_beams,
                    syntactic_generation_config.num_beams,
                    device=syn_to_sem_mapping.device
                )
                last_syntactic_hyps = [
                    hyp.syntactic_hypothesis for hyp in packed_list_of_next_syntactic_hypotheses
                ]

                unpacked_list_of_next_syntactic_hypotheses = self.syntactic_generator.unpack_unsafe_syntactic_hypotheses(
                    packed_list_of_next_syntactic_hypotheses
                )

                # rename the unpacked_list_of_next_syntactic_hypotheses["sequences"] to "input_ids"
                altered_input_ids = unpacked_list_of_next_syntactic_hypotheses["sequences"]
                altered_attention_mask = unpacked_list_of_next_syntactic_hypotheses["attention_mask"]
                last_beam_scores = unpacked_list_of_next_syntactic_hypotheses["last_beam_scores"]
                last_past_key_values = unpacked_list_of_next_syntactic_hypotheses["past_key_values"]

                #### 8. mask duplicates and set it's score low ####
                # for the same beam hyps (same token id sequence), the beam score needs to be very low
                # and is set to -1e9. This is to ensure that the same hypothesis is not considered multiple times
                # which would result in sampling over the exact same tokens (leading to multiple same hypotheses).
                mask_of_duplicates, _  = self.syntactic_generator.get_duplicates(altered_input_ids, batch_size)
                # those which are duplicates will receive a low beam score to avoid sampling multiple times
                add_to_last_beam_scores = mask_of_duplicates * -1e9
                last_beam_scores = last_beam_scores + add_to_last_beam_scores
                # update the variable lengths of the decoder prompt (due to the variable hyp size + dynamic padding)
                decoder_prompt_len = self.syntactic_generator.update_decoder_prompt_length(
                    altered_input_ids,
                    original_decoder_prompt_len_wo_padding
                )
                time_reporter.report_time("Updated next iteration values")
                # use the last model output for the next iteration
                last_model_output = {
                    "input_ids":  altered_input_ids,
                    "attention_mask": altered_attention_mask
                    }
                report_memory()
                counter += 1
                max_amount_generated_tokens = altered_input_ids.shape[-1] - decoder_prompt_len.min()
                sys.stdout.flush()

            max_length = 1e9
            if not all(beam_scorer._done):
                max_length = semantic_inputs["input_ids"].shape[-1]
            else:
                max_length = semantic_generation_config.max_length - semantic_decoder_prompt_len
            sequence_outputs = beam_scorer.finalize(
                semantic_inputs["input_ids"],
                semantic_beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.semantic_generator.tokenizer.pad_token_id,
                eos_token_id=self.semantic_generator.tokenizer.eos_token_id,
                max_length=max_length,
                beam_indices=semantic_beam_indices,
                decoder_prompt_len=semantic_decoder_prompt_len,
                other=next_semantic_tokens
            )

            final_semantic_sequences: torch.Tensor = sequence_outputs["sequences"]
            final_semantic_sequences_scores: torch.Tensor  = sequence_outputs["sequence_scores"]
            final_semantic_scores: torch.Tensor = semantic_scores
            final_semantic_beam_indices: torch.Tensor = sequence_outputs["beam_indices"]
            final_semantic_tokens: List[SemanticToken] = sequence_outputs["other"]

            # the transition scores summed at dim 1 and / (generated_len ** length penalty) equals to 
            # the sequence scores
            # need to check if properly aligned for stopped generation due to eos token
            final_semantic_transition_scores = self.semantic_generator.compute_transition_scores(
                final_semantic_sequences,
                final_semantic_scores,
                final_semantic_beam_indices,
                self.semantic_tokenizer,
                final_semantic_tokens,
            )

            final_syntactic_sequences = torch.nn.utils.rnn.pad_sequence(
                [
                    synt_hyp.syntactic_hypothesis.sequences 
                    for sem_tok in final_semantic_tokens if sem_tok is not None
                    for synt_hyp in sem_tok.syntactic_hypotheses
                ],
                batch_first=True,
                padding_value=self.syntactic_generator.tokenizer.eos_token_id
            )
            final_syntactic_scores = torch.nn.utils.rnn.pad_sequence(
                [
                    synt_hyp.syntactic_hypothesis.transition_scores
                    for sem_tok in final_semantic_tokens if sem_tok is not None
                    for synt_hyp in sem_tok.syntactic_hypotheses
                ],
                batch_first=True,
                padding_value=0
            )
            # to calc the `semantic_sequences_scores`, sum the transition_scores 
            # of the `final_semantic_transition_scores` and divide by `length**length_penalty`
            # >>> sem_sequences_scores = torch.div(
            #       final_semantic_transition_scores.sum(-1),
            #       torch.pow(
            #           (final_semantic_beam_indices >= 0).sum(-1),
            #           1.0
            #       )
            #    )

            return {
                "semantic_sequences": final_semantic_sequences,
                "semantic_scores": final_semantic_scores,
                "semantic_beam_indices": final_semantic_beam_indices,
                "semantic_sequences_scores": final_semantic_sequences_scores,
                "semantic_transition_scores": final_semantic_transition_scores,
                "syntactic_transition_scores": final_syntactic_scores,
                "syntactic_sequences": final_syntactic_sequences
            }

        elif generation_mode == SemanticGenerationMode.BEAM_SEARCH_NESTED:
            # use #syntactic_beams / #semantic_beams for each new semantic beam
            # for separate syntactic decodings
            if syntactic_generation_config.num_beams % semantic_generation_config.num_beams != 0:
                raise ValueError(
                    f"When using `nested_beam_search`, the number of syntactic beams must be divisible by the number of semantic beams. "
                    f"Currently, {syntactic_generation_config.num_beams} syntactic beams and {semantic_generation_config.num_beams} semantic beams are used."
                )
            syntactic_nested_beam_size = syntactic_generation_config.num_beams // semantic_generation_config.num_beams
            semantic_decoder_prompt_len = semantic_inputs["input_ids"].shape[-1]
            # expand semantic inputs to match the amount of semantic beams
            semantic_inputs["input_ids"] = self.semantic_generator.expand_semantic_sequences(
                    semantic_inputs["input_ids"],
                    semantic_generation_config.num_beams
                )
            semantic_inputs["input_ids"] = semantic_inputs["input_ids"].to(self.first_device)

            # setup for beam search
            semantic_batch_beam_size = batch_size * semantic_generation_config.num_beams
            semantic_beam_indices = (
                tuple(() for _ in range(semantic_batch_beam_size))
            )

            beam_scorer = BeamSearchScorer(
                            batch_size=batch_size,
                            num_beams=semantic_generation_config.num_beams,
                            device=self.first_device,
                            length_penalty=semantic_generation_config.length_penalty,
                            do_early_stopping=semantic_generation_config.early_stopping,
                            num_beam_hyps_to_keep=semantic_generation_config.num_return_sequences,
                            max_length=semantic_generation_config.max_length,
                        )

            semantic_beam_scores = torch.zeros(
                (batch_size * semantic_generation_config.num_beams,),
                dtype=torch.float, device=self.first_device
            )
            semantic_scores = torch.empty((semantic_beam_scores.shape[-1], 0)).to(semantic_beam_scores.device)

            # empty vars to set up
            last_syntactic_hyps = None
            counter = 0
            is_done = torch.tensor([False] * batch_size).to(self.first_device)
            max_amount_generated_tokens = 0
            last_semantic_tokens = None
            amount_syntactic_generation_steps = 1

            while (
                iter_output is None or
                (
                    iter_output["sequences"].size(1) < semantic_generation_config.max_overall_tokens
                    and max_amount_generated_tokens < semantic_generation_config.max_overall_generated_tokens
                )
                and not torch.all(beam_scorer._done)
            ):
                print(counter, f"[{max_amount_generated_tokens}]")
                time_reporter.reset_timer()
                time_reporter.report_time("Start")
                #### 3. run model syntactic ####
                inputs = model_inputs if last_model_output is None else last_model_output
                iter_output = None
                transition_scores = None
                
                if counter == 0:
                    iter_output = self.syntactic_generator.generate(
                        **inputs, # type: ignore
                        generation_config=syntactic_generation_config,
                        renormalize_logits = True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        resume_generation=True if last_model_output is not None else False,
                        past_key_values=last_past_key_values if last_model_output is not None else None,
                        # ? last_beam_scores is used to avoid sampling of same sequences
                        last_beam_scores=last_beam_scores if last_model_output is not None else None,
                        dynamic_decoder_prompt_length=decoder_prompt_len,
                    )
                    transition_scores = self.syntactic_generator.compute_transition_scores(
                        iter_output.sequences,
                        iter_output.scores,
                        iter_output.beam_indices
                    )
                else:
                    remembered_syntactic_beam_size = syntactic_generation_config.num_beams
                    remembered_returned_sequences = syntactic_generation_config.num_return_sequences
                    syntactic_generation_config.num_beams, syntactic_generation_config.num_return_sequences = syntactic_nested_beam_size, syntactic_nested_beam_size
                    outputs = []
                    all_temp_inputs = self.syntactic_generator.gather_semantic_token_batches(
                        syntactic_nested_beam_size,
                        remembered_syntactic_beam_size,
                        inputs,
                        last_beam_scores,
                        last_past_key_values,
                        decoder_prompt_len
                    )
                    for sem_tok_idx in range(amount_syntactic_generation_steps):
                        (
                            temp_inputs,
                            temp_last_beam_scores,
                            temp_last_past_key_values,
                            temp_decoder_prompt_len,
                        ) = all_temp_inputs[sem_tok_idx]
                        # ? if temp_last_beam_scores is self.low_score for a sem_tok_batch, this
                        # will produce gibberish and beam indices will be equal for that sem_tok.
                        # this is due to now previously found token which get filled up and are 
                        # masked with low scores to filter out. The filtering will happen automaitcally
                        # during the semantic_token generation as the outputs of this sem_tok are very low
                        sem_tok_iter_output = self.syntactic_generator.generate(
                            **temp_inputs, # type: ignore
                            generation_config=syntactic_generation_config,
                            renormalize_logits = True,
                            return_dict_in_generate=True,
                            output_scores=True,
                            resume_generation=True if last_model_output is not None else False,
                            past_key_values=temp_last_past_key_values if last_model_output is not None else None,
                            # ? last_beam_scores is used to avoid sampling of same sequences
                            last_beam_scores=temp_last_beam_scores if last_model_output is not None else None,
                            dynamic_decoder_prompt_length=temp_decoder_prompt_len,
                        )
                        temp_transition_scores = self.syntactic_generator.compute_transition_scores(
                            sem_tok_iter_output.sequences,
                            sem_tok_iter_output.scores,
                            sem_tok_iter_output.beam_indices
                        )
                        outputs.append((sem_tok_iter_output, temp_transition_scores))
                        # clean up
                        del temp_inputs, temp_last_beam_scores, temp_last_past_key_values, temp_decoder_prompt_len
                    del all_temp_inputs
                    syntactic_generation_config.num_beams = remembered_syntactic_beam_size
                    syntactic_generation_config.num_return_sequences = remembered_returned_sequences
                    iter_output, transition_scores = self.syntactic_generator.scatter_semantic_token_batches(
                        outputs,
                        syntactic_generation_config.num_beams,
                        syntactic_nested_beam_size,
                        correct_beam_indices=True,
                    )
                    

                time_reporter.report_time("After synt gen")

                #### 4. run semantic model ####
                # prepare generation output for semantic model - batch_decode to get sequences in strings
                hyps_decoded = self.syntactic_generator.batch_decode(iter_output["sequences"])
                _, input_length_chars = self.syntactic_generator.get_input_length(
                        inputs["input_ids"], iter_output["beam_indices"]
                    )
                # # output_length = syntactic_generator.get_output_length(iter_output.sequences)
                # run semantic model -> List of (batch_size, )
                semantic_data, _ = self.semantic_generator.generate(
                    hyps_decoded,
                    input_length_chars,
                    False,
                    iter_output["sequences"],
                    self.syntactic_generator.tokenizer.eos_token_id
                )
                time_reporter.report_time("After sem gen")
                
                #### 5. compute transition_scores ####
                # unlike in other decoding modes, transition scores are calculated earlier here, 
                # as the scores are split into individual semantic beam pools and moving the 
                # transition score calculation up avoids the necessity to reshape the synt scores
                # -> transition scores are calculated right after synt generation
                
                time_reporter.report_time("Computed transition scores")
                #### 6. shorten til right after newest entity ####
                syntactic_source_hyp = self.syntactic_generator.compute_source_hypothesis_indices(
                    iter_output["beam_indices"]
                )
                unshortened_syntactic_hyps = self.syntactic_generator.pack_syntactic_hypotheses(
                    iter_output["sequences"],
                    transition_scores,
                    iter_output["last_beam_scores"],
                    iter_output["past_key_values"],
                    iter_output["attention_mask"],
                    last_syntactic_hyps,
                    syntactic_source_hyp
                )
                time_reporter.report_time("Packed syntactic hyps")

                shortened_hyps = self.syntactic_generator.shorten_hyp_to_first_semantic_data_point(
                    semantic_data,
                    unshortened_syntactic_hyps,
                    syn_to_sem_mapping.flatten()[syntactic_source_hyp],
                    syntactic_source_hyp,
                    empty_token=self.semantic_generator.tokenizer.decode(torch.tensor(self.semantic_generator.tokenizer.empty_token_id)),
                    shorten_left_when_possible=True,
                )
                time_reporter.report_time("Shortened hyps")

                #### 8. semantic decoding ####
                semantic_tokens = self.semantic_generator.compute_semantic_tokens(
                    shortened_hyps,
                    syntactic_generation_config.num_beams,
                    semantic_generation_config.num_beams
                )
                # group semantic token by source beam idx, expanding from list
                # of shape (batch_size * num_beams, num_tokens) to
                # (batch_size, num_beams, num_tokens)
                semantic_tokens = self.semantic_generator.gather_tokens_by_source_beam(
                    semantic_tokens,
                    batch_size,
                    semantic_generation_config.num_beams
                )

                time_reporter.report_time("Created semantic tokens")
                # if any of the the beams has no semantic tokens, fill with an empty
                # semantic token and set score to -1e9
                # ? this (semantic_tokens_filled_hyps) is an interesting data point that could well be used to record the progress
                (
                    semantic_tokens_filled_hyps,
                    semantic_beam_scores
                ) = self.semantic_generator.fill_empty_beam_hyps(
                    semantic_tokens,
                    semantic_beam_scores,
                    non_special_token_id
                )

                # now as tensors
                # 3 is an empty token (shell for all hyps when not a single semantic token found)
                # 0 is a padding token to be able to provide the min shape
                next_tokens, next_token_scores = self.semantic_generator.gather_next_tokens(
                    semantic_tokens_filled_hyps,
                    self.first_device
                )
                pure_token_scores = next_token_scores.clone()
                # until here, the scores are just for the final token. Now, add beam scores
                # to them to get the final scores
                next_token_scores = next_token_scores + semantic_beam_scores[:, None].expand_as(
                    next_token_scores
                )
                # next_token_indices = torch.arange(0, next_tokens.shape[0])[:, None].expand_as(
                #     next_tokens
                # ).to(device) % amount_semantic_beams
                dynamic_vocab_size = next_token_scores.shape[-1]

                # prepare inputs for beam scorer
                # get the next_token_scores
                # 1. gather from semantic tokens
                # 2. add beam scores to them

                ## pass all the necessary arguments to the beam scorer
                # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
                # non eos token per beam allowing expansion if required at all time for all hyps.
                sem_eos_token_id = self.semantic_generator.tokenizer.eos_token_id
                n_eos_tokens = torch.tensor([sem_eos_token_id]).shape[0] if sem_eos_token_id is not None else 0
                n_tokens_to_keep = max(2, 1 + n_eos_tokens) * semantic_generation_config.num_beams
                
                at_least_n_tokens_per_beam = all(
                    [
                        len(beam) >= n_tokens_to_keep for batch in semantic_tokens_filled_hyps for beam in batch
                    ]
                )
                if not at_least_n_tokens_per_beam and counter > 0:
                    logger.warning_once(f"At least one beam has less than {n_tokens_to_keep} tokens. Expansion of the beam strongly hindered. Consider increasing the syntactic hypothesis to semantic hypothesis ratio.")
                at_least_n_tokens_in_tensor = next_token_scores.shape[-1] >= n_tokens_to_keep
                if not at_least_n_tokens_in_tensor:
                    next_tokens = torch.nn.functional.pad(next_tokens, (0,1), value=self.semantic_generator.tokenizer.pad_token_id)
                    dynamic_vocab_size += 1
                    next_token_scores = torch.nn.functional.pad(next_token_scores, (0,1), value=self.semantic_generator.low_score)
                    pure_token_scores = torch.nn.functional.pad(pure_token_scores, (0,1), value=self.semantic_generator.low_score)
                next_tokens = next_tokens.view((batch_size, semantic_generation_config.num_beams*dynamic_vocab_size))    
                next_token_scores = next_token_scores.view((batch_size,semantic_generation_config.num_beams*dynamic_vocab_size))
                pure_token_scores = pure_token_scores.view((batch_size,semantic_generation_config.num_beams*dynamic_vocab_size))
    
                if semantic_generation_config.do_sample is True:
                    # todo implement sampling
                    raise NotImplementedError("Sampling not implemented yet.")
                else:
                    # get the next n_tokens_to_keep tokens and indeces from the list
                    # nts = next_token_scores.clone()
                    next_token_scores, next_token_indices = torch.topk(
                        next_token_scores, n_tokens_to_keep, dim=-1, largest=True, sorted=True
                    )
                pure_token_scores = pure_token_scores.gather(1, next_token_indices)
                next_indices = torch.div(next_token_indices, dynamic_vocab_size, rounding_mode='floor')
                next_tokens = next_tokens.gather(1, next_token_indices)
                next_semantic_tokens = self.semantic_generator.gather_semantic_tokens_by_index(
                    semantic_tokens_filled_hyps,
                    next_indices,
                    next_tokens
                )
                time_reporter.report_time("Beam scorer input prepeared")
                beam_outputs = beam_scorer.process(
                    semantic_inputs["input_ids"],   	# of shape (batch_size * num_beams, cur_len): input_ids up to this point
                    next_token_scores,                  # of shape (batch_size, n_tokens_to_keep): scores of next tokens
                    next_tokens,                        # of shape (batch_size, n_tokens_to_keep): next_tokens (0-vocab_size for all batches)
                    next_indices,                       # of shape (batch_size, n_tokens_to_keep): indices of next tokens (0-beam_size)
                    pad_token_id=self.semantic_generator.tokenizer.pad_token_id,
                    eos_token_id=self.semantic_generator.tokenizer.eos_token_id,
                    beam_indices=semantic_beam_indices, # tuples of tuples (batch_size * num_beams, ?)
                    decoder_prompt_len=semantic_decoder_prompt_len,
                    other=next_semantic_tokens
                )
                time_reporter.report_time("Beam scorer done")
                # 1. update input_ids with beam_idx and beam_next_tokens
                semantic_beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                semantic_next_beam_indices = beam_outputs["next_beam_indices"] # of shape (batch_size * sem_beam_size,); per beam values [batch_idx*sem_beam_size, batch_idx*sem_beam_size+sem_beam_size)

                # add pure token scores to record
                pure_token_scores = self.semantic_generator.calc_next_pure_semantic_scores(
                    pure_token_scores,
                    beam_next_tokens,
                    next_tokens,
                    replace_nan_with=-float("inf")
                )
                semantic_scores = torch.cat(
                    (semantic_scores, pure_token_scores.flatten().unsqueeze(1)),
                    dim=-1
                )
                semantic_inputs["input_ids"] = torch.cat([semantic_inputs["input_ids"][semantic_next_beam_indices, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                # theoretically: udpate attention_mask as well, but we do not need it

                semantic_beam_indices = tuple((semantic_beam_indices[semantic_next_beam_indices[i]] + (semantic_next_beam_indices[i],) for i in range(len(semantic_beam_indices))))

                # get the source semantic hyps (tokens) and use their snytactic hyps 
                # for the next iteration input
                last_semantic_tokens = self.semantic_generator.filter_next_semantic_tokens(
                    semantic_tokens_filled_hyps,
                    semantic_next_beam_indices,
                    beam_next_tokens,
                    semantic_generation_config.num_beams,
                    padding_token_id=self.semantic_generator.tokenizer.pad_token_id
                )
                # the length of last_semantic_tokens will define how many syntactic runs will be done as each semantic tokens
                # is assigned a budget of #syntactic_beams / #semantic_beams syntactic beams
                # this means, that if #semantic_tokens < #semantic_beams, the remaining beams do not need to be explored and
                # syntactic generation can be skipped for these beams
                amount_syntactic_generation_steps = len(last_semantic_tokens) // batch_size

                if any(beam_scorer._done):
                    if not torch.all(beam_scorer._done == is_done):
                        print(f"{beam_scorer._done.sum()}/{len(beam_scorer._done)} batches [Done with: {beam_scorer._done.nonzero().flatten()}]")
                        is_done = beam_scorer._done.clone()

                    first_non_empty = None
                    for sem_tok in last_semantic_tokens:
                        if sem_tok is not None:
                            first_non_empty = sem_tok
                            break
                    # check if any batch has all None in last_semantic_tokens
                    batch_beams = [
                        last_semantic_tokens[batch_idx*semantic_generation_config.num_beams:(batch_idx+1)*semantic_generation_config.num_beams]
                        for batch_idx in range(batch_size)
                    ]
                    last_sem_toks = list(last_semantic_tokens)
                    for batch_idx, batch in enumerate(batch_beams):
                        if all([sem_tok is None for sem_tok in batch]):
                            last_sem_toks[batch_idx] = first_non_empty
                    last_semantic_tokens = tuple(last_sem_toks)
                    del batch_beams, last_sem_toks, first_non_empty

                if (
                    all(beam_scorer._done) or
                    iter_output["sequences"].size(1) > semantic_generation_config.max_overall_tokens or
                    max_amount_generated_tokens > semantic_generation_config.max_overall_generated_tokens
                ):
                    # ? do not compute next syntactic hyps, no need
                    continue

                # if only eos tokens and otherwise only empty tokens, end the generation
                if all([sem_tok is None for sem_tok in last_semantic_tokens]):
                    break

                time_reporter.report_time("Beam scorer output processed")
                packed_list_of_next_syntactic_hypotheses, syn_to_sem_mapping = self.semantic_generator.unpack_semantic_hypotheses_batched(
                    last_semantic_tokens,
                    semantic_generation_config.num_beams,
                    syntactic_generation_config.num_beams,
                    device=syn_to_sem_mapping.device
                )
                last_syntactic_hyps = [
                    hyp.syntactic_hypothesis for hyp in packed_list_of_next_syntactic_hypotheses
                ]

                unpacked_list_of_next_syntactic_hypotheses = self.syntactic_generator.unpack_unsafe_syntactic_hypotheses(
                    packed_list_of_next_syntactic_hypotheses
                )

                # rename the unpacked_list_of_next_syntactic_hypotheses["sequences"] to "input_ids"
                altered_input_ids = unpacked_list_of_next_syntactic_hypotheses["sequences"]
                altered_attention_mask = unpacked_list_of_next_syntactic_hypotheses["attention_mask"]
                last_beam_scores = unpacked_list_of_next_syntactic_hypotheses["last_beam_scores"]
                last_past_key_values = unpacked_list_of_next_syntactic_hypotheses["past_key_values"]

                #### 8. mask duplicates and set it's score low ####
                # for the same beam hyps (same token id sequence), the beam score needs to be very low
                # and is set to -1e9. This is to ensure that the same hypothesis is not considered multiple times
                # which would result in sampling over the exact same tokens (leading to multiple same hypotheses).
                mask_of_duplicates, _  = self.syntactic_generator.get_duplicates(altered_input_ids, batch_size)
                # those which are duplicates will receive a low beam score to avoid sampling multiple times
                add_to_last_beam_scores = mask_of_duplicates * -1e9
                last_beam_scores = last_beam_scores + add_to_last_beam_scores
                # update the variable lengths of the decoder prompt (due to the variable hyp size + dynamic padding)
                decoder_prompt_len = self.syntactic_generator.update_decoder_prompt_length(
                    altered_input_ids,
                    original_decoder_prompt_len_wo_padding
                )
                time_reporter.report_time("Updated next iteration values")
                # use the last model output for the next iteration
                last_model_output = {
                    "input_ids":  altered_input_ids,
                    "attention_mask": altered_attention_mask
                    }
                report_memory()
                counter += 1
                max_amount_generated_tokens = altered_input_ids.shape[-1] - decoder_prompt_len.min()
                sys.stdout.flush()

            max_length = 1e9
            if not all(beam_scorer._done):
                max_length = semantic_inputs["input_ids"].shape[-1]
            else:
                max_length = semantic_generation_config.max_length - semantic_decoder_prompt_len
            sequence_outputs = beam_scorer.finalize(
                semantic_inputs["input_ids"],
                semantic_beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.semantic_generator.tokenizer.pad_token_id,
                eos_token_id=self.semantic_generator.tokenizer.eos_token_id,
                max_length=max_length,
                beam_indices=semantic_beam_indices,
                decoder_prompt_len=semantic_decoder_prompt_len,
                other=next_semantic_tokens
            )

            final_semantic_sequences: torch.Tensor = sequence_outputs["sequences"]
            final_semantic_sequences_scores: torch.Tensor  = sequence_outputs["sequence_scores"]
            final_semantic_scores: torch.Tensor = semantic_scores
            final_semantic_beam_indices: torch.Tensor = sequence_outputs["beam_indices"]
            final_semantic_tokens: List[SemanticToken] = sequence_outputs["other"]

            # the transition scores summed at dim 1 and / (generated_len ** length penalty) equals to 
            # the sequence scores
            # need to check if properly aligned for stopped generation due to eos token
            final_semantic_transition_scores = self.semantic_generator.compute_transition_scores(
                final_semantic_sequences,
                final_semantic_scores,
                final_semantic_beam_indices,
                self.semantic_tokenizer,
                final_semantic_tokens,
            )

            final_syntactic_sequences = torch.nn.utils.rnn.pad_sequence(
                [
                    synt_hyp.syntactic_hypothesis.sequences 
                    for sem_tok in final_semantic_tokens if sem_tok is not None
                    for synt_hyp in sem_tok.syntactic_hypotheses
                ],
                batch_first=True,
                padding_value=self.syntactic_generator.tokenizer.eos_token_id
            )
            final_syntactic_scores = torch.nn.utils.rnn.pad_sequence(
                [
                    synt_hyp.syntactic_hypothesis.transition_scores
                    for sem_tok in final_semantic_tokens if sem_tok is not None
                    for synt_hyp in sem_tok.syntactic_hypotheses
                ],
                batch_first=True,
                padding_value=0
            )
            # to calc the `semantic_sequences_scores`, sum the transition_scores 
            # of the `final_semantic_transition_scores` and divide by `length**length_penalty`
            # >>> sem_sequences_scores = torch.div(
            #       final_semantic_transition_scores.sum(-1),
            #       torch.pow(
            #           (final_semantic_beam_indices >= 0).sum(-1),
            #           1.0
            #       )
            #    )

            return {
                "semantic_sequences": final_semantic_sequences,
                "semantic_scores": final_semantic_scores,
                "semantic_beam_indices": final_semantic_beam_indices,
                "semantic_sequences_scores": final_semantic_sequences_scores,
                "semantic_transition_scores": final_semantic_transition_scores,
                "syntactic_transition_scores": final_syntactic_scores,
                "syntactic_sequences": final_syntactic_sequences
            }

        else:
            raise ValueError(f"Generation mode {generation_mode} not supported.\n\
                Supported modes: {[mode.value for mode in SemanticGenerationMode]}")

from syntactic import SyntacticGenerator
from semantic import SemanticGenerator
from data_structures import SemanticToken
from generator import SemanticGenerationConfig
import torch
from transformers.generation.utils import GenerationConfig
from transformers import logging

# read access token from environment variable
import os
import time
logger = logging.get_logger()

start_time = time.time()
access_token = os.getenv("HF_TOKEN")
# some models are gated and require a hf token (make sure to request access to the model)
if access_token is not None:
    print(f"Access token: {access_token[:3]}{'*' * 16}")
else:
    print("No access token found.")
    # sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# print all available devices
print(f"Available devices: {torch.cuda.device_count()}")
print( f"Device names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

# selection of models
checkpoints = [
    "gpt2",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-v0.3",
]

# select the model you want to test
model_name = checkpoints[0]

#### 0. Experiments setup ####
# examples with batching and wo batching
example = ["Obama was born"]
# recommended: always compute in single batches, more batches 
# will not make scores reproduceable
examples = example + [
                # "Angela Merkel was born in",
                # "What is"
            ]
# chose the example you want to test (singular or batched)
# be warned: batching produces different results (just as masking)
prompt = example

#### Models config ####
# todo merge into generation config dicts
max_syntactic_tokens_per_iteration = 8
amount_syntactic_beams = 20
total_max_tokens = 1000
# this should remane 1 for greedy
amount_semantic_beams = 1


# generation configs
semantic_generation_config = SemanticGenerationConfig()
syntactic_generation_config = GenerationConfig(
    no_repeat_ngram_size=2,
    repetition_penalty = 1.0, # 1.0 is no penalty
)

#### 1. loading models ####
# syntactic generator
syntactic_generator = SyntacticGenerator(model_name, device, access_token)
tokenizer = syntactic_generator.tokenizer
syntactic_generation_config.pad_token_id = tokenizer.pad_token_id

# semantic generator
semantic_generator = SemanticGenerator("dslim/distilbert-NER", device)

#### 2. prepare inputs and outputs ####
model_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
non_special_token_id = torch.tensor(0).to(device)
while non_special_token_id in syntactic_generator.tokenizer.all_special_ids:
    if non_special_token_id > syntactic_generator.tokenizer.vocab_size:
        raise ValueError("No non special token id found.")
    non_special_token_id += 1
# the decoder prompt length will be used to update the new prompt len based on the added
# padding and masking due to the dynamic size of the hypotheses
original_decoder_prompt_len_wo_padding = syntactic_generator.get_decoder_prompt_length_wo_padding(
    model_inputs["input_ids"]
)
# repeat_interleave to match the amount of syntactic beams
original_decoder_prompt_len_wo_padding = original_decoder_prompt_len_wo_padding.repeat_interleave(
    amount_syntactic_beams
    ).view(len(prompt), amount_syntactic_beams)
original_decoder_prompt_len = model_inputs["input_ids"].shape[-1]
# this needs to be updated on every iteration
decoder_prompt_len = original_decoder_prompt_len

# empty variables to be updated in the loop
last_model_output = None
last_past_key_values = None
iter_output = None

semantic_output = None
last_beam_scores = None

# for generation
# initial semantic token extraction simply grabs all semantic tokens
input_length_chars = torch.zeros((len(prompt),), dtype=torch.long)
initial_semantic_data, all_initial_semantic_data = semantic_generator.generate(
    prompt,
    input_length_chars,
    include_all=True,
    syntactic_sequences=model_inputs["input_ids"],
    syntactic_eos_token_id=syntactic_generator.tokenizer.eos_token_id
)

semantic_inputs = semantic_generator.encode_semantic_sequences_from_semantic_data(all_initial_semantic_data)
semantic_inputs["input_ids"] = semantic_inputs["input_ids"].to(device)
# # attention_mask is not really needed
# semantic_inputs["attention_mask"] = semantic_generator.expand_semantic_sequences(semantic_inputs["attention_mask"], amount_semantic_beams)

# values necessary to be initialized
# general
batch_size = len(prompt)

semantic_scores = torch.empty((batch_size,0)).to(device)

# map syntactic hyps to semantic hyps
syn_to_sem_mapping = torch.arange(0, batch_size, dtype=torch.long, device=device)
syn_to_sem_mapping = syn_to_sem_mapping.repeat_interleave(amount_syntactic_beams).view(batch_size, amount_syntactic_beams)

last_syntactic_hyps = None
counter = 0
results = [
    None for _ in range(len(prompt))
]
while (iter_output is None or iter_output.sequences.size(1) < total_max_tokens):
    #### 3. run model syntactic ####
    inputs = model_inputs if last_model_output is None else last_model_output

    iter_output = syntactic_generator.generate(
    **inputs, # type: ignore
    max_new_tokens=max_syntactic_tokens_per_iteration,
    renormalize_logits = True,
    num_beams=amount_syntactic_beams,
    num_return_sequences=amount_syntactic_beams,
    return_dict_in_generate=True,
    output_scores=True,
    resume_generation=True if last_model_output is not None else False,
    past_key_values=last_past_key_values if last_model_output is not None else None,
    # ? last_beam_scores is used to avoid sampling of same sequences
    last_beam_scores=last_beam_scores if last_model_output is not None else None,
    dynamic_decoder_prompt_length=decoder_prompt_len,
    generation_config=syntactic_generation_config
    # last_scores = None if iter_output is None else iter_output.scores, # ? not used by default
    # length_penalty = 0,
    # # any sampling should be done with reproducibility = True
    # reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
    # do_sample = True,                         # if do_sample is True, use reproducibility = True
    # # use parameters at will
    # temperature = 0.2,                        # temperature for sampling
    # top_k = 50,                               # top_k for sampling
    )
    print(counter)

    #### 4. run semantic model ####
    # prepare generation output for semantic model - batch_decode to get sequences in strings
    hyps_decoded = syntactic_generator.batch_decode(iter_output.sequences)
    input_length, input_length_chars = syntactic_generator.get_input_length(
            inputs["input_ids"], iter_output.beam_indices
        )
    # # output_length = syntactic_generator.get_output_length(iter_output.sequences)
    # run semantic model -> List of (batch_size, )
    semantic_data, all_semantic_data = semantic_generator.generate(
        hyps_decoded,
        input_length_chars,
        False,
        iter_output.sequences,
        syntactic_generator.tokenizer.eos_token_id
    )
    
    #### 6. compute transition_scores ####
    transition_scores = syntactic_generator.compute_transition_scores(
        iter_output.sequences,
        iter_output.scores,
        iter_output.beam_indices
    )
    
    #### 7. shorten til right after newest entity ####
    syntactic_source_hyp = syntactic_generator.compute_source_hypothesis_indices(
        iter_output.beam_indices
    )
    unshortened_syntactic_hyps = syntactic_generator.pack_syntactic_hypotheses(
        iter_output.sequences,
        transition_scores,
        iter_output.last_beam_scores,
        iter_output.past_key_values,
        iter_output.attention_mask,
        last_syntactic_hyps,
        syntactic_source_hyp
    )

    shortened_hyps = syntactic_generator.shorten_hyp_to_first_semantic_data_point(
        semantic_data,
        unshortened_syntactic_hyps,
        syn_to_sem_mapping.flatten()[syntactic_source_hyp],
        syntactic_source_hyp,
        empty_token=semantic_generator.tokenizer.decode(torch.tensor(semantic_generator.tokenizer.empty_token_id)),
        shorten_left_when_possible=True
    )

    #### 8. semantic decoding ####
    semantic_tokens = semantic_generator.compute_semantic_tokens(
        shortened_hyps,
        amount_syntactic_beams,
        1, # remenants of bs, greedy is like bs with one beam
    )
    # group semantic token by source beam idx, expanding from list
    # of shape (batch_size * num_beams, num_tokens) to
    # (batch_size, num_beams, num_tokens)
    semantic_tokens = semantic_generator.gather_tokens_by_source_beam(
        semantic_tokens,
        batch_size,
        1 # remenants of bs, greedy is like bs with one beam
    )

    semantic_tokens_filled_hyps = semantic_tokens

    # now as tensors
    # 3 is an empty token (shell for all hyps when not a single semantic token found)
    # 0 is a padding token to be able to provide the min shape
    next_tokens, next_token_scores = semantic_generator.gather_next_tokens(
        semantic_tokens_filled_hyps,
        device
    )
    dynamic_vocab_size = next_token_scores.shape[-1]
    next_token_scores = next_token_scores.view((batch_size,amount_semantic_beams*dynamic_vocab_size))
    
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


    semantic_inputs["input_ids"] = torch.cat([semantic_inputs["input_ids"], next_tokens], dim=-1)
    semantic_scores = torch.cat(
        (semantic_scores, next_token_scores), dim=-1
    )

    # get the source semantic hyps (tokens) and use their snytactic hyps 
    # for the next iteration input
    last_semantic_tokens = semantic_generator.filter_next_semantic_tokens(
        semantic_tokens_filled_hyps,
        torch.arange(next_indices.view(batch_size).numel()),
        next_tokens.view(batch_size),
        1,
        padding_token_id=semantic_generator.tokenizer.pad_token_id
    )

    # check if any of the hyps is done
    already_has_result = torch.tensor([res is not None for res in results], dtype=torch.bool).to(device)
    contains_eos_token = (next_tokens == semantic_generator.tokenizer.eos_token_id).flatten()
    if contains_eos_token.any():
        contains_eos_token_indices = contains_eos_token.nonzero().flatten()
        eos_token_indices_already_in_results = torch.logical_and(already_has_result, contains_eos_token).nonzero().flatten()
        for eos_candidate_idx in contains_eos_token_indices:
            # those with eos token and which are already in results
            if eos_candidate_idx in eos_token_indices_already_in_results:
                pkv_like = last_semantic_tokens[eos_candidate_idx].syntactic_hypotheses[0].syntactic_hypothesis.stack_past_key_values()
                empty_token = SemanticToken.create_empty(
                    f"{semantic_generator.tokenizer.decode(torch.tensor(semantic_generator.tokenizer.eos_token_id))}-continuation",
                    semantic_generator.tokenizer.eos_token_id,
                    non_special_token_id,
                    device,
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
                    f"{semantic_generator.tokenizer.decode(torch.tensor(semantic_generator.tokenizer.eos_token_id))}-continuation",
                    semantic_generator.tokenizer.eos_token_id,
                    non_special_token_id,
                    device,
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
                semantic_generator.low_score,
            )
            for sem_tok_idx, sem_tok in enumerate(last_semantic_tokens)
        )
            

    packed_list_of_next_syntactic_hypotheses, syn_to_sem_mapping = semantic_generator.unpack_semantic_hypotheses(
        last_semantic_tokens,
        amount_semantic_beams,
        amount_syntactic_beams,
        device=syn_to_sem_mapping.device
    )
    last_syntactic_hyps = [
        hyp.syntactic_hypothesis for hyp in packed_list_of_next_syntactic_hypotheses
    ]

    unpacked_list_of_next_syntactic_hypotheses = syntactic_generator.unpack_unsafe_syntactic_hypotheses(
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
    mask_of_duplicates, occurences  = syntactic_generator.get_duplicates(altered_input_ids)
    # those which are duplicates will receive a low beam score to avoid sampling multiple times
    add_to_last_beam_scores = mask_of_duplicates * -1e9
    last_beam_scores = last_beam_scores + add_to_last_beam_scores
    # update the variable lengths of the decoder prompt (due to the variable hyp size + dynamic padding)
    decoder_prompt_len = syntactic_generator.update_decoder_prompt_length(
        altered_input_ids,
        original_decoder_prompt_len_wo_padding
    )

    # use the last model output for the next iteration
    last_model_output = {
        "input_ids":  altered_input_ids,
        "attention_mask": altered_attention_mask
        }
    counter += 1
    # torch.cuda.empty_cache()
    if all([True if res is not None else False for res in results]):
        break

print("Results:")
print(
    syntactic_generator.tokenizer.batch_decode(
        torch.nn.utils.rnn.pad_sequence(
            [max(res[0].syntactic_hypotheses).syntactic_hypothesis.sequences for res in results],
            True,
            syntactic_generator.tokenizer.pad_token_id
        ),
        skip_special_tokens=True
    )
)
print("Semantic output")
print(
    semantic_generator.tokenizer.batch_decode(
        torch.nn.utils.rnn.pad_sequence(
            [res[2] for res in results],
            True,
            semantic_generator.tokenizer.pad_token_id
        ),
    )
)
print("Scores")
print(
    [
        res[1] for res in results
    ]
)

print(f"Final time: {time.time() - start_time:.2f}")


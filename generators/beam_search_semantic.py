from syntactic import SyntacticGenerator
from semantic import SemanticGenerator
from generator import SemanticGenerationConfig
import torch
from utils import deep_compare
from transformers.generation.utils import GenerationConfig
from transformers.generation.beam_search import BeamSearchScorer
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


#### 0. Experiment setup ####
# examples with batching and wo batching
example = ["Obama was born"]
# recommended: always compute in single batches, more batches 
# will not make scores reproduceable
examples = example + [
                # "Angela Merkel was born in",
                "What is"
            ]
# chose the example you want to test (singular or batched);
# be warned: batching produces different results (just as masking)
prompt = example


#### Models config ####
# todo merge into generation config dicts
max_syntactic_tokens_per_iteration = 8
amount_syntactic_beams = 20
total_max_tokens = 1000
amount_semantic_beams = 2


# generator configs
semantic_generation_config = SemanticGenerationConfig(
    num_beams=amount_semantic_beams,
    num_return_sequences=amount_semantic_beams,
    length_penalty=-.7,
)
syntactic_generation_config = GenerationConfig(
    no_repeat_ngram_size=2,
    repetition_penalty = 1.0, # 1.0 is no penalty
    length_penalty=-.7
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
semantic_decoder_prompt_len = semantic_inputs["input_ids"].shape[-1]
# expand semantic inputs to match the amount of semantic beams
semantic_inputs["input_ids"] = semantic_generator.expand_semantic_sequences(
        semantic_inputs["input_ids"],
        amount_semantic_beams
    )
semantic_inputs["input_ids"] = semantic_inputs["input_ids"].to(device)
# # attention_mask is not really needed
# semantic_inputs["attention_mask"] = semantic_generator.expand_semantic_sequences(semantic_inputs["attention_mask"], amount_semantic_beams)

# values necessary to be initialized
# general
batch_size = len(prompt)
# bs
semantic_batch_beam_size = batch_size * amount_semantic_beams
semantic_beam_indices = (
    tuple(() for _ in range(semantic_batch_beam_size))
)

beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=amount_semantic_beams,
                device=device,
                length_penalty=semantic_generation_config.length_penalty,
                do_early_stopping=semantic_generation_config.early_stopping,
                num_beam_hyps_to_keep=semantic_generation_config.num_return_sequences,
                max_length=semantic_generation_config.max_length,
            )

semantic_beam_scores = torch.zeros((batch_size * amount_semantic_beams,), dtype=torch.float, device=device)
semantic_scores = torch.zeros_like(semantic_beam_scores)[:, None]

# map syntactic hyps to semantic hyps
syn_to_sem_mapping = torch.arange(0, batch_size, dtype=torch.long, device=device) * amount_semantic_beams
syn_to_sem_mapping = syn_to_sem_mapping.repeat_interleave(amount_syntactic_beams).view(batch_size, amount_syntactic_beams)

last_syntactic_hyps = None
counter = 0
is_done = torch.tensor([False] * batch_size).to(device)
last_semantic_tokens = None
while (iter_output is None or iter_output.sequences.size(1) < total_max_tokens and not torch.all(beam_scorer._done)):
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
    generation_config=syntactic_generation_config,
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
        shorten_left_when_possible=True,
    )

    #### 8. semantic decoding ####
    semantic_tokens = semantic_generator.compute_semantic_tokens(
        shortened_hyps,
        amount_syntactic_beams,
        amount_semantic_beams
    )
    # group semantic token by source beam idx, expanding from list
    # of shape (batch_size * num_beams, num_tokens) to
    # (batch_size, num_beams, num_tokens)
    semantic_tokens = semantic_generator.gather_tokens_by_source_beam(
        semantic_tokens,
        batch_size,
        amount_semantic_beams
    )
    # if any of the the beams has no semantic tokens, fill with an empty
    # semantic token and set score to -1e9
    # ? this (semantic_tokens_filled_hyps) is an interesting data point that could well be used to record the progress
    (
        semantic_tokens_filled_hyps,
        semantic_beam_scores
    ) = semantic_generator.fill_empty_beam_hyps(
        semantic_tokens,
        semantic_beam_scores,
        non_special_token_id
    )

    # now as tensors
    # 3 is an empty token (shell for all hyps when not a single semantic token found)
    # 0 is a padding token to be able to provide the min shape
    next_tokens, next_token_scores = semantic_generator.gather_next_tokens(
        semantic_tokens_filled_hyps,
        device
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
    sem_eos_token_id = semantic_generator.tokenizer.eos_token_id
    n_eos_tokens = torch.tensor([sem_eos_token_id]).shape[0] if sem_eos_token_id is not None else 0
    n_tokens_to_keep = max(2, 1 + n_eos_tokens) * amount_semantic_beams
    
    at_least_n_tokens_per_beam = all(
        [
            len(beam) >= n_tokens_to_keep for batch in semantic_tokens_filled_hyps for beam in batch
        ]
    )
    if not at_least_n_tokens_per_beam and counter > 0:
        logger.warning_once(f"At least one beam has less than {n_tokens_to_keep} tokens. Expansion of the beam strongly hindered. Consider increasing the syntactic hypothesis to semantic hypothesis ratio.")
    at_least_n_tokens_in_tensor = next_token_scores.shape[-1] >= n_tokens_to_keep
    if not at_least_n_tokens_in_tensor:
        next_tokens = torch.nn.functional.pad(next_tokens, (0,1), value=semantic_generator.tokenizer.pad_token_id)
        dynamic_vocab_size += 1
        next_token_scores = torch.nn.functional.pad(next_token_scores, (0,1), value=semantic_generator.low_score)
        pure_token_scores = torch.nn.functional.pad(pure_token_scores, (0,1), value=semantic_generator.low_score)
    next_tokens = next_tokens.view((batch_size, amount_semantic_beams*dynamic_vocab_size))    
    next_token_scores = next_token_scores.view((batch_size,amount_semantic_beams*dynamic_vocab_size))
    pure_token_scores = pure_token_scores.view((batch_size,amount_semantic_beams*dynamic_vocab_size))
    
    if semantic_generation_config.do_sample is True:
        # todo implement sampling
        raise NotImplementedError("Sampling not implemented yet.")
    else:
        # get the next n_tokens_to_keep tokens and indeces from the list
        nts = next_token_scores.clone()
        next_token_scores, next_token_indices = torch.topk(
            next_token_scores, n_tokens_to_keep, dim=-1, largest=True, sorted=True
        )
    pure_token_scores = pure_token_scores.gather(1, next_token_indices)
    next_indices = torch.div(next_token_indices, dynamic_vocab_size, rounding_mode='floor')
    next_tokens = next_tokens.gather(1, next_token_indices)
    next_semantic_tokens = semantic_generator.gather_semantic_tokens_by_index(
        semantic_tokens_filled_hyps,
        next_indices,
        next_tokens
    )

    beam_outputs = beam_scorer.process(
        semantic_inputs["input_ids"],   	# of shape (batch_size * num_beams, cur_len): input_ids up to this point
        next_token_scores,                  # of shape (batch_size, n_tokens_to_keep): scores of next tokens
        next_tokens,                        # of shape (batch_size, n_tokens_to_keep): next_tokens (0-vocab_size for all batches)
        next_indices,                       # of shape (batch_size, n_tokens_to_keep): indices of next tokens (0-beam_size)
        pad_token_id=semantic_generator.tokenizer.pad_token_id,
        eos_token_id=semantic_generator.tokenizer.eos_token_id,
        beam_indices=semantic_beam_indices, # tuples of tuples (batch_size * num_beams, ?)
        decoder_prompt_len=semantic_decoder_prompt_len,
        other=next_semantic_tokens
    )
    # 1. update input_ids with beam_idx and beam_next_tokens
    # # ntodo just a debugging variable
    # old_semantic_beam_scores = semantic_beam_scores.clone()
    semantic_beam_scores = beam_outputs["next_beam_scores"]
    beam_next_tokens = beam_outputs["next_beam_tokens"]
    semantic_next_beam_indices = beam_outputs["next_beam_indices"] # of shape (batch_size * sem_beam_size,); per beam values [batch_idx*sem_beam_size, batch_idx*sem_beam_size+sem_beam_size)

    # add pure token scores to record
    pure_token_scores = semantic_generator.calc_next_pure_semantic_scores(
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
    # # ntodo just for me, remove in prod
    # sbi = semantic_generator.beam_indices_tuple_to_tensor(semantic_beam_indices)

    # get the source semantic hyps (tokens) and use their snytactic hyps 
    # for the next iteration input
    # # ntodo just for me, remove in prod
    # old_last_semantic_tokens = last_semantic_tokens if last_semantic_tokens is not None else None
    last_semantic_tokens = semantic_generator.filter_next_semantic_tokens(
        semantic_tokens_filled_hyps,
        semantic_next_beam_indices,
        beam_next_tokens,
        amount_semantic_beams,
        padding_token_id=semantic_generator.tokenizer.pad_token_id
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
        batch_beams = [last_semantic_tokens[batch_idx*amount_semantic_beams:(batch_idx+1)*amount_semantic_beams] for batch_idx in range(batch_size)]
        last_sem_toks = list(last_semantic_tokens)
        for batch_idx, batch in enumerate(batch_beams):
            if all([sem_tok is None for sem_tok in batch]):
                last_sem_toks[batch_idx] = first_non_empty
        last_semantic_tokens = tuple(last_sem_toks)
        del batch_beams, last_sem_toks, first_non_empty

    if all(beam_scorer._done):
        # ? do not compute next syntactic hyps, no need
        continue

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
    # # ntodo just for debugging
    # trans_scores = semantic_generator.compute_transition_scores(
    #     semantic_beam_indices,
    #     semantic_scores
    # )

    # use the last model output for the next iteration
    last_model_output = {
        "input_ids":  altered_input_ids,
        "attention_mask": altered_attention_mask
        }
    counter += 1

sequence_outputs = beam_scorer.finalize(
    semantic_inputs["input_ids"],
    semantic_beam_scores,
    next_tokens,
    next_indices,
    pad_token_id=semantic_generator.tokenizer.pad_token_id,
    eos_token_id=semantic_generator.tokenizer.eos_token_id,
    max_length=semantic_generation_config.max_length,
    beam_indices=semantic_beam_indices,
    decoder_prompt_len=decoder_prompt_len,
    other=next_semantic_tokens
)

final_semantic_sequences = sequence_outputs["sequences"]
final_semantic_sequences_scores = sequence_outputs["sequence_scores"]
final_semantic_scores = semantic_scores
final_semantic_beam_indices = sequence_outputs["beam_indices"]
final_semantic_tokens = sequence_outputs["other"]

final_transition_scores = semantic_generator.compute_transition_scores(
    final_semantic_beam_indices,
    final_semantic_scores
)
# add last transition score (the eos token, if applicable
last_transition_scores = torch.stack([sem_tok.score for sem_tok in final_semantic_tokens])
final_transition_scores = torch.cat((final_transition_scores, last_transition_scores), dim=-1)
# the transition scores summed at dim 1 and / (generated_len ** length penalty) equals to 
# the sequence scores

final_syntactic_sequences = torch.nn.utils.rnn.pad_sequence(
    [
        synt_hyp.syntactic_hypothesis.sequences 
        for sem_tok in final_semantic_tokens
        for synt_hyp in sem_tok.syntactic_hypotheses
    ],
    batch_first=True,
    padding_value=syntactic_generator.tokenizer.eos_token_id
)

print(semantic_generator.tokenizer.batch_decode(final_semantic_sequences))
print(syntactic_generator.tokenizer.batch_decode(final_syntactic_sequences, skip_special_tokens=True))

print(f"Final time: {time.time() - start_time:.2f}")
print("done")

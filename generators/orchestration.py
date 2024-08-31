from syntactic import SyntacticGenerator
from semantic import SemanticGenerator
from ner_model import NERUtilities
from generator import SemanticGenerationConfig
import torch
from utils import deep_compare
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer

# read access token from environment variable
import os
import time

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


#### Experiments setup ####
max_syntactic_tokens_per_iteration = 5
# one will 
amount_syntactic_beams = 20
total_max_tokens = 20
amount_semantic_beams = 3


# examples with batching and wo batching
example = "Obama was born"
examples = [example, "Abraham Lincoln was born in"]
# chose the example you want to test (singular or batched)
prompt = examples

# select the model you want to test
model_name = checkpoints[0]


#### 1. loading models ####
# syntactic generator
syntactic_generator = SyntacticGenerator(model_name, device, access_token)
# model = syntactic_generator.model
tokenizer = syntactic_generator.tokenizer

# semantic generator
semantic_generator = SemanticGenerator("dslim/distilbert-NER", device)

#### 2. prepare inputs and outputs ####
model_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

last_model_output = None
last_past_key_values = None
iter_output = None

semantic_output = None
last_beam_scores = None

# for generation
# initial semantic token extraction simply grabs all semantic tokens
initial_semantic_output = semantic_generator.generate(prompt)
initial_semantic_output = semantic_generator.merge_entities(initial_semantic_output)

semantic_inputs = semantic_generator.encode_semantic_sequences_from_semantic_data(initial_semantic_output)
# expand semantic inputs to match the amount of semantic beams
semantic_inputs["input_ids"] = semantic_generator.expand_semantic_sequences(semantic_inputs["input_ids"], amount_semantic_beams)
# # attention_mask is not really needed
# semantic_inputs["attention_mask"] = semantic_generator.expand_semantic_sequences(semantic_inputs["attention_mask"], amount_semantic_beams)
decoder_prompt_len = semantic_inputs["input_ids"].shape[-1]

# values necessary to be initialized
# general
batch_size = len(prompt)
num_beams = amount_syntactic_beams
# bs
semantic_batch_beam_size = batch_size * amount_semantic_beams
semantic_beam_indices = (
    tuple(() for _ in range(semantic_batch_beam_size))
)

semantic_generation_config = SemanticGenerationConfig(3)
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
# semantic_beam_scores = torch.zeros((batch_size, amount_semantic_beams), dtype=torch.float, device=device)
# semantic_beam_scores[:, 1:] = -1e9
# semantic_beam_scores = semantic_beam_scores.view((batch_size * amount_semantic_beams,))

# map syntactic hyps to semantic hyps
syn_to_sem_mapping = torch.zeros((batch_size, amount_syntactic_beams), dtype=torch.long, device=device)
for batch_idx, batch in enumerate(syn_to_sem_mapping):
    batch[:] = batch_idx * amount_semantic_beams

last_syntactic_hyps = None
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
    original_prompt_length=decoder_prompt_len,
    # last_scores = None if iter_output is None else iter_output.scores, # ? not used by default
    # length_penalty = 0,
    # # any sampling should be done with reproducibility = True
    # reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
    # do_sample = True,                         # if do_sample is True, use reproducibility = True
    # # use parameters at will
    # temperature = 0.2,                        # temperature for sampling
    # top_k = 50,                               # top_k for sampling
    )

    #### 4. run semantic model ####
    # prepare generation output for semantic model - batch_decode to get sequences in strings
    hyps_decoded = syntactic_generator.batch_decode(iter_output.sequences)
    # run semantic model -> List of (batch_size, )
    semantic_output = semantic_generator.generate(hyps_decoded)

    #### 5. find new semantic data ####
    # todo rework so this is generic semantic data, not just entities
    input_length, input_length_chars = syntactic_generator.get_input_length(
            inputs["input_ids"], iter_output.beam_indices
        )
    # output_length = syntactic_generator.get_output_length(iter_output.sequences)
    first_new_entities, new_entities = NERUtilities.get_generated_entities(
            semantic_output, input_length_chars
        )
    
    merged_new_entities = semantic_generator.merge_entities(new_entities)
    merged_first_new_entities = semantic_generator.merge_entities(first_new_entities)

    semantic_data = semantic_generator.get_semantic_data_from_first_entities(
        merged_first_new_entities
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
        syntactic_source_hyp
    )

    #### 8. semantic decoding ####
    semantic_tokens = semantic_generator.compute_semantic_tokens(
        shortened_hyps,
        amount_syntactic_beams
    )
    # group semantic token by source beam idx, expanding from list
    # of shape (batch_size * num_beams, num_tokens) to
    # (batch_size, num_beams, num_tokens)
    semantic_tokens = semantic_generator.gather_tokens_by_source_beam(
        semantic_tokens,
        batch_size,
        amount_semantic_beams
    )
    # if any of the the beams has no semantic tokens, duplicate one of the
    # other beams and set the score to -1e9
    # ? this (semantic_tokens_filled_hyps) is an interesting data point that could well be used to record the progress
    semantic_tokens_filled_hyps, semantic_beam_scores = semantic_generator.fill_empty_beam_hyps(
        semantic_tokens,
        semantic_beam_scores
    )

    # now as tensors
    next_tokens, next_token_scores = semantic_generator.gather_next_tokens(
        semantic_tokens_filled_hyps,
        device
    )
    next_tokens_debug = next_tokens.clone()
    next_token_scores_debug = next_token_scores.clone()
    # until here, the scores are just for the final token. Now, add beam scores
    # to them to get the final scores
    next_token_scores = next_token_scores + semantic_beam_scores[:, None].expand_as(
        next_token_scores
    )
    dynamic_vocab_size = next_token_scores.shape[-1]
    next_token_scores = next_token_scores.view((batch_size,amount_semantic_beams*dynamic_vocab_size))
    
    # prepare inputs for beam scorer
    # get the next_token_scores
    # 1. gather from semantic tokens
    # 2. add beam scores to them

    ## pass all the necessary arguments to the beam scorer
    # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
    # non eos token per beam allowing expansion if required at all time for all hyps.
    eos_token_id = semantic_generator.tokenizer.eos_token
    n_eos_tokens = torch.tensor([eos_token_id]).shape[0] if eos_token_id is not None else 0
    n_tokens_to_keep = max(2, 1 + n_eos_tokens) * amount_semantic_beams
    
    at_least_n_tokens = all(
        [
            len(hyp) >= n_tokens_to_keep for batch in semantic_tokens_filled_hyps for hyp in batch
        ]
    )
    if not at_least_n_tokens:
        # ! what to do in this case?
        # is simply using fewer tokens an option?
        raise ValueError("Not enough tokens to keep.")
    
    if semantic_generation_config.do_sample is True:
        # todo implement sampling
        raise NotImplementedError("Sampling not implemented yet.")
    else:
        # get the next n_tokens_to_keep tokens and indeces from the list
        next_token_scores, next_token_indices = torch.topk(
            next_token_scores, n_tokens_to_keep, dim=-1, largest=True, sorted=True
        )
    next_indices = torch.div(next_token_indices, dynamic_vocab_size, rounding_mode='floor')
    next_tokens = next_tokens.view((batch_size, amount_semantic_beams*dynamic_vocab_size))    
    next_tokens = next_tokens.gather(1, next_token_indices)

    beam_outputs = beam_scorer.process(
        semantic_inputs["input_ids"],   	# of shape (batch_size * num_beams, cur_len): input_ids up to this point
        next_token_scores,                  # of shape (batch_size, n_tokens_to_keep): scores of next tokens
        next_tokens,                        # of shape (batch_size, n_tokens_to_keep): next_tokens (0-vocab_size for all batches)
        next_indices,                       # of shape (batch_size, n_tokens_to_keep): indices of next tokens (0-beam_size)
        pad_token_id=semantic_generator.tokenizer.pad_token,
        eos_token_id=semantic_generator.tokenizer.eos_token,
        beam_indices=semantic_beam_indices, # tuples of tuples (batch_size * num_beams, ?)
        decoder_prompt_len=decoder_prompt_len,
    )
    # 1. update input_ids with beam_idx and beam_next_tokens
    semantic_beam_scores = beam_outputs["next_beam_scores"]
    beam_next_tokens = beam_outputs["next_beam_tokens"]
    semantic_next_beam_indices = beam_outputs["next_beam_indices"]

    semantic_inputs["input_ids"] = torch.cat([semantic_inputs["input_ids"][semantic_next_beam_indices, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
    # theoretically: udpate attention_mask as well, but we do not need it
    semantic_beam_indices = tuple((semantic_beam_indices[semantic_next_beam_indices[i]] + (semantic_next_beam_indices[i],) for i in range(len(semantic_beam_indices))))

    # 2. prepare inputs for next iteration
    #    - [x] store the old selected sem_hyps
    #    - [x] pick the selected sem_hyps
    #    - [ ] make sure sem_source_hyp is findable
    #    - [x] extract shortened syntactic hyps and use them for next input
    #    - [x] pad syntactic hyps to be valide new input (should actually already be the case)
    #    - [x] fill up syntactic hyps with some dummy hyp and use low scores for runnable next iteration
    #    - [ ] update syn_to_sem_mapping
    #    - [ ] take a look at "stopping criteria" in utils (need the same?)
    # get the source semantic hyps (tokens) and use their snytactic hyps 
    # for the next iteration input
    last_semantic_tokens = semantic_generator.filter_next_semantic_tokens(
        semantic_tokens_filled_hyps,
        semantic_next_beam_indices,
        beam_next_tokens,
        amount_semantic_beams
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

    # use the last model output for the next iteration
    last_model_output = {
        "input_ids":  altered_input_ids,
        "attention_mask": altered_attention_mask
        }

# todo look at beam_scorer.finalize

print("Syntactic output:")
print(tokenizer.batch_decode(iter_output.sequences, skip_special_tokens=True))
print("Semantic output:")
print(semantic_output)

print(f"Final time: {time.time() - start_time:.2f}")

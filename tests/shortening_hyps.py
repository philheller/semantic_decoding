from transformers.generation.utils import GenerateBeamDecoderOnlyOutput
import os
import sys
import torch
from score_differences.utils import report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../generators')))

from syntactic import SyntacticGenerator

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
    # "meta-llama/Meta-Llama-3-8B-Instruct",
    # "meta-llama/Meta-Llama-3-70B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "mistralai/Mistral-7B-v0.3",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
]

###############################################
########### Notes about Experiments ###########
###############################################
# This experiment is designed to address packing and unpacking
# hypotheses for continued decoding. The experiment works as follows:
# 0. Imports, setup, etc
# 1. Loading tokenizer and model
# 2. Preparing inputs and outputs
# 3. 1) and 2a) are the baselines (continued decoding with direct input output mapping)
#    2b) is the same as 2a) but with packed and unpacked hyps (thereby showing, that the
#    packed hyps are the same as the direct input output mapping)
#    3) shortens the hypotheses from the output of 2b) and continues decoding them
#    4) shortens the hypotheses from the output of 2b) and simulates additional padding
#    via masking and expansion. This simulates a hypothesis being shortened and expanded
#    due to batching. The scores of 3) and 4) are compared to show that the scores are
#    within expected differences, as they are in essence the same hypothesis.
# 4. Comparing and running tests
#
# DISCLAIMER: This experiment is not working for any and all sequences. It is designed 
# to work with a few conditions in mind. For that, assertions are made to ensure
# the informative value of the experiment.

#### Experiments setup ####
# the amount of tokens will also defined the amount of
# concatenated beam searches that will be performed which 
# is i = amount_of_tokens / 2 (16 tokens will be run through 8 runs)
amount_of_tokens = 20   # amount of tokens generated
amount_of_beams = 4     # amount of beams used for generation

# examples with batching and wo batching
example = "Obama was born"
examples = [example, "Michelle Obama was born"]
example = [example]
# chose the example you want to test (singular or batched)
prompt = example

# select the model you want to test
model_name = checkpoints[0]


#### 1. loading model ####
# loading tokenizer and model
syntactic_generator = SyntacticGenerator(model_name, device, access_token)
# model = syntactic_generator.model
tokenizer = syntactic_generator.tokenizer

print(f"Model {model_name} loaded successfully")


#### 2. prepare inputs and outputs ####
model_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
original_prompt_length = model_inputs["input_ids"].shape[-1]



#########################################
######## Experiment 1 
#########################################
# this is just to make sure continued decoding is fully functional
output_baseline = syntactic_generator.generate(
    **model_inputs,
    max_new_tokens=amount_of_tokens,
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
)

output_step_1 = syntactic_generator.generate(
    **model_inputs,
    max_new_tokens=int(amount_of_tokens / 2),
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)

input_step_2a = {
    "input_ids": output_step_1.sequences.clone(),
    "attention_mask": output_step_1.attention_mask.clone()
}

output_step_2a = syntactic_generator.generate(
    **input_step_2a,
    max_new_tokens=int(amount_of_tokens / 2),
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
    resume_generation = True,
    past_key_values = output_step_1.past_key_values,
    last_beam_scores = output_step_1.last_beam_scores,
    original_prompt_length = original_prompt_length,
    length_penalty = 1,
)

assert all(
    [
        torch.equal(
            output_baseline.sequences, output_step_2a.sequences
        ),
        torch.equal(
            output_baseline.sequences_scores, output_step_2a.sequences_scores
        ),
    ]
), "The sequences and sequence scores should be the same for the baseline and the concatenated beam search."
del output_baseline


# in this concrete example, all hyps in output_step_2a stem from hyp [1] in 
# output_step_1. Finding a singular source hyp is not always the case but 
# necessary for the experiment to be of value.
assert torch.all(
    output_step_2a.sequences[:, :13] == output_step_1.sequences[1]
), "For this experiment, the end hyps should all match the source hyp from output_step_1.\n\
    Have you changed the example? Either change it back or make sure to adjust the source hyp accordingly."

packed_hyps = syntactic_generator.pack_hypotheses(
    output_step_1.sequences,
    output_step_1.last_beam_scores,
    output_step_1.past_key_values,
    output_step_1.attention_mask,
    output_step_1.scores,
    output_step_1.beam_indices
)

# only use the source hyp from output_step_1, use it 4 times
packed_hyps_of_proper_source_hyp = packed_hyps[1:2] * amount_of_beams
unpacked_hyps, _ = syntactic_generator.unpack_hypotheses(packed_hyps_of_proper_source_hyp)

# need to adjust last_beam_scores to avoid sampling the same tokens (as they are duplicates)
unpacked_hyps["last_beam_scores"][1:] = -1e9	
input_step_2b = {
    "input_ids": unpacked_hyps["sequences"],
    "attention_mask": unpacked_hyps["attention_mask"]
}


output_step_2b = syntactic_generator.generate(
    **input_step_2b,
    max_new_tokens=int(amount_of_tokens / 2),
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
    resume_generation = True,
    past_key_values = unpacked_hyps["past_key_values"],
    last_beam_scores = unpacked_hyps["last_beam_scores"],
    original_prompt_length = original_prompt_length,
    length_penalty = 1,
)


# this is not necessarily true for every hyp, but necessary for the experiment to be of value
assert all(
    [
        torch.equal(
            output_step_2a.sequences, output_step_2b.sequences
        ),
        torch.equal(
            output_step_2a.sequences_scores, output_step_2b.sequences_scores
        )
    ]
), "The sequences and sequence scores should be the same for the concatenated beam search and the packed hypotheses."


# now that we have asserted, that 2b leads to the same result as 2a, 
# we can simulate shortening of the hyps and add masking to see, if 
# the results would be the same at the end.
# For that, use a hyp from the output of 2a (or b, they are the same)
# which is also a source hyp for all output beams (this could be a hyp with
# extra tokens y = 2-4 compared to the input of step 2b). 
# We do this twice here, once with the new length being inferred from the
# shortened hyps (3), once we force the original length being kept (4). 
# The latter is a good simulation of another length being forced onto the hyp
# due to batching.
# Ideally, the scores are as close as possible between 3 and 4 (reasons as to
# why they won't be the exact same can be found in the tests
# ./scores_differences in which both batching and masking are shown to have
# an impact on the scores for the same hypothesis). Expected differences lie
# in the range of 1e-5 to 1e-3 for the scores.

# step 1, pack hypotheses from output_step_2b
# step 2, shorten the hyps to input input_step_2b.sequences.__len__() + y [here: 2]
# step 3, unpack the shortened hyps
# step 4, continue decoding the shortened hyps

shortened_to_y_extra_tokens = 2
shorten_to_idx = input_step_2b["input_ids"].shape[-1] + shortened_to_y_extra_tokens

source_beam_indices_step_2b = syntactic_generator.compute_source_hypothesis_indices(
    beam_indices=output_step_2b.beam_indices
)
output_2b_hyps = syntactic_generator.pack_hypotheses(
    output_step_2b.sequences,
    output_step_2b.last_beam_scores,
    output_step_2b.past_key_values,
    output_step_2b.attention_mask,
    output_step_2b.scores,
    output_step_2b.beam_indices,
    source_hyps = packed_hyps_of_proper_source_hyp,
    source_beam_indices = source_beam_indices_step_2b
)
# since at point output_step_2b[:shorten_to_idx] all hyps are the same, we can simply do:
# todo move to _shorten_hyp_right_by_...
shortened_hyps = [
    syntactic_generator._shorten_hyp_right_to_token_idx(
        hyp,
        shorten_to_idx,
    ) for hyp in output_2b_hyps
]
unpacked_hyps_step_3, _ = syntactic_generator.unpack_hypotheses(shortened_hyps)

# need to adjust last beam scores to avoid sampling the same tokens (as they are duplicates)
unpacked_hyps_step_3["last_beam_scores"][1:] = -1e9
input_step_3 = {
    "input_ids": unpacked_hyps_step_3["sequences"],
    "attention_mask": unpacked_hyps_step_3["attention_mask"]
}

output_step_3 = syntactic_generator.generate(
    **input_step_3,
    max_new_tokens=int((amount_of_tokens / 2) - shortened_to_y_extra_tokens),
    renormalize_logits=True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores=True,
    output_logits=True,
    resume_generation=True,
    past_key_values=unpacked_hyps_step_3["past_key_values"],
    last_beam_scores=unpacked_hyps_step_3["last_beam_scores"],
    original_prompt_length=original_prompt_length,
    length_penalty=1,
)

assert all(
    [
        torch.equal(
            output_step_2b.sequences, output_step_3.sequences
        ),
        torch.equal(
            output_step_2b.sequences_scores, output_step_3.sequences_scores
        )
    ]
), "The results for the hyps shortened in packed hypotheses should match the results from step 2b."

# original length is kept (this could be anything, here we just arbitrarily set this length)
original_length = 22

# reuse the hyps from input 3
shortened_hyps_padded_and_masked = [
    syntactic_generator._expand_hyp_to_batch_length(
        hyp,
        original_length,
        pad_token_id=tokenizer.pad_token_id
    ) for hyp in shortened_hyps
]
shortened_input_padded_and_masked, _ = syntactic_generator.unpack_hypotheses(
    shortened_hyps_padded_and_masked
)

# need to adjust last beam scores to avoid sampling the same tokens (as they are duplicates)
shortened_input_padded_and_masked["last_beam_scores"][1:] = -1e9
input_step_4 = {
    "input_ids": shortened_input_padded_and_masked["sequences"],
    "attention_mask": shortened_input_padded_and_masked["attention_mask"]
}

output_step_4 = syntactic_generator.generate(
    **input_step_4,
    max_new_tokens=int((amount_of_tokens / 2) - shortened_to_y_extra_tokens),
    renormalize_logits=True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores=True,
    output_logits=True,
    resume_generation=True,
    past_key_values=shortened_input_padded_and_masked["past_key_values"],
    last_beam_scores=shortened_input_padded_and_masked["last_beam_scores"],
    original_prompt_length=original_prompt_length + 7,
    length_penalty=1,
)


print("Differences between hyp and simulation of shortened hyp:")
print("Scores")
print(
    *report(
        torch.stack(output_step_3.scores).exp(),
        torch.stack(output_step_4.scores).exp(),
        compare_top=True
    )
)

assert all([
    torch.allclose(
        torch.stack(output_step_3.scores).exp(),
        torch.stack(output_step_4.scores).exp(),
        atol=9e-4
    )
]), "From other tests, the score differences should lie below an error of 9e-4."

print(f"Final time: {time.time() - start_time:.2f}")

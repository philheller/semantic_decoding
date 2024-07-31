from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch

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
# device = "cpu" # if you want to use the cpu instead (also change the device_map in the model loading)

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
# This experiment is designed to explore logits (normalized and unnormalized),
# conditioned probabilities (which normalized logits along the vocab would be),
# scores (which are the same as the conditioned_probabilities but may include length penalty, therefore
# no longer necessarily adding up to one [hence the term score, no longer probs]),
# transition scores (which are recomputed scores from the logits or scores of generation time),
# and last but not least sequence scores (which are the sum of the scores (could be log probs) along the sequence).
# 
# The experiment is set up as follows:
# 0. Imports, setup, etc
# 1. Loading tokenizer and model
# 2. Preparing inputs and outputs
# 3. Running the model with a greedy search approach with and without normalization
# 4. Manually recomputing the transition scores from logits and scores
# 5. Running the model with a beam search approach with and without normalization
# 6. Manually recomputing the sequence scores from the logits and scores
# 
# RECOMMENDATION: Use a debugger to dynamically explore the variables, what they represent and experiment with them.
# 
# What we have taken away from it:
# - the scores in GS may be normalized or not, that depends on arguments passed to the model
# - the scores in BS seem to only be gotten unnormalized with logits
# - the sequence scores are the sum of the scores along the sequence (which can easily be recomputed)


#### Experiments setup ####
# the amount of tokens will also defined the amount of
# concatenated greedy searches that will be performed which 
# is i = amount_of_tokens / 2 (16 tokens will be run through 8 runs)
amount_of_tokens = 50   # amount of tokens generated
amount_of_beams = 3    # amount of beams used for generation

# examples with batching and wo batching
example = "Obama was born"
examples = [example, "Michelle Obama was born"]
# chose the example you want to test (singular or batched)
prompt = example

# select the model you want to test
model_name = checkpoints[0]



#### 1. loading model ####
# loading tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if tokenizer.pad_token is None:
    print(f"Setting pad token to eos token: {tokenizer.eos_token}")
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, token=access_token, 
    device_map="auto" # comment out for cpu (and change device to cpu)
).to(device)

print(f"Model {model_name} loaded successfully")


#### 2. prepare inputs and outputs ####
model_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

last_model_output = None
iter_output = None
output1 = None # is the first output of the model with the entire sequence

total_amount_of_steps = int(amount_of_tokens / 2)

# 3. Run greedy search approach with and without normalization
output_gr_unnormalized = model.generate(
    **model_inputs,
    max_new_tokens=amount_of_tokens,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits=True,
    # # any sampling should be done with reproducibility = True
    # reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
    # do_sample = True,                         # if do_sample is True, use reproducibility = True
    # # use parameters at will
    # temperature = 0.2,                        # temperature for sampling
    # top_k = 50,                               # top_k for sampling
)
output_gr_normalized = model.generate(
    **model_inputs,
    max_new_tokens=amount_of_tokens,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits=True,
    renormalize_logits = True,
    # # any sampling should be done with reproducibility = True
    # reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
    # do_sample = True,                         # if do_sample is True, use reproducibility = True
    # # use parameters at will
    # temperature = 0.2,                        # temperature for sampling
    # top_k = 50,                               # top_k for sampling
)

# stack tensor of 50 tuples to be of shape (50, x)
unnormalized_scores = torch.stack(output_gr_unnormalized.scores, dim=0)
normalized_scores = torch.stack(output_gr_normalized.scores, dim=0)

# log_softmax on last axis
unnormalized_scores_renormalized = torch.nn.functional.log_softmax(unnormalized_scores, dim=-1)
# reshape to make it from (50, x) to tuple of 50 (x)
unnormalized_scores_renormalized_in_output_shape = unnormalized_scores_renormalized.unbind(dim=0)

assert torch.equal(
    normalized_scores, unnormalized_scores_renormalized
), "The normalized scores should be equal to the renormalized unnormalized scores"

transition_scores_unnormalized = model.compute_transition_scores(
    output_gr_unnormalized.sequences, output_gr_unnormalized.scores, normalize_logits=True
)

transition_scores_normalized = model.compute_transition_scores(
    output_gr_normalized.sequences, output_gr_normalized.scores, normalize_logits=False
)

transition_scores_normalized_renormalized = model.compute_transition_scores(
    output_gr_normalized.sequences, output_gr_normalized.scores, normalize_logits=True
)

transition_scores_using_renormalized = model.compute_transition_scores(
    output_gr_unnormalized.sequences, unnormalized_scores_renormalized_in_output_shape, normalize_logits=False
)



print(f"Output not Normalized | Output Renormalized | Transition Untouched")
print(transition_scores_using_renormalized)
print(f"Output not Normalized | Output not Renormalized | Transition Normalized")
print(transition_scores_unnormalized)
print(f"Output Normalized | Output not Renormalized | Transition Untouched")
print(transition_scores_normalized)
print(f"Output Normalized | Output not Renormalized | Transition Normalized")
print(transition_scores_normalized_renormalized)


normalized_probs_sum = torch.exp(normalized_scores).sum(dim=-1)
manually_renormalized_probs_sum = torch.exp(unnormalized_scores_renormalized).sum(dim=-1)
expected_summed_probs = torch.ones_like(unnormalized_scores_renormalized)

# ? this is working, though float precision is requires lower precision
assert all(
    [
        torch.allclose(normalized_probs_sum, expected_summed_probs, atol=1e-3),
        torch.allclose(manually_renormalized_probs_sum, expected_summed_probs, atol=1e-3),
    ]
)

# -> output not normalized but renormalized is the same as output normalized
output_bs_unnormalized = model.generate(
    **model_inputs,
    max_new_tokens=amount_of_tokens,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits=True,
    renormalize_logits=False,                   # does not seem to have an effect
    # ! the renormalize_logits **may** have effects when logit warpers change the scores (does not seem to have an effect here)
    # length_penalty = 0,                       # ensures fair comparison
    # # any sampling should be done with reproducibility = True
    # reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
    # do_sample = True,                         # if do_sample is True, use reproducibility = True
    # # use parameters at will
    # temperature = 0.2,                        # temperature for sampling
    # top_k = 50,                               # top_k for sampling
)

output_bs_normalized = model.generate(
    **model_inputs,
    max_new_tokens=amount_of_tokens,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits=True,
    renormalize_logits = True,
    # length_penalty = 0,                       # ensures fair comparison
    # # any sampling should be done with reproducibility = True
    # reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
    # do_sample = True,                         # if do_sample is True, use reproducibility = True
    # # use parameters at will
    # temperature = 0.2,                        # temperature for sampling
    # top_k = 50,                               # top_k for sampling
)


# the simplest way to garantee same transition scores and sequence scores, is by always using the
# renormalize_logits = True flag in the generation method
# reproduce the sequences_score

transition_scores = model.compute_transition_scores(
    output_bs_normalized.sequences, output_bs_normalized.scores, output_bs_normalized.beam_indices, normalize_logits=False
)

# If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
# Tip 1: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
# use case, you might want to recompute it with `normalize_logits=True`.
# Tip 2: the output length does NOT include the input length
output_length = torch.sum(transition_scores < 0, dim=1)
length_penalty = model.generation_config.length_penalty
reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)


print(torch.allclose(
    reconstructed_scores, output_bs_normalized.sequences_scores
), "The reconstructed scores should be equal to the sequence scores")


print("Done.")

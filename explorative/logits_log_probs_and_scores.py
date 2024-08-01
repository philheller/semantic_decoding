
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
device = "cpu"

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
# - it is the easiest and most reproduceable to renormalize as with the argument in the model.generate function
#     - otherwise scores may not be reproduced so easily
#     - some of it is not properly documented in the docs
#     - @jao (maintainer of the .generate function) has admitted difficulties with namings and reproducibility
#     - see recommendation @link https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.renormalize_logits


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
    device_map="auto"
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


assert all(
    [
        torch.equal(
            output_gr_unnormalized.logits[x], output_gr_normalized.logits[x],
        ) for x in range(len(output_gr_unnormalized.logits))
    ]
), "The logits from the unnormalized and normalized greedy search should be equal."

assert all(
    [
        torch.equal(
            output_gr_unnormalized.scores[x], output_gr_unnormalized.logits[x]
        ) for x in range(len(output_gr_unnormalized.scores))
    ]
), "The scores from the unnormalized greedy search should be equal to it's logits."

# check out transition scores with native compute_transition_scores
# no normalization during generation, no normalization during transition score computation
transition_scores_gr_unnormalized_transition_untouched = model.compute_transition_scores(
    output_gr_unnormalized.sequences, output_gr_unnormalized.scores, normalize_logits=False
)
# no normalization during generation, but normalization during transition score computation
transition_scores_gr_unnormalized_transition_renormalized = model.compute_transition_scores(
    output_gr_unnormalized.sequences, output_gr_unnormalized.scores, normalize_logits=True
)
# normalization during generation, but not during transition score computation
transition_scores_gr_normalized_transition_untouched = model.compute_transition_scores(
    output_gr_normalized.sequences, output_gr_normalized.scores, normalize_logits=False
)

# this does not actually have to be true bc scores may be adapted with length penalty or logit warpers
assert torch.allclose(
    transition_scores_gr_unnormalized_transition_renormalized,
    transition_scores_gr_normalized_transition_untouched,
    atol=1e-5
), "The transition scores with normalization are only equal if the scores are not altered."

# recreate transition_scores_gr_unnormalized_transition_renormalized manually
# as seen in @link https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores
normalized_gr_scores_from_logits = tuple(
    torch.log_softmax(logits, dim=-1) for logits in output_gr_unnormalized.logits
)
normalized_gr_scores_from_scores = tuple(
    torch.log_softmax(scores, dim=-1) for scores in output_gr_unnormalized.scores
)
assert all(
    [
        torch.equal(normalized_gr_scores_from_logits[0], normalized_gr_scores_from_scores[0]),
        torch.equal(normalized_gr_scores_from_logits[-1], normalized_gr_scores_from_scores[-1])
    ]
), "The normalized scores from logits and scores should be equal."

assert all(
    [
        torch.equal(
            torch.log_softmax(output_gr_unnormalized.logits[0], dim=-1),
            torch.nn.functional.log_softmax(output_gr_unnormalized.scores[0], dim=-1)
        )
    ]
), "The torch.log_softmax and the torch.nn.functional.log_softmax should be equal."

normalized_gr_scores = normalized_gr_scores_from_scores

assert all(
    [
        torch.equal(
            normalized_gr_scores[0], output_gr_normalized.scores[0]
        ),
        torch.equal(
            normalized_gr_scores[-1], output_gr_normalized.scores[-1]
        )
    ]
), "The manually normalized scores do not match the natively normalized scores."

max_normalized_gr_scores = tuple(
    torch.max(scores, dim=-1).values for scores in normalized_gr_scores
)


# make transition_scores_recomputed_from_scratch as a tensor of shape sequence_length, 1 from tuple of len (sequence_length, beam_size)
transition_scores_gr_recomputed_from_scratch = torch.stack(
    # [max_normalized_gr_scores[0][0], max_normalized_gr_scores[1][0]]
    [
        max_normalized_gr_scores[x][0] for x in range(len(max_normalized_gr_scores))
    ]
    ).unsqueeze(0)

print("Final scores:")
print("Originally unnormalized, transition scores untouched")
print(transition_scores_gr_unnormalized_transition_untouched)
print("Recomputed from scratch")
print(transition_scores_gr_recomputed_from_scratch)
print("Originally unnormalized, transition scores renoramlized")
print(transition_scores_gr_unnormalized_transition_renormalized)
print("Originally normalized, transition scores untouched")
print(transition_scores_gr_normalized_transition_untouched)

assert all(
    [
        torch.allclose(
            transition_scores_gr_recomputed_from_scratch,
            transition_scores_gr_unnormalized_transition_renormalized,
            atol=1e-3
        ),
        torch.equal(
            transition_scores_gr_recomputed_from_scratch,
            transition_scores_gr_normalized_transition_untouched
        ),
    ]
), "The recomputed scores should be equal to the transition scores from the model."

print("The log probs correspond to:")
print(torch.exp(transition_scores_gr_recomputed_from_scratch))


input_length = model_inputs.input_ids.shape[1]
generated_tokens = output_gr_unnormalized.sequences[:, input_length:]
for tok, score in zip(generated_tokens[0], transition_scores_gr_normalized_transition_untouched[0]):
    # | token | token string | logits | probability
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")

print("Play around with the greedy transition scores here.")


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

# no normalization during generation, no normalization during transition score computation
transition_scores_bs_unnormalized_transition_untouched = model.compute_transition_scores(
    output_bs_unnormalized.sequences, output_bs_unnormalized.scores, output_bs_unnormalized.beam_indices, normalize_logits=False
)
# no normalization during generation, but normalization during transition score computation
transition_scores_bs_unnormalized_transition_renormalized = model.compute_transition_scores(
    output_bs_unnormalized.sequences, output_bs_unnormalized.scores, output_bs_unnormalized.beam_indices, normalize_logits=True
)
# normalization during generation, but not during transition score computation
transition_scores_bs_normalized_transition_untouched = model.compute_transition_scores(
    output_bs_normalized.sequences, output_bs_normalized.scores, output_bs_normalized.beam_indices, normalize_logits=False
)
# normalization during generation, and normalization during transition score computation
transition_scores_bs_normalized_transition_renormalized = model.compute_transition_scores(
    output_bs_normalized.sequences, output_bs_normalized.scores, output_bs_normalized.beam_indices, normalize_logits=True
)
print("Play around to look into the transition scores here")

assert all(
    [
        torch.allclose(
            transition_scores_bs_unnormalized_transition_untouched,
            transition_scores_bs_unnormalized_transition_renormalized,
            atol=1e-5
        ),
        torch.allclose(
            transition_scores_bs_unnormalized_transition_untouched,
            transition_scores_bs_normalized_transition_untouched,
            atol=1e-5
        ),
        torch.allclose(
            transition_scores_bs_unnormalized_transition_untouched,
            transition_scores_bs_normalized_transition_renormalized,
            atol=1e-5
        ),
    ]
), "The transition scores should be equal."
# continue with one variable since they are all the same:
transition_scores_bs = transition_scores_bs_unnormalized_transition_renormalized

print("The log probs correspond to:")
print(torch.exp(transition_scores_bs))

# now to recompute the sequences scores (see @link https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores)
# If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
# Tip 1: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
# use case, you might want to recompute it with `normalize_logits=True`.
# Tip 2: the output length does NOT include the input length
output_length = torch.sum(transition_scores_bs < 0, axis=1)
length_penalty = model.generation_config.length_penalty

reconstructed_scores = torch.sum(transition_scores_bs, axis=1) / (output_length**length_penalty)

assert torch.allclose(output_bs_unnormalized.sequences_scores, reconstructed_scores, atol=1e-9), "The recomputed scores should be equal to the sequences scores from the model."

print("Here are the sequence scores")
print(output_bs_unnormalized.sequences_scores, torch.exp(output_bs_unnormalized.sequences_scores))
print(reconstructed_scores, torch.exp(reconstructed_scores))
print(f"Final time: {time.time() - start_time:.2f}")
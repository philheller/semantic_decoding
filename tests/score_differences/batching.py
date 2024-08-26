import os
import sys
import torch
from utils import report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../generators')))

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
# device = "cpu" # comment in and out to quickly switch between cpu and gpu

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
# This experiment is here to show differences in logits 
# (and ultimately scores) that occur due to batching.
# For that, the simple forward pass and generate 
# with greedy search and beam search are compared.
# 0. Imports, setup, etc
# 1. Loading tokenizer and model
# 2. Preparing inputs and outputs
# 3. a) direct forward pass
#   - single sequence
#   - multiple sequences
# 3. b) generate with greedy search
#   - single sequence
#   - multiple sequences
# 3. c) generate with beam search
#   - single sequence
#   - multiple sequences (though not really necessary,
#    as beam search is like greedy search from a batching
#    perspective)
# 4. Comparing and running tests

#### Experiments setup ####
amount_of_tokens = 2    # amount of tokens generated
amount_of_beams = 4     # amount of beams used for generation

# examples with batching and wo batching
example = "Obama was born"
prompt = [example]

# select the model you want to test
model_name = checkpoints[0]


#### 1. loading model ####
# loading tokenizer and model
syntactic_generator = SyntacticGenerator(model_name, device, access_token)
model = syntactic_generator.model
tokenizer = syntactic_generator.tokenizer

print(f"Model {model_name} loaded successfully")


#### 2. prepare inputs and outputs ####
model_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
model_inputs_batched = tokenizer(prompt * 4, return_tensors="pt", padding=True).to(device)
position_ids = torch.arange(0, model_inputs["input_ids"].shape[-1]).unsqueeze(0).to(device)
original_input_length = model_inputs["input_ids"].shape[-1]

#### 1. Experiments ####
# a) direct forward pass with single sequence
out_forward_single_sequence = model(**model_inputs, position_ids=position_ids)
# a) direct forward pass with multiple sequences
out_forward_multiple_sequences = model(**model_inputs_batched, position_ids=None)


# b) generate with greedy search with single sequence
out_greedy_single_sequence = syntactic_generator.generate(
    **model_inputs,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)
# b) generate with greedy search with multiple sequences
out_greedy_multiple_sequences = syntactic_generator.generate(
    **model_inputs_batched,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)

# c) generate with beam search with single sequence
out_bs_single_sequence = syntactic_generator.generate(
    **model_inputs,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)
# c) generate with beam search with multiple sequences
out_bs_multiple_sequences = syntactic_generator.generate(
    **model_inputs_batched,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)

print("\n\n", "\t\tResults".upper())
print("#" * 40)
print("1. Compare the processes against itself unbatched vs batched:")
print("a) Direct forward pass")

print("Logits")
print(
    *report(
        out_forward_single_sequence.logits[:, -1:, :],
        out_forward_multiple_sequences.logits[:1, -1:, :]
    )
)
print("Scores")
print(
    *report(
        out_forward_single_sequence.logits[:, -1:, :].log_softmax(dim=-1).exp(),
        out_forward_multiple_sequences.logits[:1, -1:, :].log_softmax(dim=-1).exp(),
        compare_top=True
    )
)


print("\nb) Generate with greedy search")

print("Logits")
print(
    *report(
        out_greedy_single_sequence.logits[0],
        out_greedy_multiple_sequences.logits[0][:1, :]
    )
)
print("Scores") # .scores is same as .logits from above with .log_softmax(dim=-1)
print(
    *report(
        out_greedy_single_sequence.scores[0].exp(),
        out_greedy_multiple_sequences.scores[0][:1, :].exp(),
        compare_top=True
    )
)


print("\nc) Generate with beam search")
print("Logits")
print(
    *report(
        out_bs_single_sequence.logits[0],
        out_bs_multiple_sequences.logits[0][:4, :]
    )
)
print("Scores")
print(
    *report(
        out_bs_single_sequence.scores[0].exp(),
        out_bs_multiple_sequences.scores[0][:4, :].exp()
    )
)

print()
print("Are the single sequences the same regardless of generation type?")
print("Forward vs Greedy?\t")
print(
    *report(
        out_forward_single_sequence.logits[:, -1, :],
        out_greedy_single_sequence.logits[0]
    )
)
print("Greedy vs Beam?\t\t")
print(
    *report(
        out_greedy_single_sequence.logits[0],
        out_bs_single_sequence.logits[0][:1, :]	
    )
)

print()
print("Are the batched sequences the same regardless of generation type?")
print("Forward vs Greedy?\t")
print(
    *report(
        out_forward_multiple_sequences.logits[:1, -1, :],
        out_greedy_multiple_sequences.logits[0][:1, :]
    )
)
print("Greedy vs Beam?\t\t")
print(
    *report(
        out_greedy_multiple_sequences.logits[0][:1, :],
        out_bs_multiple_sequences.logits[0][:1, :]
    )
)

print("Done")
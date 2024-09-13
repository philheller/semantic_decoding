import os
import sys
import torch
from utils import report, compare_top_k

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
# (and ultimately scores) that occur due to 
# batching and masking.
# For that, the simple forward pass and generate 
# with greedy search and beam search are compared.
# 0. Imports, setup, etc
# 1. Loading tokenizer and model
# 2. Preparing inputs and outputs
# 3. a) direct forward pass
#       - no masking, no batching
#       - masking and batching
# 3. b) generate with greedy search
#       - no masking, no batching
#       - masking and batching
# 3. c) generate with beam search
#       - no masking, no batching
#       - masking and batching
# 4. Comparing and running tests
# 
# masking will occur in the following size:
# - no masking
# - 10 token masked
# batching is done with 4 sequences (always the same)

#### Experiments setup ####
amount_of_tokens = 2    # amount of tokens generated
long_tokens = 90         # amount of tokens generated in second setup
amount_of_beams = 4     # amount of beams used for generation

# examples with batching and wo batching
example = ["One of the greatest things is that"]
example_1_masked = example + [example[0] + " fill up mask to 5"]
example_10_masked = example + [example[0] + " these are words filling up to a mask of 10"]
# chose the example you want to test (singular or batched)


# select the model you want to test
model_name = checkpoints[0]


#### 1. loading model ####
# loading tokenizer and model
syntactic_generator = SyntacticGenerator(model_name, device, access_token)
model = syntactic_generator.model
tokenizer = syntactic_generator.tokenizer

print(f"Model {model_name} loaded successfully")


#### 2. prepare inputs and outputs ####
model_inputs = tokenizer(example, return_tensors="pt", padding=True).to(device)
model_inputs_1_masked = tokenizer(example_1_masked, return_tensors="pt", padding=True).to(device)
model_inputs_10_masked = tokenizer(example_10_masked, return_tensors="pt", padding=True).to(device)
original_input_length = model_inputs["input_ids"].shape[-1]
original_input_length_1_masked = model_inputs_1_masked["input_ids"].shape[-1]
original_input_length_10_masked = model_inputs_10_masked["input_ids"].shape[-1]
assert all(
    [
        original_input_length_10_masked - 10 == original_input_length,
        original_input_length_1_masked - 5 == original_input_length,
    ]
), "Mask length is not as expected"

model_inputs["input_ids"] = model_inputs["input_ids"][:1]
model_inputs["attention_mask"] = model_inputs["attention_mask"][:1]
# use the same sentence multiple times (batching) with mask
model_inputs_1_masked["input_ids"] = model_inputs_1_masked["input_ids"][:1]#.repeat(4, 1)
model_inputs_1_masked["attention_mask"] = model_inputs_1_masked["attention_mask"][:1]#.repeat(4, 1)
model_inputs_10_masked["input_ids"] = model_inputs_10_masked["input_ids"][:1].repeat(4, 1)
model_inputs_10_masked["attention_mask"] = model_inputs_10_masked["attention_mask"][:1].repeat(4, 1)


#### 3. Run experiment ####
# a) direct forward pass
out_forward_no_mask = model(**model_inputs)
out_forward_10_masked = model(**model_inputs_10_masked)

# b) generate with greedy search
out_greedy_no_mask = syntactic_generator.generate(
    **model_inputs,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)
out_greedy_10_masked = syntactic_generator.generate(
    **model_inputs_10_masked,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)

# c) generate with beam search
out_bs_no_mask = syntactic_generator.generate(
    **model_inputs,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)
out_bs_10_masked = syntactic_generator.generate(
    **model_inputs_10_masked,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)

#### 4. Run tests ####
print("\n\n", "\t\tResults".upper())
print("#" * 40)
print()
print("a) Direct forward pass")

print("No masking, no batching vs 10 token masked, batched (4 times)")
print("Logits")
print(
    *report(
        out_forward_no_mask.logits[:, -1:, :],
        out_forward_10_masked.logits[0, -1:, :]
    )
)
print("Scores")
print(
    *report(
        out_forward_no_mask.logits[:, -1:, :].log_softmax(dim=-1).exp(),
        out_forward_10_masked.logits[0, -1:, :].log_softmax(dim=-1).exp(),
        compare_top = True
        )
)
print()
print("b) Generate with greedy search")

print("Logits")
print(
    *report(
        out_greedy_no_mask.logits[0],
        out_greedy_10_masked.logits[0][:1, :]
    )
)
print("Scores")
print(
    *report(
        out_greedy_no_mask.scores[0].exp(),
        out_greedy_10_masked.scores[0][:1, :].exp(),
        compare_top = True
    )
)

print()
print("c) Generate with beam search")
print("Logits")
print(
    *report(
        out_bs_no_mask.logits[0][:1, :],
        out_bs_10_masked.logits[0][:1, :]
    )
)

print("Scores")
print(
    *report(
        out_bs_no_mask.scores[0][:1, :].exp(),
        out_bs_10_masked.scores[0][:1, :].exp(),
        compare_top = True
    )
)


# now over a larger amount of tokens
# b) generate with greedy search
out_greedy_no_mask_long = syntactic_generator.generate(
    **model_inputs,
    max_new_tokens=int(long_tokens),
    renormalize_logits=True,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)
out_greedy_10_masked_long = syntactic_generator.generate(
    **model_inputs_10_masked,
    max_new_tokens=int(long_tokens),
    renormalize_logits=True,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)

# c) generate with beam search
out_bs_no_mask_long = syntactic_generator.generate(
    **model_inputs,
    max_new_tokens=int(long_tokens),
    renormalize_logits=True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)
out_bs_10_masked_long = syntactic_generator.generate(
    **model_inputs_10_masked,
    max_new_tokens=int(long_tokens),
    renormalize_logits=True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)


print()
print("b) Generate with greedy search (long)")

print("Logits")
print(
    *report(
        out_greedy_no_mask_long.logits[-1],
        out_greedy_10_masked_long.logits[-1]
    )
)
print("Scores")
print(
    *report(
        out_greedy_no_mask_long.scores[-1].exp(),
        out_greedy_10_masked_long.scores[-1].exp(),
        compare_top = True
    )
)

print()
print("c) Generate with beam search (long)")
print("Logits")
print(
    *report(
        torch.stack(out_bs_no_mask_long.logits)[:, :4],
        torch.stack(out_bs_10_masked_long.logits)[:, :4],
    )
)
print("Scores (first in log score and second exponentiated)")
print(
    *report(
        torch.stack(out_bs_no_mask_long.scores)[:, :4],
        torch.stack(out_bs_10_masked_long.scores)[:, :4],
        compare_top = True
    )
)
print(
    *report(
        torch.stack(out_bs_no_mask_long.scores)[:, :4].exp(),
        torch.stack(out_bs_10_masked_long.scores)[:, :4].exp(),
        compare_top = True
    )
)

# b) generate with greedy search
out_greedy_1_masked = syntactic_generator.generate(
    **model_inputs_1_masked,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)
out_greedy_1_masked = syntactic_generator.generate(
    **model_inputs_1_masked,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)
# c) generate with beam search
out_bs_1_masked = syntactic_generator.generate(
    **model_inputs_1_masked,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)
out_bs_1_masked = syntactic_generator.generate(
    **model_inputs_1_masked,
    max_new_tokens=int(amount_of_tokens),
    renormalize_logits=True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits = True,
)

print()
print("Differences in beams")
print("Would the beams have been the same for 1, 2, 3, 4, 5, 8, 10, 50 beams (No masking vs 1 masked & no batching)?")
indices_same_1, _, _ = compare_top_k(
        torch.stack(out_bs_no_mask.scores)[:, :4],
        torch.stack(out_bs_1_masked.scores)[:, :4],
        1,
        -1
)
indices_same_2, _, _ = compare_top_k(
        torch.stack(out_bs_no_mask.scores)[:, :4],
        torch.stack(out_bs_1_masked.scores)[:, :4],
        2,
        -1
)
indices_same_3, _, _ = compare_top_k(
        torch.stack(out_bs_no_mask.scores)[:, :4],
        torch.stack(out_bs_1_masked.scores)[:, :4],
        3,
        -1
)
indices_same_4, _, _ = compare_top_k(
        torch.stack(out_bs_no_mask.scores)[:, :4],
        torch.stack(out_bs_1_masked.scores)[:, :4],
        4,
        -1
)
indices_same_5, _, _ = compare_top_k(
        torch.stack(out_bs_no_mask.scores)[:, :4],
        torch.stack(out_bs_1_masked.scores)[:, :4],
        5,
        -1
)
indices_same_8, _, _ = compare_top_k(
    torch.stack(out_bs_no_mask.scores)[:, :4],
    torch.stack(out_bs_1_masked.scores)[:, :4],
    8,
    -1
)
indices_same_5, _, _ = compare_top_k(
    torch.stack(out_bs_no_mask.scores)[:, :4],
    torch.stack(out_bs_1_masked.scores)[:, :4],
    10,
    -1
)
indices_same_50, _, _ = compare_top_k(
    torch.stack(out_bs_no_mask.scores)[:, :4],
    torch.stack(out_bs_1_masked.scores)[:, :4],
    50,
    -1
)
print(f"{'1🌿':^5} {'2🌿':^5} {'3🌿':^5} {'4🌿':^5} {'5🌿':^5} {'8🌿':^5} {'10🌿':^5} {'50🌿':^5}")
print(f"{indices_same_1:^5} {indices_same_2:^5} {indices_same_3:^5} {indices_same_4:^5} {indices_same_5:^5} {indices_same_8:^5} {indices_same_5:^5} {indices_same_50:^5}")
print("Would the beams have been the same for 1, 2, 3, 4, 5, 8, 10, 50 beams (No masking vs 10 masked & batching)?")
indices_same_1, _, _ = compare_top_k(
        torch.stack(out_bs_no_mask.scores)[:, :4],
        torch.stack(out_bs_10_masked.scores)[:, :4],
        1,
        -1
)
indices_same_2, _, _ = compare_top_k(
        torch.stack(out_bs_no_mask.scores)[:, :4],
        torch.stack(out_bs_10_masked.scores)[:, :4],
        2,
        -1
)
indices_same_3, _, _ = compare_top_k(
        torch.stack(out_bs_no_mask.scores)[:, :4],
        torch.stack(out_bs_10_masked.scores)[:, :4],
        3,
        -1
)
indices_same_4, _, _ = compare_top_k(
        torch.stack(out_bs_no_mask.scores)[:, :4],
        torch.stack(out_bs_10_masked.scores)[:, :4],
        4,
        -1
)
indices_same_5, _, _ = compare_top_k(
        torch.stack(out_bs_no_mask.scores)[:, :4],
        torch.stack(out_bs_10_masked.scores)[:, :4],
        5,
        -1
)
indices_same_8, _, _ = compare_top_k(
    torch.stack(out_bs_no_mask.scores)[:, :4],
    torch.stack(out_bs_10_masked.scores)[:, :4],
    8,
    -1
)
indices_same_10, _, _ = compare_top_k(
    torch.stack(out_bs_no_mask.scores)[:, :4],
    torch.stack(out_bs_10_masked.scores)[:, :4],
    10,
    -1
)
indices_same_50, _, _ = compare_top_k(
    torch.stack(out_bs_no_mask.scores)[:, :4],
    torch.stack(out_bs_10_masked.scores)[:, :4],
    50,
    -1
)
print(f"{'1🌿':^5} {'2🌿':^5} {'3🌿':^5} {'4🌿':^5} {'5🌿':^5} {'8🌿':^5} {'10🌿':^5} {'50🌿':^5}")
print(f"{indices_same_1:^5} {indices_same_2:^5} {indices_same_3:^5} {indices_same_4:^5} {indices_same_5:^5} {indices_same_8:^5} {indices_same_10:^5} {indices_same_50:^5}")
print("Done")
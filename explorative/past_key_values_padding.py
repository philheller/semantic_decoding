from transformers.generation.utils import GenerateBeamDecoderOnlyOutput
import os
import sys
import torch

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
# This is explorative work to understand the behavior of past_key_values.
# Key findings: the first tokens in the past_key_values are the same
# if it is padding which is masked out. They are however only the same
# within the same sequence. Another sequence with padding up front will
# have different past_key_values.
# There is no easy way to (re-)produce a past_key_value for a masked token up front.
# 0. Imports, setup, etc
# 1. Loading tokenizer and model
# 2. Preparing inputs and outputs
# 3. Experiments 
# 4. Comparing and running tests


#### Experiments setup ####
# the amount of tokens will also defined the amount of
# concatenated beam searches that will be performed which 
# is i = amount_of_tokens / 2 (16 tokens will be run through 8 runs)
amount_of_tokens = 20   # amount of tokens generated
amount_of_beams = 4     # amount of beams used for generation

# examples with batching and wo batching
example = "Obama was born"
examples = [
    example,
    "Michelle Obama was born in a city which is known to be in",
    "Micheal Jordan is one of the most famous basketball players in the world"
]
example = [example]
# chose the example you want to test (singular or batched)
prompt_1 = examples[:2]
prompt_2 = [examples[0], examples[2]]

# select the model you want to test
model_name = checkpoints[0]


#### 1. loading model ####
# loading tokenizer and model
syntactic_generator = SyntacticGenerator(model_name, device, access_token)
model = syntactic_generator.model
tokenizer = syntactic_generator.tokenizer

print(f"Model {model_name} loaded successfully")


#### 2. prepare inputs and outputs ####
model_inputs_1 = tokenizer(prompt_1, return_tensors="pt", padding=True).to(device)
model_inputs_2 = tokenizer(prompt_2, return_tensors="pt", padding=True).to(device)
original_prompt_length = model_inputs_1["input_ids"].shape[-1]
original_prompt_length_2 = model_inputs_2["input_ids"].shape[-1]

model_inputs_one_sequence_1 = {
    "input_ids": model_inputs_1["input_ids"][:4],
    "attention_mask": model_inputs_1["attention_mask"][:4],
}
model_input_one_sequence_2 = {
    "input_ids": model_inputs_2["input_ids"][:4],
    "attention_mask": model_inputs_2["attention_mask"][:4],
}

# annotation:
# 0)  only one sequence
# 0a) baseline
# 0b) same as 0a; see if the same pkv for the same padding is produced (it is)
# 0c) same sequence, different amount of padding (and thus the pkv tokens of the padding is different)
# 
# 1)  two sequences
# 1a) baseline
# 1b) same as 1a; see if the same pkv for the same padding is produced (it is)
# 1c) also two sequences, but the second sequence has a different amount of padding
# 
# 2)  two sequences, simple forward pass
# 
#########################################
######## Experiment 1 
#########################################
output_0_a = syntactic_generator.generate(
    **model_inputs_one_sequence_1,
    max_new_tokens=amount_of_tokens,
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits=True,
)

output_0_b = syntactic_generator.generate(
    **model_inputs_one_sequence_1,
    max_new_tokens=amount_of_tokens,
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits=True,
)

output_0_c = syntactic_generator.generate(
    **model_input_one_sequence_2,
    max_new_tokens=amount_of_tokens,
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits=True,
)

output_1_a = syntactic_generator.generate(
    **model_inputs_1,
    max_new_tokens=amount_of_tokens,
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits=True,
)

output_1_b = syntactic_generator.generate(
    **model_inputs_1,
    max_new_tokens=amount_of_tokens,
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits=True,
)

output_1_c = syntactic_generator.generate(
    **model_inputs_2,
    max_new_tokens=amount_of_tokens,
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    output_logits=True,
)

output_2 = model(
    input_ids = model_inputs_1["input_ids"],
    attention_mask=model_inputs_1["attention_mask"]
)


pkv_0_a = output_0_a.past_key_values
pkv_0_a = syntactic_generator._stack_past_key_values(pkv_0_a)

pkv_0_b = output_0_b.past_key_values
pkv_0_b = syntactic_generator._stack_past_key_values(pkv_0_b)

pkv_0_c = output_0_c.past_key_values
pkv_0_c = syntactic_generator._stack_past_key_values(pkv_0_c)

pkv_1_a = output_1_a.past_key_values
pkv_1_a = syntactic_generator._stack_past_key_values(pkv_1_a)

pkv_1_b = output_1_b.past_key_values
pkv_1_b = syntactic_generator._stack_past_key_values(pkv_1_b)

pkv_1_c = output_2.past_key_values
pkv_1_c = syntactic_generator._stack_past_key_values(pkv_1_c)

pkv_2 = output_2.past_key_values
pkv_2 = syntactic_generator._stack_past_key_values(pkv_2)

print("Done")
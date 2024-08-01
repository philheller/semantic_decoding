from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput
from utils import report_output
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


#### Experiments setup ####
# the amount of tokens will also defined the amount of
# concatenated beam searches that will be performed which 
# is i = amount_of_tokens / 2 (16 tokens will be run through 8 runs)
amount_of_tokens = 10   # amount of tokens generated
amount_of_beams = 3     # amount of beams used for generation

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

total_amount_of_steps = int(amount_of_tokens / 2)

#### 3. run models ####
### decoded one iteration
output_1 = model.generate(
**model_inputs,
max_new_tokens=int(amount_of_tokens / 2),
renormalize_logits = True,
num_beams=amount_of_beams,
num_return_sequences=amount_of_beams,
return_dict_in_generate=True,
output_scores = True,
length_penalty = 0,                       # ensures fair comparison
# # any sampling should be done with reproducibility = True
# reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
# do_sample = True,                         # if do_sample is True, use reproducibility = True
# # use parameters at will
# temperature = 0.2,                        # temperature for sampling
# top_k = 50,                               # top_k for sampling
)

### decoded piece by piece
inputs = {
    "input_ids":  output_1.sequences,
    "attention_mask": output_1.attention_mask
}

# continue decoding without changes
output_2 = model.generate(
**inputs,
max_new_tokens=int(amount_of_tokens / 2),
renormalize_logits = True,
num_beams=amount_of_beams,
num_return_sequences=amount_of_beams,
return_dict_in_generate=True,
output_scores = True,
resume_generation = True,
# past_key_values = output_1.past_key_values,
# last_beam_scores = output_1.last_beam_scores, # should be same as sequences_scores if length_penalty = 0
# last_scores = output_1.scores,
# length_penalty = 0,                       # ensures fair comparison
# # any sampling should be done with reproducibility = True
# reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
# do_sample = True,                         # if do_sample is True, use reproducibility = True
# # use parameters at will
# temperature = 0.2,                        # temperature for sampling
# top_k = 50,                               # top_k for sampling
)

# change the input to have an even attention mask and compare to output_2
# ✅ this experiment is successful 
# the same inputs but with an attention mask masking out even left padding works
inputs_2_5 = {
    key: value.clone() for key, value in inputs.items()
}
zeroes = torch.zeros((amount_of_beams, 2), dtype=torch.long).to(device)
# add two zeros to first token of every beam
inputs_2_5["input_ids"] = torch.cat((zeroes, inputs["input_ids"]), dim=-1)
# adapt ateention mask accordingly
inputs_2_5["attention_mask"] = torch.cat((zeroes, inputs["attention_mask"]), dim=-1)


output_2_5 = model.generate(
**inputs_2_5,
max_new_tokens=int(amount_of_tokens / 2),
renormalize_logits = True,
num_beams=amount_of_beams,
num_return_sequences=amount_of_beams,
return_dict_in_generate=True,
output_scores = True,
resume_generation = True,
# past_key_values = output_1.past_key_values,
# last_beam_scores = output_1.last_beam_scores, # should be same as sequences_scores if length_penalty = 0
# last_scores = output_1.scores,
# length_penalty = 0,                       # ensures fair comparison
# # any sampling should be done with reproducibility = True
# reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
# do_sample = True,                         # if do_sample is True, use reproducibility = True
# # use parameters at will
# temperature = 0.2,                        # temperature for sampling
# top_k = 50,                               # top_k for sampling
)

assert torch.equal(
    output_2.sequences, output_2_5.sequences[:,2:]
), "Same sequences with attention mask adapted do not match but should match."


# same as 2_5 but different order of inputs
# ✅ this experiment is successful 
# the same inputs but with an attention mask evenly masking out left padding; works with another order as well
input_2_6 = {
    key: value.clone() for key, value in inputs_2_5.items()
}

# reorder beams by 1
input_2_6["input_ids"] = input_2_6["input_ids"][[1, 0, 2]]

output_2_6 = model.generate(
**inputs_2_5,
max_new_tokens=int(amount_of_tokens / 2),
renormalize_logits = True,
num_beams=amount_of_beams,
num_return_sequences=amount_of_beams,
return_dict_in_generate=True,
output_scores = True,
resume_generation = True,
# past_key_values = output_1.past_key_values,
# last_beam_scores = output_1.last_beam_scores, # should be same as sequences_scores if length_penalty = 0
# last_scores = output_1.scores,
# length_penalty = 0,                       # ensures fair comparison
# # any sampling should be done with reproducibility = True
# reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
# do_sample = True,                         # if do_sample is True, use reproducibility = True
# # use parameters at will
# temperature = 0.2,                        # temperature for sampling
# top_k = 50,                               # top_k for sampling
)

assert torch.equal(
    output_2_5.sequences, output_2_6.sequences
), "Same sequences with different order of beams do not match but should match."

altered_input = {key: value.clone() for key, value in inputs.items()}
first_input_seq = inputs["input_ids"][2]
last_input_seq = inputs["input_ids"][0]
altered_first_input_seq = torch.cat((torch.tensor([0]).to(device), first_input_seq[:-1]))
altered_last_input_seq = torch.cat((torch.tensor([0, 0]).to(device), last_input_seq[:-2]))
altered_first_mask = inputs["attention_mask"][2]
altered_last_mask = inputs["attention_mask"][0]
altered_first_mask = torch.cat((torch.tensor([0]).to(device), altered_first_mask[:-1]))
altered_last_mask = torch.cat((torch.tensor([0, 0]).to(device), altered_last_mask[:-2]))

altered_input["input_ids"][2] = altered_first_input_seq
altered_input["attention_mask"][2] = altered_first_mask
altered_input["input_ids"][0] = altered_last_input_seq
altered_input["attention_mask"][0] = altered_last_mask
print("Input before and after alteration")
print(inputs)
print(altered_input)


# ✅ this experiment is successful 
output_3 = model.generate(
**altered_input,
max_new_tokens=int(amount_of_tokens / 2),
renormalize_logits = True,
num_beams=amount_of_beams,
num_return_sequences=amount_of_beams,
return_dict_in_generate=True,
output_scores = True,
resume_generation = True,
past_key_values = None,
# last_beam_scores = , # should be same as sequences_scores if length_penalty = 0
# last_scores = None if iter_output is None else iter_output.scores,
# length_penalty = 0,                       # ensures fair comparison
# # any sampling should be done with reproducibility = True
# reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
# do_sample = True,                         # if do_sample is True, use reproducibility = True
# # use parameters at will
# temperature = 0.2,                        # temperature for sampling
# top_k = 50,                               # top_k for sampling
)

# compute transition scores
transition_scores_3 = model.compute_transition_scores(
    output_3.sequences, output_3.scores, output_3.beam_indices, normalize_logits=False
)

altered_order_3_inputs = {key: value.clone() for key, value in altered_input.items()}
altered_order_3_inputs["input_ids"] = altered_order_3_inputs["input_ids"][[1, 0, 2]]
altered_order_3_inputs["attention_mask"] = altered_order_3_inputs["attention_mask"][[1, 0, 2]]
# just for testing purposes
output_3_5 = model.generate(
**altered_order_3_inputs,
max_new_tokens=int(amount_of_tokens / 2),
renormalize_logits = True,
num_beams=amount_of_beams,
num_return_sequences=amount_of_beams,
return_dict_in_generate=True,
output_scores = True,
resume_generation = True,
past_key_values = None,
# last_beam_scores = , # should be same as sequences_scores if length_penalty = 0
# last_scores = None if iter_output is None else iter_output.scores,
# length_penalty = 0,                       # ensures fair comparison
# # any sampling should be done with reproducibility = True
# reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
# do_sample = True,                         # if do_sample is True, use reproducibility = True
# # use parameters at will
# temperature = 0.2,                        # temperature for sampling
# top_k = 50,                               # top_k for sampling
)

transition_scores_3_5 = model.compute_transition_scores(
    output_3_5.sequences, output_3_5.scores, output_3_5.beam_indices, normalize_logits=False
)

assert torch.equal(
    output_3.sequences, output_3_5.sequences
), "Same sequences with different order of beams do not match but should match."

assert torch.equal(
    transition_scores_3, transition_scores_3_5
), "Transition scores do not match but should match."
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput
import torch

# read access token from environment variable
import os
import time
import sys

start_time = time.time()
access_token = os.getenv("HF_TOKEN")
# if access_token is not None:
#     print(f"Access token: {access_token[:3]}{'*' * 16}")
# else:
#     print("No access token found.")
    # sys.exit(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

# print all available devices
print(f"Available devices: {torch.cuda.device_count()}")
# print devices names
print(
    f"Device names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
)

checkpoints = [
    # "meta-llama/Meta-Llama-3-8B-Instruct",
    # "meta-llama/Meta-Llama-3-70B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "mistralai/Mistral-7B-v0.3",
    "EleutherAI/pythia-70m-deduped",
    # "EleutherAI/pythia-160m-deduped",
    # "EleutherAI/pythia-410m-deduped",
    # "EleutherAI/pythia-1b-deduped",
    # "EleutherAI/pythia-1.4b-deduped",
    # "EleutherAI/pythia-2.8b-deduped",
    # "EleutherAI/pythia-6.9b-deduped",
    # "EleutherAI/pythia-12b-deduped",
]


# Experiments setup
amount_of_tokens = 2
amount_of_beams = 3

elapsed_time = time.time() - start_time
print(f"Time elapsed: {elapsed_time:.2f} seconds")
model_name = checkpoints[0]

print(40 * "#")
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if tokenizer.pad_token is None:
    print(f"Setting pad token to eos token: {tokenizer.eos_token}")
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, token=access_token, device_map="auto"
)

print(f"Model {model_name} loaded successfully")
example = "Obama was born in"
examples = [example, "Michelle Obama was born"]


model_inputs = tokenizer(example, return_tensors="pt").to(device)
batched_model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to(device)

inference_start_time = time.time()
output1 = model.generate(
    **model_inputs,
    # **batched_model_inputs,
    max_new_tokens=amount_of_tokens,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    # output_attentions = True
    )

print(30 * "+", " 1st generation", 30 * "+")
if (isinstance(output1, GenerateBeamDecoderOnlyOutput)):
    print(f"Scores of shape [{len(output1.scores)}]")
    print(output1.scores)
    print("Sequences scores")
    print(output1.sequences_scores)
    print("Beam indices")
    print(output1.beam_indices)
    print("Sequences")
    print(output1.sequences)
    print("Decoded sequences")
    print(tokenizer.batch_decode(output1[0], skip_special_tokens=True))
    # print("Attention mask")
    # print(output1.attentions)
else: 
    print("Not a GenerateBeamDecoderOnlyOutput")


inference_elapsed_time = time.time() - inference_start_time

elapsed_time = time.time() - start_time
# print(f"Time elapsed: {elapsed_time:.2f} seconds")
print(f"Inference time: {inference_elapsed_time:.2f} seconds")
# print(f"Done with {model_name}!\n\n")

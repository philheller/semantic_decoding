
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

elapsed_time = time.time() - start_time
print(f"Time elapsed: {elapsed_time:.2f} seconds")
model_name = checkpoints[0]

print(40 * "#")
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    print(f"Setting pad token to eos token: {tokenizer.eos_token}")
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, token=access_token, device_map="auto"
)

print(f"Model {model_name} loaded successfully")
example = "Obama was born"
examples = [example, "Michelle Obama was born"]


model_inputs = tokenizer(example, return_tensors="pt").to(device)
batched_model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to(device)

inference_start_time = time.time()
output1 = model.generate(
    **model_inputs,
    # **batched_model_inputs,
    max_new_tokens=10,
    num_beams=4,
    num_return_sequences=4,
    return_dict_in_generate=True,
    output_scores = True,
    # output_attentions = True
    )
print(40 * "#" + f"Output:")
print(tokenizer.batch_decode(output1[0], skip_special_tokens=True)[0])
print("end of output1")

print("Output scores")
if (isinstance(output1, GenerateBeamDecoderOnlyOutput)):
    print(f"Scores of shape [{len(output1.scores)}]")
    print(output1.scores)
    print("Sequences scores")
    print(output1.sequences_scores)
    print("Beam indices")
    print(output1.beam_indices)
    print("Sequences")
    print(output1.sequences)
    # print("Attention mask")
    # print(output1.attentions)
else: 
    print("Not a GenerateBeamDecoderOnlyOutput")


inference_elapsed_time = time.time() - inference_start_time

elapsed_time = time.time() - start_time
# print(f"Time elapsed: {elapsed_time:.2f} seconds")
print(f"Inference time: {inference_elapsed_time:.2f} seconds")
# print(f"Done with {model_name}!\n\n")

# print(model_inputs)
# decoded_beams = tokenizer.batch_decode(output1[0], skip_special_tokens=True)
# print("Decoded beams:")
# print(decoded_beams)
# print("Reencoded beams:")
# print(tokenizer(decoded_beams, return_tensors="pt").to(device))

print("\n\n", 40 * "*", "Next generation")
output2 = model.generate(
    **model_inputs,
    max_new_tokens=5,
    num_beams=4,
    num_return_sequences=4,
    return_dict_in_generate=True,
    output_scores = True,
    # output_attentions = True   
)


if (isinstance(output2, GenerateBeamDecoderOnlyOutput)):
    print(f"Scores of shape [{len(output2.scores)}]")
    print(output2.scores)
    print("Sequences scores")
    print(output2.sequences_scores)
    print("Beam indices")
    print(output2.beam_indices)
    print("Sequences")
    print(output2.sequences)
    # print("Attention scores")
    # print(output2.attentions)


decoded_beams = tokenizer.batch_decode(output2[0], skip_special_tokens=True)
reencoded_beams = tokenizer(decoded_beams, return_tensors="pt").to(device)

output3 = model.generate(
    **reencoded_beams,
    past_outputs=output2,
    max_new_tokens=5,
    num_beams=4,
    num_return_sequences=4,
    return_dict_in_generate=True,
    output_scores = True,
    # output_attentions = True   
    resume_generation = True
)

if (isinstance(output3, GenerateBeamDecoderOnlyOutput)):
    print(f"Scores of shape [{len(output3.scores)}]")
    print(output3.scores)
    print("Sequences scores")
    print(output3.sequences_scores)
    print("Beam indices")
    print(output3.beam_indices)
    print("Sequences")
    print(output3.sequences)
    # print("Attention mask")
    # print(output3.attentions)


# output2 = model.generate(
#     **model_inputs,
#     past_outputs=output1,
#     # **batched_model_inputs,
#     max_new_tokens=10,
#     num_beams=4,
#     num_return_sequences=4,
#     return_dict_in_generate=True,
#     output_scores = True,
#     # output_attentions = True   
#     resume_generation = True
# )

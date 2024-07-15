from transformers import AutoModelForCausalLM, AutoTokenizer
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

print("Starting model downloads")

elapsed_time = time.time() - start_time
print(f"Time elapsed: {elapsed_time:.2f} seconds")
for model_name in checkpoints:
    print(40 * "#")
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=access_token, device_map="auto"
    )

    print(f"Model {model_name} loaded successfully")

    example = "Describe a day in the life of a citizen in a newly discovered ancient civilization, focusing on their culture, technology, and daily activities."

    # print(f"Generating model inputs")
    model_inputs = tokenizer(example, return_tensors="pt").to(device)
    # print(f"Generating output")

    inference_start_time = time.time()
    output = model.generate(**model_inputs, max_new_tokens=10, num_beams=4, num_return_sequences=4, return_dict_in_generate=True, output_scores = True)
    output1 = model.generate(**model_inputs, max_new_tokens=10, num_beams=4, num_return_sequences=4, return_dict_in_generate=True, output_scores = True)
    print(model_inputs)
    print("up model inputs, down output")
    # print output but do not show "past_key_values" and "decoder_hidden_states" keys
    print({k: v for k, v in output.items() if k not in ["past_key_values", "decoder_hidden_states"]})
    print(output.scores[0].shape)
    print(output1)
    print(40 * "*", "Next generation")
    output2 = model.generate(**model_inputs, max_new_tokens=10, num_beams=4, num_return_sequences=4, return_dict_in_generate=True, resume_generation = True)
    inference_elapsed_time = time.time() - inference_start_time

    print(40 * "#" + f"Output:")
    print(tokenizer.batch_decode(output[0], skip_special_tokens=True)[0])
    elapsed_time = time.time() - start_time
    # print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Inference time: {inference_elapsed_time:.2f} seconds")
    # print(f"Done with {model_name}!\n\n")

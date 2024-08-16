
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput
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
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-v0.3",
]

#### Experiments setup ####
# the amount of tokens will also defined the amount of
# concatenated beam searches that will be performed which 
# is i = amount_of_tokens / 2 (16 tokens will be run through 8 runs)
amount_of_tokens = 20   # amount of tokens generated
amount_of_beams = 2     # amount of beams used for generation

# examples with batching and wo batching
example = "Obama was born"
examples = [example, "Michelle Obama was born"]
# chose the example you want to test (singular or batched)
prompt = examples

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
output1 = None

total_amount_of_steps = int(amount_of_tokens / 2)

#### 3. run models ####
output_entirely = model.generate(
**model_inputs,
max_new_tokens=amount_of_tokens,
renormalize_logits = True,
num_beams=amount_of_beams,
num_return_sequences=amount_of_beams,
return_dict_in_generate=True,
output_scores = True,
# # any sampling should be done with reproducibility = True
# reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
# do_sample = True,                         # if do_sample is True, use reproducibility = True
# # use parameters at will
# temperature = 0.2,                        # temperature for sampling
# top_k = 50,                               # top_k for sampling
)


print(f"Final time: {time.time() - start_time:.2f}")
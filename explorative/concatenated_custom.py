from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput
from utils import report_output
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
print( f"Device names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

checkpoints = [
    # "meta-llama/Meta-Llama-3-8B-Instruct",
    # "meta-llama/Meta-Llama-3-70B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "mistralai/Mistral-7B-v0.3",
    # "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    # "EleutherAI/pythia-410m-deduped",
    # "EleutherAI/pythia-1b-deduped",
    # "EleutherAI/pythia-1.4b-deduped",
    # "EleutherAI/pythia-2.8b-deduped",
    # "EleutherAI/pythia-6.9b-deduped",
    # "EleutherAI/pythia-12b-deduped",
]


# Experiments setup
amount_of_tokens = 10
amount_of_beams = 4

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
    model_name, token=access_token, 
    device_map="auto"
).to(device)

print(f"Model {model_name} loaded successfully")
example = "Obama was born"
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
    report_output(output1, tokenizer)
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

decoded_beams = None
reencoded_beams = None
iter_output = None

time.sleep(1)
for i in range(int(amount_of_tokens / 2)):
    # time sleep
    inputs = model_inputs if reencoded_beams is None else reencoded_beams
    print(30 * "+", " 2nd generation", 30 * "+")
    iter_output = model.generate(
    **inputs,
    # **batched_model_inputs,
    max_new_tokens=int(amount_of_tokens / (amount_of_tokens / 2)),
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    resume_generation = True if iter_output is not None else False,
    past_outputs = iter_output,
    last_beam_scores = None if iter_output is None else iter_output.last_beam_scores,
    # output_attentions = True   
    )


    if (isinstance(iter_output, GenerateBeamDecoderOnlyOutput)):
        report_output(iter_output, tokenizer)


    decoded_beams = tokenizer.batch_decode(iter_output[0], skip_special_tokens=True)
    reencoded_beams = tokenizer(decoded_beams, return_tensors="pt", padding=True).to(device)


print(30 * "~", " comparison".upper(), 30 * "~")
print("Are the sequences the same?")
print(output1.sequences, iter_output.sequences)
# print(output1.sequences == output3.sequences)
# use torch to compare two tensors
print(
    torch.equal(
        output1.sequences,iter_output.sequences
    )
)


print("Are the scores the same?")
print(output1.scores, iter_output.scores)
print(output1.scores[0].shape)
print(iter_output.scores[0].shape)
print(len(output1.scores))
print(len(iter_output.scores))
# Function to check if two tuples of tensors are identical or nearly identical
def are_tuples_almost_equal(t1, t2, tol=1e-5):
    return all(torch.allclose(tensor1, tensor2, atol=tol) for tensor1, tensor2 in zip(t1, t2))

for i in range(len(iter_output.scores)):
    from_behind = i - len(iter_output.scores)
    print("compare ", from_behind)
    print(output1.scores[from_behind].shape, iter_output.scores[from_behind].shape)
    print(output1.scores[from_behind])
    print(iter_output.scores[from_behind])
    print(torch.allclose(output1.scores[from_behind], iter_output.scores[from_behind], atol=1e-5)   )
    # check if tensors are the same
    print(torch.equal(output1.scores[from_behind], iter_output.scores[from_behind]))
    print("Sequence scores")
    print(output1.sequences_scores[from_behind])
    print(iter_output.sequences_scores[from_behind])

print(output1.scores[0][0])
print(iter_output.scores[0][0])

print(output1.sequences_scores, iter_output.sequences_scores)
# print(output1.scores[0][0], output2.scores[0][0])


print(f"Final time: {time.time() - start_time:.2f}")
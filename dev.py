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
    # "EleutherAI/pythia-70m-deduped",
    # "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
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
    print("Last Beam Scores")
    print(output1.last_beam_scores)
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

print(30 * "+", " 2nd generation", 30 * "+")
output2 = model.generate(
    **model_inputs,
    # **batched_model_inputs,
    max_new_tokens=int(amount_of_tokens / 2),
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
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
    print("Decoded sequences")
    print(tokenizer.batch_decode(output2[0], skip_special_tokens=True))
    print("Last Beam Scores")
    print(output2.last_beam_scores)
    # print("Attention scores")
    # print(output2.attentions)


decoded_beams = tokenizer.batch_decode(output2[0], skip_special_tokens=True)
reencoded_beams = tokenizer(decoded_beams, return_tensors="pt", padding=True).to(device)

input_ids = {
    "input_ids": output1.sequences,
    "attention_mask": output1.attention_mask
}


print("Reencoded beams:")
print(reencoded_beams)
print(30 * "+", " 3rd generation", 30 * "+")
output3 = model.generate(
    **input_ids,
    # past_outputs=output2,
    max_new_tokens=int(amount_of_tokens / 2),
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    resume_generation = True
    # past_outputs = iter_output,
    last_beam_scores = output2.last_beam_scores, # should be same as sequences_scores if length_penalty = 0
    last_scores = output2.last_scores,
    length_penalty = 0,                       # ensures fair comparison
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
    print("Decoded sequences")
    print(tokenizer.batch_decode(output3[0], skip_special_tokens=True))
    print("Last Beam Scores")
    print(output3.last_beam_scores)
    # print("Attention mask")
    # print(output3.attentions)

    

print(30 * "~", " comparison".upper(), 30 * "~")
print("Are the sequences the same?")
print(output1.sequences, output3.sequences)
# print(output1.sequences == output3.sequences)
# use torch to compare two tensors
print(
    torch.equal(
        output1.sequences,output3.sequences
    )
)


print("Are the scores the same?")
print(output1.scores, output3.scores)
print(output1.scores[0].shape)
print(output3.scores[0].shape)
print(len(output1.scores))
print(len(output3.scores))
# Function to check if two tuples of tensors are identical or nearly identical
def are_tuples_almost_equal(t1, t2, tol=1e-5):
    return all(torch.allclose(tensor1, tensor2, atol=tol) for tensor1, tensor2 in zip(t1, t2))

for i in range(len(output3.scores)):
    from_behind = i - len(output3.scores)
    print("compare ", from_behind)
    print(output1.scores[from_behind].shape, output3.scores[from_behind].shape)
    print(output1.scores[from_behind])
    print(output3.scores[from_behind])
    print(torch.allclose(output1.scores[from_behind], output3.scores[from_behind], atol=1e-5)   )
    print("Sequence scores")
    print(output1.sequences_scores[from_behind])
    print(output3.sequences_scores[from_behind])

print(output1.scores[0][0])
print(output3.scores[0][0])

print(output1.sequences_scores, output3.sequences_scores)
print(output1.scores[0][0], output2.scores[0][0])


print(f"Final time: {time.time() - start_time:.2f}")
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

###############################################
########### Notes about Experiments ###########
###############################################
# These experiments are designed to ensure that the concatenated diverse beam search
# works as expected and can reproduce the exact same results as diverse beam search
# without passing inputs and outputs to a model multiple times.
# The experiment works as follows:
# 0. Imports, setup, etc
# 1. Loading tokenizer and model
# 2. Preparing inputs and outputs
# 3. Running the model i times with a concatenated diverse beam search approach:
#    - first time the model is run with a normal diverse beam search approach for the range of 
#       tokens [0, i]
#    - the second run appends to the output of previous concatenated diverse beam search 
# 4. Both results are compared and tests being run on it
#
# For this specifically, a few things have been adapted:
# - the length penalty is set to 0 to ensure that the scores are directly comparable
#   as the sequence_scores would otherwise differ depending on how much was decoded
# - group beam search finalizes the results with the beam scorer and with that changes the order
#   of the sequences. However, for a main logic loop continuation, the same input_ids as with
#   regular beam search are needed. Therefore, the `next_input_ids` are passed as
#   the `input_ids` for the next iteration.


#### Experiments setup ####
# the amount of tokens will also defined the amount of
# concatenated diverse beam searches that will be performed which 
# is i = amount_of_tokens / 2 (16 tokens will be run through 8 runs)
amount_of_tokens = 50   # amount of tokens generated
amount_of_beams = 4     # amount of beams used for generation
num_beam_groups = 4     # amount of beam groups (G = B for maximum search space [see paper]; amount_of_beams % num_beam_groups === 0)
diversity_penalty = 0.5 # diversity penalty (from paper: best when [0.2, 0.8])
# (taken from this paper)[https://arxiv.org/pdf/1610.02424]

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
# i is every step (here: every 2 tokens)
for i in range(total_amount_of_steps):
    print(50 * "#", f"Step {i}", 50 * "#")
    print(30 * "+", f"Entirely decoded [{i}] ({i*2} tokens)", 30 * "+")
    ### decoded entirely
    output_entirely = model.generate(
    **model_inputs,
    max_new_tokens=i*2 + 2,
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_beam_groups=num_beam_groups,
    diversity_penalty = diversity_penalty,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    length_penalty = 0,                       # ensures fair comparison
    )
    if i == 0:
        output1 = output_entirely

    ### decoded piece by piece
    inputs = model_inputs if last_model_output is None else last_model_output
    print(30 * "+", f"Piecewise decoded [{i}] ({i}th time 2 tokens)", 30 * "+")

    iter_output = model.generate(
    **inputs,
    max_new_tokens=int(amount_of_tokens / total_amount_of_steps),
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_beam_groups=num_beam_groups,
    diversity_penalty = diversity_penalty,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    resume_generation = True if iter_output is not None else False,
    past_key_values = None if iter_output is None else iter_output.past_key_values,
    last_beam_scores = None if iter_output is None else iter_output.last_beam_scores, # should be same as sequences_scores if length_penalty = 0
    last_scores = None if iter_output is None else iter_output.scores,
    length_penalty = 0,                       # ensures fair comparison
    )
    
    #### 4. compare and run tests ####
    ### comparison of inputs
    print("\n", 30 * "~", " Inputs".upper(), 30 * "~")
    print(model_inputs)
    print(inputs)

    ### comparison of outputs (newest scores)
    print("\n\n", 30 * "~", " scores".upper(), 30 * "~")
    # scores of the last (newest) generated tokens
    # tensors of shape (beam_size * batch_size, vocab_size)
    print(output_entirely.scores[-1])
    print(iter_output.scores[-1])

    ### tests
    # run tests to compare outputs of concatenated diverse beam search vs regular diverse beam search
    # at every step i
    if (isinstance(iter_output, GenerateBeamDecoderOnlyOutput)):
        report_output(output_entirely, tokenizer)
        report_output(iter_output, tokenizer)
        print("Are the scores the same?")
        print(
            "✅" if torch.allclose(
                output_entirely.scores[-1], iter_output.scores[-1], atol=1e-3
                ) is True else "❌",
            "✅" if torch.allclose(
                output_entirely.scores[-1], iter_output.scores[-1], atol=1e-5
                ) is True else "❌",
            "\t(with tolerances)"
            )
        print(
            "✅" if torch.equal(
                output_entirely.scores[-1], iter_output.scores[-1]
                ) is True else "❌", " \t(exact)"
            )

        print("Are the sequence_scores the same?")
        print(
            "✅" if torch.allclose(
                iter_output.sequences_scores, output_entirely.sequences_scores, atol=1e-3
                ) is True else "❌",
            "✅" if torch.allclose(
                iter_output.sequences_scores, output_entirely.sequences_scores, atol=1e-5
                ) is True else "❌",
            "\t(with tolerances)"
        )
        print(
            "✅" if torch.equal(
                iter_output.sequences_scores, output_entirely.sequences_scores
                ) is True else "❌", " \t(exact)"
            )

        print("Are the sequences the same?")
        print(
            "✅" if torch.equal(
                output_entirely.sequences,iter_output.sequences
            ) is True else "❌"
        )
        if not torch.allclose(iter_output.sequences_scores, output_entirely.sequences_scores, atol=1e-3):
            print("Difference in scores, exiting")
            break

    # use the last model output for the next iteration
    last_model_output = {
        "input_ids":  iter_output.next_input_ids,
        "attention_mask": iter_output.attention_mask
        }


print(f"Final time: {time.time() - start_time:.2f}")
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput
from utils import report_output
import os
import sys
import torch

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
# This experiment is designed to address packing and unpacking
# hypotheses for continued decoding. The experiment works as follows:
# 0. Imports, setup, etc
# 1. Loading tokenizer and model
# 2. Preparing inputs and outputs
# 3a. Running the model entirely once for a baseling
# 3b. Running the model in two steps with a concatenated beam search approach:
#   - first time the model is run with a normal beam search approach for half tokens
#   - the second run appends to the output of previous concatenated beam search
#   - outputs are packed into hypotheses and unpacked afterwards to test
#     if the scores are the same (if packing and unpacking works as expected)
#   - the third run uses the packed hypotheses and unpacks them in a different
#     order. The final scores should be the same (even the order should match,
#     as the hypotheses are ordered by which hyp is the most likely).
# 4. Comparing and running tests


#### Experiments setup ####
# the amount of tokens will also defined the amount of
# concatenated beam searches that will be performed which 
# is i = amount_of_tokens / 2 (16 tokens will be run through 8 runs)
amount_of_tokens = 50   # amount of tokens generated
amount_of_beams = 4     # amount of beams used for generation

# examples with batching and wo batching
example = "Obama was born"
examples = [example, "Michelle Obama was born"]
# chose the example you want to test (singular or batched)
prompt = examples

# select the model you want to test
model_name = checkpoints[0]


#### 1. loading model ####
# loading tokenizer and model
syntactic_generator = SyntacticGenerator(model_name, device, access_token)
# model = syntactic_generator.model
tokenizer = syntactic_generator.tokenizer

print(f"Model {model_name} loaded successfully")


#### 2. prepare inputs and outputs ####
model_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

last_model_output = None

# experiment
output_entirely = syntactic_generator.generate(
    **model_inputs,
    max_new_tokens=amount_of_tokens,
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    # use_cache=False,
    length_penalty = 0,                       # ensures fair comparison
    # # any sampling should be done with reproducibility = True
    # reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
    # do_sample = True,                         # if do_sample is True, use reproducibility = True
    # # use parameters at will
    # temperature = 0.2,                        # temperature for sampling
    # top_k = 50,                               # top_k for sampling
)

output_1 = syntactic_generator.generate(
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


# experiment: 
# get scores with compute_transition_scores (not to be passed but to be saved)
# get sequence_ids
# get past_key_values (should match sequence_ids alignment)
# recreate beam_indices (all always based on the same beam as current)
# attention mask should not matter for this yet
packed_hyps = syntactic_generator.pack_hypotheses(
    sequences=output_1.sequences,
    scores=output_1.scores,
    attention_mask=output_1.attention_mask,
    beam_indices=output_1.beam_indices,
    past_key_values=output_1.past_key_values,
    last_beam_scores=output_1.last_beam_scores,
    keep_original_data=True
)
recreated_model_output, _ = syntactic_generator.unpack_hypotheses(
    packed_hyps
)

# use the last model output for the next iteration
last_model_output = {
    "input_ids":  output_1.sequences,
    "attention_mask": output_1.attention_mask
    }

last_model_output_recreated = {
    "input_ids":  recreated_model_output["sequences"],
    "attention_mask": recreated_model_output["attention_mask"]
}

output_2 = syntactic_generator.generate(
    **last_model_output,
    max_new_tokens=int(amount_of_tokens / 2),
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    resume_generation = True if output_1 is not None else False,
    past_key_values = None if output_1 is None else output_1.past_key_values,
    last_beam_scores = None if output_1 is None else output_1.last_beam_scores, # should be same as sequences_scores if length_penalty = 0
    # last_scores = None if output_1 is None else output_1.scores,
    length_penalty = 0,                       # ensures fair comparison
    # # any sampling should be done with reproducibility = True
    # reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
    # do_sample = True,                         # if do_sample is True, use reproducibility = True
    # # use parameters at will
    # temperature = 0.2,                        # temperature for sampling
    # top_k = 50,                               # top_k for sampling
)

output_2_recreated = syntactic_generator.generate(
    **last_model_output_recreated,
    max_new_tokens=int(amount_of_tokens / 2),
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    resume_generation = True if output_1 is not None else False,
    past_key_values = None if output_1 is None else recreated_model_output["past_key_values"],
    last_beam_scores = None if output_1 is None else recreated_model_output["last_beam_scores"],
    # last_scores = None if output_1 is None else output_1.scores,
    length_penalty = 0,                       # ensures fair comparison
    # # any sampling should be done with reproducibility = True
    # reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
    # do_sample = True,                         # if do_sample is True, use reproducibility = True
    # # use parameters at will
    # temperature = 0.2,                        # temperature for sampling
    # top_k = 50,                               # top_k for sampling
)

# reorder hyps
reordered_packed_hyps = [
    packed_hyps[2], packed_hyps[1], packed_hyps[0], packed_hyps[3], packed_hyps[4], packed_hyps[5], packed_hyps[6], packed_hyps[7]
]
recreated_model_output_reordered, _ = syntactic_generator.unpack_hypotheses(
    reordered_packed_hyps
)

last_model_output_recreated_reordered = {
    "input_ids":  recreated_model_output_reordered["sequences"],
    "attention_mask": recreated_model_output_reordered["attention_mask"]
}

output_2_recreated_different_order = syntactic_generator.generate(
    **last_model_output_recreated_reordered,
    max_new_tokens=int(amount_of_tokens / 2),
    renormalize_logits = True,
    num_beams=amount_of_beams,
    num_return_sequences=amount_of_beams,
    return_dict_in_generate=True,
    output_scores = True,
    resume_generation = True if output_1 is not None else False,
    past_key_values = None if output_1 is None else recreated_model_output_reordered["past_key_values"],
    last_beam_scores = None if output_1 is None else recreated_model_output_reordered["last_beam_scores"],
    # last_scores = None if output_1 is None else output_1.scores,
    length_penalty = 0,                       # ensures fair comparison
    # # any sampling should be done with reproducibility = True
    # reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
    # do_sample = True,                         # if do_sample is True, use reproducibility = True
    # # use parameters at will
    # temperature = 0.2,                        # temperature for sampling
    # top_k = 50,                               # top_k for sampling
)

#### 4. compare and run tests ####
### comparison of inputs
print("\n", 30 * "~", " Inputs".upper(), 30 * "~")
print(model_inputs)
print(last_model_output)
print(last_model_output_recreated)

### comparison of outputs (newest scores)
print("\n\n", 30 * "~", " scores".upper(), 30 * "~")
# scores of the last (newest) generated tokens
# tensors of shape (beam_size * batch_size, vocab_size)
print(output_entirely.scores[-1])
print(output_2.scores[-1])
print(output_2_recreated.scores[-1])
print(output_2_recreated_different_order.scores[-1])

### tests
# run tests to compare outputs of concatenated beam search vs regular beam search
# at every step i
if (isinstance(output_2, GenerateBeamDecoderOnlyOutput)):
    report_output(output_entirely, tokenizer)
    report_output(output_2, tokenizer)
    print("Are the scores the same?")
    print(
        "✅" if 
            all(
                [
                    torch.allclose(
                    output_entirely.scores[-1], output_2.scores[-1], atol=1e-3
                    ),
                    torch.allclose(
                        output_2.scores[-1], output_2_recreated.scores[-1], atol=1e-3
                    ),
                    torch.allclose(
                        output_2.scores[-1], output_2_recreated_different_order.scores[-1], atol=1e-3
                    )
                ]
            )
        is True else "❌",
        "✅" if all(
                [
                    torch.allclose(
                    output_entirely.scores[-1], output_2.scores[-1], atol=1e-5
                    ),
                    torch.allclose(
                        output_2.scores[-1], output_2_recreated.scores[-1], atol=1e-5
                    ),
                    torch.allclose(
                        output_2.scores[-1], output_2_recreated_different_order.scores[-1], atol=1e-5
                    ),
                ]
            ) 
        is True else "❌",
        "\t(with tolerances)"
        )
    print(
        "✅" if all(
                [
                    torch.equal(
                    output_entirely.scores[-1], output_2.scores[-1]
                    ),
                    torch.equal(
                        output_2.scores[-1], output_2_recreated.scores[-1]
                    ),
                    torch.equal(
                        output_2.scores[-1], output_2_recreated_different_order.scores[-1]
                    ),
                ]
            )
        is True else "❌", " \t(exact)"
        )

    print("Are the sequence_scores the same?")
    print(
        "✅" if all(
                [
                    torch.allclose(
                        output_2.sequences_scores, output_entirely.sequences_scores, atol=1e-3
                    ),
                    torch.allclose(
                        output_2.sequences_scores, output_2_recreated.sequences_scores, atol=1e-3
                    ),
                    torch.allclose(
                        output_2.sequences_scores, output_2_recreated_different_order.sequences_scores, atol=1e-3
                    )
                ]
            )
        is True else "❌",
        "✅" if all(
                [
                    torch.allclose(
                        output_2.sequences_scores, output_entirely.sequences_scores, atol=1e-5
                    ),
                    torch.allclose(
                        output_2.sequences_scores, output_2_recreated.sequences_scores, atol=1e-5
                    ), 
                    torch.allclose(
                        output_2.sequences_scores, output_2_recreated_different_order.sequences_scores, atol=1e-5
                    ), 
                ]
            )
        is True else "❌",
        "\t(with tolerances)"
    )
    print(
        "✅" if all(
            [
                torch.equal(
                    output_2.sequences_scores, output_entirely.sequences_scores
                    ),
                torch.equal(
                    output_2.sequences_scores, output_2_recreated.sequences_scores
                ),
                torch.equal(
                    output_2.sequences_scores, output_2_recreated_different_order.sequences_scores
                )
            ]
        )
        is True else "❌", " \t(exact)"
        )

    print("Are the sequences the same?")
    print(
        "✅" if all(
                [
                    torch.equal(
                        output_entirely.sequences,output_2.sequences
                    ),
                    torch.equal(
                        output_2.sequences, output_2_recreated.sequences
                    ),
                    torch.equal(
                        output_2.sequences, output_2_recreated_different_order.sequences
                    )
                ]
            )
        is True else "❌"
    )
    if not torch.allclose(output_2.sequences_scores, output_entirely.sequences_scores, atol=1e-3):
        print("Difference in scores!!")


print(f"Final time: {time.time() - start_time:.2f}")

from syntactic import SyntacticGenerator
from ner_model import HuggingFaceNERModel, NERUtilities
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
max_syntactic_tokens_per_iteration = 5
amount_syntactic_beams = 10
total_max_tokens = 20
amount_semantic_beams = 3


# examples with batching and wo batching
example = "Obama was born"
examples = [example, "Abraham Lincoln was born"]
# chose the example you want to test (singular or batched)
prompt = examples

# select the model you want to test
model_name = checkpoints[0]


#### 1. loading models ####
syntacticGenerator = SyntacticGenerator(model_name, device, access_token)
# model = syntacticGenerator.model
tokenizer = syntacticGenerator.tokenizer

# loading tokenizer and model (semantic)
ner = HuggingFaceNERModel("dslim/distilbert-NER", device)

#### 2. prepare inputs and outputs ####
model_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

last_model_output = None
iter_output = None

semantic_output = None
last_beam_scores = None
# extract semantic tokens first
# todo:phil extract semantic tokens in a first run
# initial semantic token extraction simply grabs all semantic tokens

while (iter_output is None or iter_output.sequences.size(1) < total_max_tokens):
    #### 3. run model syntactic ####
    inputs = model_inputs if last_model_output is None else last_model_output

    iter_output = syntacticGenerator.generate(
    **inputs,
    max_new_tokens=max_syntactic_tokens_per_iteration,
    renormalize_logits = True,
    num_beams=amount_syntactic_beams,
    num_return_sequences=amount_syntactic_beams,
    return_dict_in_generate=True,
    output_scores=True,
    resume_generation=True if iter_output is not None else False,
    past_key_values=None,
    # ? last_beam_scores is used to avoid sampling of same sequences
    last_beam_scores=None if last_beam_scores is None else last_beam_scores,
    # past_key_values = None if iter_output is None else iter_output.past_key_values, # ? not used
    # last_scores = None if iter_output is None else iter_output.scores, # ? not used by default
    # length_penalty = 0,
    # # any sampling should be done with reproducibility = True
    # reproducibility = True,                   # ensures fair comparison by f.e. setting seeds at every gen loop step
    # do_sample = True,                         # if do_sample is True, use reproducibility = True
    # # use parameters at will
    # temperature = 0.2,                        # temperature for sampling
    # top_k = 50,                               # top_k for sampling
    )

    
    #### 4. run semantic model ####
    # prepare generation output for semantic model - batch_decode to get sequences in strings
    semantic_input = syntacticGenerator.batch_decode(iter_output.sequences)
    # run semantic model -> List of (batch_size, )
    semantic_output = ner.predict(semantic_input)

    #### 5. find new entities ####
    input_length, input_length_chars = syntacticGenerator.get_input_length(
            inputs["input_ids"], iter_output.beam_indices
        )
    output_length = syntacticGenerator.get_output_length(iter_output.sequences)

    first_new_entities, new_entities = NERUtilities.get_generated_entities(
            semantic_output, input_length_chars
        )
        
    # ? maybe stop shortening the hyps if ending condition is met
    # if (iter_output is None or iter_output.sequences.size(1) < total_max_tokens):
    #     break

    
    #### 6. shorten til right after newest entity ####
    altered_input_ids, altered_attention_mask = syntacticGenerator.shorten_hyps_to_first_entity(
        first_new_entities,
        iter_output.sequences,
        semantic_input,
        iter_output.attention_mask,
    )

    #### 7. mask duplicates and set it's score low ####
    # ? for the same beam hyps (same token id sequence), the beam score needs to be very low
    # and is set to -1e9. This is to ensure that the same hypothesis is not considered multiple times
    # which would result in sampling over the exact same tokens (leading to multiple same hypotheses).
    mask_of_duplicates, occurences  = syntacticGenerator.get_duplicates(altered_input_ids)
    last_beam_scores = mask_of_duplicates.view((len(prompt), amount_syntactic_beams))
    # those which are duplicates will receive a low beam score to avoid sampling multiple times
    last_beam_scores = last_beam_scores * -1e9



    # use the last model output for the next iteration
    last_model_output = {
        "input_ids":  altered_input_ids,
        "attention_mask": altered_attention_mask
        }

print("Syntactic output:")
print(tokenizer.batch_decode(iter_output.sequences, skip_special_tokens=True))
print("Semantic output:")
print(semantic_output)

print(f"Final time: {time.time() - start_time:.2f}")

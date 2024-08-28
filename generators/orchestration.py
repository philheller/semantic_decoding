from syntactic import SyntacticGenerator
from semantic import SemanticGenerator
from ner_model import NERUtilities
import torch
from utils import deep_compare

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
# one will 
amount_syntactic_beams = 20
total_max_tokens = 20
amount_semantic_beams = 3


# examples with batching and wo batching
example = "Obama was born"
examples = [example, "Abraham Lincoln was born in"]
# chose the example you want to test (singular or batched)
prompt = examples

# select the model you want to test
model_name = checkpoints[0]


#### 1. loading models ####
# syntactic generator
syntactic_generator = SyntacticGenerator(model_name, device, access_token)
# model = syntactic_generator.model
tokenizer = syntactic_generator.tokenizer

# semantic generator
semantic_generator = SemanticGenerator("dslim/distilbert-NER", device)

#### 2. prepare inputs and outputs ####
model_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

last_model_output = None
iter_output = None

semantic_output = None
last_beam_scores = None

# for generation
# initial semantic token extraction simply grabs all semantic tokens
initial_semantic_output = semantic_generator.generate(prompt)
initial_semantic_output = semantic_generator.merge_entities(initial_semantic_output)

semantic_inputs = semantic_generator.encode_semantic_sequences_from_entities(initial_semantic_output)
# expand semantic inputs to match the amount of semantic beams
semantic_inputs["input_ids"] = semantic_generator.expand_semantic_sequences(semantic_inputs["input_ids"], amount_semantic_beams)
semantic_inputs["attention_mask"] = semantic_generator.expand_semantic_sequences(semantic_inputs["attention_mask"], amount_semantic_beams)

# values necessary to be initialized
# general
batch_size = len(prompt)
# bs
semantic_batch_beam_size = batch_size * amount_semantic_beams
semantic_beam_indices = (
    tuple(() for _ in range(semantic_batch_beam_size))
)

semantic_beam_scores = torch.zeros((batch_size, amount_semantic_beams), dtype=torch.float, device=device)
semantic_beam_scores[:, 1:] = -1e9
semantic_beam_scores = semantic_beam_scores.view((batch_size * amount_semantic_beams,))

# map syntactic hyps to semantic hyps
syn_to_sem_mapping = torch.zeros((batch_size, amount_syntactic_beams), dtype=torch.long, device=device)
for batch_idx, batch in enumerate(syn_to_sem_mapping):
    batch[:] = batch_idx * amount_semantic_beams

last_syntactic_hyps = None
while (iter_output is None or iter_output.sequences.size(1) < total_max_tokens):
    #### 3. run model syntactic ####
    inputs = model_inputs if last_model_output is None else last_model_output

    iter_output = syntactic_generator.generate(
    **inputs, # type: ignore
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
    semantic_input = syntactic_generator.batch_decode(iter_output.sequences)
    # run semantic model -> List of (batch_size, )
    semantic_output = semantic_generator.generate(semantic_input)

    #### 5. find new semantic data ####
    # todo rework so this is generic semantic data, not just entities
    input_length, input_length_chars = syntactic_generator.get_input_length(
            inputs["input_ids"], iter_output.beam_indices
        )
    # output_length = syntactic_generator.get_output_length(iter_output.sequences)
    first_new_entities, new_entities = NERUtilities.get_generated_entities(
            semantic_output, input_length_chars
        )
    
    merged_new_entities = semantic_generator.merge_entities(new_entities)
    merged_first_new_entities = semantic_generator.merge_entities(first_new_entities)

    semantic_data = semantic_generator.get_semantic_data_from_first_entities(
        merged_first_new_entities
    )
        
    #### 6. compute transition_scores ####
    transition_scores = syntactic_generator.compute_transition_scores(
        iter_output.sequences,
        iter_output.scores,
        iter_output.beam_indices
    )
    
    #### 7. shorten til right after newest entity ####
    syntactic_source_hyp = syntactic_generator.compute_source_hypothesis_indices(
        iter_output.beam_indices
    )
    unshortened_syntactic_hyps = syntactic_generator.pack_syntactic_hypotheses(
        iter_output.sequences,
        transition_scores,
        iter_output.last_beam_scores,
        iter_output.past_key_values,
        iter_output.attention_mask,
        last_syntactic_hyps,
        syntactic_source_hyp
    )
    shortened_hyps = syntactic_generator.shorten_hyp_to_first_semantic_data_point(
        semantic_data,
        unshortened_syntactic_hyps,
        syn_to_sem_mapping.flatten()[syntactic_source_hyp],
        syntactic_source_hyp
    )

    # legacy
    (
        altered_input_ids,
        altered_attention_mask,
        altered_transition_scores,
        amount_of_tokens_shortened
    ) = syntactic_generator.shorten_hyps_to_first_entity(
        first_new_entities,
        iter_output.sequences,
        semantic_input,
        iter_output.attention_mask, # type: ignore
        transition_scores
    )

    #### 8. semantic decoding ####
    semantic_hyps = semantic_generator.compute_semantic_hypotheses(
        shortened_hyps,
        amount_syntactic_beams
    )

    # legacy
    semantic_generator.compute_semantic_scores_legacy(
        altered_transition_scores,
        amount_syntactic_beams,
        first_new_entities,
        syn_to_sem_mapping
    )

    
    # reconstruct new input from here on out
    #### 8. mask duplicates and set it's score low ####
    # ? for the same beam hyps (same token id sequence), the beam score needs to be very low
    # and is set to -1e9. This is to ensure that the same hypothesis is not considered multiple times
    # which would result in sampling over the exact same tokens (leading to multiple same hypotheses).
    mask_of_duplicates, occurences  = syntactic_generator.get_duplicates(altered_input_ids)
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

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification
import torch
from collections import defaultdict

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
amount_syntactic_beams = 300
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
# loading tokenizer and model (syntactic)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if tokenizer.pad_token is None:
    print(f"Setting pad token to eos token: {tokenizer.eos_token}")
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, token=access_token, 
    device_map="auto"
).to(device)

print(f"Model {model_name} loaded successfully")

# loading tokenizer and model (semantic)
ner_tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device=device)

print(f"NER model loaded successfully")

#### 2. prepare inputs and outputs ####
model_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

last_model_output = None
iter_output = None

semantic_output = None
last_beam_scores = None
# extract semantic tokens first
# todo:phil extract semantic tokens in a first run
# initial semantic token extraction simply grabs all semantic tokens

def _does_token_length_match_raw_string_length(sequence_ids: list[str], raw_sequence: str, tokenizer):
    """
    Check if the tokenized sequence length matches the length of the raw string
    """
    recomputed_string_length, raw_string_length = _get_token_vs_raw_string_length(sequence_ids, raw_sequence, tokenizer)
    return recomputed_string_length == raw_string_length

def _get_token_vs_raw_string_length(sequence_ids: list[str], raw_sequence: str, tokenizer):
    """
    Get the tokenized sequence length and the length of the raw string
    """
    recomputed_string = tokenizer.decode(sequence_ids, skip_special_tokens=True)
    return len(recomputed_string), len(raw_sequence)

def _token_sequence_contains_raw_string(sequence_ids: list[str], raw_sequence: str, tokenizer):
    """
    Check if the tokenized sequence contains the raw string (ignoring whitespace)
    """
    recomputed_string = tokenizer.decode(sequence_ids, skip_special_tokens=True)
    return raw_sequence.strip() in recomputed_string

    
def _get_source_hypothesis_idx(beam_indices, beam_idx, step=-1):
    """
    Get the source hypothesis index for the given beam index
    """
    if beam_indices[beam_idx, step] == -1:
        return _get_source_hypothesis_idx(beam_indices, beam_idx, step -1)
    else: 
        prior_index = beam_indices[beam_idx, step]
        if step == 0 or step == -(beam_indices.shape[1] + 1):
            return prior_index
        return _get_source_hypothesis_idx(beam_indices, prior_index, step -1)
        
count = 0
while (iter_output is None or iter_output.sequences.size(1) < total_max_tokens):
    #### 3. run model syntactic ####
    # decoded piece by piece
    inputs = model_inputs if last_model_output is None else last_model_output

    iter_output = model.generate(
    **inputs,
    max_new_tokens=max_syntactic_tokens_per_iteration,
    renormalize_logits = True,
    num_beams=amount_syntactic_beams,
    num_return_sequences=amount_syntactic_beams,
    return_dict_in_generate=True,
    output_scores = True,
    resume_generation = True if iter_output is not None else False,
    past_key_values = None,
    # ? last_beam_scores is used to avoid sampling of same sequences
    last_beam_scores = None if last_beam_scores is None else last_beam_scores,
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
    semantic_input = tokenizer.batch_decode(iter_output.sequences, skip_special_tokens=True)
    
    # run semantic model -> List of (batch_size, )
    semantic_output = ner_pipeline(semantic_input)

    #### 5. find new entities ####
    # a) start with first entity after the last entity (which we save)
    # get the length of chars before generation and after generation (for idx offset)
    if last_model_output is None:
        # if first run (iter_output is None), expand inputs_sequences to get right shape
        expanded_inputs = inputs.input_ids.repeat_interleave(amount_syntactic_beams, dim=0)
        hyps_in_plain_string = tokenizer.batch_decode(expanded_inputs, skip_special_tokens=True)
        input_len_chars = [len(hyp) if hyp is not None else None for hyp in hyps_in_plain_string]
        input_length = torch.sum(expanded_inputs != tokenizer.eos_token_id, axis=1)
    else:
        # if bs, input_len depends on the correct source hyp (find with beam_indices)
        hyps_in_plain_string = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        input_len_chars = [len(hyp) for hyp in hyps_in_plain_string]
        input_length = torch.sum(inputs["input_ids"] != tokenizer.eos_token_id, axis=1)

        if hasattr(iter_output, "beam_indices"):
            beam_hyp_input_len_chars = input_len_chars.copy()
            beam_hyp_input_length = input_length.clone()
            # get the right input length for the correct source hyp
            first_non_negative_beam_idx = (iter_output.beam_indices >= 0).sum(dim=1) - 1

            for beam_idx in range(len(iter_output.beam_indices)):
                source_hyp_idx = _get_source_hypothesis_idx(iter_output.beam_indices, beam_idx, step=first_non_negative_beam_idx[beam_idx].item())
                beam_hyp_input_length[beam_idx] = input_length[source_hyp_idx]
                beam_hyp_input_len_chars[beam_idx] = input_len_chars[source_hyp_idx]

            input_len_chars = beam_hyp_input_len_chars
            input_length = beam_hyp_input_length
            del beam_hyp_input_len_chars, beam_hyp_input_length
            
    # output length should always match shape
    output_length = torch.sum(iter_output.sequences != tokenizer.eos_token_id, axis=1)

    new_entities = []
    first_new_entities = []
    for i, entities in enumerate(semantic_output):
        entities_of_current_output = []
        for entity in entities:
            if entity["start"] > input_len_chars[i]:
                entities_of_current_output.append(entity)
        
        # if the first of entities_of_current_output["entity"] does not start with a "B", remove it
        while (len(entities_of_current_output) > 0 and entities_of_current_output[0]["entity"][0] != "B"):
            entities_of_current_output = entities_of_current_output[1:]
        new_entities.append(entities_of_current_output)
        entities_of_current_output = []

    # keep track of first new entity
    for hyp in new_entities:
        first_entity = []
        for entity_idx, entity in enumerate(hyp):
            if entity_idx > 0 and entity["entity"][0] == "B":
                break
            first_entity.append(entity)
        first_new_entities.append(first_entity)

        
    # ? maybe stop shortening the hyps if ending condition is met
    # if (iter_output is None or iter_output.sequences.size(1) < total_max_tokens):
    #     break

    
    #### 6. shorten til right after newest entity ####
    # approach: 
    # 1. shorten raw string to fit the last entity
    # 2. encode with tokenizer
    # 3. find the last matching tokens between original tokens and recomputed tokens
    # 4. check if the recomputed tokens match the original tokens
    #   4a. if they do, proceed with 5.
    #   4b. if they do not, shave off token at a time and see if it matches the length of the trimmed string
    # 5. cut the tokens at the last matching token

    # using the following variables for that
    # a) iter_output.sequences
    # b) input_len_chars
    # c) first_new_entities
    altered_input_ids = torch.empty_like(iter_output.sequences).to(device)
    altered_attention_mask = torch.zeros_like(iter_output.attention_mask).to(device)
    target_size = altered_input_ids.shape[-1]

    for beam_hyp_idx, entity_in_hypothis in enumerate(first_new_entities):
        if len(entity_in_hypothis) == 0:
            # if no entity found, simply use the tokens as is
            altered_input_ids[beam_hyp_idx] = iter_output.sequences[beam_hyp_idx].clone()
            altered_attention_mask[beam_hyp_idx] = iter_output.attention_mask[beam_hyp_idx].clone()
            continue
        last_char = entity_in_hypothis[-1]["end"]
        
        shortened_output = semantic_input[beam_hyp_idx][:last_char]
        recomputed_tokens = tokenizer(shortened_output, return_tensors="pt", padding=True).to(device)
        last_sequence_id_from_recomputed = recomputed_tokens.input_ids[0][-1]

        # sequence_id of output without padding
        trimmed_sequence = iter_output.sequences[
                beam_hyp_idx,
                (
                    iter_output.sequences[beam_hyp_idx] != tokenizer.pad_token_id
                ).nonzero().min():
            ].clone()

        if torch.equal(recomputed_tokens.input_ids[0], trimmed_sequence[:len(recomputed_tokens.input_ids[0])]):
            current_size = recomputed_tokens.input_ids.shape[-1]
            altered_input_ids[beam_hyp_idx] = torch.concat((
                    torch.tensor(
                            (target_size-current_size) * [tokenizer.pad_token_id]
                        ).to(device),
                    recomputed_tokens.input_ids[0]
                ), dim=0
            )
            altered_attention_mask[beam_hyp_idx, -current_size:] = 1
        else:
            # the first optimistic approach does not work as there is a mismatch
            # between decoding and reencoding
            
            # to find the last matching token, remove syntactic tokens until the string lenght matches
            match = False
            piecewise_shortened_output = iter_output.sequences[
                beam_hyp_idx,
                (
                    iter_output.sequences[beam_hyp_idx] != tokenizer.pad_token_id
                ).nonzero().min():
            ].clone()
            while (not match and len(piecewise_shortened_output) > 0):
                # check if decoded is the same length as the end of the entity (first one wo shortening if entity ends with string)
                decoded_piecewise = tokenizer.decode(piecewise_shortened_output, skip_special_tokens=True)
                if len(decoded_piecewise) == last_char:
                    # add padding to beginning of sequence and fix attention mask
                    current_size = len(piecewise_shortened_output)
                    altered_input_ids[beam_hyp_idx] = torch.concat((
                            torch.tensor(
                                    (target_size-current_size) * [tokenizer.pad_token_id]
                                ).to(device),
                                piecewise_shortened_output
                        ), dim=0
                    )
                    altered_attention_mask[beam_hyp_idx, -current_size:] = 1
                    match = True
                    break
                else:
                    piecewise_shortened_output = piecewise_shortened_output[:-1]
            if match:
                continue
            # if no match can be found at all, sth is wrong
            raise ValueError("Unable to find match between syntactic tokens and raw string")

    # ? for the same beam hyps (same token id sequence), the beam score needs to be very low
    # and is set to -1e9. This is to ensure that the same hypothesis is not considered multiple times
    # which would result in sampling over the exact same tokens (leading to multiple same hypotheses).

    # check if a sequence is present multiple times
    altered_input_ids_as_tuple = [tuple(seq.tolist()) if seq is not None else None for seq in altered_input_ids]
    # needs to be of size (batch_size, num_hyps_size)
    mask_of_duplicates = torch.zeros((len(prompt) * amount_syntactic_beams)).to(device)

    occurrences = defaultdict(int)
    for i, t in enumerate(altered_input_ids_as_tuple):
        occurrences[t] += 1
        mask_of_duplicates[i] = 1 if occurrences[t] > 1 else 0
    last_beam_scores = mask_of_duplicates.view((len(prompt), amount_syntactic_beams))
    # those which are duplicates will receive a low beam score to avoid sampling multiple times
    last_beam_scores = last_beam_scores * -1e9



    # use the last model output for the next iteration
    last_model_output = {
        "input_ids":  altered_input_ids,
        "attention_mask": altered_attention_mask
        }
    last_input_len = [
        len(seq) for seq in semantic_input
    ]
    count += 1

print("Syntactic output:")
print(tokenizer.batch_decode(iter_output.sequences, skip_special_tokens=True))
print("Semantic output:")
print(semantic_output)

print(f"Final time: {time.time() - start_time:.2f}")

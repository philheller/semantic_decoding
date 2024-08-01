import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

###############################################
########### Notes about Experiments ###########
###############################################
# This experiment is designed to show, that the continuated decoding methods produce
# the same scores. For that, we compare different scenarios to a custom forward loop.
# 
# The following generations are compared:
# 1. Custom forward loop
# 2. Greedy search (to compare to custom forward loop, proof of concept)
# 3. Beam search 
# 4. Continued beam search with hyps of different lengths (checking that the scores still match)
# 
# The test shows that all of these methods produce the same scores within margin of error.


# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "EleutherAI/pythia-70m-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if tokenizer.pad_token is None:
    print(f"Setting pad token to eos token: {tokenizer.eos_token}")
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
# model.eval()
model.to(device)

# Define the prompt and tokenize
prompt = "Obama was born"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

# Generate a target output using greedy search for comparison
# target_output = [39302, 369, 5686, 275, 253, 1986, 2077, 13]
target_output = [39302, 369, 5686, 275, 253, 1986, 2077, 15, 187, 187, 510, 6729, 5286]

# Original prompt length
original_prompt_length = input_ids.shape[1]
# random_initial_hyp = torch.randint(0, model.config.vocab_size, (original_prompt_length+2,)).to(device)
random_initial_hyp = torch.tensor([49557, 19499, 12296, 43912,  3463]).to(device)
random_hyp = None
def add_random_token(input_ids, amount_to_add):
    new_random_tokens = torch.randint(0, model.config.vocab_size, (amount_to_add,)).to(input_ids.device)
    return torch.cat([input_ids, new_random_tokens], dim=-1)

def pad_left(hyp1, hyp2):
    max_len = max(hyp1.shape[0], hyp2.shape[0])
    hyp1 = torch.nn.functional.pad(hyp1, (max_len - hyp1.shape[0], 0), value=tokenizer.pad_token_id)
    hyp2 = torch.nn.functional.pad(hyp2, (max_len - hyp2.shape[0], 0), value=tokenizer.pad_token_id)
    return hyp1, hyp2

# Custom forward loop to mimic greedy search
for i in range(len(target_output) - original_prompt_length):
    # Prepare inputs
    inputs = {
        "input_ids": torch.tensor([target_output[:original_prompt_length + i + 1]]).to(device),
        "attention_mask": torch.ones(1, original_prompt_length + i + 1).to(device)
    }

    if random_hyp is None:
        random_hyp = random_initial_hyp
    else:
        random_hyp = add_random_token(random_hyp, 1)

    input_ids_2, random_hyp = pad_left(inputs["input_ids"][0].clone(), random_hyp)
    hyps = torch.stack([input_ids_2, random_hyp], dim=0)
    attention_mask_2 = (hyps != 0).int()
    
    inputs_2 = {
        "input_ids": hyps,
        "attention_mask": attention_mask_2
    }

    # Custom forward loop
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits

    # Greedy search equivalent
    output_1 = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=original_prompt_length + i + 2,  # +2 to generate the next token
        num_beams=1,  # Ensuring greedy search
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True
    )

    # Beam search equivalent
    output_2 = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        # max_length=original_prompt_length + i + 2,
        max_new_tokens=1,
        num_beams=2,  # Beam search
        num_return_sequences=2,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
        renormalize_logits=False,  # Disable renormalization to compare raw logits
    )

    # Beam search continued with one longer (but presumably way less likely) hypotheses and the original prompt
    output_3 = model.generate(
        input_ids=inputs_2["input_ids"],
        attention_mask=inputs_2["attention_mask"],
        # max_length=inputs_2["input_ids"].shape[-1] + i,
        max_new_tokens=1,
        num_beams=2,  # Beam search
        num_return_sequences=2,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
        renormalize_logits=False,  # Disable renormalization to compare raw logits
        resume_generation=True,
    )

    # Print comparison
    print(f"Step {i+1}:")
    print("Custom Loop Logits:", logits[:, -1, :])
    print("Greedy Search Logits:", output_1.logits[0])
    print("Beam Search Logits:", output_2.logits[0])
    print("Beam Search custom hyps Logits:", output_3.logits[0])

    # Check the difference
    diff_greedy = torch.abs(logits[:, -1, :] - output_1.logits[0])
    diff_beam = torch.abs(logits[:, -1, :] - output_2.logits[0][0])
    diff_beam_2 = torch.abs(logits[:, -1, :] - output_3.logits[0][0])
    print("Difference with Greedy Search:", diff_greedy.max())
    print("Difference with Beam Search:", diff_beam.max())
    print("Difference with Beam Search custom hyps:", diff_beam_2.max(), f"Beam indices: {output_3.beam_indices}")
    print()

    assert all(
        [
            torch.allclose(logits[:, -1, :], output_1.logits[-1], atol=1e-6),
            torch.allclose(logits[:, -1, :], output_2.logits[-1][0], atol=1e-6),
            torch.allclose(output_2.logits[-1][0], output_3.logits[-1][0], atol=1e-6),
        ]
    ), "EXITING: Logits are not allclose!"


print("All tests passed ✅")
print("Done")
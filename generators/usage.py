import os
import time
import torch
from generator import Generator
from transformers.generation.utils import GenerationConfig
from semantic import SemanticGenerationConfig

start_time = time.time()
access_token = os.getenv("HF_TOKEN")
# some models are gated and require a hf token (make sure to request access to the model)
if access_token is not None:
    print(f"Access token: {access_token[:3]}{'*' * 16}")
else:
    print("No access token found.")
    # sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# print all available devices
print(f"Available devices: {torch.cuda.device_count()}")
print( f"Device names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")


checkpoints = [
    "gpt2",
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

# select the model you want to test
model_name = checkpoints[0]

#### 0. Experiments setup ####
# examples with batching and wo batching
example = ["Obama was born"]
# recommended: always compute in single batches, more batches 
# will not make scores reproduceable
examples = example + [
                # "Angela Merkel was born in",
                # "What is"
            ]
# chose the example you want to test (singular or batched)
# be warned: batching produces different results (just as masking)
prompt = example

# init models
generator = Generator(model_name, "dslim/distilbert-NER", device)

# set up generation configs
syntactic_generation_config: GenerationConfig = GenerationConfig(
    max_new_tokens=8,
    num_beams=20,
    num_return_sequences=20,
    do_sample=False,
    access_token=access_token,
    no_repeat_ngram_size=2,
    repetition_penalty=1.0,
    length_penalty=-.7,
)
semantic_generation_config: SemanticGenerationConfig = SemanticGenerationConfig(
    num_beams=2,
    num_return_sequences=2,
    length_penalty=-.7,
    max_overall_tokens=1000
)


generator.generate(
    prompts=prompt,
    syntactic_generation_config=syntactic_generation_config,
    semantic_generation_config=semantic_generation_config,
)

print("Done")
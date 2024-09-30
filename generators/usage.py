import os
import time
import torch
from generator import Generator
from transformers.generation.utils import GenerationConfig
from semantic import SemanticGenerationConfig
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Model and generation configuration options")
parser.add_argument(
    "-m",
    "--model", 
    type=str, 
    help="Model name or index from the predefined list. If an int is passed, it will select from the preselected model list.",
    required=True
)
parser.add_argument(
    "--syntactic_beams", 
    type=int, 
    default=20, 
    help="Number of syntactic beams for generation (default: 20)"
)
parser.add_argument(
    "-i",
    "--input",
    "--prompt",
    type=str,
    help="Input prompt for the model."
)
parser.add_argument(
    "-a",
    "--aggregation_key",
    type=str,
    choices=["text", "word", "type"],
    default="word"
)
parser.add_argument(
    "-s",
    "--semantic_token",
    type=str,
    choices=["ner", "noun_chunks"],
    default="noun_chunks",
)
args = parser.parse_args()


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

# Select model by index or string
if args.model.isdigit():
    model_index = int(args.model)
    if 0 <= model_index < len(checkpoints):
        model_name = checkpoints[model_index]
    else:
        raise ValueError(f"Model index {model_index} is out of range. Choose between 0 and {len(checkpoints)-1}.")
else:
    model_name = args.model

#### 0. Experiments setup ####
# examples with batching and wo batching
example = ["Obama was born"]
# recommended: always compute in single batches, more batches 
# will not make scores reproduceable
examples = example + [
    "Angela Merkel was born in",
    "What is",
    "Sir Charles William Fremantle KCB JP FRSA (12 August 1834 - 8 October 1914) was a British governmental official who served 26 years as deputy master of the Royal Mint. As the chancellor of the exchequer was ex officio master of the Royal Mint beginning in 1870, Fremantle was its executive head for almost a quarter century.",
    "In 1894, at the age of sixty, Fremantle retired from the Royal Mint and thereafter spent time as a corporate director and as a magistrate. He died in 1914, just under two months after his eightieth birthday.",
    "The quick brown fox jumps over the lazy dog.",
]
# chose the example you want to test (singular or batched)
# be warned: batching produces different results (just as masking)
prompt = example
if args.input is not None:
    prompt = [args.input]

generator = None
if args.semantic_token == "ner":
    generator = Generator(model_name, "dslim/distilbert-NER", device, unique_key=args.aggregation_key)
elif args.semantic_token == "noun_chunks":
    generator = Generator(model_name, "en_core_web_sm", device, unique_key=args.aggregation_key)
else:
    raise ValueError(f"Semantic token {args.semantic_token} is not supported.")

beam_size = args.syntactic_beams
# set up generation configs
syntactic_generation_config: GenerationConfig = GenerationConfig(
    max_new_tokens=8,
    num_beams=beam_size,
    num_return_sequences=beam_size,
    access_token=access_token,
)
semantic_generation_config: SemanticGenerationConfig = SemanticGenerationConfig(
    num_beams=2,
    num_return_sequences=2,
    max_overall_tokens=1000,
    max_overall_generated_tokens=256,
    nest_beam_search=True,
)


res = generator.generate(
    prompts=prompt,
    syntactic_generation_config=syntactic_generation_config,
    semantic_generation_config=semantic_generation_config,
)

duration = time.time() - start_time
print(f"End time: {duration // 60:.0f}m {duration % 60:.0f}s")
print("Done")
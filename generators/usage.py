import os
import re
import time
import torch
from semantic_decoding.generators.generator import Generator
from transformers.generation.utils import GenerationConfig
from semantic_decoding.generators.semantic import SemanticGenerationConfig
from semantic_decoding.generators.utils import TimeReporter, report_memory
import argparse

tr = TimeReporter()
gtr = TimeReporter()
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
    "gpt2",
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
    "Sir Charles William Fremantle KCB JP FRSA (12 August 1834 - 8 October 1914) was a British governmental official who served 26 years as deputy master of the Royal Mint. As the chancellor of the exchequer was ex officio master of the Royal Mint beginning in 1870, Fremantle was its executive head for almost a quarter century.",
    "In 1894, at the age of sixty, Fremantle retired from the Royal Mint and thereafter spent time as a corporate director and as a magistrate. He died in 1914, just under two months after his eightieth birthday.",
    "The quick brown fox jumps over the lazy dog.",
]
# chose the example you want to test (singular or batched)
# be warned: batching produces different results (just as masking)
prompt = examples
if args.input is not None:
    prompt = [args.input]
tr.report_time("args processed")

generator = None
tr.reset_timer()
tr.report_time("t-model-load>")
if args.semantic_token == "ner":
    generator = Generator(model_name, "dslim/distilbert-NER", device, unique_key=args.aggregation_key)
elif args.semantic_token == "noun_chunks":
    generator = Generator(model_name, "en_core_web_sm", device, unique_key=args.aggregation_key)
else:
    raise ValueError(f"Semantic token {args.semantic_token} is not supported.")
tr.report_time(f"t-model-load-end-<{model_name}>")

beam_size = args.syntactic_beams
# set up generation configs
syntactic_generation_config: GenerationConfig = GenerationConfig(
    max_new_tokens=8,
    num_beams=beam_size,
    num_return_sequences=beam_size,
    access_token=access_token,
    # length_penalty=-.7,
    no_repeat_ngram_size=2,
)
semantic_generation_config: SemanticGenerationConfig = SemanticGenerationConfig(
    num_beams=2,
    num_return_sequences=2,
    max_overall_tokens=1000,
    max_overall_generated_tokens=1000,
    nest_beam_search=True,
    # length_penalty=-.7,
)

print("MODEL_INFO** ", f"{model_name} [{args.model}]; <{args.syntactic_beams}>")
status = "success".upper()
gtr.reset_timer()
final_token_length = []
final_sem_token_length = []
try:
    for p_idx, p in enumerate(prompt):
        tr.reset_timer()
        tr.report_time(f"t-start-generation-{p_idx}->")
        res = generator.generate(
            prompts=[p],
            syntactic_generation_config=syntactic_generation_config,
            semantic_generation_config=semantic_generation_config,
        )
        ftl = res["syntactic_sequences"].shape[-1]
        stl = res["semantic_sequences"].shape[-1]
        tr.report_time(f"t-end-generation-{p_idx}-> ftl:{ftl}")
        mem_sum = report_memory("", False)
        print(f"m-{p_idx}> {mem_sum[-2]}")
        final_token_length.append(ftl)
        final_sem_token_length.append(stl)
except torch.cuda.CudaError as e:
    error_message = str(e).lower()
    if re.search(r'out.*memory', error_message):
        status = "cuda_memory".upper()
    else:
        status = "cuda".upper()
    print(f"CUDA Error: {e}")
except Exception as e:
    status = "error".upper()
    print(f"Error: {e}")
gtr.report_time(f"t-total-generation-time> {model_name}")
print(f"ftl-total-length> {','.join(str(final_token_length))}")
print(f"stl-total-length> {','.join(str(final_sem_token_length))}")
mem_sum = report_memory("", False)
print(f"m-total-max-> {max(mem_sum[-2][0])}GB ({max(mem_sum[-2][1]):.3f}%)")


duration = time.time() - start_time
print(f"End time: {duration // 60:.0f}m {duration % 60:.0f}s")
print(f"Status: {status}")
print("Done")
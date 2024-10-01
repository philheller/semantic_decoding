from semantic_decoding.generators.argparser import create_argparser
from typing import Tuple

import os
import csv
from semantic_decoding.generators.generator import Generator
from transformers.generation.utils import GenerationConfig
from semantic_decoding.generators.semantic import SemanticGenerationConfig

import torch
from datasets import load_dataset
from tqdm import tqdm

CHECKPOINTS = [
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


def postprocess_args(args):
    try:
        args.model = int(args.model)
    except ValueError:
        pass
    if isinstance(args.model, int):
        args.model = CHECKPOINTS[args.model]
    return args

def create_generation_configs(args) -> Tuple[GenerationConfig, SemanticGenerationConfig]:
    # synt config
    syntactic_generation_config: GenerationConfig = GenerationConfig(
        max_new_tokens=args.synt_max_new_tokens,
        num_beams=args.syntactic_beams,
        num_return_sequences=args.syntactic_beams,
        access_token=access_token,
    )

    semantic_generation_config: SemanticGenerationConfig = SemanticGenerationConfig(
        num_beams=args.semantic_beams,
        num_return_sequences=args.semantic_beams,
        max_overall_tokens=1000,
        max_overall_generated_tokens=1,
        nest_beam_search=True,
    )

    return syntactic_generation_config, semantic_generation_config

def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = Generator(
        model_name=args.model,
        semantic_generators=args.semantic_model_name,
        device=device,
        access_token=None,
        unique_key=args.aggregation_key
    )
    return generator

def write_to_target(res, file_path):
    # get path of this script
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    with open(file_path, mode='a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        try:
            writer.writerows(res)
        except Exception as e:
            print(f"Error writing to file: {e}")

def infer_file_name(
    model_name: str, 
    semantic_token_type: str,
    syntactic_amount_tokens: int,
) -> str:
    return (
         f"{model_name.split('/')[-1]}_{semantic_token_type}_{syntactic_amount_tokens:02d}.csv",
         f"{model_name.split('/')[-1]}_{semantic_token_type}_{syntactic_amount_tokens:02d}_generated.csv",
        )

# if called as module
if __name__ == "__main__":
    # Argument parser
    parser = create_argparser()
    args = parser.parse_args()
    args = postprocess_args(args)
    access_token = os.getenv("HF_TOKEN")
    # some models are gated and require a hf token (make sure to request access to the model)
    if access_token is not None:
        print(f"Access token: {access_token[:3]}{'*' * 16}")
    else:
        print("No access token found.")
        # sys.exit(1)

    synt_conf, sem_conf = create_generation_configs(args)
    generator = load_models()

    # generate texts
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    shuffled_dataset = ds.shuffle(buffer_size=10_000, seed=42)

    shuffled_dataset = iter(shuffled_dataset)
    file_name, file_name_generations = infer_file_name(
        args.model,
        args.semantic_token_type,
        args.synt_max_new_tokens,
    )
    
    metric_res = [
        ["num_semantic_tokens", "num_unique_semantic_tokens"]
    ]
    generated_res = [
        ["id", "prompt", "syntactic", "semantic"]
    ]

    p_bar = tqdm(total=10_000, desc="Generating")
    for i in range(10_000):
        next_prompt = next(shuffled_dataset)
        constructed_prompt = " ".join(next_prompt["text"].split(" ")[:20])
        res = None
        try: 
            res = generator.generate(
                [constructed_prompt],
                syntactic_generation_config=synt_conf,
                semantic_generation_config=sem_conf,
            )
        except Exception as e:
            tqdm.write(f"Error at [{next_prompt["id"]}]: {e}")
            continue

        metrics_list = [
            len([sem_data for sem_data in res["last_semantic_data"] if sem_data is not None]),
            len(set(res["last_semantic_data"]))
        ]

        semantic_tokens_txt = generator.semantic_tokenizer.batch_decode(res["semantic_sequences"])
        syntactic_txt = generator.syntactic_tokenizer.batch_decode(res["syntactic_sequences"], skip_special_tokens=True)
        generated_texts_list = [
            next_prompt["id"],
            constructed_prompt,
            syntactic_txt,
            semantic_tokens_txt,
        ]

        metric_res.append(metrics_list)
        generated_res.append(generated_texts_list)
        
        if i % 1000 == 0:
            p_bar.set_description(f"Saving to file")
            write_to_target(metric_res, file_name)
            write_to_target(generated_res, file_name_generations)
            metric_res = []
            generated_res = []
        p_bar.set_description(f"Generating")
        p_bar.update(1)


    print("Done")
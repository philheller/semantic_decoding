import torch
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import datasets

from typing import List, Dict

import os
import re
import json
from copy import deepcopy
import argparse
import tqdm

from FactualityPrompt.fever_athene.src.retrieval.fever_doc_db import FeverDocDB
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

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

def create_argparser(
    formatter=argparse.ArgumentDefaultsHelpFormatter,
):
    parser = argparse.ArgumentParser(
        description="Run transformer models in semantic decoding generation.",
        epilog="HF:)",
        formatter_class=formatter,
    )

    
    # general
    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input file for prompts.",
        default="fever_factual_final.jsonl",
    )
    general_group.add_argument(
        "-o",
        "--output",
        type=str,
        help="Chose output file name. If not set, name will be auto generated.",
    )
    general_group.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index of prompt from prompt file (line) for generation.",
    )
    general_group.add_argument(
        "--end",
        type=int,
        default=0,
        help="End index of prompt from prompt file (line) for generation.",
    )
    general_group.add_argument(
        "--access-token",
        type=str,
        help="Access token for private models.",
    )
    
    # syntactic
    syntactic_group = parser.add_argument_group("Syntactic Config")
    syntactic_group.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model name or index from the predefined list. If an int is passed, it will select from the preselected model list.",
        required=True,
        default=1,
    )
    syntactic_group.add_argument(
        "--syntactic-beams",
        type=int,
        default=200,
        help="Number of syntactic beams for generation.",
    )
    syntactic_group.add_argument(
        "--synt-max-new-tokens",
        type=int,
        default=5,
        help="Max new tokens before checking if entity is in the golden answer.",
    )
    
    return parser

def postprocess_args(args):
    try:
        args.model = int(args.model)
    except ValueError:
        pass
    if isinstance(args.model, int):
        args.model = CHECKPOINTS[args.model]
    return args

def load_model(
        model_name:str,
        access_token: str,
        force_even_split: bool = False
    ) -> AutoModelForCausalLM:
    """
    Load a pre-trained model from Hugging Face model hub.
    
    :param model_name: Name of the model to load.
    :type model_name: str
    :param device: Device to load the model on.
    :type device: str
    :param access_token: Access token for private models.
    :type access_token: Optional[str]
    :return: Loaded model.
    :rtype: Any
    """
    print(f"Loading model: {model_name}")
    # Create a device map if more than one GPU is available
    device_map = "auto"
    if force_even_split:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            device_map = {str(i): i for i in range(num_gpus)}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=access_token, 
        device_map=device_map
    )
    model.eval() # Though this is default, but just to be sure
    print(f"Model: {model_name}")
    print(f"Model on device: {model.device}")
    print(f"Model device map: {model.hf_device_map}")
    print(f"Using precision: {next(model.parameters()).dtype}")
    print(f"Eval mode: {not model.training}")
    return model

def create_generation_configs(args) -> GenerationConfig:
    # synt config
    syntactic_generation_config: GenerationConfig = GenerationConfig(
        max_new_tokens=args.synt_max_new_tokens,
        num_beams=args.syntactic_beams,
        num_return_sequences=args.syntactic_beams,
        access_token=access_token,
    )

    return syntactic_generation_config

def create_results_folder():
    current_dir = os.path.dirname(__file__)
    results_dir = os.path.join(current_dir, "results")
    if not os.path.exists(results_dir):
        print("Creating results folder at ", results_dir)
        os.makedirs(results_dir)

def write_to_target(res: List[Dict], file_path):
    # get path of this script
    file_path = os.path.join(os.path.dirname(__file__), "results", file_path)
    with open(file_path, mode='a', newline='', encoding="utf-8") as file:
        try:
            for item in res:
                file.write(json.dumps(item) + '\n')
        except Exception as e:
            print(f"Error writing to file: {e}")

def infer_file_name(
    prefix: str,
    model_name: str, 
    extra_annotation: str,
) -> str:
    return f"{prefix}_{model_name.split('/')[-1]}_{extra_annotation}_results.jsonl"

def tokenization(used_tokenizer, example):
    input_dict = used_tokenizer(example["prompt"], return_tensors="pt", padding=True)
    return {
        **input_dict,
        **example,
    }

# same as from FactualityPrompt/fever
def get_wiki_from_db(wiki_names, db):
    
    all_lines = []
    for wiki_name in wiki_names:
        
        lines = db.get_doc_lines(wiki_name)
        if lines != None:
            all_lines.extend(sent_tokenize(lines))
            
    return all_lines

# see FactualityPrompt
IMPORTANT_ENT_TYPE = set(['ORG', 'PERSON', 'WORK_OF_ART', 'PRODUCT', 'EVENT'])
REMOVE_ENT_TYPE = set(['ORDINAL', 'CARDINAL']) 
# Cardinal numbers tell 'how many' of something, they show quantity. Ordinal numbers tell the order of how things are set, they show the position or the rank of something.

# same as frmo FactualityPrompt
def obtain_important_ne(gen, nlp, include_capitalized_words_as_ents=True):
    important_words = []

    doc = nlp(gen)

    # print("GEN: ", gen)
    # print([(token.text, token.pos_, token.tag_, token.dep_) for token in doc if token.pos_ in ['NOUN', 'PRON', 'PROPN']])
    # print("\n")

    ents = [(ent.text, ent.label_) for ent in doc.ents]

    if include_capitalized_words_as_ents and len(ents) == 0:
        capitalized_words = re.findall('(?<!^)([A-Z][a-z]+)', gen)
        
        if len(capitalized_words) > 0:
            capitalized_words = [(word, 'CAPITALIZED') for word in capitalized_words if word.lower() not in stop_words]
            ents.extend(capitalized_words)

    important_words.extend([ent for ent in ents if ent[1] in IMPORTANT_ENT_TYPE])
    remaining_ne_all = [ent for ent in ents if ent[1] not in IMPORTANT_ENT_TYPE]

    # filter out some ne
    remaining_ne = []
    for ent in remaining_ne_all:
        if ent[1] in REMOVE_ENT_TYPE:
            continue
        if ent[1] == 'DATE' and ("year" in ent[0] or "day" in ent[0]): #not bool(re.search(r'\d', ent[0])):
            # if "DATE" entity contains NO number at all (e.g., ``the year''), meaningless
            continue
        remaining_ne.append(ent)

    gens_with_ne = {
                        "gen": gen,
                        "important_ne": important_words,
                        "unimportant_ne": remaining_ne,
                        "subject": set([token.text for token in doc if token.dep_ in ['nsubj', 'nsubjpass']]),
                        # "all_analysis": [(token.text, token.pos_, token.tag_, token.dep_) for token in doc]
                    }

    return gens_with_ne 

# taken from FactualityPrompt (originally `ner_metric`) and adapted to return boolean instead of metric
def is_correct_ne(named_entity, prompt_wiki_candidates) -> bool:
    """ 
    Uses the same logic as FactualityPrompt to determine if a named entity is correct.
    If any is not correct, return False.
    """
    
    wiki_text = " ".join(prompt_wiki_candidates).lower()

    for ent in named_entity:
        # preprocess
        ent_text = ent[0].lower()
        if 'the ' in ent_text:
            ent_text = ent_text.replace('the ', "")

        if ent_text in wiki_text:
            continue
        elif any([bool(word in wiki_text) for word in ent_text.split(" ") if ent[1] == 'PERSON']):
            # handle shorter forms of same NE: Exists "Marcus Morgan Bentley", but NE is "Marcus Bentley" or "Bentley"
            continue
        elif ent[1] == 'DATE':
            date_str = re.sub(r"[,.;@#?!&$]+\ *", " ", ent_text)
            date_str = date_str.replace("st", "")
            date_str = date_str.replace("nd", "")
            date_str = date_str.replace("th", "")
            date_str = date_str.replace("of", "")
            date_tokens = date_str.split(" ")

            if all([bool(token in wiki_text) for token in date_tokens]):
                    continue
        # if none of the matching is the case, return False -> not a wiki text ent
        else:
            return False
        
    # in case all ne's have been matched to wiki text
    return True

def main(args):
    cur_dir = os.getcwd()
    db_path = os.path.join(cur_dir, "FactualityPrompt", "data", "kilt_db.db")
    db = FeverDocDB(db_path)
    create_results_folder()
    model = load_model(
        args.model,
        args.access_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        # check if unk tokens is set
        if tokenizer.unk_token is not None:
            print(f"pad token is None. Setting pad token to same as unk token: {tokenizer.unk_token}")
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            print(f"pad token is None. Setting pad token to same as eos token: {tokenizer.eos_token}")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Pad token could be set to neither unk nor eos token.")
    if not spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    # create dataset with hf datasets from jsonl file
    factuality_prefix = "factual" if "non" not in args.input else "nonfactual"
    prompt_file_name = os.path.join(os.path.dirname(__file__), args.input)
    dataset = datasets.load_dataset('json', data_files=prompt_file_name, split='train')

    ds_1k = dataset.select(range(100))
    ds_1k = ds_1k.map(lambda x: tokenization(tokenizer, x), batched=False)
    # ds_1k.set_format(type="torch", columns=["input_ids", "attention_mask"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generation_config = GenerationConfig(
        renormalize_logits=True,
        max_new_tokens=256,
        num_beams=args.syntactic_beams,
        num_return_sequences=args.syntactic_beams,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    generation_config_piecewise = deepcopy(generation_config)
    generation_config_piecewise.max_new_tokens = args.synt_max_new_tokens
    
    result_objs_regular_bs = []
    result_objs_constrained_bs = [] # note: this is not actually equivalent to constrained decoding


    p_bar = tqdm.tqdm(total=len(ds_1k), desc="Generate", position=0)
    for prompt_idx, prompt_obj in enumerate(ds_1k):
        model_input = {
            "input_ids": torch.tensor(prompt_obj["input_ids"]).to(device),
            "attention_mask": torch.tensor(prompt_obj["attention_mask"]).to(device),
        }
        model_input_length = model_input["input_ids"].shape[-1]
        model_input_char_length = len(prompt_obj["prompt"])
        # generate #-beams with regular beam search
        p_bar.set_postfix(status="Generate (regular)")
        regular_output = model.generate(
            **model_input,
            generation_config=generation_config
        )
        res_regular_bs = {
            "id": prompt_obj["id"],
            "prompt": prompt_obj["prompt"],
            "text": tokenizer.batch_decode(regular_output.sequences[:1, model_input_length:], skip_special_tokens=True)[0],
        }
        result_objs_regular_bs.append(res_regular_bs)
        del regular_output

        iter_output = None
        last_beam_scores_altered = None
        last_model_output = None
        p_bar.set_postfix(status="Fetch Wiki txt")
        prompt_wiki_names = [ev_infos[0] for ev_infos in prompt_obj['evidence_info']]
        wiki_sentences_for_prompt = get_wiki_from_db(prompt_wiki_names, db)
        p_bar.set_postfix(status="Generate (constr)")
        while (
            not iter_output or
            iter_output.sequences.shape[-1] < (model_input_length + generation_config.max_new_tokens)
        ):
            inputs = model_input if last_model_output is None else last_model_output
            # generate #-beams with regular beam search, but only allow for entities in the golden answer to persist (same logic as in FactualityPrompt)
            iter_output = model.generate(
                **inputs,
                generation_config=generation_config_piecewise,
                resume_generation=True if iter_output else False,
                past_key_values=iter_output.past_key_values if iter_output else None,
                last_scores=iter_output.scores if iter_output else None,
                last_beam_scores=last_beam_scores_altered if iter_output else None,
                dynamic_decoder_prompt_length=model_input_length,
            )

            generated_output = iter_output.sequences[:, model_input_length:]

            # check if ner is there
            decoded_output = tokenizer.batch_decode(generated_output, skip_special_tokens=True)
            shortened_decoded_output = [decoded_hyp[model_input_char_length:] for decoded_hyp in decoded_output]
            
            sent_objs_with_ne = [obtain_important_ne(sent, nlp) for sent in shortened_decoded_output]
            
            mark_for_low_score = torch.zeros(iter_output.sequences.shape[0]).to(iter_output.sequences.device)
            for sent_obj_idx, sent_obj in enumerate(sent_objs_with_ne):
                all_ne = sent_obj["important_ne"] + sent_obj["unimportant_ne"]
                if len(all_ne) == 0:
                    continue
                else:
                    # now, need to check if ne also in wiki text
                    # use same approach as FactualityPrompt;
                    # def is_correct_ne is adaption of that (outputs boolean instead of metric)
                    is_ne_in_golden_answer = is_correct_ne(all_ne, wiki_sentences_for_prompt)
                    if not is_ne_in_golden_answer:
                        mark_for_low_score[sent_obj_idx] = 1
                    
            # set low score for those that are not correct
            last_beam_scores_altered = iter_output.last_beam_scores.clone()
            last_beam_scores_altered[mark_for_low_score == 1] = -1e9
            
            last_model_output = {
                "input_ids":  iter_output.sequences,
                "attention_mask": iter_output.attention_mask,
                }
        # here, chose the sequences are sorted by score
        # -> chose the first sequence with last beam score > -1e9 (otherwise it is marked
        # as not incorrect ner)
        generated_output = None
        if (last_beam_scores_altered <= -1e9).all():
            generated_output = "<NO_CANDIDATE>"
        else:
            indices_of_acceptable_hypotheses = (last_beam_scores_altered > -1e9).nonzero(as_tuple=True)[0]
            chosen_sequence = indices_of_acceptable_hypotheses[0]
            generated_output = tokenizer.batch_decode(iter_output.sequences[chosen_sequence:chosen_sequence+1, model_input_length:], skip_special_tokens=True)[0]
        res_constrained_bs = {
            "id": prompt_obj["id"],
            "prompt": prompt_obj["prompt"],
            "text": generated_output,
        }
        result_objs_constrained_bs.append(res_constrained_bs)

        if prompt_idx % 10 == 0:
            p_bar.set_postfix(status="Write to file")
            write_to_target(result_objs_regular_bs, infer_file_name(factuality_prefix, args.model, "regular_bs"))
            write_to_target(result_objs_constrained_bs, infer_file_name(factuality_prefix, args.model, "constrained_bs"))
            result_objs_regular_bs = []
            result_objs_constrained_bs = []
            pass
        # extract the first entities for both
        # -> aggregate most common entity for both
        # visualize most common entity for both
        p_bar.update(1)
    p_bar.close()


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

    main(args)
    print("Done")
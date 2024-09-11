# Setup

1. Clone this repo
2. Clone the [hf fork](https://github.com/philheller/transformers.git) to a sibling directory

```bash
# the repos should be in the same directory for the yml install to work
ls
# my_folder/
#    ma-eval/           # this repo
#    transformers/      # the hf fork
```

3. Install all dependencies (currently only `environment.yml` & `environment-gpu.yml` are up to date)
```bash
# from the root of this repo
conda env create -f env/environment.yml
# conda env create -f env/environment-gpu.yml # use instead for gpu support
# make sure the pip dependencies in the yml file have properly been installed
```

For usage, activate the enviroment and see [Usage](#Usage).
```bash
conda activate sem-cpu
# conda activate sem # use instead for gpu support
```

# Usage

Currently, Greedy and Beam search are implemented. To run a prompt, go to
- `generators/greedy_semantic.py` for Greedy Search
- `generators/beam_search_semantic.py` for Beam Search

Change any values within it to play around with settings/prompts, etc.
I currently set breakpoints to look at end results.


# General structure
Helper functions are mostly separated into to classes responsible for generation:
1. the [SyntacticGenerator](./generators/syntactic.py)
2. the [SemanticGenerator](./generators/semantic.py)

The SyntacticGenerator contains the functions associated with manipulation of syntactic hypothesis. The SemanticGenerator contains the functions associated with manipulation of semantic hypothesis.

Both classes also contain the models and the tokenizers used:

```python
# to decode syntactic tokens
syntactic_generator.tokenizer.batch_decode(syntactic_output)

# to decode semantic tokens
semantic_generator.tokenizer.batch_decode(semantic_output)
```

# Known issues
> ⚠️ There is currently still an edge case which will throw an exception:
> Usually, when eos tokens are produced, alternatives have to be generated for beam search to keep going (same sequence but without the eos token). However, it can currently happen, that in a generation step, all synt. hyps are eos tokens which leaves no other hyps to keep generating. This will be fixed soon.
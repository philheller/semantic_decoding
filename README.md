# Setup
There are two ways to set up the environment. [One simply installs all dependencies](#build-the-environment) to get you up and running. The [other uses the transformers fork](#build-the-environment-with-the-hf-fork-in-editable-mode) necessary for the project in editable mode. This setup is recommended if you want to make changes to the transformers library while working on any implementation details.

## Build the environment
1. Clone this repo

```bash
cd semantic_decoding
# build env
conda env create -f env/environment.yml
# conda env create -f env/environment-gpu.yml # use instead for gpu support

conda activate sem
```

## Build the environment with the HF fork in editable mode
1. Clone this repo
2. Clone the [hf fork](https://github.com/philheller/transformers.git) to a sibling directory

```bash
# the repos should be in the same directory for the yml install to work; otherwise adapt path in yml file
ls
# my_folder/
#    semantic_decoding/     # this repo
#    transformers/          # the hf fork
```

3. comment out the remote source of transformers in the `environment*.yml` file and point to the the local directory instead
```diff
name: sem
channels:
  - ...
dependencies:
  - ...
  - pip
  - pip:
-      - git+https://github.com/philheller/transformers.git
+     # - git+https://github.com/philheller/transformers.git
-     # - -e ../transformers
+      - -e ../transformers

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
conda activate sem
```

# Usage

Currently, Greedy and Beam search are implemented. To run a prompt, go to
- [`generators/greedy_semantic.py`](./generators/greedy_semantic.py) for Greedy Search
- [`generators/beam_search_semantic.py`](./generators/beam_search_semantic.py) for Beam Search

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
1. Only EOS tokens\\
⚠️ There is currently still an edge case which will throw an exception:
Usually, when eos tokens are produced, alternatives have to be generated for beam search to keep going (same sequence but without the eos token). However, it can currently happen, that in a generation step, all synt. hyps are eos tokens which leaves no other hyps to keep generating. This will be fixed soon.

2. Batching and scores\\
Scores are not resolving to be the same based on batching and masking. This can change the results of beam search (more on that in [tests regarding differences in scores](./tests/score_differences/different_beams.py)). To make a result reproducible (and thus easily accessible), batching should be avoided. Not batched computations can be reproduced.

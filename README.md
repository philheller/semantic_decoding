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

The usage of semantic decoding is provided through the `Generator` class. Here is simple usage:
  
```python
# generator
from semantic_decoding.generators.generator import Generator
# generation config for syntactic and semantic level
from transformers.generation.utils import GenerationConfig
from semantic_decoding.generators.semantic import SemanticGenerationConfig

# load the generator
generator = Generator(
  model_name,
  "en_core_web_sm",
  device,
  unique_key=args.aggregation_key
)

# generation configs
# syntactic
syntactic_generation_config: GenerationConfig = GenerationConfig(
    max_new_tokens=4,
    num_beams=200,
    num_return_sequences=200,
    access_token=access_token,
    # ...
)
# semantic
semantic_generation_config: SemanticGenerationConfig = SemanticGenerationConfig(
    num_beams=2,
    num_return_sequences=2,
    max_overall_tokens=1000,
    max_overall_generated_tokens=1000,
    nest_beam_search=True,
)

# generate
res = generator.generate(
    prompts=["Obama was born in"],
    syntactic_generation_config=syntactic_generation_config,
    semantic_generation_config=semantic_generation_config,
)
```

## Model Choices
Syntactic models are the [HF models](https://huggingface.co/models).\
Available models for the semantic generation can be viewed and implemented in [`smeantic_model.py`](./generators/semantic_model.py). Currently, some ner models and some spacy models are supported. Adding new ones requires implementing the `SemanticDataModel` class, the already implemented models serve as examples.

## Generation Modes

The `Generator.generate` function is structurally kept analogous to the `transformers` library. Currently, these decoding modes are supported:
1. Greedy decoding
2. Beam Search decoding
3. Nested Beam Search decoding

The appropriate mode is selected based on the semantic and syntactic generation config. For more details, see the [SemanticGenerationConfig](./generators/semantic.py).

# General structure
Central to the generation is the [`Generator`](./generators/generator.py) class which orchestrates the generation. Helper functions are mostly for syntactic and semantic generation structure code further:
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
1. Batching and scores
  Scores are not resolving to be the same based on batching and masking. This can change the results of beam search (more on that in [tests regarding differences in scores](./tests/score_differences/different_beams.py)). To make a result reproducible (and thus easily accessible), batching should be avoided. Not batched computations can be reproduced.

# Prerequisites

This experiment also requires another repository for it to work. Specifically, it needs the pipeline provided by the FactualityPrompt repo. As it is somewhat out of date and support for windows is not there, a [fork](https://github.com/philheller/FactualityPrompt.git#1-setup) has been created detailing how to proceed. Set up the [FactualityPrompt repo](https://github.com/philheller/FactualityPrompt.git#1-setup) first and then come back here to set up this experiment. Note: the script assumes that FactualityPrompt is within a sibling directory to this repo:

```
semantic_decoding/
FactualityPrompt/
```

# Setup

> The `environment.yml` provided here is meant to be used for the generation. For the evaluation, use the fork itself which has its own instructions and environment.

1. To set up an environment that can deal with both the packages necessary for generation (and this repo's requirements) and the FactualityPrompt repo, an explicit `environment.yml` is provided. To set up the environment, run the following command:
    ```bash
    conda env create -f environment.yml
    # ...

    conda activate semantic_decoding
    ```
2. Copy the `fever_factual_final.jsonl` file from the [FactualityPrompt repo](https://github.com/nayeon7lee/FactualityPrompt.git) to this directory.

# Running the experiment
To run the experiment:
```bash
ls -l
>> FactaulityPrompt/
>> semantic_decoding/

python semantic_decoding/experiments/factuality_prompt/factuality_prompt.py
```

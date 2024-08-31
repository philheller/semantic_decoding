from transformers.generation.utils import GenerateBeamDecoderOnlyOutput, GenerationConfig


GenerationConfig(
    max_length=20,
    num_beams=5,
    num_return_sequences=5,
    length_penalty=1.0,
    do_sample=False,
    use_cache=True,
)

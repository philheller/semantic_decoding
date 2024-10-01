import argparse


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
        "--prompt",
        type=str,
        help="Input prompt for the model.",
    )
    general_group.add_argument(
        "-o",
        "--output",
        type=str,
        help="Chose output file name. If not set, name will be auto generated.",
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
        default=20,
        help="Number of syntactic beams for generation.",
    )
    syntactic_group.add_argument(
        "--synt-max-new-tokens",
        type=int,
        default=8,
        help="Maximum number of new tokens for generation.",
    )
    
    # semantic
    semantic_group = parser.add_argument_group("Semantic Config")
    semantic_group.add_argument(
        "--semantic-model-name",
        type=str,
        default="en_core_web_sm",
        help="Name of the semantic model to use. Make sure it is properly setup and supported.",
    )
    semantic_group.add_argument(
        "--semantic-beams",
        type=int,
        default=4,
        help="Number of semantic beams for generation.",
    )
    semantic_group.add_argument(
        "-a",
        "--aggregation-key",
        type=str,
        choices=["text", "word", "type"],
        default="word",
        help="This is what the semantic tokens are aggregated over."
        "Word and text are the strings of a semantic token with the former being smaller, the latter being wider."
        "Type is the type of the semantic token. This can f.e. be interesting for NER.",
    )
    semantic_group.add_argument(
        "--semantic-token-type",
        type=str,
        choices=["ner", "noun_chunks"],
        default="noun_chunks",
        help="Use either noun_chunks or ner for the semantic tokens. Some models support both (spacy)."
    )
    return parser

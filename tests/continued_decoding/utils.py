def report_output(output, tokenizer):
    if hasattr(output, "sequences_scores"):
        # only some have sequences_scores (f.e. greedy does not)
        print("Sequences scores")
        print(output.sequences_scores)
    print("Sequences")
    print(output.sequences)
    max_sequence_len = max([len(seq) for seq in output.sequences])
    print(f"Decoded sequences [of length {max_sequence_len}]")
    print(tokenizer.batch_decode(output[0], skip_special_tokens=True))
    if hasattr(output, "last_beam_scores"):
        print("Last Beam Scores")
        print(output.last_beam_scores)
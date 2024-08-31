from semantic import SemanticGenerator


ner = SemanticGenerator("dslim/distilbert-NER")
ner2 = SemanticGenerator("lxyuan/span-marker-bert-base-multilingual-uncased-multinerd")

example = "Obama was born in New York, New Jersey, New York, New Jersey,"

ner.generate([example])
ner.generate([example])
print("test")
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)

# import spacy
# from flair.data import Sentence
# from flair.models import SequenceTagger

class NERModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    @abstractmethod
    def predict(self, text: List[str]) -> List[Dict[str, Any]]:
        """
        Run NER prediction on a list of texts.
        
        :param text: List of texts to run NER prediction on.
        :type text: List[str]
        :return: List of dictionaries containing NER predictions.
        :rtype: List[Dict[str, Any]]
        """
        raise NotImplementedError("This is an abstract method.")

# see @link https://huggingface.co/lxyuan/span-marker-bert-base-multilingual-uncased-multinerd
class HuggingFaceNERModel(NERModel):
    def __init__(self, model_name: str, device):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=device)
    
    def predict(self, text: List[str]) -> List[Dict[str, Any]]:
        return self.pipeline(text)


# todo:phil implement SpacyNERModel and FlairNERModel
# see @link https://huggingface.co/spacy/en_core_web_md and https://huggingface.co/spacy/en_core_web_trf
# see implementation @link https://spacy.io/
# class SpacyNERModel(NERModel):
# # is in iob notation
#     def __init__(self, model_name: str):
#         self.nlp = spacy.load(model_name)
    
#     def predict(self, text: List[str]) -> List[Dict[str, Any]]:
#         results = []
#         for doc in self.nlp.pipe(text):
#             entities = [{"entity": ent.label_, "text": ent.text} for ent in doc.ents]
#             results.append(entities)
#         return results

# see @link https://huggingface.co/flair/ner-multi
# class FlairNERModel(NERModel):
#     def __init__(self, model_name: str = 'ner'):
#         self.tagger = SequenceTagger.load(model_name)
    
#     def predict(self, text: List[str]) -> List[Dict[str, Any]]:
#         results = []
#         for sentence_text in text:
#             sentence = Sentence(sentence_text)
#             self.tagger.predict(sentence)
#             entities = [{"entity": entity.tag, "text": entity.text} for entity in sentence.get_spans('ner')]
#             results.append(entities)
#         return results

class NERModelFactory:
    """ 
    Factory class to create NERModel instances.
    """
    @staticmethod
    def create(model_name: str, device: str) -> NERModel:
        if model_name.startswith("dbmdz/bert-large-cased-finetuned-conll03-english"):
            return HuggingFaceNERModel(model_name, device)
        # elif model_name.startswith("en_core_web_sm"):
        #     return SpacyNERModel(model_name)
        # elif model_name.startswith("ner"):
        #     return FlairNERModel(model_name)
        else:
            raise ValueError("Model not supported.")

            
class NERUtilities:
    @staticmethod
    def get_generated_entities(
        entities: List[Dict[str, Any]],
        input_length_chars: torch.Tensor    
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """ 
        Get the generated entities from the NER model output.
        
        :param entities: List of entities predicted by the NER model.
        :type entities: List[Dict[str, Any]]
        :param input_length_chars: Length of the input text in characters.
            This is needed in order to only keep the entities generated after the input text.
        :type input_length_chars: torch.Tensor
        :return: Tuple containing only the first generated entity and all generated entities.
        :rtype: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        """
        new_entities = []
        first_new_entities = []
        for i, entities in enumerate(entities):
            entities_of_current_output = []
            for entity in entities:
                if entity["start"] > input_length_chars[i]:
                    entities_of_current_output.append(entity)
            
            # if the first of entities_of_current_output["entity"] does not start with a "B", remove it
            while (len(entities_of_current_output) > 0 and entities_of_current_output[0]["entity"][0] != "B"):
                entities_of_current_output = entities_of_current_output[1:]
            new_entities.append(entities_of_current_output)

        # keep track of first new entity
        for hyp in new_entities:
            first_entity = []
            for entity_idx, entity in enumerate(hyp):
                if entity_idx > 0 and entity["entity"][0] == "B":
                    break
                first_entity.append(entity)
            first_new_entities.append(first_entity)
        return first_new_entities, new_entities


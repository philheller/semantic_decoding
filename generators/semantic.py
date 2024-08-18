import torch
from typing import Any, Dict, List, Tuple
from ner_model import HuggingFaceNERModel, NERUtilities

class SemanticGenerator:
    def __init__(self, ner_models: List[str]):
        self.ner_models = ner_models
        # todo generation config here
        pass

    def generate(self):
        pass

    def _get_generated_entities(self, entities: List[Dict[str, Any]], input_length_chars: torch.Tensor) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        return new_entities, first_new_entities

    def _load_ner_models(self):
        # todo implement
        pass
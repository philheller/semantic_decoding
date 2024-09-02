from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)
from data_structures import SemanticData

# import spacy
# from flair.data import Sentence
# from flair.models import SequenceTagger

SemanticDataModelOutputType = List[List[Dict[str, Any]]]

class SemanticDataModel(ABC):
    def __init__(self, model_name: str, device: Optional[str]):
        self.model_name = model_name
        self.device = device
        
    @abstractmethod
    def predict(self, text: List[str]) -> SemanticDataModelOutputType:
        """
        Run NER prediction on a list of texts.
        
        :param text: List of texts to run NER prediction on.
        :type text: List[str]
        :return: List of dictionaries containing NER predictions.
        :rtype: List[Dict[str, Any]]
        """
        raise NotImplementedError("This is an abstract method.")
    
    @abstractmethod
    def get_generated_semantic_data(
        self,
        semantic_data_points: SemanticDataModelOutputType,
        input_length_chars: torch.Tensor,
        include_all: bool = False
    ) -> Tuple[SemanticDataModelOutputType, SemanticDataModelOutputType]:
        """ 
        Returns a tuple of two lists of entities. The first list contains the first generated entity
        after the input text, and the second list contains all generated entities after the input text.

        :param semantic_data_points: List of semantic data points.
        :type semantic_data_points: SemanticDataModelOutputType
        :param input_length_chars: Length of the input text in characters.
        :type input_length_chars: torch.Tensor
        :param include_all: Whether to include all semantic data points (not only those after the
            input_length_chars). This is not to be confused with the second Tuple output.
        :type include_all: bool
        :return: Tuple containing the first generated entity and all generated entities.
        :rtype: Tuple[SemanticDataModelOutputType, SemanticDataModelOutputType]
        """
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def merge_semantic_data(
        self,
        semantic_data_points: SemanticDataModelOutputType
    ) -> SemanticDataModelOutputType:
        """ 
        Merges the semantic data points belonging together but remains in same data structure
        of SemanticDataModelOutputType.
        """
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def to_generic_semantic_data(
        self,
        semantic_data_points: SemanticDataModelOutputType,
        unique_key: str
    ) -> List[List[Union[SemanticData, None]]]:
        """ 
        Returns a list of SemanticData objects from the semantic data predicted by
        the SemanticDataModel.
        """
        raise NotImplementedError("This is an abstract method.")

# see @link https://huggingface.co/lxyuan/span-marker-bert-base-multilingual-uncased-multinerd
class BIOModel(SemanticDataModel):
    def __init__(self, model_name: str, device="cpu"):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=device)
    
    def predict(self, text: List[str]) -> SemanticDataModelOutputType:
        return self.pipeline(text) # type: ignore

    def get_generated_semantic_data(
        self,
        semantic_data_points: SemanticDataModelOutputType,
        input_length_chars: torch.Tensor,
        include_all: bool = False
    ) -> Tuple[SemanticDataModelOutputType, SemanticDataModelOutputType]:
        new_semantic_data_points = []
        for i, sem_datas in enumerate(semantic_data_points):
            sem_data_of_current_output = []
            for sem_data in sem_datas:
                if include_all or sem_data["start"] > input_length_chars[i]:
                    sem_data_of_current_output.append(sem_data)
            
            # if the first of entities_of_current_output["entity"] does not start with a "B", remove it
            while (len(sem_data_of_current_output) > 0 and sem_data_of_current_output[0]["entity"][0] != "B"):
                sem_data_of_current_output = sem_data_of_current_output[1:]
            new_semantic_data_points.append(sem_data_of_current_output)

        first_new_semantic_data_points = []
        # keep track of first new data points
        for hyp in new_semantic_data_points:
            first_sem_data = []
            for sem_data_idx, sem_data in enumerate(hyp):
                if sem_data_idx > 0 and sem_data["entity"][0] == "B":
                    break
                first_sem_data.append(sem_data)
            first_new_semantic_data_points.append(first_sem_data)
        return first_new_semantic_data_points, new_semantic_data_points

    def merge_semantic_data(
        self,
        semantic_data: SemanticDataModelOutputType
    ) -> SemanticDataModelOutputType:
        list_of_semantic_data_sequences = []
        for hyp_sem_data in semantic_data:
            list_of_whole_sem_data = []
            amount_sem_data_chunks = len(hyp_sem_data)
            if amount_sem_data_chunks == 0:
                list_of_semantic_data_sequences.append(list_of_whole_sem_data)
                continue
            counter = 0
            counter_start = counter
            while counter <= amount_sem_data_chunks:
                if counter == amount_sem_data_chunks:
                    # all chunks from counter_start to counter are a whole entity
                    sem_data_chunks_belonging_together = hyp_sem_data[counter_start:counter]
                    merged_sem_data = self._merge_semantic_data_chunks(sem_data_chunks_belonging_together)
                    list_of_whole_sem_data.append(merged_sem_data)
                elif hyp_sem_data[counter]["entity"][0] == "B":
                    # all the chunks before this B are a whole entity (not if counter == 0)
                    if counter != 0:
                        sem_data_chunks_belonging_together = hyp_sem_data[counter_start:counter]
                        merged_sem_data = self._merge_semantic_data_chunks(sem_data_chunks_belonging_together) 
                        list_of_whole_sem_data.append(merged_sem_data)
                        counter_start = counter
                counter += 1
            list_of_semantic_data_sequences.append(list_of_whole_sem_data)
        return list_of_semantic_data_sequences

    def _merge_semantic_data_chunks(
        self,
        semantic_data_chunks_belonging_together: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        _type = self._extract_entity_type(
            semantic_data_chunks_belonging_together[0]["entity"]
        )
        start = semantic_data_chunks_belonging_together[0]["start"]
        end = semantic_data_chunks_belonging_together[-1]["end"]
        word = " ".join([sem_data["word"] for sem_data in semantic_data_chunks_belonging_together])
        
        return {
            "entity": _type,
            "word": word,
            "start": start,
            "end": end,
            "amount_of_entity_chunks": len(semantic_data_chunks_belonging_together)
        }
        
    def _extract_entity_type(
        self,
        entity: str
    ) -> str:
        return entity.split("-")[-1]
        
    def to_generic_semantic_data(
        self,
        semantic_data: SemanticDataModelOutputType,
        unique_key: str
    ) -> List[List[Union[SemanticData, None]]]:
        hyps = []
        for hyp in semantic_data:
            generic_sem_data = []
            for entire_sem_data_point in hyp:
                sem_dat = SemanticData(
                    entire_sem_data_point[unique_key],
                    entire_sem_data_point["start"],
                    entire_sem_data_point["end"],
                    entire_sem_data_point["entity"],
                    entire_sem_data_point["amount_of_entity_chunks"],
                    hyp
                )
                generic_sem_data.append(sem_dat)
            if len(generic_sem_data) == 0:
                generic_sem_data.append(None)
            hyps.append(generic_sem_data)
        return hyps

# todo:phil implement SpacyNERModel and FlairNERModel
# see @link https://huggingface.co/spacy/en_core_web_md and https://huggingface.co/spacy/en_core_web_trf
# see implementation @link https://spacy.io/
# class SpacyNERModel(SemanticDataModel):
# # is in iob notation
#     def __init__(self, model_name: str, device="cpu"):
#         if device == "cuda":
#             spacy.prefer_gpu()
#         self.nlp = spacy.load(model_name)
    
#     def predict(self, text: List[str]) -> NER_OutputType:
#         results = []
#         for doc in self.nlp.pipe(text):
#             entities = [{"entity": ent.label_, "text": ent.text} for ent in doc.ents]
#             results.append(entities)
#         return results

# see @link https://huggingface.co/flair/ner-multi
# class FlairNERModel(SemanticDataModel):
#     def __init__(self, model_name: str = 'ner', device="cpu"):
#         self.tagger = SequenceTagger.load(model_name)
    
#     def predict(self, text: List[str]) -> List[Dict[str, Any]]:
#         results = []
#         for sentence_text in text:
#             sentence = Sentence(sentence_text)
#             self.tagger.predict(sentence)
#             entities = [{"entity": entity.tag, "text": entity.text} for entity in sentence.get_spans('ner')]
#             results.append(entities)
#         return results

class SemanticModelFactory:
    """ 
    Factory class to create SemanticDataModel instances.
    """
    @staticmethod
    def create(model_name: Union[str, List[str]], device: str) -> List[SemanticDataModel]:
        ner_models = []
        if isinstance(model_name, list):
            for name in model_name:
                ner_models.append(SemanticModelFactory._create_singular(name, device))
            return ner_models
        else:
            return [SemanticModelFactory._create_singular(model_name, device)]

    @staticmethod
    def _create_singular(model_name: str, device: str) -> SemanticDataModel:
        hf_ner_models = ("lxyuan/", "dslim/")
        if model_name.startswith(hf_ner_models):
            return BIOModel(model_name, device)
        # elif model_name.startswith("en_core"):
        #     return SpacyNERModel(model_name)
        # elif model_name.startswith("ner"):
        #     return FlairNERModel(model_name)
        else:
            raise ValueError("Model not supported.")

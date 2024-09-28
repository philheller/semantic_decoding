from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Optional, Literal, Iterable
import spacy.tokens
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)
from data_structures import SemanticData
import spacy


SemanticDataModelOutputType = List[List[Dict[str, Any]]]

class SemanticDataModel(ABC):
    def __init__(
        self,
        model_name: str,
        device: Optional[str],
        normalize_unique_key: bool,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_unique_key = normalize_unique_key
        
    @abstractmethod
    def predict(self, text: List[str]) -> SemanticDataModelOutputType:
        """
        Run prediction on a list of texts.
        
        :param text: List of texts to run prediction on.
        :type text: List[str]
        :return: List of dictionaries containing predictions.
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
        Returns a tuple of two lists of SemanticDataModelOutputTyp. The first list contains the first
        generated semantic data point after the input text, and the second list contains
        all generated data points after the input text.

        :param semantic_data_points: List of semantic data points.
        :type semantic_data_points: SemanticDataModelOutputType
        :param input_length_chars: Length of the input text in characters.
        :type input_length_chars: torch.Tensor
        :param include_all: Whether to include all semantic data points (not only those after the
            input_length_chars). This is not to be confused with the second Tuple output.
        :type include_all: bool
        :return: Tuple containing the first generated semantic data points and all generated
            semantic data points.
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
        unique_key: Literal["text", "word", "type"],
        syntactic_sequences: Optional[torch.Tensor] = None,
        syntactic_eos_token_id: Optional[int] = None,
        semantic_eos_token: Optional[str] = None,
        other: Optional[List[List[Dict[str, Any]]]] = None
    ) -> List[List[Union[SemanticData, None]]]:
        """ 
        Returns a list of SemanticData objects from the semantic data predicted by
        the SemanticDataModel. Also includes an end of sentence token as semantic
        token if it is present as last token in the syntactc sequence.
        """
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def _select_with_unique_key(self, unique_key: str, semantic_data_point: Any) -> str:
        """ 
        Maps the unique key to the correct key for the selected model.
        """
        raise NotImplementedError("This is an abstract method.")

    def _normalize_unique_key(self, unique_key: str) -> str:
        """ 
        Normalize the unique key to be used in the SemanticData object.

        Normalization steps:
        1. Remove anything that is not a-z, A-Z, 0-9
        2. Make string lowercase
        """
        unique_key = "".join([c for c in unique_key if c.isalnum()])
        return unique_key.lower()
        

# see @link https://huggingface.co/lxyuan/span-marker-bert-base-multilingual-uncased-multinerd
class BIOModel(SemanticDataModel):
    def __init__(self, model_name: str, normalize_unique_key, device="cpu"):
        super().__init__(model_name, device, normalize_unique_key)
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
            
            # if the first of sem_data_of_current_output["entity"] does not start with a "B", remove it
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
        unique_key: str,
        syntactic_sequences: Optional[torch.Tensor] = None,
        syntactic_eos_token_id: Optional[int] = None,
        semantic_eos_token: Optional[str] = None,
        other: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[List[Union[SemanticData, None]]]:
        hyps = []
        can_find_eos = all([
            syntactic_sequences is not None,
            syntactic_eos_token_id is not None,
            semantic_eos_token is not None]
        )
        for hyp_idx, hyp in enumerate(semantic_data):
            generic_sem_data = []
            for entire_sem_data_point in hyp:
                selected_unique_key = self._select_with_unique_key(unique_key, entire_sem_data_point)
                if self.normalize_unique_key:
                    selected_unique_key = self._normalize_unique_key(selected_unique_key)
                sem_dat = SemanticData(
                    selected_unique_key,
                    entire_sem_data_point["word"],
                    entire_sem_data_point["start"],
                    entire_sem_data_point["end"],
                    entire_sem_data_point["entity"],
                    entire_sem_data_point["amount_of_entity_chunks"],
                    other[hyp_idx] if other is not None else hyp,
                )
                generic_sem_data.append(sem_dat)
            # check if synt eos token is last syt token in sequence
            if can_find_eos and syntactic_sequences[hyp_idx, -1] == syntactic_eos_token_id:
                eos_sem_data = SemanticData(
                        semantic_eos_token,
                        semantic_eos_token,
                        syntactic_sequences[hyp_idx].shape[-1],
                        syntactic_sequences[hyp_idx].shape[-1],
                        semantic_eos_token,
                        1,
                        None,
                        True,
                        is_eos_token=True
                    )
                generic_sem_data.append(eos_sem_data)
            if len(generic_sem_data) == 0:
                generic_sem_data.append(None)
            hyps.append(generic_sem_data)
        return hyps

    def _select_with_unique_key(self, unique_key: str, semantic_data_point: Dict[str, Any]) -> str:
        # map unique keys to correct keys for the selected bios model
        if unique_key in ("text", "word"):
            unique_key = "word"
        elif unique_key in ("type",):
            unique_key = "entity"
        else:
            raise ValueError(f"Unique key {unique_key} not supported for model {self.__class__.__name__}.")
        selected_unique_key = semantic_data_point[unique_key]
        return selected_unique_key


SemanticDataModelIterator = List[Iterable[spacy.tokens.Span]]
class SpacyModel(SemanticDataModel):
    def __init__(self, model_name: str, normalize_unique_key: bool):
        super().__init__(model_name, None, normalize_unique_key)
        self.spacy = spacy.load(model_name)
    
    def predict(
        self,
        text: List[str], 
        aggregation_type: Literal["ner", "noun_chunks"] = "noun_chunks"
    ) -> SemanticDataModelIterator:
        results = []
        
        docs = self.spacy.pipe(text)
        for doc in docs:
            if aggregation_type == "noun_chunks":
                noun_chunks = doc.noun_chunks
                results.append(noun_chunks)
            # todo could also be used (ner)
            elif aggregation_type == "ner": 
                entities = doc.ents
                results.append(entities)
            else:
                raise ValueError("Aggregation type not supported.")
        return results
    
    def get_generated_semantic_data(
        self,
        semantic_data_points: SemanticDataModelIterator,
        input_length_chars: torch.Tensor,
        include_all: bool = False
    ) -> Tuple[SemanticDataModelIterator, SemanticDataModelIterator]:
        new_semantic_data_points = []
        
        for hyp_idx, sem_datas in enumerate(semantic_data_points):
            sem_data_of_current_output = []
            for sem_data in sem_datas:
                if include_all or sem_data.start_char > input_length_chars[hyp_idx]:
                    sem_data_of_current_output.append(sem_data)
            new_semantic_data_points.append(sem_data_of_current_output)

        first_new_semantic_data_points = []
        for hyp in new_semantic_data_points:
            if len(hyp) == 0:
                first_new_semantic_data_points.append([])
                continue
            first_new_semantic_data_points.append([hyp[0]])

        return first_new_semantic_data_points, new_semantic_data_points

    def merge_semantic_data(
        self,
        semantic_data: SemanticDataModelIterator,
    ) -> SemanticDataModelIterator:
        # i pass-through function, merging not needed in spacy
        return semantic_data

    def to_generic_semantic_data(
        self,
        semantic_data: SemanticDataModelIterator,
        unique_key: str,
        syntactic_sequences: Optional[torch.Tensor] = None,
        syntactic_eos_token_id: Optional[int] = None,
        semantic_eos_token: Optional[str] = None,
        other: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[List[Union[SemanticData, None]]]:
        hyps = []
        can_find_eos = all([
            syntactic_sequences is not None,
            syntactic_eos_token_id is not None,
            semantic_eos_token is not None]
        )

        for hyp_idx, hyp in enumerate(semantic_data):
            generic_sem_data = []
            for entire_sem_data_point in hyp:
                selected_unique_key = self._select_with_unique_key(unique_key, entire_sem_data_point)
                if self.normalize_unique_key:
                    selected_unique_key = self._normalize_unique_key(selected_unique_key)
                sem_dat = SemanticData(
                    selected_unique_key,
                    entire_sem_data_point.text,
                    entire_sem_data_point.start_char,
                    entire_sem_data_point.end_char,
                    entire_sem_data_point.label_,
                    1,
                    None
                )
                generic_sem_data.append(sem_dat)
            # check if synt eos token is last syt token in sequence
            if can_find_eos and syntactic_sequences[hyp_idx, -1] == syntactic_eos_token_id:
                eos_sem_data = SemanticData(
                        semantic_eos_token,
                        semantic_eos_token,
                        syntactic_sequences[hyp_idx].shape[-1],
                        syntactic_sequences[hyp_idx].shape[-1],
                        semantic_eos_token,
                        1,
                        None,
                        True,
                        is_eos_token=True
                    )
                generic_sem_data.append(eos_sem_data)
            if len(generic_sem_data) == 0:
                generic_sem_data.append(None)
            hyps.append(generic_sem_data)
        return hyps

    def _select_with_unique_key(self, unique_key: str, semantic_data_point: spacy.tokens.Span) -> str:
        if unique_key in ("text",):
            selected_unique_key = semantic_data_point.text
        elif unique_key in ("word",):
            selected_unique_key = semantic_data_point.root.text
        # elif unique_key in ("type",):
        #     selected_unique_key = semantic_data_point.label_
        else:
            raise ValueError(f"Unique key `{unique_key}` not supported for model {self.__class__.__name__}.")
        return selected_unique_key


class SemanticModelFactory:
    """ 
    Factory class to create SemanticDataModel instances.
    """
    @staticmethod
    def create(
        model_name: Union[str, List[str]],
        device: str,
        normalize_unique_key: bool
    ) -> List[SemanticDataModel]:
        ner_models = []
        if isinstance(model_name, list):
            for name in model_name:
                ner_models.append(SemanticModelFactory._create_singular(name, device, normalize_unique_key))
            return ner_models
        else:
            return [SemanticModelFactory._create_singular(model_name, device, normalize_unique_key)]

    @staticmethod
    def _create_singular(model_name: str, device: str, normalize_unique_key: bool) -> SemanticDataModel:
        hf_ner_models = ("lxyuan/", "dslim/")
        spacy_models = ("en_core")
        if model_name.startswith(hf_ner_models):
            return BIOModel(model_name, normalize_unique_key, device)
        elif model_name.startswith(spacy_models):
            return SpacyModel(model_name, normalize_unique_key)
        else:
            raise ValueError("Model not supported.")

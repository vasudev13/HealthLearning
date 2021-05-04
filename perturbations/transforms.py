"""torchvision.transforms style augmentations for Clincal Text"""
import torch
import torch.nn.functional as F
import spacy
from scispacy.linking import EntityLinker
import random


__all__ = ["Compose", "ClinicalSynonymSubstitution"]


class Compose:
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, text):
        for t in self.transforms:
            text = t(text)
        return text

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ClinicalSynonymSubstitution(torch.nn.Module):
    """
        Substitutes clincal term with synonymous concepts from Universal Medical Language System
        For now using only `en_core_sci_sm` as UMLS is huge and memory requirements can get out of hand.
    """

    def __init__(self, p: float = 0.5, substitution_probability: float = 0.7, scispacy_entity_model="en_core_sci_sm"):
        super().__init__()
        self.nlp = spacy.load(scispacy_entity_model)
        self.nlp.add_pipe("scispacy_linker", config={
                          "resolve_abbreviations": True, "linker_name": "umls"})
        self.p = p
        self.substitution_probability = substitution_probability
        self.linker = self.nlp.get_pipe("scispacy_linker")

    def forward(self, text: str) -> str:
        augmented_text = str(text)
        if torch.rand(1).item() < self.p:
            doc = self.nlp(text)
            for entity in doc.ents:
                if torch.rand(1).item() < self.substitution_probability:
                    for umls_entity in entity._.kb_ents[:1]:
                        aliases = self.linker.kb.cui_to_entity[umls_entity[0]].aliases
                        if aliases:
                            alias = random.sample(aliases, 1)[0]
                            augmented_text = augmented_text.replace(
                                entity.text, alias)
        return augmented_text

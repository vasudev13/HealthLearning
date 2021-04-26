"""torchvision.transforms style augmentations but for text"""
import torch
import torch.nn.functional as F
from googletrans import Translator


__all__ = ["Compose", "BackTranslation"]


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


class BackTranslation(torch.nn.Module):
    """Translates text into another language and back to souce language


    When installing googletrans, please keep in mind:
    `pip uninstall googletrans`
    `pip install googletrans==3.1.0a0`
    """

    def __init__(self, source_lang: str, dest_lang: str, p: float = 0.5):
        self.source_lang = source_lang
        self.dest_lang = dest_lang
        self.translator = Translator()
        self.p = p

    def forward(self, text):
        if torch.rand(1).item() < self.p:
            text = self.translator.translate(text, dest=self.dest_lang).text
            text = self.translator.translate(text, dest=self.source_lang).text
        return text
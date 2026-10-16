"""The fixed label list behind the automatic tags (zero-shot, by text-image similarity).

Each label is an English tag, the sentence the model reads for it, and a few words in other
languages that a human may type in a tag filter (the model itself, SigLIP 2, is multilingual,
so a search sentence can be written in French, Spanish, German and so on). Changing the list or
a sentence bumps ``LABELS_VERSION`` through ``labels_signature``: entries are re-tagged from the
embeddings they already have, without decoding any video again.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# SigLIP's own scale and bias suit caption pairs, not a plain label against a real frame (they
# give probabilities near zero, or saturate for any indoor scene). Measured on real footage with
# SigLIP 2 Base: a label that is on screen scores a cosine of 0.11 to 0.14, one that is not
# 0.03 to 0.10. The match probability is centred and softened to that range.
TAG_CENTER = 0.11  # cosine at which a label counts as 50 % likely
TAG_SCALE = 60.0
AUTO_TAG_MIN = 0.5  # match probability above which a label becomes an automatic tag
AUTO_TAG_LIMIT = 10  # at most this many automatic tags per entry (the best ones)


@dataclass(frozen=True)
class Label:
    tag: str
    prompt: str
    aliases: tuple[str, ...] = ()


LABELS: tuple[Label, ...] = (
    Label("person", "a photo of a person", ("personne", "persona", "mensch")),
    Label("face close-up", "a close-up photo of a person's face", ("gros plan visage", "visage")),
    Label("hands", "a close-up photo of hands", ("mains", "manos", "hande")),
    Label(
        "product close-up",
        "a close-up photo of a product package held or placed in front of the camera",
        ("produit", "gros plan produit", "producto"),
    ),
    Label("bathroom", "a photo taken in a bathroom", ("salle de bain", "bano")),
    Label("kitchen", "a photo taken in a kitchen", ("cuisine", "cocina", "kuche")),
    Label("bedroom", "a photo taken in a bedroom", ("chambre", "dormitorio")),
    Label("living room", "a photo taken in a living room", ("salon", "sala")),
    Label("outdoors", "a photo taken outdoors", ("exterieur", "dehors", "exterior")),
    Label("street", "a photo of a city street", ("rue", "calle")),
    Label("car", "a photo taken inside a car", ("voiture", "coche")),
    Label("mirror", "a person looking at a mirror", ("miroir", "espejo", "spiegel")),
    Label("selfie", "a selfie taken with a phone", ("selfie",)),
    Label("text on screen", "a picture with a lot of written text on it", ("texte", "texto")),
    Label("diagram", "a diagram, chart or infographic", ("schema", "infographie")),
    Label("screen recording", "a screenshot of a phone or computer screen", ("capture d'ecran",)),
    Label("before and after", "a before and after comparison of two pictures", ("avant apres",)),
    Label("sad", "a person with a sad face", ("triste", "triste")),
    Label("smiling", "a person smiling", ("sourire", "souriant", "sonrisa")),
    Label("surprised", "a person with a surprised face", ("surpris", "sorprendido")),
    Label("worried", "a worried person", ("inquiet", "preocupado")),
    Label("touching face", "a person touching their own face or jaw", ("toucher visage",)),
    Label("applying cream", "a person applying a cream on their skin", ("creme", "appliquer")),
    Label("skin close-up", "a macro photo of human skin", ("peau", "piel", "haut")),
    Label("woman", "a photo of a woman", ("femme", "mujer", "frau")),
    Label("man", "a photo of a man", ("homme", "hombre", "mann")),
    Label("older person", "a photo of an elderly person", ("personne agee", "mayor")),
    Label("crowd", "a photo of a crowd of people", ("foule", "multitud")),
    Label("red carpet", "a celebrity on a red carpet", ("tapis rouge",)),
    Label("doctor", "a doctor or a medical professional", ("medecin", "docteur", "medico")),
    Label("food", "a photo of food", ("nourriture", "comida")),
    Label("animal", "a photo of an animal", ("animal", "tier")),
)


def label_prompts() -> list[str]:
    return [label.prompt for label in LABELS]


def labels_signature() -> str:
    """Changes when the list or any sentence changes."""

    lines = [f"{label.tag}|{label.prompt}" for label in LABELS]
    head = f"{TAG_CENTER}|{TAG_SCALE}|{AUTO_TAG_MIN}"
    text = "\n".join([head, *lines])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def resolve_tag(word: str) -> str | None:
    """The English tag a word stands for (the tag itself or one of its aliases), else ``None``."""

    folded = _fold(word)
    for label in LABELS:
        if folded == _fold(label.tag) or folded in {_fold(alias) for alias in label.aliases}:
            return label.tag
    return None


def _fold(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", value.casefold().strip())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


__all__ = [
    "AUTO_TAG_LIMIT",
    "AUTO_TAG_MIN",
    "LABELS",
    "TAG_CENTER",
    "TAG_SCALE",
    "Label",
    "label_prompts",
    "labels_signature",
    "resolve_tag",
]

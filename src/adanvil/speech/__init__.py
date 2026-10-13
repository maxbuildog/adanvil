"""Speech of the footage: transcripts with word timestamps, phrases, fillers, takes, captions.

Importing this package stays light on purpose (the library and the storyboard both use it):
import the submodules you need, for example ``adanvil.speech.transcribe``.
"""

from .model import SpeechData, SpeechWord

__all__ = ["SpeechData", "SpeechWord"]

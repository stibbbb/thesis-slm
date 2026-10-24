from slm_api import translate_signs, translate_speech, translate_speech_with_confidence
from vocab_pool import CATEGORIES
from onnx_config import GLOSS_MAP

PERSON_WORDS = {w for w, cat in CATEGORIES.items() if cat == "Person / Family"}

PERSON_ALIASES = {alias: canonical for alias, canonical in GLOSS_MAP.items()
                   if canonical in PERSON_WORDS}


def _canonical_person(word):
    """Returns the canonical person-word form if `word` refers to a
    person (directly, or via a GLOSS_MAP alias), else None."""
    if word in PERSON_WORDS:
        return word
    return PERSON_ALIASES.get(word)


class ConversationSession:
    """
    Stateful session wrapping the stateless slm_api functions.
    Implements the context_memory variable from Algorithm 1/2, plus
    one concrete rule-based use of that history: subject-ellipsis
    recovery for Direction 1 (signs -> sentence). Also surfaces a
    confidence signal for Direction 2 (speech -> signs), based on
    out-of-vocabulary word density.
    """

    def __init__(self):
        self.context_memory = []       # matches Algorithm 1/2's context_memory
        self._last_person_word = None  # most recently mentioned mother/father
        self.last_recovery = None      # set by sign_to_sentence(): the word
                                        # that was recovered on the most
                                        # recent call, or None if no
                                        # recovery was needed that turn
        self.last_confidence = None    # set by speech_to_signs(): the full
                                        # confidence dict from
                                        # translate_speech_with_confidence(),
                                        # or None before any speech turn

    def reset(self):
        """Start a new conversation -- clears all history."""
        self.context_memory = []
        self._last_person_word = None
        self.last_recovery = None
        self.last_confidence = None

    def _recover_subject(self, glosses):
        """
        If this turn's glosses contain no person-reference word and a
        previous turn established one, prepend it. Returns the
        (possibly modified) word list and whether recovery happened.
        Does nothing on a genuinely empty turn -- an empty input stays
        empty rather than fabricating content from history.
        """
        words = glosses.split() if isinstance(glosses, str) else list(glosses)
        if not words:
            return words, False
        has_person = any(_canonical_person(w) is not None for w in words)
        if not has_person and self._last_person_word is not None:
            return [self._last_person_word] + words, True
        return words, False

    def sign_to_sentence(self, glosses):
        """
        Direction 1 with context: recovers an omitted subject from
        recent history if needed, calls the existing stateless
        translate_signs(), then records this turn in context_memory
        and updates the tracked subject for future turns.
        """
        recovered_words, recovered = self._recover_subject(glosses)
        self.last_recovery = recovered_words[0] if recovered else None
        sentence = translate_signs(recovered_words)

        for w in recovered_words:
            canonical = _canonical_person(w)
            if canonical:
                self._last_person_word = canonical

        self.context_memory.append(sentence)
        return sentence

    def speech_to_signs(self, sentence):
        """
        Direction 2 with context: no ellipsis recovery applied here --
        speech-to-text transcripts are expected to already be complete
        sentences -- but this turn is still recorded in context_memory
        and its subject tracked, so a later Direction-1 turn can
        recover from it too. Also reports a confidence signal (see
        translate_speech_with_confidence()) so low-confidence turns can
        be flagged rather than silently trusted.
        """
        result = translate_speech_with_confidence(sentence)
        glosses = result["glosses"]
        self.last_confidence = result

        for w in glosses.split():
            if w in PERSON_WORDS:
                self._last_person_word = w

        self.context_memory.append(glosses)
        return glosses


if __name__ == "__main__":
    print("=== ConversationSession demo ===\n")
    session = ConversationSession()

    print("Turn 1  sign_to_sentence('mother drive')")
    out1 = session.sign_to_sentence("mother drive")
    print(f"        -> {out1!r}")
    print(f"        context_memory: {session.context_memory}\n")

    print("Turn 2  sign_to_sentence('where')   [no subject -- should recover 'mother']")
    out2 = session.sign_to_sentence("where")
    print(f"        -> {out2!r}")
    print(f"        recovered: {session.last_recovery!r}")
    print(f"        context_memory: {session.context_memory}\n")

    print("Turn 3  speech_to_signs('Can you help me?')   [high OOV -- should flag low confidence]")
    out3 = session.speech_to_signs("Can you help me?")
    print(f"        -> {out3!r}")
    print(f"        confidence: {session.last_confidence}\n")

    print("Resetting session...")
    session.reset()
    print(f"        context_memory after reset: {session.context_memory}")
    print(f"        tracked subject after reset: {session._last_person_word}")
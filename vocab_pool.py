# SLM Vocabulary Pool
# CONFIRMED_GESTURES: the words the recognition module is actually
#   trained to recognize -- the 44 signs confirmed by the recognition team.
# HELPER_WORDS: words NOT signed by the user. Two roles:
#   (a) grammatical connectors the SLM adds during sentence completion
#   (b) words that appear in a HEARING person's natural speech, which
#       Reduce must be able to SEE in order to learn whether to keep or
#       drop them. A word that is not in the vocabulary becomes a generic
#       unknown-token, and the model cannot learn a specific rule about a
#       word it cannot distinguish -- so filler words like "is" and "the"
#       must be present here precisely so the model can learn to DROP them.
#
# Used for both tasks:
#   1. Complete (Deaf-to-NonDeaf): signs in, English sentence out
#   2. Reduce   (NonDeaf-to-Deaf): English in, signs out

CONFIRMED_GESTURES = [
    # The 44 signs confirmed by the recognition team. These are the
    # ONLY words the camera can produce. Everything else in this file
    # is speech- or grammar-only.

    # Responses
    "yes", "good", "fine", "understand", "not",
    # Greetings & politeness ("thank" is the sign THANK YOU)
    "hello", "please", "sorry", "nice", "again", "thank",
    # Question words
    "what", "where", "why", "how",
    # Pronouns
    "you", "my",
    # Everyday needs
    "help", "eat", "water", "phone", "sleep", "go", "want",
    # People & identity
    "mother", "father", "family", "friend", "deaf", "teacher", "person",
    # The camera cannot tell MOTHER from FATHER, so the Deaf person's signs
    # report both as PARENT; "mother" and "father" stay for the hearing
    # person's words on the Deaf screen.
    "parent",
    # Places
    "home", "school", "class", "here",
    # School & learning
    "learn", "study", "sign",
    # Time
    "time", "now", "today", "tomorrow", "yesterday",
]

HELPER_WORDS = [
    # --- grammatical connectors
    "i", "your", "am", "is", "are", "going", "doing", "and", "at", "in",
    "he", "she", "it", "the", "to", "will", "can", "do", "does",
    "have", "for", "we",

    # --- past tense, for YESTERDAY ("i went to school yesterday")
    "was", "were", "did", "went", "ate", "studied", "learned", "slept", "signed",
    # --- for WANT with a person ("i want father to be here now",
    #     "teacher wants to learn")
    "be", "wants",
    # --- SIGN as a subject studied ("i am studying sign language")
    "language",

    # --- words a hearing person says that have NO sign in the 44.
    # Present so the model can learn to DROP them (see header note).
    "his", "her", "me", "who", "when", "need", "needs", "some",
    "tell", "know", "this", "that", "say", "come", "right", "meet",
    "there", "look",

    # --- "no" has no sign, but it must be SEEN so the meaning is not
    # inverted; it is shown as NOT (see word_picker.py).
    "no",

    # --- verb forms appearing in natural speech
    "studying", "learning", "eating", "sleeping", "signing",

    # --- informal equivalents (also in onnx_config.GLOSS_MAP)
    "mom", "dad", "house", "okay", "alright", "hi",

    # --- everyday filler the hearing person says
    "very", "really", "so", "still", "much",
]

# Combined master list (deduplicated, sorted)
ALL_VOCAB = sorted(set(CONFIRMED_GESTURES + HELPER_WORDS))

# Category of every sign. slm_pipeline.py uses these to know which words
# name a PERSON (they can be remembered and filled in later), which are
# questions, states, activities and places.
CATEGORIES = {
    # --- Person / Family: these can be recovered as a subject later
    "mother": "Person / Family",
    "father": "Person / Family",
    "teacher": "Person / Family",
    "family": "Person / Family",
    "friend": "Person / Family",
    "person": "Person / Family",
    "parent": "Person / Family",

    # --- Pronoun / Reference: deliberately NOT Person/Family. "you" is
    # the person being spoken to, who is always present.
    "my": "Pronoun / Reference",
    "you": "Pronoun / Reference",

    # --- Questions
    "what": "Basic Question",
    "where": "Basic Question",
    "why": "Basic Question",
    "how": "Basic Question",

    # --- States
    "fine": "Response / State",
    "good": "Response / State",
    "deaf": "Response / State",

    # --- Activities
    "study": "Daily Activity",
    "learn": "Daily Activity",
    "eat": "Daily Activity",
    "sleep": "Daily Activity",
    "go": "Daily Activity",
    "sign": "Daily Activity",

    # --- Places and things
    "home": "Location / Place",
    "school": "Location / Place",
    "class": "Location / Place",
    "here": "Location / Place",
    "water": "Object",
    "phone": "Object",

    # --- Time
    "time": "Time",
    "now": "Time",
    "today": "Time",
    "tomorrow": "Time",
    "yesterday": "Time",

    # --- Needs
    "help": "Need",
    "want": "Need",

    # --- Social
    "hello": "Social",
    "sorry": "Social",
    "please": "Social",
    "nice": "Social",
    "yes": "Social",
    "understand": "Social",
    "again": "Social",
    "thank": "Social",

    # --- Negation: shown and written by a rule, never learned (word_picker.py,
    # slm_pipeline.negate), so the meaning can never invert
    "not": "Negation",
}

assert len(CONFIRMED_GESTURES) == 44, len(CONFIRMED_GESTURES)
assert set(CATEGORIES) == set(CONFIRMED_GESTURES)

if __name__ == "__main__":
    print(f"Confirmed gesture words: {len(CONFIRMED_GESTURES)}")
    print(f"Helper words: {len(set(HELPER_WORDS) - set(CONFIRMED_GESTURES))}")
    print(f"Total vocabulary: {len(ALL_VOCAB)}")
    print()
    for word in ALL_VOCAB:
        tag = "GESTURE" if word in CONFIRMED_GESTURES else "helper"
        print(f"  [{tag}] {word}")

# SLM Vocabulary Pool
# CONFIRMED_GESTURES: the words the recognition module is actually
#   trained to recognize -- the 34 signs confirmed by the recognition team.
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
    # The 34 signs confirmed by the recognition team. These are the
    # ONLY words the camera can produce. Everything else in this file
    # is speech- or grammar-only.

    # Responses
    "yes", "good", "fine", "understand",
    # Greetings & politeness
    "hello", "please", "sorry", "nice", "again",
    # Question words
    "what", "where", "why", "how",
    # Pronouns
    "you", "my",
    # Everyday needs
    "help", "eat", "water",
    # People & identity
    "mother", "father", "family", "friend", "deaf", "teacher",
    # Places
    "home", "school", "class", "here",
    # School & learning
    "learn", "study",
    # Time
    "time", "now", "today", "tomorrow",
]

HELPER_WORDS = [
    # --- grammatical connectors
    "i", "your", "am", "is", "are", "going", "doing", "and", "at", "in",
    "he", "she", "it", "the", "to", "will", "can", "do", "does", "go",
    "have", "for", "we",

    # --- words a hearing person says that have NO sign in the 34.
    # Present so the model can learn to DROP them (see header note).
    "his", "her", "me", "who", "when", "want", "need", "needs", "some",
    "tell", "know", "this", "that", "say", "come", "right", "meet",
    "there", "look",

    # --- negation. There is no sign for it, but it must be SEEN so the
    # meaning is not inverted ("i do not understand"); see word_picker.py.
    "not", "no",

    # --- verb forms appearing in natural speech
    "studying", "learning", "eating",

    # --- informal equivalents (also in onnx_config.GLOSS_MAP)
    "mom", "dad", "house", "okay", "alright", "hi",

    # --- everyday filler the hearing person says
    "very", "really", "so", "still",
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

    # --- Places and things
    "home": "Location / Place",
    "school": "Location / Place",
    "class": "Location / Place",
    "here": "Location / Place",
    "water": "Object",

    # --- Time
    "time": "Time",
    "now": "Time",
    "today": "Time",
    "tomorrow": "Time",

    # --- Needs
    "help": "Need",

    # --- Social
    "hello": "Social",
    "sorry": "Social",
    "please": "Social",
    "nice": "Social",
    "yes": "Social",
    "understand": "Social",
    "again": "Social",
}

assert len(CONFIRMED_GESTURES) == 34, len(CONFIRMED_GESTURES)
assert set(CATEGORIES) == set(CONFIRMED_GESTURES)

if __name__ == "__main__":
    print(f"Confirmed gesture words: {len(CONFIRMED_GESTURES)}")
    print(f"Helper words: {len(set(HELPER_WORDS) - set(CONFIRMED_GESTURES))}")
    print(f"Total vocabulary: {len(ALL_VOCAB)}")
    print()
    for word in ALL_VOCAB:
        tag = "GESTURE" if word in CONFIRMED_GESTURES else "helper"
        print(f"  [{tag}] {word}")

# SLM Vocabulary Pool
# CONFIRMED_GESTURES: the words the recognition module is actually
#   trained to recognize. Source: [Dalumpines et al.] Pipeline
#   Reference, Section 3, plus the recognition team's Phase 4 list.
# HELPER_WORDS: words NOT signed by the user. Two roles:
#   (a) grammatical connectors the SLM adds during sentence completion
#   (b) words that appear in a HEARING person's natural speech, which
#       Reduce must be able to SEE in order to learn whether to keep or
#       drop them. A word that is not in the vocabulary becomes a generic
#       unknown-token, and the model cannot learn a specific rule about a
#       word it cannot distinguish -- so filler words like "my" and "the"
#       must be present here precisely so the model can learn to DROP them.
#
# Used for all three tasks:
#   1. Restructure (Deaf-to-NonDeaf, step A)
#   2. Complete    (Deaf-to-NonDeaf, step B)
#   3. Reduce      (NonDeaf-to-Deaf)

CONFIRMED_GESTURES = [
    # The 39 signs confirmed by the recognition team, grouped as they
    # were supplied. These are the ONLY words the camera can produce.
    # Everything else in this file is speech- or grammar-only.

    # Responses / glue
    "yes", "no", "good", "fine", "understand", "finish",
    # Greetings & politeness
    "hello", "please", "sorry", "nice",
    # Question words
    "what", "where", "why", "how", "who", "when",
    # Pronouns (his/her retained for possessive use: "her mother")
    "you", "my", "his", "her",
    # Everyday needs
    "want", "help", "eat", "water", "bathroom", "home",
    # Feelings / states
    "tired", "sick", "sad",
    # People & identity
    "mother", "father", "family", "friend", "deaf", "teacher",
    # School & learning
    "school", "learn", "study",
    # Other activities
    "drive", "laundry",
]

HELPER_WORDS = [
    # --- original grammatical connectors
    "i",
    "you",
    "your",
    "am",
    "is",
    "are",
    "going",
    "doing",
    "and",
    "at",
    "driving",
    "he",
    "she",

    # --- referential pronouns.
    # Added so Reduce can preserve WHO is being referred to. Directly
    # addresses the adviser's scenario: with two Deaf people present,
    # "what is his name?" reducing to just "name" loses the critical
    # information about whose name is being asked.
    "his",
    "her",
    "me",
    "my",

    # --- filler words a hearing person naturally says.
    # Present so the model can learn to DROP them (see header note).
    "the",
    "to",
    "please",
    "today",
    "tell",
    "know",
    "will",
    "can",
    "want",
    "this",

    # --- verb forms appearing in natural speech
    "do",
    "does",
    "drives",
    "go",
    "studying",   # PHASE 4: mirrors drive -> driving

    # --- informal equivalents appearing in natural speech.
    # NOTE: these are also in onnx_config.GLOSS_MAP, which handles the
    # gesture-input direction. They are needed HERE as well because
    # Reduce receives raw speech, which GLOSS_MAP does not preprocess.
    "mom",
    "dad",
    "house",
    "okay",
    "alright",

    # --- PHASE 5: inflected forms and connectives that appear in the
    # hearing person's speech but are never signed.
    "eating",
    "learning",
    "finished",
    "need",
    "needs",
    "meet",
    "there",
    "it",
    "not",

    # --- PHASE 5b: everyday filler the hearing person says. Present so
    # the model can learn to DROP them, same reason as "my" and "the".
    "very",
    "really",
    "so",
    "still",
    "now",
    "hi",
    "done",
    "look",
]

# Combined master list (deduplicated, sorted)
ALL_VOCAB = sorted(set(CONFIRMED_GESTURES + HELPER_WORDS))

# Category mapping, mirrors the group leader's reference document
# (Section 3 / Section 4 of the Pipeline Reference)
CATEGORIES = {
    # Every one of the 40 signs is categorised. conversation_session.py
    # uses this to know which words name a PERSON, because those are the
    # ones it can recover as the subject of a later bare gesture
    # ("where?" after "mother" was mentioned means "where is mother?").
    # A sign missing from here is invisible to that feature.

    # --- Person / Family: these can be recovered as a subject later
    "mother": "Person / Family",
    "father": "Person / Family",
    "teacher": "Person / Family",
    "family": "Person / Family",
    "friend": "Person / Family",

    # --- Pronoun / Reference: deliberately NOT Person/Family.
    # "his" refers BACK to someone already established rather than
    # establishing a new subject, and "you" is the person being spoken
    # to, who is always present. Treating either as a recoverable
    # subject would make context recovery behave incorrectly.
    "his": "Pronoun / Reference",
    "her": "Pronoun / Reference",
    "my": "Pronoun / Reference",
    "you": "Pronoun / Reference",

    # --- Questions
    "what": "Basic Question",
    "where": "Basic Question",
    "why": "Basic Question",
    "how": "Basic Question",
    "who": "Basic Question",
    "when": "Basic Question",

    # --- States
    "fine": "Response / State",
    "sad": "Response / State",
    "tired": "Response / State",
    "sick": "Response / State",
    "good": "Response / State",
    "deaf": "Response / State",

    # --- Activities
    "drive": "Daily Activity",
    "laundry": "Daily Activity",
    "study": "Daily Activity",
    "learn": "Daily Activity",
    "eat": "Daily Activity",

    # --- Places and things
    "home": "Location / Place",
    "school": "Location / Place",
    "bathroom": "Location / Place",
    "water": "Object",

    # --- Needs
    "want": "Need",
    "help": "Need",

    # --- Social
    "hello": "Social",
    "sorry": "Social",
    "please": "Social",
    "nice": "Social",
    "yes": "Social",
    "no": "Social",
    "understand": "Social",
    "finish": "Social",
}

if __name__ == "__main__":
    print(f"Confirmed gesture words: {len(CONFIRMED_GESTURES)}")
    print(f"Helper words: {len(HELPER_WORDS)}")
    print(f"Total vocabulary: {len(ALL_VOCAB)}")
    print()
    for word in ALL_VOCAB:
        tag = "GESTURE" if word in CONFIRMED_GESTURES else "helper"
        print(f"  [{tag}] {word}")
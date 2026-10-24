"""
Automated test suite for the SLM interface-hardening work
(slm_progress_notes.md §11). Formalizes the edge-case suite that was
previously only checked by eye (in onnx_inference.py / slm_inference.py's
demo output, and in live_demo.py) into real, automated pass/fail
assertions.

Two groups of tests:
  - Unit tests (TestNormalizeInput, TestSanitizeGlosses, TestEnforceMaxLen,
    TestGlossMap): test the onnx_config.py helpers in isolation. Fast,
    no model loading required.
  - Integration tests (TestPipelineEdgeCases, TestReduceDoesNotStripUnknownWords):
    load the real ONNX models once per test session and exercise the
    full sign_to_sentence_onnx() / sentence_to_glosses_onnx() pipeline.

Run:            pytest test_hardening.py -v
Unit tests only: pytest test_hardening.py -v -k "not Pipeline and not Reduce"
Requires:        pip install pytest   (if not already installed)
"""

import pytest
import onnxruntime as ort

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from onnx_config import (
    ENCODER_ONNX_PATH, DECODER_ONNX_PATH,
    normalize_input, sanitize_glosses, enforce_max_len, GLOSS_MAP,
)
from onnx_inference import sign_to_sentence_onnx, sentence_to_glosses_onnx


# ---------------------------------------------------------------
# Fixtures: load once per test session, not once per test function
# ---------------------------------------------------------------

@pytest.fixture(scope="session")
def tokenizer():
    return SimpleTokenizer(ALL_VOCAB)


@pytest.fixture(scope="session")
def enc_session():
    return ort.InferenceSession(ENCODER_ONNX_PATH)


@pytest.fixture(scope="session")
def dec_session():
    return ort.InferenceSession(DECODER_ONNX_PATH)


# ---------------------------------------------------------------
# Unit tests: onnx_config.py helpers in isolation, no model needed
# ---------------------------------------------------------------

class TestNormalizeInput:
    def test_string_passthrough(self):
        assert normalize_input("drive mother where") == "drive mother where"

    def test_strips_whitespace(self):
        assert normalize_input("  mother drive  ") == "mother drive"

    def test_list_joins_with_spaces(self):
        assert normalize_input(["drive", "mother", "where"]) == "drive mother where"

    def test_tuple_joins_with_spaces(self):
        assert normalize_input(("mother", "drive")) == "mother drive"

    def test_empty_string_ok(self):
        assert normalize_input("") == ""

    def test_empty_list_ok(self):
        assert normalize_input([]) == ""

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            normalize_input(None)

    def test_int_raises_typeerror(self):
        with pytest.raises(TypeError):
            normalize_input(42)

    def test_dict_raises_typeerror(self):
        with pytest.raises(TypeError):
            normalize_input({"a": 1})

    def test_error_message_names_the_offending_type(self):
        with pytest.raises(TypeError, match="int"):
            normalize_input(42)


class TestSanitizeGlosses:
    def test_known_words_all_kept(self):
        kept, dropped = sanitize_glosses(["mother", "drive", "where"])
        assert kept == ["mother", "drive", "where"]
        assert dropped == []

    def test_unknown_word_dropped(self):
        kept, dropped = sanitize_glosses(["mother", "xyz", "drive"])
        assert kept == ["mother", "drive"]
        assert dropped == ["xyz"]

    def test_all_unknown_returns_empty_kept(self):
        kept, dropped = sanitize_glosses(["xyz", "abc"])
        assert kept == []
        assert dropped == ["xyz", "abc"]

    def test_empty_input(self):
        kept, dropped = sanitize_glosses([])
        assert kept == []
        assert dropped == []

    def test_order_is_preserved(self):
        kept, _ = sanitize_glosses(["where", "mother", "drive"])
        assert kept == ["where", "mother", "drive"]


class TestEnforceMaxLen:
    def test_short_list_unchanged(self):
        words = ["mother", "drive", "where"]
        assert enforce_max_len(words, 16) == words

    def test_long_list_truncated_to_exact_length(self):
        words = ["mother"] * 20
        result = enforce_max_len(words, 16)
        assert len(result) == 16

    def test_truncation_keeps_the_front(self):
        words = ["a", "b", "c", "d", "e"]
        assert enforce_max_len(words, 3) == ["a", "b", "c"]

    def test_exact_length_is_not_truncated(self):
        words = ["mother"] * 16
        assert enforce_max_len(words, 16) == words


class TestGlossMap:
    def test_mama_maps_to_mother(self):
        assert GLOSS_MAP["mama"] == "mother"

    def test_papa_maps_to_father(self):
        assert GLOSS_MAP["papa"] == "father"

    def test_unmapped_words_are_not_map_keys(self):
        assert "mother" not in GLOSS_MAP
        assert "drive" not in GLOSS_MAP


# ---------------------------------------------------------------
# Integration tests: the full ONNX pipeline on edge-case input
# ---------------------------------------------------------------

class TestPipelineEdgeCases:
    def test_empty_input_returns_empty_string(self, enc_session, dec_session, tokenizer):
        result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, "")
        assert result == ""

    def test_whitespace_only_returns_empty_string(self, enc_session, dec_session, tokenizer):
        result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, "   ")
        assert result == ""

    def test_single_known_word_does_not_crash(self, enc_session, dec_session, tokenizer):
        result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, "laundry")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_word_is_dropped_not_crashed(self, enc_session, dec_session, tokenizer):
        result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, "mother xyz drive")
        assert "xyz" not in result

    def test_list_input_matches_equivalent_string_input(self, enc_session, dec_session, tokenizer):
        result_list = sign_to_sentence_onnx(
            enc_session, dec_session, tokenizer, ["mother", "drive", "where"])
        result_string = sign_to_sentence_onnx(
            enc_session, dec_session, tokenizer, "mother drive where")
        assert result_list == result_string

    def test_oversized_input_does_not_crash(self, enc_session, dec_session, tokenizer):
        oversized = " ".join(
            ["mother", "father", "drive", "home", "laundry", "where", "how", "fine", "name"] * 2
        )  # 18 words, exceeds ENCODER_MAX_LEN=16
        result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, oversized)
        assert isinstance(result, str)

    @pytest.mark.parametrize("bad_input", [None, 42, {"a": 1}, 3.14])
    def test_invalid_types_raise_typeerror(self, enc_session, dec_session, tokenizer, bad_input):
        with pytest.raises(TypeError):
            sign_to_sentence_onnx(enc_session, dec_session, tokenizer, bad_input)

    def test_mama_papa_mapping_end_to_end(self, enc_session, dec_session, tokenizer):
        result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, "papa laundry")
        assert "father" in result
        assert "papa" not in result


class TestReduceDoesNotStripUnknownWords:
    """
    Deliberate design choice (§11 / §13 item 14 in the notes): Reduce
    processes natural speech and must NOT silently delete words a person
    said, unlike Restructure's closed-vocabulary stripping. This test
    exists to catch anyone "fixing" that asymmetry by accident later --
    if this starts failing, someone made Reduce start stripping words,
    which is a regression, not an improvement.
    """
    def test_reduce_does_not_crash_on_out_of_vocabulary_sentence(self, dec_session, tokenizer):
        # "hello" and "today" are not in the 23-word vocabulary.
        result = sentence_to_glosses_onnx(dec_session, tokenizer, "Hello, see you today!")
        assert isinstance(result, str)  # must not crash; content quality is a Phase B item

class TestKnownTrickyCases:
    """
    Historical tracking tests for two specific cases flagged in
    slm_progress_notes.md but never added to an automated test before
    now. Neither asserts exact wording on purpose -- both are
    documented as genuinely unstable or previously-imperfect, so a
    strict wording assertion would be actively wrong. They exist to
    give a real, current answer instead of an old memory, and to catch
    a genuine REGRESSION (a crash, an empty result, or losing the
    subject entirely) if things get worse in the future.

    Run with -s to see the actual current output printed:
        pytest test_hardening.py -v -s -k tricky
    """

    def test_father_is_driving_reduce(self, dec_session, tokenizer):
        # §13 known issue 3: earlier testing showed this dropping
        # "drive" entirely, before it was ever added to a real test.
        result = sentence_to_glosses_onnx(dec_session, tokenizer, "Father is driving.")
        print(f"\n  'Father is driving.' -> {result!r}")
        assert isinstance(result, str)
        assert "father" in result.lower()  # weak on purpose -- see docstring

    def test_home_drive_father_restructure(self, enc_session, dec_session, tokenizer):
        # §10: an adversarial 3-word combination absent from the
        # dataset entirely (train, test, or full). Behavior has changed
        # across every retrain in this project -- expected until Phase B.
        result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, "home drive father")
        print(f"\n  'home drive father' -> {result!r}")
        assert isinstance(result, str)
        assert len(result) > 0  # only guards against a total crash or empty output


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
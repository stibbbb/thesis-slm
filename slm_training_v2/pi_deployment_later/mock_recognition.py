"""
mock_recognition.py -- A MINIMAL, FAKE stand-in for the real gesture-
recognition module, built only to demonstrate the buffering pattern
described in INTEGRATION_CONTRACT.md. This contains NO real computer
vision -- it simulates a stream of per-frame gesture detections (most
frames have no new sign) and shows how to buffer them into a complete
utterance before calling the SLM.

VERBOSE MODE (default here): shows all three pipeline stages
separately on every flush --
    gloss mapping (mama/papa -> mother/father)
    Restructure   (DistilBERT -- reorders the glosses)
    Complete      (TinyLlama  -- writes the full sentence)
-- instead of hiding them behind one combined call, since Restructure
and Complete are two genuinely separate model calls, not one step
wearing two hats.

This is a reference implementation of the CALLING side of the
contract for Veron to look at (and reuse the buffering logic from, if
useful) -- not a preview of the real recognition module itself.

Run: python mock_recognition.py
"""

from slm_api import translate_signs_verbose

# Simulated per-frame detections. None = no gesture detected that
# frame (the normal case -- most frames have nothing new). A real
# recognition module would produce this stream continuously from the
# CNN-LSTM classifier's output; here it's hardcoded purely to
# demonstrate the buffering pattern with two test utterances.
SIMULATED_FRAME_STREAM = [
    None, None, "drive", None, None, "mother", None, None, None,
    "where", None, None, None, None, None,          # 5 empty frames -> flush
    None, "papa", None, "laundry", None, None, None, None, None,  # flush
]

PAUSE_FRAMES_TO_FLUSH = 5  # how many empty frames in a row = "end of utterance"


def _flush(buffer, verbose):
    """Runs the buffered words through the SLM and prints the result.
    Separated out since both the mid-stream and end-of-stream flush
    points need to do exactly the same thing."""
    utterance = " ".join(buffer)
    print(f"    -> sending {utterance!r} to the SLM")
    result = translate_signs_verbose(utterance)
    if verbose:
        if result["mapped"] != result["input"]:
            print(f"       gloss mapping (mama/papa): {result['mapped']!r}")
        print(f"       RESTRUCTURE (DistilBERT):  {result['restructured']!r}")
        print(f"       COMPLETE    (TinyLlama):   {result['sentence']!r}\n")
    else:
        print(f"       -> SLM says: {result['sentence']!r}\n")
    return result["sentence"]


def simulate_recognition_stream(frame_stream, pause_frames=PAUSE_FRAMES_TO_FLUSH, verbose=True):
    """
    Buffers individual per-frame gesture detections into complete
    utterances, flushing (calling the SLM) after a run of empty
    frames -- exactly the pattern described in
    INTEGRATION_CONTRACT.md's "Timing" section. That section was
    previously only prose; this is the same idea as runnable code.
    """
    buffer = []
    empty_run = 0

    for frame_num, detection in enumerate(frame_stream):
        if detection is not None:
            buffer.append(detection)
            empty_run = 0
            print(f"  frame {frame_num:2d}: detected {detection!r:10s} "
                  f"(buffer so far: {buffer})")
        else:
            empty_run += 1

        if buffer and empty_run >= pause_frames:
            print(f"  frame {frame_num:2d}: pause detected -- flushing buffer")
            _flush(buffer, verbose)
            buffer = []
            empty_run = 0

    if buffer:  # stream ended mid-utterance -- flush whatever's left
        print(f"  stream ended -- flushing remaining buffer")
        _flush(buffer, verbose)


if __name__ == "__main__":
    print("=== Mock recognition stream -> SLM demo (verbose) ===")
    print("(NOT real computer vision -- a hardcoded stand-in for the")
    print(" per-frame detection stream, to demonstrate the buffering")
    print(" pattern described in INTEGRATION_CONTRACT.md)\n")
    simulate_recognition_stream(SIMULATED_FRAME_STREAM, verbose=True)
    print("Done.")
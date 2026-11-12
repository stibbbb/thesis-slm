"""
latency_log.py -- saves every turn of the SLM to a CSV file, with the time
each pipeline box took, so the system's speed can be proven with logs.

Nothing is shown on screen (printing the times would slow the system
down). One file is written per run of the program, in the logs folder
next to full_demo.py:

    logs/latency_2026-11-12_14-30-05.csv

One line per turn: what came in (and from where), what was shown on the
reader's screen, the milliseconds of every box, and the total. A box the
turn did not use is left empty. The same file is the record of actual
testing: the last column is left empty, to mark each turn correct or not
afterwards. It opens directly in Excel.

Safe while the system runs:
- If the file is open in Excel (which locks it), the line is kept and
  written with the next turn once the file is closed; the system never
  stops because of the log.
- A file name that already exists (for example when the Raspberry Pi's
  clock was reset without internet) gets _2, _3 ..., so an old log is
  never added to.

The pipeline writes the log when it is created with a log folder:
    SLMPipeline(log_dir=...)
full_demo.py does this; the accuracy test (Conversation_test.py) does not.
"""

import csv
import os
from datetime import datetime

# the timed boxes of both paths; a box the path does not use is left empty
BOX_COLUMNS = ["Speech to text", "Normalization", "Reduction", "Tokenize", "Embedding",
               "Contextualize", "Space Restriction", "Active Context",
               "Sentence Completion", "Transcribed Words", "Display"]

HEADER = (["date", "time", "turn", "direction", "input from", "input as received",
           "unsure signs (dropped)", "output on screen", "rules applied", "topic after",
           "voice recording s"]
          + [f"{box} ms" for box in BOX_COLUMNS]
          + ["processing total ms", "correct? (mark later)"])


class LatencyLog:
    """Appends one CSV line per turn to <log_dir>/latency_<start time>.csv."""

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        name = f"latency_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        self.path = os.path.join(log_dir, name + ".csv")
        n = 2
        while os.path.exists(self.path):          # never add to an older log
            self.path = os.path.join(log_dir, f"{name}_{n}.csv")
            n += 1
        self.turns = 0
        self.pending = []                         # lines not written yet (file was locked)
        self.warned = False

    def write(self, turn, topic=None):
        """Add one finished Turn to the file (the file is made on the first turn)."""
        self.turns += 1
        ms = {}
        for (box, _), took in zip(turn.steps, turn.times):
            ms[box] = ms.get(box, 0.0) + took
        recording = ms.pop("Voice Recording", None)
        if turn.source != "microphone":
            ms.pop("Speech to text", None)        # typed input: Vosk did not run
        now = datetime.now()
        row = [f"{now:%Y-%m-%d}", f"{now:%H:%M:%S}", self.turns, turn.direction,
               turn.source, turn.received, ", ".join(turn.unsure), turn.output,
               "; ".join(turn.notes), topic or "neutral",
               f"{recording / 1000:.2f}" if turn.source == "microphone" and recording else ""]
        row += [f"{ms[box]:.2f}" if box in ms else "" for box in BOX_COLUMNS]
        row += [f"{sum(ms.values()):.2f}", ""]
        self.pending.append(row)
        self._flush()

    def _flush(self):
        """Write every waiting line; if the file is locked, keep them for the next turn."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            new_file = not os.path.exists(self.path)
            # utf-8-sig so Excel shows every character; written per turn so a crash loses nothing
            with open(self.path, "a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(HEADER)
                w.writerows(self.pending)
            self.pending = []
        except OSError:
            if not self.warned:
                print(f"  [log] {self.path} is open in another program; "
                      "the turns are kept and saved when it is closed")
                self.warned = True

    def close(self):
        """At the end of a run: save what is still waiting. If the file is
        still locked, the lines go to <name>_unsaved.csv instead of being lost."""
        if not self.pending:
            return
        self._flush()
        if self.pending:
            path = self.path
            self.path = path[:-4] + "_unsaved.csv"
            self._flush()
            if not self.pending:
                print(f"  [log] {path} was still open; the last turns are in {self.path}")

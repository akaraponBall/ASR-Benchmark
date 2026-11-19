"""Run the speaker-diarized LFM3 benchmark against local Llama.app."""

import sys

from asr_bench.cli import app

if __name__ == "__main__":
    app(["run", "--model", "lfm3", *sys.argv[1:]])

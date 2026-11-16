"""Small CLI; model calls live in adapters, experiment policy lives in runner."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .schema import ASRConfig, Settings

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
console = Console()


@app.command()
def models(config: Path = Path("models.toml")):
    """List configured ASR models. Every pipeline includes speaker detection."""
    settings = Settings.load(config)
    for name, model in settings.asr.items():
        mode = f"{model.latency_ms} ms" if isinstance(model, ASRConfig) else "diarized turns, offline ASR"
        console.print(f"{name}: {model.repo} ({mode})")
    console.print(f"Speaker detector: {settings.diarization.repo}")


@app.command()
def fetch(model: Annotated[list[str], typer.Option("--model", "-m")], config: Path = Path("models.toml")):
    """Download pinned checkpoints once, before benchmarking."""
    from .adapters import resolve_checkpoint

    settings = Settings.load(config)
    for name in model:
        if name not in settings.asr:
            raise typer.BadParameter(f"unknown model {name!r}")
    for name in model:
        console.print(resolve_checkpoint(settings.asr[name], offline=False))
    console.print(resolve_checkpoint(settings.diarization, offline=False))


@app.command()
def run(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    model: Annotated[list[str], typer.Option("--model", "-m")],
    out: Annotated[Path, typer.Option(help="New result directory; existing directories are refused")],
    config: Path = Path("models.toml"),
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda")] = "auto",
    repeat: Annotated[int, typer.Option(min=1)] = 1,
    warmup: Annotated[int, typer.Option(min=0)] = 0,
    threads: Annotated[int, typer.Option(min=1)] = 8,
    offline: bool = False,
):
    """Benchmark complete ASR + diarization pipelines on a JSONL manifest."""
    from .runner import run_benchmark

    if device not in {"auto", "cpu", "cuda"}:
        raise typer.BadParameter("device must be auto, cpu, or cuda")
    summary = run_benchmark(manifest, config, model, out, device, repeat, warmup, offline, threads)
    table = Table("Model", "RTF ↓", "CER ↓", "WER ↓", "DER ↓")
    for row in summary:
        table.add_row(
            row["model"],
            f"{row['rtf']:.3f}",
            *[f"{row[k]:.2%}" if row[k] is not None else "no reference" for k in ("cer", "wer", "der")],
        )
    console.print(table)
    console.print(f"Results: {out.resolve()}")
    console.print("Readable speaker transcripts: the .txt files in this results directory.")


if __name__ == "__main__":
    app()

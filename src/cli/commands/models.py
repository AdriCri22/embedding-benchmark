"""
Model management commands.
"""
import typer
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

from embedding_benchmark.models.registry import get_registry
from embedding_benchmark.models.hf_repos import get_hf_repo_for_download
from embedding_benchmark.utils import check_gguf_exists


def list_models(
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Filter by preset"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    enabled_only: bool = typer.Option(True, "--enabled/--all", help="Show only enabled models"),
):
    """
    List available models for benchmarking.
    """
    registry = get_registry()
    
    if preset:
        models = registry.get_preset(preset)
    elif tag:
        models = registry.get_by_tags(tag)
    else:
        models = registry.get_all(enabled_only)
    
    typer.echo(f"\nAvailable Models ({len(models)}):")
    typer.echo("-" * 100)
    for m in models:
        status = "✅" if m.enabled else "❌"
        tags = f"[{', '.join(m.tags)}]" if m.tags else ""
        typer.echo(f"  {status} {m.name:<40} {m.type:<12} {m.dim:<6} dim  {tags}")


def list_presets():
    """
    List available model presets.
    """
    registry = get_registry()
    typer.echo("\nAvailable Presets:")
    for name in registry.list_presets():
        models = registry.get_preset(name)
        typer.echo(f"  {name}: {len(models)} models - {', '.join(m.name for m in models)}")


def download_models(
    models: Optional[str] = typer.Option(
        None,
        "--models", "-m",
        help="Comma-separated list of models to download (default: all from full preset)",
    ),
    preset: str = typer.Option(
        "full",
        "--preset", "-p",
        help="Model preset to download",
    ),
    output_dir: str = typer.Option(
        "models",
        "--output", "-o",
        help="Output directory for models",
    ),
):
    """
    Download GGUF models from Hugging Face.
    """
    registry = get_registry()
    
    if models:
        model_names = [m.strip() for m in models.split(",")]
        model_list = [registry.get(n) for n in model_names if registry.get(n)]
    else:
        model_list = registry.get_preset(preset)
    
    llama_models = [m for m in model_list if m.type == "llama_cpp" and m.gguf_path]
    
    if not llama_models:
        typer.echo("No GGUF models to download")
        return
    
    hf_bin = shutil.which("hf")
    if not hf_bin:
        typer.echo("❌ hf CLI not found. Run setup first.")
        raise typer.Exit(1)
    
    models_dir = Path(output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for model in llama_models:
        if check_gguf_exists(model.to_dict()):
            typer.echo(f"✅ {model.name} already exists")
            continue
        
        repo = get_hf_repo_for_download(model.name)
        if not repo:
            typer.echo(f"⚠️  Unknown repo for {model.name}")
            continue
        
        typer.echo(f"📥 Downloading {model.name} from {repo}...")
        result = subprocess.run(
            ["hf", "download", repo, "--local-dir", str(models_dir)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            typer.echo(f"✅ {model.name} downloaded")
        else:
            typer.echo(f"❌ Failed: {result.stderr}")
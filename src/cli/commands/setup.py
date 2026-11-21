"""
Setup command - complete environment setup.
"""
import typer
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from embedding_benchmark.types import SetupTarget
from embedding_benchmark.models.registry import get_registry
from embedding_benchmark.models.hf_repos import get_hf_repo_for_download
from embedding_benchmark.utils import get_project_root, check_gguf_exists


def setup(
    target: SetupTarget = typer.Argument(
        SetupTarget.AUTO,
        help="Target platform (auto-detected if not specified)",
        case_sensitive=False,
    ),
    install_models: bool = typer.Option(
        True,
        "--install-models/--no-install-models",
        help="Whether to download GGUF models",
    ),
    install_deps: bool = typer.Option(
        True,
        "--install-deps/--no-install-deps",
        help="Whether to install system dependencies",
    ),
    install_python_deps: bool = typer.Option(
        True,
        "--install-python-deps/--no-install-python-deps",
        help="Whether to install Python dependencies",
    ),
    python_version: str = typer.Option(
        "3.11",
        "--python-version",
        help="Python version to use for virtual environment",
    ),
    models_preset: str = typer.Option(
        "full",
        "--models-preset",
        help="Model preset to download (quick, formats, full, local_only)",
    ),
    venv_path: str = typer.Option(
        ".venv",
        "--venv",
        help="Virtual environment path",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force reinstall even if already installed",
    ),
):
    """
    Setup the complete benchmark environment.
    
    Installs system dependencies, Python dependencies, creates venv, and downloads models.
    
    TARGET: mac, win, linux, or auto (default)
    """
    if target == SetupTarget.AUTO:
        system = platform.system().lower()
        if system == "darwin":
            target = SetupTarget.MAC
        elif system == "windows":
            target = SetupTarget.WIN
        elif system == "linux":
            target = SetupTarget.LINUX
        else:
            typer.echo(f"❌ Unsupported platform: {system}")
            raise typer.Exit(1)
    
    typer.echo(f"🔧 Setting up for platform: {target.value}")
    typer.echo("=" * 60)
    
    project_root = get_project_root()
    venv_dir = project_root / venv_path
    
    if install_deps:
        _install_system_deps(target, force)
    
    _setup_venv(venv_dir, python_version, force)
    
    if install_python_deps:
        _install_python_deps(venv_dir, project_root)
    
    if install_models:
        _download_models(venv_dir, project_root, models_preset)
    
    typer.echo("=" * 60)
    typer.echo("✅ Setup complete!")
    typer.echo(f"\nTo run benchmarks:")
    typer.echo(f"  source {venv_path}/bin/activate")
    typer.echo(f"  python main.py benchmark --mode quick --lang en")


def _install_system_deps(target: SetupTarget, force: bool):
    """Install system dependencies based on platform."""
    typer.echo("\n📦 Installing system dependencies...")
    
    if target == SetupTarget.MAC:
        _install_macos_deps(force)
    elif target == SetupTarget.LINUX:
        _install_linux_deps(force)
    elif target == SetupTarget.WIN:
        _install_windows_deps(force)
    
    typer.echo("✅ System dependencies installed")


def _install_macos_deps(force: bool):
    """Install dependencies on macOS via Homebrew."""
    if not shutil.which("brew"):
        typer.echo("Installing Homebrew...")
        subprocess.run(
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"',
            shell=True, check=True
        )
    
    subprocess.run("brew update", shell=True, check=not force)
    
    packages = ["python@3.11", "llama.cpp", "hf"]
    for pkg in packages:
        typer.echo(f"  Installing {pkg}...")
        result = subprocess.run(f"brew install {pkg}", shell=True, capture_output=True)
        if result.returncode != 0 and "already installed" not in result.stderr.decode():
            if not force:
                typer.echo(f"  ⚠️  {pkg} may already be installed")


def _install_linux_deps(force: bool):
    """Install dependencies on Linux."""
    if shutil.which("apt"):
        _install_apt_deps(force)
    elif shutil.which("dnf"):
        _install_dnf_deps(force)
    elif shutil.which("pacman"):
        _install_pacman_deps(force)
    else:
        typer.echo("⚠️  Unknown package manager. Install manually: python3.11, llama.cpp, hf")


def _install_apt_deps(force: bool):
    """Install via apt (Ubuntu/Debian)."""
    subprocess.run("sudo apt update", shell=True, check=not force)
    
    packages = [
        "python3.11", "python3.11-venv", "python3.11-dev",
        "git", "curl", "wget", "build-essential"
    ]
    
    for pkg in packages:
        typer.echo(f"  Installing {pkg}...")
        subprocess.run(f"sudo apt install -y {pkg}", shell=True, check=not force)
    
    if not shutil.which("llama-server") or force:
        typer.echo("  Installing llama.cpp...")
        _install_llama_cpp_linux()
    
    if not shutil.which("hf") or force:
        typer.echo("  Installing hf CLI...")
        _install_hf_cli_linux()


def _install_dnf_deps(force: bool):
    """Install via dnf (Fedora/RHEL)."""
    packages = ["python3.11", "python3.11-venv", "git", "curl", "wget", "gcc", "gcc-c++", "make"]
    subprocess.run(f"sudo dnf install -y {' '.join(packages)}", shell=True, check=not force)


def _install_pacman_deps(force: bool):
    """Install via pacman (Arch)."""
    packages = ["python", "git", "curl", "wget", "base-devel"]
    subprocess.run(f"sudo pacman -S --needed {' '.join(packages)}", shell=True, check=not force)


def _install_llama_cpp_linux():
    """Install llama.cpp on Linux."""
    import json
    import urllib.request
    
    with urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest") as resp:
        release = json.load(resp)
    
    tag = release["tag_name"]
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-server-linux-x64"
    
    subprocess.run(f"curl -fsSL {url} -o /usr/local/bin/llama-server", shell=True, check=True)
    subprocess.run("chmod +x /usr/local/bin/llama-server", shell=True, check=True)


def _install_hf_cli_linux():
    """Install Hugging Face CLI on Linux."""
    subprocess.run(
        "curl -fsSL https://huggingface.co/hf/hf/releases/latest/download/hf-linux-amd64 -o /usr/local/bin/hf",
        shell=True, check=True
    )
    subprocess.run("chmod +x /usr/local/bin/hf", shell=True, check=True)


def _install_windows_deps(force: bool):
    """Windows setup - recommend WSL2."""
    typer.echo("⚠️  Windows native not fully supported. Use WSL2:")
    typer.echo("  wsl --install -d Ubuntu")
    typer.echo("  Then run setup inside WSL2")


def _setup_venv(venv_dir: Path, python_version: str, force: bool):
    """Create Python virtual environment."""
    typer.echo(f"\n🐍 Setting up Python {python_version} virtual environment...")
    
    if venv_dir.exists() and not force:
        typer.echo(f"  Virtual environment already exists at {venv_dir}")
        return
    
    python_bin = shutil.which(f"python{python_version}") or shutil.which("python3") or shutil.which("python")
    if not python_bin:
        typer.echo(f"❌ Python {python_version} not found")
        raise typer.Exit(1)
    
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    
    typer.echo(f"  Creating venv at {venv_dir}...")
    subprocess.run([python_bin, "-m", "venv", str(venv_dir)], check=True)
    typer.echo("✅ Virtual environment created")


def _install_python_deps(venv_dir: Path, project_root: Path):
    """Install Python dependencies."""
    typer.echo("\n📦 Installing Python dependencies...")
    
    pip_bin = venv_dir / "bin" / "pip"
    if platform.system() == "Windows":
        pip_bin = venv_dir / "Scripts" / "pip.exe"
    
    subprocess.run([str(pip_bin), "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
    
    req_file = project_root / "requirements.txt"
    if req_file.exists():
        subprocess.run([str(pip_bin), "install", "-r", str(req_file)], check=True)
        typer.echo("✅ Python dependencies installed")
    else:
        typer.echo("⚠️  requirements.txt not found")


def _download_models(venv_dir: Path, project_root: Path, preset: str):
    """Download GGUF models using hf CLI."""
    typer.echo(f"\n📥 Downloading models (preset: {preset})...")
    
    try:
        from embedding_benchmark.models.registry import get_registry
        registry = get_registry()
        models = registry.get_preset(preset)
        
        if not models:
            typer.echo(f"⚠️  Unknown preset: {preset}")
            return
        
        llama_models = [m for m in models if m.type == "llama_cpp" and m.gguf_path]
        
        if not llama_models:
            typer.echo("  No GGUF models in preset")
            return
        
        hf_bin = shutil.which("hf")
        if not hf_bin:
            typer.echo("⚠️  hf CLI not found, skipping model download")
            return
        
        models_dir = Path(llama_models[0].gguf_path).parent
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model in llama_models:
            if check_gguf_exists(model.to_dict()):
                typer.echo(f"  ✅ {model.name} already exists")
                continue
            
            typer.echo(f"  Downloading {model.name}...")
            repo = get_hf_repo_for_download(model.name)
            if repo:
                result = subprocess.run(
                    ["hf", "download", repo, "--local-dir", str(models_dir)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    typer.echo(f"  ✅ {model.name} downloaded")
                else:
                    typer.echo(f"  ❌ Failed: {result.stderr}")
        
        typer.echo("✅ Model download complete")

    except ImportError:
        typer.echo("⚠️  Model registry not available, skipping")
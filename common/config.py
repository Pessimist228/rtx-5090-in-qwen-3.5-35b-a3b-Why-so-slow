"""Конфигурация хоста.

Единственное, что отличается между ноутом и арендованной картой, — файл в config/.
Код одинаковый на обеих машинах (критерий приёмки 8).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = HARNESS_ROOT / "config"


class ConfigError(RuntimeError):
    """Конфиг не найден, не подходит этой машине или указывает в никуда."""


def _expand(raw: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def detect_gpu_name() -> str | None:
    """Имя GPU 0 по nvidia-smi. None, если nvidia-smi недоступен."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[0] if lines else None


class HostConfig:
    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

    # --- загрузка -----------------------------------------------------

    @classmethod
    def load(cls, host: str | None = None) -> "HostConfig":
        """Загрузить конфиг по id/пути. Без аргумента — автоопределение по GPU.

        Автоопределение позволяет run_all.sh запускаться одной и той же командой
        на обеих машинах.
        """
        if host:
            candidate = Path(host)
            if not candidate.is_file():
                candidate = CONFIG_DIR / f"{host}.json"
            if not candidate.is_file():
                raise ConfigError(f"конфиг не найден: {host}")
            return cls(candidate, json.loads(candidate.read_text(encoding="utf-8")))

        env_host = os.environ.get("HARNESS_HOST")
        if env_host:
            return cls.load(env_host)

        gpu = detect_gpu_name()
        if gpu is None:
            raise ConfigError(
                "nvidia-smi недоступен — автоопределение хоста невозможно. "
                "Укажите --host явно."
            )
        matches = []
        for cfg_path in sorted(CONFIG_DIR.glob("*.json")):
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            needle = data.get("gpu", {}).get("expected_name_substring", "")
            if needle and needle.lower() in gpu.lower():
                matches.append(cls(cfg_path, data))
        if not matches:
            raise ConfigError(
                f"GPU '{gpu}' не соответствует ни одному конфигу в {CONFIG_DIR}. "
                "Добавьте конфиг или укажите --host."
            )
        if len(matches) > 1:
            ids = ", ".join(m.host_id for m in matches)
            raise ConfigError(f"GPU '{gpu}' подходит под несколько конфигов: {ids}")
        return matches[0]

    # --- поля ---------------------------------------------------------

    @property
    def host_id(self) -> str:
        return self.data["host_id"]

    @property
    def phase(self) -> int:
        return self.data["phase"]

    @property
    def gpu(self) -> dict:
        return self.data["gpu"]

    @property
    def safety(self) -> dict:
        return self.data["safety"]

    @property
    def bench_matrix(self) -> dict:
        return self.data["bench_matrix"]

    @property
    def bin_dir(self) -> Path:
        return _expand(self.data["llama_cpp"]["bin_dir"])

    @property
    def results_dir(self) -> Path:
        return _expand(self.data["results_dir"])

    # --- разрешение путей ---------------------------------------------

    def exe(self, name: str) -> Path:
        """Путь к бинарю llama.cpp. Падает, если его нет."""
        suffix = self.data["llama_cpp"].get("exe_suffix", "")
        path = self.bin_dir / f"{name}{suffix}"
        if not path.is_file():
            raise ConfigError(f"бинарь llama.cpp не найден: {path}")
        return path

    def find_model(self, filename: str) -> Path:
        """Найти GGUF по имени файла в каталогах поиска."""
        direct = _expand(filename)
        if direct.is_file():
            return direct
        searched = []
        for raw in self.data["models"]["search_dirs"]:
            d = _expand(raw)
            searched.append(str(d))
            candidate = d / filename
            if candidate.is_file():
                return candidate
        raise ConfigError(
            f"модель '{filename}' не найдена. Искали в: {', '.join(searched)}"
        )

    def list_models(self) -> list[Path]:
        """Все GGUF в каталогах поиска, без дублей, отсортированные по имени."""
        seen: dict[str, Path] = {}
        for raw in self.data["models"]["search_dirs"]:
            d = _expand(raw)
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.gguf")):
                seen.setdefault(p.name, p)
        return [seen[k] for k in sorted(seen)]

    # --- каталог прогона ----------------------------------------------

    def run_dir(self, model: str, quant: str, timestamp: str | None = None) -> Path:
        """results/<host>_<gpu>_<model>_<quant>_<timestamp>/ — создаётся сразу."""
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        gpu_slug = _slug(detect_gpu_name() or self.gpu["expected_name_substring"])
        name = f"{_slug(self.host_id)}_{gpu_slug}_{_slug(model)}_{_slug(quant)}_{ts}"
        path = self.results_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path


def _slug(text: str) -> str:
    """Имя, безопасное для файловой системы: без пробелов и спецсимволов."""
    text = re.sub(r"\.gguf$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(NVIDIA|GeForce|Laptop|GPU)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return re.sub(r"-{2,}", "-", text)

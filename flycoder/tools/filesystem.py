"""Safe filesystem tools for FLY-CODER."""

import os
import tempfile
from pathlib import Path


class Workspace:
    """Restrict file operations to one workspace directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _safe_path(self, relative_path: str | Path) -> Path:
        """Return a path that must remain inside the workspace."""

        target = (self.root / relative_path).resolve()

        if target != self.root and self.root not in target.parents:
            raise ValueError("Path is outside the workspace.")

        return target

    def list_files(self) -> list[str]:
        """List readable source files inside the workspace."""

        if not self.root.exists():
            return []

        ignored_directories = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
        }

        ignored_extensions = {
            ".pyc",
            ".pyo",
            ".so",
            ".dll",
            ".exe",
            ".db",
            ".sqlite",
            ".sqlite3",
        }

        files = []

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue

            relative_path = path.relative_to(self.root)

            if any(
                directory in ignored_directories
                for directory in relative_path.parts
            ):
                continue

            if path.suffix.lower() in ignored_extensions:
                continue

            files.append(relative_path.as_posix())

        return sorted(files)

    def read_file(self, relative_path: str) -> str:
        """Read a text file inside the workspace."""

        path = self._safe_path(relative_path)

        if not path.is_file():
            raise FileNotFoundError(relative_path)

        return path.read_text(encoding="utf-8")

    def write_file(self, relative_path: str, content: str) -> None:
        """Write a text file inside the workspace."""

        path = self._safe_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def atomic_write_file(
        self,
        relative_path: str,
        content: str,
    ) -> None:
        """Atomically replace a text file inside the workspace."""

        path = self._safe_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, path)
            temporary_path = None

        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

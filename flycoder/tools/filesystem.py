"""Safe filesystem tools for FLY-CODER."""

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
        """List files inside the workspace."""

        if not self.root.exists():
            return []

        return sorted(
            str(path.relative_to(self.root))
            for path in self.root.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and ".venv" not in path.parts
        )

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
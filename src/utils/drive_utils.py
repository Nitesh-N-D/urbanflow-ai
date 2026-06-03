"""Google Drive checkpoint synchronization helpers."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class DriveCheckpointManager:
    """Syncs training checkpoints to Google Drive when mounted in Colab."""

    def __init__(self, gdrive_dir: str | None = None) -> None:
        """Initialize manager.

        Args:
            gdrive_dir: Google Drive checkpoint directory.

        Returns:
            None.

        Raises:
            None.
        """
        self.gdrive_dir = Path(gdrive_dir) if gdrive_dir else None
        self.available = self._check_drive()
        if self.gdrive_dir and self.available:
            self.gdrive_dir.mkdir(parents=True, exist_ok=True)

    def _check_drive(self) -> bool:
        """Check whether Google Drive is mounted.

        Args:
            None.

        Returns:
            True if Drive is mounted.

        Raises:
            None.
        """
        try:
            import google.colab  # noqa: F401

            return os.path.ismount("/content/drive")
        except ImportError:
            return False

    def save_checkpoint(self, checkpoint_path: str | Path, epoch: int) -> bool:
        """Copy checkpoint to Google Drive.

        Args:
            checkpoint_path: Local checkpoint path.
            epoch: Epoch number.

        Returns:
            True when copied.

        Raises:
            None.
        """
        source = Path(checkpoint_path)
        if not self.available or not self.gdrive_dir or not source.exists():
            return False
        try:
            dest = self.gdrive_dir / f"checkpoint_epoch_{epoch}.pt"
            shutil.copy2(source, dest)
            return True
        except Exception as exc:
            logger.warning("Drive save failed: %s", exc)
            return False

    def load_latest(self, local_dir: str | Path) -> str | None:
        """Find most recent checkpoint locally or on Drive.

        Args:
            local_dir: Local checkpoint directory.

        Returns:
            Latest checkpoint path or None.

        Raises:
            None.
        """
        candidates = list(Path(local_dir).glob("*.pt")) if Path(local_dir).exists() else []
        if self.available and self.gdrive_dir and self.gdrive_dir.exists():
            candidates.extend(self.gdrive_dir.glob("*.pt"))
        if not candidates:
            return None
        return str(max(candidates, key=lambda path: path.stat().st_mtime))

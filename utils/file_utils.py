"""
utils/file_utils.py – File-system helper utilities.
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


def ensure_directories(*paths: Path) -> None:
    """Create all supplied *paths* (and their parents) if they do not exist.

    Args:
        *paths: Any number of :class:`pathlib.Path` objects to create.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", path)


def save_pickle(data: Any, path: Path) -> None:
    """Serialise *data* to *path* using :mod:`pickle`.

    Args:
        data: Python object to serialise.
        path: Destination file path (parent directory must exist).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.debug("Saved pickle to %s", path)


def load_pickle(path: Path) -> Any:
    """Deserialise a pickle file from *path*.

    Args:
        path: Path to the pickle file.

    Returns:
        Deserialised Python object.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as fh:
        data = pickle.load(fh)  # noqa: S301 – internal pickle, controlled input
    logger.debug("Loaded pickle from %s", path)
    return data


def delete_file(path: Path) -> bool:
    """Delete a single file at *path*.

    Args:
        path: File to delete.

    Returns:
        ``True`` if the file was deleted, ``False`` if it did not exist.
    """
    if path.exists() and path.is_file():
        path.unlink()
        logger.debug("Deleted file: %s", path)
        return True
    return False


def delete_directory(path: Path) -> bool:
    """Recursively delete a directory at *path*.

    Args:
        path: Directory to delete.

    Returns:
        ``True`` if the directory was deleted, ``False`` if it did not exist.
    """
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        logger.debug("Deleted directory: %s", path)
        return True
    return False


def list_files(directory: Path, pattern: str = "*") -> list[Path]:
    """Return a sorted list of files in *directory* matching *pattern*.

    Args:
        directory: Directory to search.
        pattern:   Glob pattern (default ``"*"`` – all files).

    Returns:
        Sorted list of :class:`~pathlib.Path` objects.
    """
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def count_files(directory: Path, pattern: str = "*") -> int:
    """Count files in *directory* matching *pattern*.

    Args:
        directory: Directory to inspect.
        pattern:   Glob pattern.

    Returns:
        Number of matching files.
    """
    return len(list_files(directory, pattern))


def get_student_image_dir(student_id: str, base_upload_dir: Path) -> Path:
    """Return the image storage directory for a student.

    Args:
        student_id:      Student identifier.
        base_upload_dir: Root upload directory.

    Returns:
        :class:`~pathlib.Path` for ``<base_upload_dir>/<student_id>``.
    """
    return base_upload_dir / student_id


def get_encoding_path(student_id: str, encoding_dir: Path) -> Path:
    """Return the pickle file path for a student's face encodings.

    Args:
        student_id:   Student identifier.
        encoding_dir: Root encodings directory.

    Returns:
        :class:`~pathlib.Path` for ``<encoding_dir>/<student_id>.pickle``.
    """
    return encoding_dir / f"{student_id}.pickle"


def cleanup_temp_dir(temp_dir: Path) -> None:
    """Remove all files from the temp directory.

    Args:
        temp_dir: Temporary files directory.
    """
    if temp_dir.exists():
        for file in temp_dir.iterdir():
            if file.is_file():
                file.unlink()
    logger.debug("Cleaned up temp directory: %s", temp_dir)

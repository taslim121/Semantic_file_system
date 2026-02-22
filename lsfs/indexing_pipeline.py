import os
import fnmatch
import string
from typing import Iterable, List, Optional, Tuple


_PRINTABLE = set(bytes(string.printable, "ascii"))


def _matches_any(name: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


def is_hidden_path(path: str) -> bool:
    name = os.path.basename(path)
    if name.startswith("."):
        return True
    if os.name == "nt":
        try:
            import ctypes

            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            if attrs != -1:
                # FILE_ATTRIBUTE_HIDDEN = 0x2, FILE_ATTRIBUTE_SYSTEM = 0x4
                return bool(attrs & 0x2) or bool(attrs & 0x4)
        except Exception:
            return False
    return False


def should_skip_dir(dir_name: str, dir_path: str, exclude_dirs: List[str], skip_hidden: bool) -> bool:
    if skip_hidden and is_hidden_path(dir_path):
        return True
    if exclude_dirs and _matches_any(dir_name, exclude_dirs):
        return True
    return False


def iter_files(root_dir: str, exclude_dirs: List[str], skip_hidden: bool) -> Iterable[str]:
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [
            d
            for d in dirs
            if not should_skip_dir(d, os.path.join(root, d), exclude_dirs, skip_hidden)
        ]
        for file_name in files:
            yield os.path.join(root, file_name)


def is_probably_text(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return False
    # Count non-printable characters (excluding common whitespace)
    non_printable = 0
    for b in sample:
        if b in (9, 10, 13):  # tab, lf, cr
            continue
        if b not in _PRINTABLE:
            non_printable += 1
    ratio = non_printable / max(1, len(sample))
    return ratio < 0.2


def should_index_file(
    file_path: str,
    *,
    max_file_size_mb: int,
    exclude_extensions: List[str],
    include_extensions: List[str],
    skip_hidden: bool,
    max_text_bytes: int,
) -> bool:
    if skip_hidden and is_hidden_path(file_path):
        return False

    ext = os.path.splitext(file_path)[1].lower()
    if exclude_extensions and ext in exclude_extensions:
        return False
    if include_extensions and ext not in include_extensions:
        return False

    try:
        size = os.path.getsize(file_path)
    except OSError:
        return False

    if size == 0:
        return False
    if size > max_file_size_mb * 1024 * 1024:
        return False

    try:
        with open(file_path, "rb") as f:
            sample = f.read(min(8192, max_text_bytes))
        return is_probably_text(sample)
    except OSError:
        return False


def read_text_limited(file_path: str, max_text_bytes: int) -> Optional[str]:
    try:
        with open(file_path, "rb") as f:
            raw = f.read(max_text_bytes)
    except OSError:
        return None

    if not raw:
        return None

    for encoding in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1", "cp1252"):
        try:
            text = raw.decode(encoding, errors="ignore")
            if text:
                return text
        except Exception:
            continue
    return None


def chunk_text(
    text: str, chunk_size: int, chunk_overlap: int, max_chunks: int
) -> List[Tuple[str, int, int]]:
    if chunk_size <= 0:
        return [(text, 0, len(text))]

    chunks: List[Tuple[str, int, int]] = []
    start = 0
    text_len = len(text)
    overlap = max(0, min(chunk_overlap, chunk_size - 1))

    while start < text_len and len(chunks) < max_chunks:
        end = min(text_len, start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append((chunk, start, end))
        if end >= text_len:
            break
        start = end - overlap

    return chunks

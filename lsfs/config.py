from dataclasses import dataclass, field
from typing import List
import yaml
import os


DEFAULT_ROOT_DIR = os.path.expanduser("~/lsfs-root")
DEFAULT_VECTOR_DB_DIR = "./.lsfs_db"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RESULTS = 5
DEFAULT_MIN_SIMILARITY = 0.3
DEFAULT_MAX_FILE_SIZE_MB = 10

DEFAULT_EXCLUDE_DIRS = [
    ".git",
    ".venv",
    ".lsfs_db",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "System Volume Information",
    "$RECYCLE.BIN",
]

DEFAULT_EXCLUDE_EXTENSIONS = [
    # binaries
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".dat",
    # archives
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".bz2",
    ".iso",
    # media
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".svg",
    ".webp",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    # documents that need special parsers
    ".pdf",
]

@dataclass
class LSFSConfig:
    """Configuration for Local Semantic File System"""
    root_dir: str = field(default_factory=lambda: DEFAULT_ROOT_DIR)
    vector_db_dir: str = DEFAULT_VECTOR_DB_DIR
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    default_results: int = DEFAULT_RESULTS
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    auto_mount: bool = True
    batch_size: int = 32
    max_text_bytes: int = 512 * 1024
    chunk_size: int = 1200
    chunk_overlap: int = 150
    max_chunks_per_file: int = 8
    skip_hidden: bool = True
    use_default_excludes: bool = True
    exclude_dirs: List[str] = field(default_factory=list)
    exclude_extensions: List[str] = field(default_factory=list)
    include_extensions: List[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f) or {}

        embedding_cfg = config.get('embedding', {}) or {}
        search_cfg = config.get('search', {}) or {}
        indexing_cfg = config.get('indexing', {}) or {}
        
        return cls(
            root_dir=os.path.expanduser(config.get('root_dir', DEFAULT_ROOT_DIR)),
            vector_db_dir=os.path.expanduser(
                config.get('vector_db_dir', DEFAULT_VECTOR_DB_DIR)
            ),
            embedding_model=embedding_cfg.get('model', DEFAULT_EMBEDDING_MODEL),
            default_results=search_cfg.get('default_results', DEFAULT_RESULTS),
            min_similarity=search_cfg.get('min_similarity', DEFAULT_MIN_SIMILARITY),
            max_file_size_mb=indexing_cfg.get(
                'max_file_size_mb', DEFAULT_MAX_FILE_SIZE_MB
            ),
            auto_mount=config.get('auto_mount', True),
            batch_size=indexing_cfg.get('batch_size', 32),
            max_text_bytes=indexing_cfg.get('max_text_bytes', 512 * 1024),
            chunk_size=indexing_cfg.get('chunk_size', 1200),
            chunk_overlap=indexing_cfg.get('chunk_overlap', 150),
            max_chunks_per_file=indexing_cfg.get('max_chunks_per_file', 8),
            skip_hidden=indexing_cfg.get('skip_hidden', True),
            use_default_excludes=indexing_cfg.get('use_default_excludes', True),
            exclude_dirs=indexing_cfg.get('exclude_dirs', []),
            exclude_extensions=indexing_cfg.get('exclude_extensions', []),
            include_extensions=indexing_cfg.get('include_extensions', []),
        )
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)

        # Normalize lists
        self.exclude_dirs = [d.strip() for d in self.exclude_dirs if d]
        self.exclude_extensions = [e.lower().strip() for e in self.exclude_extensions if e]
        self.include_extensions = [e.lower().strip() for e in self.include_extensions if e]

        if self.use_default_excludes:
            for d in DEFAULT_EXCLUDE_DIRS:
                if d not in self.exclude_dirs:
                    self.exclude_dirs.append(d)
            for e in DEFAULT_EXCLUDE_EXTENSIONS:
                if e not in self.exclude_extensions:
                    self.exclude_extensions.append(e)

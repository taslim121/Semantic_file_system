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
DEFAULT_INTENT_MIN_CONFIDENCE = 0.15
DEFAULT_INTENT_FALLBACK_TARGET = "default"
DEFAULT_MAX_THREADS = 4
DEFAULT_OLLAMA_ENABLED = False
DEFAULT_OLLAMA_MODEL = "llama3:8b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

DEFAULT_CODE_EXTENSIONS = [
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".scala",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
]

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
    embedding_default_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_code_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_intent_model: str = DEFAULT_EMBEDDING_MODEL
    code_extensions: List[str] = field(default_factory=list)
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
    intent_enabled: bool = True
    intent_labels: List[dict] = field(default_factory=list)
    intent_min_confidence: float = DEFAULT_INTENT_MIN_CONFIDENCE
    intent_fallback_target: str = DEFAULT_INTENT_FALLBACK_TARGET
    max_threads: int = DEFAULT_MAX_THREADS
    ollama_enabled: bool = DEFAULT_OLLAMA_ENABLED
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f) or {}

        embedding_cfg = config.get('embedding', {}) or {}
        search_cfg = config.get('search', {}) or {}
        indexing_cfg = config.get('indexing', {}) or {}
        intent_cfg = config.get('intent_routing', {}) or {}
        performance_cfg = config.get('performance', {}) or {}
        ollama_cfg = config.get('ollama', {}) or {}

        default_model = embedding_cfg.get('default_model', embedding_cfg.get('model', DEFAULT_EMBEDDING_MODEL))
        return cls(
            root_dir=os.path.expanduser(config.get('root_dir', DEFAULT_ROOT_DIR)),
            vector_db_dir=os.path.expanduser(
                config.get('vector_db_dir', DEFAULT_VECTOR_DB_DIR)
            ),
            embedding_default_model=default_model,
            embedding_code_model=embedding_cfg.get('code_model', default_model),
            embedding_intent_model=embedding_cfg.get('intent_model', default_model),
            code_extensions=embedding_cfg.get('code_extensions', DEFAULT_CODE_EXTENSIONS),
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
            intent_enabled=intent_cfg.get('enabled', True),
            intent_labels=intent_cfg.get('labels', []),
            intent_min_confidence=intent_cfg.get(
                'min_confidence', DEFAULT_INTENT_MIN_CONFIDENCE
            ),
            intent_fallback_target=intent_cfg.get(
                'fallback_target', DEFAULT_INTENT_FALLBACK_TARGET
            ),
            max_threads=performance_cfg.get('max_threads', DEFAULT_MAX_THREADS),
            ollama_enabled=ollama_cfg.get('enabled', DEFAULT_OLLAMA_ENABLED),
            ollama_model=ollama_cfg.get('model', DEFAULT_OLLAMA_MODEL),
            ollama_url=ollama_cfg.get('url', DEFAULT_OLLAMA_URL),
        )
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)

        # Normalize lists
        self.exclude_dirs = [d.strip() for d in self.exclude_dirs if d]
        def _norm_ext(ext: str) -> str:
            ext = ext.lower().strip()
            if ext and not ext.startswith("."):
                ext = f".{ext}"
            return ext

        self.exclude_extensions = [_norm_ext(e) for e in self.exclude_extensions if e]
        self.include_extensions = [_norm_ext(e) for e in self.include_extensions if e]
        self.code_extensions = [_norm_ext(e) for e in self.code_extensions if e]

        if not self.code_extensions:
            self.code_extensions = list(DEFAULT_CODE_EXTENSIONS)

        if self.use_default_excludes:
            for d in DEFAULT_EXCLUDE_DIRS:
                if d not in self.exclude_dirs:
                    self.exclude_dirs.append(d)
            for e in DEFAULT_EXCLUDE_EXTENSIONS:
                if e not in self.exclude_extensions:
                    self.exclude_extensions.append(e)

        # Normalize intent labels
        normalized_labels = []
        for label in self.intent_labels:
            if not isinstance(label, dict):
                continue
            name = str(label.get('name', '')).strip()
            description = str(label.get('description', '')).strip()
            target = str(label.get('target', '')).strip() or DEFAULT_INTENT_FALLBACK_TARGET
            if not name and not description:
                continue
            normalized_labels.append(
                {"name": name, "description": description, "target": target}
            )
        self.intent_labels = normalized_labels

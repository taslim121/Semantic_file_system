from dataclasses import dataclass
import yaml
import os

@dataclass
class LSFSConfig:
    """Configuration for Local Semantic File System"""
    root_dir: str
    vector_db_dir: str
    embedding_model: str
    default_results: int
    max_file_size_mb: int
    auto_mount: bool = True
    batch_size: int = 32
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            root_dir=os.path.expanduser(config['root_dir']),
            vector_db_dir=os.path.expanduser(config['vector_db_dir']),
            embedding_model=config['embedding']['model'],
            default_results=config['search']['default_results'],
            max_file_size_mb=config['indexing']['max_file_size_mb'],
            auto_mount=config.get('auto_mount', True),
            batch_size=config['indexing'].get('batch_size', 32)
        )
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)

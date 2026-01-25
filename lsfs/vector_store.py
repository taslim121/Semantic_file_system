import os
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import csv
from tqdm import tqdm
from datetime import datetime

class VectorStore:
    """Handles embeddings and semantic search using ChromaDB with CSV indexing"""
    
    def __init__(self, db_path: str, embedding_model: str):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(embedding_model)
        self.csv_index_path = os.path.join(db_path, "file_index.csv")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="files",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load CSV index
        self.file_index = self._load_csv_index()
    
    def _load_csv_index(self) -> Dict[str, Dict]:
        """Load file index from CSV"""
        index = {}
        if os.path.exists(self.csv_index_path):
            with open(self.csv_index_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    index[row['file_path']] = {
                        'hash': row['hash'],
                        'size': int(row['size']),
                        'modified': row['modified'],
                        'indexed_at': row['indexed_at']
                    }
        return index
    
    def _save_csv_index(self):
        """Save file index to CSV"""
        os.makedirs(os.path.dirname(self.csv_index_path) if os.path.dirname(self.csv_index_path) else self.db_path, exist_ok=True)
        with open(self.csv_index_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file_path', 'hash', 'size', 'modified', 'indexed_at'])
            writer.writeheader()
            for file_path, data in self.file_index.items():
                writer.writerow({
                    'file_path': file_path,
                    'hash': data['hash'],
                    'size': data['size'],
                    'modified': data['modified'],
                    'indexed_at': data['indexed_at']
                })
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Generate hash for file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def _generate_file_id(self, file_path: str) -> str:
        """Generate unique ID for file"""
        return hashlib.sha256(file_path.encode()).hexdigest()
    
    def _read_file_content(self, file_path: str, max_size_mb: int = 10) -> Optional[str]:
        """Safely read file content"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > max_size_mb * 1024 * 1024:
                return None
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                    return content
                except:
                    continue
            return None
        except Exception:
            return None
    
    def _needs_reindex(self, file_path: str) -> bool:
        """Check if file needs reindexing based on CSV"""
        if file_path not in self.file_index:
            return True
        
        current_hash = self._generate_file_hash(file_path)
        return current_hash != self.file_index[file_path]['hash']
    
    def index_file(self, file_path: str, root_dir: str) -> str:
        """Index a single file. Returns 'indexed', 'unchanged', or 'error'."""
        try:
            # Check if reindex needed
            if not self._needs_reindex(file_path):
                return 'unchanged'
            
            # Read content
            content = self._read_file_content(file_path)
            if not content or not content.strip():
                return 'error'
            
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Prepare metadata
            rel_path = os.path.relpath(file_path, root_dir)
            file_id = self._generate_file_id(file_path)
            file_hash = self._generate_file_hash(file_path)
            
            metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "relative_path": rel_path,
                "extension": os.path.splitext(file_path)[1],
                "size": os.path.getsize(file_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                "hash": file_hash
            }
            
            # Upsert to ChromaDB
            self.collection.upsert(
                ids=[file_id],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            
            # Update CSV index
            self.file_index[file_path] = {
                'hash': file_hash,
                'size': metadata['size'],
                'modified': metadata['modified'],
                'indexed_at': datetime.now().isoformat()
            }
            
            return 'indexed'
        
        except Exception as e:
            return 'error'
    
    def index_directory(self, directory: str, batch_size: int = 32) -> Dict[str, int]:
        """Index all files in directory with batching."""
        stats = {'indexed': 0, 'unchanged': 0, 'error': 0}
        
        # Collect all files
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith('.'):
                    continue
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        
        # Batch process
        with tqdm(total=len(file_paths), desc="Indexing files") as pbar:
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i + batch_size]
                
                for file_path in batch:
                    result = self.index_file(file_path, directory)
                    stats[result] += 1
                    pbar.update(1)
        
        # Save CSV index
        self._save_csv_index()
        
        return stats
    
    def search(self, query: str, k: int = 5, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Semantic search for files - INSTANT (no LLM)"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare where filter
            where_filter = {}
            if file_type:
                where_filter["extension"] = file_type
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                where=where_filter if where_filter else None
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance
                    
                    formatted_results.append({
                        "file_path": results['metadatas'][0][i]['file_path'],
                        "file_name": results['metadatas'][0][i]['file_name'],
                        "relative_path": results['metadatas'][0][i]['relative_path'],
                        "extension": results['metadatas'][0][i]['extension'],
                        "size": results['metadatas'][0][i]['size'],
                        "modified": results['metadatas'][0][i]['modified'],
                        "similarity": round(similarity, 4)
                    })
            
            return formatted_results
        
        except Exception as e:
            return []
    
    def remove_file(self, file_path: str) -> bool:
        """Remove file from index"""
        try:
            file_id = self._generate_file_id(file_path)
            self.collection.delete(ids=[file_id])
            
            if file_path in self.file_index:
                del self.file_index[file_path]
                self._save_csv_index()
            
            return True
        except:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        return {
            "total_files": self.collection.count(),
            "indexed_files": len(self.file_index),
            "db_path": self.db_path,
            "csv_index": self.csv_index_path
        }

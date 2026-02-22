import os
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import csv
from tqdm import tqdm
from datetime import datetime
from .indexing_pipeline import (
    iter_files,
    should_index_file,
    read_text_limited,
    chunk_text,
)
from .intent_router import IntentRouter

class VectorStore:
    """Handles embeddings and semantic search using ChromaDB with CSV indexing"""

    def __init__(self, config):
        self.config = config
        self.db_path = config.vector_db_dir
        self.csv_index_path = os.path.join(self.db_path, "file_index.csv")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Models and collections per target
        self.models: Dict[str, SentenceTransformer] = {}
        self.collections: Dict[str, Any] = {}

        model_cache: Dict[str, SentenceTransformer] = {}
        targets = {
            "default": config.embedding_default_model,
            "code": config.embedding_code_model,
        }

        # Limit CPU spikes if configured
        if getattr(config, "max_threads", 0):
            try:
                import os as _os
                _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                import torch

                max_threads = max(1, int(config.max_threads))
                torch.set_num_threads(max_threads)
                torch.set_num_interop_threads(max(1, max_threads // 2))
            except Exception:
                pass

        for target, model_name in targets.items():
            if not model_name:
                continue
            if model_name in model_cache:
                model = model_cache[model_name]
            else:
                model = SentenceTransformer(model_name)
                model_cache[model_name] = model
            self.models[target] = model
            self.collections[target] = self.client.get_or_create_collection(
                name=f"files_{target}",
                metadata={"hnsw:space": "cosine"},
            )

        # Fallback to default model if code model not available
        if "default" not in self.models:
            self.models["default"] = SentenceTransformer(config.embedding_default_model)
            self.collections["default"] = self.client.get_or_create_collection(
                name="files_default",
                metadata={"hnsw:space": "cosine"},
            )
        if "code" not in self.models:
            self.models["code"] = self.models["default"]
            self.collections["code"] = self.client.get_or_create_collection(
                name="files_code",
                metadata={"hnsw:space": "cosine"},
            )

        # Intent router
        self.intent_router: Optional[IntentRouter] = None
        if config.intent_enabled and config.intent_labels:
            self.intent_router = IntentRouter(
                config.embedding_intent_model,
                config.intent_labels,
                config.intent_min_confidence,
                config.intent_fallback_target,
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
                        'hash': row.get('hash', ''),
                        'size': int(row.get('size', 0)),
                        'modified': row.get('modified', ''),
                        'indexed_at': row.get('indexed_at', ''),
                        'chunks': int(row.get('chunks', 0)),
                        'target': row.get('target', 'default'),
                    }
        return index
    
    def _save_csv_index(self):
        """Save file index to CSV"""
        os.makedirs(
            os.path.dirname(self.csv_index_path)
            if os.path.dirname(self.csv_index_path)
            else self.db_path,
            exist_ok=True,
        )
        with open(self.csv_index_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'file_path',
                    'hash',
                    'size',
                    'modified',
                    'indexed_at',
                    'chunks',
                    'target',
                ],
            )
            writer.writeheader()
            for file_path, data in self.file_index.items():
                writer.writerow(
                    {
                        'file_path': file_path,
                        'hash': data.get('hash', ''),
                        'size': data.get('size', 0),
                        'modified': data.get('modified', ''),
                        'indexed_at': data.get('indexed_at', ''),
                        'chunks': data.get('chunks', 0),
                        'target': data.get('target', 'default'),
                    }
                )
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Generate hash for file content (streaming, low memory)"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def _generate_file_id(self, file_path: str) -> str:
        """Generate unique ID for file"""
        return hashlib.sha256(file_path.encode()).hexdigest()

    def _resolve_target_for_extension(self, extension: str) -> str:
        ext = (extension or "").lower()
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        if ext in self.config.code_extensions:
            return "code" if "code" in self.collections else "default"
        return "default"

    def _delete_from_all_collections(self, file_path: str) -> None:
        for collection in self.collections.values():
            try:
                collection.delete(where={"file_path": file_path})
            except Exception:
                continue

    def _get_collection_and_model(self, target: str) -> Tuple[Any, SentenceTransformer]:
        if target not in self.collections:
            target = "default"
        return self.collections[target], self.models[target]

    def _route_query_targets(self, query: str, file_type: Optional[str]) -> List[str]:
        if file_type:
            target = self._resolve_target_for_extension(file_type)
            return [target]

        if self.intent_router:
            target, _, _ = self.intent_router.route(query)
            if target == "all":
                return list(self.collections.keys())
            if target in self.collections:
                return [target]

        fallback = self.config.intent_fallback_target
        if fallback == "all":
            return list(self.collections.keys())
        return ["default"] if "default" in self.collections else list(self.collections.keys())
    
    def _read_file_content(self, file_path: str) -> Optional[str]:
        """Safely read limited file content"""
        return read_text_limited(file_path, self.config.max_text_bytes)
    
    def _needs_reindex(self, file_path: str) -> bool:
        """Check if file needs reindexing based on CSV"""
        if file_path not in self.file_index:
            return True

        try:
            stat = os.stat(file_path)
        except OSError:
            return False

        current_size = stat.st_size
        current_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
        entry = self.file_index[file_path]

        if entry.get('size') == current_size and entry.get('modified') == current_modified:
            return False

        current_hash = self._generate_file_hash(file_path)
        return current_hash != entry.get('hash')
    
    def index_file(self, file_path: str, root_dir: str) -> str:
        """Index a single file. Returns 'indexed', 'unchanged', 'skipped', or 'error'."""
        try:
            # Skip non-indexable files early
            if not should_index_file(
                file_path,
                max_file_size_mb=self.config.max_file_size_mb,
                exclude_extensions=self.config.exclude_extensions,
                include_extensions=self.config.include_extensions,
                skip_hidden=self.config.skip_hidden,
                max_text_bytes=self.config.max_text_bytes,
            ):
                # Remove if previously indexed
                if file_path in self.file_index:
                    self.remove_file(file_path)
                return 'skipped'

            # Check if reindex needed
            if not self._needs_reindex(file_path):
                return 'unchanged'

            # Read content (limited bytes)
            content = self._read_file_content(file_path)
            if not content or not content.strip():
                return 'skipped'

            extension = os.path.splitext(file_path)[1].lower()
            target = self._resolve_target_for_extension(extension)
            collection, model = self._get_collection_and_model(target)

            # Chunk content
            chunks = chunk_text(
                content,
                self.config.chunk_size,
                self.config.chunk_overlap,
                self.config.max_chunks_per_file,
            )
            if not chunks:
                return 'skipped'

            # Generate embeddings in batch
            chunk_texts = [c[0] for c in chunks]
            embeddings = model.encode(
                chunk_texts, batch_size=self.config.batch_size, show_progress_bar=False
            ).tolist()

            # Prepare metadata
            rel_path = os.path.relpath(file_path, root_dir)
            file_id = self._generate_file_id(file_path)
            file_hash = self._generate_file_hash(file_path)
            size = os.path.getsize(file_path)
            modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            # Remove previous chunks (if any)
            self._delete_from_all_collections(file_path)

            ids = []
            metadatas = []
            for idx, (chunk, start, end) in enumerate(chunks):
                chunk_id = f"{file_id}:{idx}"
                ids.append(chunk_id)
                metadatas.append(
                    {
                        "file_path": file_path,
                        "file_name": os.path.basename(file_path),
                        "relative_path": rel_path,
                        "extension": extension,
                        "size": size,
                        "modified": modified,
                        "hash": file_hash,
                        "chunk_index": idx,
                        "chunk_start": start,
                        "chunk_end": end,
                        "preview": chunk[:200].replace("\n", " ").strip(),
                    }
                )

            # Upsert to ChromaDB
            collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

            # Update CSV index
            self.file_index[file_path] = {
                'hash': file_hash,
                'size': size,
                'modified': modified,
                'indexed_at': datetime.now().isoformat(),
                'chunks': len(chunks),
                'target': target,
            }

            return 'indexed'

        except Exception:
            return 'error'
    
    def index_directory(self, directory: str, progress=None) -> Dict[str, int]:
        """Index all files in directory with batching."""
        stats = {'indexed': 0, 'unchanged': 0, 'skipped': 0, 'removed': 0, 'error': 0}
        seen_files = set()

        file_paths = list(
            iter_files(directory, self.config.exclude_dirs, self.config.skip_hidden)
        )
        total = len(file_paths)

        if progress:
            progress(0, total, stats, None, "start")
            for i, file_path in enumerate(file_paths, 1):
                seen_files.add(file_path)
                result = self.index_file(file_path, directory)
                stats[result] += 1
                progress(i, total, stats, file_path, result)
        else:
            with tqdm(total=total, desc="Indexing files") as pbar:
                for file_path in file_paths:
                    seen_files.add(file_path)
                    result = self.index_file(file_path, directory)
                    stats[result] += 1
                    pbar.update(1)

        # Remove stale files from index
        stale_paths = [p for p in self.file_index.keys() if p not in seen_files]
        for stale in stale_paths:
            if self.remove_file(stale):
                stats['removed'] += 1

        # Save CSV index
        self._save_csv_index()

        if progress:
            progress(total, total, stats, None, "done")

        return stats
    
    def search(
        self,
        query: str,
        k: int = 5,
        file_type: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search for files - INSTANT (no LLM)"""
        try:
            # Prepare where filter
            where_filter = {}
            if file_type:
                ext = file_type.lower()
                if not ext.startswith("."):
                    ext = f".{ext}"
                where_filter["extension"] = ext

            targets = self._route_query_targets(query, file_type)
            if not targets:
                return []

            threshold = (
                min_similarity if min_similarity is not None else self.config.min_similarity
            )

            # Over-sample chunks to return top unique files
            oversample = max(k * 4, k)

            best_by_file: Dict[str, Dict[str, Any]] = {}

            for target in targets:
                collection, model = self._get_collection_and_model(target)
                total = collection.count()
                if total == 0:
                    continue

                query_embedding = model.encode(query).tolist()
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(oversample, total),
                    where=where_filter if where_filter else None,
                )

                if results['ids'] and results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        distance = results['distances'][0][i]
                        similarity = 1 - distance
                        metadata = results['metadatas'][0][i]
                        file_path = metadata['file_path']

                        current = best_by_file.get(file_path)
                        if current is None or similarity > current['similarity']:
                            best_by_file[file_path] = {
                                "file_path": file_path,
                                "file_name": metadata['file_name'],
                                "relative_path": metadata['relative_path'],
                                "extension": metadata['extension'],
                                "size": metadata['size'],
                                "modified": metadata['modified'],
                                "similarity": similarity,
                                "preview": metadata.get('preview', ''),
                            }

            sorted_results = sorted(
                best_by_file.values(), key=lambda x: x['similarity'], reverse=True
            )
            filtered = [
                {**r, "similarity": round(r["similarity"], 4)}
                for r in sorted_results
                if r["similarity"] >= threshold
            ]

            return filtered[:k]

        except Exception:
            return []
    
    def remove_file(self, file_path: str) -> bool:
        """Remove file from index"""
        try:
            self._delete_from_all_collections(file_path)

            if file_path in self.file_index:
                del self.file_index[file_path]
                self._save_csv_index()

            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        total = 0
        per_target = {}
        for target, collection in self.collections.items():
            try:
                count = collection.count()
            except Exception:
                count = 0
            per_target[target] = count
            total += count
        return {
            "total_files": total,
            "indexed_files": len(self.file_index),
            "db_path": self.db_path,
            "csv_index": self.csv_index_path,
            "collections": per_target,
        }

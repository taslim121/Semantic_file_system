import csv
import hashlib
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .indexing_pipeline import (
    chunk_text,
    iter_files,
    read_text_limited,
    should_index_file,
)
from .intent_router import IntentRouter


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


class VectorStore:
    """Embeddings + hybrid retrieval (semantic + lexical) with CSV file index."""

    def __init__(self, config):
        self.config = config
        self.db_path = config.vector_db_dir
        self.csv_index_path = os.path.join(self.db_path, "file_index.csv")

        # Keep CPU usage stable on low/mid-range laptops.
        if getattr(config, "max_threads", 0):
            try:
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                import torch

                max_threads = max(1, int(config.max_threads))
                torch.set_num_threads(max_threads)
                torch.set_num_interop_threads(max(1, max_threads // 2))
            except Exception:
                pass

        try:
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    chroma_product_telemetry_impl="lsfs.chroma_no_telemetry.NoOpTelemetry",
                    chroma_telemetry_impl="lsfs.chroma_no_telemetry.NoOpTelemetry",
                ),
            )
        except Exception:
            os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
            self.client = chromadb.PersistentClient(path=self.db_path)

        # Models are loaded lazily to avoid startup memory spikes.
        self.model_names: Dict[str, str] = {
            "default": config.embedding_default_model,
            "code": config.embedding_code_model,
        }
        self.models: Dict[str, SentenceTransformer] = {}

        self.collections: Dict[str, Any] = {
            "default": self.client.get_or_create_collection(
                name="files_default", metadata={"hnsw:space": "cosine"}
            ),
            "code": self.client.get_or_create_collection(
                name="files_code", metadata={"hnsw:space": "cosine"}
            ),
        }

        self.intent_router: Optional[IntentRouter] = None
        if config.intent_enabled and config.intent_labels:
            self.intent_router = IntentRouter(
                config.embedding_intent_model,
                config.intent_labels,
                config.intent_min_confidence,
                config.intent_fallback_target,
            )

        self.file_index = self._load_csv_index()

    def _get_model(self, target: str) -> SentenceTransformer:
        if target not in self.model_names:
            target = "default"
        if target not in self.models:
            model_name = self.model_names.get(target) or self.model_names["default"]
            self.models[target] = SentenceTransformer(model_name)
        return self.models[target]

    def _load_csv_index(self) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.csv_index_path):
            with open(self.csv_index_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    path = row.get("file_path", "")
                    if not path:
                        continue
                    index[path] = {
                        "hash": row.get("hash", ""),
                        "size": _safe_int(row.get("size"), 0),
                        "modified": row.get("modified", ""),
                        "indexed_at": row.get("indexed_at", ""),
                        "chunks": _safe_int(row.get("chunks"), 0),
                        "target": row.get("target", "default"),
                    }
        return index

    def _save_csv_index(self) -> None:
        os.makedirs(
            os.path.dirname(self.csv_index_path)
            if os.path.dirname(self.csv_index_path)
            else self.db_path,
            exist_ok=True,
        )
        with open(self.csv_index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file_path",
                    "hash",
                    "size",
                    "modified",
                    "indexed_at",
                    "chunks",
                    "target",
                ],
            )
            writer.writeheader()
            for file_path, data in self.file_index.items():
                writer.writerow(
                    {
                        "file_path": file_path,
                        "hash": data.get("hash", ""),
                        "size": data.get("size", 0),
                        "modified": data.get("modified", ""),
                        "indexed_at": data.get("indexed_at", ""),
                        "chunks": data.get("chunks", 0),
                        "target": data.get("target", "default"),
                    }
                )

    def _generate_file_hash(self, file_path: str) -> str:
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    def _generate_file_id(self, file_path: str) -> str:
        return hashlib.sha256(file_path.encode()).hexdigest()

    def _resolve_target_for_extension(self, extension: str) -> str:
        ext = (extension or "").lower()
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        if ext in self.config.code_extensions:
            return "code"
        return "default"

    def _delete_from_all_collections(self, file_path: str) -> None:
        for collection in self.collections.values():
            try:
                collection.delete(where={"file_path": file_path})
            except Exception:
                continue

    def _route_query_targets(self, query: str, file_type: Optional[str]) -> List[str]:
        if file_type:
            return [self._resolve_target_for_extension(file_type)]

        if self.intent_router:
            target, _, _ = self.intent_router.route(query)
            if target == "all":
                return ["default", "code"]
            if target in self.collections:
                return [target]

        if self.config.intent_fallback_target == "all":
            return ["default", "code"]
        return ["default"]

    def _needs_reindex(self, file_path: str) -> bool:
        if file_path not in self.file_index:
            return True
        try:
            stat = os.stat(file_path)
        except OSError:
            return False

        current_size = stat.st_size
        current_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
        entry = self.file_index[file_path]
        if entry.get("size") == current_size and entry.get("modified") == current_modified:
            return False
        return self._generate_file_hash(file_path) != entry.get("hash")

    def _lexical_score(self, query_tokens: List[str], row: Dict[str, Any]) -> float:
        if not query_tokens:
            return 0.0
        content_tokens = set(
            _tokenize(
                f"{row.get('file_name', '')} "
                f"{row.get('relative_path', '')} "
                f"{row.get('preview', '')}"
            )
        )
        if not content_tokens:
            return 0.0
        hits = sum(1 for t in query_tokens if t in content_tokens)
        return hits / max(1, len(set(query_tokens)))

    def _hybrid_score(self, semantic: float, lexical: float) -> float:
        alpha = self.config.hybrid_alpha
        return (alpha * semantic) + ((1.0 - alpha) * lexical)

    def index_file(self, file_path: str, root_dir: str) -> str:
        """Index one file: returns indexed|unchanged|skipped|error."""
        try:
            if not should_index_file(
                file_path,
                max_file_size_mb=self.config.max_file_size_mb,
                exclude_extensions=self.config.exclude_extensions,
                include_extensions=self.config.include_extensions,
                skip_hidden=self.config.skip_hidden,
                max_text_bytes=self.config.max_text_bytes,
            ):
                if file_path in self.file_index:
                    self.remove_file(file_path)
                return "skipped"

            if not self._needs_reindex(file_path):
                return "unchanged"

            content = read_text_limited(file_path, self.config.max_text_bytes)
            if not content or not content.strip():
                return "skipped"

            extension = os.path.splitext(file_path)[1].lower()
            target = self._resolve_target_for_extension(extension)
            model = self._get_model(target)
            collection = self.collections[target]

            # Slightly tighter chunks for code improves precision.
            is_code = extension in self.config.code_extensions
            chunk_size = max(300, int(self.config.chunk_size * (0.75 if is_code else 1.0)))
            chunk_overlap = self.config.chunk_overlap
            chunks = chunk_text(
                content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_chunks=self.config.max_chunks_per_file,
            )
            if not chunks:
                return "skipped"

            chunk_texts = [c[0] for c in chunks]
            embeddings = model.encode(
                chunk_texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            ).tolist()

            rel_path = os.path.relpath(file_path, root_dir)
            file_id = self._generate_file_id(file_path)
            file_hash = self._generate_file_hash(file_path)
            size = os.path.getsize(file_path)
            modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

            self._delete_from_all_collections(file_path)

            ids: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            documents: List[str] = []
            for idx, (chunk, start, end) in enumerate(chunks):
                preview = chunk[:220].replace("\n", " ").strip()
                ids.append(f"{file_id}:{idx}")
                documents.append(preview)
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
                        "preview": preview,
                        "target": target,
                    }
                )

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            self.file_index[file_path] = {
                "hash": file_hash,
                "size": size,
                "modified": modified,
                "indexed_at": datetime.now().isoformat(),
                "chunks": len(chunks),
                "target": target,
            }
            return "indexed"
        except Exception:
            return "error"

    def index_directory(self, directory: str, progress=None) -> Dict[str, int]:
        stats = {"indexed": 0, "unchanged": 0, "skipped": 0, "removed": 0, "error": 0}
        seen_files: Set[str] = set()

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

        stale_paths = [p for p in self.file_index.keys() if p not in seen_files]
        for stale in stale_paths:
            if self.remove_file(stale):
                stats["removed"] += 1

        self._save_csv_index()
        if progress:
            progress(total, total, stats, None, "done")
        return stats

    def search(
        self,
        query: str,
        k: int = 5,
        file_type: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        try:
            k = max(1, int(k))
            query_tokens = _tokenize(query)

            where_filter = None
            single_type = None
            if file_type:
                single_type = file_type.lower()
                if not single_type.startswith("."):
                    single_type = f".{single_type}"
                where_filter = {"extension": single_type}

            normalized_types: Optional[Set[str]] = None
            if file_types:
                normalized = set()
                for ext in file_types:
                    x = (ext or "").strip().lower()
                    if not x:
                        continue
                    if not x.startswith("."):
                        x = f".{x}"
                    normalized.add(x)
                if normalized:
                    normalized_types = normalized

            targets = self._route_query_targets(query, single_type)
            if not targets:
                targets = ["default"]

            threshold = (
                float(min_similarity)
                if min_similarity is not None
                else float(self.config.min_similarity)
            )
            threshold = max(0.0, min(1.0, threshold))

            oversample = max(k * 6, int(self.config.search_candidates))
            best_by_file: Dict[str, Dict[str, Any]] = {}

            for target in targets:
                collection = self.collections.get(target)
                if collection is None:
                    continue
                total = collection.count()
                if total == 0:
                    continue

                model = self._get_model(target)
                query_embedding = model.encode(query).tolist()
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(oversample, total),
                    where=where_filter,
                    include=["metadatas", "distances"],
                )

                ids = results.get("ids", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                if not ids:
                    continue

                for i in range(len(ids)):
                    meta = metadatas[i]
                    if not meta:
                        continue
                    file_path = meta.get("file_path")
                    if not file_path:
                        continue
                    ext = (meta.get("extension") or "").lower()
                    if normalized_types and ext not in normalized_types:
                        continue

                    semantic_score = 1.0 - float(distances[i])
                    lexical_score = self._lexical_score(query_tokens, meta)
                    hybrid_score = self._hybrid_score(semantic_score, lexical_score)

                    current = best_by_file.get(file_path)
                    if current is None or hybrid_score > current["hybrid_score"]:
                        best_by_file[file_path] = {
                            "file_path": file_path,
                            "file_name": meta.get("file_name", ""),
                            "relative_path": meta.get("relative_path", ""),
                            "extension": ext,
                            "size": _safe_int(meta.get("size"), 0),
                            "modified": meta.get("modified", ""),
                            "preview": meta.get("preview", ""),
                            "semantic_score": round(semantic_score, 4),
                            "lexical_score": round(lexical_score, 4),
                            "hybrid_score": hybrid_score,
                        }

            if not best_by_file and targets != ["default", "code"] and file_type is None:
                # Broader fallback search when intent routing is too narrow.
                for target in ["default", "code"]:
                    if target in targets:
                        continue
                    collection = self.collections.get(target)
                    if collection is None:
                        continue
                    total = collection.count()
                    if total == 0:
                        continue
                    model = self._get_model(target)
                    query_embedding = model.encode(query).tolist()
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(oversample, total),
                        where=where_filter,
                        include=["metadatas", "distances"],
                    )
                    ids = results.get("ids", [[]])[0]
                    metadatas = results.get("metadatas", [[]])[0]
                    distances = results.get("distances", [[]])[0]
                    for i in range(len(ids)):
                        meta = metadatas[i]
                        if not meta:
                            continue
                        file_path = meta.get("file_path")
                        if not file_path:
                            continue
                        ext = (meta.get("extension") or "").lower()
                        if normalized_types and ext not in normalized_types:
                            continue
                        semantic_score = 1.0 - float(distances[i])
                        lexical_score = self._lexical_score(query_tokens, meta)
                        hybrid_score = self._hybrid_score(semantic_score, lexical_score)
                        current = best_by_file.get(file_path)
                        if current is None or hybrid_score > current["hybrid_score"]:
                            best_by_file[file_path] = {
                                "file_path": file_path,
                                "file_name": meta.get("file_name", ""),
                                "relative_path": meta.get("relative_path", ""),
                                "extension": ext,
                                "size": _safe_int(meta.get("size"), 0),
                                "modified": meta.get("modified", ""),
                                "preview": meta.get("preview", ""),
                                "semantic_score": round(semantic_score, 4),
                                "lexical_score": round(lexical_score, 4),
                                "hybrid_score": hybrid_score,
                            }

            rows = sorted(
                best_by_file.values(),
                key=lambda x: x["hybrid_score"],
                reverse=True,
            )

            filtered = [
                r for r in rows if r["semantic_score"] >= threshold or r["lexical_score"] > 0.0
            ]
            if self.config.always_return_k and len(filtered) < k:
                # Keep top-ranked items even below threshold to guarantee usable output.
                filtered = rows[: max(k, len(rows))]

            final = []
            for r in filtered[:k]:
                final.append(
                    {
                        **r,
                        "similarity": round(r["hybrid_score"], 4),
                    }
                )

            if self.config.always_return_k and len(final) < k and self.file_index:
                used_paths = {r.get("file_path") for r in final}
                fallback_rows = []
                for path, data in self.file_index.items():
                    if path in used_paths:
                        continue
                    rel_path = os.path.relpath(path, self.config.root_dir)
                    file_name = os.path.basename(path)
                    lexical = self._lexical_score(
                        query_tokens,
                        {
                            "file_name": file_name,
                            "relative_path": rel_path,
                            "preview": "",
                        },
                    )
                    fallback_rows.append(
                        {
                            "file_path": path,
                            "file_name": file_name,
                            "relative_path": rel_path,
                            "extension": os.path.splitext(path)[1].lower(),
                            "size": _safe_int(data.get("size"), 0),
                            "modified": data.get("modified", ""),
                            "preview": "",
                            "semantic_score": 0.0,
                            "lexical_score": round(lexical, 4),
                            "hybrid_score": lexical * (1.0 - self.config.hybrid_alpha),
                            "similarity": round(lexical * (1.0 - self.config.hybrid_alpha), 4),
                        }
                    )
                fallback_rows.sort(key=lambda x: x["similarity"], reverse=True)
                for row in fallback_rows:
                    final.append(row)
                    if len(final) >= k:
                        break

            return final
        except Exception:
            return []

    def remove_file(self, file_path: str) -> bool:
        try:
            self._delete_from_all_collections(file_path)
            if file_path in self.file_index:
                del self.file_index[file_path]
                self._save_csv_index()
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        total_chunks = 0
        per_target = {}
        for target, collection in self.collections.items():
            try:
                count = collection.count()
            except Exception:
                count = 0
            per_target[target] = count
            total_chunks += count
        return {
            "total_files": len(self.file_index),
            "total_chunks": total_chunks,
            "indexed_files": len(self.file_index),
            "db_path": self.db_path,
            "csv_index": self.csv_index_path,
            "collections": per_target,
        }

import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import LSFSConfig
from .llm_handler import OllamaHandler
from .vector_store import VectorStore


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\._\-]+", (text or "").lower())


class LocalLSFS:
    """Local semantic file system with hybrid retrieval and NL parsing."""

    def __init__(self, config: LSFSConfig):
        self.config = config
        self.root_dir = config.root_dir
        self.vector_store = VectorStore(config)
        self.ollama_handler: Optional[OllamaHandler] = None
        if config.ollama_enabled:
            try:
                self.ollama_handler = OllamaHandler(
                    model=config.ollama_model, base_url=config.ollama_url
                )
            except Exception:
                self.ollama_handler = None

        if config.auto_mount:
            self._mount_root()

    def _mount_root(self) -> None:
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

        if self.vector_store.get_stats().get("total_files", 0) == 0:
            self.reindex_all()

    def _resolve_path(self, rel_path: str) -> str:
        rel = (rel_path or "").strip().replace("\\", os.sep).replace("/", os.sep)
        abs_path = os.path.abspath(os.path.join(self.root_dir, rel))
        root_abs = os.path.abspath(self.root_dir)
        if not abs_path.startswith(root_abs):
            raise ValueError("Path escapes root directory")
        return abs_path

    def create_file(self, file_name: str, content: str = "") -> Dict[str, Any]:
        try:
            file_path = self._resolve_path(file_name)
            parent = os.path.dirname(file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            if content.strip():
                self.vector_store.index_file(file_path, self.root_dir)
                self.vector_store._save_csv_index()
            return {
                "success": True,
                "file_path": file_path,
                "relative_path": file_name,
                "message": f"Created file: {file_name}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_directory(self, dir_name: str) -> Dict[str, Any]:
        try:
            dir_path = self._resolve_path(dir_name)
            os.makedirs(dir_path, exist_ok=True)
            return {"success": True, "dir_path": dir_path, "message": f"Created directory: {dir_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def write_file(self, file_name: str, content: str, append: bool = False) -> Dict[str, Any]:
        try:
            file_path = self._resolve_path(file_name)
            parent = os.path.dirname(file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)
            self.vector_store.index_file(file_path, self.root_dir)
            self.vector_store._save_csv_index()
            return {
                "success": True,
                "file_path": file_path,
                "mode": "appended" if append else "written",
                "message": f"Content {'appended to' if append else 'written to'} {file_name}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_file(self, file_name: str) -> Dict[str, Any]:
        try:
            file_path = self._resolve_path(file_name)
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_name}"}
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return {"success": True, "file_name": file_name, "content": content, "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_files(
        self,
        query: str,
        k: Optional[int] = None,
        file_type: Optional[str] = None,
        file_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        k = max(1, int(k or self.config.default_results))
        candidate_k = max(k, int(self.config.search_candidates))

        candidates = self.vector_store.search(
            query=query,
            k=candidate_k,
            file_type=file_type,
            file_types=file_types,
            min_similarity=self.config.min_similarity,
        )

        final_results = candidates
        if (
            self.ollama_handler
            and self.config.ollama_rerank_enabled
            and len(candidates) > 1
        ):
            top_n = min(len(candidates), int(self.config.ollama_rerank_top_n))
            reranked = self.ollama_handler.rerank_results(
                query=query, candidates=candidates[:top_n], final_k=k
            )
            if reranked:
                used_paths = {r.get("file_path") for r in reranked}
                merged = reranked + [c for c in candidates if c.get("file_path") not in used_paths]
                final_results = merged

        results = final_results[:k]
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
        }

    def list_files(self, subdir: str = "") -> Dict[str, Any]:
        try:
            target_dir = self._resolve_path(subdir) if subdir else self.root_dir
            if not os.path.exists(target_dir):
                return {"success": False, "error": f"Directory not found: {subdir}"}

            files = []
            dirs = []
            for item in os.listdir(target_dir):
                item_path = os.path.join(target_dir, item)
                if os.path.isdir(item_path):
                    dirs.append(item)
                else:
                    stat = os.stat(item_path)
                    files.append(
                        {
                            "name": item,
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        }
                    )

            return {
                "success": True,
                "path": subdir or "/",
                "directories": sorted(dirs),
                "files": sorted(files, key=lambda x: x["name"].lower()),
                "total": len(files) + len(dirs),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_file(self, file_name: str) -> Dict[str, Any]:
        try:
            file_path = self._resolve_path(file_name)
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_name}"}

            if os.path.isfile(file_path):
                os.remove(file_path)
                self.vector_store.remove_file(file_path)
            else:
                shutil.rmtree(file_path)
                self.vector_store.remove_file(file_path)
            return {"success": True, "message": f"Deleted: {file_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        try:
            src_path = self._resolve_path(source)
            dst_path = self._resolve_path(destination)
            if not os.path.exists(src_path):
                return {"success": False, "error": f"Source not found: {source}"}

            parent = os.path.dirname(dst_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            self.vector_store.remove_file(src_path)
            shutil.move(src_path, dst_path)
            if os.path.isfile(dst_path):
                self.vector_store.index_file(dst_path, self.root_dir)
                self.vector_store._save_csv_index()
            return {"success": True, "message": f"Moved {source} to {destination}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        try:
            src_path = self._resolve_path(source)
            dst_path = self._resolve_path(destination)
            if not os.path.exists(src_path):
                return {"success": False, "error": f"Source not found: {source}"}

            parent = os.path.dirname(dst_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            shutil.copy2(src_path, dst_path)
            if os.path.isfile(dst_path):
                self.vector_store.index_file(dst_path, self.root_dir)
                self.vector_store._save_csv_index()
            return {"success": True, "message": f"Copied {source} to {destination}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def reindex_all(self, progress=None) -> Dict[str, Any]:
        try:
            stats = self.vector_store.index_directory(self.root_dir, progress=progress)
            return {
                "success": True,
                "stats": stats,
                "message": (
                    f"Indexed {stats['indexed']} files, "
                    f"{stats['unchanged']} unchanged, "
                    f"{stats.get('skipped', 0)} skipped, "
                    f"{stats.get('removed', 0)} removed, "
                    f"{stats['error']} errors"
                ),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        return {"success": True, "root_dir": self.root_dir, "vector_stats": self.vector_store.get_stats()}

    def _parse_search_query(self, raw_query: str) -> Tuple[str, List[str]]:
        query = (raw_query or "").strip()
        lower = query.lower()
        file_types: List[str] = []

        split_idx = -1
        if " with " in lower:
            split_idx = lower.find(" with ")
            filter_part = lower[split_idx + 6 :]
        elif " in " in lower:
            split_idx = lower.find(" in ")
            filter_part = lower[split_idx + 4 :]
        else:
            filter_part = ""

        if split_idx != -1:
            query = raw_query[:split_idx].strip()

        for prefix in ["files related to ", "files about ", "related to ", "about "]:
            if query.lower().startswith(prefix):
                query = query[len(prefix) :].strip()
                break

        mapping = {
            "json": [".json"],
            "typescript": [".ts", ".tsx"],
            "javascript": [".js", ".jsx"],
            "python": [".py"],
            "yaml": [".yaml", ".yml"],
            "markdown": [".md"],
            "text": [".txt"],
        }
        if filter_part:
            tokens = [t.strip() for t in re.split(r",|and", filter_part) if t.strip()]
            for token in tokens:
                if token in mapping:
                    file_types.extend(mapping[token])
                elif token.startswith("."):
                    file_types.append(token.lower())
                elif token.isalpha():
                    file_types.append(f".{token.lower()}")

        for ext in re.findall(r"\.[a-z0-9]{1,8}", lower):
            file_types.append(ext.lower())

        file_types = list(dict.fromkeys(file_types))
        return (query if query else raw_query), file_types

    def _levenshtein(self, a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
            prev = curr
        return prev[-1]

    def _best_keyword_score(self, tokens: List[str], keywords: List[str]) -> float:
        best = 0.0
        for token in tokens:
            for kw in keywords:
                dist = self._levenshtein(token, kw)
                score = 1.0 - (dist / max(len(token), len(kw), 1))
                if score > best:
                    best = score
        return best

    def _extract_quoted(self, text: str) -> Optional[str]:
        match = re.search(r'"([^"]+)"|\'([^\']+)\'', text)
        if not match:
            return None
        return match.group(1) or match.group(2)

    def _parse_two_paths(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        m = re.search(r"from\s+(.+?)\s+to\s+(.+)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        m = re.search(r"(.+?)\s*->\s*(.+)", text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        m = re.search(r"(.+?)\s+to\s+(.+)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return None, None

    def _parse_command_heuristic(self, raw: str) -> Tuple[Optional[str], Dict[str, Any], float]:
        text = (raw or "").strip()
        lower = text.lower()
        tokens = _tokenize(text)
        if not tokens:
            return None, {}, 0.0

        if lower.startswith(("search ", "find ")):
            query = text.split(" ", 1)[1] if " " in text else ""
            parsed_query, file_types = self._parse_search_query(query)
            return "search", {"query": parsed_query, "file_types": file_types}, 1.0
        if lower in ("index", "reindex", "scan"):
            return "reindex", {}, 1.0
        if lower in ("status", "stats", "info"):
            return "stats", {}, 1.0
        if lower.startswith("list"):
            subdir = text[5:].strip()
            return "list", {"subdir": subdir}, 1.0
        if lower.startswith("read "):
            return "read", {"file_name": text[5:].strip()}, 1.0
        if lower.startswith("delete "):
            return "delete", {"file_name": text[7:].strip()}, 1.0

        ops = {
            "search": ["search", "find", "lookup", "query", "seek"],
            "list": ["list", "show", "ls", "dir", "browse"],
            "read": ["read", "open", "view", "cat"],
            "create": ["create", "make", "new", "touch", "write"],
            "delete": ["delete", "remove", "rm", "erase"],
            "move": ["move", "rename"],
            "copy": ["copy", "duplicate"],
            "reindex": ["index", "reindex", "scan"],
            "stats": ["status", "stats", "info"],
        }
        scores = {op: self._best_keyword_score(tokens, kws) for op, kws in ops.items()}
        best_op = max(scores, key=scores.get)
        best_score = scores[best_op]
        if best_score < 0.70:
            return None, {}, best_score

        if best_op == "search":
            parsed_query, file_types = self._parse_search_query(text)
            return "search", {"query": parsed_query, "file_types": file_types}, best_score
        if best_op == "list":
            subdir = text.split(" in ", 1)[1].strip() if " in " in lower else ""
            return "list", {"subdir": subdir}, best_score
        if best_op == "read":
            return "read", {"file_name": self._extract_quoted(text) or tokens[-1]}, best_score
        if best_op == "delete":
            return "delete", {"file_name": self._extract_quoted(text) or tokens[-1]}, best_score
        if best_op == "move":
            src, dst = self._parse_two_paths(text)
            if src and dst:
                return "move", {"source": src, "destination": dst}, best_score
        if best_op == "copy":
            src, dst = self._parse_two_paths(text)
            if src and dst:
                return "copy", {"source": src, "destination": dst}, best_score
        if best_op == "create":
            if any(t in tokens for t in ["dir", "folder", "directory"]):
                return "create_dir", {"dir_name": self._extract_quoted(text) or tokens[-1]}, best_score
            return "create_file", {"file_name": self._extract_quoted(text) or tokens[-1]}, best_score
        if best_op == "reindex":
            return "reindex", {}, best_score
        if best_op == "stats":
            return "stats", {}, best_score
        return None, {}, best_score

    def _execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if operation == "search":
            return self.search_files(
                params.get("query", ""),
                params.get("k", self.config.default_results),
                file_types=params.get("file_types"),
            )
        if operation == "list":
            return self.list_files(params.get("subdir", ""))
        if operation == "read":
            return self.read_file(params.get("file_name", ""))
        if operation == "create_file":
            return self.create_file(params.get("file_name", ""), params.get("content", ""))
        if operation == "create_dir":
            return self.create_directory(params.get("dir_name", ""))
        if operation == "write":
            return self.write_file(
                params.get("file_name", ""),
                params.get("content", ""),
                params.get("append", False),
            )
        if operation == "delete":
            return self.delete_file(params.get("file_name", ""))
        if operation == "move":
            return self.move_file(params.get("source", ""), params.get("destination", ""))
        if operation == "copy":
            return self.copy_file(params.get("source", ""), params.get("destination", ""))
        if operation == "reindex":
            return self.reindex_all()
        if operation == "stats":
            return self.get_stats()
        return {"success": False, "error": "Unknown command"}

    def parse_and_execute(self, user_input: str) -> Dict[str, Any]:
        raw = (user_input or "").strip()
        if not raw:
            return {"success": False, "error": "Empty command"}

        op, params, _ = self._parse_command_heuristic(raw)
        if op:
            return self._execute_operation(op, params)

        if self.ollama_handler:
            parsed = self.ollama_handler.parse_command(raw)
            operation = parsed.get("operation")
            params = parsed.get("parameters", {})
            confidence = float(parsed.get("confidence", 0.0))
            if operation and operation != "error" and confidence >= 0.35:
                return self._execute_operation(operation, params)

        return {"success": False, "error": "Unknown command. Type 'help' for available commands."}

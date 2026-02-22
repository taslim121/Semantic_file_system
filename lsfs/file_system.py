from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from .vector_store import VectorStore
from .config import LSFSConfig
import os
import shutil

class LocalLSFS:
    """Local Semantic File System with instant search (no LLM in search loop)"""
    
    def __init__(self, config: LSFSConfig):
        self.config = config
        self.root_dir = config.root_dir
        self.vector_store = VectorStore(config)
        self.ollama_handler = None
        if config.ollama_enabled:
            try:
                from .llm_handler import OllamaHandler

                self.ollama_handler = OllamaHandler(
                    model=config.ollama_model, base_url=config.ollama_url
                )
            except Exception:
                self.ollama_handler = None
        
        # Mount and index
        if config.auto_mount:
            self._mount_root()
    
    def _mount_root(self):
        """Mount and index root directory"""
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        
        # Initial index if empty
        if self.vector_store.get_stats().get("total_files", 0) == 0:
            self.reindex_all()
    
    def create_file(self, file_name: str, content: str = "") -> Dict[str, Any]:
        """Create a new file"""
        try:
            file_path = os.path.join(self.root_dir, file_name)
            
            # Create parent directories if needed
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_name) else self.root_dir, exist_ok=True)
            
            # Create file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Index file if it has content
            if content.strip():
                self.vector_store.index_file(file_path, self.root_dir)
                self.vector_store._save_csv_index()
            
            return {
                "success": True,
                "file_path": file_path,
                "relative_path": file_name,
                "message": f"Created file: {file_name}"
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_directory(self, dir_name: str) -> Dict[str, Any]:
        """Create a directory"""
        try:
            dir_path = os.path.join(self.root_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            
            return {
                "success": True,
                "dir_path": dir_path,
                "message": f"Created directory: {dir_name}"
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def write_file(self, file_name: str, content: str, append: bool = False) -> Dict[str, Any]:
        """Write content to file"""
        try:
            file_path = os.path.join(self.root_dir, file_name)
            
            mode = 'a' if append else 'w'
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)
            
            # Reindex file
            self.vector_store.index_file(file_path, self.root_dir)
            self.vector_store._save_csv_index()
            
            return {
                "success": True,
                "file_path": file_path,
                "mode": "appended" if append else "written",
                "message": f"Content {'appended to' if append else 'written to'} {file_name}"
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def read_file(self, file_name: str) -> Dict[str, Any]:
        """Read file content"""
        try:
            file_path = os.path.join(self.root_dir, file_name)
            
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_name}"}
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return {
                "success": True,
                "file_name": file_name,
                "content": content,
                "size": len(content)
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_files(
        self,
        query: str,
        k: int = 5,
        file_type: Optional[str] = None,
        file_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Semantic search for files - INSTANT (no LLM)"""
        results = self.vector_store.search(query, k, file_type, file_types)
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results)
        }

    def _parse_search_query(self, raw_query: str) -> Tuple[str, List[str]]:
        """Extract file-type filters from a query like '... with json and typescript'."""
        import re

        query = raw_query.strip()
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

        # Remove filler phrases
        for prefix in ["files related to ", "files about ", "related to ", "about "]:
            if query.lower().startswith(prefix):
                query = query[len(prefix) :].strip()
                break

        if filter_part:
            filter_part = filter_part.replace("files", " ").replace("file", " ")
            tokens = [t.strip() for t in re.split(r",|and", filter_part) if t.strip()]

            mapping = {
                "json": [".json"],
                "typescript": [".ts", ".tsx"],
                "ts": [".ts"],
                "tsx": [".tsx"],
                "javascript": [".js", ".jsx"],
                "js": [".js"],
                "jsx": [".jsx"],
                "python": [".py"],
                "py": [".py"],
                "yaml": [".yaml", ".yml"],
                "yml": [".yml"],
                "md": [".md"],
                "markdown": [".md"],
                "text": [".txt"],
                "txt": [".txt"],
            }

            for token in tokens:
                if token in mapping:
                    file_types.extend(mapping[token])
                elif token.startswith("."):
                    file_types.append(token)
                elif token.isalpha():
                    file_types.append(f".{token}")

        # Also capture explicit .ext in the full query
        for ext in re.findall(r"\.[a-z0-9]{1,6}", lower):
            file_types.append(ext)

        # Deduplicate
        file_types = list(dict.fromkeys(file_types))
        return query if query else raw_query, file_types

    def _tokenize(self, text: str) -> List[str]:
        import re

        return re.findall(r"[a-z0-9\._\-]+", text.lower())

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

    def _fuzzy_score(self, token: str, word: str) -> float:
        if not token or not word:
            return 0.0
        if token == word:
            return 1.0
        dist = self._levenshtein(token, word)
        max_len = max(len(token), len(word))
        return 1.0 - (dist / max_len)

    def _best_keyword_score(self, tokens: List[str], keywords: List[str]) -> float:
        best = 0.0
        for token in tokens:
            for kw in keywords:
                score = self._fuzzy_score(token, kw)
                if score > best:
                    best = score
        return best

    def _extract_quoted(self, text: str) -> Optional[str]:
        import re

        match = re.search(r'"([^"]+)"|\'([^\']+)\'', text)
        if not match:
            return None
        return match.group(1) or match.group(2)

    def _parse_two_paths(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        import re

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
        text = raw.strip()
        lower = text.lower()
        tokens = self._tokenize(text)

        if not tokens:
            return None, {}, 0.0

        # Direct matches for common operations
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

        # Fuzzy detection
        ops = {
            "search": ["search", "find", "lookup", "look", "query", "seek"],
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

        # Threshold tuned for typo tolerance
        if best_score < 0.72:
            return None, {}, best_score

        if best_op == "search":
            parsed_query, file_types = self._parse_search_query(text)
            return "search", {"query": parsed_query, "file_types": file_types}, best_score
        if best_op == "list":
            subdir = ""
            if " in " in lower:
                subdir = text.lower().split(" in ", 1)[1].strip()
            return "list", {"subdir": subdir}, best_score
        if best_op == "read":
            file_name = self._extract_quoted(text) or tokens[-1]
            return "read", {"file_name": file_name}, best_score
        if best_op == "delete":
            file_name = self._extract_quoted(text) or tokens[-1]
            return "delete", {"file_name": file_name}, best_score
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
                dir_name = self._extract_quoted(text) or tokens[-1]
                return "create_dir", {"dir_name": dir_name}, best_score
            file_name = self._extract_quoted(text) or tokens[-1]
            return "create_file", {"file_name": file_name}, best_score
        if best_op == "reindex":
            return "reindex", {}, best_score
        if best_op == "stats":
            return "stats", {}, best_score

        return None, {}, best_score

    def _execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if operation == "search":
            return self.search_files(
                params.get("query", ""),
                self.config.default_results,
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
    
    def list_files(self, subdir: str = "") -> Dict[str, Any]:
        """List files in directory"""
        try:
            target_dir = os.path.join(self.root_dir, subdir) if subdir else self.root_dir
            
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
                    files.append({
                        "name": item,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            return {
                "success": True,
                "path": subdir or "/",
                "directories": dirs,
                "files": files,
                "total": len(files) + len(dirs)
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_file(self, file_name: str) -> Dict[str, Any]:
        """Delete a file or directory"""
        try:
            file_path = os.path.join(self.root_dir, file_name)
            
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_name}"}
            
            if os.path.isfile(file_path):
                os.remove(file_path)
                self.vector_store.remove_file(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                # Remove all files in directory from index
            
            return {
                "success": True,
                "message": f"Deleted: {file_name}"
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Move or rename a file"""
        try:
            src_path = os.path.join(self.root_dir, source)
            dst_path = os.path.join(self.root_dir, destination)
            
            if not os.path.exists(src_path):
                return {"success": False, "error": f"Source not found: {source}"}
            
            # Create destination directory
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Move file
            # Remove from index first
            self.vector_store.remove_file(src_path)
            shutil.move(src_path, dst_path)
            
            # Update index
            self.vector_store.remove_file(src_path)
            if os.path.isfile(dst_path):
                self.vector_store.index_file(dst_path, self.root_dir)
                self.vector_store._save_csv_index()
            
            return {
                "success": True,
                "message": f"Moved {source} to {destination}"
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy a file"""
        try:
            src_path = os.path.join(self.root_dir, source)
            dst_path = os.path.join(self.root_dir, destination)
            
            if not os.path.exists(src_path):
                return {"success": False, "error": f"Source not found: {source}"}
            
            # Create destination directory
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Copy file
            # Copy and index
            shutil.copy2(src_path, dst_path)
            
            # Index new file
            if os.path.isfile(dst_path):
                self.vector_store.index_file(dst_path, self.root_dir)
                self.vector_store._save_csv_index()
            
            return {
                "success": True,
                "message": f"Copied {source} to {destination}"
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def reindex_all(self, progress=None) -> Dict[str, Any]:
        """Re-index all files in root directory"""
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
                )
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file system statistics"""
        return {
            "success": True,
            "root_dir": self.root_dir,
            "vector_stats": self.vector_store.get_stats()
        }
    
    def parse_and_execute(self, user_input: str) -> Dict[str, Any]:
        """Parse and execute commands WITHOUT LLM - instant execution"""
        raw = user_input.strip()
        if not raw:
            return {"success": False, "error": "Empty command"}

        # Heuristic parsing first (fast, typo-tolerant)
        op, params, score = self._parse_command_heuristic(raw)
        if op:
            return self._execute_operation(op, params)

        # Optional LLM parsing fallback
        if self.ollama_handler:
            parsed = self.ollama_handler.parse_command(raw)
            operation = parsed.get("operation")
            parameters = parsed.get("parameters", {})
            if operation and operation != "error":
                return self._execute_operation(operation, parameters)

        return {
            "success": False,
            "error": "Unknown command. Type 'help' for available commands.",
        }

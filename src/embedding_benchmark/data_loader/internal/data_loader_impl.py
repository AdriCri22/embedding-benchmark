"""
Data loader implementation for the embedding benchmark suite.
"""
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from ..interface import DataLoaderInterface
from embedding_benchmark.utils import get_skip_dirs


class DataLoaderImpl(DataLoaderInterface):
    """Implementation of data loading."""

    def __init__(self):
        self._skip_dirs = get_skip_dirs()
        self._heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    
    def load_chunks(self, test_data_dir: Path) -> List[Dict[str, Any]]:
        """Load all documents from test_data directory and return chunks."""
        files = self._candidate_files(test_data_dir)
        all_chunks = []
        for f in files:
            all_chunks.extend(self._chunk_file(f, test_data_dir))
        return all_chunks
    
    def load_all_documents(self, test_data_dir: Path) -> List[Dict[str, Any]]:
        """Load all documents with format-specific chunking."""
        files = self._candidate_files(test_data_dir)
        all_chunks = []
        for f in files:
            all_chunks.extend(self._chunk_file_format_aware(f, test_data_dir))
        return all_chunks
    
    def _candidate_files(self, test_data_dir: Path) -> List[Path]:
        files = []
        for folder in ("docs", "api", "hr", "legal", "engineering", "ops", "finance"):
            base = test_data_dir / folder
            if not base.is_dir():
                continue
            for p in base.rglob("*"):
                if p.suffix in {".md", ".json", ".jsonl", ".csv"} and not any(part in self._skip_dirs for part in p.parts):
                    files.append(p)
        return sorted(set(files))
    
    def _chunk_file(self, path: Path, test_data_dir: Path) -> List[Dict[str, Any]]:
        rel = path.relative_to(test_data_dir).as_posix()
        try:
            if path.suffix == ".md":
                lines = path.read_text(encoding="utf-8").splitlines()
            elif path.suffix == ".json":
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return [{"text": json.dumps(data, ensure_ascii=False), "path": rel, "format": "json", "heading": "", "line": 0}]
            elif path.suffix == ".jsonl":
                chunks = []
                with open(path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line: continue
                        try:
                            record = json.loads(line)
                            text = json.dumps(record, ensure_ascii=False)
                            chunks.append({"text": text, "path": rel, "format": "jsonl", "heading": f"line_{line_num}", "line": line_num})
                        except: continue
                return chunks
            elif path.suffix == ".csv":
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    chunks = []
                    for row_num, row in enumerate(reader, 1):
                        text = " | ".join(f"{k}: {v}" for k, v in row.items())
                        chunks.append({"text": text, "path": rel, "format": "csv", "heading": f"row_{row_num}", "line": row_num})
                    return chunks
            else:
                return []
        except:
            return []
        
        if path.suffix != ".md":
            return []
        
        chunks = []
        heading = ""
        start = 1
        body = []
        def flush():
            text = "\n".join(body).strip()
            if text:
                chunks.append({"text": text, "path": rel, "format": "md", "heading": heading, "line": start})
        for number, line in enumerate(lines, start=1):
            match = self._heading_re.match(line)
            if match:
                flush()
                heading = match.group(2).strip()
                start = number
                body = [line]
            else:
                body.append(line)
        flush()
        return chunks
    
    def _chunk_file_format_aware(self, path: Path, test_data_dir: Path) -> List[Dict[str, Any]]:
        rel = path.relative_to(test_data_dir).as_posix()
        try:
            if path.suffix == ".md":
                content = path.read_text(encoding="utf-8")
                return self._chunk_markdown(content, rel)
            elif path.suffix == ".json":
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._chunk_json(data, rel)
            elif path.suffix == ".jsonl":
                chunks = []
                with open(path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            text = json.dumps(record, ensure_ascii=False)
                            chunks.append({"text": text, "path": rel, "format": "jsonl", "heading": f"line_{line_num}", "line": line_num})
                        except:
                            continue
                return chunks
            elif path.suffix == ".csv":
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    chunks = []
                    for row_num, row in enumerate(reader, 1):
                        text = " | ".join(f"{k}: {v}" for k, v in row.items())
                        chunks.append({"text": text, "path": rel, "format": "csv", "heading": f"row_{row_num}", "line": row_num})
                    return chunks
            else:
                return []
        except Exception as e:
            print(f"   ⚠️  Error loading {path}: {e}")
            return []
    
    def _chunk_markdown(self, content: str, rel_path: str) -> List[Dict[str, Any]]:
        lines = content.splitlines()
        chunks = []
        heading = ""
        start = 1
        body = []
        def flush():
            text = "\n".join(body).strip()
            if text:
                chunks.append({"text": text, "path": rel_path, "format": "md", "heading": heading, "line": start})
        for number, line in enumerate(lines, start=1):
            match = self._heading_re.match(line)
            if match:
                flush()
                heading = match.group(2).strip()
                start = number
                body = [line]
            else:
                body.append(line)
        flush()
        return chunks
    
    def _chunk_json(self, data: Any, rel_path: str, max_chunk_chars: int = 2000) -> List[Dict[str, Any]]:
        text = json.dumps(data, ensure_ascii=False)
        if len(text) <= max_chunk_chars:
            return [{"text": text, "path": rel_path, "format": "json", "heading": "", "line": 0}]
        chunks = []
        for i in range(0, len(text), max_chunk_chars):
            chunk_text = text[i:i+max_chunk_chars]
            chunks.append({"text": chunk_text, "path": rel_path, "format": "json", "heading": f"chunk_{i//max_chunk_chars}", "line": i})
        return chunks
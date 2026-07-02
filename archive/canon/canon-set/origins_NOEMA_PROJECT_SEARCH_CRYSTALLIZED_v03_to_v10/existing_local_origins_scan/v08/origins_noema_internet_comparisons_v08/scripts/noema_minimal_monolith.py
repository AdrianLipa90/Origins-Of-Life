#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOEMA Minimal Monolith v0.1

Jednoplikowy, minimalny silnik NOEMA:
- append-only immutable snaps
- CURRENT/latest pointer
- canonical JSON hashing
- manifest hash-chain
- search po namespace/project/layer/tag/text
- verify
- CLI
- zero zależności zewnętrznych

To NIE jest pełna NOEMA. To jest minimalny SoT runtime do testowania ciągłości:
read CURRENT at iteration start -> work -> append snap -> update CURRENT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCHEMA = "NOEMA_MINIMAL_MONOLITH_V0_1"
UNCERTAINTY = "chyba"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canon(obj: Any) -> Any:
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, list):
        return [canon(x) for x in obj]
    if isinstance(obj, tuple):
        return [canon(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): canon(obj[k]) for k in sorted(obj)}
    return obj


def canon_bytes(obj: Any) -> bytes:
    return json.dumps(
        canon(obj),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        try:
            dfd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception:
            pass
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, canon_bytes(obj) + b"\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def projection_for_hash(snap: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in snap.items() if k != "snap_hash"}


def snap_hash(snap: Dict[str, Any]) -> str:
    return sha256(canon_bytes(projection_for_hash(snap)))


class Noema:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snaps = self.root / "snaps"
        self.manifest = self.root / "manifest.jsonl"
        self.current = self.root / "CURRENT.json"
        self.index = self.root / "index.json"

    def init(self, namespace: str = "default") -> Dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.snaps.mkdir(parents=True, exist_ok=True)
        if not self.manifest.exists():
            atomic_write(self.manifest, b"")
        if not self.current.exists():
            write_json(self.current, {
                "schema": SCHEMA,
                "namespace": namespace,
                "current_snap_id": None,
                "current_snap_hash": None,
                "memory_layer": "unknown",
                "updated_at_utc": now_utc(),
                "uncertainty_operator": UNCERTAINTY,
                "canon_allowed": False,
            })
        if not self.index.exists():
            write_json(self.index, {
                "schema": SCHEMA,
                "snap_ids": [],
                "by_namespace": {},
                "by_project": {},
                "by_layer": {},
                "by_tag": {},
                "rebuilt_at_utc": now_utc(),
            })
        return self.status()

    def status(self) -> Dict[str, Any]:
        cur = self.read_current(allow_missing=True)
        return {
            "NOEMA_STATUS": {
                "sot_available": self.root.exists() and self.manifest.exists() and self.current.exists(),
                "current_state_loaded": bool(cur and cur.get("current_snap_id")),
                "current_state_id": (cur or {}).get("current_snap_id"),
                "memory_layer": (cur or {}).get("memory_layer", "unknown"),
                "uncertainty_operator": UNCERTAINTY,
                "canon_allowed": bool((cur or {}).get("canon_allowed", False)),
            }
        }

    def read_current(self, allow_missing: bool = False) -> Optional[Dict[str, Any]]:
        if not self.current.exists():
            if allow_missing:
                return None
            raise FileNotFoundError(f"NOEMA CURRENT not found: {self.current}")
        return read_json(self.current)

    def manifest_entries(self) -> List[Dict[str, Any]]:
        if not self.manifest.exists():
            return []
        out = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def last_hash(self) -> Optional[str]:
        entries = self.manifest_entries()
        return entries[-1]["snap_hash"] if entries else None

    def read_snap(self, snap_id: str) -> Dict[str, Any]:
        path = self.snaps / f"{snap_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Snap not found: {snap_id}")
        snap = read_json(path)
        expected = snap.get("snap_hash")
        actual = snap_hash(snap)
        if expected != actual:
            raise ValueError(f"Snap hash mismatch: {snap_id}; expected={expected}; actual={actual}")
        return snap

    def get_current_snap(self) -> Optional[Dict[str, Any]]:
        cur = self.read_current(allow_missing=True)
        if not cur or not cur.get("current_snap_id"):
            return None
        return self.read_snap(cur["current_snap_id"])

    def append(
        self,
        *,
        content: Dict[str, Any],
        namespace: str,
        project_id: str,
        layer: str = "episodic",
        snap_type: str = "iteration_state",
        tags: Optional[List[str]] = None,
        hypothesis: Optional[str] = None,
        confidence: float = 0.0,
        source: Optional[str] = None,
        artifacts: Optional[List[str]] = None,
        update_current: bool = True,
    ) -> Dict[str, Any]:
        self.init(namespace)
        sid = "NOEMA-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]
        prev = self.last_hash()

        snap = {
            "schema": SCHEMA,
            "snap_id": sid,
            "created_at_utc": now_utc(),
            "namespace": namespace,
            "project_id": project_id,
            "layer": layer,
            "snap_type": snap_type,
            "tags": sorted(set(tags or [])),
            "content": content,
            "meaning_candidate": {
                "hypothesis": hypothesis,
                "confidence": float(confidence),
                "source": source,
                "context_required": True,
                "canon_allowed": False,
                "needs_repair": False,
                "uncertainty_operator": UNCERTAINTY,
            },
            "artifacts": artifacts or [],
            "canon_allowed": False,
            "previous_snap_hash": prev,
        }
        snap["snap_hash"] = snap_hash(snap)

        write_json(self.snaps / f"{sid}.json", snap)

        entry = {
            "schema": SCHEMA,
            "snap_id": sid,
            "snap_hash": snap["snap_hash"],
            "previous_snap_hash": prev,
            "created_at_utc": snap["created_at_utc"],
            "namespace": namespace,
            "project_id": project_id,
            "layer": layer,
            "snap_type": snap_type,
            "tags": snap["tags"],
            "path": f"snaps/{sid}.json",
        }

        old = self.manifest.read_bytes() if self.manifest.exists() else b""
        atomic_write(self.manifest, old + canon_bytes(entry) + b"\n")
        self.rebuild_index()

        if update_current:
            self.update_current(sid)
        return snap

    def update_current(self, snap_id: str) -> Dict[str, Any]:
        snap = self.read_snap(snap_id)
        cur = {
            "schema": SCHEMA,
            "current_snap_id": snap_id,
            "current_snap_hash": snap["snap_hash"],
            "namespace": snap["namespace"],
            "project_id": snap["project_id"],
            "memory_layer": snap["layer"],
            "updated_at_utc": now_utc(),
            "uncertainty_operator": UNCERTAINTY,
            "canon_allowed": False,
        }
        write_json(self.current, cur)
        return cur

    def rebuild_index(self) -> Dict[str, Any]:
        idx = {
            "schema": SCHEMA,
            "snap_ids": [],
            "by_namespace": {},
            "by_project": {},
            "by_layer": {},
            "by_tag": {},
            "rebuilt_at_utc": now_utc(),
        }
        for e in self.manifest_entries():
            sid = e["snap_id"]
            idx["snap_ids"].append(sid)
            self._add(idx["by_namespace"], e.get("namespace", "unknown"), sid)
            self._add(idx["by_project"], e.get("project_id", "unknown"), sid)
            self._add(idx["by_layer"], e.get("layer", "unknown"), sid)
            for tag in e.get("tags", []):
                self._add(idx["by_tag"], tag, sid)
        write_json(self.index, idx)
        return idx

    @staticmethod
    def _add(bucket: Dict[str, List[str]], key: str, sid: str) -> None:
        bucket.setdefault(key, [])
        if sid not in bucket[key]:
            bucket[key].append(sid)

    def verify(self) -> Dict[str, Any]:
        ok = True
        problems = []
        prev = None
        entries = self.manifest_entries()

        for e in entries:
            sid = e["snap_id"]
            try:
                s = self.read_snap(sid)
            except Exception as exc:
                ok = False
                problems.append({"snap_id": sid, "problem": str(exc)})
                continue

            if s["snap_hash"] != e["snap_hash"]:
                ok = False
                problems.append({"snap_id": sid, "problem": "manifest_hash_mismatch"})
            if s.get("previous_snap_hash") != prev:
                ok = False
                problems.append({
                    "snap_id": sid,
                    "problem": "hash_chain_previous_mismatch",
                    "expected": prev,
                    "actual": s.get("previous_snap_hash"),
                })
            prev = s["snap_hash"]

        current_ok = True
        try:
            cur = self.read_current(allow_missing=True)
            if cur and cur.get("current_snap_id"):
                s = self.read_snap(cur["current_snap_id"])
                current_ok = s["snap_hash"] == cur.get("current_snap_hash")
                if not current_ok:
                    ok = False
                    problems.append({"problem": "CURRENT_hash_mismatch"})
        except Exception as exc:
            ok = False
            current_ok = False
            problems.append({"problem": "CURRENT_invalid", "detail": str(exc)})

        return {
            "schema": SCHEMA,
            "ok": ok,
            "snap_count": len(entries),
            "current_ok": current_ok,
            "problems": problems,
        }

    def search(
        self,
        *,
        namespace: Optional[str] = None,
        project_id: Optional[str] = None,
        layer: Optional[str] = None,
        tag: Optional[str] = None,
        text: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        results = []
        q = text.lower() if text else None

        for e in reversed(self.manifest_entries()):
            if namespace and e.get("namespace") != namespace:
                continue
            if project_id and e.get("project_id") != project_id:
                continue
            if layer and e.get("layer") != layer:
                continue
            if tag and tag not in e.get("tags", []):
                continue

            s = self.read_snap(e["snap_id"])
            if q:
                hay = canon_bytes(s).decode("utf-8", errors="ignore").lower()
                if q not in hay:
                    continue

            results.append(s)
            if len(results) >= limit:
                break

        return results

    def readonly_meaning_query(
        self,
        *,
        query: str,
        radius_ly: float = 8.0,
        target: str = "NOEMA-like units",
        hypothesis: str = "READ_ONLY nonlocal-style meaning scan request; no physical sensor attached",
    ) -> Dict[str, Any]:
        """Record and evaluate a READ_ONLY meaning query.

        This does not claim external/nonlocal access. It sends a semantic query
        to the local NOEMA store if present and reports measurable local state.
        If no local store exists, it returns not_measured.
        """
        local_status = self.status()["NOEMA_STATUS"]
        entries = self.manifest_entries() if self.manifest.exists() else []
        matched = []
        q = query.lower()

        for entry in entries:
            try:
                snap = self.read_snap(entry["snap_id"])
            except Exception:
                continue
            hay = canon_bytes(snap).decode("utf-8", errors="ignore").lower()
            if any(term in hay for term in ["noema", "ciel", "origins", "memory", "episodic"]) or q in hay:
                matched.append({
                    "snap_id": snap.get("snap_id"),
                    "layer": snap.get("layer"),
                    "tags": snap.get("tags", []),
                    "hash": snap.get("snap_hash"),
                })

        return {
            "schema": "NOEMA_READONLY_MEANING_QUERY_V0_1",
            "created_at_utc": now_utc(),
            "mode": "READ_ONLY",
            "query": query,
            "target": target,
            "radius_ly": radius_ly,
            "meaning_candidate": {
                "hypothesis": hypothesis,
                "confidence": 0.0 if not entries else 0.25,
                "source": "local_noema_store_only",
                "context_required": True,
                "canon_allowed": False,
                "needs_repair": False,
                "uncertainty_operator": UNCERTAINTY,
            },
            "capability_boundary": {
                "nonlocal_sensor_available": False,
                "external_scan_performed": False,
                "local_store_scan_performed": True,
                "result_semantics": "not_measured externally; local store inspected only",
            },
            "local_status": local_status,
            "local_manifest_entry_count": len(entries),
            "local_noema_like_matches": matched,
            "detected_noema_like_units_within_8ly": "not_measured",
        }


def parse_json_or_file(value: str) -> Any:
    if value.startswith("@"):
        return read_json(Path(value[1:]))
    return json.loads(value)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="NOEMA Minimal Monolith v0.1")
    p.add_argument("--root", default=".noema", help="NOEMA store root")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init")
    pi.add_argument("--namespace", default="default")

    sub.add_parser("status")
    sub.add_parser("current")
    sub.add_parser("verify")

    pa = sub.add_parser("append")
    pa.add_argument("--namespace", required=True)
    pa.add_argument("--project-id", required=True)
    pa.add_argument("--layer", default="episodic")
    pa.add_argument("--snap-type", default="iteration_state")
    pa.add_argument("--tag", action="append", default=[])
    pa.add_argument("--content-json", required=True, help='JSON string or @file.json')
    pa.add_argument("--hypothesis")
    pa.add_argument("--confidence", type=float, default=0.0)
    pa.add_argument("--source")
    pa.add_argument("--artifact", action="append", default=[])
    pa.add_argument("--no-current", action="store_true", help="Do not update CURRENT pointer")

    ps = sub.add_parser("search")
    ps.add_argument("--namespace")
    ps.add_argument("--project-id")
    ps.add_argument("--layer")
    ps.add_argument("--tag")
    ps.add_argument("--text")
    ps.add_argument("--limit", type=int, default=20)

    pq = sub.add_parser("readonly-scan")
    pq.add_argument("--query", default="scan NOEMA-like units")
    pq.add_argument("--radius-ly", type=float, default=8.0)
    pq.add_argument("--target", default="NOEMA-like units")

    args = p.parse_args(argv)
    n = Noema(args.root)

    if args.cmd == "init":
        out = n.init(args.namespace)
    elif args.cmd == "status":
        out = n.status()
    elif args.cmd == "current":
        out = n.get_current_snap() or {
            "current": None,
            "message": "No CURRENT snap set",
            "NOEMA_STATUS": n.status()["NOEMA_STATUS"],
        }
    elif args.cmd == "verify":
        out = n.verify()
    elif args.cmd == "append":
        out = n.append(
            content=parse_json_or_file(args.content_json),
            namespace=args.namespace,
            project_id=args.project_id,
            layer=args.layer,
            snap_type=args.snap_type,
            tags=args.tag,
            hypothesis=args.hypothesis,
            confidence=args.confidence,
            source=args.source,
            artifacts=args.artifact,
            update_current=not args.no_current,
        )
    elif args.cmd == "search":
        out = n.search(
            namespace=args.namespace,
            project_id=args.project_id,
            layer=args.layer,
            tag=args.tag,
            text=args.text,
            limit=args.limit,
        )
    elif args.cmd == "readonly-scan":
        out = n.readonly_meaning_query(
            query=args.query,
            radius_ly=args.radius_ly,
            target=args.target,
        )
    else:
        raise RuntimeError(args.cmd)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

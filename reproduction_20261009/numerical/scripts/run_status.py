"""Record the latest attempt; a failed attempt never retains a current PASS."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def last_success(path):
    if not path.is_file():
        return None
    try:
        previous = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    if previous.get("passed") is True:
        return {key: value for key, value in previous.items() if key != "last_success"}
    return previous.get("last_success")


def write_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(record, handle, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def mark_not_run(path, parent_attempt_id):
    path = Path(path)
    record = {"passed": False, "status": "not_run", "updated_at_UTC": now(),
              "parent_attempt_id": parent_attempt_id,
              "reason": "This comparison has not run in the current full-run attempt."}
    previous = last_success(path)
    if previous is not None:
        record["last_success"] = previous
    write_record(path, record)


class RunStatus:
    def __init__(self, path, manifests):
        self.path = Path(path)
        self.manifests = manifests

    def __enter__(self):
        self.start = time.perf_counter()
        self.record = {"passed": False, "status": "running", "attempt_id": uuid.uuid4().hex,
                       "started_at_UTC": now(),
                       "manifest_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                           if Path(path).is_file() else None
                                           for name, path in self.manifests.items()}}
        previous = last_success(self.path)
        if previous is not None:
            self.record["last_success"] = previous
        write_record(self.path, self.record)
        return self

    def succeed(self, values):
        self.record.update(values)
        self.record.update({"passed": True, "status": "passed", "finished_at_UTC": now(),
                            "runtime_seconds": time.perf_counter() - self.start})
        self.record.pop("last_success", None)
        write_record(self.path, self.record)

    def __exit__(self, kind, error, traceback):
        if kind is None and self.record["status"] == "passed":
            return False
        if kind is None:
            kind, error = RuntimeError, RuntimeError("Run finished without declaring success.")
        self.record.update({"passed": False, "status": "failed", "finished_at_UTC": now(),
                            "runtime_seconds": time.perf_counter() - self.start,
                            "error_type": kind.__name__, "error": str(error)})
        write_record(self.path, self.record)
        if traceback is None:
            raise error
        return False

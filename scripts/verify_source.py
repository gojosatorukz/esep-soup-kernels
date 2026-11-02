"""Check that the portfolio retains the measured safe-6 runtime bytes. No GPU needed."""
from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "evidence/source-manifest.json").read_text())
for name, expected in manifest["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print(f"Verified {len(manifest['files'])} source files against measured safe-6.")

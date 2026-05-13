from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from model import export_model, load_model_from_nn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("net", help="Input Berserk-format .nn")
    args = ap.parse_args()

    src = Path(args.net)
    model = load_model_from_nn(src)
    with tempfile.NamedTemporaryFile(suffix=".nn", delete=False) as tmp:
        out = Path(tmp.name)

    try:
        export_model(model, out)
        a = src.read_bytes()
        b = out.read_bytes()
        if a != b:
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    raise SystemExit(
                        f"roundtrip mismatch at byte {i}: {x} != {y}")
            raise SystemExit(f"roundtrip size mismatch {len(a)} != {len(b)}")
        print(f"roundtrip ok: {src}")
    finally:
        out.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

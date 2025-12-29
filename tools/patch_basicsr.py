
import pathlib
base = pathlib.Path(r"path")
for p in base.rglob("*.py"):
    s = p.read_text(encoding="utf-8")
    if "functional_tensor" in s:
        print("Found in:", p)


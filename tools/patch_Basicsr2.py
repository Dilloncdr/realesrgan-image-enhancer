
import pathlib, shutil, sys

# adjust this path if your venv folder name or layout differs:
base = pathlib.Path(r"path")
target = base / "basicsr" / "data" / "degradations.py"

if not target.exists():
    print("ERROR: Could not find degradations.py at:", target)
    sys.exit(1)

bak = target.with_suffix(".py.bak")
shutil.copy2(target, bak)
print("Backup created:", bak)

text = target.read_text(encoding="utf-8")

old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
if old not in text:
    print("Note: expected import line not found. Searching for alternative occurrences...")
    # show some context and exit so you can inspect
    for line_no, line in enumerate(text.splitlines(), start=1):
        if "functional_tensor" in line or "rgb_to_grayscale" in line:
            print(f"{line_no}: {line.strip()}")
    print("If you want, paste the exact file contents here and I'll craft a tailored patch.")
    sys.exit(0)

replacement = (
    "try:\n"
    "    from torchvision.transforms.functional_tensor import rgb_to_grayscale\n"
    "except Exception:\n"
    "    # fallback for newer torchvision where functional_tensor was moved/renamed\n"
    "    from torchvision.transforms.functional import rgb_to_grayscale\n"
)

text2 = text.replace(old, replacement)
target.write_text(text2, encoding="utf-8")
print("Patched successfully:", target)


"""Download the exact public checkpoints used by Ripple; never train a model."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (
    ("NeoQuasar/Kronos-base", "2b554741eca47781b64468546e77fef3e85130e6", "Kronos-base"),
    ("NeoQuasar/Kronos-Tokenizer-base", "0e0117387f39004a9016484a186a908917e22426", "Kronos-Tokenizer-base"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only check whether checkpoint files exist; no network access")
    args = parser.parse_args()
    missing = []
    for repo_id, revision, folder in CHECKPOINTS:
        target = ROOT / "scenario-lab" / "models" / folder
        if not args.check:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id, revision=revision, local_dir=target,
                              allow_patterns=["config.json", "model.safetensors", "README.md", "LICENSE*"])
        for filename in ("config.json", "model.safetensors"):
            if not (target / filename).is_file():
                missing.append(str((target / filename).relative_to(ROOT)))
        print(f"{repo_id}@{revision}: {target.relative_to(ROOT)}")
    if missing:
        raise SystemExit("Missing checkpoint files: " + ", ".join(missing))
    print("Both local checkpoints are available.")


if __name__ == "__main__":
    main()

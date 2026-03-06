import argparse
import json
import subprocess


def _run(cmd: str) -> None:
    print(f"[reproduce] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducibility pipeline from config.")
    parser.add_argument("--config", default="configs/repro_full_and_v2.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    commands = cfg.get("commands", {})
    for key in ("validate", "benchmark", "evaluate"):
        if key in commands:
            _run(commands[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

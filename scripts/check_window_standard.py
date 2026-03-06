import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src_replica.runtime.standards import (
    CAN_WINDOW_SIZE_STANDARD,
    ETH_WINDOW_SIZE_STANDARD,
    enforce_window_standard,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check window-size standard compliance.")
    parser.add_argument("--config", required=True, help="Path to deployment JSON config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    can_w = int(cfg.get("can_window_size", -1))
    eth_w = int(cfg.get("eth_window_size", -1))
    enforce_window_standard(can_w, eth_w)
    print(
        f"PASS: can_window_size={can_w}, eth_window_size={eth_w} "
        f"(expected {CAN_WINDOW_SIZE_STANDARD}/{ETH_WINDOW_SIZE_STANDARD})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

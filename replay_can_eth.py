import sys

from src_replica.ids_cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] != "replay_can_eth":
        sys.argv.insert(1, "replay_can_eth")
    raise SystemExit(main())

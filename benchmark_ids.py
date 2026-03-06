import sys

from src_replica.ids_cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] != "benchmark_ids":
        sys.argv.insert(1, "benchmark_ids")
    raise SystemExit(main())

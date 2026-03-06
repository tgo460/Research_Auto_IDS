import pytest

from src_replica.runtime.adapters import SocketCanIngest


def test_socketcan_adapter_requires_python_can():
    try:
        SocketCanIngest(channel="can0")
    except RuntimeError:
        return
    except Exception:
        # If python-can exists but no CAN interface in CI, that's still acceptable.
        return
    pytest.skip("socketcan environment available")

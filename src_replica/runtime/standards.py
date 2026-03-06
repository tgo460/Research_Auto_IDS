CAN_WINDOW_SIZE_STANDARD = 100
ETH_WINDOW_SIZE_STANDARD = 1


def enforce_window_standard(can_window_size: int, eth_window_size: int) -> None:
    if int(can_window_size) != CAN_WINDOW_SIZE_STANDARD:
        raise ValueError(
            f"CAN window must be {CAN_WINDOW_SIZE_STANDARD}, got {can_window_size}"
        )
    if int(eth_window_size) != ETH_WINDOW_SIZE_STANDARD:
        raise ValueError(
            f"ETH window must be {ETH_WINDOW_SIZE_STANDARD}, got {eth_window_size}"
        )

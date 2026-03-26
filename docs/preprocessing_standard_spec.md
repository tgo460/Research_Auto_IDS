# Standard Preprocessing Spec

This project now uses a shared preprocessing contract for CAN and Ethernet across
training, validation, and testing.

## CAN

- Canonical raw fields:
  - `CAN_ID`, `DLC`, `D0`-`D7`, `Timestamp`, `Label`
- Normalization:
  - `CAN_ID`
    - keep values already in `[0, 1]`
    - scale 11-bit IDs by `2047`
    - scale extended IDs by `536870911`
  - `DLC`
    - convert byte-normalized values like `8/255` to `1.0`
    - scale classic CAN values by `8`
    - scale CAN-FD values by `64`
  - payload bytes
    - keep values already in `[0, 1]`
    - otherwise scale by `255`
- Engineered features:
  - `can_id_freq_global`
    - causal prefix frequency, not full-file frequency
  - `can_id_freq_win`
    - rolling local ID frequency
  - `payload_entropy`
    - per-frame entropy normalized by theoretical maximum
  - `inter_arrival`
    - causal running-max normalization
  - `inter_arrival_roll_mean`
    - rolling mean of normalized inter-arrival
  - `id_switch_rate_win`
    - rolling CAN ID switch rate

## Ethernet

- Canonical source for the hybrid light model:
  - packet metadata CSV, preferably `replica_eth_smoke/*_replica_packets.csv`
- Canonical packet fields:
  - `timestamp_sec`, `timestamp_usec`, `captured_len`, `original_len`
- Supervision contract:
  - replay, training, and evaluation CSVs must include `Label`
  - filename-derived ETH labels are not allowed
- Fallback packet fields:
  - `Packet_Length`
  - timestamps are synthesized if absent
- Standardized packet features:
  - `captured_len_norm`
  - `original_len_norm`
  - `length_ratio_norm`
  - `inter_arrival_norm`
  - `rolling_len_mean_norm`
  - `rolling_gap_mean_norm`
  - `packet_delta_norm`
- Image representation:
  - deterministic `32x32` grayscale image
  - built from window summary statistics and their complements
  - encoded as a metadata outer-product image
  - representation tag: `metadata_outer_v1`

## Research rationale

- preprocessing is causal and reproducible
- train, validation, and test share one feature definition
- runtime replay can reuse the same normalization and Ethernet image encoding
- non-causal file-global CAN statistics are removed from the actual computation path

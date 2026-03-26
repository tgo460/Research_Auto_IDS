from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PairSpec:
    name: str
    group: str
    can_file: str
    eth_npy_file: str


def training_pair_specs() -> List[PairSpec]:
    return [
        PairSpec("baseline_driving_01", "baseline", "can_normal_train.csv", "eth_driving_01_original_images-006.npy"),
        PairSpec("baseline_indoors_01", "baseline", "can_normal_train.csv", "eth_indoors_01_original_images.npy"),
        PairSpec("can_only_dos_driving_01", "can_only", "can_dos_train.csv", "eth_driving_01_original_images-006.npy"),
        PairSpec("can_only_fuzzy_driving_02", "can_only", "can_fuzzy_train.csv", "eth_driving_02_original_images-005.npy"),
        PairSpec("can_only_gear_driving_02", "can_only", "can_gear_train.csv", "eth_driving_02_original_images-005.npy"),
        PairSpec("can_only_rpm_indoors_01", "can_only", "can_rpm_train.csv", "eth_indoors_01_original_images.npy"),
        PairSpec("eth_only_driving_01", "eth_only", "can_normal_train.csv", "eth_driving_01_injected_images-003.npy"),
        PairSpec("eth_only_driving_02", "eth_only", "can_normal_train.csv", "eth_driving_02_injected_images-008.npy"),
        PairSpec("eth_only_indoors_01", "eth_only", "can_normal_train.csv", "eth_indoors_01_injected_images.npy"),
        PairSpec("eth_only_indoors_02", "eth_only", "can_normal_train.csv", "eth_indoors_02_injected_images.npy"),
        PairSpec("coordinated_dos_driving_01", "coordinated", "can_dos_train.csv", "eth_driving_01_injected_images-003.npy"),
        PairSpec("coordinated_fuzzy_driving_02", "coordinated", "can_fuzzy_train.csv", "eth_driving_02_injected_images-008.npy"),
        PairSpec("coordinated_gear_indoors_01", "coordinated", "can_gear_train.csv", "eth_indoors_01_injected_images.npy"),
        PairSpec("coordinated_rpm_indoors_02", "coordinated", "can_rpm_train.csv", "eth_indoors_02_injected_images.npy"),
    ]


def evaluation_pair_specs() -> List[PairSpec]:
    return [
        PairSpec("baseline_driving_01", "baseline", "can_normal_train.csv", "eth_driving_01_original_images-006.npy"),
        PairSpec("baseline_indoors_01", "baseline", "can_normal_train.csv", "eth_indoors_01_original_images.npy"),
        PairSpec("can_only_dos_driving_01", "can_only", "can_dos_train.csv", "eth_driving_01_original_images-006.npy"),
        PairSpec("can_only_fuzzy_driving_02", "can_only", "can_fuzzy_train.csv", "eth_driving_02_original_images-005.npy"),
        PairSpec("can_only_gear_driving_02", "can_only", "can_gear_train.csv", "eth_driving_02_original_images-005.npy"),
        PairSpec("can_only_rpm_indoors_01", "can_only", "can_rpm_train.csv", "eth_indoors_01_original_images.npy"),
        PairSpec("eth_only_driving_01", "eth_only", "can_normal_train.csv", "eth_driving_01_injected_images-003.npy"),
        PairSpec("eth_only_driving_02", "eth_only", "can_normal_train.csv", "eth_driving_02_injected_images-008.npy"),
        PairSpec("eth_only_indoors_01", "eth_only", "can_normal_train.csv", "eth_indoors_01_injected_images.npy"),
        PairSpec("eth_only_indoors_02", "eth_only", "can_normal_train.csv", "eth_indoors_02_injected_images.npy"),
        PairSpec("coordinated_dos_driving_01", "coordinated", "can_dos_train.csv", "eth_driving_01_injected_images-003.npy"),
        PairSpec("coordinated_fuzzy_driving_02", "coordinated", "can_fuzzy_train.csv", "eth_driving_02_injected_images-008.npy"),
        PairSpec("coordinated_gear_indoors_01", "coordinated", "can_gear_train.csv", "eth_indoors_01_injected_images.npy"),
        PairSpec("coordinated_rpm_indoors_02", "coordinated", "can_rpm_train.csv", "eth_indoors_02_injected_images.npy"),
    ]


def group_pair_specs(specs: List[PairSpec]) -> Dict[str, List[PairSpec]]:
    out: Dict[str, List[PairSpec]] = {}
    for spec in specs:
        out.setdefault(spec.group, []).append(spec)
    return out

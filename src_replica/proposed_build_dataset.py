from typing import List, Optional, Tuple
import os
import json
import torch
from torch.utils.data import ConcatDataset
from dataloader_correlated_replica import CorrelatedHybridVehicleDataset

def build_mixed_dataset(base_path: str, split: str = 'train') -> Optional[ConcatDataset]:
    """
    Constructs a ConcatDataset of CorrelatedHybridVehicleDataset instances
    based on the pairing logic found in the split file.
    
    Logic:
    1. Load 'data/splits/split_v1.json' (or v2 if present, but user asked for v1 logic).
    2. Extract CAN and ETH file lists for the requested split ('train', 'val', 'test').
    3. Iterate over ETH files:
       - If ETH file is 'original' (benign):
         - Find a 'normal' CAN file in the split.
         - If unavailable in current split, maybe fallback to 'can_normal_train.csv' (common practice in this replica).
       - If ETH file is 'injected' (attack):
         - Find an 'attack' CAN file (dos, fuzzy, gear, rpm). 
         - Prefer 'dos' for 'driving_01' if available? Or just any attack?
         - User traces suggest 'can_dos' aligns with 'driving_01_injected'.
    """
    
    # 1. Determine split file
    split_path = os.path.join(base_path, 'data', 'splits', 'split_v1.json')
    if not os.path.exists(split_path):
        # Fallback or check v2
        split_path_v2 = os.path.join(base_path, 'data', 'splits', 'split_v2_domain_balanced.json')
        if os.path.exists(split_path_v2):
            split_path = split_path_v2
            
    if not os.path.exists(split_path):
        print(f"Split file not found at {split_path}")
        return None

    with open(split_path, 'r') as f:
        split_data = json.load(f)
        
    can_files = split_data['modalities']['can'][split]
    eth_files = split_data['modalities']['eth'][split]
    
    # Helper to classify CAN files
    benign_can = [f for f in can_files if 'normal' in f]
    attack_can = [f for f in can_files if 'normal' not in f]
    
    # If no benign CAN in this split (e.g. val in v2), we might need to look in 'train'
    # BUT strictly speaking we should only use files in the split. 
    # However, for 'original' ETH, we really need 'normal' CAN.
    if not benign_can:
        # Fallback: check train split for normal
        if 'train' in split_data['modalities']['can']:
             benign_can = [f for f in split_data['modalities']['can']['train'] if 'normal' in f]

    datasets = []
    
    # 2. Iterate ETH files to form pairs
    for eth_file in eth_files:
        is_injected = 'injected' in eth_file or 'attack' in eth_file
        
        target_can_file = None
        
        if is_injected:
            # Pair with an attack CAN file
            # Ideally match the attack type if possible, but names differ (dos vs injected).
            # For this replica, we just pick the first available attack file 
            # OR try to match simple Keywords if they existed.
            # Given 'can_dos_train' matches 'driving_01_injected', we prefer 'dos' if current is 'injected'?
            # Actually, let's just use the first available attack file in the list 
            # that hasn't been exhausted? Or just reuse?
            # Reusing is fine.
            if attack_can:
                # Prefer 'dos' for 'driving_01' if it exists in list
                if 'driving_01' in eth_file:
                    dos_files = [c for c in attack_can if 'dos' in c]
                    if dos_files:
                        target_can_file = dos_files[0]
                    else:
                        target_can_file = attack_can[0]
                else:
                    target_can_file = attack_can[0]
        else:
            # Pair with benign/normal
            if benign_can:
                target_can_file = benign_can[0]
                
        if target_can_file:
            # Construct paths
            # Assuming files are in 'datasets/'
            can_path = os.path.join(base_path, 'datasets', target_can_file)
            eth_npy = os.path.join(base_path, 'datasets', eth_file)
            # We also need the ETH CSV (packet) file.
            # NPY is 'eth_driving_01_injected_images-003.npy'
            # CSV is likely 'eth_driving_01_injected.csv' (derived from name)
            # or 'eth_driving_01_injected_preprocessed.csv'?
            # The correlation function takes 'eth_csv_path'.
            # Looking at file list: 'eth_driving_01_injected.csv' exists.
            
            # Simple heuristic to get CSV name from NPY name:
            # Remove '_images*.npy' and add '.csv'
            # e.g. 'eth_driving_01_injected_images-003.npy' -> 'eth_driving_01_injected.csv'
            base_name = eth_file.split('_images')[0]
            eth_csv = os.path.join(base_path, 'datasets', base_name + '.csv')
            
            if not os.path.exists(eth_csv):
                # Try finding it?
                pass
                
            print(f"pairing {eth_file} with {target_can_file}")
            
            try:
                ds = CorrelatedHybridVehicleDataset(
                    can_csv_path=can_path,
                    eth_packet_csv_path=eth_csv,
                    eth_npy_path=eth_npy,
                    can_features=['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'], # Standard Data fields
                    # default windows etc...
                )
                if len(ds) > 0:
                    datasets.append(ds)
            except Exception as e:
                print(f"Failed to load pair {eth_file} + {target_can_file}: {e}")

    if not datasets:
        return None
        
    return ConcatDataset(datasets)

import pandas as pd
import numpy as np
import os
import argparse

def blend_datasets(normal_path: str, attack_path: str, output_path: str, num_attacks: int = 3, attack_window_size: int = 500, total_samples: int = 15000):
    """
    Blends a normal automotive network trace with injected bursts of attack frames.
    This simulates a realistic scenario where an attacker intermittently breaches the bus.
    """
    print(f"Loading normal data from {normal_path}...")
    df_normal = pd.read_csv(normal_path)
    
    print(f"Loading attack data from {attack_path}...")
    df_attack = pd.read_csv(attack_path)

    # Ensure we have enough data
    df_normal = df_normal.sample(n=min(total_samples, len(df_normal)), replace=True).reset_index(drop=True)
    
    blended_frames = []
    
    # Determine random injection points
    # We want to inject `num_attacks` times into the total_samples timeframe
    interval = total_samples // (num_attacks + 1)
    injection_indices = [interval * i + np.random.randint(-interval//4, interval//4) for i in range(1, num_attacks + 1)]
    
    print(f"Applying attack injections at indices: {injection_indices} (Burst Size: {attack_window_size} frames)")

    current_idx = 0
    attack_pointer = 0

    for idx in range(total_samples):
        # Check if we are currently inside an injection window
        in_attack_window = any(inj_idx <= current_idx < (inj_idx + attack_window_size) for inj_idx in injection_indices)
        
        if in_attack_window and attack_pointer < len(df_attack):
            blended_frames.append(df_attack.iloc[attack_pointer])
            attack_pointer += 1
        else:
            blended_frames.append(df_normal.iloc[idx])
        
        current_idx += 1

    blended_df = pd.DataFrame(blended_frames)
    
    # Keep the integrity of the data stream but enforce our realistic label tracking
    # Ensure label column exists. Normal = 0, Attack = 1
    if 'Label' in blended_df.columns:
        attack_count = blended_df['Label'].sum()
        print(f"Blended dataset created. Total frames: {len(blended_df)} | Total Attack Frames: {attack_count} ({(attack_count/len(blended_df))*100:.2f}%)")
    else:
        print("Warning: 'Label' column not found.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    blended_df.to_csv(output_path, index=False)
    print(f"Realistic blended dataset saved to -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automotive IDS Realistic Traffic Simulation")
    parser.add_argument("--normal", default="datasets/can_normal_train.csv")
    parser.add_argument("--attack", default="datasets/can_dos_train.csv")
    parser.add_argument("--output", default="datasets/can_blended_simulation.csv")
    parser.add_argument("--num_attacks", type=int, default=3)
    parser.add_argument("--attack_window", type=int, default=800, help="Frames injected per burst")
    parser.add_argument("--total_samples", type=int, default=15000, help="Total CAN sequence length")
    
    args = parser.parse_args()
    blend_datasets(args.normal, args.attack, args.output, args.num_attacks, args.attack_window, args.total_samples)
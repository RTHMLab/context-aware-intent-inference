import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import argparse

# --- Define Constants and Column Names (VERIFY THESE CAREFULLY AGAINST YOUR ACTUAL CSVs) ---
TRIALS_GESTURE_COL = "Gesture"
TRIALS_START_TIME_COL = "Start Time (Xsens)"
TRIALS_END_TIME_COL = "End Time (Xsens)"
# NEW: Column for the ground truth object in trials_df
GROUND_TRUTH_OBJECT_COL = "Object" # <--- VERIFY THIS COLUMN NAME IN YOUR sub3-trials.csv

OBJ_TIME_COL = "Time (in ms)"
OBJ_DEPTH_COLS = ["chair", "backpack", "dining table", "couch"] # These should be in meters

HUMAN_TIME_COL = "Time (in ms)"
HUMAN_POS_COLS = ["Pelvis x", "Pelvis y", "Pelvis z"]

ORI_TIME_COL = "Time (in ms)"
HUMAN_ORI_COLS = ["Pelvis x", "Pelvis y", "Pelvis z"] # Assuming Pelvis z is Yaw/Heading

# --- Command-line arguments ---
parser = argparse.ArgumentParser(description="Map detected object depths to human-object distances using Xsens pelvis pose.")
parser.add_argument("--trials_csv", required=True, help="Trial CSV generated from ANVIL annotations")
parser.add_argument("--object_positions_csv", required=True, help="Per-trial object depth CSV from extractObjectPositions.py")
parser.add_argument("--segment_position_csv", required=True, help="Xsens pelvis segment position CSV")
parser.add_argument("--segment_orientation_csv", required=True, help="Xsens pelvis segment orientation Euler CSV")
parser.add_argument("--output_csv", required=True, help="Output mapped distances CSV")
parser.add_argument("--output_plot_dir", default=None, help="Optional output directory for distance plots")
parser.add_argument("--validation_output_csv", default=None, help="Optional validation CSV path")
parser.add_argument("--skip_first_n_trials", type=int, default=0, help="Optional number of initial trials to skip")
args = parser.parse_args()

# --- Load Data ---
try:
    trials_df = pd.read_csv(args.trials_csv)
    object_data = pd.read_csv(args.object_positions_csv)
    human_pose_data = pd.read_csv(args.segment_position_csv)
    orientation_data = pd.read_csv(args.segment_orientation_csv)
except FileNotFoundError as e:
    print(f"Error loading data file: {e}. Please ensure all CSV files are in the correct directory.")
    exit()
except KeyError as e:
    print(f"Error: Missing expected column in trials.csv: {e}. Please check GROUND_TRUTH_OBJECT_COL.")
    exit()

# --- Apply initial filtering for trials (Trial 16 onwards) ---
# This ensures we only process trials that have depth data
# We'll use the original_trials_df to get the Ground Truth Object later if needed,
# or simply access it before slicing. Let's make a copy.
original_trials_df_for_gt = trials_df.copy() # Keep a copy of original trials_df before slicing

if args.skip_first_n_trials > 0:
    trials_df = trials_df.iloc[args.skip_first_n_trials:].reset_index(drop=True)
else:
    trials_df = trials_df.reset_index(drop=True)


# Ensure output folder exists
if args.output_plot_dir:
    os.makedirs(args.output_plot_dir, exist_ok=True)

if args.validation_output_csv:
    os.makedirs(os.path.dirname(args.validation_output_csv), exist_ok=True)

# Constants
DEPTH_POSITION_JUMP_THRESHOLD = 0.2  # Max allowed absolute position change (meters) for an object

all_trial_data = []  # Store final CSV data
validation_results = [] # NEW: To store results for validation report

# Function to compute absolute object position
def compute_absolute_position(human_x, human_y, depth, orientation_deg):
    orientation_rad = math.radians(orientation_deg)
    new_x = human_x + depth * math.cos(orientation_rad)
    new_y = human_y + depth * math.sin(orientation_rad)
    return new_x, new_y

# Process each trial separately
for trial_idx_in_filtered_df, trial in trials_df.iterrows():
    trial_num_orig = int(trial["Trial Number"]) if "Trial Number" in trials_df.columns else trial_idx_in_filtered_df + 1
    
    trial_gesture = trial[TRIALS_GESTURE_COL]
    # NEW: Get the ground truth object for this specific trial
    ground_truth_object = trial[GROUND_TRUTH_OBJECT_COL]
    
    trial_start_time = trial[TRIALS_START_TIME_COL]
    trial_end_time = trial[TRIALS_END_TIME_COL]

    print(f"Processing Trial {trial_num_orig} ({trial_gesture}). Ground Truth Object: {ground_truth_object}...")

    # --- Initialize/Reset variables for EACH NEW TRIAL ---
    last_valid_positions = {obj: (np.nan, np.nan) for obj in OBJ_DEPTH_COLS}
    
    human_x, human_y, human_z = 0.0, 0.0, 0.0
    orientation = 0.0 
    
    human_pose_idx, orient_idx, obj_data_idx = 0, 0, 0 

    trial_obj_df = object_data[
        (object_data[OBJ_TIME_COL] >= trial_start_time) & (object_data[OBJ_TIME_COL] <= trial_end_time)
    ].sort_values(by=OBJ_TIME_COL).reset_index(drop=True)
    
    trial_human_df = human_pose_data[
        (human_pose_data[HUMAN_TIME_COL] >= trial_start_time) & (human_pose_data[HUMAN_TIME_COL] <= trial_end_time)
    ].sort_values(by=HUMAN_TIME_COL).reset_index(drop=True)
    
    trial_orient_df = orientation_data[
        (orientation_data[ORI_TIME_COL] >= trial_start_time) & (orientation_data[ORI_TIME_COL] <= trial_end_time)
    ].sort_values(by=ORI_TIME_COL).reset_index(drop=True)

    all_timestamps = sorted(set(
        trial_human_df[HUMAN_TIME_COL].tolist() + 
        trial_obj_df[OBJ_TIME_COL].tolist() +
        trial_orient_df[ORI_TIME_COL].tolist()
    ))

    time_steps_for_plot = []
    distances_for_plot = {obj: [] for obj in OBJ_DEPTH_COLS}

    # Main loop: Iterate through every unique timestamp across all data streams for the current trial
    for current_timestamp in all_timestamps:
        # --- 1. Update Human Position and Orientation to the latest available ---
        while human_pose_idx < len(trial_human_df) and \
              trial_human_df.loc[human_pose_idx, HUMAN_TIME_COL] <= current_timestamp:
            human_row_temp = trial_human_df.loc[human_pose_idx]
            human_x, human_y, human_z = human_row_temp[HUMAN_POS_COLS].values
            human_pose_idx += 1 
        
        while orient_idx < len(trial_orient_df) and \
              trial_orient_df.loc[orient_idx, ORI_TIME_COL] <= current_timestamp:
            orientation_row_temp = trial_orient_df.loc[orient_idx]
            orientation = orientation_row_temp[HUMAN_ORI_COLS[2]] # Assuming Pelvis Z is yaw/heading
            orient_idx += 1
            
        # --- 2. Initialize current frame's object positions and distances ---
        current_object_abs_pos = {obj: last_valid_positions[obj] for obj in OBJ_DEPTH_COLS}
        current_object_distances = {obj: np.nan for obj in OBJ_DEPTH_COLS} 

        # --- 3. Process any new object detection data available at or before current_timestamp ---
        while obj_data_idx < len(trial_obj_df) and \
              trial_obj_df.loc[obj_data_idx, OBJ_TIME_COL] <= current_timestamp:
            
            obj_row_for_update = trial_obj_df.loc[obj_data_idx]
            
            for obj_name in OBJ_DEPTH_COLS: 
                depth_value = obj_row_for_update[obj_name]

                if depth_value > 0: 
                    new_obj_x, new_obj_y = compute_absolute_position(human_x, human_y, depth_value, orientation)

                    if not np.isnan(last_valid_positions[obj_name][0]):
                        prev_obj_x, prev_obj_y = last_valid_positions[obj_name]
                        pos_change = np.linalg.norm([new_obj_x - prev_obj_x, new_obj_y - prev_obj_y])

                        if pos_change > DEPTH_POSITION_JUMP_THRESHOLD:
                            new_obj_x, new_obj_y = prev_obj_x, prev_obj_y

                    last_valid_positions[obj_name] = (new_obj_x, new_obj_y)
                    current_object_abs_pos[obj_name] = (new_obj_x, new_obj_y)
            
            obj_data_idx += 1 

        # --- 4. Calculate Distances for the current_timestamp ---
        for obj_name in OBJ_DEPTH_COLS:
            obj_x, obj_y = current_object_abs_pos[obj_name]
            
            if not np.isnan(obj_x): 
                current_object_distances[obj_name] = np.linalg.norm([human_x - obj_x, human_y - obj_y])
            else:
                current_object_distances[obj_name] = np.nan 

        # --- 5. Store data for CSV output for the current_timestamp ---
        all_trial_data.append([
            current_timestamp, trial_num_orig, trial_gesture,
            current_object_abs_pos["chair"][0], current_object_abs_pos["chair"][1],
            current_object_abs_pos["backpack"][0], current_object_abs_pos["backpack"][1],
            current_object_abs_pos["dining table"][0], current_object_abs_pos["dining table"][1],
            current_object_abs_pos["couch"][0], current_object_abs_pos["couch"][1],
            human_x, human_y,
            current_object_distances["chair"], current_object_distances["backpack"],
            current_object_distances["dining table"], current_object_distances["couch"]
        ])

        # --- 6. Store data for plotting this trial ---
        time_steps_for_plot.append(current_timestamp - trial_start_time)
        for obj_name in distances_for_plot.keys():
            distances_for_plot[obj_name].append(current_object_distances[obj_name])

    # --- Save Distance vs Time Plot for Each Trial ---
    plt.figure(figsize=(10, 7)) 
    for obj_name in distances_for_plot.keys():
        if not all(np.isnan(d) for d in distances_for_plot[obj_name]):
             plt.plot(time_steps_for_plot, distances_for_plot[obj_name], label=obj_name)
    
    plt.xlabel("Time (ms from trial start)")
    plt.ylabel("Distance (m)")
    plt.title(f"Trial {trial_num_orig}_{trial_gesture}: Human-Object Distance")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if args.output_plot_dir:
        plot_filename = os.path.join(args.output_plot_dir, f"trial_{trial_num_orig}_{trial_gesture}_distance.png")
        plt.savefig(plot_filename, dpi=300)
        print(f"✅ Saved plot for Trial {trial_num_orig}, Gesture: {trial_gesture}")
    plt.close()

# Save final aggregated CSV file
columns = [
    "Time (ms)", "Trial Number", "Gesture",
    "Chair X", "Chair Y", "Backpack X", "Backpack Y",
    "Dining Table X", "Dining Table Y", "Couch X", "Couch Y",
    "Human X", "Human Y",
    "Distance Chair", "Distance Backpack", "Distance Dining Table", "Distance Couch"
]

df_final = pd.DataFrame(all_trial_data, columns=columns)
output_csv_filename = args.output_csv
df_final.to_csv(output_csv_filename, index=False)
print(f"\n✅ All distances computed, CSV saved successfully as {output_csv_filename}!")

# --- NEW: Validation Section ---
print("\n--- Running Validation Check at 3000ms ---")

validation_results = []

# Re-iterate through the original trials_df (including the 'Object' column)
# Using original_trials_df_for_gt to get ground truth for all trials including ones before filtered 15,
# then filtering based on the range we processed (trial_num_orig from 16 onwards)
for idx, trial in original_trials_df_for_gt.iterrows():
    trial_num = idx + 1 # Original trial number (1 to N)
    
    if trial_num < 16: # Skip trials that were not processed by the main loop
        continue

    ground_truth_object = trial[GROUND_TRUTH_OBJECT_COL]
    trial_start_time = trial[TRIALS_START_TIME_COL]

    # Calculate the target timestamp (3000ms from trial start)
    target_timestamp_ms = trial_start_time + 3000

    # Find the row in df_final closest to the target_timestamp for this trial
    # Filter for the correct trial number first
    trial_data_in_final_df = df_final[df_final["Trial Number"] == trial_num].copy()

    if trial_data_in_final_df.empty:
        print(f"Warning: No processed data found for Trial {trial_num}.")
        validation_results.append({
            "Trial Number": trial_num,
            "Ground Truth Object": ground_truth_object,
            "Target Time (ms)": target_timestamp_ms,
            "Nearest Detected Object": "N/A",
            "Match": "N/A",
            "Notes": "No processed data for trial"
        })
        continue

    # Find the row closest to target_timestamp_ms
    # Using idxmin() on the absolute difference to find the index of the closest timestamp
    closest_row_idx = (trial_data_in_final_df["Time (ms)"] - target_timestamp_ms).abs().idxmin()
    closest_row = trial_data_in_final_df.loc[closest_row_idx]
    
    # Extract distances at this closest timestamp
    distances_at_3000ms = {}
    for obj_col in OBJ_DEPTH_COLS:
        dist_col_name = f"Distance {obj_col.replace('dining table', 'Dining Table').title()}" # Adjust for column names in df_final
        distances_at_3000ms[obj_col] = closest_row[dist_col_name]

    # Find the object with the minimum non-NaN distance
    min_dist = np.inf
    nearest_detected_object = "None Detected"
    
    # Filter out NaN distances before finding the minimum
    valid_distances = {obj: dist for obj, dist in distances_at_3000ms.items() if not np.isnan(dist)}

    if valid_distances:
        min_dist_obj = min(valid_distances, key=valid_distances.get)
        min_dist = valid_distances[min_dist_obj]
        nearest_detected_object = min_dist_obj
    
    # Compare with ground truth
    is_match = (nearest_detected_object.lower() == ground_truth_object.lower())

    # --- NEW ADDITIONS FOR VALIDATION ---
    # Get the distance of the ground truth object at 3000ms
    # Ensure column name matches df_final: e.g., "Distance Chair", "Distance Backpack", etc.
    # Convert ground_truth_object name to match the column format in df_final
    gt_obj_col_name_in_df = f"Distance {ground_truth_object.replace('dining table', 'Dining Table').title()}"
    gt_object_distance_at_3000ms = closest_row[gt_obj_col_name_in_df]

    # Check if ground truth object is 0.8m or closer
    # Value is 1 if True, 0 if False (or if distance is NaN)
    is_gt_0_8m_closer = 1 if not np.isnan(gt_object_distance_at_3000ms) and gt_object_distance_at_3000ms <= 0.8 else 0
    
    # Check if ground truth object is 1m or closer
    # Value is 1 if True, 0 if False (or if distance is NaN)
    is_gt_1_0m_closer = 1 if not np.isnan(gt_object_distance_at_3000ms) and gt_object_distance_at_3000ms <= 1.0 else 0
    # --- END NEW ADDITIONS ---

    print(f"Trial {trial_num} (GT: {ground_truth_object}): Nearest detected at 3000ms: {nearest_detected_object} (Dist: {min_dist:.2f}m). Match: {is_match}")

    validation_results.append({
        "Trial Number": trial_num,
        "Ground Truth Object": ground_truth_object,
        "Target Time (ms)": target_timestamp_ms,
        "Actual Timestamp Used (ms)": closest_row["Time (ms)"],
        "Nearest Detected Object": nearest_detected_object,
        "Distance (m)": min_dist,
        "Match": 1 if is_match else 0, # Kept as 1/0 as per previous request
        # NEW COLUMNS
        "Ground Truth Object 0.8m or Closer": is_gt_0_8m_closer,
        "Ground Truth Object 1.0m or Closer": is_gt_1_0m_closer
    })

# Save validation results to CSV
validation_df = pd.DataFrame(validation_results)
if args.validation_output_csv:
    validation_df.to_csv(args.validation_output_csv, index=False)
    print(f"\n✅ Validation results saved to {args.validation_output_csv}")
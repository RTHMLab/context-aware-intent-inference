import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Extract per-trial object depths from YOLO/depth detections.")
parser.add_argument("--trials_csv", required=True, help="Trial CSV generated from ANVIL annotations")
parser.add_argument("--detections_csv", required=True, help="YOLO/depth detection CSV from processRawData.py")
parser.add_argument("--output_csv", required=True, help="Output object positions/depths CSV")
parser.add_argument("--skip_first_n_trials", type=int, default=0, help="Optional number of initial trials to skip")
args = parser.parse_args()

# Load trials CSV
testing_trials_df = pd.read_csv(args.trials_csv)

if args.skip_first_n_trials > 0:
    testing_trials_df = testing_trials_df.iloc[args.skip_first_n_trials:].reset_index(drop=True)
else:
    testing_trials_df = testing_trials_df.reset_index(drop=True)

# Load object detection data
object_data_df = pd.read_csv(args.detections_csv)

# Define column names from object_data_df
TIME_COL = "Time (in ms)"
OBJECT_CLASS_COL = "YOLO Class"
CONFIDENCE_COL = "YOLO Confidence"
DEPTH_COL = "Min Cluster Depth (mm)"

# Convert depth from mm to meters
# This conversion is necessary as the input CSV has depth in mm
object_data_df[DEPTH_COL] = object_data_df[DEPTH_COL] / 1000.0

# Object classes to track
OBJECTS_TO_KEEP = ["chair", "backpack", "dining table", "couch"]

# List to store processed data
processed_data = []

# Iterate through each testing trial
for trial_idx, row in testing_trials_df.iterrows():
    gesture = row["Gesture"]  # Gesture name
    start_time = row["Start Time (Xsens)"]  # Trial start time (Xsens)
    end_time = row["End Time (Xsens)"]  # Trial end time (Xsens)

    # Filter object data strictly within the trial window
    trial_object_data = object_data_df[
        (object_data_df[TIME_COL] >= start_time) & (object_data_df[TIME_COL] <= end_time)
    ]

    # Get unique timestamps in this trial
    unique_timestamps = trial_object_data[TIME_COL].unique()

    for timestamp in unique_timestamps:
        frame_data = trial_object_data[trial_object_data[TIME_COL] == timestamp]

        # Dictionary to store the best depth for each object (default to 0)
        object_depths = {obj: 0 for obj in OBJECTS_TO_KEEP}

        for obj in OBJECTS_TO_KEEP:
            obj_instances = frame_data[frame_data[OBJECT_CLASS_COL] == obj]

            if not obj_instances.empty:
                # Select the highest confidence detection
                best_instance = obj_instances.sort_values(by=CONFIDENCE_COL, ascending=False).iloc[0]

                # Use depth if it's non-zero, otherwise check the second-best instance
                # This logic is kept as is, but it will now apply to the 'Min Cluster Depth (mm)'
                depth_value = best_instance[DEPTH_COL]
                # The assumption here is that a '0' depth value from the clustering
                # also signifies an invalid reading that should be avoided if possible.
                if depth_value == 0 and len(obj_instances) > 1:
                    depth_value = obj_instances.sort_values(by=CONFIDENCE_COL, ascending=False).iloc[1][DEPTH_COL]

                object_depths[obj] = depth_value  # Store depth

        # Append processed row
        # trial_idx+1 is now based on the filtered DataFrame's new index (0, 1, 2, ... for the remaining trials)
        trial_number = row["Trial Number"] if "Trial Number" in testing_trials_df.columns else trial_idx + 1
        processed_data.append([timestamp, trial_number, gesture] + [object_depths[obj] for obj in OBJECTS_TO_KEEP])

# Convert processed data to DataFrame
output_df = pd.DataFrame(processed_data, columns=["Time (in ms)", "Trial Number", "Gesture"] + OBJECTS_TO_KEEP)

# Save to CSV
output_filename = args.output_csv
output_df.to_csv(output_filename, index=False)

print(f"✅ Object positions saved in {output_filename}")
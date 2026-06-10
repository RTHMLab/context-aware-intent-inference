import pandas as pd

# Load trials CSV
testing_trials_df = pd.read_csv("sub3-trials.csv")

# Ignore first 15 trials (rows)
testing_trials_df = testing_trials_df.iloc[15:].reset_index(drop=True)
# .reset_index(drop=True) is important to reset the DataFrame index after slicing,
# otherwise, row.iterrows() might behave unexpectedly with a fragmented index.

# Load object detection data (output from previous script with clustered depth)
object_data_df = pd.read_csv("yolov8nFT-conf6-output.csv")

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
        processed_data.append([timestamp, trial_idx+1+15, gesture] + [object_depths[obj] for obj in OBJECTS_TO_KEEP])

# Convert processed data to DataFrame
output_df = pd.DataFrame(processed_data, columns=["Time (in ms)", "Trial Number", "Gesture"] + OBJECTS_TO_KEEP)

# Save to CSV
output_filename = "object_positions_by_trial-sub3.csv"
output_df.to_csv(output_filename, index=False)

print(f"✅ Object positions saved in {output_filename}")
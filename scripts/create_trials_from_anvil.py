from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
import pandas as pd


GESTURE_TO_OBJECT = {
    "Sit-on-chair": "chair",
    "Sit-on-couch": "couch",
    "Sit-on-table": "dining table",
    "Sit-at-table": "dining table",
    "Clean-table": "dining table",
    "Pick-up-backpack": "backpack",
    "Pickup-backpack": "backpack",
    "Wear-backpack": "backpack",
    "Push-chair": "chair",
    "Stand-on-couch": "couch",
}


def normalize_session_name(name: str) -> str:
    return (
        name.lower()
        .replace("_averaged.anvil", "")
        .replace("_ja.csv", "")
        .replace(".anvil", "")
        .replace(".csv", "")
        .replace("-", "")
        .replace("_", "")
    )


def read_offsets(offset_csv: Path) -> dict:
    df = pd.read_csv(offset_csv)
    print("Offset CSV columns:", df.columns.tolist())

    # Try to infer columns
    session_col = None
    offset_col = None

    for col in df.columns:
        low = col.lower()
        if "file" in low or "session" in low or "subject" in low:
            session_col = col
        if "offset" in low:
            offset_col = col

    if session_col is None or offset_col is None:
        raise ValueError(
            f"Could not infer session/offset columns from {df.columns.tolist()}"
        )

    offsets = {}
    for _, row in df.iterrows():
        session_key = normalize_session_name(str(row[session_col]))
        offsets[session_key] = float(row[offset_col])

    return offsets


def parse_anvil(anvil_path: Path):
    tree = ET.parse(anvil_path)
    root = tree.getroot()

    rows = []
    for el in root.iter("el"):
        if "start" not in el.attrib:
            continue

        start_s = float(el.attrib["start"])
        end_s = float(el.attrib.get("end", start_s))

        gesture = None
        for child in list(el):
            if child.tag == "attribute" and child.attrib.get("name") == "type":
                gesture = (child.text or "").strip()
                break

        if not gesture:
            continue

        rows.append({
            "anvil_index": int(el.attrib.get("index", len(rows))),
            "gesture": gesture,
            "anvil_start_s": start_s,
            "anvil_end_s": end_s,
        })

    rows = sorted(rows, key=lambda r: r["anvil_start_s"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anvil_dir", default="context-a-if/annotations/Averaged_Annotations")
    parser.add_argument("--offset_csv", default="context-a-if/GroundTruth-Annotations-Offset.csv")
    parser.add_argument("--ja_dir", default="data/extracted_JointAngles")
    parser.add_argument("--rgbd_dir", default="data/Synced-color-depthPNG")
    parser.add_argument("--output_dir", default="data/trials")
    parser.add_argument("--pre_ms", type=float, default=2500.0)
    parser.add_argument("--post_ms", type=float, default=1000.0)
    args = parser.parse_args()

    anvil_dir = Path(args.anvil_dir)
    ja_dir = Path(args.ja_dir)
    rgbd_dir = Path(args.rgbd_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    offsets = read_offsets(Path(args.offset_csv))

    # Use RGB-D folder names as canonical output session names
    rgbd_sessions = {
        normalize_session_name(p.name): p.name
        for p in rgbd_dir.iterdir()
        if p.is_dir()
    }

    ja_files = {
        normalize_session_name(p.name): p
        for p in ja_dir.glob("*_ja.csv")
    }

    anvil_files = sorted(anvil_dir.glob("*_averaged.anvil"))

    for anvil_path in anvil_files:
        key = normalize_session_name(anvil_path.name)
        session_id = rgbd_sessions.get(key, key)

        print("\n" + "=" * 100)
        print(f"ANVIL: {anvil_path.name}")
        print(f"Session: {session_id}")

        if key not in offsets:
            print("  SKIP: offset not found")
            continue

        if key not in ja_files:
            print("  SKIP: matching JA CSV not found")
            continue

        offset_s = offsets[key]
        ja_path = ja_files[key]

        ja_df = pd.read_csv(ja_path, usecols=[0])
        first_xsens_ms = float(ja_df.iloc[0, 0])

        annotations = parse_anvil(anvil_path)

        output_rows = []
        for i, ann in enumerate(annotations, start=1):
            gesture = ann["gesture"]
            obj = GESTURE_TO_OBJECT.get(gesture, "")

            action_start_relative_s = ann["anvil_start_s"] - offset_s
            action_end_relative_s = ann["anvil_end_s"] - offset_s

            action_start_xsens_ms = first_xsens_ms + action_start_relative_s * 1000.0
            action_end_xsens_ms = first_xsens_ms + action_end_relative_s * 1000.0

            trial_start_ms = action_start_xsens_ms - args.pre_ms
            trial_end_ms = action_start_xsens_ms + args.post_ms

            output_rows.append({
                "Trial Number": i,
                "Gesture": gesture,
                "Object": obj,
                "Action Start Time (ANVIL s)": ann["anvil_start_s"],
                "Action End Time (ANVIL s)": ann["anvil_end_s"],
                "Offset (s)": offset_s,
                "Action Start Time (Xsens)": action_start_xsens_ms,
                "Action End Time (Xsens)": action_end_xsens_ms,
                "Start Time (Xsens)": trial_start_ms,
                "End Time (Xsens)": trial_end_ms,
            })

        out_df = pd.DataFrame(output_rows)
        out_csv = output_dir / f"{session_id}_trials.csv"
        out_df.to_csv(out_csv, index=False)

        print(f"  Offset: {offset_s}")
        print(f"  First Xsens ms: {first_xsens_ms}")
        print(f"  Trials: {len(out_df)}")
        print(f"  Wrote: {out_csv}")


if __name__ == "__main__":
    main()

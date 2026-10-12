"""Build a test-only VPR dataset from the Ford Multi-AV Seasonal dataset.

The Ford source tree is treated as read-only. Images are linked (by default)
or copied into the standard VPR-datasets-downloader layout. Image labels are
obtained by timestamp-aligning camera PNGs with ``/gps`` and converting GPS
coordinates to UTM. ``/pose_ground_truth`` is aligned and retained as metadata,
but is not used across dates because Ford's seasonal map frames differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import utm
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial import cKDTree
from tqdm import tqdm


CAMERAS = ("Center", "FL", "FR", "SL", "SR", "RL", "RR")
POSE_TOPIC = "/pose_ground_truth"
GPS_TOPIC = "/gps"
MIN_LOG_SYNC_RATIO = 0.90


@dataclass(frozen=True)
class ImageRecord:
    split: str
    source_path: Path
    image_timestamp_us: int
    pose_timestamp_ns: int
    gps_timestamp_ns: int
    pose_sync_error_ms: float
    gps_sync_error_ms: float
    utm_easting: float
    utm_northing: float
    utm_zone_number: int
    utm_zone_letter: str
    latitude: float
    longitude: float
    altitude: float
    map_x: float
    map_y: float
    map_z: float
    qx: float
    qy: float
    qz: float
    qw: float
    date: str
    vehicle: str
    log: str
    camera: str

    @property
    def output_name(self) -> str:
        # VPR convention:
        # easting, northing, zone, zone letter, lat, lon, pano id, tile,
        # heading, pitch, roll, height, timestamp, note, extension.
        fields = [
            f"{self.utm_easting:.3f}",
            f"{self.utm_northing:.3f}",
            str(self.utm_zone_number),
            self.utm_zone_letter,
            f"{self.latitude:.7f}",
            f"{self.longitude:.7f}",
            str(self.image_timestamp_us),
            "",
            "",
            "",
            "",
            "",
            self.date.replace("-", ""),
            f"{self.date}_{self.vehicle}_{self.log}_{self.camera}",
            self.source_path.suffix.lower(),
        ]
        return "@" + "@".join(fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Ford Multi-AV Seasonal data to the standard VPR test layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/Ford_seasonal"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "ford_seasonal",
    )
    parser.add_argument("--database-date", default="2017-07-24")
    parser.add_argument("--query-date", default="2017-10-26")
    parser.add_argument("--database-vehicle", default="V2")
    parser.add_argument("--query-vehicle", default="V2")
    parser.add_argument("--database-logs", nargs="+", default=None, metavar="LOG")
    parser.add_argument("--query-logs", nargs="+", default=None, metavar="LOG")
    parser.add_argument("--camera", choices=CAMERAS, default="Center")
    parser.add_argument(
        "--sample-distance",
        type=float,
        default=5.0,
        help="minimum planar distance in metres between consecutive retained frames",
    )
    parser.add_argument(
        "--max-sync-error-ms",
        type=float,
        default=50.0,
        help="discard images farther than this from the nearest ground-truth pose",
    )
    parser.add_argument(
        "--query-overlap-threshold",
        type=float,
        default=25.0,
        help="discard queries farther than this many metres from every database image; 0 disables filtering",
    )
    parser.add_argument(
        "--transfer-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="how selected source PNGs are represented in the output dataset",
    )
    parser.add_argument(
        "--max-images-per-log",
        type=int,
        default=0,
        help="debug/testing limit applied before spatial sampling; 0 means all images",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep compatible files already present in a partially generated output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read and validate inputs and report counts without creating output files",
    )
    args = parser.parse_args()

    if args.sample_distance < 0:
        parser.error("--sample-distance must be non-negative")
    if args.max_sync_error_ms <= 0:
        parser.error("--max-sync-error-ms must be greater than zero")
    if args.query_overlap_threshold < 0:
        parser.error("--query-overlap-threshold must be non-negative")
    if args.max_images_per_log < 0:
        parser.error("--max-images-per-log must be non-negative")
    return args


def natural_log_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix("Log")
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def discover_logs(raw_root: Path, date: str, vehicle: str, requested: Sequence[str] | None) -> list[str]:
    vehicle_dir = raw_root / date / vehicle
    if not vehicle_dir.is_dir():
        raise FileNotFoundError(f"Ford vehicle directory does not exist: {vehicle_dir}")

    available = sorted(
        (path.name for path in vehicle_dir.iterdir() if path.is_dir() and path.name.startswith("Log")),
        key=natural_log_key,
    )
    if not available:
        raise FileNotFoundError(f"No Log* directories found in {vehicle_dir}")

    if requested is None:
        return available

    missing = sorted(set(requested) - set(available), key=natural_log_key)
    if missing:
        raise FileNotFoundError(f"Requested logs are missing from {vehicle_dir}: {missing}")
    return sorted(dict.fromkeys(requested), key=natural_log_key)


def timestamp_stem_to_us(path: Path) -> int:
    try:
        timestamp = int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Ford image filename is not an integer timestamp: {path}") from exc

    digits = len(path.stem)
    if digits == 16:  # Ford release convention: Unix microseconds.
        return timestamp
    if digits == 19:  # Allow already-nanosecond timestamps.
        return timestamp // 1_000
    if digits == 13:  # Unix milliseconds.
        return timestamp * 1_000
    if digits == 10:  # Unix seconds.
        return timestamp * 1_000_000
    raise ValueError(f"Unsupported timestamp precision in Ford image filename: {path.name}")


def ford_log_paths(raw_root: Path, date: str, vehicle: str, log: str, camera: str) -> tuple[Path, Path]:
    log_dir = raw_root / date / vehicle / log
    prefix = f"{date}-{vehicle}-{log}"
    bag_path = log_dir / f"{prefix}.bag"
    camera_dir = log_dir / f"{prefix}-{camera}"
    if not bag_path.is_file():
        raise FileNotFoundError(f"Ford ROS bag does not exist: {bag_path}")
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Ford camera directory does not exist: {camera_dir}")
    return bag_path, camera_dir


def message_timestamp_ns(message: object, bag_timestamp: int) -> int:
    stamp = message.header.stamp
    timestamp = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return timestamp if timestamp else int(bag_timestamp)


def load_navigation_samples(
    bag_path: Path,
    start_ns: int,
    stop_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read ground-truth poses and GPS fixes over the camera time range."""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    pose_timestamps: list[int] = []
    poses: list[tuple[float, ...]] = []
    gps_timestamps: list[int] = []
    gps_fixes: list[tuple[float, float, float]] = []

    with Reader(bag_path) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in (POSE_TOPIC, GPS_TOPIC)
        ]
        topics = {connection.topic for connection in connections}
        missing_topics = {POSE_TOPIC, GPS_TOPIC} - topics
        if missing_topics:
            raise ValueError(f"{bag_path} is missing required topics: {sorted(missing_topics)}")

        messages = reader.messages(connections=connections, start=start_ns, stop=stop_ns)
        for connection, bag_timestamp, rawdata in tqdm(
            messages,
            desc=f"Read navigation {bag_path.stem}",
            unit="msg",
        ):
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            timestamp = message_timestamp_ns(message, bag_timestamp)
            if connection.topic == POSE_TOPIC:
                position = message.pose.position
                orientation = message.pose.orientation
                values = (
                    float(position.x),
                    float(position.y),
                    float(position.z),
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                )
                if all(math.isfinite(value) for value in values):
                    pose_timestamps.append(timestamp)
                    poses.append(values)
            else:
                values = (
                    float(message.latitude),
                    float(message.longitude),
                    float(message.altitude),
                )
                if (
                    all(math.isfinite(value) for value in values)
                    and -90 <= values[0] <= 90
                    and -180 <= values[1] <= 180
                ):
                    gps_timestamps.append(timestamp)
                    gps_fixes.append(values)

    if not pose_timestamps:
        raise ValueError(f"No valid {POSE_TOPIC} messages in the camera time range of {bag_path}")
    if not gps_timestamps:
        raise ValueError(f"No valid {GPS_TOPIC} messages in the camera time range of {bag_path}")

    pose_order = np.argsort(np.asarray(pose_timestamps, dtype=np.int64))
    gps_order = np.argsort(np.asarray(gps_timestamps, dtype=np.int64))
    return (
        np.asarray(pose_timestamps, dtype=np.int64)[pose_order],
        np.asarray(poses, dtype=np.float64)[pose_order],
        np.asarray(gps_timestamps, dtype=np.int64)[gps_order],
        np.asarray(gps_fixes, dtype=np.float64)[gps_order],
    )


def nearest_pose_indices(image_ns: np.ndarray, pose_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(pose_ns, image_ns, side="left")
    right = np.clip(right, 0, len(pose_ns) - 1)
    left = np.clip(right - 1, 0, len(pose_ns) - 1)
    choose_left = np.abs(image_ns - pose_ns[left]) <= np.abs(pose_ns[right] - image_ns)
    indices = np.where(choose_left, left, right)
    errors_ns = np.abs(image_ns - pose_ns[indices])
    return indices, errors_ns


def spatially_sample(records: Iterable[ImageRecord], distance: float) -> list[ImageRecord]:
    if distance == 0:
        return list(records)

    selected: list[ImageRecord] = []
    last_xy: tuple[float, float] | None = None
    for record in records:
        xy = (record.utm_easting, record.utm_northing)
        if last_xy is None or math.dist(xy, last_xy) >= distance:
            selected.append(record)
            last_xy = xy
    return selected


def records_from_log(
    raw_root: Path,
    split: str,
    date: str,
    vehicle: str,
    log: str,
    camera: str,
    sample_distance: float,
    max_sync_error_ms: float,
    max_images_per_log: int,
) -> list[ImageRecord]:
    bag_path, camera_dir = ford_log_paths(raw_root, date, vehicle, log, camera)
    image_paths = sorted(camera_dir.glob("*.png"), key=timestamp_stem_to_us)
    if max_images_per_log:
        image_paths = image_paths[:max_images_per_log]
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {camera_dir}")

    image_us = np.asarray([timestamp_stem_to_us(path) for path in image_paths], dtype=np.int64)
    image_ns = image_us * 1_000
    margin_ns = int(max_sync_error_ms * 1_000_000) + 1
    pose_ns, poses, gps_ns, gps_fixes = load_navigation_samples(
        bag_path,
        start_ns=int(image_ns[0] - margin_ns),
        stop_ns=int(image_ns[-1] + margin_ns),
    )
    pose_indices, pose_errors_ns = nearest_pose_indices(image_ns, pose_ns)
    gps_indices, gps_errors_ns = nearest_pose_indices(image_ns, gps_ns)

    aligned: list[ImageRecord] = []
    max_error_ns = max_sync_error_ms * 1_000_000
    for path, timestamp_us, pose_index, gps_index, pose_error_ns, gps_error_ns in zip(
        image_paths, image_us, pose_indices, gps_indices, pose_errors_ns, gps_errors_ns
    ):
        if pose_error_ns > max_error_ns or gps_error_ns > max_error_ns:
            continue
        x, y, z, qx, qy, qz, qw = poses[pose_index]
        latitude, longitude, altitude = gps_fixes[gps_index]
        easting, northing, zone_number, zone_letter = utm.from_latlon(latitude, longitude)
        aligned.append(
            ImageRecord(
                split=split,
                source_path=path.resolve(),
                image_timestamp_us=int(timestamp_us),
                pose_timestamp_ns=int(pose_ns[pose_index]),
                gps_timestamp_ns=int(gps_ns[gps_index]),
                pose_sync_error_ms=float(pose_error_ns / 1_000_000),
                gps_sync_error_ms=float(gps_error_ns / 1_000_000),
                utm_easting=float(easting),
                utm_northing=float(northing),
                utm_zone_number=int(zone_number),
                utm_zone_letter=str(zone_letter),
                latitude=float(latitude),
                longitude=float(longitude),
                altitude=float(altitude),
                map_x=float(x),
                map_y=float(y),
                map_z=float(z),
                qx=float(qx),
                qy=float(qy),
                qz=float(qz),
                qw=float(qw),
                date=date,
                vehicle=vehicle,
                log=log,
                camera=camera,
            )
        )

    if not aligned:
        raise ValueError(
            f"No images in {camera_dir} matched a pose within {max_sync_error_ms:g} ms"
        )
    sync_ratio = len(aligned) / len(image_paths)
    if sync_ratio < MIN_LOG_SYNC_RATIO:
        print(
            f"WARNING: skipping {split} {date}/{vehicle}/{log}/{camera}: only "
            f"{len(aligned)}/{len(image_paths)} images ({sync_ratio:.1%}) synchronized. "
            "The camera directory and same-named bag likely cover different time ranges."
        )
        return []
    selected = spatially_sample(aligned, sample_distance)
    print(
        f"{split:8s} {date}/{vehicle}/{log}/{camera}: "
        f"{len(image_paths)} images -> {len(aligned)} synchronized -> {len(selected)} sampled"
    )
    return selected


def build_split_records(
    raw_root: Path,
    split: str,
    date: str,
    vehicle: str,
    logs: Sequence[str],
    args: argparse.Namespace,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for log in logs:
        records.extend(
            records_from_log(
                raw_root=raw_root,
                split=split,
                date=date,
                vehicle=vehicle,
                log=log,
                camera=args.camera,
                sample_distance=args.sample_distance,
                max_sync_error_ms=args.max_sync_error_ms,
                max_images_per_log=args.max_images_per_log,
            )
        )
    deduplicated = deduplicate_image_timestamps(records, split)
    ensure_unique_output_names(deduplicated, split)
    return deduplicated


def deduplicate_image_timestamps(
    records: Sequence[ImageRecord], split: str
) -> list[ImageRecord]:
    """Drop repeated camera frames accidentally present in multiple Ford logs."""
    deduplicated: list[ImageRecord] = []
    seen: set[int] = set()
    for record in records:
        if record.image_timestamp_us not in seen:
            deduplicated.append(record)
            seen.add(record.image_timestamp_us)
    removed = len(records) - len(deduplicated)
    if removed:
        print(f"WARNING: removed {removed} duplicate image timestamps from {split}")
    return deduplicated


def ensure_unique_output_names(records: Sequence[ImageRecord], split: str) -> None:
    names = [record.output_name for record in records]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate generated filenames detected in {split} split")


def filter_query_overlap(
    database: Sequence[ImageRecord],
    queries: Sequence[ImageRecord],
    threshold: float,
) -> tuple[list[ImageRecord], np.ndarray]:
    if not database:
        raise ValueError("Database split is empty")
    if not queries:
        raise ValueError("Query split is empty")

    database_zones = {(record.utm_zone_number, record.utm_zone_letter) for record in database}
    query_zones = {(record.utm_zone_number, record.utm_zone_letter) for record in queries}
    if len(database_zones | query_zones) != 1:
        raise ValueError(
            f"All records must share one UTM zone, got database={database_zones}, queries={query_zones}"
        )
    database_xy = np.asarray([(record.utm_easting, record.utm_northing) for record in database])
    query_xy = np.asarray([(record.utm_easting, record.utm_northing) for record in queries])
    distances, _ = cKDTree(database_xy).query(query_xy, k=1)
    if threshold == 0:
        return list(queries), distances
    keep = distances <= threshold
    return [record for record, accepted in zip(queries, keep) if accepted], distances[keep]


def coverage_summary(nearest_distances: np.ndarray) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"query_count": int(len(nearest_distances))}
    if len(nearest_distances) == 0:
        return summary
    summary.update(
        {
            "nearest_database_distance_min_m": float(np.min(nearest_distances)),
            "nearest_database_distance_median_m": float(np.median(nearest_distances)),
            "nearest_database_distance_max_m": float(np.max(nearest_distances)),
            "queries_within_5m_percent": float(np.mean(nearest_distances <= 5) * 100),
            "queries_within_10m_percent": float(np.mean(nearest_distances <= 10) * 100),
            "queries_within_25m_percent": float(np.mean(nearest_distances <= 25) * 100),
        }
    )
    return summary


def transfer_file(source: Path, destination: Path, mode: str, resume: bool) -> None:
    if destination.exists() or destination.is_symlink():
        if not resume:
            raise FileExistsError(f"Output already exists: {destination}; use --resume to keep it")
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return
        raise FileExistsError(f"Existing output is incompatible with source: {destination}")

    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def materialize_split(
    records: Sequence[ImageRecord],
    destination: Path,
    mode: str,
    resume: bool,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for record in tqdm(records, desc=f"Write {destination.name}", unit="image"):
        transfer_file(record.source_path, destination / record.output_name, mode, resume)

    paths_file = destination.parent / f"{destination.name}_images_paths.txt"
    with paths_file.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(record.output_name + "\n")


def write_manifest(records: Sequence[ImageRecord], output_root: Path) -> None:
    manifest_path = output_root / "manifest.csv"
    fieldnames = [
        "split",
        "relative_output_path",
        "source_path",
        "image_timestamp_us",
        "pose_timestamp_ns",
        "gps_timestamp_ns",
        "pose_sync_error_ms",
        "gps_sync_error_ms",
        "utm_easting",
        "utm_northing",
        "utm_zone_number",
        "utm_zone_letter",
        "latitude",
        "longitude",
        "altitude",
        "map_x",
        "map_y",
        "map_z",
        "qx",
        "qy",
        "qz",
        "qw",
        "date",
        "vehicle",
        "log",
        "camera",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["source_path"] = str(record.source_path)
            row["relative_output_path"] = str(
                Path("images") / "test" / record.split / record.output_name
            )
            writer.writerow({field: row[field] for field in fieldnames})


def validate_roots(raw_root: Path, output_root: Path) -> tuple[Path, Path]:
    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Ford raw root does not exist: {raw_root}")
    if output_root == raw_root or output_root.is_relative_to(raw_root) or raw_root.is_relative_to(output_root):
        raise ValueError(
            "Raw and output roots must be disjoint; refusing to write in or above the Ford source tree"
        )
    return raw_root, output_root


def main() -> None:
    args = parse_args()
    raw_root, output_root = validate_roots(args.raw_root, args.output_root)
    database_logs = discover_logs(
        raw_root, args.database_date, args.database_vehicle, args.database_logs
    )
    query_logs = discover_logs(raw_root, args.query_date, args.query_vehicle, args.query_logs)

    print(f"Ford raw root (read-only): {raw_root}")
    print(f"Output root: {output_root}")
    print(
        f"Database: {args.database_date}/{args.database_vehicle}/{database_logs}; "
        f"queries: {args.query_date}/{args.query_vehicle}/{query_logs}; camera: {args.camera}"
    )

    database = build_split_records(
        raw_root,
        "database",
        args.database_date,
        args.database_vehicle,
        database_logs,
        args,
    )
    all_queries = build_split_records(
        raw_root,
        "queries",
        args.query_date,
        args.query_vehicle,
        query_logs,
        args,
    )
    queries, nearest_distances = filter_query_overlap(
        database, all_queries, args.query_overlap_threshold
    )
    if not queries:
        raise ValueError(
            "Query-overlap filtering removed every query. Select overlapping logs/dates "
            "or set --query-overlap-threshold 0 to inspect the unfiltered split."
        )
    ensure_unique_output_names(queries, "queries")
    removed_queries = len(all_queries) - len(queries)
    coverage = coverage_summary(nearest_distances)

    print(
        f"Selected {len(database)} database images and {len(queries)} queries "
        f"({removed_queries} non-overlapping queries removed)."
    )
    print("Retained-query coverage: " + json.dumps(coverage, indent=2, sort_keys=True))

    if args.dry_run:
        print("Dry run complete; no output files were created.")
        return

    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output root is not empty: {output_root}; choose another path or use --resume"
        )

    database_dir = output_root / "images" / "test" / "database"
    queries_dir = output_root / "images" / "test" / "queries"
    materialize_split(database, database_dir, args.transfer_mode, args.resume)
    materialize_split(queries, queries_dir, args.transfer_mode, args.resume)
    write_manifest([*database, *queries], output_root)

    metadata = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "database_date": args.database_date,
        "database_vehicle": args.database_vehicle,
        "database_logs": database_logs,
        "query_date": args.query_date,
        "query_vehicle": args.query_vehicle,
        "query_logs": query_logs,
        "camera": args.camera,
        "sample_distance_m": args.sample_distance,
        "max_sync_error_ms": args.max_sync_error_ms,
        "query_overlap_threshold_m": args.query_overlap_threshold,
        "transfer_mode": args.transfer_mode,
        "database_count": len(database),
        "query_count_before_overlap_filter": len(all_queries),
        "query_count": len(queries),
        "removed_query_count": removed_queries,
        "database_records_per_log": {
            log: sum(record.log == log for record in database) for log in database_logs
        },
        "query_records_per_log": {
            log: sum(record.log == log for record in queries) for log in query_logs
        },
        "coverage": coverage,
        "coordinate_note": "GPS fixes were converted to absolute UTM coordinates for cross-date evaluation.",
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    print(f"Ford VPR dataset written to {output_root}")


if __name__ == "__main__":
    main()

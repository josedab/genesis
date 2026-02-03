"""Synthetic Video Generation for Genesis.

This module provides synthetic video generation capabilities, particularly
focused on privacy-safe surveillance footage, anonymized faces, and
motion synthesis.

Example:
    >>> from genesis.video import VideoGenerator, VideoConfig
    >>>
    >>> # Create video generator
    >>> generator = VideoGenerator(config=VideoConfig(
    ...     resolution=(640, 480),
    ...     fps=30,
    ...     duration_seconds=10,
    ... ))
    >>>
    >>> # Generate synthetic surveillance footage
    >>> video = generator.generate_surveillance(
    ...     n_people=5,
    ...     scene_type="office",
    ... )
    >>>
    >>> # Save the video
    >>> video.save("synthetic_footage.mp4")
"""

from __future__ import annotations

import io
import json
import math
import random
import struct
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np

from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Optional imports
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SceneType(Enum):
    """Types of scenes for video generation."""

    OFFICE = "office"
    RETAIL = "retail"
    STREET = "street"
    PARKING = "parking"
    WAREHOUSE = "warehouse"
    LOBBY = "lobby"
    CORRIDOR = "corridor"
    CUSTOM = "custom"


class MotionType(Enum):
    """Types of motion patterns."""

    WALKING = "walking"
    RUNNING = "running"
    STANDING = "standing"
    SITTING = "sitting"
    RANDOM_WALK = "random_walk"
    PATH_FOLLOWING = "path_following"
    LOITERING = "loitering"


class ObjectType(Enum):
    """Types of objects that can appear in videos."""

    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    OBJECT = "object"
    PLACEHOLDER = "placeholder"


@dataclass
class BoundingBox:
    """A bounding box for object tracking."""

    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    object_id: Optional[str] = None
    object_type: ObjectType = ObjectType.PERSON

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "object_id": self.object_id,
            "object_type": self.object_type.value,
        }

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this box intersects with another."""
        return not (
            self.x + self.width < other.x
            or other.x + other.width < self.x
            or self.y + self.height < other.y
            or other.y + other.height < self.y
        )

    def center(self) -> Tuple[float, float]:
        """Get the center point of the box."""
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class VideoConfig:
    """Configuration for video generation."""

    # Resolution and format
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    duration_seconds: float = 10.0
    codec: str = "mp4v"

    # Scene settings
    scene_type: SceneType = SceneType.OFFICE
    background_color: Tuple[int, int, int] = (200, 200, 200)

    # Object settings
    n_objects: int = 5
    object_types: List[ObjectType] = field(default_factory=lambda: [ObjectType.PERSON])
    min_object_size: int = 30
    max_object_size: int = 100

    # Motion settings
    motion_type: MotionType = MotionType.RANDOM_WALK
    speed_range: Tuple[float, float] = (1.0, 5.0)

    # Privacy settings
    anonymize_faces: bool = True
    add_noise: bool = False
    noise_level: float = 0.05

    # Overlays
    add_timestamp: bool = True
    add_bounding_boxes: bool = False
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"

    # Output
    generate_metadata: bool = True
    generate_annotations: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution": self.resolution,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "scene_type": self.scene_type.value,
            "n_objects": self.n_objects,
            "motion_type": self.motion_type.value,
            "anonymize_faces": self.anonymize_faces,
        }


@dataclass
class ObjectState:
    """State of an object in a frame."""

    object_id: str
    object_type: ObjectType
    frame: int
    bbox: BoundingBox
    velocity: Tuple[float, float] = (0.0, 0.0)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type.value,
            "frame": self.frame,
            "bbox": self.bbox.to_dict(),
            "velocity": self.velocity,
            "attributes": self.attributes,
        }


@dataclass
class VideoFrame:
    """A single frame in the video."""

    frame_number: int
    timestamp: float
    image: np.ndarray
    objects: List[ObjectState] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "objects": [o.to_dict() for o in self.objects],
            "metadata": self.metadata,
        }


@dataclass
class VideoMetadata:
    """Metadata for a generated video."""

    video_id: str
    created_at: str
    config: VideoConfig
    n_frames: int
    duration_seconds: float
    object_tracks: Dict[str, List[Dict[str, Any]]]
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "created_at": self.created_at,
            "config": self.config.to_dict(),
            "n_frames": self.n_frames,
            "duration_seconds": self.duration_seconds,
            "n_objects": len(self.object_tracks),
            "statistics": self.statistics,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class MotionGenerator(ABC):
    """Abstract base class for motion generation."""

    @abstractmethod
    def generate_trajectory(
        self,
        n_frames: int,
        start_position: Tuple[float, float],
        bounds: Tuple[int, int, int, int],
    ) -> List[Tuple[float, float]]:
        """Generate a trajectory for an object.

        Args:
            n_frames: Number of frames
            start_position: Starting (x, y) position
            bounds: (x_min, y_min, x_max, y_max) movement bounds

        Returns:
            List of (x, y) positions for each frame
        """
        pass


class RandomWalkMotion(MotionGenerator):
    """Random walk motion generator."""

    def __init__(
        self,
        speed_range: Tuple[float, float] = (1.0, 5.0),
        direction_change_prob: float = 0.1,
    ):
        self.speed_range = speed_range
        self.direction_change_prob = direction_change_prob

    def generate_trajectory(
        self,
        n_frames: int,
        start_position: Tuple[float, float],
        bounds: Tuple[int, int, int, int],
    ) -> List[Tuple[float, float]]:
        trajectory = [start_position]
        x, y = start_position
        x_min, y_min, x_max, y_max = bounds

        # Initial random direction
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*self.speed_range)

        for _ in range(n_frames - 1):
            # Maybe change direction
            if random.random() < self.direction_change_prob:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(*self.speed_range)

            # Update position
            dx = speed * math.cos(angle)
            dy = speed * math.sin(angle)
            x += dx
            y += dy

            # Bounce off walls
            if x < x_min or x > x_max:
                angle = math.pi - angle
                x = max(x_min, min(x_max, x))
            if y < y_min or y > y_max:
                angle = -angle
                y = max(y_min, min(y_max, y))

            trajectory.append((x, y))

        return trajectory


class PathFollowingMotion(MotionGenerator):
    """Motion that follows predefined paths."""

    def __init__(
        self,
        speed_range: Tuple[float, float] = (2.0, 4.0),
        waypoints: Optional[List[Tuple[float, float]]] = None,
    ):
        self.speed_range = speed_range
        self.waypoints = waypoints

    def generate_trajectory(
        self,
        n_frames: int,
        start_position: Tuple[float, float],
        bounds: Tuple[int, int, int, int],
    ) -> List[Tuple[float, float]]:
        trajectory = []
        x_min, y_min, x_max, y_max = bounds

        # Generate random waypoints if not provided
        if not self.waypoints:
            n_waypoints = random.randint(3, 7)
            waypoints = [start_position]
            for _ in range(n_waypoints - 1):
                waypoints.append((
                    random.uniform(x_min, x_max),
                    random.uniform(y_min, y_max),
                ))
        else:
            waypoints = [start_position] + list(self.waypoints)

        # Interpolate between waypoints
        speed = random.uniform(*self.speed_range)
        current_pos = list(start_position)
        waypoint_idx = 1

        for _ in range(n_frames):
            trajectory.append(tuple(current_pos))

            if waypoint_idx >= len(waypoints):
                continue

            target = waypoints[waypoint_idx]
            dx = target[0] - current_pos[0]
            dy = target[1] - current_pos[1]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < speed:
                waypoint_idx += 1
                if waypoint_idx < len(waypoints):
                    current_pos = list(waypoints[waypoint_idx - 1])
            else:
                current_pos[0] += (dx / distance) * speed
                current_pos[1] += (dy / distance) * speed

        return trajectory


class LoiteringMotion(MotionGenerator):
    """Motion pattern for loitering behavior."""

    def __init__(
        self,
        loiter_radius: float = 50.0,
        speed_range: Tuple[float, float] = (0.5, 2.0),
    ):
        self.loiter_radius = loiter_radius
        self.speed_range = speed_range

    def generate_trajectory(
        self,
        n_frames: int,
        start_position: Tuple[float, float],
        bounds: Tuple[int, int, int, int],
    ) -> List[Tuple[float, float]]:
        trajectory = [start_position]
        center = start_position

        angle = random.uniform(0, 2 * math.pi)
        angular_speed = random.uniform(0.02, 0.05)
        radius_variation = random.uniform(0.8, 1.2)

        for i in range(n_frames - 1):
            # Circular motion around loiter center
            angle += angular_speed
            radius = self.loiter_radius * radius_variation * (0.5 + 0.5 * math.sin(i * 0.1))

            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)

            # Add some randomness
            x += random.gauss(0, 2)
            y += random.gauss(0, 2)

            trajectory.append((x, y))

        return trajectory


class ObjectRenderer:
    """Renders objects onto frames."""

    def __init__(self, anonymize: bool = True):
        self.anonymize = anonymize
        self._colors = self._generate_colors()

    def _generate_colors(self) -> Dict[ObjectType, Tuple[int, int, int]]:
        """Generate colors for different object types."""
        return {
            ObjectType.PERSON: (0, 255, 0),  # Green
            ObjectType.VEHICLE: (255, 0, 0),  # Blue
            ObjectType.ANIMAL: (255, 255, 0),  # Cyan
            ObjectType.OBJECT: (255, 0, 255),  # Magenta
            ObjectType.PLACEHOLDER: (128, 128, 128),  # Gray
        }

    def render_person(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        object_id: str,
    ) -> np.ndarray:
        """Render a synthetic person onto the frame."""
        x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)

        # Ensure within bounds
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        if w <= 0 or h <= 0:
            return frame

        # Generate a consistent color based on object_id
        color_seed = hash(object_id) % 256
        body_color = ((color_seed * 37) % 200 + 55, (color_seed * 73) % 200 + 55, (color_seed * 131) % 200 + 55)

        # Draw body (simplified stick figure / blob)
        head_size = max(w // 3, 5)
        head_y = y + head_size

        # Head (circle)
        head_center = (x + w // 2, head_y)
        if self.anonymize:
            # Pixelated/blurred head for anonymization
            if head_y - head_size >= 0 and head_y + head_size < frame_h:
                if head_center[0] - head_size >= 0 and head_center[0] + head_size < frame_w:
                    head_region = frame[
                        head_y - head_size:head_y + head_size,
                        head_center[0] - head_size:head_center[0] + head_size,
                    ]
                    if head_region.size > 0:
                        # Pixelate
                        small = head_region[::4, ::4]
                        if small.size > 0:
                            pixelated = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)
                            target_h, target_w = head_region.shape[:2]
                            pixelated = pixelated[:target_h, :target_w]
                            if pixelated.shape == head_region.shape:
                                frame[
                                    head_y - head_size:head_y + head_size,
                                    head_center[0] - head_size:head_center[0] + head_size,
                                ] = pixelated

        if CV2_AVAILABLE:
            # Draw using OpenCV
            cv2.ellipse(frame, head_center, (head_size, head_size), 0, 0, 360, body_color, -1)

            # Body (rectangle)
            body_top = head_y + head_size
            body_bottom = y + int(h * 0.7)
            cv2.rectangle(frame, (x + w // 4, body_top), (x + 3 * w // 4, body_bottom), body_color, -1)

            # Legs
            leg_top = body_bottom
            leg_bottom = y + h
            cv2.line(frame, (x + w // 3, leg_top), (x + w // 4, leg_bottom), body_color, max(w // 10, 2))
            cv2.line(frame, (x + 2 * w // 3, leg_top), (x + 3 * w // 4, leg_bottom), body_color, max(w // 10, 2))
        else:
            # Fallback: draw simple rectangle
            frame[max(0, y):min(frame_h, y + h), max(0, x):min(frame_w, x + w)] = body_color

        return frame

    def render_vehicle(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        object_id: str,
    ) -> np.ndarray:
        """Render a synthetic vehicle onto the frame."""
        x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)

        # Vehicle color
        color_seed = hash(object_id) % 256
        car_color = ((color_seed * 41) % 200 + 55, (color_seed * 67) % 200 + 55, (color_seed * 97) % 200 + 55)

        if CV2_AVAILABLE:
            # Car body
            cv2.rectangle(frame, (x, y + h // 3), (x + w, y + h), car_color, -1)

            # Roof
            roof_points = np.array([
                [x + w // 5, y + h // 3],
                [x + w // 4, y],
                [x + 3 * w // 4, y],
                [x + 4 * w // 5, y + h // 3],
            ], np.int32)
            cv2.fillPoly(frame, [roof_points], car_color)

            # Windows (darker)
            window_color = tuple(max(0, c - 50) for c in car_color)
            cv2.rectangle(frame, (x + w // 4, y + 5), (x + 3 * w // 4, y + h // 3 - 5), window_color, -1)

            # Wheels
            wheel_color = (30, 30, 30)
            wheel_size = max(h // 5, 3)
            cv2.circle(frame, (x + w // 4, y + h), wheel_size, wheel_color, -1)
            cv2.circle(frame, (x + 3 * w // 4, y + h), wheel_size, wheel_color, -1)
        else:
            frame[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)] = car_color

        return frame

    def render_placeholder(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        object_id: str,
        object_type: ObjectType,
    ) -> np.ndarray:
        """Render a placeholder object (colored rectangle)."""
        x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
        color = self._colors.get(object_type, (128, 128, 128))

        if CV2_AVAILABLE:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)
        else:
            frame[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)] = color

        return frame


class SceneGenerator:
    """Generates background scenes."""

    def __init__(self, config: VideoConfig):
        self.config = config

    def generate_background(self) -> np.ndarray:
        """Generate a background for the scene type."""
        width, height = self.config.resolution
        background = np.full((height, width, 3), self.config.background_color, dtype=np.uint8)

        if self.config.scene_type == SceneType.OFFICE:
            background = self._add_office_elements(background)
        elif self.config.scene_type == SceneType.RETAIL:
            background = self._add_retail_elements(background)
        elif self.config.scene_type == SceneType.STREET:
            background = self._add_street_elements(background)
        elif self.config.scene_type == SceneType.PARKING:
            background = self._add_parking_elements(background)
        elif self.config.scene_type == SceneType.CORRIDOR:
            background = self._add_corridor_elements(background)

        return background

    def _add_office_elements(self, background: np.ndarray) -> np.ndarray:
        """Add office-like elements to background."""
        height, width = background.shape[:2]

        # Floor (darker at bottom)
        floor_start = int(height * 0.7)
        background[floor_start:, :] = (150, 150, 150)

        # Add grid pattern for ceiling tiles
        tile_size = 60
        for y in range(0, floor_start, tile_size):
            if CV2_AVAILABLE:
                cv2.line(background, (0, y), (width, y), (180, 180, 180), 1)
            else:
                background[y, :] = (180, 180, 180)

        for x in range(0, width, tile_size):
            if CV2_AVAILABLE:
                cv2.line(background, (x, 0), (x, floor_start), (180, 180, 180), 1)
            else:
                background[:floor_start, x] = (180, 180, 180)

        # Add some desks (rectangles)
        desk_color = (139, 90, 43)  # Brown
        n_desks = random.randint(2, 4)
        for i in range(n_desks):
            desk_x = random.randint(50, width - 150)
            desk_y = random.randint(floor_start - 50, floor_start - 20)
            desk_w = random.randint(80, 120)
            desk_h = random.randint(30, 50)

            if CV2_AVAILABLE:
                cv2.rectangle(background, (desk_x, desk_y), (desk_x + desk_w, desk_y + desk_h), desk_color, -1)

        return background

    def _add_retail_elements(self, background: np.ndarray) -> np.ndarray:
        """Add retail store elements."""
        height, width = background.shape[:2]

        # Floor
        background[int(height * 0.75):, :] = (200, 180, 160)

        # Shelving units
        shelf_color = (100, 80, 60)
        n_shelves = random.randint(3, 6)
        shelf_width = width // (n_shelves + 2)

        for i in range(n_shelves):
            x = (i + 1) * shelf_width
            if CV2_AVAILABLE:
                cv2.rectangle(background, (x, int(height * 0.2)), (x + 30, int(height * 0.75)), shelf_color, -1)

        return background

    def _add_street_elements(self, background: np.ndarray) -> np.ndarray:
        """Add street scene elements."""
        height, width = background.shape[:2]

        # Sky
        background[:int(height * 0.3), :] = (135, 206, 235)  # Light blue

        # Buildings
        building_color = (120, 120, 120)
        n_buildings = random.randint(3, 5)
        building_width = width // n_buildings

        for i in range(n_buildings):
            bh = random.randint(int(height * 0.3), int(height * 0.6))
            x = i * building_width
            if CV2_AVAILABLE:
                cv2.rectangle(background, (x, int(height * 0.3)), (x + building_width - 10, bh), building_color, -1)

        # Road
        road_start = int(height * 0.7)
        background[road_start:, :] = (50, 50, 50)

        # Road markings
        if CV2_AVAILABLE:
            for x in range(0, width, 100):
                cv2.rectangle(background, (x, height - 30), (x + 50, height - 25), (255, 255, 255), -1)

        return background

    def _add_parking_elements(self, background: np.ndarray) -> np.ndarray:
        """Add parking lot elements."""
        height, width = background.shape[:2]

        # Asphalt
        background[:, :] = (80, 80, 80)

        # Parking lines
        line_color = (255, 255, 255)
        spot_width = 80
        n_spots = width // spot_width

        for i in range(n_spots + 1):
            x = i * spot_width
            if CV2_AVAILABLE:
                cv2.line(background, (x, 0), (x, height), line_color, 2)

        return background

    def _add_corridor_elements(self, background: np.ndarray) -> np.ndarray:
        """Add corridor elements."""
        height, width = background.shape[:2]

        # Walls with perspective
        wall_color = (220, 220, 210)
        background[:, :] = wall_color

        # Floor
        floor_color = (180, 180, 180)
        background[int(height * 0.7):, :] = floor_color

        # Ceiling lights
        if CV2_AVAILABLE:
            for x in range(width // 4, width, width // 2):
                cv2.circle(background, (x, 30), 20, (255, 255, 200), -1)

        return background


class SyntheticVideo:
    """Container for a synthetic video and its metadata."""

    def __init__(
        self,
        frames: List[VideoFrame],
        metadata: VideoMetadata,
        config: VideoConfig,
    ):
        self.frames = frames
        self.metadata = metadata
        self.config = config

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[VideoFrame]:
        return iter(self.frames)

    def __getitem__(self, idx: int) -> VideoFrame:
        return self.frames[idx]

    def save(self, path: Union[str, Path], include_metadata: bool = True) -> None:
        """Save the video to a file.

        Args:
            path: Output path
            include_metadata: Whether to save metadata alongside
        """
        path = Path(path)

        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, saving as raw numpy")
            np.savez(
                path.with_suffix(".npz"),
                frames=[f.image for f in self.frames],
            )
            if include_metadata:
                with open(path.with_suffix(".json"), "w") as f:
                    f.write(self.metadata.to_json())
            return

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            self.config.fps,
            self.config.resolution,
        )

        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)

        writer.release()
        logger.info(f"Saved video to {path}")

        # Save metadata
        if include_metadata:
            meta_path = path.with_suffix(".json")
            with open(meta_path, "w") as f:
                f.write(self.metadata.to_json())
            logger.info(f"Saved metadata to {meta_path}")

    def get_annotations(self) -> List[Dict[str, Any]]:
        """Get frame-by-frame annotations (for ML training)."""
        annotations = []

        for frame in self.frames:
            annotation = {
                "frame_number": frame.frame_number,
                "timestamp": frame.timestamp,
                "objects": [],
            }

            for obj in frame.objects:
                annotation["objects"].append({
                    "object_id": obj.object_id,
                    "object_type": obj.object_type.value,
                    "bbox": [obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height],
                })

            annotations.append(annotation)

        return annotations

    def export_annotations_coco(self) -> Dict[str, Any]:
        """Export annotations in COCO format."""
        images = []
        annotations = []
        annotation_id = 1

        category_map = {
            ObjectType.PERSON: 1,
            ObjectType.VEHICLE: 2,
            ObjectType.ANIMAL: 3,
        }

        for frame in self.frames:
            images.append({
                "id": frame.frame_number,
                "width": self.config.resolution[0],
                "height": self.config.resolution[1],
                "file_name": f"frame_{frame.frame_number:06d}.jpg",
            })

            for obj in frame.objects:
                annotations.append({
                    "id": annotation_id,
                    "image_id": frame.frame_number,
                    "category_id": category_map.get(obj.object_type, 1),
                    "bbox": [obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height],
                    "area": obj.bbox.width * obj.bbox.height,
                    "iscrowd": 0,
                })
                annotation_id += 1

        return {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "vehicle"},
                {"id": 3, "name": "animal"},
            ],
        }


class VideoGenerator:
    """Main class for synthetic video generation."""

    def __init__(self, config: Optional[VideoConfig] = None):
        """Initialize the video generator.

        Args:
            config: Video configuration
        """
        self.config = config or VideoConfig()

        self._scene_generator = SceneGenerator(self.config)
        self._object_renderer = ObjectRenderer(anonymize=self.config.anonymize_faces)

        self._motion_generators: Dict[MotionType, MotionGenerator] = {
            MotionType.RANDOM_WALK: RandomWalkMotion(speed_range=self.config.speed_range),
            MotionType.PATH_FOLLOWING: PathFollowingMotion(speed_range=self.config.speed_range),
            MotionType.LOITERING: LoiteringMotion(),
        }

    def generate(
        self,
        n_objects: Optional[int] = None,
        object_types: Optional[List[ObjectType]] = None,
        motion_type: Optional[MotionType] = None,
    ) -> SyntheticVideo:
        """Generate a synthetic video.

        Args:
            n_objects: Number of objects (overrides config)
            object_types: Object types to include (overrides config)
            motion_type: Motion pattern (overrides config)

        Returns:
            Generated SyntheticVideo
        """
        n_objects = n_objects or self.config.n_objects
        object_types = object_types or self.config.object_types
        motion_type = motion_type or self.config.motion_type

        # Calculate number of frames
        n_frames = int(self.config.fps * self.config.duration_seconds)
        width, height = self.config.resolution

        # Generate background
        background = self._scene_generator.generate_background()

        # Initialize objects
        objects = self._initialize_objects(n_objects, object_types, width, height)

        # Generate trajectories
        motion_gen = self._motion_generators.get(motion_type, self._motion_generators[MotionType.RANDOM_WALK])
        bounds = (50, 50, width - 100, height - 100)

        trajectories: Dict[str, List[Tuple[float, float]]] = {}
        for obj_id, obj in objects.items():
            start_pos = (obj["x"], obj["y"])
            trajectories[obj_id] = motion_gen.generate_trajectory(n_frames, start_pos, bounds)

        # Generate frames
        frames = []
        object_tracks: Dict[str, List[Dict[str, Any]]] = {obj_id: [] for obj_id in objects}
        start_time = datetime.now()

        for frame_num in range(n_frames):
            timestamp = frame_num / self.config.fps

            # Copy background
            frame_image = background.copy()

            # Update and render objects
            frame_objects = []

            for obj_id, obj in objects.items():
                pos = trajectories[obj_id][frame_num]

                # Create bounding box
                bbox = BoundingBox(
                    x=pos[0] - obj["width"] / 2,
                    y=pos[1] - obj["height"] / 2,
                    width=obj["width"],
                    height=obj["height"],
                    object_id=obj_id,
                    object_type=obj["type"],
                )

                # Calculate velocity
                if frame_num > 0:
                    prev_pos = trajectories[obj_id][frame_num - 1]
                    velocity = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
                else:
                    velocity = (0.0, 0.0)

                # Render object
                if obj["type"] == ObjectType.PERSON:
                    frame_image = self._object_renderer.render_person(frame_image, bbox, obj_id)
                elif obj["type"] == ObjectType.VEHICLE:
                    frame_image = self._object_renderer.render_vehicle(frame_image, bbox, obj_id)
                else:
                    frame_image = self._object_renderer.render_placeholder(frame_image, bbox, obj_id, obj["type"])

                # Create object state
                state = ObjectState(
                    object_id=obj_id,
                    object_type=obj["type"],
                    frame=frame_num,
                    bbox=bbox,
                    velocity=velocity,
                )
                frame_objects.append(state)

                # Track
                object_tracks[obj_id].append(state.to_dict())

            # Add bounding boxes overlay if requested
            if self.config.add_bounding_boxes and CV2_AVAILABLE:
                for state in frame_objects:
                    bbox = state.bbox
                    color = self._object_renderer._colors.get(state.object_type, (0, 255, 0))
                    cv2.rectangle(
                        frame_image,
                        (int(bbox.x), int(bbox.y)),
                        (int(bbox.x + bbox.width), int(bbox.y + bbox.height)),
                        color,
                        2,
                    )

            # Add timestamp overlay
            if self.config.add_timestamp:
                frame_time = start_time + timedelta(seconds=timestamp)
                timestamp_str = frame_time.strftime(self.config.timestamp_format)

                if CV2_AVAILABLE:
                    cv2.putText(
                        frame_image,
                        timestamp_str,
                        (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            # Add noise if requested
            if self.config.add_noise:
                noise = np.random.normal(0, self.config.noise_level * 255, frame_image.shape).astype(np.int16)
                frame_image = np.clip(frame_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Create frame object
            video_frame = VideoFrame(
                frame_number=frame_num,
                timestamp=timestamp,
                image=frame_image,
                objects=frame_objects,
            )
            frames.append(video_frame)

        # Create metadata
        metadata = VideoMetadata(
            video_id=f"vid_{uuid.uuid4().hex[:12]}",
            created_at=datetime.now().isoformat(),
            config=self.config,
            n_frames=n_frames,
            duration_seconds=self.config.duration_seconds,
            object_tracks=object_tracks,
            statistics={
                "total_objects": len(objects),
                "total_frames": n_frames,
                "avg_objects_per_frame": len(objects),
            },
        )

        return SyntheticVideo(frames=frames, metadata=metadata, config=self.config)

    def generate_surveillance(
        self,
        n_people: int = 5,
        scene_type: SceneType = SceneType.OFFICE,
        duration_seconds: float = 10.0,
        add_anomalies: bool = False,
    ) -> SyntheticVideo:
        """Generate synthetic surveillance footage.

        Args:
            n_people: Number of people in the scene
            scene_type: Type of scene
            duration_seconds: Video duration
            add_anomalies: Whether to add anomalous behaviors

        Returns:
            Generated surveillance video
        """
        # Update config
        self.config.scene_type = scene_type
        self.config.duration_seconds = duration_seconds
        self.config.n_objects = n_people

        # Mix of motion types for realism
        motion_types = [MotionType.RANDOM_WALK, MotionType.PATH_FOLLOWING]
        if add_anomalies:
            motion_types.append(MotionType.LOITERING)

        return self.generate(
            n_objects=n_people,
            object_types=[ObjectType.PERSON],
            motion_type=random.choice(motion_types),
        )

    def generate_traffic(
        self,
        n_vehicles: int = 10,
        duration_seconds: float = 30.0,
    ) -> SyntheticVideo:
        """Generate synthetic traffic footage.

        Args:
            n_vehicles: Number of vehicles
            duration_seconds: Video duration

        Returns:
            Generated traffic video
        """
        self.config.scene_type = SceneType.STREET
        self.config.duration_seconds = duration_seconds
        self.config.min_object_size = 50
        self.config.max_object_size = 150

        return self.generate(
            n_objects=n_vehicles,
            object_types=[ObjectType.VEHICLE],
            motion_type=MotionType.PATH_FOLLOWING,
        )

    def _initialize_objects(
        self,
        n_objects: int,
        object_types: List[ObjectType],
        width: int,
        height: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Initialize objects with random positions and sizes."""
        objects = {}

        for i in range(n_objects):
            obj_id = f"obj_{uuid.uuid4().hex[:8]}"
            obj_type = random.choice(object_types)

            # Random size
            size = random.randint(self.config.min_object_size, self.config.max_object_size)
            aspect_ratio = 0.5 if obj_type == ObjectType.PERSON else 1.5

            objects[obj_id] = {
                "type": obj_type,
                "x": random.randint(100, width - 100),
                "y": random.randint(100, height - 100),
                "width": int(size * aspect_ratio),
                "height": size,
            }

        return objects


def generate_synthetic_video(
    duration_seconds: float = 10.0,
    resolution: Tuple[int, int] = (640, 480),
    n_objects: int = 5,
    scene_type: str = "office",
    output_path: Optional[str] = None,
) -> SyntheticVideo:
    """Generate a synthetic video with default settings.

    Args:
        duration_seconds: Video duration
        resolution: Video resolution (width, height)
        n_objects: Number of objects
        scene_type: Scene type
        output_path: Optional path to save video

    Returns:
        Generated SyntheticVideo
    """
    config = VideoConfig(
        resolution=resolution,
        duration_seconds=duration_seconds,
        n_objects=n_objects,
        scene_type=SceneType(scene_type),
    )

    generator = VideoGenerator(config=config)
    video = generator.generate()

    if output_path:
        video.save(output_path)

    return video


__all__ = [
    # Main classes
    "VideoGenerator",
    "SyntheticVideo",
    "VideoConfig",
    # Data classes
    "VideoFrame",
    "VideoMetadata",
    "BoundingBox",
    "ObjectState",
    # Types
    "SceneType",
    "MotionType",
    "ObjectType",
    # Motion generators
    "MotionGenerator",
    "RandomWalkMotion",
    "PathFollowingMotion",
    "LoiteringMotion",
    # Scene generation
    "SceneGenerator",
    "ObjectRenderer",
    # Convenience function
    "generate_synthetic_video",
]

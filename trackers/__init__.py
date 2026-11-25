"""Multi-object tracking: gives each detected person a stable ID across frames."""
from trackers.person_tracker import PersonTracker, TrackerConfig, TrackerConfigError

__all__ = ["PersonTracker", "TrackerConfig", "TrackerConfigError"]

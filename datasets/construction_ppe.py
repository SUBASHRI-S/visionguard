"""Constants for the Ultralytics Construction-PPE dataset (November 2026 release).

Source: https://docs.ultralytics.com/datasets/detect/construction-ppe (CC BY 4.0).
The class order below must match the dataset's own ``construction-ppe.yaml``.
"""

SOURCE_CLASS_NAMES: list[str] = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
    "forklift",
    "no_vest",
]

# Each profile is an ordered list of (source class name, new class name).
# The position in the list becomes the new class id.
PROFILES: dict[str, list[tuple[str, str]]] = {
    # Violations are inferred later from person/helmet/vest boxes (decision D1).
    "core": [
        ("Person", "person"),
        ("helmet", "helmet"),
        ("vest", "vest"),
        ("forklift", "forklift"),
    ],
    # Same as core plus the two "missing equipment" classes, for the ablation
    # "detect missing PPE directly" vs "infer it from absence".
    "core_missing": [
        ("Person", "person"),
        ("helmet", "helmet"),
        ("vest", "vest"),
        ("forklift", "forklift"),
        ("no_helmet", "no_helmet"),
        ("no_vest", "no_vest"),
    ],
    # Every class, unchanged ids.
    "all": [(name, name) for name in SOURCE_CLASS_NAMES],
}


def profile_mapping(
    profile: str, source_names: list[str] | None = None
) -> tuple[dict[int, int], list[str]]:
    """Return (source_id -> new_id, new class names) for a profile."""
    source_names = source_names or SOURCE_CLASS_NAMES
    if profile not in PROFILES:
        raise ValueError(f"unknown profile '{profile}', choose from {sorted(PROFILES)}")
    id_map: dict[int, int] = {}
    new_names: list[str] = []
    for new_id, (source_name, new_name) in enumerate(PROFILES[profile]):
        if source_name not in source_names:
            raise ValueError(f"profile '{profile}' needs unknown class '{source_name}'")
        id_map[source_names.index(source_name)] = new_id
        new_names.append(new_name)
    return id_map, new_names

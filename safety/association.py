"""Decide, for each detected person, whether they wear a helmet and a vest.

The model only detects things that are present. A missing helmet is inferred: we look for a
helmet at the person's head. Three answers are possible, and only 'no' can raise a violation:

    yes      a helmet / vest was matched to this person
    no       nothing matched AND we could see the relevant body part clearly
    unknown  nothing matched but we cannot be sure (tiny, cut off, hidden, low confidence)

Positive evidence always wins: a matched helmet is 'yes' even for a small person.
"""
from safety.config import PPEConfig
from safety.contracts import FORKLIFT, HELMET, PERSON, VEST, Box, Detection, PPEStatus


def _head_region(box: Box, config: PPEConfig) -> Box:
    return Box(box.x1, box.y1, box.x2, box.y1 + config.head_region_fraction * box.height)


def _torso_region(box: Box, config: PPEConfig) -> Box:
    return Box(
        box.x1,
        box.y1 + config.torso_top_fraction * box.height,
        box.x2,
        box.y1 + config.torso_bottom_fraction * box.height,
    )


def _helmet_score(person: Box, helmet: Detection, config: PPEConfig) -> float | None:
    """Score a helmet for a person, or None if it cannot belong to them."""
    head = _head_region(person, config)
    cx, cy = helmet.box.center
    if not head.contains_point(cx, cy):
        return None
    if person.width <= 0:
        return None
    ratio = helmet.box.width / person.width
    if not config.helmet_min_width_ratio <= ratio <= config.helmet_max_width_ratio:
        return None
    # Prefer helmets nearer the middle of the head region, weighted by detector confidence.
    head_cx = (head.x1 + head.x2) / 2
    offset = abs(cx - head_cx) / max(person.width / 2, 1e-9)
    return helmet.confidence * (1.0 - min(offset, 1.0) * 0.5)


def _vest_score(person: Box, vest: Detection, config: PPEConfig) -> float | None:
    torso = _torso_region(person, config)
    if vest.box.area <= 0:
        return None
    overlap = torso.intersection_area(vest.box) / vest.box.area
    if overlap < config.vest_overlap_threshold:
        return None
    return vest.confidence * overlap


def _assign(scores: dict[tuple[int, int], float]) -> dict[int, int]:
    """Greedy one-to-one matching, best score first: each PPE item goes to one person only."""
    person_to_item: dict[int, int] = {}
    used_items: set[int] = set()
    for (person_i, item_i), _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        if person_i in person_to_item or item_i in used_items:
            continue
        person_to_item[person_i] = item_i
        used_items.add(item_i)
    return person_to_item


def _occlusion(index: int, boxes: list[Box]) -> float:
    """Largest share of this person's area covered by any other person."""
    own = boxes[index]
    if own.area <= 0:
        return 0.0
    return max(
        (own.intersection_area(other) / own.area for i, other in enumerate(boxes) if i != index),
        default=0.0,
    )


def _is_operator(person: Box, forklifts: list[Detection], config: PPEConfig) -> bool:
    if person.area <= 0:
        return False
    return any(
        person.intersection_area(f.box) / person.area >= config.operator_containment
        for f in forklifts
    )


def associate(
    detections: list[Detection],
    frame_width: int,
    frame_height: int,
    config: PPEConfig | None = None,
) -> list[PPEStatus]:
    """Return one PPEStatus per detected person, in the order the persons appear."""
    config = config or PPEConfig()
    persons = [d for d in detections if d.class_name == PERSON]
    helmets = [d for d in detections if d.class_name == HELMET]
    vests = [d for d in detections if d.class_name == VEST]
    forklifts = [d for d in detections if d.class_name == FORKLIFT]
    person_boxes = [p.box for p in persons]

    helmet_scores = {
        (pi, hi): s
        for pi, p in enumerate(persons)
        for hi, h in enumerate(helmets)
        if (s := _helmet_score(p.box, h, config)) is not None
    }
    vest_scores = {
        (pi, vi): s
        for pi, p in enumerate(persons)
        for vi, v in enumerate(vests)
        if (s := _vest_score(p.box, v, config)) is not None
    }
    helmet_of = _assign(helmet_scores)
    vest_of = _assign(vest_scores)

    statuses: list[PPEStatus] = []
    for index, person in enumerate(persons):
        box = person.box
        general: list[str] = []
        if person.confidence < config.min_person_confidence:
            general.append("person confidence too low")
        if box.height < config.min_person_height_px:
            general.append("person too small to judge")
        if _occlusion(index, person_boxes) > config.max_occlusion:
            general.append("person mostly hidden by another person")

        helmet_reasons = list(general)
        if box.y1 <= config.edge_margin_px:
            helmet_reasons.append("head may be cut off by the top of the frame")
        vest_reasons = list(general)
        if box.x1 <= config.edge_margin_px or box.x2 >= frame_width - config.edge_margin_px:
            vest_reasons.append("torso may be cut off by the side of the frame")

        if index in helmet_of:
            helmet, helmet_reasons = "yes", []
        else:
            helmet = "unknown" if helmet_reasons else "no"
        if index in vest_of:
            vest, vest_reasons = "yes", []
        else:
            vest = "unknown" if vest_reasons else "no"

        statuses.append(
            PPEStatus(
                person_index=index,
                person=person,
                helmet=helmet,
                vest=vest,
                is_operator=_is_operator(box, forklifts, config),
                reasons=tuple(dict.fromkeys(helmet_reasons + vest_reasons)),
            )
        )
    return statuses

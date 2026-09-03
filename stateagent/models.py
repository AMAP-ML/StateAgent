"""Core data models for StateAgent.

All cross-module data structures use Pydantic BaseModel for validation,
serialization, and JSON schema generation.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    """Types of entities tracked in the memory bank."""
    PERSON = "person"
    OBJECT = "object"
    CONTAINER = "container"
    CLOTHING = "clothing"
    UNKNOWN = "unknown"


class VisibilityState(str, Enum):
    """Visibility status of an entity in the current scene."""
    VISIBLE = "visible"
    HIDDEN = "hidden"
    PARTIALLY_VISIBLE = "partially_visible"


class OpenState(str, Enum):
    """Open/closed status for containers."""
    OPEN = "open"
    CLOSED = "closed"
    NOT_APPLICABLE = "n/a"


class CheckResult(str, Enum):
    """Result of a verification check."""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    SKIP = "skip"


# ---------------------------------------------------------------------------
# Unified Memory Bank Models
# ---------------------------------------------------------------------------

class Relation(BaseModel):
    """A relation between two entities."""
    type: str  # "inside", "holding", "wearing", "on_top_of", "covered_by"
    subject: str  # entity_id
    object: str   # entity_id
    value: bool = True


class HistoryEvent(BaseModel):
    """A recorded event in the state history."""
    time: int
    description: str
    details: dict[str, str] = Field(default_factory=dict)


class EntityMemory(BaseModel):
    """Unified entity memory: identity + state + appearance.

    This is the core data structure. Each entity in the world has:
    - Identity: entity_id, type, attributes (color, material, etc.)
    - State: visibility, location, open_state, relations
    - Appearance: crop image, size/shape descriptions
    """
    entity_id: str
    type: EntityType = EntityType.UNKNOWN
    # State information
    attributes: dict[str, str] = Field(default_factory=dict)
    visibility: VisibilityState = VisibilityState.VISIBLE
    current_location: Optional[str] = None  # e.g. "inside:box_1", "holding:person_1"
    open_state: OpenState = OpenState.NOT_APPLICABLE
    relations: list[Relation] = Field(default_factory=list)
    # Appearance information
    appearance_image: Optional[str] = None  # crop image path
    size_description: Optional[str] = None  # e.g. "small marble, ~2cm diameter"
    shape_description: Optional[str] = None  # e.g. "perfect sphere, smooth surface"
    state_description: Optional[str] = None  # e.g. "contains green mixed paint", "broken into two pieces"
    # Metadata
    source_shot: Optional[int] = None  # which shot this entity first appeared


class MemoryBank(BaseModel):
    """Unified state memory bank — the single source of truth for world state.

    Contains all known entities with their current state and appearance.
    """
    time: int = 0
    entities: dict[str, EntityMemory] = Field(default_factory=dict)
    history: list[HistoryEvent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prediction Models
# ---------------------------------------------------------------------------

class PredictionResult(BaseModel):
    """Result of VLM state prediction."""
    predicted_memory: MemoryBank
    entities_becoming_visible: list[str] = Field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Verification Models
# ---------------------------------------------------------------------------

class VerificationResult(BaseModel):
    """Result of verifying generated content against expected state."""
    scs: CheckResult = CheckResult.SKIP
    identity: CheckResult = CheckResult.SKIP
    visibility: CheckResult = CheckResult.SKIP
    failure_reason: str = ""


# ---------------------------------------------------------------------------
# Frame Selection & Edit Verification Models
# ---------------------------------------------------------------------------

class FrameSelectionResult(BaseModel):
    """Result of VLM-based frame selection from video history."""
    selected_frame_path: str
    reasoning: str = ""
    needs_editing: bool = True
    satisfaction_score: int = 0


class EditVerifyResult(BaseModel):
    """Result of verifying an edited frame against predicted state and prompt."""
    passed: bool = False
    feedback: str = ""
    state_ok: bool = True
    scene_ok: bool = True


# ---------------------------------------------------------------------------
# Pipeline I/O Models
# ---------------------------------------------------------------------------

class ShotOutput(BaseModel):
    """Complete output for a single shot in the pipeline."""
    shot_id: int
    prompt: str
    memory_snapshot: MemoryBank
    image_prompt: str = ""
    reference_images: list[str] = Field(default_factory=list)
    generated_end_frame: Optional[str] = None
    generated_video: Optional[str] = None
    verification: VerificationResult = Field(default_factory=VerificationResult)


class PipelineInput(BaseModel):
    """Input specification for the pipeline."""
    sample_id: str
    prompts: list[str]
    previous_end_frame: Optional[str] = None

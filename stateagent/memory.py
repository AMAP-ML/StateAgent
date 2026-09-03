"""MemoryBank manager — the single source of truth for world state.

MemoryBankManager provides a clean interface to read and update the unified
MemoryBank, which stores all entities with their identity, state, and appearance.
"""

import logging
from typing import Optional

from stateagent.models import (
    EntityMemory,
    MemoryBank,
    VisibilityState,
)

logger = logging.getLogger(__name__)


class MemoryBankManager:
    """Manages the unified MemoryBank."""

    def __init__(self):
        self._memory = MemoryBank()

    # ------------------------------------------------------------------
    # Memory access
    # ------------------------------------------------------------------

    def get_memory(self) -> MemoryBank:
        """Return a deep copy of the current memory bank."""
        return self._memory.model_copy(deep=True)

    def set_memory(self, memory: MemoryBank) -> None:
        """Replace the current memory bank."""
        self._memory = memory

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[EntityMemory]:
        """Get an entity by ID (returns reference, not copy)."""
        return self._memory.entities.get(entity_id)

    def update_entity(self, entity: EntityMemory) -> None:
        """Add or update an entity in the memory bank."""
        self._memory.entities[entity.entity_id] = entity

    def has_entity(self, entity_id: str) -> bool:
        """Check if an entity exists."""
        return entity_id in self._memory.entities

    def get_all_entities(self) -> dict[str, EntityMemory]:
        """Get all entities."""
        return self._memory.entities

    def get_entities_by_visibility(self, visibility: VisibilityState) -> list[EntityMemory]:
        """Get all entities with a specific visibility state."""
        return [
            e for e in self._memory.entities.values()
            if e.visibility == visibility
        ]

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    def get_entities_becoming_visible(self, predicted: MemoryBank) -> list[str]:
        """Compare current memory with predicted memory.

        Returns entity IDs that are currently HIDDEN or PARTIALLY_VISIBLE
        but will become VISIBLE in the predicted state. These entities need
        their appearance_image as reference for ending frame generation.
        """
        becoming_visible = []
        for eid, predicted_entity in predicted.entities.items():
            current_entity = self._memory.entities.get(eid)
            if current_entity is None:
                if predicted_entity.visibility == VisibilityState.VISIBLE:
                    becoming_visible.append(eid)
            elif (current_entity.visibility != VisibilityState.VISIBLE
                  and predicted_entity.visibility == VisibilityState.VISIBLE):
                becoming_visible.append(eid)
        return becoming_visible

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the memory bank to empty state."""
        self._memory = MemoryBank()

"""Guards against per-ability config dicts drifting out of sync.

Each ability must appear in all three timing dicts; a contributor adding a new
ability and forgetting one dict would otherwise hit a KeyError only at runtime
when that ability is first used.
"""
from __future__ import annotations

import config


def test_ability_timing_dicts_share_one_key_set() -> None:
    charge = set(config.ABILITY_CHARGE_TIME)
    cooldown = set(config.ABILITY_COOLDOWN)
    active = set(config.ABILITY_ACTIVE_DURATION)

    assert charge == cooldown == active, (
        "Ability timing dicts disagree:\n"
        f"  only in CHARGE_TIME:      {sorted(charge - (cooldown & active))}\n"
        f"  only in COOLDOWN:         {sorted(cooldown - (charge & active))}\n"
        f"  only in ACTIVE_DURATION:  {sorted(active - (charge & cooldown))}"
    )

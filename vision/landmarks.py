"""MediaPipe Hands landmark index constants.

Single source of truth for the 21 hand landmark indices used throughout the
Conjure pipeline.  Import from here rather than re-defining locally:

    from vision.landmarks import WRIST, INDEX_TIP, FINGER_TIPS, ...

Index layout (MediaPipe canonical):
    0  = WRIST
    1  = THUMB_CMC,  2 = THUMB_MCP,  3 = THUMB_IP,   4 = THUMB_TIP
    5  = INDEX_MCP,  6 = INDEX_PIP,  7 = INDEX_DIP,   8 = INDEX_TIP
    9  = MIDDLE_MCP, 10 = MIDDLE_PIP, 11 = MIDDLE_DIP, 12 = MIDDLE_TIP
    13 = RING_MCP,   14 = RING_PIP,   15 = RING_DIP,   16 = RING_TIP
    17 = PINKY_MCP,  18 = PINKY_PIP,  19 = PINKY_DIP,  20 = PINKY_TIP
"""

from __future__ import annotations

# Wrist
WRIST = 0

# Thumb
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4

# Index finger
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8

# Middle finger
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12

# Ring finger
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16

# Pinky finger
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Convenience groups
FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_MCPS = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FINGER_PIPS = (THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)

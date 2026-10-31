"""Anatomy of the 22-joint DHG-2016 hand skeleton and dataset constants."""

NUM_JOINTS = 22
NUM_CLASSES = 14
TARGET_FRAMES = 30

JOINT_NAMES = [
    "Wrist", "Palm_center",
    "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_tip",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_tip",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_tip",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_tip",
]

# 25 anatomical segments (same graph as the original report)
SKELETON_EDGES = [
    (0, 1),
    (1, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9),
    (1, 10), (10, 11), (11, 12), (12, 13),
    (1, 14), (14, 15), (15, 16), (16, 17),
    (1, 18), (18, 19), (19, 20), (20, 21),
    (2, 6), (6, 10), (10, 14), (14, 18),
]

# Three functional induced subgraphs (same partition as the original report)
SUBGRAPH_NODES = {
    "doigts_longs": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    "doigts_courts": [2, 3, 4, 5, 18, 19, 20, 21],
    "paume": [0, 1, 2, 6, 10, 14, 18],
}

# Class index = gesture_id - 1
GESTURE_NAMES = [
    "Grab", "Tap", "Expand", "Pinch", "Rotation CW", "Rotation CCW",
    "Swipe Right", "Swipe Left", "Swipe Up", "Swipe Down",
    "Swipe X", "Swipe V", "Swipe +", "Shake",
]

def classify_direction(history, mode, thresh=15.0):
    if len(history) < 5:
        return None
    first_cx, first_cy = history[0]
    last_cx, last_cy = history[-1]
    dx = last_cx - first_cx
    dy = last_cy - first_cy
    
    detected_direction = None
    if mode == "vertical_down_is_in":
        if dy > thresh: detected_direction = "in"
        elif dy < -thresh: detected_direction = "out"
    elif mode == "vertical_up_is_in":
        if dy < -thresh: detected_direction = "in"
        elif dy > thresh: detected_direction = "out"
    elif mode == "horizontal_right_is_in":
        if dx > thresh: detected_direction = "in"
        elif dx < -thresh: detected_direction = "out"
    elif mode == "horizontal_left_is_in":
        if dx < -thresh: detected_direction = "in"
        elif dx > thresh: detected_direction = "out"
    return detected_direction

# Test Suite
assert classify_direction([(100, 100), (100, 110), (100, 120), (100, 130), (100, 140)], "vertical_down_is_in") == "in"
assert classify_direction([(100, 140), (100, 130), (100, 120), (100, 110), (100, 100)], "vertical_down_is_in") == "out"
assert classify_direction([(100, 100), (110, 100), (120, 100), (130, 100), (140, 100)], "horizontal_right_is_in") == "in"
assert classify_direction([(140, 100), (130, 100), (120, 100), (110, 100), (100, 100)], "horizontal_right_is_in") == "out"

print("[SUCCESS] All motion-vector classification test assertions passed!")

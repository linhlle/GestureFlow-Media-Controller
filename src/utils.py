import subprocess

def normalize_landmarks(landmark_list):
    base_x, base_y, base_z = landmark_list[0].x, landmark_list[0].y, landmark_list[0].z

    relative_coords = []
    for lm in landmark_list:
        relative_coords.append([lm.x - base_x, lm.y - base_y, lm.z - base_z])

    flattened = [coord for sublist in relative_coords for coord in sublist]

    max_val = max(list(map(abs, flattened)))
    if max_val == 0:
        return [0.0] * 63
    
    normalized = [val / max_val for val in flattened]

    return normalized


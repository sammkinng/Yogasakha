import numpy as np
def are_points_collinear(p1, p2, p3, tolerance=0.00334): #.002458
    # Vector differences (2D - ignoring z-coordinate)
    v1 = (p2.x - p1.x, p2.y - p1.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)

    # Cross product in 2D is reduced to checking area of the parallelogram
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    # Check if the cross product is close to zero
    return abs(cross_product) <=tolerance

def is_same_side_of_line(p1, p2, a, b):
    # Calculate cross products to determine which side of the line the points lie on
    cross1 = (b.x - a.x) * (p1.y - a.y) - (b.y - a.y) * (p1.x - a.x)
    cross2 = (b.x - a.x) * (p2.y - a.y) - (b.y - a.y) * (p2.x - a.x)

    # Points are on the same side if the cross products have the same sign
    return cross1 * cross2 >= 0


# Function to determine orientation
def determine_orientation(shoulder, hip):
    vertical_diff = abs(shoulder[1] - hip[1])
    horizontal_diff = abs(shoulder[0] - hip[0])
    
    if vertical_diff > horizontal_diff * 1.5:
        return 0  # Upright
    elif horizontal_diff > vertical_diff * 1.5:
        return 1  # Horizontal
    else:
        return 2  # Midway
    
    
# Function to calculate angles between keypoints
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)






'''
# Input parameters of the AWS DeepRacer reward function

{
    "all_wheels_on_track": Boolean,        # flag to indicate if the agent is on the track
    "x": float,                            # agent's x-coordinate in meters
    "y": float,                            # agent's y-coordinate in meters
    "closest_objects": [int, int],         # zero-based indices of the two closest objects to the agent's current position of (x, y).
    "closest_waypoints": [int, int],       # indices of the two nearest waypoints.
    "distance_from_center": float,         # distance in meters from the track center 
    "is_crashed": Boolean,                 # Boolean flag to indicate whether the agent has crashed.
    "is_left_of_center": Boolean,          # Flag to indicate if the agent is on the left side to the track center or not. 
    "is_offtrack": Boolean,                # Boolean flag to indicate whether the agent has gone off track.
    "is_reversed": Boolean,                # flag to indicate if the agent is driving clockwise (True) or counter clockwise (False).
    "heading": float,                      # agent's yaw in degrees
    "objects_distance": [float, ],         # list of the objects' distances in meters between 0 and track_length in relation to the starting line.
    "objects_heading": [float, ],          # list of the objects' headings in degrees between -180 and 180.
    "objects_left_of_center": [Boolean, ], # list of Boolean flags indicating whether elements' objects are left of the center (True) or not (False).
    "objects_location": [(float, float),], # list of object locations [(x,y), ...].
    "objects_speed": [float, ],            # list of the objects' speeds in meters per second.
    "progress": float,                     # percentage of track completed
    "speed": float,                        # agent's speed in meters per second (m/s)
    "steering_angle": float,               # agent's steering angle in degrees
    "steps": int,                          # number steps completed
    "track_length": float,                 # track length in meters.
    "track_width": float,                  # width of the track
    "waypoints": [(float, float), ]        # list of (x,y) as milestones along the track center

}
'''

import math

def reward_function(params):
    # Constants
    SPEED_WEIGHT = 100
    HEADING_WEIGHT = 100
    STEERING_WEIGHT = 100
    SPEED_THRESHOLD = 1.0
    DIRECTION_THRESHOLD = 10.0
    SLOW_DOWN_THRESHOLD = 0.5
    WINDOW_SIZE = 2
    TARGET_PERCENTAGE = 0.3

    # Extract parameters
    speed = params['speed']
    heading = params['heading']
    steering_angle = params['steering_angle']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    all_wheels_on_track = params['all_wheels_on_track']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    is_left_of_center = params['is_left_of_center']

    # Calculate speed reward
    max_speed_reward = 10 * 10
    min_speed_reward = 3.33 * 3.33
    abs_speed_reward = speed * speed
    speed_reward = (abs_speed_reward - min_speed_reward) / (max_speed_reward - min_speed_reward) * SPEED_WEIGHT

    # Base reward on track status and speed
    if not all_wheels_on_track:
        reward = 1e-3
    elif speed < SPEED_THRESHOLD:
        reward = 0.5
    else:
        reward = 1.0

    # Calculate track direction
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    track_direction = math.degrees(track_direction)

    # Calculate track direction changes over waypoints
    all_track_directions = [
        math.degrees(math.atan2(waypoints[i + 1][1] - waypoints[i][1], waypoints[i + 1][0] - waypoints[i][0]))
        for i in range(len(waypoints) - 1)
    ]
    # This is not correct. 
    avg_diff = sum(
        sum(abs(all_track_directions[i + j] - all_track_directions[i + j - 1]) for j in range(1, WINDOW_SIZE)) / (WINDOW_SIZE - 1)
        for i in range(len(all_track_directions) - WINDOW_SIZE)
    ) / (len(all_track_directions) - WINDOW_SIZE)

    # Adjust speed reward based on track curvature
    if avg_diff > SLOW_DOWN_THRESHOLD:
        speed_reward *= 0.8

    # Calculate direction difference
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    # Penalize for large direction difference
    if direction_diff > DIRECTION_THRESHOLD:
        reward *= 0.5

    # Calculate heading reward
    heading_reward = (1 - (direction_diff / 180.0)) * HEADING_WEIGHT

    # Calculate steering reward
    steering_reward = (1 - (abs(steering_angle - direction_diff) / 180.0)) * STEERING_WEIGHT

    # Check if the vehicle is left of center and target 30% track width
    if is_left_of_center:
        target_position = 0.5 * track_width - (0.5 - TARGET_PERCENTAGE) * track_width
    else:
        target_position = 0.5 * track_width + (0.5 - TARGET_PERCENTAGE) * track_width

    # Penalize if the vehicle is too far from the target position
    distance_reward = max(1e-3, 1 - (abs(distance_from_center - target_position) / (0.5 * track_width)))

    # Combine rewards
    total_reward = reward + speed_reward + heading_reward + steering_reward + distance_reward

    return float(total_reward)

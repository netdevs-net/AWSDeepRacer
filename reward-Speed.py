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
    # Reward weights
    speed_weight = 100
    heading_weight = 100
    steering_weight = 100
    
    # Initialize the reward based on current speed
    max_speed_reward = 10 * 10
    min_speed_reward = 5 * 5
    abs_speed_reward = params['speed'] * params['speed']
    speed_reward = (abs_speed_reward - min_speed_reward) / (max_speed_reward - min_speed_reward) * speed_weight
    
    # Adjust this to only slightly Penalize if the car goes off track
    # Instead, focus on speed and driving toward the furthest point in the window
    if not params['all_wheels_on_track']:
        return 1e-3
    
    # Calculate the direction of the center line based on the closest waypoints
    next_point = params['waypoints'][params['closest_waypoints'][1]]
    prev_point = params['waypoints'][params['closest_waypoints'][0]]
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    track_direction = math.degrees(track_direction)
    
    # Initialize variables
    all_track_directions = []
    window_size = 2
    slow_down_threshold = 0.5
    
    # Calculate track directions for the entire track
    all_track_directions = [
        math.degrees(math.atan2(params['waypoints'][i + 1][1] - params['waypoints'][i][1], params['waypoints'][i + 1][0] - params['waypoints'][i][0]))
        for i in range(len(params['waypoints']) - 1)
    ]
    
    # Calculate average absolute difference of directions within a window
    average_diff = sum(
        sum(abs(all_track_directions[i + j] - all_track_directions[i + j - 1]) for j in range(1, window_size)) / (window_size - 1)
        for i in range(len(all_track_directions) - window_size)
    ) / (len(all_track_directions) - window_size)
    
    # Adjust speed based on average difference
    adjusted_speed_reward = speed_reward
    if average_diff > slow_down_threshold:
        adjusted_speed_reward *= 0.8
    
    # Calculate the difference between the car's heading and the track direction
    direction_diff = abs(track_direction - params['heading'])
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    
    abs_heading_reward = 1 - (direction_diff / 180.0)
    heading_reward = abs_heading_reward * heading_weight
    
    # Reward if steering angle is aligned with direction difference
    abs_steering_reward = 1 - (abs(params['steering_angle'] - direction_diff) / 180.0)
    steering_reward = abs_steering_reward * steering_weight
    
    return adjusted_speed_reward + heading_reward + steering_reward

import math

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


def reward_function(params):

    # Reward weights
    speed_weight = 100
    heading_weight = 100
    steering_weight = 50
    # Initialize the reward based on current speed
    # This section calculates a reward based on the car's speed (params['speed']).
    # It defines maximum and minimum speed rewards based on squares.
    max_speed_reward = 10 * 10
    min_speed_reward = 5 * 5
    # calculate absolute speed 
    abs_speed_reward = params['speed'] * params['speed']
    speed_reward = (abs_speed_reward - min_speed_reward) / (max_speed_reward - min_speed_reward) * speed_weight
    
    # - - - - - 
    
    # Penalize if the car goes off track
    if not params['all_wheels_on_track']:
        return 1e-3
    
    # - - - - - 
    
    # This section calculates a reward based on how well the car is aligned with the center line of the track.
    # It retrieves the waypoints (reference points on the track) from params['waypoints'] and the car's closest waypoints using params['closest_waypoints'].
    # Calculate the direction of the center line based on the closest waypoints
    next_point = params['waypoints'][params['closest_waypoints'][1]]
    prev_point = params['waypoints'][params['closest_waypoints'][0]]

    # It calculates the direction of the center line between the two closest waypoints using math.atan2 and converts it to degrees.
    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])  
    # Convert to degrees
    track_direction = math.degrees(track_direction)


    # Initialize variables
    all_track_directions = []  # List to store track directions for all steps
    window_size = 3  # Number of waypoints to consider for average difference
    slow_down_threshold = 0.5  # Threshold for slowing down based on average difference

    # Loop through waypoints to calculate track directions for the entire track
    for i in range(len(params['waypoints']) - 1):
        next_point = params['waypoints'][i + 1]
        prev_point = params['waypoints'][i]
        direction = math.degrees(math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]))
        all_track_directions.append(direction)

    # Calculate average absolute difference of radians within a window
    # ---- There is an error here, and the code does not validate, please fix
    average_diff = 0
    for i in range(len(all_track_directions) - window_size):
        window_diff = sum(abs(all_track_directions[i + j] - all_track_directions[i + j - 1]) 
                         for j in range(1, window_size)) / (window_size - 1)
        average_diff += window_diff

    average_diff /= (len(all_track_directions) - window_size)  # Normalize by number of windows

    # Adjust speed based on average difference
    adjusted_speed_reward = speed_reward
    if average_diff > slow_down_threshold:
    # Slow down for significant difference
        adjusted_speed_reward *= 0.5  # Adjust based on your desired slowdown factor

    # It calculates the difference between the car's heading (params['heading']) and the track direction.
    direction_diff = abs(track_direction - params['heading'])
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    abs_heading_reward = 1 - (direction_diff / 180.0)
    heading_reward = abs_heading_reward * heading_weight

    # Reward if steering angle is aligned with direction difference
    # It penalizes larger differences between the car's direction and the track direction.
    abs_steering_reward = 1 - (abs(params['steering_angle'] - direction_diff) / 180.0)
    steering_reward = abs_steering_reward * steering_weight
    
    return adjusted_speed_reward + heading_reward + steering_reward

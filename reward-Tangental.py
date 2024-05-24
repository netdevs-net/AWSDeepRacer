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
    # Convert to degreex
    track_direction = math.degrees(track_direction)

    #  ----- ADD CODE HERE
    # Calculate the track direction for every step around the track. 
    # Store all these values in an array, and output the values to the screen. 

    # If the average absolute difference of the radians is less than a set value, then accelerate the car towards max acceleration. 
    # If there is a significant difference in radians, within 3 waypoints, then slow the car down to prepare for the turn 
    # Create an expression here for slowing down for average absolute difference, focused on max_speed_reward and highest min_speed_reward
    #  ----- END ADD CODE HERE
 
   

    # It calculates the difference between the car's heading (params['heading']) and the track direction.
    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = abs(track_direction - params['heading'])
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    abs_heading_reward = 1 - (direction_diff / 180.0)
    heading_reward = abs_heading_reward * heading_weight
    
    # - - - - -
    # Reward if steering angle is aligned with direction difference
    # It penalizes larger differences between the car's direction and the track direction.
    abs_steering_reward = 1 - (abs(params['steering_angle'] - direction_diff) / 180.0)
    steering_reward = abs_steering_reward * steering_weight
    # Higher alignment gets a higher reward.    
    # - - - - -
    
    return speed_reward + heading_reward + steering_reward


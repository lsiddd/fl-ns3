import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from itertools import combinations

# Step 1: Read the file
data = pd.read_csv('campus.csv', delim_whitespace=True, header=None, names=['node_id', 'time', 'x', 'y'])

# Step 2: Function to interpolate the positions for each second
def interpolate_positions(df):
    # Group by node_id
    nodes = df['node_id'].unique()
    
    interpolated_data = []

    for node in nodes:
        # Extract the node's data
        node_data = df[df['node_id'] == node].sort_values(by='time')
        
        # Create a time range from the min to the max second, filling every second
        time_range = np.arange(np.floor(node_data['time'].min()), np.ceil(node_data['time'].max()) + 1)

        # Interpolate x and y positions for each second in the time range
        x_interp = np.interp(time_range, node_data['time'], node_data['x'])
        y_interp = np.interp(time_range, node_data['time'], node_data['y'])

        # Create a new DataFrame with the interpolated data
        node_interpolated = pd.DataFrame({
            'node_id': node,
            'time': time_range,
            'x': x_interp,
            'y': y_interp
        })

        interpolated_data.append(node_interpolated)

    # Concatenate all the interpolated data
    return pd.concat(interpolated_data)

# Step 3: Interpolate positions
interpolated_data = interpolate_positions(data)

# Save the interpolated data to a new CSV file (optional)
# interpolated_data.to_csv('interpolated_positions.csv', index=False)

# Step 4: Set up the figure and axes for animation
fig, ax = plt.subplots(figsize=(10, 8))

# Set axis labels
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Node Movement over Time')
ax.grid(True)

# Step 5: Initialize the plot elements we will update
node_paths = {node_id: ax.plot([], [], 'o-', label=f'Node {node_id}')[0] for node_id in interpolated_data['node_id'].unique()}
link_lines = []
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Set limits based on the data
ax.set_xlim(interpolated_data['x'].min() - 10, interpolated_data['x'].max() + 10)
ax.set_ylim(interpolated_data['y'].min() - 10, interpolated_data['y'].max() + 10)

# Show legend
ax.legend()

# Step 6: Define distance threshold for links
distance_threshold = 20.0  # You can adjust this value

# Step 7: Animation function
def update(frame):
    # Clear old link lines
    for line in link_lines:
        line.remove()
    link_lines.clear()

    # Update the plot for each node
    current_time = frame
    time_text.set_text(f'Time: {current_time:.0f}s')
    
    positions = {}
    for node_id, line in node_paths.items():
        node_data = interpolated_data[(interpolated_data['node_id'] == node_id) & (interpolated_data['time'] == current_time)]
        line.set_data(node_data['x'], node_data['y'])
        if not node_data.empty:
            positions[node_id] = (node_data['x'].iloc[-1], node_data['y'].iloc[-1])  # Latest position

    # Check distances between all pairs of nodes and draw links if within threshold
    for (node1, node2) in combinations(positions.keys(), 2):
        x1, y1 = positions[node1]
        x2, y2 = positions[node2]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if distance <= distance_threshold:
            # Draw a line between the two nodes
            link_line, = ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1)  # 'k-' is a black solid line
            link_lines.append(link_line)

    return list(node_paths.values()) + link_lines + [time_text]

# Step 8: Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(interpolated_data['time'].min(), interpolated_data['time'].max() + 1),
                    interval=1, blit=True)

# Show the animation (Optional, you can remove this if not needed)
plt.show()

# Step 9: Initialize a dictionary to store connections and their start times
active_links = {}
link_durations = []
interactions_per_node = {node: set() for node in interpolated_data['node_id'].unique()}

# Step 10: Function to calculate the distances and log interactions
def log_connections(frame):
    current_time = frame
    positions = {}
    
    # Track positions of all nodes at the current time
    for node_id in interpolated_data['node_id'].unique():
        node_data = interpolated_data[(interpolated_data['node_id'] == node_id) & (interpolated_data['time'] == current_time)]
        if not node_data.empty:
            positions[node_id] = (node_data['x'].iloc[-1], node_data['y'].iloc[-1])  # Latest position

    # Check distances between all pairs of nodes
    for (node1, node2) in combinations(positions.keys(), 2):
        x1, y1 = positions[node1]
        x2, y2 = positions[node2]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # If nodes are within the threshold, track the link
        if distance <= distance_threshold:
            # If a link between these nodes does not exist, start tracking it
            if (node1, node2) not in active_links:
                active_links[(node1, node2)] = current_time
            # Add these nodes as having interacted with each other
            interactions_per_node[node1].add(node2)
            interactions_per_node[node2].add(node1)
        else:
            # If the nodes were previously linked but are no longer within threshold, log the duration
            if (node1, node2) in active_links:
                start_time = active_links.pop((node1, node2))
                link_durations.append(current_time - start_time)

# Step 11: Loop through all time frames to log connections
for time_frame in np.arange(interpolated_data['time'].min(), interpolated_data['time'].max() + 1):
    log_connections(time_frame)

# Step 12: Calculate average link duration
if link_durations:
    average_link_duration = np.mean(link_durations)
else:
    average_link_duration = 0  # No connections

# Step 13: Calculate average number of nodes interacted with per node
average_interactions_per_node = np.mean([len(interactions) for interactions in interactions_per_node.values()])

# Step 14: Display the results
print(f"Average Connectedness Link Duration: {average_link_duration:.2f} seconds")
print(f"Average Number of Nodes Each Node Interacts With: {average_interactions_per_node:.2f}")

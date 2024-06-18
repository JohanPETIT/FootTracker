import os
from collections import defaultdict

# Base directory path
base_dir = "/storage8To/student_projects/foottracker/detectionData/outputjerem"

# List of target directories
target_dirs = [
    "1606b0e6_0", "35bd9041_1", "407c5a9e_1", "cfbe2e94_0",
    "1606b0e6_1", "3c993bd2_0", "4ffd5986_0", "cfbe2e94_1",
    "35bd9041_0", "3c993bd2_1", "9a97dae4_1", "ecf251d4_0"
]

# Event labels
event_labels = ["noevent", "challenge", "play", "throwin"]

# Dictionary to store counts
total_counts = defaultdict(int)
dir_counts = {}

# Function to count events in a directory
def count_events_in_dir(dir_path):
    counts = defaultdict(int)
    for entry in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, entry)):
            event = entry.split('_')[-1]
            if event in event_labels:
                counts[event] += 1
                total_counts[event] += 1
    return counts

# Traverse each target directory and count events
for dir_name in target_dirs:
    dir_path = os.path.join(base_dir, dir_name)
    if os.path.exists(dir_path):
        dir_counts[dir_name] = count_events_in_dir(dir_path)

# Print results
for dir_name, counts in dir_counts.items():
    print(f"In {dir_name}:")
    for event in event_labels:
        print(f"{event}: {counts[event]}")
    print()

# Print total counts
print("Total:")
for event in event_labels:
    print(f"{event}: {total_counts[event]}")

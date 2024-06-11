import os
import re
from collections import defaultdict

def count_batch_events(main_directory):
    # Regex pattern to match batchxxxxx_event files
    pattern = re.compile(r'batch\d+_(throwin|play|challenge|noevent)')
    
    # Dictionary to store counts of each event type
    event_counts = defaultdict(int)
    
    # Walk through the main directory and its subdirectories
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            match = pattern.match(file)
            if match:
                event = match.group(1)
                event_counts[event] += 1
    
    return event_counts

# Path to the main directory
main_directory = '/storage8To/student_projects/foottracker/detectionData/outputjerem'
event_counts = count_batch_events(main_directory)

# Print the counts
for event, count in event_counts.items():
    print(f"{event}: {count}")

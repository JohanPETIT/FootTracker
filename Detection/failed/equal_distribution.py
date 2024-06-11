import os
import random
import shutil

# Path to the directory containing the files
input_dir = '/storage8To/student_projects/foottracker/detectionData/output'
# Output directory to store the balanced dataset
output_dir = '/storage8To/student_projects/foottracker/detectionData/balanced_output'
# List all files in the directory
all_files = os.listdir(input_dir)
# Initialize dictionaries to store files based on their labels
file_dict = {
    'play': [],
    'challenge': [],
    'touchin': [],
    'no_event': []#,
    #'start':[],
    #'end':[]
}

# Function to equilibrate folders
def equal_distribution(input_dir,output_dir,all_files,file_dict):
 # Create directory if is not created yet 
 if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Classify files based on their labels
 for file in all_files:
    if 'play' in file:
        file_dict['play'].append(file)
        print('\n 1')
    elif 'challenge' in file:
        file_dict['challenge'].append(file)
        print('\n 2')
    elif 'touchin' in file:
        file_dict['touchin'].append(file)
        print('\n 3')
    elif 'no_event' in file:
        file_dict['no_event'].append(file)
        print('\n 4')
   # elif 'start' in file:
      #  file_dict['start'].append(file)
      #  print('\n 5')
   # elif 'end' in file:
        #file_dict['end'].append(file)
      #  print('\n 6')
     # Determine the minimum number of files across all labels
    min_count = min(len(file_dict['play']), len(file_dict['challenge']), len(file_dict['touchin']), len(file_dict['no_event']))#,len(file_dict['start']),len(file_dict['end']))
    # Randomly select files to match the target count
    selected_files = {
    'play': random.sample(file_dict['play'], min_count),
    'challenge': random.sample(file_dict['challenge'], min_count),
    'touchin': random.sample(file_dict['touchin'], min_count),
    'no_event': random.sample(file_dict['no_event'], min_count),
    #'start': random.sample(file_dict['start'], min_count), 
    #'end': random.sample(file_dict['end'], min_count)
     }
    # Insert selected files to the output directory
    for label, files in selected_files.items():
       for file in files:
        src = os.path.join(input_dir, file)
        dst = os.path.join(output_dir, file)
        shutil.copyfile(src, dst)
    print(f"Balanced dataset created with {min_count} files for each label.")
    # Unmatch the next line if works with debugger
    # return all_files,selected_files,label,files,min_count,file_dict
    
test_variable = equal_distribution(input_dir,output_dir,all_files,file_dict)
print('ok')
from Detection.failed.video_dataset_valid import VideoDataset,transform,DataLoader


def main():
    # Initialize dataset and DataLoader for the specific video
    video_dataset = VideoDataset(
        video_directory='/storage8To/student_projects/foottracker/detectionData/train',
        csv_file='/storage8To/student_projects/foottracker/detectionData/train.csv',
        transform=transform,
        frame_count=30,
        specific_video='1606b0e6_0.mp4'  # Specify the video file
    )

    video_loader = DataLoader(video_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Loop to demonstrate data loading
    #for frames, video_id, event_times, event_names in video_loader:
        #print("Processing batch...")
        #if len(frames)== 0:
        #    print("Received an empty batch of frames.")
        #else:
        #    formatted_event_times = [f"[{float(t)}]" for t in event_times]
         #   print(f"Event times: {formatted_event_times}\nEvent names: {event_names}")
         #   break  # Stop after first batch for demonstration

    print("Total videos in dataset:", len(video_dataset))

if __name__ == "__main__":
    main()


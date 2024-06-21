import os


# Enregistre la vidéo une fois les modifications apportées (détections, annotations etc...)
def save_video(output_video_bytes, output_video_path):
    # Checks and deletes the output file
    # You cant have a existing file or it will through an error
    if os.path.isfile(output_video_path):
        os.remove(output_video_path)

    out_file = open(output_video_path, "wb") # open for [w]riting as [b]inary

    batch_size = 20
    for i in range(0, len(output_video_bytes), batch_size):
        out_file.write(output_video_bytes[i:i+batch_size])
    out_file.close()
    
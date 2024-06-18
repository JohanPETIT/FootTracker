from PIL import Image
import matplotlib.pyplot as plt

# Path to the uploaded .png file
image_path = '/storage8To/student_projects/foottracker/detectionData/outputjerem/1606b0e6_0/batch01849_challenge/frame000.png'

# Open an image file
def open_and_display_image(image_path):
    print(f"Opening image: {image_path}")
    with Image.open(image_path) as img:
        print("Image opened successfully")
        # Display image
        plt.imshow(img)
        plt.axis('off')  # Hide axes
        plt.show()

open_and_display_image(image_path)

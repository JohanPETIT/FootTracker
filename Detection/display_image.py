from PIL import Image
import os
from torchvision import transforms

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
  #  transforms.Resize((244, 244)),
   # transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
 #   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor()
])

def display_and_save_images(image_path):
    try:
        # Open original image file
        img = Image.open(image_path)
        
        # Display original image with title
        img.show(title="Original Image")
        
        # Save original image
        img.save('original.png')
        
        # Apply transformation
        img_transformed = transform(img)
        
        # Display transformed image with title
        img_transformed.show(title="Transformed Image")
        
        # Save transformed image
        img_transformed.save('transformed.png')

    except Exception as e:
        print(f"Error displaying or saving image: {e}")

if __name__ == "__main__":
    image_file = "/storage8To/student_projects/foottracker/detectionData/outputjerem/407c5a9e_1/batch03561_play/frame001.png"
    display_and_save_images(image_file)


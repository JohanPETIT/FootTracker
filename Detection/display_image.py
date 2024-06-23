from PIL import Image
import os
from torchvision import transforms

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Resize((244, 244)),
    # Uncomment any additional transforms if needed
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToPILImage()  # Convert tensor back to PIL image
])


def display_and_save_images(image_path):
    try:
        # Open original image file
        img = Image.open(image_path)
        
        # Display original image (if GUI environment is available)
        try:
            img.show(title="Original Image")
        except Exception as e:
            print(f"Could not display the original image: {e}")
        
        # Save original image
        img.save('ordinal2.png')
        
        # Apply transformation
        img_transformed_tensor = transform(img)
        
        # Convert transformed tensor back to PIL Image for display and saving
       # img_transformed_pil = transforms.ToPILImage()(img_transformed_tensor)
        
        # Display transformed image (if GUI environment is available)
        try:
            img_transformed_tensor.show(title="Transformed Image")
        except Exception as e:
            print(f"Could not display the transformed image: {e}")
        
        # Save transformed image
        img_transformed_tensor.save('transformed2_pil.png')

    except Exception as e:
        print(f"Error displaying or saving image: {e}")

if __name__ == "__main__":
    image_file = "/storage8To/student_projects/foottracker/detectionData/outputjerem/407c5a9e_1/batch03561_play/frame009.png"
    display_and_save_images(image_file)



import os
from PIL import Image

filepath = "boston_test/image"

# Loop through all provided arguments
for filename in os.listdir(filepath):
    if "." not in filename:
        continue
    ending = filename.split(".")[1]
    if ending not in ["jpg"]:
        continue

    try:
        # Attempt to open an image file
        image = Image.open(os.path.join(filepath, filename))
    except IOError:
        # Report error, and then skip to the next argument
        print("Problem opening", filepath, ":", IOError)
        continue

    # Perform operations on the image here
    image = image.crop((0, 0, 328, 328))

    # Split our origional filename into name and extension
    name, extension = os.path.splitext(filename)

    # Save the image as "(origional_name)_thumb.jpg
    print(name + '_cropped.jpg')
    image.save(os.path.join("boston_test/cropped_image", name + '_cropped.jpg'))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def display_image_with_bboxes(image: np.ndarray, bboxes: np.ndarray, image_name, path_folder:str='./docs'):
    # Get the image dimensions
    height, width, _ = image.shape
    # Create a figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(image)
    # Loop through each bounding box
    for bbox in bboxes:
        # Denormalize the bounding box coordinates
        x_min, y_min, x_max, y_max = bbox
        x_min *= width
        y_min *= height
        x_max *= width
        y_max *= height
        # Create a rectangle patch to represent the bounding box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='r', facecolor='none')
        # Add the rectangle to the plot
        ax.add_patch(rect)
    # Show the image with bounding boxes
    plt.savefig(path_folder+f'/{image_name}.png', format='png')
    plt.show()
    
def plot_losses(train_losses, val_losses, path_folder:str='./docs'):
    plt.figure(figsize=(15, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path_folder+'/losses.png', format='png')
    plt.show()
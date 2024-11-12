import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def display_image_with_bboxes(image: np.ndarray, bboxes: np.ndarray, image_name, path_folder:str='./docs'):
    """
    Display the image with bounding boxes
    Args:
        image (np.ndarray): The image to display.
        bboxes (np.ndarray): The bounding boxes to display.
        image_name (str): The name of the image.
        path_folder (str): The path to save the image.
    """
    height, width, _ = image.shape
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(image)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min *= width
        y_min *= height
        x_max *= width
        y_max *= height
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig(path_folder+f'/{image_name}.png', format='png')
    plt.show()
    
def plot_losses(train_losses, val_losses, path_folder:str='./docs'):
    """
    Plot the training and validation losses
    Args:
        train_losses (_type_): _description_
        val_losses (_type_): _description_
        path_folder (str, optional): _description_. Defaults to './docs'.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path_folder+'/losses.png', format='png')
    plt.show()
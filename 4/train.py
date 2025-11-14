import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader import create_meshgrid, normalize_color, read_img, sample_pixels
from positional_encoding import positional_encoding
from network import NeuralField2D

# compute psnr loss from mse (assumes img normalized to [0,1])
def psnr(mse):
    return 10 * np.log10(1.0/mse)

# train a 2D neural field to fit an image
def train_2D_neural_field(model, coords, colors, num_iterations=2000, batch_size=10000, learning_rate=1e-2, device='cpu', log_every=100, visualize_every=500, height=None, width=None, original_img=None):
    """
    Arugments:
    model: the neural field to fit an img
    coords: all pixel coords w shape (num_pixels, 2) normalized [0,1]
    colors: all pixel colors w shape (num_pixels, 3) normalized [0,1]
    num_iterations: number of times to iterate training loop
    batch_size: number of pixels to sample each iteration
    learning_rate: learning rate for Adam optimizer
    log_every: print the training stats every __ iterations
    visualize_every: display current reconstruction of image every __ iterations

    Returns:
    loss_log: list of loss values over training
    psnr_log: list of PSNR values over training
    """

    # move model and data to a device, start w cpu for now
    model = model.to(device)
    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    colors = torch.tensor(colors, dtype=torch.float32, device=device)

    num_pixels = coords.shape[0]

    # now we can set up loss fn and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # track the metrics and train!
    loss_log = [] 
    psnr_log = []

    print(f"Starting Training")
    print(f"==================================================================")
    print(f"Device: {device}")
    print(f"Number of pixels in image: {num_pixels}")
    print(f"Batch Size (num pixels inputted to network): {batch_size}")
    print(f"Iterations: {num_iterations}")
    print(f"Learning rate: {learning_rate}")
    print(f"==================================================================")

    if visualize_every and height and width and original_img is not None:
        plt.ion()
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        fig.suptitle('Training Progress')

    
    for iteration in tqdm(range(num_iterations), desc="Training"):
        # set up random pixels, make batches
        idxs = torch.randint(0, num_pixels, (batch_size,), device=device)
        batch_coords = coords[idxs]
        batch_colors = colors[idxs]

        # forward pass
        predicted_colors = model(batch_coords)

        # calculate loss
        loss = loss_fn(predicted_colors, batch_colors)

        # backwards pass
        optimizer.zero_grad() # clear gradients from prev iteration
        loss.backward() # calc gradients
        optimizer.step() # update weights

        # track the loss and psnr
        loss_value = loss.item()
        psnr_value = psnr(loss_value)
        loss_log.append(loss_value)
        psnr_log.append(psnr_value)

        # print progress
        if (iteration) % log_every == 0:
            print(f"Iteration {iteration} / {num_iterations} ; Loss: {loss_value} ; PSNR: {psnr_value}")

        if visualize_every and (iteration % visualize_every) == 0:
            print(f"Rendering visualization at iteration {iteration} ...")

            reconstructed_img = render_img(model, height, width, coords, device)

            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Training Progress - Iteration {iteration}', fontsize=16)
            
            # Plot 1: Original image
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image', fontsize=12)
            axes[0].axis('off')
            
            # Plot 2: Current reconstruction
            axes[1].imshow(np.clip(reconstructed_img, 0, 1))
            axes[1].set_title(f'Reconstruction\nPSNR: {psnr_value:.2f} dB', fontsize=12)
            axes[1].axis('off')
            
            # Plot 3: Loss curve
            axes[2].plot(loss_log, color='blue', alpha=0.7, linewidth=2)
            axes[2].set_xlabel('Iteration', fontsize=11)
            axes[2].set_ylabel('MSE Loss', fontsize=11)
            axes[2].set_title('Training Loss', fontsize=12)
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the figure
            filename = f'progress_iter_{iteration:04d}.png'
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"    Saved: {filename}")
            
            # Display in notebook
            plt.show()
            
            # Close to free memory
            plt.close(fig)
            
            print(f">>> Visualization complete.\n")

        


    
    print("=========================================================")
    print("Training Complete!!")
    print("Final Loss: {loss_value} ; Final PSNR: {psnr_value}")
    print("=========================================================")
    return loss_log, psnr_log


def render_img(model, height, width, coords, device='cpu'):
    # generate a full image by querying model at every pixel.... (dont sample pixels)
    # coords should be positionally encoded
    model.eval()
    if not isinstance(coords, torch.Tensor):
        coords=torch.tensor(coords, dtype=torch.float32, device=device)
    else:
        coords = coords.to(device)

    # need to render in batches bc memory issues
    batch_size = 10000
    num_pixels = coords.shape[0]
    predicted_colors = []

    with torch.no_grad(): # no need to compute gradients during inference
        for i in range(0, num_pixels, batch_size):
            batch = coords[i:i+batch_size]
            colors = model(batch)
            predicted_colors.append(colors.cpu().numpy())

    # must concatenate batches and reshape img
    predicted_colors = np.concatenate(predicted_colors, axis=0) # bc batched along axis=0
    image = predicted_colors.reshape(height, width, 3)\
    
    model.train() # reset to training mode

    return image

def visualize_training_run(loss_log, psnr_log, old_img, new_img):
    # old_img = original image , new_img = reconstructed image, both are numpy arrays
    fig = plt.figure(figsize=(10, 10))

    # plot the loss curve
    ax1 = plt.subplot(2,2,1)
    ax1.plot(loss_log)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # plot the psnr curve
    ax2 = plt.subplot(2,2,2)
    ax2.plot(psnr_log)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR')
    ax2.set_title('PSNR along Training')
    ax2.grid(True, alpha=0.3)
    
    # original img
    ax3 = plt.subplot(2,2,3)
    ax3.imshow(old_img)
    ax3.set_title("Original Image")
    ax3.axis('off')

    # reconstructed img
    ax4 = plt.subplot(2,2,4)
    ax4.imshow(np.clip(new_img, 0, 1))
    ax4.set_title(f"Reconstructed Image (Best PSNR = {psnr_log[-1]})")
    ax4.axis('off')


    plt.tight_layout()
    plt.show()



def visualize_images_during_training(model, old_img, coords, device='cpu'):
    h, w = old_img.shape[:2]
    new_img = render_img(model, h, w, device)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

    ax1.imshow(old_img)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(new_img)
    ax2.set_title('Current Reconstruction')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()



    
def run_train_script(device):
    # entire training pipeline
    # set hyperparams
    L = 5
    num_iterations = 2000
    batch_size = 10000
    learning_rate = 1e-2
    device=device
    log_every=100
    visualize_every=500

    # load and preprocess image
    img_path = "C:/Users/sammi/Sammiesmith.github.io/4/fox.jpg"
    image = read_img(img_path)
    h,w = image.shape[:2]
    coords = create_meshgrid(h,w)
    colors = normalize_color(image)

    # apply positional encoding
    encoded = positional_encoding(coords, L)

    # create model
    model = NeuralField2D(input_dim=encoded.shape[1])

    # train
    loss_log, psnr_log = train_2D_neural_field(model=model, coords=encoded, colors=colors, num_iterations=num_iterations, batch_size=batch_size, learning_rate=learning_rate,device=device, log_every=log_every, visualize_every=visualize_every, height=h, width=w, original_img=image)

    # render final result
    reconstructed_img = render_img(model, h, w, encoded, device)

    # visualize loss curves
    visualize_training_run(loss_log, psnr_log, image, reconstructed_img)

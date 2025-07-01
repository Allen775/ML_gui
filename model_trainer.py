import time
import os
import numpy as np

def train_model(data_type, linear_type, output_dir, num_epochs=20):
    """
    Simulates a model training loop.

    This is a placeholder function. You should replace the logic here
    with your actual PyTorch/TensorFlow training code. The key is to
    `yield` the progress after each epoch.

    Args:
        data_type (str): 'non-deformed' or 'deformed'.
        linear_type (str): The type of linear layer configuration.
        output_dir (str): Directory to save the final model.
        num_epochs (int): The number of epochs to train for.

    Yields:
        dict: A dictionary containing training progress for each epoch.
              {'epoch': int, 'total_epochs': int, 'loss': float}
    """
    print(f"--- Starting Training ---")
    print(f"Data Type: {data_type}")
    print(f"Linear Layer: {linear_type}")
    print(f"Output Directory: {output_dir}")
    print("-------------------------")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- REPLACE THIS SECTION ---
    # In a real scenario, you would load your datasets and build your model here
    # based on the data_type and linear_type arguments.
    time.sleep(2) # Simulate data loading and model setup

    # --- Training Loop Simulation ---
    initial_loss = np.random.uniform(0.8, 1.2)
    for epoch in range(1, num_epochs + 1):
        # Simulate one epoch of training (forward/backward pass)
        time.sleep(0.5)
        loss = initial_loss / np.log1p(epoch) + np.random.uniform(-0.05, 0.05)

        # Yield progress to the GUI
        yield {'epoch': epoch, 'total_epochs': num_epochs, 'loss': loss}

    # --- Save Model Simulation ---
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    print(f"\nTraining complete. Saving model to {final_model_path}")
    # In a real scenario, you'd use torch.save(model.state_dict(), final_model_path)
    with open(final_model_path, 'w') as f:
        f.write(f"Model trained with:\nData: {data_type}\nLinear: {linear_type}\n")
    print("--- Model Saved ---")
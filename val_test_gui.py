"""
Main application for data control and model training GUI.

To run this script, you need an environment that can render ipywidgets,
such as a Jupyter Notebook, JupyterLab, or Google Colab.

You can run it from a notebook cell using:
%run main_app.py

Or from an IPython terminal:
ipython main_app.py
"""

import os
import time
import pandas as pd
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# --- Import from external files ---
try:
    from prediction_model import predict_age_with_model
    model_available = True
except ImportError:
    print("Warning: 'prediction_model.py' not found. Prediction functionality will be disabled.")
    model_available = False

try:
    from model_trainer import train_model
    trainer_available = True
except ImportError:
    print("Warning: 'model_trainer.py' not found. Training functionality will be disabled.")
    trainer_available = False

# --- Global Configuration & State ---
DATA_DIR = 'data'
COLUMNS = ['subject', 'age', 'sex', 'b0']
state = {
    'INIT': pd.DataFrame(columns=COLUMNS),
    'train': pd.DataFrame(columns=COLUMNS),
    'val': pd.DataFrame(columns=COLUMNS),
    'test': pd.DataFrame(columns=COLUMNS)
}


def setup_sample_data():
    """Creates a 'data' directory and sample CSV files for testing."""
    print("Setting up sample data...")
    os.makedirs(DATA_DIR, exist_ok=True)
    data1 = {'subject': ['S01', 'S02'], 'age': [25, 34], 'sex': ['M', 'F'], 'b0': [1.15, 1.23]}
    pd.DataFrame(data1).to_csv(os.path.join(DATA_DIR, 'sample1.csv'), index=False)
    data2 = {'subject': ['S03', 'S04'], 'age': [45, 22], 'sex': ['M', 'F'], 'b0': [0.98, 1.41]}
    pd.DataFrame(data2).to_csv(os.path.join(DATA_DIR, 'sample2.csv'), index=False)
    print(f"Created '{DATA_DIR}' directory with sample CSV files.")


def _create_data_panel(data_key: str, title: str):
    """Creates a full set of data control widgets for a single dataset (e.g., 'val' or 'test')."""
    # --- Widget Definition ---
    output = widgets.Output(layout={'border': '1px solid black', 'padding': '5px'})
    create_btn = widgets.Button(description=f'New {title}', button_style='success', icon='plus-square')
    try:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    except FileNotFoundError:
        csv_files = []
    file_selector = widgets.Dropdown(options=csv_files, description='CSV File:', disabled=not csv_files)
    add_data_btn = widgets.Button(description='Add from CSV', button_style='info', icon='upload')
    reset_index_btn = widgets.Button(description='Reset Index', button_style='warning', icon='refresh')
    split_data_btn = widgets.Button(description='Split data', button_style='warning', icon='refresh')
    save_filename_input = widgets.Text(value='new_dataset.csv', description='Save As:')
    save_btn = widgets.Button(description='Save to CSV', button_style='primary', icon='save')
    subject_selector = widgets.SelectMultiple(description='Subjects:', disabled=True, rows=8)
    delete_btn = widgets.Button(description='Delete Selected', button_style='danger', icon='trash')

    # --- Helper & Event Handlers ---
    def update_subject_selector():
        df = state[data_key]
        if not df.empty and 'subject' in df.columns:
            subjects = sorted(df['subject'].unique().tolist())
            subject_selector.options = subjects
            subject_selector.rows = min(10, len(subjects))
            subject_selector.disabled = False
        else:
            subject_selector.options = []
            subject_selector.disabled = True

    def on_create_clicked(b):
        with output:
            clear_output(wait=True)
            state[data_key] = pd.DataFrame(columns=COLUMNS)
            print("Empty dataset created.")
            update_subject_selector()
            display(state[data_key])

    def on_add_data_clicked(b):
        csv_file = file_selector.value
        if not csv_file: return
        file_path = os.path.join(DATA_DIR, csv_file)
        with output:
            clear_output(wait=True)
            try:
                new_data = pd.read_csv(file_path)
                state[data_key] = pd.concat([state[data_key], new_data], ignore_index=True)
                print(f"Data from '{csv_file}' added successfully.")
                update_subject_selector()
                display(state[data_key])
            except Exception as e:
                print(f"Error reading file '{csv_file}': {e}")
                display(state[data_key])

    def on_delete_clicked(b):
        selected_subjects = subject_selector.value
        with output:
            clear_output(wait=True)
            if not selected_subjects:
                print("No subjects selected for deletion.")
            else:
                df = state[data_key]
                initial_rows = len(df)
                state[data_key] = df[~df['subject'].isin(selected_subjects)].reset_index(drop=True)
                final_rows = len(state[data_key])
                print(f"Deleted {initial_rows - final_rows} row(s) for subjects: {', '.join(selected_subjects)}")
            update_subject_selector()
            display(state[data_key])

    def on_reset_index_clicked(b):
        with output:
            clear_output(wait=True)
            if not state[data_key].empty:
                state[data_key] = state[data_key].reset_index(drop=True)
                print("Dataset index has been reset.")
            else:
                print("Dataset is empty. Nothing to reset.")
            display(state[data_key])
    
    def on_split_data_clicked(b):
        with output:
            clear_output(wait=True)
            if state[data_key].empty:
                print("Dataset is empty. Nothing to split.")
                return
            else: 
                state['train'] = state[data_key].sample(frac=0.7, random_state=42)
                state['val'] = state[data_key].drop(state['train'].index).sample(frac=0.5, random_state=42)
                state['test'] = state[data_key].drop(state['train'].index).drop(state['val'].index)
            print("Dataset split into train, val, and test sets.")
            display(state[data_key])

    def on_save_clicked(b):
        filename = save_filename_input.value.strip()
        with output:
            clear_output(wait=True)
            if state[data_key].empty or not filename:
                print("Dataset is empty or filename is missing.")
                display(state[data_key])
                return
            if not filename.lower().endswith('.csv'):
                filename += '.csv'
            save_path = os.path.join(DATA_DIR, filename)
            try:
                state[data_key].to_csv(save_path, index=False)
                print(f"Dataset saved successfully to '{save_path}'")
                file_selector.options = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
            except Exception as e:
                print(f"Error saving file: {e}")
            display(state[data_key])

    # --- Link Handlers to Buttons ---
    create_btn.on_click(on_create_clicked)
    add_data_btn.on_click(on_add_data_clicked)
    split_data_btn.on_click(on_split_data_clicked)
    delete_btn.on_click(on_delete_clicked)
    reset_index_btn.on_click(on_reset_index_clicked)
    save_btn.on_click(on_save_clicked)

    # --- GUI Layout ---
    layout = widgets.VBox([
        widgets.HBox([create_btn, add_data_btn, split_data_btn, reset_index_btn]),
        file_selector,
        widgets.HBox([save_filename_input, save_btn]),
        widgets.HBox([subject_selector, delete_btn]),
        output
    ], layout={'padding': '10px'})

    with output:
        print(f"{title} panel is ready.")
        update_subject_selector()
        display(state[data_key])

    return layout

def _create_data_panel_other(data_key: str, title: str):
    """Creates a full set of data control widgets for a single dataset (e.g., 'val' or 'test')."""
    # --- Widget Definition ---
    output = widgets.Output(layout={'border': '1px solid black', 'padding': '5px'})
    create_btn = widgets.Button(description=f'New {title}', button_style='success', icon='plus-square')
    try:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    except FileNotFoundError:
        csv_files = []
    file_selector = widgets.Dropdown(options=csv_files, description='CSV File:', disabled=not csv_files)
    add_data_btn = widgets.Button(description='Add from CSV', button_style='info', icon='upload')
    reset_index_btn = widgets.Button(description='Reset Index', button_style='warning', icon='refresh')
    save_filename_input = widgets.Text(value='new_dataset.csv', description='Save As:')
    save_btn = widgets.Button(description='Save to CSV', button_style='primary', icon='save')
    subject_selector = widgets.SelectMultiple(description='Subjects:', disabled=True, rows=8)
    delete_btn = widgets.Button(description='Delete Selected', button_style='danger', icon='trash')

    # --- Helper & Event Handlers ---
    def update_subject_selector():
        df = state[data_key]
        if not df.empty and 'subject' in df.columns:
            subjects = sorted(df['subject'].unique().tolist())
            subject_selector.options = subjects
            subject_selector.rows = min(10, len(subjects))
            subject_selector.disabled = False
        else:
            subject_selector.options = []
            subject_selector.disabled = True

    def on_create_clicked(b):
        with output:
            clear_output(wait=True)
            state[data_key] = pd.DataFrame(columns=COLUMNS)
            print("Empty dataset created.")
            update_subject_selector()
            display(state[data_key])

    def on_add_data_clicked(b):
        csv_file = file_selector.value
        if not csv_file: return
        file_path = os.path.join(DATA_DIR, csv_file)
        with output:
            clear_output(wait=True)
            try:
                new_data = pd.read_csv(file_path)
                state[data_key] = pd.concat([state[data_key], new_data], ignore_index=True)
                print(f"Data from '{csv_file}' added successfully.")
                update_subject_selector()
                display(state[data_key])
            except Exception as e:
                print(f"Error reading file '{csv_file}': {e}")
                display(state[data_key])

    def on_delete_clicked(b):
        selected_subjects = subject_selector.value
        with output:
            clear_output(wait=True)
            if not selected_subjects:
                print("No subjects selected for deletion.")
            else:
                df = state[data_key]
                initial_rows = len(df)
                state[data_key] = df[~df['subject'].isin(selected_subjects)].reset_index(drop=True)
                final_rows = len(state[data_key])
                print(f"Deleted {initial_rows - final_rows} row(s) for subjects: {', '.join(selected_subjects)}")
            update_subject_selector()
            display(state[data_key])

    def on_reset_index_clicked(b):
        with output:
            clear_output(wait=True)
            if not state[data_key].empty:
                state[data_key] = state[data_key].reset_index(drop=True)
                print("Dataset index has been reset.")
            else:
                print("Dataset is empty. Nothing to reset.")
            display(state[data_key])

    def on_save_clicked(b):
        filename = save_filename_input.value.strip()
        with output:
            clear_output(wait=True)
            if state[data_key].empty or not filename:
                print("Dataset is empty or filename is missing.")
                display(state[data_key])
                return
            if not filename.lower().endswith('.csv'):
                filename += '.csv'
            save_path = os.path.join(DATA_DIR, filename)
            try:
                state[data_key].to_csv(save_path, index=False)
                print(f"Dataset saved successfully to '{save_path}'")
                file_selector.options = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
            except Exception as e:
                print(f"Error saving file: {e}")
            display(state[data_key])

    # --- Link Handlers to Buttons ---
    create_btn.on_click(on_create_clicked)
    add_data_btn.on_click(on_add_data_clicked)
    delete_btn.on_click(on_delete_clicked)
    reset_index_btn.on_click(on_reset_index_clicked)
    save_btn.on_click(on_save_clicked)

    # --- GUI Layout ---
    layout = widgets.VBox([
        reset_index_btn,
        file_selector,
        widgets.HBox([save_filename_input, save_btn]),
        widgets.HBox([subject_selector, delete_btn]),
        output
    ], layout={'padding': '10px'})

    with output:
        print(f"{title} panel is ready.")
        update_subject_selector()
        display(state[data_key])

    return layout

def create_data_control_gui():
    """Creates the data control tab with separate panels for val and test data."""
    init_panel= _create_data_panel('INIT', 'INIT Dataset')
    train_panel= _create_data_panel_other('train', 'Training Dataset')
    val_panel = _create_data_panel_other('val', 'Validation Dataset')
    test_panel = _create_data_panel_other('test', 'Test Dataset')
    
    accordion = widgets.Accordion(children=[init_panel,train_panel,val_panel, test_panel])
    accordion.set_title(0, 'INIT Dataset')
    accordion.set_title(1, 'Training Dataset')
    accordion.set_title(2, 'Validation Dataset')
    accordion.set_title(3, 'Test Dataset')
    
    return accordion


def create_training_gui():
    """Creates all widgets and layouts for the model training tab."""
    # --- Configuration Widgets ---
    data_type_selector = widgets.RadioButtons(
        options=['non-deformed', 'deformed'], description='Data Type:')
    linear_layer_options = {
        '1. T1 Only': 'T1_only', '2. T1 + Sex': 'T1_sex',
        '3. T1 + b0': 'T1_b0', '4. T1 + Sex + b0': 'T1_sex_b0'
    }
    linear_type_selector = widgets.RadioButtons(
        options=linear_layer_options, description='Linear Layer:')

    # --- Control & Display Widgets ---
    start_training_btn = widgets.Button(
        description='Start Training', button_style='success', icon='play',
        disabled=not trainer_available)
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100, description='Progress:',
        style={'bar_color': 'green'}, orientation='horizontal')
    training_output = widgets.Output(layout={'border': '1px solid black', 'padding': '5px'})

    # --- Event Handler ---
    def on_start_training_clicked(b):
        start_training_btn.disabled = True
        with training_output:
            training_output.clear_output(wait=True)
            data_type = data_type_selector.value
            linear_type_key = linear_type_selector.label
            linear_type_val = linear_type_selector.value
            output_dir = os.path.join('models', data_type, linear_type_val)
            
            try:
                all_losses = []
                for progress in train_model(data_type, linear_type_key, output_dir):
                    progress_bar.max = progress['total_epochs']
                    progress_bar.value = progress['epoch']
                    progress_bar.description = f"Epoch {progress['epoch']}/{progress['total_epochs']}"
                    loss = progress['loss']
                    all_losses.append(loss)
                    print(f"Epoch {progress['epoch']}: Loss = {loss:.4f}")
                
                # Plot training loss at the end
                print("\nPlotting training loss...")
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, len(all_losses) + 1), all_losses, marker='o', linestyle='-')
                plt.title(f'Training Loss for {data_type} / {linear_type_key}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.show()

            except Exception as e:
                print(f"An error occurred during training: {e}")
        start_training_btn.disabled = False

    # --- Link Handler to Button ---
    start_training_btn.on_click(on_start_training_clicked)

    # --- GUI Layout ---
    if not trainer_available:
        with training_output:
            print("Error: 'model_trainer.py' not found. Please create the file and restart the kernel.")

    config_box = widgets.HBox([data_type_selector, linear_type_selector])
    control_box = widgets.VBox([start_training_btn, progress_bar])
    return widgets.VBox([config_box, control_box, training_output])


def create_validation_gui():
    """Creates the GUI for the Validation & Testing tab."""
    # --- Widget Definition ---
    model_selector = widgets.Dropdown(description='Select Model:')
    dataset_selector = widgets.RadioButtons(options={'Validation': 'val', 'Test': 'test'}, description='Dataset:')
    run_btn = widgets.Button(description='Run Prediction', button_style='success', icon='rocket', disabled=True)
    output = widgets.Output(layout={'border': '1px solid black', 'padding': '5px'})
    
    def find_models():
        """Scans the ./models directory for trained model files."""
        model_paths = {}
        models_dir = 'models'
        if not os.path.isdir(models_dir):
            return model_paths
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith('final_model.pth'):
                    full_path = os.path.join(root, file)
                    # Create a user-friendly name from the path
                    name = os.path.relpath(full_path, models_dir).replace('\\', '/')
                    model_paths[name] = full_path
        return model_paths

    def refresh_models_list(b=None):
        """Updates the model selector dropdown."""
        with output:
            model_paths = find_models()
            if not model_paths:
                model_selector.options = []
                model_selector.disabled = True
                run_btn.disabled = True
                print("No trained models found in './models' directory.")
            else:
                model_selector.options = model_paths
                model_selector.disabled = False
                run_btn.disabled = False

    def on_run_prediction_clicked(b):
        model_path = model_selector.value
        data_key = dataset_selector.value
        df = state[data_key]

        with output:
            clear_output(wait=True)
            if not model_path:
                print("Error: Please select a model.")
                return
            if df.empty:
                print(f"Error: The {dataset_selector.label} dataset is empty. Please load data first.")
                return
            
            print(f"Running prediction on {dataset_selector.label} dataset...")
            try:
                # Call the external prediction function
                state[data_key]['predicted_age'] = predict_age_with_model(df.copy(), model_path)
                
                print("Prediction complete. Added 'predicted_age' column.")

                # Plot results
                plt.figure(figsize=(7, 7))
                plt.scatter(df['age'], df['predicted_age'], alpha=0.7, label='Predictions')
                lims = [np.min([df['age'].min(), df['predicted_age'].min()]), np.max([df['age'].max(), df['predicted_age'].max()])]
                plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Identity Line')
                plt.xlabel("True Age")
                plt.ylabel("Predicted Age")
                plt.title("Predicted Age vs. True Age")
                plt.legend()
                plt.grid(True)
                plt.show()

                display(state[data_key])
            except Exception as e:
                print(f"An error occurred during prediction: {e}")

    run_btn.on_click(on_run_prediction_clicked)
    model_selector.observe(refresh_models_list, names='options') # Refresh on change
    refresh_models_list() # Initial population
    return widgets.VBox([widgets.HBox([dataset_selector, model_selector]), run_btn, output])

def create_bias_cor_gui():
    """Creates the GUI for the Validation & Testing tab."""
    # --- Widget Definition ---
    
    run_btn = widgets.Button(description='Run Bias Correction ', button_style='success', icon='rocket', disabled=False)
    output = widgets.Output(layout={'border': '1px solid black', 'padding': '20px'})
    

    from sklearn.linear_model import LinearRegression
    # val_true_age=df_val['age'].values
    # val_predicted_age=df_val['predicted_age'].values
    # def check_data_ready(b=None,df=state['test']):
    #     if df.empty or 'predicted_age' not in df.columns:
    #                 print("Error: Please run val and test data prediction.")
    #                 return
    #     else: 
    #         run_btn.disabled = False

    def build_bias_correction_model(state_val,state_test):
        if len(state_val) != len(state_test):
            raise ValueError("The lengths of true_age and predicted_age must be the same.")
        
        PAD = (state_val['predicted_age'].values -state_val['age'].values)
        reg=LinearRegression().fit(PAD.reshape(-1, 1),state_val['age'].values)
        # print('val R squared:', reg.score(PAD.reshape(-1, 1), state_val['age'].values))
        # print('val Coefficients:', reg.coef_)
        # print('val Intercept:', reg.intercept_)
        
        corrected_val_PAD=PAD-reg.predict(state_val['age'].values.reshape(-1, 1))
        test_PAD = (state_test['predicted_age'].values - state_test['age'].values)
        corrected_test_PAD = test_PAD - reg.predict(state_test['age'].values.reshape(-1, 1))
        
        reg=LinearRegression().fit(state_val['age'].values.reshape(-1, 1)+corrected_val_PAD.reshape(-1, 1),state_val['age'].values)
        print("val  predicted age(corrected)/age R Squared:", reg.score(state_val['age'].values.reshape(-1, 1), state_val['age'].values))
        
        reg=LinearRegression().fit(state_test['age'].values.reshape(-1, 1)+corrected_test_PAD.reshape(-1, 1),state_test['age'].values)
        print("test  predicted age(corrected)/age R Squared:", reg.score(state_test['age'].values.reshape(-1, 1), state_test['age'].values))
        return corrected_val_PAD,corrected_test_PAD
        

            
    

    def on_run_bias_cor_clicked(b):
        df= state['test']
        with output:
            clear_output(wait=True)
            

            print(f"Running bias correction on valiadation dataset...")
            try:
                # Call the external prediction function
                corrected_val_PAD,corrected_test_PAD=build_bias_correction_model(state['val'],state['test'])
                state['val']['corrected PAD'] = corrected_val_PAD
                state['val']['predicted_age corrected'] = corrected_val_PAD + state['val']['age']
                print("Bias correction complete. Added 'corrected PAD' and 'predicted_age corrected' column to validation dataset.")
                print("valiadation corrected PAD:", sum(state['val']['corrected PAD'].values)/len(state['val']['corrected PAD'].values))
                df['corrected PAD'] = corrected_test_PAD
                df['predicted_age corrected'] = corrected_test_PAD + df['age']
                print("Bias correction complete. Added 'corrected PAD' and 'predicted_age corrected' column to test dataset.")
                print("valiadation corrected PAD:", sum(state['test']['corrected PAD'].values)/len(state['test']['corrected PAD'].values))
                # Plot results
                plt.figure(1,figsize=(7, 7))
                plt.scatter(state['val']['age'], state['val']['predicted_age corrected'], alpha=0.7, label='Predictions')
                lims = [np.min([state['val']['age'].min(), state['val']['predicted_age corrected'].min()]), np.max([state['val']['age'].max(), state['val']['predicted_age corrected'].max()])]
                plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Identity Line')
                plt.xlabel("True Age")
                plt.ylabel("Predicted Age")
                plt.title(" VALIDATION Predicted Age (corrected) vs. True Age")
                plt.legend()
                plt.grid(True)
                plt.show()
                
                plt.figure(2,figsize=(7, 7))
                plt.scatter(df['age'], df['predicted_age corrected'], alpha=0.7, label='Predictions')
                lims = [np.min([df['age'].min(), df['predicted_age corrected'].min()]), np.max([df['age'].max(), df['predicted_age corrected'].max()])]
                plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Identity Line')
                plt.xlabel("True Age")
                plt.ylabel("Predicted Age")
                plt.title("TESTING Predicted Age (corrected) vs. True Age")
                plt.legend()
                plt.grid(True)
                plt.show()

                print('val dataset after bias correction:')
                display(state['val'])
                print('test dataset after bias correction:')
                display(state['test'])
            except Exception as e:
                print(f"An error occurred during prediction: {e}")

    run_btn.on_click(on_run_bias_cor_clicked)
    # check_data_ready(b=None,df=state['test'])
    return widgets.VBox([ run_btn, output])

def create_transfer_learning_gui():
    """Creates all widgets and layouts for the model training tab."""
    # --- Configuration Widgets ---
    data_type_selector = widgets.RadioButtons(
        options=['non-deformed', 'deformed'], description='Data Type:')
    linear_layer_options = {
        '1. T1 Only': 'T1_only', '2. T1 + Sex': 'T1_sex',
        '3. T1 + b0': 'T1_b0', '4. T1 + Sex + b0': 'T1_sex_b0'
    }
    linear_type_selector = widgets.RadioButtons(
        options=linear_layer_options, description='Linear Layer:')

    # --- Control & Display Widgets ---
    start_training_btn = widgets.Button(
        description='Start transfer learningTraining', button_style='success', icon='play',
        disabled=not trainer_available)
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100, description='Progress:',
        style={'bar_color': 'green'}, orientation='horizontal')
    training_output = widgets.Output(layout={'border': '1px solid black', 'padding': '5px'})

    # --- Event Handler ---
    def on_start_training_clicked(b):
        start_training_btn.disabled = True
        with training_output:
            training_output.clear_output(wait=True)
            data_type = data_type_selector.value
            linear_type_key = linear_type_selector.label
            linear_type_val = linear_type_selector.value
            output_dir = os.path.join('models', data_type, linear_type_val)
            
            try:
                all_losses = []
                for progress in train_model(data_type, linear_type_key, output_dir):
                    progress_bar.max = progress['total_epochs']
                    progress_bar.value = progress['epoch']
                    progress_bar.description = f"Epoch {progress['epoch']}/{progress['total_epochs']}"
                    loss = progress['loss']
                    all_losses.append(loss)
                    print(f"Epoch {progress['epoch']}: Loss = {loss:.4f}")
                
                # Plot training loss at the end
                print("\nPlotting training loss...")
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, len(all_losses) + 1), all_losses, marker='o', linestyle='-')
                plt.title(f'Training Loss for {data_type} / {linear_type_key}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.show()

            except Exception as e:
                print(f"An error occurred during training: {e}")
        start_training_btn.disabled = False

    # --- Link Handler to Button ---
    start_training_btn.on_click(on_start_training_clicked)

    # --- GUI Layout ---
    if not trainer_available:
        with training_output:
            print("Error: 'model_trainer.py' not found. Please create the file and restart the kernel.")

    config_box = widgets.HBox([data_type_selector, linear_type_selector])
    control_box = widgets.VBox([start_training_btn, progress_bar])
    return widgets.VBox([config_box, control_box, training_output])

def main():
    """
    Main function to assemble and display the complete GUI.
    """
    # 1. Setup sample data if needed
    if not os.path.exists(DATA_DIR):
        setup_sample_data()

    # 2. Create the individual GUI components
    data_gui = create_data_control_gui()
    training_gui = create_training_gui()
    validation_gui = create_validation_gui()
    bias_cor_gui=create_bias_cor_gui()
    tranfer_gui=create_transfer_learning_gui()

    # 3. Combine them into a tabbed interface
    app = widgets.Tab()
    app.children = [data_gui, training_gui, validation_gui,bias_cor_gui,tranfer_gui]
    app.set_title(0, 'Data Control')
    app.set_title(1, 'Model Training')
    app.set_title(2, 'Validation & Testing')
    app.set_title(3, 'Bias Correction')
    app.set_title(4,'Transfer Learning')

    # 4. Display the final application
    print("Application is ready.")
    display(app)


if __name__ == "__main__":
    main()
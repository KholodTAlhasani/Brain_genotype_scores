
import os
import numpy as np
import pandas as pd
import medicalnet, dataset_mn, training_mn_v2, loss
import importlib
import random
from collections import Counter
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Reload modules to ensure updates are considered
importlib.reload(medicalnet)
importlib.reload(dataset_mn)
importlib.reload(training_mn_v2)
importlib.reload(loss)
# Additional imports
from medicalnet import multitask_resnet10, multitask_resnet18, multitask_resnet50
from dataset_mn import T1_dataset
from training_mn_v2 import trainer, tester, create_saliency
from loss import MaskedLoss
from datetime import datetime
# Torch imports
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
import gc


##################   Global Functions ####################

def reset_random_seeds(seed_value=43):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_parameters(label_file):
    # Define and update parameters based on the label file
    params = {
        'SNP_measure': 'genotype',
        'test_percent': 0.1,
        'val_percent': 0.1,
        'nb_epochs': 25,         # how does it differes from the number of epoch
        'batch_size': 20,
        'input_size': 1728,
        'num_workers': 16,       # normal 16, for debugging 0, or you will get an error in the DataLoaders
        'shuffle': True,
        'lr': 5e-06,
        'weight_decay': 8.01e-05,
        'class_nb': 3,
        'num_snps':10,
        'channels': [32, 64, 128, 128, 128, 64],
        'flip': False,
        'model': 'ResNet10',
        'version': 'b0', #b0, b1, b2, b3, b4, b5, b6, b7 for EfficientNet
        'load_pretrained': False,
        'freeze_pretrained': False,
        'pretrained_path': '/mnt/sdc/kholod/T1_NN/MedicalNet/pretrain/resnet_10_23dataset.pth',
        'model_dir': '/mnt/sdc/kholod/T1_NN/models',
        'test_file': '/path/to/results/test_',
        'train_file': '/path/to/results/train_',
        'val_file': '/path/to/results/val_',
        'sal_maps': '/mnt/sdc/kholod/T1_NN/results/nt_maps/GWAS12', # nt_maps nt_sq_maps GradCam_maps IntGrad_Maps GGC_maps
        'image_path':'/mnt/sdd/kholod/T1_images/T1_mni',
        'label_path': label_file,
        'device': 2,
        'sal_batch': 1,
        'sal_workers': 10,
        'shap_samples': 10,
        'activation_type': 'NT_IntGrad',
        'create_maps': True,
        'saliency_tasks': [5, 6, 7]
    }
    return params

def load_data(params):
    '''
    In this function is used to load the dataset and create the train, validation and test sets splits 
    using StratifiedShuffleSplit. This function also ensures that there is no overlap between the sets as
    well as print class distribution per set.
    '''

    # Read data and create datasets
    label_full_table = pd.read_csv(params['label_path'], sep=',')  
    initial_count = label_full_table.shape[0]
    label_full_table = label_full_table.drop_duplicates(subset='iid')
    duplicates_removed = initial_count - label_full_table.shape[0]
    print(f"Removed {duplicates_removed} duplicates from dataset.")
    
    # Prepare for class weight calculations
    class_weights = {}
    snp_columns = label_full_table.columns[1:] #first column is 'sample ID' and subsequent columns are SNPs
    # Calculate class weights for each SNP (task)
    for snp in snp_columns:
        class_counts = label_full_table[snp].value_counts().sort_index()
        # Ensure all classes 0, 1, 2 are present
        for cls in [0, 1]:
            if cls not in class_counts:
                class_counts[cls] = 0
        class_counts = class_counts.sort_index()

        weights = 1 / class_counts  # Calculate inverse frequency
        weights /= weights.max()  # Normalize weights by the maximum value

        class_weights[snp] = torch.tensor(weights.values, dtype=torch.float32)
    
    # Prepare labels for stratification, focusing on the first SNP as a placeholder
    labels = label_full_table[snp_columns].fillna(2).astype(int).values

    # Create dataset
    dataset = T1_dataset(params)
    dataset_size = len(dataset)
    print(f'Dataset length: {dataset_size}')
    
    
    # First split: train_val vs. test
    sss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=params['test_percent'], random_state=42)
    train_val_indices, test_indices = next(sss.split(np.zeros(len(labels)), labels))

    # Second split: train vs. val (within train_val)
    val_size = params['val_percent'] / (1 - params['test_percent'])
    sss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    train_indices_rel, val_indices_rel = next(sss_val.split(np.zeros(len(train_val_indices)), labels[train_val_indices]))

    # Convert relative indices to absolute (global index space)
    train_indices = train_val_indices[train_indices_rel]
    val_indices = train_val_indices[val_indices_rel]
    
    for task_idx, snp in enumerate(snp_columns):
        task_labels = label_full_table.iloc[train_indices][snp].fillna(2).astype(int).values
        print(f"Task {task_idx} ({snp}) - Class distribution:", Counter(task_labels))

    return dataset, train_indices, val_indices, test_indices, class_weights

    

def setup_model(params,class_weights= None):
    if not torch.cuda.is_available():
        raise Exception("Sorry, CUDA is necessary.")
    
    device = torch.device(params['device'])
    # Initialize the model based on the type specified in params

    if params['model'] == 'ResNet10':
        model = multitask_resnet10(num_tasks=params['num_snps'], num_classes_per_task=params['class_nb'], shortcut_type='B', no_cuda=False,  use_se=True, reduction=16, use_sa=False, sa_kernel_size=7).to(device)
    elif params['model'] == 'ResNet18':
        model = multitask_resnet18(num_tasks=params['num_snps'], num_classes_per_task=params['class_nb'], shortcut_type='B', no_cuda=False,  use_se=True, reduction=16, use_sa=False, sa_kernel_size=7).to(device)
    elif params['model'] == 'ResNet50':
        model = multitask_resnet50(num_tasks=params['num_snps'], num_classes_per_task=params['class_nb'], shortcut_type='B', no_cuda=False,  use_se=True, reduction=16, use_sa=True, sa_kernel_size=7).to(device)   
    else:
        raise ValueError(f"Unknown model type: {params['model']}")
    
    
    # Load pretrained weights if requested
    if params.get('load_pretrained', False):
        pretrained_path = params.get('pretrained_path', None)
        if pretrained_path is None:
            raise ValueError("Pretrained path must be provided when load_pretrained is True.")
        print(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        # Optionally filter out layers that don't match (for example, if you modified the head)
        model.load_state_dict(state_dict, strict=False)
    

    if params.get('freeze_pretrained', False):
        for name, param in model.named_parameters():
            # If the parameter name doesn't include 'fc_tasks', freeze it.
            if 'fc_tasks' not in name:
                param.requires_grad = False
        
    # Using DataParallel to utilize multiple GPUs
    model = nn.DataParallel(model, device_ids=[2, 3]).to(device)

    # Switch to evaluation mode to avoid side effects like dropout when summarizing
    model.eval()
    print(summary(model, input_size=(1, 1, 182, 218, 182)))  # Adjust input dimensions for summary if needed

    # Define the loss function, optimizer, and scheduler
    print("Class weights keys:", class_weights) 
        
    criterion = MaskedLoss(class_weights=None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    return model, criterion, optimizer, scheduler

def get_file_label(label_path):
    # Extracts a unique part of the label file name, because I will run multiple label files for APOE andother genes
    return os.path.basename(label_path).split('.')[0]

def train_and_evaluate(params):
    print("Entered train_and_evaluate")
    # Here I add current time just for naming the saved files
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset, train_indices, val_indices, test_indices, class_weights  = load_data(params)

    # Use the name of the label files to create a unique file name
    label = get_file_label(params['label_path'])
    params['label_file_name'] = label
    
    model, criterion, optimizer, scheduler = setup_model(params)
    
    model, history = trainer(model, dataset, train_indices, val_indices, params, criterion, optimizer, scheduler)
    
    print(type(history))
    print(history)
    if all(len(v) == len(history['train_loss']) for v in [history['valid_loss'], history['train_bAccuracy'], history['valid_bAccuracy']]):
        
        # Convert history dictionary to pandas DataFrame
        df_overall = pd.DataFrame({
            "ID": list(range(len(history['train_loss']))),
            "train_loss": history['train_loss'],
            "valid_loss": history['valid_loss'],
            "train_bAccuracy": history['train_bAccuracy'],
            "valid_bAccuracy": history['valid_bAccuracy'],

        })

        
        file_name_overall = f"/path/to/results/train_valid_{params['label_file_name']}_{params['model']}_model_{params['flip']}_augment_{params['nb_epochs']}_eps_{params['class_nb']}_class_{params['lr']}_lr_{params['activation_type']}_{current_time}.csv"
        
        # Save DataFrame to CSV
        try:
            
            df_overall.to_csv(file_name_overall, sep=',', index=False)
            print(f"Saved training and validation results to {file_name_overall}")
            
        except Exception as e:
            
            print(f"Failed to save CSV: {e}")

    else:
        print("Error: Mismatch in history data lengths, cannot save to CSV.")
        
    # Save task-specific metrics
 
    task_data = {'ID': []}
    
    for task_id in range(params['num_snps']):
        task_data[f'train_task_{task_id+1}_loss'] = history[f'train_task_{task_id+1}_loss']
        task_data[f'train_task_{task_id+1}_accuracy'] = history[f'train_task_{task_id+1}_accuracy']
        task_data[f'valid_task_{task_id+1}_loss'] = history[f'valid_task_{task_id+1}_loss']
        task_data[f'valid_task_{task_id+1}_accuracy'] = history[f'valid_task_{task_id+1}_accuracy']
    # Adjust ID to match the number of epochs or steps
    task_data['ID'] = list(range(len(history[f'train_task_1_loss'])))
    
    df_task_specific = pd.DataFrame(task_data)
    
    file_name_task_specific = f"/path/to/results/task_specific_{params['label_file_name']}_{params['model']}_model_{params['flip']}_augment_{params['nb_epochs']}_eps_{params['class_nb']}_class_{params['lr']}_lr_{params['activation_type']}_{current_time}.csv"

    try:
        df_task_specific.to_csv(file_name_task_specific, sep=',', index=False)
        print(f"Saved task-specific metrics to {file_name_task_specific}")
    except Exception as e:
        print(f"Failed to save task-specific CSV: {e}")
    
    print("Loading best model for testing...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_results = tester(model, dataset, test_indices, params, criterion)
    # Create saliency maps
    if params['create_maps'] == True:
        create_saliency(model, dataset, test_indices, params)
        
    return model, history, test_results


def main():
    label_files = [
                    '/path/to/data/example_label_file.csv',
                   ]

    for label_file in label_files:

        # Reset environment and clearing GPU for each training session
        reset_random_seeds()
        torch.cuda.empty_cache()

        # Setup parameters and specify the model and device
        params = setup_parameters(label_file)

        # Train and evaluate the model
        model, history, test_results = train_and_evaluate(params)

        print(f"Completed processing for: {label_file} ")

        # Explicit memory management
        del model, history, test_results
        gc.collect()  # Collect garbage to free memory
        torch.cuda.empty_cache()  # Clear GPU cache again after cleanup

if __name__ == "__main__":
    main()
"""
training_mn_v2: F1 and MCC metrics, also to change bAccuracy from batch wise to epoch wise calculation
Provides a training function, test function and fucntion to create saliency maps
"""
import matplotlib
matplotlib.use('Agg')  # Set a non-GUI backend

import os
import warnings
import time
import torch

import numpy as np
import pandas as pd
import nibabel as nib

from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import WeightedRandomSampler
# Calculate the sampling weights of each sample
from numpy import bincount

from torch.utils.data import DataLoader, Subset
from captum.attr import Saliency, NoiseTunnel
import shap
from captum.attr._utils import visualization as viz
from captum.attr._utils import attribution as attr_
from sklearn.metrics import f1_score, matthews_corrcoef

from torch.utils.tensorboard import SummaryWriter
from early_stopping import EarlyStopping
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime



# This function is used to save the model and optimizer state dict and the scheduler state dict 
def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")

# creating a function that account for missing classes in the batches and calculate the balanced accuracy accordingly by giving zero value to the missing classes.
def balanced_accuracy_with_zero_recall(y_true, y_pred, num_classes):
    # Convert inputs to tensor if they are not already
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    class_recall = torch.zeros(num_classes, device=y_true.device)
    for c in range(num_classes):
        class_mask = (y_true == c)
        if class_mask.any():
            correct_preds = torch.sum((y_pred == c) & class_mask).float()
            class_recall[c] = correct_preds / class_mask.sum() if class_mask.sum() > 0 else 0
    return torch.mean(class_recall).item()


            
def trainer(model, dataset, train_indices, val_indices, params, criterion, optimizer, scheduler):  #wheres the files of these parameters?
    '''
    Contains
    1. split of training set into train and validation
    2. Train the given model
    Returns
    1. Best performing model
    2. Arrays with train and validation loss
    '''
    # For tensor board logging 
    run_name = params['label_file_name']
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    writer = SummaryWriter(log_dir=f'runs/{run_name}') 
    
    # Put device correctly
    device = torch.device(params['device'])
    
    model.to(device)                         #this is to  train the model to a desired device (GPU), asign the data, model and loss func(criterion) to it using .to(device)
    criterion.to(device)

    # labels_pS can access get_labels which access row data from csv
    labels_pST = dataset.get_labels(range(len(dataset)))  # DataFrame of shape (num_samples, num_snps) (41382, 10), 
    labels_pST = labels_pST.replace( -1, 2 ).astype( int ) # When calculating the weight of each sample, we consider -1 as 2 in order to downsample a bit the people with missing values
    count_lTnC = [ bincount( labels_pST.iloc[ :, iT ] + 1 )[ 1: ]
                   for iT in range( labels_pST.shape[ 1 ] ) ]

    count_nTC = np.stack( count_lTnC, axis = 0 )
    assert all( count_lTnC[ 0 ] == count_nTC[ 0, : ] )
    weight_nS = np.zeros(labels_pST.shape[0])
    for iT in range( labels_pST.shape[ 1 ] ) :
        weight_nS += count_nTC[iT, labels_pST.iloc[:, iT]]
    # Invert and normalize
    weight_nS = 1.0 / (weight_nS + 1e-8)  # Add small constant to avoid division by zero
    weight_nS /= weight_nS.sum()
    

    # Step 5: Apply these weights to the training and validation indices
    train_weights_nS = weight_nS[train_indices]
    val_weights_nS = weight_nS[val_indices]

    # Step 6: Create subsets and samplers
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
      
    # Create samples and data loaders
    train_sampler = WeightedRandomSampler(train_weights_nS, bReplacement=True)
    val_sampler = SequentialSampler(val_subset)

    train_loader =DataLoader(train_subset, batch_size=params['batch_size'], sampler=train_sampler, prefetch_factor=2, num_workers=params['num_workers'])
    val_loader = DataLoader(val_subset, batch_size=params['batch_size'], sampler=val_sampler, num_workers=params['num_workers'])

    # instantiate history dict and array for epoch timing
    num_tasks = params['num_snps'] 
    # To store the averaged accuracy across all tasks 
    history = { 'train_loss': [],
                'train_bAccuracy': [],  
                'valid_loss': [],
                'valid_bAccuracy': []
            }

    data_store = {}
  
    
    # Initialize task-specific keys for each task
    history[f'valid_count_nETC'] = np.full( fill_value = 0,
                                            shape = ( params['nb_epochs'], num_tasks, 4 ) )
    history[f'train_count_nETC'] = np.full( fill_value = 0,
                                            shape = ( params['nb_epochs'], num_tasks, 4 ) )
    for task_id in range(num_tasks):
        history[f'train_task_{task_id+1}_loss'] = []
        history[f'train_task_{task_id+1}_accuracy'] = []
        history[f'valid_task_{task_id+1}_loss'] = []
        history[f'valid_task_{task_id+1}_accuracy'] = []
        history[f'valid_task_{task_id+1}_epoch_pval']=[]

    # Initialize a global dictionary to store gradients for all tasks across all epochs
    all_gradients = {task_id: [] for task_id in range(num_tasks)}
        
    epoch_step_time = []
    
    # Initialize a GradScaler for Mixed precision training
    scaler = GradScaler()   
    early_stopping = EarlyStopping(patience=10, delta=0.001, path='best_model.pth', verbose=True)
    
    try:
        # TRAINING/TESTING LOOP
        for epoch in range(0, params['nb_epochs']):
            # TRAINING EPOCH LOOP

            # Hale the user
            print("---------------------------------------------START TRAINING EPOCH---------------------------------------------")
            print( "Epoch: {}".format( epoch ) )
            print( "Number of minibatches: ", len( train_loader ) )
            
            # Save the NN in a checkpoint in order to be able to re-start training if there are any error
            checkpoint_path = os.path.join(params['model_dir'], f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
            
            # Initialise mintoring diagnostic variables
            step_start_time = time.time()
            train_loss=0.0
            train_bAcc = 0.0

            num_classes = params['class_nb']
            dataset.flip = params['flip'] # Initialize flip data augmentation according to parameters
            
            # initialize lists to track the cumulative accuracy and the number of valid samples for each task during an epoch
            # Initialize task losses and accuracies at the start of each epoch
            train_task_losses = [[] for _ in range(num_tasks)]
            train_task_accuracies = [[] for _ in range(num_tasks)]
            valid_task_losses = [[] for _ in range(num_tasks)]
            valid_task_accuracies = [[] for _ in range(num_tasks)]
            # Data lists for saving, restructured by task
            train_task_data = {task_id: {'preds': [], 'trues': [], 'ids': []} for task_id in range(num_tasks)}
            valid_task_data = {task_id: {'preds': [], 'trues': [], 'ids': []} for task_id in range(num_tasks)}
            data_store[epoch] = []
            # Initialize a dictionary to store gradients for each task
            epoch_gradients = {task_id: [] for task_id in range(num_tasks)}

                        
            # Run training loop
            model.train()
            
            for iB, (image, label_true, mask, name, label_val, aff_mat) in enumerate(train_loader):   # The program gets stuck if you try to get a sample from the data loader while debugging. To avoid the problem you need to create the data loader with "num_workers = 0"

                # backpropagate and optimize
                optimizer.zero_grad()
                # Show the user some of the monitoring variables
                ## Tell the user that training has started, such that he/she knows that the programm didn't get stuck in the beginning of the training loop
                if iB == 0 :
                    print( "Training of first sample" )
                ## Give to the user an estimation of how long the training is going to take
                if iB == 1 :
                    rTime = time.time()
                if iB == 4 :
                    print( "Estimated time for training one epoch (mins): ", ( time.time() - rTime ) * len( train_loader ) / 3 / 60 )
                # Wrap the training forward and backward passes with autocast and GradScaler
                with autocast():    
                    # Forwardpropagate input image                
                    image = image.float().to(device)
                    label_true = label_true.to(device)  #its shape, torch.Size([20, 10])
                    label_pred = model(image) # label_pred will be a list of outputs, one per task
                
                    # Initialize total loss for the batch
                    # total_loss is the aggregated loss for all tasks in a single batch. It’s calculated during the training loop and is used for backpropagation to update the model weights.
                    total_loss = 0
                    gradient_norms = []  # To store gradient norms for each task
                    
                    # This loop over each task in a batch to calcualte loss and accuracy per task then aggregate them
                    for task_id in range(num_tasks):
                        # Calculate loss for each task
                        task_output = label_pred[task_id].float()
                        task_target = label_true[:, task_id].long()
                        task_mask = mask[:, task_id].to(device)

                        valid = (task_target >= 0) & task_mask.bool()
                            # Calculate the loss for this task
                        if valid.any(): 
                            # Calculate the loss for this task
                            task_loss = criterion(task_output, task_target, task_mask, task_id=task_id) 
                            total_loss += task_loss.sum() # Aggregate losses from all tasks to be used later for backpropagation

                            # Accumulate the loss for each task separately, then average losses for the batch for the specific task
                            # Computing the average loss for each task is appended to train_task_losses, to track the loss for each task
                            train_task_losses[task_id].append(task_loss.mean().item())
                             
                            task_pred_classes = task_output[valid].argmax(dim=1).cpu().tolist()
                            task_true_classes = task_target[valid].cpu().tolist()
                            # Collect task-specific data for valid samples only
                            train_task_data[task_id]['preds'].extend(task_pred_classes)
                            train_task_data[task_id]['trues'].extend(task_true_classes)
                            train_task_data[task_id]['ids'].extend([name[i] for i in valid.nonzero(as_tuple=True)[0]]) # Collect valid IDs per task

                            # Count the number of samples per class
                            history[ f'train_count_nETC' ][ epoch, task_id, : ] += torch.bincount( task_target + 1, 
                                                                                                    minlength = 4 ).detach().cpu().numpy()
                # Scales the loss, and calls backward() to create scaled gradients
                scaler.scale(total_loss).backward()
                # Clip the gradients of all model parameters (shared + heads)

                scaler.step(optimizer)
                scaler.update()

            # Compute overall average loss and accuracy for the epoch for all tasks (for monitoring and logging purposes only not for backpropagation)
            # train_loss and avg_bAccuracy, avg_pval are metrics for all tasks combined)
            # Accumulate the total number of valid samples across all batches
            total_valid_samples_for_loss = sum([len(t) for t in train_task_losses])
            # Compute the average training loss by dividing by the total number of valid samples
            train_loss = sum([sum(t) for t in train_task_losses]) / total_valid_samples_for_loss  # The inner sum(t) sums up the losses for each task, the outer sum([sum(t), sums up these sums to get the total loss for all tasks.
            
            # Calculate the average balanced accuracy across tasks
            epoch_task_accuracies  = []
            epoch_task_f1 = []
            epoch_task_mcc = []
            for task_id in range(num_tasks):
                task_true = train_task_data[task_id]['trues']
                task_preds = train_task_data[task_id]['preds']
                if len(task_true) > 0:
                    # Compute balanced accuracy for the whole epoch for this task
                    task_accuracy = balanced_accuracy_with_zero_recall(task_true, task_preds, num_classes)
                    task_f1 = f1_score(task_true, task_preds, average='macro')
                    task_mcc = matthews_corrcoef(task_true, task_preds)
                else:
                    task_accuracy = task_f1 = task_mcc = 0.0
                epoch_task_accuracies.append(task_accuracy)
                epoch_task_f1.append(task_f1)
                epoch_task_mcc.append(task_mcc)
                print(f"  Task {task_id+1} - Epoch-wise Train Balanced Accuracy: {task_accuracy:.4f}, F1: {task_f1:.4f}, MCC: {task_mcc:.4f}")

            # Compute overall training accuracy averaged across tasks:
            avg_bAccuracy = sum(epoch_task_accuracies) / num_tasks if num_tasks > 0 else 0
            avg_f1 = sum(epoch_task_f1) / num_tasks if num_tasks > 0 else 0
            avg_mcc = sum(epoch_task_mcc) / num_tasks if num_tasks > 0 else 0

            history['train_loss'].append(train_loss)
            history['train_bAccuracy'].append(avg_bAccuracy)
            history.setdefault('train_F1', []).append(avg_f1)
            history.setdefault('train_MCC', []).append(avg_mcc)
            
            print(f"Epoch [{epoch+1}/{params['nb_epochs']}]:")
            print(f" Train average loss(all tasks): {train_loss:.4f}")
            print(f"  Train average bAccuracy(all tasks): {avg_bAccuracy:.4f}")
            
            ## Send some info to tensorboard
            writer.add_scalar('train_bAccuracy', avg_bAccuracy, global_step=epoch)              
            writer.add_scalar('Training Loss', train_loss, global_step=epoch)
            writer.add_scalar('train_F1', avg_f1, global_step=epoch)
            writer.add_scalar('train_MCC', avg_mcc, global_step=epoch)
            
            # Save the average loss and accuracy per task for this epoch, These per-task metrics are stored in the history dictionary for later analysis or plotting
            for task_id in range(num_tasks):
                avg_task_loss = sum(train_task_losses[task_id]) / len(train_task_losses[task_id])
                
                # Use the epoch-level accuracy computed above for this task
                epoch_task_accuracy = epoch_task_accuracies[task_id]
                history[f'train_task_{task_id+1}_loss'].append(avg_task_loss)
                history[f'train_task_{task_id+1}_accuracy'].append(epoch_task_accuracy)
                print(f"  Task {task_id+1} - Loss: {avg_task_loss:.4f}, Balanced Accuracy: {epoch_task_accuracy:.4f}")

                
            # Reset task losses and accuracies for the next epoch
            train_task_losses = [[] for _ in range(num_tasks)]
            
            for task_id in range(num_tasks):
                train_task_data[task_id]['preds'] = []
                train_task_data[task_id]['trues'] = []

            # Save time of epoch
            epoch_step_time.append(time.time() - step_start_time)

            # VALIDATION EPOCH LOOP
            
            # Hale the user
            print("---------------------------------------------START VALIDATION EPOCH ---------------------------------------------")
            print( "Number of minibatches: ", len( val_loader ) )
            num_classes = params['class_nb']
            # Set model in evaluation loop
            model.eval()
            
            # Disable flip in data augmentation
            dataset.flip = False
            
            # Initialise monitoring variables
            valid_loss = 0.0
            
            # Disable warnings 
            warnings.filterwarnings('ignore', 'y_pred contains classes not in y_true')
             
            
            
            # Run the validation loop
            with torch.no_grad():
                for iB, (image, label_true, mask, name, label_val, aff_mat) in enumerate(val_loader):

                    
                    with autocast():
                        # Give to the user an estimation of how long the training is going to take
                        if iB == 1 :
                            rTime = time.time()
                        if iB == 4 :
                            print( "Estimated time for validating this epoch (mins): ", ( time.time() - rTime ) * len( val_loader ) / 3 / 60 )
                    
                        # Read image and label, run model
                        image = image.float().to(device)
                        label_true = label_true.to(device)
                        label_pred = model(image)
                        # Initialize total loss for the batch
                        total_loss = 0
                        processed_pairs = set()  # Initialize a set to track processed (sample_id, task_id) pairs

                        for task_id in range(num_tasks):
                            # Calculate loss for each task
                            task_output = label_pred[task_id].float()
                            task_target = label_true[:, task_id].long()
                            task_mask = mask[:, task_id].to(device)

                            # Calculate the loss for this task
                            if task_mask.any(): # If any True, Only calculate accuracy if there are valid samples for this task
                                valid_indices = task_mask.nonzero(as_tuple=True)[0]    # Gets the indices of the True values in task_mask, i.e., the indices of the valid samples.
                                task_loss = criterion(task_output, task_target, task_mask) 
                                total_loss += task_loss.sum() # Aggregate losses from all tasks to be used later for backpropagation
                                
                                # Accumulate the loss for each task separately, then average losses for the batch for the specific task
                                # Computing the average loss for each task is appended to train_task_losses, to track the loss for each task
                                valid_task_losses[task_id].append(task_loss.mean().item())
                                
                                # Get predictions and ground truths for valid indices
                                task_pred_classes = task_output[valid_indices].argmax(dim=1).cpu().tolist()
                                task_true_classes = task_target[valid_indices].cpu().tolist()
                                task_probs = torch.softmax(task_output, dim=1).cpu().tolist()
                                
                                # Sanity check
                                valid_indices_list = valid_indices.tolist() if isinstance(valid_indices, torch.Tensor) else valid_indices
                                assert len(valid_indices_list) == len([task_probs[i] for i in valid_indices_list]), "Mismatch in valid indices and probabilities!"
                                
                                valid_task_data[task_id]['preds'].extend(task_pred_classes)
                                valid_task_data[task_id]['trues'].extend(task_true_classes)
                                valid_task_data[task_id]['ids'].extend([name[i] for i in valid_indices])  # Collect valid IDs per task

                                
                                # Store the predicted probabilities and true labels with sample ID, epoch, and task number
                                for idx in valid_indices:
                                    sample_id = name[idx]
                                    probabilities = task_probs[idx]
                                    true_label = task_target[idx].item()
                                    data_store[epoch].append([sample_id, task_id, epoch, *probabilities, true_label])
                                
                                # Count the number of samples per class
                                history[ f'valid_count_nETC' ][ epoch, task_id, : ] += torch.bincount( task_target + 1, 
                                                                                                       minlength = 4 ).detach().cpu().numpy()
                # Compute overall validation loss over all tasks:       
                total_valid_samples_for_loss = sum([len(t) for t in valid_task_losses])
                valid_loss = sum([sum(t) for t in valid_task_losses]) / total_valid_samples_for_loss
                print (valid_loss)
                
                
                # Compute epoch-level balanced accuracy per tasks
                epoch_task_accuracies = []
                epoch_task_f1 = []
                epoch_task_mcc = [] 
                
                for task_id in range(num_tasks):
                    task_true = valid_task_data[task_id]['trues']
                    task_preds = valid_task_data[task_id]['preds']
                    if len(task_true) > 0:
                        task_accuracy = balanced_accuracy_with_zero_recall(task_true, task_preds, num_classes)
                        task_f1 = f1_score(task_true, task_preds, average='macro')
                        task_mcc = matthews_corrcoef(task_true, task_preds)

                    else:
                        task_accuracy = 0.0
                        task_f1 = 0.0
                        task_mcc = 0.0
                        
                    epoch_task_accuracies.append(task_accuracy)
                    epoch_task_f1.append(task_f1)
                    epoch_task_mcc.append(task_mcc)
                    print(f"  Task {task_id+1} - Epoch-wise Valid Balanced Accuracy: {task_accuracy:.4f}, F1: {task_f1:.4f}, MCC: {task_mcc:.4f}")


                avg_bAccuracy = sum(epoch_task_accuracies) / num_tasks if num_tasks > 0 else 0
                avg_f1 = sum(epoch_task_f1) / num_tasks if num_tasks > 0 else 0
                avg_mcc = sum(epoch_task_mcc) / num_tasks if num_tasks > 0 else 0

                print(f"  valid loss: {valid_loss:.4f}")
                print(f"  valid average balanced accuracy: {avg_bAccuracy:.4f}")
                print(f"  valid average F1: {avg_f1:.4f}")
                print(f"  valid average MCC: {avg_mcc:.4f}")
                 
                
                # Save the average loss and accuracy per task for this epoch, These per-task metrics are stored in the history dictionary for later analysis or plotting
                for task_id in range(num_tasks):

                    avg_task_loss = sum(valid_task_losses[task_id]) / len(valid_task_losses[task_id])
                    # Use the epoch-level accuracy computed above for this task
                    epoch_task_accuracy = epoch_task_accuracies[task_id]
                    epoch_task_f1_val = epoch_task_f1[task_id]
                    epoch_task_mcc_val = epoch_task_mcc[task_id]

                    history[f'valid_task_{task_id+1}_loss'].append(avg_task_loss)
                    history[f'valid_task_{task_id+1}_accuracy'].append(epoch_task_accuracy)
                    history.setdefault(f'valid_task_{task_id+1}_f1', []).append(epoch_task_f1_val)
                    history.setdefault(f'valid_task_{task_id+1}_mcc', []).append(epoch_task_mcc_val)

                    print(f"  Task {task_id+1} - Loss: {avg_task_loss:.4f}, Balanced Accuracy: {epoch_task_accuracy:.4f}, F1: {epoch_task_f1_val:.4f}, MCC: {epoch_task_mcc_val:.4f}")


                # Reset task losses and accuracies for the next epoch
                valid_task_losses = [[] for _ in range(num_tasks)] 
                for task_id in range(num_tasks):
                    valid_task_data[task_id]['preds'] = []
                    valid_task_data[task_id]['trues'] = []

                # Reset the warning
                warnings.filterwarnings('default', 'y_pred contains classes not in y_true')

                # Early stopping
                early_stopping(avg_bAccuracy, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break
                
                
                # Send some info to tensorboard
                writer.add_scalar('valid_bAccuracy', avg_bAccuracy, global_step=epoch)      
                writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
                writer.add_scalar('valid_F1', avg_f1, global_step=epoch)
                writer.add_scalar('valid_MCC', avg_mcc, global_step=epoch)
                # Log the overall validation loss and accuracy
                history['valid_loss'].append(valid_loss)
                history['valid_bAccuracy'].append(avg_bAccuracy)
                
                # Update learning Rate Scheduler
                scheduler.step((valid_loss))

                
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        if input("Do you want to save the current model? (yes/no): ").lower() == "yes":
            checkpoint_path = os.path.join(params['model_dir'], 'interrupted_checkpoint.pth')
            # Save the model and optimizer state dict and the scheduler state dict, used in case I want to resume training
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
        
        # Close the TensorBoard writer
        writer.close()
        print("TensorBoard writer closed and data has been flushed to disk.")

    # save final model, this saved model can be loaded and used for testing
    torch.save(model.state_dict(keep_vars=True),os.path.join(params['model_dir'], params['SNP_measure'] + "_" +str(params['label_file_name'])+"_" + str(params['model'])+ "_model_" + str(params['flip']) + "_augment_" + \
    str(params['nb_epochs']) + "_eps_" + str(params['class_nb'])+'_class_'+ str(params['lr'])+"_lr_" +params['activation_type'] + 'final%04d.pt' % epoch))


    # Write labels to CSV
    import csv
    
    # Writing predictions to CSV
    output_file_path = f"/path/to/results/{params['label_file_name']}_{current_time}_data.csv"
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header row with clear column names
        writer.writerow([
            'Sample_ID', 'Task_ID', 'Epoch', 
            'Prob_Class_0', 'Prob_Class_1', 'Prob_Class_2', 
            'True_Label'
        ])
        # Iterate through the dictionary to flatten the data
        for epoch, entries in data_store.items():
            for entry in entries:
                writer.writerow(entry)  # Write each row
    
    return model, history #, normalized_difficulty


def tester(model, dataset, test_indices, params,criterion):
    
    # Generate dataset & DataLoaders
    test_subset = Subset(dataset, test_indices) 
    test_sampler = RandomSampler(test_subset)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=params['batch_size'], 
                                              sampler=test_sampler, num_workers=params['num_workers'])

    device = torch.device(params['device'])
    num_classes = params['class_nb']
    num_tasks = params['num_snps']
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize storage for per-task predictions, labels, and IDs
    test_task_data = {task_id: {'preds': [], 'trues': [], 'ids': []} for task_id in range(num_tasks)}    

    print("---------------------------------------------START TEST---------------------------------------------")
    print( "Number of minibatches: ", len( test_loader ) )
    model.eval()
    dataset.flip = False # Disable flip augmentation for testing
    
    # Initialize accumulators
    total_loss = 0
    total_valid_samples = [0] * num_tasks
    test_task_losses = [[] for _ in range(num_tasks)]
    test_data_store = []

    with torch.no_grad():
        for iB, (image, label_true, mask, name, label_val, aff_mat) in enumerate(test_loader):
            image = image.float().to(device)
            label_true = label_true.to(device)
            label_pred = model(image)
            mask = mask.to(device)

            for task_id in range(num_tasks):
                task_output = label_pred[task_id].float()
                task_target = label_true[:, task_id].long()
                task_mask = mask[:, task_id].to(device)
                
                
                # Calculate accuracy for this task, applying the mask
                if task_mask.any(): # If any True, Only calculate accuracy if there are valid samples for this task
                    valid_indices = task_mask.nonzero(as_tuple=True)[0]    # Gets the indices of the True values in task_mask, i.e., the indices of the valid samples.
                    # Calculate the loss for this task
                    task_loss = criterion(task_output, task_target, task_mask)
                    total_loss += task_loss.sum().item() # Aggregate losses from all tasks to be used later for backpropagation
                    
                    # Accumulate the loss for each task separately, then average losses for the batch for the specific task
                    test_task_losses[task_id].append(task_loss.mean().item())
                    
                    task_pred_classes = task_output[valid_indices].argmax(dim=1).cpu().tolist()
                    task_true_classes = task_target[valid_indices].cpu().tolist()
                    task_probs = torch.softmax(task_output, dim=1).cpu().tolist()
                    
                    # Collect task-specific data for valid samples only
                    test_task_data[task_id]['preds'].extend(task_pred_classes)
                    test_task_data[task_id]['trues'].extend(task_true_classes)
                    test_task_data[task_id]['ids'].extend([name[i] for i in valid_indices])  # Collect valid IDs per task
                    # Store the predicted probabilities and true labels with sample ID, epoch, and task number
                    for idx in valid_indices:
                        sample_id = name[idx]
                        probabilities = task_probs[idx]
                        true_label = task_target[idx].item()
                        # Store each probability separately
                        test_data_store.append([sample_id, task_id, *probabilities, true_label])

                    total_valid_samples[task_id] += valid_indices.size(0)
                else:
                    # For batches with no valid sample, nothing is accumulated.
                    continue
                    
        # Compute average metrics
        total_valid_samples_for_loss = sum(total_valid_samples)
        avg_test_loss = total_loss / total_valid_samples_for_loss if total_valid_samples_for_loss > 0 else 0

        # Compute epoch-level balanced accuracy per task using the aggregated data
        epoch_task_accuracies = []
        epoch_task_f1 = []
        epoch_task_mcc = []
        for task_id in range(num_tasks):
            task_true = test_task_data[task_id]['trues']
            task_preds = test_task_data[task_id]['preds']
            if len(task_true) > 0:
                task_accuracy = balanced_accuracy_with_zero_recall(task_true, task_preds, num_classes)
                task_f1 = f1_score(task_true, task_preds, average='macro')
                task_mcc = matthews_corrcoef(task_true, task_preds)
            else:
                task_accuracy = task_f1 = task_mcc = 0.0
            
            epoch_task_accuracies.append(task_accuracy)
            epoch_task_f1.append(task_f1)
            epoch_task_mcc.append(task_mcc)
            print(f"  Task {task_id+1} - Test BAcc: {task_accuracy:.4f}, F1: {task_f1:.4f}, MCC: {task_mcc:.4f}")

        avg_test_accuracy = sum(epoch_task_accuracies) / num_tasks if num_tasks > 0 else 0
        avg_test_f1 = sum(epoch_task_f1) / num_tasks if num_tasks > 0 else 0
        avg_test_mcc = sum(epoch_task_mcc) / num_tasks if num_tasks > 0 else 0

        print(f'Overall test results: Test set loss: {avg_test_loss:.4f}, BAcc: {avg_test_accuracy:.4f}, F1: {avg_test_f1:.4f}, MCC: {avg_test_mcc:.4f}')
 
        # Save the average loss and accuracy per task for this epoch, These per-task metrics are stored in the history dictionary for later analysis or plotting
        task_data = {}
        for task_id in range(num_tasks):
            avg_task_loss = (sum(test_task_losses[task_id]) / len(test_task_losses[task_id]) 
                            if len(test_task_losses[task_id]) > 0 else 0)

            avg_task_accuracy = epoch_task_accuracies[task_id]
            epoch_task_f1_val = epoch_task_f1[task_id]
            epoch_task_mcc_val = epoch_task_mcc[task_id]

            print(f"  Task {task_id+1} - Loss: {avg_task_loss:.4f}, BAcc: {avg_task_accuracy:.4f}, F1: {epoch_task_f1_val:.4f}, MCC: {epoch_task_mcc_val:.4f}")
            task_data[f'test_task_{task_id+1}_loss'] = [avg_task_loss]
            task_data[f'test_task_{task_id+1}_accuracy'] = [avg_task_accuracy]
            task_data[f'test_task_{task_id+1}_f1'] = [epoch_task_f1_val]
            task_data[f'test_task_{task_id+1}_mcc'] = [epoch_task_mcc_val]
            
        task_data['ID'] = [0]   # Since the metrics are now overall/averaged, there's no need for a unique ID per row

    df_task_specific = pd.DataFrame(task_data)
    file_name_task_specific = f"/path/to/results/test_task_specific_{params['label_file_name']}_{params['model']}_model_{params['flip']}_augment_{params['nb_epochs']}_eps_{params['class_nb']}_class_{params['lr']}_lr_{params['activation_type']}.csv"

    try:
        df_task_specific.to_csv(file_name_task_specific, sep=',', index=False)
        print(f"Saved task-specific metrics to {file_name_task_specific}")
    except Exception as e:
        print(f"Failed to save task-specific CSV: {e}")

    
    # Write labels to CSV
    import csv
    
    # Writing predictions to CSV
    output_file_path = f"/path/to/results/{params['label_file_name']}_{current_time}_test_data.csv"
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header row with clear column names
        writer.writerow([
            'Sample_ID', 'Task_ID',
            'Prob_Class_0', 'Prob_Class_1', 'Prob_Class_2',
            'True_Label'
        ])
        # Write each record
        for entry in test_data_store:
            writer.writerow(entry)
            
    for task_id in range(num_tasks):
       # Once all batches are processed, convert tensor IDs to integers
        test_task_data[task_id]['ids'] = [int(id) for id in test_task_data[task_id]['ids']]
    
    for task_id in range(num_tasks):
        #Save test results
        test_df = pd.DataFrame({
            "ID": test_task_data[task_id]['ids'],
            "Prediction": test_task_data[task_id]['preds'],
            "True_Label": test_task_data[task_id]['trues']
            })
            
        test_df.to_csv(
            f"{params['test_file']}_task_{task_id+1}_{params['SNP_measure']}_{params['label_file_name']}_{params['model']}_model_{params['flip']}_augment_{params['nb_epochs']}_eps_{params['class_nb']}_class_{params['lr']}_lr_{params['activation_type']}.csv", 
            sep=',', 
            index=False
        )

    
    # Return test loss and balanced accuracy
    return avg_test_loss, avg_test_accuracy, avg_task_loss, epoch_task_accuracies

class HeadWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, head_idx: int):
        super().__init__()
        self.model    = model
        self.head_idx = head_idx

    def forward(self, x):
        # model(x) returns a list/tuple of logits per head
        out = self.model(x)
        return out[self.head_idx] if isinstance(out, (list,tuple)) else out[:, self.head_idx]

def create_saliency(model, dataset, test_indices, params):
    """
    Compute saliency maps for selected tasks,
    1) Pre-filter: only correctly classified samples
    2) For each: compute SmoothGrad saliency (σ=0.01, 25 samples)
    3) Per-subject min-max normalize to [0,1]
    4) Save one NIfTI per subject×task
    """
    device = torch.device(params['device'])
    
    model.to(device).eval()
    dataset.flip = False
    
    # 1) which tasks?
    num_tasks = params['num_snps']
    selected  = params.get('saliency_tasks', list(range(num_tasks)))
    assert all(0 <= t < num_tasks for t in selected), f"Invalid tasks {selected}"
    noise_std  = params.get('noise_std', 0.01)
    n_samples  = params.get('n_samples', 25)
    
    # build a wrapper+explainer per task
    tunnel_dict   = {}
    for t in selected:
        wrapper = HeadWrapper(model, t).to(device)
        tunnel = NoiseTunnel(Saliency(wrapper))
        tunnel_dict[t] = tunnel
    
    
    
    # 2) DataLoader
    test_subset = Subset(dataset, test_indices)
    test_loader = DataLoader(
        test_subset,
        batch_size=params['sal_batch'], 
        num_workers=params['sal_workers']
    )
    print(f"→ Generating SmoothGrad maps for tasks {selected}")

    act = params['activation_type']

    # 4) Now loop over your data and *use* those explainers
    for image, label_true, mask, name, label_val, aff_mat in test_loader:
        image      = image.to(device).float()

        label_true = label_true.to(device).long()[0]  # [1, num_tasks]
        sample_id_raw = name[0] if isinstance(name, (tuple, list)) else name
        sample_id = str(int(sample_id_raw))  # now sample_id is a plain string without extra chars
        affine     = aff_mat[0].cpu().numpy()
        
        preds = {}
        with torch.no_grad():
            logits = model(image)
            if isinstance(logits, (list, tuple)):
                # [1, num_tasks, num_classes]
                logits = torch.stack(logits, dim=1)
            preds  = logits.argmax(dim=-1)[0]  # shape (num_snps,)

        # 2) for each task, if correct → compute SmoothGrad
        for t in selected:
            if preds[t] != label_true[t]:
                continue  # skip mis‐classifications
            
            # enable grad only for saliency
            x = image.clone().requires_grad_(True)
            attr = tunnel_dict[t].attribute(
                x, nt_type='smoothgrad', stdevs=noise_std,
                nt_samples=n_samples, target=label_true[t].item()
            )
            arr = attr.squeeze().cpu().numpy()
            mn, mx = arr.min(), arr.max()
            arr = (arr - mn) / (mx - mn) if mx>mn else np.zeros_like(arr)

            # 4) save per-subject, per-task NIfTI
            out_fn = os.path.join(
                params['sal_maps'],
                f"{params['SNP_measure']}_"
                f"{params['label_file_name']}_"
                f"{params['model']}_"
                f"task{t}_"
                f"{sample_id}_SmoothGrad.nii.gz"
            )
            nib.Nifti1Image(arr.astype(np.float32), affine).to_filename(out_fn)
            print(f"saved saliency for subject {sample_id}, task {t}")

    return 0


def create_shap_saliency_maps(model, dataset, test_indices, params):
    """
    Function to create SHAP saliency maps
    """
    device = torch.device(params['device'])
    model.to('cpu').eval()
    dataset.flip = False  # Ensure dataset is not in augmentation mode

    # DataLoader for the test set
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=params['sal_batch'], sampler=torch.utils.data.RandomSampler(test_indices), num_workers=params['sal_workers'])
    
    # Select a representative subset of data as the background distribution
    num_background_samples = min(2500, len(dataset))  # Set the number of background samples
    random_indices = torch.randint(len(dataset), (num_background_samples,)).tolist()  # Generate random indices
    background_samples = [torch.from_numpy(dataset[i][0]) for i in random_indices]  # Convert dataset elements to tensors
    background = next(iter(test_loader))[0].float().to('cpu')


    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.to('cpu')
    explainer = shap.DeepExplainer(model.to('cpu'), background.to('cpu'))  # Move to CPU for SHAP analysis

    for i, (input_data, _, _, name, aff_mat) in enumerate(test_loader):
        input_data = input_data.float().to('cpu')  
        shap_values = explainer.shap_values(input_data)
        shap_values_np = np.array(shap_values)

        # Print the shape of shap_values to understand its structure
        print("Shape of shap_values:", shap_values_np.shape)
        # Assuming you want to visualize the SHAP values for the first class (index 0)
        # You remove the first three dimensions, keeping the dimensions of the 3D image
        # Handle SHAP values for each class
        for class_index in range(shap_values_np.shape[0]):
            shap_values_class = shap_values_np[class_index, 0, 0]  # Adjusted SHAP values for the current class
            
            # Ensure aff_mat is correctly shaped
            aff_mat_corrected = aff_mat.squeeze(0)
            if aff_mat_corrected.shape != (4, 4):
                raise ValueError(f"Invalid affine matrix shape: {aff_mat_corrected.shape}")

            # Save the SHAP values for the current class to Nifti images
            shap_nii = nib.Nifti1Image(shap_values_class, aff_mat_corrected.numpy())
            nib.save(shap_nii, os.path.join(params['sal_maps'],params['SNP_measure'] 
            + "_" + str(params['model']) + "_model_" + str(name.item()) + "_" + str(params['flip']) 
            + "_augment_" + str(params['nb_epochs']) + "_eps_" + str(params['class_nb'])+'_class_'+params['activation_type']+".nii"))
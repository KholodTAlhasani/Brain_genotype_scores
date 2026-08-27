""" 
Dataset utilities for the brain-genotype score analysis.

This implementation was adapted from an earlier script developed by
Joshua Sammet. It has been substantially modified for the present
study, including data handling, imaging type, preprocessing, and task-specific processing.
"""
import os
import pandas as pd
import nibabel as nib
import numpy as np
import random
import imgaug.augmenters as iaa

from torch.utils import data
import torch


class T1_dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    def __init__(self,params, limit=None, transform=None, global_mean=None, global_std=None):                       
        '''                                           
        params: Parameter file containg information
        '''
        
        # ensure label file is csv
        assert params['label_path'].endswith('.csv')

        # Load label file 
        self.label_full_table = pd.read_csv(params['label_path'])
        
        # Remove duplicates based on the sample ID column (keeping only the first occurrence)
        self.label_full_table = self.label_full_table.drop_duplicates(subset='iid')
        
        # Handle NaN values immediately after loading by replacing them with -1 (applying mask later in getitem)
        self.label_full_table.fillna(-1 , inplace=True)

        # store class number and genes number
        self.class_nb = params['class_nb'] 
        self.num_snps = params['num_snps']
        # store image path a self var of dataset
        self.img_dir=params['image_path']

        # parameter if images should be augmented
        self.flip = params['flip']
        self.global_mean = global_mean
        self.global_std = global_std

        # Save the whole params object
        self.params = params
        
         # Optionally limit the number of samples
        if limit:
            self.label_full_table = self.label_full_table.head(limit)
        
        
    def __len__(self):
        return len(self.label_full_table)
     

    def augment_mri_images(self, images):
    # Define augmentation pipeline
        augmentation = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1)),
        iaa.GaussianBlur(sigma=(0.0, 1.0))
    ])
        
        # Augment each image independently
        images_aug = []
        for image in images:
          if bool(random.getrandbits(1)):
            augmented_image = augmentation.augment_image(image)
            images_aug.append(augmented_image)
          else:
            images_aug.append(image)

         # Convert images back to numpy array format
        images_aug = np.array(images_aug)
        return images_aug
    
    def drop_invalid_range(self, volume):
        """
        Crop the volume to the bounding box of nonzero voxels.
        """
        non_zero_coords = np.where(volume != 0)
        if non_zero_coords[0].size == 0:
            # If the volume is completely zero, return it unchanged.
            return volume
        z_min, z_max = non_zero_coords[0].min(), non_zero_coords[0].max()
        y_min, y_max = non_zero_coords[1].min(), non_zero_coords[1].max()
        x_min, x_max = non_zero_coords[2].min(), non_zero_coords[2].max()
        # Add 1 to include the boundary voxel.
        return volume[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    
    def intensity_normalize(self, volume):
        """
        Z-score normalize the brain (non-zero) voxels and leave background at zero.
        """
        # Create a mask of the brain region
        brain_mask = volume != 0

        # Extract brain voxels
        brain_vals = volume[brain_mask].astype(np.float32)

        # If there's no brain (empty mask), just return a float copy
        if brain_vals.size == 0:
            return volume.astype(np.float32)

        # Compute mean and std (add tiny epsilon to avoid divide-by-zero)
        mean = brain_vals.mean()
        std = brain_vals.std() if brain_vals.std() > 0 else 1.0

        # Allocate output
        normalized = np.zeros_like(volume, dtype=np.float32)

        # Z-score inside mask
        normalized[brain_mask] = (brain_vals - mean) / std

        return normalized

    def get_labels(self, indices):
        # Method to get labels for specific indices
        return self.label_full_table.iloc[indices, 1:self.num_snps + 1]  # Adjusted for multi-SNP labels
      
    
    def __getitem__(self, index):
        """Generates one sample of data
        Arguments: 
        index: Number of element in dataset
  
        Returns:
        image: T1MRI of subject for respective index
        labels: Tensor of SNP labels for the subject
        mask: Mask indicating valid SNP labels (1 for valid, 0 for NaN)
        img_id: ID of subject for respective index
        label_val: Original SNP measure of subject for respective index
        self.label_file.iloc[index, 0]: ID of subject for respective index
        nifti_img.affine: affine transformation of NIFTI file of image (needed for activation map storage)
        """
        
        # Create image path and make image to numpy array
        img_id = self.label_full_table.iloc[index, 0]
        # Create image path and make image to numpy array
        img_path = os.path.join(self.img_dir, f"{img_id}_T1_brain_to_MNI.nii.gz")
        nifti_img = nib.load(img_path)
        image = np.asarray(nifti_img.get_fdata().astype(np.float64))

        # Preprocessing steps for classification:
        image = self.intensity_normalize(image)

        # Convert numpy array to PyTorch tensor and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0)

        # rotates 50% of images with maximum angle of 10°. Returns image
        if self.flip == True:
            image = self.augment_mri_images([image])[0]
            
        # Get SNP labels and create a mask for NaN values
        label_val = self.label_full_table.iloc[index, 1:self.num_snps + 1].values
        label = torch.tensor(label_val).long()
    
        # Mask NaN values
        mask = label >= 0  # Mask where labels are valid (not -1), after discusson with Alejo, masking here will not work due to getitem way of working it will simply fitch data as it is without filtering or nasking because it output a tensor so if removing -1 removing the sample with all its labels -1 and other valid labels

        return image, label, mask, img_id, label_val, nifti_img.affine


     
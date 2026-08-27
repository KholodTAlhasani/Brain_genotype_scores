
import os
import numpy as np
import nibabel as nib
import pandas as pd

# ── PARAMETERS ────────────────────────────────────────────────────────────────
att_map_dir = '/mnt/sdc/kholod/T1_NN/results/nt_maps/GWAS7'
orig_img_dir = '/mnt/sdd/kholod/T1_images/T1_mni'
aff_path = '/home/kalhasani/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
output_dir = '/mnt/sdc/kholod/T1_NN/results/nt_maps/GWAS7'
folder_and_prefix = 'task_5_GWAS7_'

test_df = pd.read_csv('/path/to/results/example_test_predictions.csv').set_index('ID')
info_df = pd.read_csv('/path/to/data/example_phenotype_data.csv').set_index('f.eid')

# Ensure same dtype for indexing
test_df.index = test_df.index.astype(int)
info_df.index = info_df.index.astype(int)

affine = nib.load(aff_path).affine

# ── ACCUMULATORS ──────────────────────────────────────────────────────────────
sum_all, count_all = None, 0

sum_class = {c: None for c in [0,1,2]}
count_class = {c: 0 for c in [0,1,2]}

# NEW: per-sex accumulators
sum_sex = {0: None, 1: None}
count_sex = {0: 0, 1: 0}

processed = 0

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
for subj, row in test_df.iterrows():
    pred = int(row['Prediction'])
    true = int(row['True_Label'])

    if pred != true:
        continue

    # --- load maps ---
    sal_f = os.path.join(att_map_dir, f"genotype_pruned_005_02_filtered_2_subgroup_GWAS_7_ResNet10_task5_{subj}_SmoothGrad.nii.gz")
    orig_f = os.path.join(orig_img_dir, f"{subj}_T1_brain_to_MNI.nii.gz")

    try:
        sal = nib.load(sal_f).get_fdata().astype(np.float64)
    except Exception:
        print(f"[WARN] Missing map for {subj}, skipping")
        continue

    # --- get sex safely ---
    if subj not in info_df.index:
        print(f"[WARN] Missing metadata for {subj}, skipping")
        continue

    sex = info_df.loc[subj, 'Sex']
    if pd.isna(sex):
        continue
    sex = int(sex)

    processed += 1

    # --- global ---
    if sum_all is None:
        sum_all = np.zeros_like(sal)
    sum_all += sal
    count_all += 1

    # --- per class ---
    if sum_class[pred] is None:
        sum_class[pred] = np.zeros_like(sal)
    sum_class[pred] += sal
    count_class[pred] += 1

    # --- per sex ---
    if sum_sex[sex] is None:
        sum_sex[sex] = np.zeros_like(sal)
    sum_sex[sex] += sal
    count_sex[sex] += 1

print(f"Processed: {processed}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
threshold = 0.12

# --- class maps (thresholded only, unchanged) ---
for cls in [0,1,2]:
    if count_class[cls] == 0:
        continue

    mean_cls = sum_class[cls] / count_class[cls]
    mean_cls_thresh = np.where(mean_cls >= threshold, mean_cls, 0)

    out_fn = os.path.join(output_dir,
        f"{folder_and_prefix}_salmap_class{cls}_th{threshold}.nii.gz")

    nib.Nifti1Image(mean_cls_thresh.astype(np.float32), affine).to_filename(out_fn)

# --- overall maps (keep both) ---
if count_all > 0:
    mean_all = sum_all / count_all
    mean_all_thresh = np.where(mean_all >= threshold, mean_all, 0)

    nib.Nifti1Image(mean_all.astype(np.float32), affine).to_filename(
        os.path.join(output_dir, f"{folder_and_prefix}_salmap_all_nothreshold.nii.gz"))

    nib.Nifti1Image(mean_all_thresh.astype(np.float32), affine).to_filename(
        os.path.join(output_dir, f"{folder_and_prefix}_salmap_all_th{threshold}.nii.gz"))

# --- NEW: per-sex maps (NON-thresholded as requested) ---
for sex in [0,1]:
    if count_sex[sex] == 0:
        continue

    mean_sex = sum_sex[sex] / count_sex[sex]

    out_fn = os.path.join(
        output_dir,
        f"{folder_and_prefix}_salmap_sex{sex}_nothreshold_all.nii.gz"
    )

    nib.Nifti1Image(mean_sex.astype(np.float32), affine).to_filename(out_fn)

    print(f"Saved sex {sex} (n={count_sex[sex]}) → {out_fn}")


### Here I am comparing two models, full_model and reduced_model using F-test and delta R2
# the full model have two probabilities predictors, and the reduced model have only the confounders.
# The null hypothesis is that the probabilities predictors Prob_Class_0 and Prob_Class_2 do not improve the model fit over the reduced model (which only includes confounders).
# Significant p-values indicate that probabilities predictors significantly improve the model fit.
# Created by Kholod Alhasani

# REFF FOR COVAR:
# Sniekers, S., Stringer, S., Watanabe, K. et al. Genome-wide association meta-analysis of 78,308 individuals identifies new loci and genes influencing human intelligence. Nat Genet 49, 1107–1112 (2017). https://doi.org/10.1038/ng.3869
# Davies, G., Lam, M., Harris, S.E. et al. Study of 300,486 individuals identifies 148 independent genetic loci influencing general cognitive function. Nat Commun 9, 2098 (2018). https://doi.org/10.1038/s41467-018-04362-x
# de la Fuente, J., Davies, G., Grotzinger, A.D. et al. A general dimension of genetic sharing across diverse cognitive traits inferred from molecular data. Nat Hum Behav 5, 49–58 (2021). https://doi.org/10.1038/s41562-020-00936-2
# Wootton et al. 2023  preprint 

import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load your data
snp_file = '/path/to/data/example_scores.csv'
phenotype_file = '/path/to/data/example_phenotype_data.csv'
IDP_file = '/path/to/data/example_idp_data.csv' 
pca_file  = '/path/to/data/example_pca_data.csv'


snp_data = pd.read_csv(snp_file)
phenotype_data = pd.read_csv(phenotype_file)
IDP = pd.read_csv(IDP_file)
pca  = pd.read_csv(pca_file)


pca_cols = [c for c in pca.columns if c.startswith('f.22009.')]
pca_cols = sorted(pca_cols, key=lambda nm: int(nm.split('.')[-1]))[:10]

# Merging so phenotype subject have PCA data
phenotype_data = pd.merge(phenotype_data, pca[['f.eid'] + pca_cols], on='f.eid',how='inner')
# Then merge the resulting dataframe with the IDP data
phenotype_data = pd.merge(phenotype_data, IDP[['f.eid', 'head_volume','Volume_ventricul_cerebrospinal_fluid', 'Brain_volume_gray_white_matter']], on='f.eid', how='inner')

phenotype_data[pca_cols]=phenotype_data[pca_cols].apply(pd.to_numeric, errors='coerce')
# 1) Create a single Education_score that’s the max of the two
phenotype_data['Education_score'] = phenotype_data[['Education_score_0','Education_score_2']].max(axis=1)
phenotype_data.drop(columns=['Education_score_0','Education_score_2'], inplace=True)


# create dummy code for assessment centre
phenotype_data['Assessment_centre'] = phenotype_data['Assessment_centre'].astype('category')
dummies = pd.get_dummies(
    phenotype_data['Assessment_centre'],
    prefix='Assessment_centre',
    drop_first=True
)
phenotype_data = pd.concat([phenotype_data, dummies], axis=1)


dummy_cols   = [c for c in phenotype_data.columns if c.startswith('Assessment_centre_')]

base_covs = ['True_Label', 'Age', 'Sex','head_volume','Volume_ventricul_cerebrospinal_fluid', 'Brain_volume_gray_white_matter', 'Education_score'] + dummy_cols + pca_cols

# Prepare data for merging
dependent_vars = [
    'Fluid_intel_score', # Fluid intelligence /reasoning  score
    'log_reaction_time', # Reaction time score
    'Numeric_memory', # Maximum digits remembered correctly
    'Numeric_memory_online', # Maximum digits remembered correctly online
    'Matrix_pattern_puzzles',
    'Symbol_digit_matches',
    'Tower_rearranging'
    ]

# Independent variables
# Full model: includes the probability predictors
independent_vars_full = ['Prob_Class_0', 'Prob_Class_2'] + base_covs 

# Reduced model: excludes the probability predictors
independent_vars_reduced =  base_covs.copy()

# Prepare a list to store results
results = []


# Loop through each task ID and dependent variable
for snp_id in snp_data["SNP_rsid"].unique():
    print(f"Processing SNP: {snp_id}")
    # Filter SNP data for the current task and merge with phenotype data
    filtered_snp_data = snp_data[snp_data["SNP_rsid"] == snp_id]
    merged_data = pd.merge(filtered_snp_data, phenotype_data, left_on='Sample_ID', right_on='f.eid', how='inner')
    merged_data[dependent_vars] = merged_data[dependent_vars].replace(-1, np.nan)

    for dep_var in dependent_vars:
        # Define columns to check for NaNs (both independent and the dependent variable)
        relevant_columns = independent_vars_full + [dep_var]
        cleaned_data = merged_data[relevant_columns].dropna()
        
        # Skip this loop if no data is available 
        if cleaned_data.empty:
            continue
        
        n = len(cleaned_data)
        print(cleaned_data.columns.tolist())
        
        # Prepare dependent variable and predictors for full model
        y = cleaned_data[dep_var]
        X_full = cleaned_data[independent_vars_full]
        X_full_const = sm.add_constant(X_full).apply(pd.to_numeric, errors='coerce').astype(float)
        model_full = sm.OLS(y, X_full_const).fit()
        # Prepare predictors for reduced model
        X_reduced = cleaned_data[independent_vars_reduced]
        X_reduced_const = sm.add_constant(X_reduced).apply(pd.to_numeric, errors='coerce').astype(float)
        model_reduced = sm.OLS(y, X_reduced_const).fit()
        

        # Perform the F-test to compare the full and reduced models
        F_stat, p_value, df_diff = model_full.compare_f_test(model_reduced)
        
        # --- NEW: compute in-sample predictions & errors ---
        y_pred_full    = model_full.predict(X_full_const)
        y_pred_reduced = model_reduced.predict(X_reduced_const)

        rmse_full    = np.sqrt(mean_squared_error(y, y_pred_full))
        mae_full     = mean_absolute_error(y, y_pred_full)
        rmse_reduced = np.sqrt(mean_squared_error(y, y_pred_reduced))
        mae_reduced  = mean_absolute_error(y, y_pred_reduced)
        

        # Store results in the list
        results.append({
            'SNP': snp_id,
            'Dependent_Var': dep_var,
            "n_samples":       n,
            'F_stat': F_stat,
            'p_value': p_value,
            'df_diff': df_diff,
            'R2_full_raw': model_full.rsquared,
            'R2_reduced_raw': model_reduced.rsquared,
            'R2_full_adj': model_full.rsquared_adj,    
            'R2_reduced_adj': model_reduced.rsquared_adj, 
            'Beta_full': model_full.params.to_dict(),
            'Beta_reduced':  model_reduced.params.to_dict(),
            'SE_full':       model_full.bse.to_dict(),
            'SE_reduced':    model_reduced.bse.to_dict(),
            'RMSE_full':     rmse_full,
            'MAE_full':      mae_full,
            'RMSE_reduced':  rmse_reduced,
            'MAE_reduced':   mae_reduced
        })
        

# Convert the list of results into a DataFrame for easy inspection
results_df = pd.DataFrame(results)
# save the results to a csv file
results_df.to_csv('/path/to/results/cognitive_scores_associations_results.csv')
import pandas as pd
import numpy as np
import os
from scipy.stats import ks_2samp

# Define paths
governance_folder = r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\governance"
ks_path = os.path.join(governance_folder, "ks_test_results.csv")
psi_path = os.path.join(governance_folder, "psi_results.csv")

# Load structured data
old_structured = pd.read_pickle(r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\api\processed_data\stored_data\processed_structured.pkl")
new_structured = pd.read_pickle(r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\api\processed_data\new_data\processed_structured.pkl")

# Define PSI function
def psi(expected, actual, bins=10):
    expected_perc = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_perc = np.histogram(actual, bins=bins)[0] / len(actual)
    psi_values = (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)
    return np.sum(np.nan_to_num(psi_values))

ks_results = {}
psi_results = {}

for col in old_structured.columns:
    ks_stat, ks_pvalue = ks_2samp(old_structured[col], new_structured[col])
    psi_value = psi(old_structured[col], new_structured[col])
    
    ks_results[col] = {"KS_Stat": ks_stat, "p-value": ks_pvalue}
    psi_results[col] = {"PSI_Score": psi_value}

# Save KS & PSI results
pd.DataFrame.from_dict(ks_results, orient="index").to_csv(ks_path)
pd.DataFrame.from_dict(psi_results, orient="index").to_csv(psi_path)

print("✅ Data drift detection completed. Results saved.")

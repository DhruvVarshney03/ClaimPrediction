import os
import json
import pandas as pd

# Define paths
governance_folder = r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\governance"
comparison_path = os.path.join(governance_folder, "performance_comparison.csv")
ks_path = os.path.join(governance_folder, "ks_test_results.csv")
psi_path = os.path.join(governance_folder, "psi_results.csv")
report_path = os.path.join(governance_folder, "governance_report.json")

# Load results
comparison_df = pd.read_csv(comparison_path)
ks_results = pd.read_csv(ks_path, index_col=0).to_dict(orient="index")
psi_results = pd.read_csv(psi_path, index_col=0).to_dict(orient="index")

# Compute summary stats
classification_accuracy_old = (comparison_df["True_Condition"] == comparison_df["Old_Pred_Condition"]).mean()
classification_accuracy_new = (comparison_df["True_Condition"] == comparison_df["New_Pred_Condition"]).mean()
regression_error_old = (comparison_df["True_Amount"] - comparison_df["Old_Pred_Amount"]).abs().mean()
regression_error_new = (comparison_df["True_Amount"] - comparison_df["New_Pred_Amount"]).abs().mean()

# Create report
report = {
    "Performance Comparison": {
        "Classification Accuracy": {
            "Old Model": classification_accuracy_old,
            "New Model": classification_accuracy_new,
        },
        "Regression Error (MAE)": {
            "Old Model": regression_error_old,
            "New Model": regression_error_new,
        },
    },
    "Data Drift": {
        "KS Test": ks_results,
        "PSI Scores": psi_results,
    }
}

# Save JSON report
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print(f"✅ Governance report generated and saved at {report_path}")

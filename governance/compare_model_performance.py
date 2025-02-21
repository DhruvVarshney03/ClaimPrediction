import pandas as pd
import numpy as np
import os

# Define paths
governance_folder = r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\governance"
old_pred_path = os.path.join(governance_folder, "old_model_performance.pkl")
new_pred_path = os.path.join(governance_folder, "model_performance.pkl")
comparison_path = os.path.join(governance_folder, "performance_comparison.csv")

# Load old and new predictions
old_preds = pd.read_pickle(old_pred_path)
new_preds = pd.read_pickle(new_pred_path)

# Merge for comparison
comparison_df = pd.DataFrame({
    "True_Condition": old_preds["True_Condition"],
    "Old_Pred_Condition": old_preds["Predicted_Condition"],
    "New_Pred_Condition": new_preds["Predicted_Condition"],
    "True_Amount": old_preds["True_Amount"],
    "Old_Pred_Amount": old_preds["Predicted_Amount"],
    "New_Pred_Amount": new_preds["Predicted_Amount"],
})
comparison_df.to_csv(comparison_path, index=False)

# Compute accuracy & error
classification_accuracy_old = (comparison_df["True_Condition"] == comparison_df["Old_Pred_Condition"]).mean()
classification_accuracy_new = (comparison_df["True_Condition"] == comparison_df["New_Pred_Condition"]).mean()
regression_error_old = np.mean(np.abs(comparison_df["True_Amount"] - comparison_df["Old_Pred_Amount"]))
regression_error_new = np.mean(np.abs(comparison_df["True_Amount"] - comparison_df["New_Pred_Amount"]))

print("✅ Performance comparison completed.")
print(f"🔹 Old Model Accuracy: {classification_accuracy_old:.4f}")
print(f"🔹 New Model Accuracy: {classification_accuracy_new:.4f}")
print(f"🔹 Old Model MAE: {regression_error_old:.4f}")
print(f"🔹 New Model MAE: {regression_error_new:.4f}")

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# FIGURE 7: FINAL TEST CONFUSION MATRIX
# =============================================================================
# Purpose of this script
# ----------------------
# This script creates Figure 7 for the report. The figure visualises the final
# test confusion matrix for the selected model.
#
# Why this figure matters
# -----------------------
# The confusion matrix helps the reader see the balance between:
# - true negatives,
# - false positives,
# - false negatives,
# - and true positives
#
# in a direct and intuitive way. This is especially useful in an imbalanced
# classification study, where the practical meaning of errors matters as much as
# aggregate summary metrics.
# =============================================================================


# -----------------------------------------------------------------------------
# Step 1: Define project folder and possible input file
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

possible_input_files = [
    project_folder / "step12_final_test_confusion_matrix.csv",
    project_folder / "step25_distilbert_final_test_confusion_matrix.csv"
]

figure_output = project_folder / "figure7_final_test_confusion_matrix.png"


# -----------------------------------------------------------------------------
# Step 2: Try to load the final selected model confusion matrix
# -----------------------------------------------------------------------------
# The preferred file is the final selected Method 1 confusion matrix.
# If a file is not found or cannot be interpreted directly, fall back to the
# confirmed counts already reported in the project text.
# -----------------------------------------------------------------------------
input_file = None
for file_path in possible_input_files:
    if file_path.exists():
        input_file = file_path
        break

matrix = None

if input_file is not None:
    try:
        df = pd.read_csv(input_file, low_memory=False)
        print(f"Using confusion-matrix file: {input_file.name}")

        # Try common formats
        if df.shape == (2, 2):
            matrix = df.to_numpy()

        elif set(df.columns.str.lower()) >= {"tn", "fp", "fn", "tp"}:
            row = df.iloc[0]
            matrix = np.array([
                [row["tn"], row["fp"]],
                [row["fn"], row["tp"]]
            ])

    except Exception as e:
        print("Could not parse the confusion-matrix file directly.")
        print(f"Reason: {e}")

if matrix is None:
    print("Falling back to the confirmed final test confusion-matrix values.")
    matrix = np.array([
        [1152, 128],
        [102, 196]
    ])


# -----------------------------------------------------------------------------
# Step 3: Create the confusion matrix plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(matrix, cmap="Blues")

# Axis labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted non-unstable", "Predicted unstable"])
ax.set_yticklabels(["Actual non-unstable", "Actual unstable"])

ax.set_title("Final test confusion matrix")

# Add the values inside the cells
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        # Make the darkest cell use white text for readability.
        text_color = "white" if (i == 0 and j == 0) else "black"

        ax.text(
            j, i, f"{matrix[i, j]}",
            ha="center",
            va="center",
            fontsize=12,
            color=text_color
        )

# Add a color bar
fig.colorbar(im, ax=ax)

plt.tight_layout()


# -----------------------------------------------------------------------------
# Step 4: Save the figure
# -----------------------------------------------------------------------------
plt.savefig(figure_output, dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------------------------------------------------------
# Step 5: Print confirmation
# -----------------------------------------------------------------------------
print("=" * 90)
print("FIGURE 7 CREATED SUCCESSFULLY")
print("=" * 90)
print(f"Figure file created: {figure_output.name}")
print("=" * 90)
print("This PNG file is ready to be inserted into the report as Figure 7.")
print("=" * 90)
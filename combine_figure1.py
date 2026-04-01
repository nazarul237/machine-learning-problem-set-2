from PIL import Image
from pathlib import Path

# =============================================================================
# COMBINE TWO EDA FIGURES INTO ONE REPORT-READY IMAGE
# =============================================================================
# Purpose of this script:
# This script combines two existing PNG figures into one single image so that
# they can be inserted into the report as one figure.
#
# In this project, the aim is to present:
# 1. the number of ASRS reports by year, and
# 2. the unstable-approach trend by year
#
# side by side in a single visual.
#
# This is useful for the report because it allows the reader to understand both
# the temporal distribution of the corpus and the year-wise unstable trend in
# one place, without having to look at two separate images.
# =============================================================================


# -----------------------------------------------------------------------------
# Step 1: Define the file names of the two existing figures
# -----------------------------------------------------------------------------
# IMPORTANT:
# Please replace the second file name below with the exact file name from your
# folder if it is slightly different.
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

file_left = project_folder / "step6_plot_rows_by_year.png"
file_right = project_folder / "step6_plot_unstable_rate_by_year.png"

# -----------------------------------------------------------------------------
# Step 2: Check that both files exist before trying to open them
# -----------------------------------------------------------------------------
# This avoids a confusing error later in the script if one of the image names
# has been typed incorrectly.
# -----------------------------------------------------------------------------
if not file_left.exists():
    raise FileNotFoundError(f"Left image not found: {file_left.name}")

if not file_right.exists():
    raise FileNotFoundError(f"Right image not found: {file_right.name}")


# -----------------------------------------------------------------------------
# Step 3: Open both images
# -----------------------------------------------------------------------------
# PIL (Python Imaging Library) allows images to be loaded and edited in a simple
# and reliable way.
# -----------------------------------------------------------------------------
img_left = Image.open(file_left)
img_right = Image.open(file_right)


# -----------------------------------------------------------------------------
# Step 4: Make both images the same height
# -----------------------------------------------------------------------------
# The two figures may not have exactly the same dimensions. To create a clean
# side-by-side figure, both images should share a common height.
#
# The script uses the larger of the two heights as the target height, then
# rescales each image proportionally so that the original shape is preserved.
# -----------------------------------------------------------------------------
target_height = max(img_left.height, img_right.height)

new_left_width = int(img_left.width * target_height / img_left.height)
new_right_width = int(img_right.width * target_height / img_right.height)

img_left_resized = img_left.resize((new_left_width, target_height))
img_right_resized = img_right.resize((new_right_width, target_height))


# -----------------------------------------------------------------------------
# Step 5: Create a blank white canvas for the combined figure
# -----------------------------------------------------------------------------
# The total width of the new image is the sum of the two resized widths.
# The height is the shared target height.
# -----------------------------------------------------------------------------
combined_width = img_left_resized.width + img_right_resized.width
combined_height = target_height

combined_figure = Image.new("RGB", (combined_width, combined_height), "white")


# -----------------------------------------------------------------------------
# Step 6: Paste the two resized images onto the blank canvas
# -----------------------------------------------------------------------------
# The first figure is placed on the left starting at position (0, 0).
# The second figure is placed immediately to the right of the first figure.
# -----------------------------------------------------------------------------
combined_figure.paste(img_left_resized, (0, 0))
combined_figure.paste(img_right_resized, (img_left_resized.width, 0))


# -----------------------------------------------------------------------------
# Step 7: Save the final combined figure
# -----------------------------------------------------------------------------
# This creates a single PNG file that can be inserted into the report.
# -----------------------------------------------------------------------------
output_file = project_folder / "figure1_combined.png"
combined_figure.save(output_file)


# -----------------------------------------------------------------------------
# Step 8: Print confirmation
# -----------------------------------------------------------------------------
print("=" * 80)
print("Figure combination complete.")
print(f"New file created: {output_file.name}")
print("This file is ready to be used as Figure 1 in the report.")
print("=" * 80)
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# STEP 6: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
# Purpose of this step:
# At this stage, the modelling-ready base dataset has already been prepared in
# Step 5. However, before training any machine learning model, it is good
# research practice to explore the data carefully.
#
# This is especially important in the present assessment for two reasons:
# 1. The professor's earlier feedback highlighted the need for more basic
#    exploratory analysis before full modelling.
# 2. Our dataset contains both text and contextual variables, so we need to
#    understand class balance, missing values, text length, time coverage, and
#    environmental-field completeness before choosing the modelling strategy.
#
# In this step, we are NOT cleaning further and we are NOT training models.
# We are only describing and visualising the base dataset.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
# Using Path(__file__).resolve().parent makes the script work relative to its
# own folder, which is safer than depending on wherever the terminal happens to
# be opened from.
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent
input_file = project_folder / "step5_base_dataset.csv"

# Output files from this EDA step
class_balance_file = project_folder / "step6_class_balance_summary.csv"
missing_summary_file = project_folder / "step6_missing_summary.csv"
text_length_summary_file = project_folder / "step6_text_length_summary.csv"
year_summary_file = project_folder / "step6_year_summary.csv"
flight_conditions_file = project_folder / "step6_flight_conditions_summary.csv"
weather_visibility_file = project_folder / "step6_weather_visibility_summary.csv"
light_summary_file = project_folder / "step6_light_summary.csv"
ceiling_summary_file = project_folder / "step6_ceiling_summary.csv"

# Plot file names
plot_class_balance = project_folder / "step6_plot_class_balance.png"
plot_year_counts = project_folder / "step6_plot_rows_by_year.png"
plot_unstable_rate = project_folder / "step6_plot_unstable_rate_by_year.png"
plot_text_length = project_folder / "step6_plot_text_length_by_label.png"


# -----------------------------------------------------------------------------
# Load the Step 5 base dataset
# -----------------------------------------------------------------------------
# This dataset should already contain:
# - the modelling scope after cleaning
# - the binary unstable-approach label
# - the main text field called 'text_main'
# - the contextual fields that may be used later
# -----------------------------------------------------------------------------
df = pd.read_csv(input_file, low_memory=False)

print("=" * 90)
print("STEP 6: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 90)
print(f"Rows loaded from Step 5 base dataset: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")
print()


# -----------------------------------------------------------------------------
# Section 6A: Overall class balance
# -----------------------------------------------------------------------------
# Because this is a classification project, it is essential to understand how
# many positive and negative cases we have before modelling.
#
# In this project:
# - label_unstable = 1 means unstable approach
# - label_unstable = 0 means non-unstable case
#
# This class balance matters because strong imbalance can distort model
# evaluation and make accuracy look deceptively strong.
# -----------------------------------------------------------------------------
label_counts = df["label_unstable"].value_counts(dropna=False).sort_index()

class_balance_df = pd.DataFrame({
    "label_unstable": label_counts.index,
    "count": label_counts.values
})

class_balance_df["class_name"] = class_balance_df["label_unstable"].map({
    0: "non_unstable",
    1: "unstable"
})

class_balance_df["percent"] = (class_balance_df["count"] / len(df) * 100).round(2)
class_balance_df.to_csv(class_balance_file, index=False)

print("CLASS BALANCE SUMMARY")
print("-" * 90)
print(class_balance_df.to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Section 6B: Missing value summary
# -----------------------------------------------------------------------------
# Before building a context-enhanced model later, we need to understand how much
# missingness exists in the important variables. This is particularly important
# for environmental fields, because missing context values may affect how much
# benefit those variables can add beyond the narrative text.
# -----------------------------------------------------------------------------
columns_to_check = [
    "Time | Date",
    "incident_year",
    "Aircraft 1 | Flight Phase",
    "Report 1 | Narrative",
    "Report 1 | Synopsis",
    "text_main",
    "Environment | Flight Conditions",
    "Environment | Weather Elements / Visibility",
    "Environment | Light",
    "Environment | Ceiling",
    "Aircraft 1 | Make Model Name",
    "Aircraft 1 | Flight Plan",
    "Aircraft 1 | Mission",
    "Aircraft 1 | Aircraft Operator",
    "Aircraft 1 | Operating Under FAR Part",
    "Aircraft 1 | ATC / Advisory",
    "label_unstable",
]

missing_summary = pd.DataFrame({
    "column": columns_to_check,
    "missing_count": [df[col].isna().sum() for col in columns_to_check],
})

missing_summary["missing_percent"] = (
    missing_summary["missing_count"] / len(df) * 100
).round(2)

missing_summary.to_csv(missing_summary_file, index=False)

print("MISSING VALUE SUMMARY")
print("-" * 90)
print(missing_summary.to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Section 6C: Text length analysis
# -----------------------------------------------------------------------------
# Since the core of this project is text classification, it is useful to know
# how long the narratives are. Very short narratives may contain less usable
# information, while very long narratives may reflect more detailed descriptions
# of operational events.
#
# We calculate two simple length measures:
# - character length
# - word count
#
# We then compare them overall and by class.
# -----------------------------------------------------------------------------
df["text_main"] = df["text_main"].fillna("").astype(str)
df["text_char_length"] = df["text_main"].str.len()
df["text_word_count"] = df["text_main"].str.split().str.len()

overall_text_summary = pd.DataFrame({
    "metric": [
        "char_length_mean", "char_length_median", "char_length_min", "char_length_max",
        "word_count_mean", "word_count_median", "word_count_min", "word_count_max"
    ],
    "value": [
        df["text_char_length"].mean(),
        df["text_char_length"].median(),
        df["text_char_length"].min(),
        df["text_char_length"].max(),
        df["text_word_count"].mean(),
        df["text_word_count"].median(),
        df["text_word_count"].min(),
        df["text_word_count"].max(),
    ]
})

by_label_text_summary = (
    df.groupby("label_unstable")[["text_char_length", "text_word_count"]]
      .agg(["mean", "median", "min", "max"])
)

overall_text_summary.to_csv(text_length_summary_file, index=False)

print("TEXT LENGTH SUMMARY (OVERALL)")
print("-" * 90)
print(overall_text_summary.to_string(index=False))
print()

print("TEXT LENGTH SUMMARY BY LABEL")
print("-" * 90)
print(by_label_text_summary)
print()


# -----------------------------------------------------------------------------
# Section 6D: Year distribution and unstable rate by year
# -----------------------------------------------------------------------------
# Because our later evaluation will use a time-based design, it is important to
# understand how the data are distributed over time.
#
# We therefore calculate:
# - the number of rows per year
# - the number of unstable cases per year
# - the unstable-case percentage per year
# -----------------------------------------------------------------------------
year_summary = (
    df.groupby("incident_year")
      .agg(
          total_rows=("label_unstable", "size"),
          unstable_cases=("label_unstable", "sum")
      )
      .reset_index()
)

year_summary["unstable_rate_percent"] = (
    year_summary["unstable_cases"] / year_summary["total_rows"] * 100
).round(2)

year_summary.to_csv(year_summary_file, index=False)

print("YEAR SUMMARY")
print("-" * 90)
print(year_summary.to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Section 6E: Category summaries for key context variables
# -----------------------------------------------------------------------------
# These summaries help us understand which environmental conditions are most
# common in the dataset. Since some of these columns have missing values, we
# explicitly convert missing values into the text label 'Missing' so that the
# absence of information is visible in the summaries.
# -----------------------------------------------------------------------------
def save_category_summary(dataframe, column_name, output_file, top_n=15):
    temp = dataframe[column_name].fillna("Missing").astype(str).str.strip()
    summary = temp.value_counts(dropna=False).reset_index()
    summary.columns = [column_name, "count"]
    summary["percent"] = (summary["count"] / len(dataframe) * 100).round(2)
    summary.head(top_n).to_csv(output_file, index=False)
    return summary.head(top_n)

flight_conditions_summary = save_category_summary(
    df, "Environment | Flight Conditions", flight_conditions_file
)

weather_visibility_summary = save_category_summary(
    df, "Environment | Weather Elements / Visibility", weather_visibility_file
)

light_summary = save_category_summary(
    df, "Environment | Light", light_summary_file
)

ceiling_summary = save_category_summary(
    df, "Environment | Ceiling", ceiling_summary_file
)

print("TOP FLIGHT CONDITIONS")
print("-" * 90)
print(flight_conditions_summary.to_string(index=False))
print()

print("TOP WEATHER / VISIBILITY CATEGORIES")
print("-" * 90)
print(weather_visibility_summary.to_string(index=False))
print()

print("TOP LIGHT CATEGORIES")
print("-" * 90)
print(light_summary.to_string(index=False))
print()

print("TOP CEILING CATEGORIES")
print("-" * 90)
print(ceiling_summary.to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Section 6F: Create simple plots
# -----------------------------------------------------------------------------
# These plots are intentionally simple and readable. The purpose is not to make
# a flashy dashboard, but to produce clear visual evidence for the report and
# for your own understanding of the dataset before modelling.
# -----------------------------------------------------------------------------

# Plot 1: Class balance
plt.figure(figsize=(7, 5))
plt.bar(class_balance_df["class_name"], class_balance_df["count"])
plt.title("Class Balance: Unstable vs Non-Unstable Cases")
plt.xlabel("Class")
plt.ylabel("Number of reports")
plt.tight_layout()
plt.savefig(plot_class_balance, dpi=300)
plt.close()

# Plot 2: Number of rows by year
plt.figure(figsize=(8, 5))
plt.bar(year_summary["incident_year"].astype(str), year_summary["total_rows"])
plt.title("Number of Reports by Year")
plt.xlabel("Incident year")
plt.ylabel("Number of reports")
plt.tight_layout()
plt.savefig(plot_year_counts, dpi=300)
plt.close()

# Plot 3: Unstable rate by year
plt.figure(figsize=(8, 5))
plt.plot(
    year_summary["incident_year"].astype(str),
    year_summary["unstable_rate_percent"],
    marker="o"
)
plt.title("Unstable-Approach Rate by Year")
plt.xlabel("Incident year")
plt.ylabel("Unstable rate (%)")
plt.tight_layout()
plt.savefig(plot_unstable_rate, dpi=300)
plt.close()

# Plot 4: Boxplot of text word count by label
plt.figure(figsize=(7, 5))
df.boxplot(column="text_word_count", by="label_unstable")
plt.title("Narrative Word Count by Label")
plt.suptitle("")  # removes default pandas subtitle
plt.xlabel("label_unstable (0 = non-unstable, 1 = unstable)")
plt.ylabel("Word count")
plt.tight_layout()
plt.savefig(plot_text_length, dpi=300)
plt.close()


# -----------------------------------------------------------------------------
# Final printout so that the user knows which files were created
# -----------------------------------------------------------------------------
print("=" * 90)
print("FILES CREATED IN STEP 6")
print("=" * 90)
print(class_balance_file.name)
print(missing_summary_file.name)
print(text_length_summary_file.name)
print(year_summary_file.name)
print(flight_conditions_file.name)
print(weather_visibility_file.name)
print(light_summary_file.name)
print(ceiling_summary_file.name)
print(plot_class_balance.name)
print(plot_year_counts.name)
print(plot_unstable_rate.name)
print(plot_text_length.name)
print()
print("STEP 6 FINISHED")
print("EDA has been completed.")
print("No modelling has been started yet.")
print("=" * 90)
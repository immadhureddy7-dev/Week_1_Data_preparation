import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# 1. CREATE VISUALIZATION FOLDER
# ============================================================

os.makedirs("../visualizations", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

# CHANGE THIS FILE NAME TO THE CSV DOWNLOADED FROM KAGGLE

df = pd.read_csv("../data/netflix_titles.csv", encoding="latin1")


print("\n========================================")
print("DATASET LOADED SUCCESSFULLY")
print("========================================")


# ============================================================
# 3. INITIAL DATA INSPECTION
# ============================================================

print("\nFIRST 5 ROWS")
print(df.head())


print("\nDATASET SHAPE")
print(df.shape)


print("\nCOLUMN NAMES")
print(df.columns.tolist())


print("\nDATA TYPES")
print(df.dtypes)


print("\nDATASET INFORMATION")
df.info()


print("\nSTATISTICAL SUMMARY")
print(df.describe(include="all"))


# ============================================================
# 4. MISSING VALUE ANALYSIS
# ============================================================

print("\nMISSING VALUES")

missing_values = df.isnull().sum()

print(missing_values)


# Visualization

plt.figure(figsize=(10, 5))

missing_values[missing_values > 0].plot(
    kind="bar"
)

plt.title("Missing Values in Dataset")
plt.xlabel("Columns")
plt.ylabel("Number of Missing Values")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "../visualizations/01_missing_values.png",
    dpi=300
)

plt.show()


# ============================================================
# 5. DUPLICATE ANALYSIS
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nNUMBER OF DUPLICATE ROWS:")
print(duplicate_count)


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("\nDATASET SHAPE AFTER DUPLICATE REMOVAL:")
print(df.shape)


# ============================================================
# 7. DATA TYPE CHECK
# ============================================================

print("\nDATA TYPES BEFORE CLEANING:")
print(df.dtypes)


# ============================================================
# 8. NUMERICAL COLUMN IDENTIFICATION
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

print("\nNUMERICAL COLUMNS:")
print(numeric_columns.tolist())


# ============================================================
# 9. CATEGORICAL COLUMN IDENTIFICATION
# ============================================================

categorical_columns = df.select_dtypes(
    include="object"
).columns

print("\nCATEGORICAL COLUMNS:")
print(categorical_columns.tolist())


# ============================================================
# 10. HANDLE MISSING NUMERICAL VALUES
# ============================================================

for column in numeric_columns:

    if df[column].isnull().sum() > 0:

        median_value = df[column].median()

        df[column] = df[column].fillna(
            median_value
        )


# ============================================================
# 11. HANDLE MISSING CATEGORICAL VALUES
# ============================================================

for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        mode_value = df[column].mode()[0]

        df[column] = df[column].fillna(
            mode_value
        )


# ============================================================
# 12. CHECK MISSING VALUES AFTER CLEANING
# ============================================================

print("\nMISSING VALUES AFTER CLEANING:")

print(df.isnull().sum())


# ============================================================
# 13. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    "../data/cleaned_dataset.csv",
    index=False
)

print("\nCLEANED DATASET SAVED.")


# ============================================================
# 14. UPDATED STATISTICAL SUMMARY
# ============================================================

print("\nFINAL STATISTICAL SUMMARY")

print(
    df.describe()
)


# ============================================================
# 15. NUMERICAL CORRELATION
# ============================================================

if len(numeric_columns) >= 2:

    correlation = df[numeric_columns].corr()

    print("\nCORRELATION MATRIX")

    print(correlation)


    plt.figure(
        figsize=(10, 7)
    )

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        "../visualizations/02_correlation_heatmap.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 16. NUMERICAL DISTRIBUTIONS
# ============================================================

for column in numeric_columns:

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(
        "Distribution of " + column
    )

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.tight_layout()

    safe_name = column.replace(
        " ",
        "_"
    )

    plt.savefig(
        "../visualizations/"
        + safe_name
        + "_distribution.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 17. BOX PLOTS FOR OUTLIER ANALYSIS
# ============================================================

for column in numeric_columns:

    plt.figure(
        figsize=(8, 5)
    )

    sns.boxplot(data =df,y="release_year")

    plt.title(
        "Box Plot - " + column
    )

    plt.ylabel(column)

    plt.tight_layout()

    safe_name = column.replace(
        " ",
        "_"
    )

    plt.savefig(
        "../visualizations/"
        + safe_name
        + "_boxplot.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 18. CATEGORICAL ANALYSIS
# ============================================================

for column in categorical_columns:

    unique_count = df[column].nunique()

    if unique_count <= 15:

        print(
            "\nVALUE COUNTS FOR:",
            column
        )

        print(
            df[column].value_counts()
        )

        plt.figure(
            figsize=(10, 5)
        )

        sns.countplot(
            data=df,
            x=column
        )

        plt.title(
            "Distribution of " + column
        )

        plt.xlabel(column)

        plt.ylabel("Count")

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        safe_name = column.replace(
            " ",
            "_"
        )

        plt.savefig(
            "../visualizations/"
            + safe_name
            + "_countplot.png",
            dpi=300
        )

        plt.show()


# ============================================================
# 19. FINAL DATA QUALITY CHECK
# ============================================================

print("\n========================================")
print("FINAL DATA QUALITY CHECK")
print("========================================")

print(
    "Rows:",
    df.shape[0]
)

print(
    "Columns:",
    df.shape[1]
)

print(
    "Missing Values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate Rows:",
    df.duplicated().sum()
)


# ============================================================
# 20. FINAL DATASET
# ============================================================

print("\nFINAL CLEAN DATASET")

print(df.head(10))


print("\n========================================")
print("WEEK 1 EDA COMPLETED SUCCESSFULLY")
print("========================================")
from pathlib import Path

import pandas as pd

RAW_DATA_PATH = (

    Path(__file__).parents[1]
    / "data"
    / "raw"
    / "Timely_and_Effective_Care-Hospital.csv"
)

df = pd.read_csv(

    RAW_DATA_PATH,

    dtype={"Facility ID": "string"},

    low_memory=False

)

print("\nConditions:")
print(df["Condition"].value_counts())

print("\nUnique Measure IDs:")
print(df["Measure ID"].nunique())

print("\nMeasures:")
measures = (
    df[["Condition", "Measure ID", "Measure Name"]]
    .drop_duplicates()
    .sort_values(["Condition", "Measure ID"])
)

print(measures.to_string(index = False))

ed = df[
    df["Condition"]
    .str.contains("Emergency", case=False, na=False)
]
print("\nEmergency Department Measures:")
print(
    ed[["Measure ID", "Measure Name"]]
    .drop_duplicates()
    .sort_values("Measure ID")
    .to_string(index = False)
)

print("\nEmergency Department Score Examples:")
print(
    ed[
        ["Facility ID", "Facility Name",
         "Measure ID", "Measure Name", "Score"]
    ].head(30).to_string(index = False)
)

ed = df.loc[
    df["Condition"] == "Emergency Department",
].copy()
print(ed["Measure ID"].value_counts())


volume = ed.loc[
    ed["Measure ID"] == "EDV",
    ["Facility ID", "Score"]
].copy()

volume = volume.rename(
    columns ={"Score": "ed_volume"}
)

numeric_ids = [
    "OP_18a",
    "OP_18b",
    "OP_18c",
    "OP_18d",
    "OP_22",
    "OP_23"
]
numeric = ed.loc[
    ed["Measure ID"].isin(numeric_ids)
].copy()
numeric["Score"] = pd.to_numeric(
    numeric["Score"], errors="coerce"
)

print("nED Volume:")
print(volume["ed_volume"].value_counts(dropna = False))

print("\nNumeric Score Summary:")
print(numeric.groupby("Measure ID")["Score"].describe())

#pivoting the data

wide = numeric.pivot(
    index="Facility ID",
    columns="Measure ID",
    values="Score"
).reset_index()

wide = wide.merge(
    volume,
    on="Facility ID",
    how="left"
)

hospital_info = (
    ed[
        [
            "Facility ID",
            "Facility Name",
            "City/Town",
            "State",
            "ZIP Code",
            "County/Parish"
        ]
    ]
    .drop_duplicates(subset="Facility ID")
)
wide = hospital_info.merge(
    wide,
    on="Facility ID",
    how="left"
)
print(wide.head())
print(wide.shape)
print(wide.dtypes)

wide["ed_volume"] = wide["ed_volume"].replace(
    "Not Available", pd.NA
)
volume_order = [
    "low",
    "medium",
    "high",
    "very high"
]
wide["ed_volume"] = pd.Categorical(wide["ed_volume"], categories=volume_order,
                                   ordered=True)
print("\nProcessed Shape:")
print(wide.shape)

print("\nMissing Proportion:")
print(
    wide.isna()
    .mean()
    .sort_values(ascending=False)
)
print("\nED Volume Distribution:")
print(wide["ed_volume"].value_counts(dropna=False))

print("\nCore Measures:")
print(
    wide[
        ["OP_18a", "OP_18b", "OP_22"]
    ].describe()
)
print("\nCore Correlations:")
print(
    wide[
        ["OP_18a", 'OP_18b', "OP_22"]
    ].corr()
)

PROCESSED_DATA_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "processed"
    / "ed_hospital_measures.csv"
)
wide.to_csv(
    PROCESSED_DATA_PATH,
    index = False
)

GENERAL_DATA_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "raw"
    / "Hospital_General_Information.csv"
)
general = pd.read_csv(
    GENERAL_DATA_PATH,
    dtype={
        "Facility ID": "string",
        "ZIP Code": "string"
    }, low_memory=False
)

print("Shape:", general.shape)

print("\nColumns:")
print(general.columns.tolist())

print("\nFirst 5:")
print(general.head())

print("\nData Types:")
print(general.dtypes)

print("\nMissing Values:")
print(general.isna().sum())

print("\nUnique Facility IDs:")
print(general["Facility ID"].nunique())

print("\nDuplicate Facility IDs:")
print(general["Facility ID"].duplicated().sum())

model2_features = [
    "Hospital Type",
    "Hospital Ownership",
    "Emergency Services"
]

for col in model2_features:
    print(f"\n{col}:")
    print(general[col].value_counts())

print(
    general[
        ["Hospital Type",
        "Hospital Ownership",
        "Emergency Services"]
    ].nunique()
)

""" merge """

general_features = general[
    [
        "Facility ID",
        "Hospital Type",
        "Hospital Ownership",
        "Emergency Services"
    ]
].copy()

model_data = wide.merge(
    general_features,
    on="Facility ID",
    how="left",
    validate="one_to_one"
)
#sanity check
print("\nBefore Merge:", wide.shape)
print("\nAfter Merge:", model_data.shape)
matched = model_data["Hospital Type"].notna().sum()

print(
    f"\nMatched Hospitals: {matched}/{len(model_data)}"
    f"({matched / len(model_data):.2%})"
)

print("\nMissing General Information:")
print(
    model_data[
        [
            "Hospital Type",
            "Hospital Ownership",
            "Emergency Services"
        ]
    ].isna().sum()
)
print("\nHospital Type in ED Sample:")
print(model_data["Hospital Type"].value_counts(dropna=False))

print("\nOwnership in ED Sample:")
print(model_data["Hospital Ownership"].value_counts(dropna=False))

print("\nEmergency Services in ED Sample:")
print(model_data["Emergency Services"].value_counts(dropna=False))

MODEL_DATA_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "processed"
    / "ed_hospital_modeling.csv"
)

model_data.to_csv(
    MODEL_DATA_PATH,
    index=False
)
print(f"\nSaved processed data to {MODEL_DATA_PATH}")
print("Final Shape:", model_data.shape)



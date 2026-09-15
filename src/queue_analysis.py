import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
"""V = Volume, T = Hospital Type, O = Ownership"""

DATA_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "processed"
    / "ed_hospital_modeling.csv"
)


df = pd.read_csv(DATA_PATH, dtype={"Facility ID": "string"})

volume_summary = (
    df.groupby("ed_volume", observed=True)["OP_18a"]
    .agg(["count", "mean", "median", "std"])
)
#print(volume_summary) # expectation of [ED duration | volume]

abandonment_summary = (
    df.groupby("ed_volume", observed=True)["OP_22"]
    .agg(["count", "mean", "median", "std"])
)
#print(abandonment_summary) #E[left before seen | volume]

""" Baseline Model, ED congestion - ED volume"""

ols_df = df[
    ["Facility ID",
     "Facility Name",
     "State",
     "OP_18a",
     "ed_volume",
     "Hospital Type",
     "Hospital Ownership",
     "Emergency Services"
     ]
].dropna(subset=["OP_18a", "ed_volume"])

print("OLS sample size:", len(ols_df))
print(ols_df["ed_volume"].value_counts())

print(
    ols_df.groupby("ed_volume")["OP_18a"]
    .agg(["count", "mean", "median", "std"])
)

model = smf.ols(
    formula=(
        'OP_18a ~ C(ed_volume, Treatment(reference="low"))'
    ),
    data=ols_df,
).fit()

print(model.summary())

robust_model = model.get_robustcov_results(
    cov_type="HC3"
)
print(robust_model.summary())

""" Baseline Diagnostics, Y ~ V"""

bp_test = het_breuschpagan(
    model.resid,
    model.model.exog
)

labels = [
    "LM Statistic",
    "LM p-value",
    "F Statistic",
    "F p-value"
]
print(dict(zip(labels, bp_test)))

plt.scatter(
    model.fittedvalues,
    model.resid,
    alpha=0.3,
)
plt.axhline(0, linestyle="--")

#plt.xlabel("Fitted ED Duration")
#plt.ylabel("Residual")
#plt.title("Residuals vs Fitted Values")

#plt.show()

sm.qqplot(
    model.resid,
    line="45",
    fit=True
)
plt.title("Q-Q Plot of OLS Residuals")
#plt.show()

ols_df["predicted"] = model.predict(ols_df)
ols_df["residual"] = (
    ols_df["OP_18a"] - ols_df["predicted"]
)

worst_residuals = (
    ols_df.sort_values("residual", ascending=False)
    .head(10)
)
print(
    worst_residuals[
        [
            "Facility Name",
            "State",
            "Hospital Type",
            "Hospital Ownership",
            "ed_volume",
            "OP_18a",
            "predicted",
            "residual"
        ]
    ].to_string(index=False)
)
best_residuals = (
    ols_df.sort_values("residual").head(10)
)
print(
    best_residuals[
        [
            "Facility Name",
            "State",
            "ed_volume",
            "OP_18a",
            "predicted",
            "residual"
        ]
    ].to_string(index=False)
)
""" Model 2
Form, Yi = B0 + B_V*V_i + B_T*T_i + epsilon_i, Y ~ V + T
"""
model2_df = df[
    [
        "OP_18a",
        "ed_volume",
        "Hospital Type"
    ]
].dropna()

print("\nModel 2 Sample Size:")
print(len(model2_df))

print("\nHospital Types in Model 2:")
print(model2_df["Hospital Type"].value_counts())

model2 = smf.ols(
    formula=(
        'OP_18a ~ C(ed_volume, Treatment(reference="low")) +'
        'C(Q("Hospital Type"), '
        'Treatment(reference="Acute Care Hospitals"))'
    ),
    data=model2_df,
).fit()

model2_robust = model2.get_robustcov_results(
    cov_type="HC3"
)
print(model2_robust.summary())
#Model 2 improved R^2, but did not improve heavy tails from excess Kurtosis

#sanity check for why only Acute Care and Critical Access remain

coverage_by_type = (
    df.groupby("Hospital Type").agg(
        total=("Facility ID", "size"),
        op18a_available=("OP_18a", "count"),
        volume_available=("ed_volume", "count")
    )
)

coverage_by_type["model_eligible"] = (
    df.assign(
        eligible=(
                df["OP_18a"].notna() & df["ed_volume"].notna()
        )
    ).groupby("Hospital Type")["eligible"].sum()
)

coverage_by_type["eligible_pct"] = (
    coverage_by_type["model_eligible"]
    / coverage_by_type["total"]
    *100
)
print(coverage_by_type)
#other hospitals are not eligible in combination of OP_18a and ed_volume
#Model2 covers among CMS Hospitals reporting both ED Volume and OP_18a

"""Model 3, Y ~ V + T + O"""

model3_df = df[
    [
        "Facility ID",
        "Facility Name",
        "State",
        "OP_18a",
        "ed_volume",
        "Hospital Type",
        "Hospital Ownership"
    ]
].dropna(
    subset=[
        "OP_18a",
        "ed_volume",
        "Hospital Type",
        "Hospital Ownership"
    ]
).copy()

model3 = smf.ols(
    formula = (
        'OP_18a ~ C(ed_volume, Treatment(reference="low")) +'
        'C(Q("Hospital Type"), '
        'Treatment(reference="Acute Care Hospitals")) +'
        'C(Q("Hospital Ownership"))'
    ),
    data=model3_df,
).fit()

model3_robust = model3.get_robustcov_results(
    cov_type="HC3"
)
print(model3_robust.summary())
"""R^2 = 0.400, but kurtosis remains high"""
#check residuals, using reference Volunary non-profit - Private as it was the largest

model3 = smf.ols(
    formula=(
        'OP_18a ~ C(ed_volume, Treatment(reference="low")) +'
        'C(Q("Hospital Type"), '
        'Treatment(reference="Acute Care Hospitals")) +'
        'C(Q("Hospital Ownership"),'
        'Treatment(reference="Voluntary non-profit - Private"))'
    ),
    data=model3_df,
).fit()

bp3 = het_breuschpagan(
    model.resid,
    model.model.exog
)

labels = [
    "LM Statistic",
    "LM p-value",
    "F Statistic",
    "F p-value"
]
print("\nModel 3 Breush-Pagan:")
print(dict(zip(labels, bp3)))

plt.scatter(
    model3.fittedvalues,
    model3.resid,
    alpha=0.3
)

plt.axhline(0, linestyle='--')
plt.xlabel("Fitted ED Duration")
plt.ylabel("Residual")
plt.title("Model 3: Residual vs Fitted Values")
plt.show()

#Q-Q plot
sm.qqplot(
    model3.resid,
    line="45",
    fit=True
)
plt.title("Model3: Q-Q Plot of Residuals")
plt.show()
#residual rankings
model3_df = model3_df.copy()
model3_df["predicted"] = model3.fittedvalues

model3_df["residual"] = model3.resid


worst_m3_residuals = (
    model3_df
    .sort_values("residual", ascending=False)
    .head(15)
)
best_m3_residuals = (
    model3_df.sort_values("residual", ascending=True)
    .head(15)
)

display_columns = [
    "Facility ID",
    "Facility Name",
    "Hospital Type",
    "Hospital Ownership",
    "ed_volume",
    "OP_18a",
    "predicted",
    "residual"
]
print("\nModel 3, Largest Positive Residuals")
print(
    worst_m3_residuals[
        display_columns
    ].to_string(index=False)
)
print("\nModel 3, Largest Negative Residuals")
print(
    best_m3_residuals[
        display_columns
    ].to_string(index=False)
)
#largest psoitive and negative residuals evince symmetry, top 15 for both are Acute Care
#Var(epsilon_i | Xi)

plt.scatter(
    model3.fittedvalues,
    model3.resid,
    alpha=0.3
)
plt.axhline(0, linestyle='--')

plt.xlabel("Fitted ED Duration")
plt.ylabel("Residual (minutes)")
plt.title("Model 3: Residual vs Fitted Values")
plt.show()

residual_by_volume = (
    model3_df
    .groupby("ed_volume", observed=True)["residual"]
    .agg(
        n="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max"
    )
)

print("\nModel 3 Residuals by ED Volume:")
print(residual_by_volume)

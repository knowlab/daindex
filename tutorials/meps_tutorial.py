# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# We will use the [`aif360`](https://aif360.readthedocs.io/en/latest/Getting%20Started.html) package to load the UCI adult dataset, fit a simple model and then analyse the fairness of the model using the DA-AUC.

# %%
import numpy as np
from aif360.datasets import MEPSDataset19
from aif360.explainers import MetricTextExplainer
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(1)

# %% [markdown]
# The below code downloads the required files from the UCI website if they are not already present in ai360's data directory.

# %%
import os
import shutil
import subprocess

import aif360

aif360_location = os.path.dirname(aif360.__file__)
meps_data_dir = os.path.join(aif360_location, "data", "raw", "meps")
h181_file_path = os.path.join(meps_data_dir, "h181.csv")

if not os.path.isfile(h181_file_path):
    r_script_path = os.path.join(meps_data_dir, "generate_data.R")
    process = subprocess.Popen(["Rscript", r_script_path], stdin=subprocess.PIPE)
    process.communicate(input=b"y\n")

    # Move the generated CSV files to meps_data_dir
    generated_files = ["h181.csv", "h192.csv"]
    for file_name in generated_files:
        src_path = os.path.join(os.getcwd(), file_name)
        dest_path = os.path.join(meps_data_dir, file_name)
        if os.path.isfile(src_path):
            shutil.move(src_path, dest_path)


# %%
def preprocessing_w_multimorb(df):
    """
    1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
      and 'non-White' otherwise
    2. Restrict to Panel 19
    3. RENAME all columns that are PANEL/ROUND SPECIFIC
    4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
    5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
    """

    def race(row):
        if (row["HISPANX"] == 2) and (
            row["RACEV2X"] == 1
        ):  # non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
            return "White"
        return "Non-White"

    df["RACE"] = df.apply(lambda row: race(row), axis=1)

    df = df[df["PANEL"] == 19]

    # RENAME COLUMNS
    df = df.rename(
        columns={
            "FTSTU53X": "FTSTU",
            "ACTDTY53": "ACTDTY",
            "HONRDC53": "HONRDC",
            "RTHLTH53": "RTHLTH",
            "MNHLTH53": "MNHLTH",
            "CHBRON53": "CHBRON",
            "JTPAIN53": "JTPAIN",
            "PREGNT53": "PREGNT",
            "WLKLIM53": "WLKLIM",
            "ACTLIM53": "ACTLIM",
            "SOCLIM53": "SOCLIM",
            "COGLIM53": "COGLIM",
            "EMPST53": "EMPST",
            "REGION53": "REGION",
            "MARRY53X": "MARRY",
            "AGE53X": "AGE",
            "POVCAT15": "POVCAT",
            "INSCOV15": "INSCOV",
        }
    )

    df = df[df["REGION"] >= 0]  # remove values -1
    df = df[df["AGE"] >= 0]  # remove values -1

    df = df[df["MARRY"] >= 0]  # remove values -1, -7, -8, -9

    df = df[df["ASTHDX"] >= 0]  # remove values -1, -7, -8, -9

    df = df[
        (
            df[
                [
                    "FTSTU",
                    "ACTDTY",
                    "HONRDC",
                    "RTHLTH",
                    "MNHLTH",
                    "HIBPDX",
                    "CHDDX",
                    "ANGIDX",
                    "EDUCYR",
                    "HIDEG",
                    "MIDX",
                    "OHRTDX",
                    "STRKDX",
                    "EMPHDX",
                    "CHBRON",
                    "CHOLDX",
                    "CANCERDX",
                    "DIABDX",
                    "JTPAIN",
                    "ARTHDX",
                    "ARTHTYPE",
                    "ASTHDX",
                    "ADHDADDX",
                    "PREGNT",
                    "WLKLIM",
                    "ACTLIM",
                    "SOCLIM",
                    "COGLIM",
                    "DFHEAR42",
                    "DFSEE42",
                    "ADSMOK42",
                    "PHQ242",
                    "EMPST",
                    "POVCAT",
                    "INSCOV",
                ]
            ]
            >= -1
        ).all(1)
    ]  # for all other categorical features, remove values < -1

    def utilization(row):
        return row["OBTOTV15"] + row["OPTOTV15"] + row["ERTOT15"] + row["IPNGTD15"] + row["HHTOTD15"]

    df["TOTEXP15"] = df.apply(lambda row: utilization(row), axis=1)
    lessE = df["TOTEXP15"] < 10.0
    df.loc[lessE, "TOTEXP15"] = 0.0
    moreE = df["TOTEXP15"] >= 10.0
    df.loc[moreE, "TOTEXP15"] = 1.0
    df["MULTIMORBIDITY"] = (
        df.filter(regex="DX$|CHBRON$|JTPAIN$").drop(columns=["ADHDADDX"]).apply(lambda x: (x == 1).sum(), axis=1)
    )

    df = df.rename(columns={"TOTEXP15": "UTILIZATION"})
    return df


# %%
dataset_orig_panel19_train, dataset_orig_panel19_val, dataset_orig_panel19_test = MEPSDataset19(
    custom_preprocessing=preprocessing_w_multimorb,
    features_to_keep=[
        "REGION",
        "AGE",
        "SEX",
        "RACE",
        "RACEV2X",
        "MARRY",
        "FTSTU",
        "ACTDTY",
        "HONRDC",
        "RTHLTH",
        "MNHLTH",
        "HIBPDX",
        "CHDDX",
        "ANGIDX",
        "MIDX",
        "OHRTDX",
        "STRKDX",
        "EMPHDX",
        "CHBRON",
        "CHOLDX",
        "CANCERDX",
        "DIABDX",
        "JTPAIN",
        "ARTHDX",
        "ARTHTYPE",
        "ASTHDX",
        "ADHDADDX",
        "PREGNT",
        "WLKLIM",
        "ACTLIM",
        "SOCLIM",
        "COGLIM",
        "DFHEAR42",
        "DFSEE42",
        "ADSMOK42",
        "PCS42",
        "MCS42",
        "K6SUM42",
        "PHQ242",
        "EMPST",
        "POVCAT",
        "INSCOV",
        "UTILIZATION",
        "PERWT15F",
        "MULTIMORBIDITY",
    ],
).split([0.5, 0.8], shuffle=True)

sens_ind = 0
sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]


# %%
def describe(train=None, val=None, test=None):
    if train is not None:
        print(train.features.shape)
    if val is not None:
        print(val.features.shape)
    print(test.features.shape)
    print(test.favorable_label, test.unfavorable_label)
    print(test.protected_attribute_names)
    print(test.privileged_protected_attributes, test.unprivileged_protected_attributes)
    print(test.feature_names)


# %%
describe(dataset_orig_panel19_train, dataset_orig_panel19_val, dataset_orig_panel19_test)

# %%
metric_orig_panel19_train = BinaryLabelDatasetMetric(
    dataset_orig_panel19_train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
)
explainer_orig_panel19_train = MetricTextExplainer(metric_orig_panel19_train)

print(explainer_orig_panel19_train.disparate_impact())

# %%
dataset = dataset_orig_panel19_train
model = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", random_state=1))
fit_params = {"logisticregression__sample_weight": dataset.instance_weights}

lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

# %%
from collections import defaultdict


def test(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            dataset, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
        )

        metric_arrs["bal_acc"].append((metric.true_positive_rate() + metric.true_negative_rate()) / 2)
        metric_arrs["avg_odds_diff"].append(metric.average_odds_difference())
        metric_arrs["disp_imp"].append(metric.disparate_impact())
        metric_arrs["stat_par_diff"].append(metric.statistical_parity_difference())
        metric_arrs["eq_opp_diff"].append(metric.equal_opportunity_difference())
        metric_arrs["theil_ind"].append(metric.theil_index())

    return metric_arrs


# %%
thresh_arr = np.linspace(0.01, 0.5, 50)
val_metrics = test(dataset=dataset_orig_panel19_val, model=lr_orig_panel19, thresh_arr=thresh_arr)
lr_orig_best_ind = np.argmax(val_metrics["bal_acc"])


# %%
def describe_metrics(metrics, thresh_arr):
    best_ind = np.argmax(metrics["bal_acc"])
    print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    print("Best balanced accuracy: {:6.4f}".format(metrics["bal_acc"][best_ind]))
    disp_imp_at_best_ind = 1 - min(metrics["disp_imp"][best_ind], 1 / metrics["disp_imp"][best_ind])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics["avg_odds_diff"][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics["stat_par_diff"][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics["eq_opp_diff"][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics["theil_ind"][best_ind]))


# %%
describe_metrics(val_metrics, thresh_arr)

# %%
lr_orig_metrics = test(
    dataset=dataset_orig_panel19_test, model=lr_orig_panel19, thresh_arr=[thresh_arr[lr_orig_best_ind]]
)

# %%
describe_metrics(lr_orig_metrics, [thresh_arr[lr_orig_best_ind]])

# %%
df = dataset.convert_to_dataframe()[0]

# %%
df["MULTIMORBIDITY"].hist()
df_add = df.copy()
df_add["RTHLTH"] = df_add.filter(regex="RTHLTH").idxmax(axis=1)

# %%
df_add.boxplot(column="MULTIMORBIDITY", by="RTHLTH")

# %%
from daindex import DAIndex, DeteriorationFeature, Group

# %% [markdown]
# Let's first specify our deterioration feature and groups.

# %%
det_feature = DeteriorationFeature(col="MULTIMORBIDITY", threshold=1, is_discrete=True)
groups = [
    Group("White", 1, "RACEV2X"),
    Group(
        "Non-White", [12, 5, 10, 2, 3, 6, 4], "RACEV2X", det_threshold=1.2
    ),  # Modify deterioration threshold only for this group
    Group("Black", 2, "RACEV2X"),
    Group("Asian", [4, 5, 6, 10], "RACEV2X"),
]
print(det_feature)
print(groups[1])

# %% [markdown]
# We can call a group on the cohort to get the group representation in the cohort:

# %%
print(groups[0](df).RACEV2X.value_counts())
groups[0](df).head()

# %% [markdown]
# We can now instantiate the `DAIndex` with the components above and evaluate all groups via the model we have made. Note this could also be done with a list of arbitrary models. The package also supports evaluation via an existing predictions column in the cohort, see `evaluate_all_groups_from_predictions`. We could also just compare two specific groups with `evaluate_group_pair_by_models` and `evaluate_group_pair_from_predictions`.

# %%
feature_list = dataset.feature_names
index = DAIndex(
    cohort=df,
    groups=groups,
    det_feature=det_feature,
    decision_boundary=0.75,  # Set decision boundary for DA curve
)
index.evaluate_all_groups_by_models(
    models=lr_orig_panel19,
    feature_list=feature_list,
    reference_group="White",
    n_jobs=1,  # Up n_jobs to parallelize if needed
)

# %%
index.present_all_results()

# %%
index.get_all_ratios()

# %%
index.get_group_ratios("White", "Asian")

# %%
index.get_all_groups_failed_steps()

# %%

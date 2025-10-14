import json
import pandas as pd

def extract_dataset(json_file):
    """Extracts metabolite and metadata tables from a MW JSON file."""
    with open(json_file) as f:
        data = json.load(f)

    table_section = data.get("MS_METABOLITE_DATA", None)
    if table_section is None:
        raise KeyError(f"Peak table section not found in {json_file}")

    meta_data = data.get("SUBJECT_SAMPLE_FACTORS", None)
    if meta_data is None:
        raise KeyError(f"Meta data section not found in {json_file}")

    meta_data = pd.DataFrame(meta_data).set_index("Sample ID")

    metabolite_data = pd.DataFrame(table_section["Data"]).T
    metabolites = metabolite_data.iloc[0].tolist()
    metabolite_data = metabolite_data.iloc[1:].copy()
    metabolite_data.index.name = "Sample ID"
    metabolite_data.columns = metabolites

    # Drop duplicate sample IDs
    metabolite_data = metabolite_data[~metabolite_data.index.duplicated(keep="first")]
    meta_data = meta_data[~meta_data.index.duplicated(keep="first")]

    # Find common samples
    common_ids = metabolite_data.index.intersection(meta_data.index)
    metabolite_data = metabolite_data.loc[common_ids]
    meta_data = meta_data.loc[common_ids]

    # Expand "Factors" into separate columns
    factors_df = meta_data["Factors"].apply(pd.Series)
    for col in ["Genotype", "Treatment", "Sample source"]:
        if col in factors_df.columns:
            factors_df[col] = pd.factorize(factors_df[col])[0]

    return metabolite_data, meta_data, factors_df


# --- Extract both datasets ---
metabo_6992, meta_6992, factors_6992 = extract_dataset("ST004205_AN006992.json")
metabo_6994, meta_6994, factors_6994 = extract_dataset("ST004205_AN006994.json")

# --- Align datasets by common Sample IDs ---
common_samples = metabo_6992.index.intersection(metabo_6994.index)
metabo_6992_aligned = metabo_6992.loc[common_samples]
metabo_6994_aligned = metabo_6994.loc[common_samples]

meta_6992_aligned = meta_6992.loc[common_samples]
meta_6994_aligned = meta_6994.loc[common_samples]

# --- Write aligned data to CSV ---
metabo_6992_aligned.to_csv("ST004205_AN006992.csv", index=True, header=True)
metabo_6994_aligned.to_csv("ST004205_AN006994.csv", index=True, header=True)

meta_6992_aligned.to_csv("meta_ref_6992.csv", index=True, header=True)
meta_6994_aligned.to_csv("meta_ref_6994.csv", index=True, header=True)

# --- Write processed factors ---
factors_6992.to_csv("meta_ST004205_AN006992.csv", index=True, header=True)
factors_6994.to_csv("meta_ST004205_AN006994.csv", index=True, header=True)

# --- Optional: concatenate both for cross-omics alignment ---
aligned_concat = pd.concat(
    [metabo_6992_aligned.add_prefix("MS1_"),
     metabo_6994_aligned.add_prefix("MS2_")],
    axis=1
)
aligned_concat = aligned_concat.loc[:, (aligned_concat != "N.D.").any(axis=0)]
aligned_concat = aligned_concat.replace("N.D.", pd.NA)
aligned_concat.to_csv("aligned_ST004205_MS1_MS2.csv", index=True, header=True)

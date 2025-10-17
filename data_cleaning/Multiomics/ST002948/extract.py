import json
import pandas as pd


def extract_dataset(json_file):
    """Extracts metabolite data, metadata, and expanded factors table from a MW JSON file."""
    with open(json_file) as f:
        data = json.load(f)

    # --- Extract metabolite data ---
    table_section = data.get("MS_METABOLITE_DATA", None)
    if table_section is None:
        raise KeyError(f"Peak table section not found in {json_file}")

    metabolite_data = pd.DataFrame(table_section["Data"]).T
    metabolites = metabolite_data.iloc[0].tolist()
    metabolite_data = metabolite_data.iloc[1:].copy()
    metabolite_data.index.name = "Sample ID"
    metabolite_data.columns = metabolites

    # --- Extract metadata ---
    meta_data = data.get("SUBJECT_SAMPLE_FACTORS", None)
    if meta_data is None:
        raise KeyError(f"Meta data section not found in {json_file}")

    meta_data = pd.DataFrame(meta_data).set_index("Sample ID")

    # --- Expand nested fields ---
    factors_df = meta_data["Factors"].apply(pd.Series)
    add_data_df = meta_data["Additional sample data"].apply(pd.Series)

    # Combine essential metadata fields
    combined_meta = pd.concat([
        meta_data["Subject ID"],
        factors_df[["Time", "Treatment"]],
        add_data_df[["Gender"]]
    ], axis=1)

    # --- Drop duplicate sample IDs ---
    metabolite_data = metabolite_data[~metabolite_data.index.duplicated(keep="first")]
    meta_data = meta_data[~meta_data.index.duplicated(keep="first")]
    combined_meta = combined_meta[~combined_meta.index.duplicated(keep="first")]

    # --- Align sample IDs ---
    common_ids = metabolite_data.index.intersection(combined_meta.index)
    metabolite_data = metabolite_data.loc[common_ids]
    meta_data = meta_data.loc[common_ids]
    combined_meta = combined_meta.loc[common_ids]

    numerical_meta = combined_meta.copy()
    for col in ["Subject ID", "Time", "Treatment", "Gender"]:
        #if col in factors_df.columns:
            numerical_meta[col] = pd.factorize(combined_meta[col])[0]

    return metabolite_data, combined_meta, numerical_meta


# --- Extract both datasets ---
metabo_4834, meta_4834, factors_4834 = extract_dataset("ST002948_AN004834.json")
metabo_4835, meta_4835, factors_4835 = extract_dataset("ST002948_AN004835.json")

# --- Align datasets by common Sample IDs ---
common_samples = metabo_4834.index.intersection(metabo_4835.index)
metabo_4834_aligned = metabo_4834.loc[common_samples]
metabo_4835_aligned = metabo_4835.loc[common_samples]

meta_4834_aligned = meta_4834.loc[common_samples]
meta_4835_aligned = meta_4835.loc[common_samples]

# --- Write aligned data to CSV ---
metabo_4834_aligned.to_csv("ST002948_AN004834.csv", index=True, header=True)
metabo_4835_aligned.to_csv("ST002948_AN004835.csv", index=True, header=True)

meta_4834_aligned.to_csv("meta_ref_4834.csv", index=True, header=True)
meta_4835_aligned.to_csv("meta_ref_4835.csv", index=True, header=True)

# --- Write processed factors ---
factors_4834.to_csv("meta_ST002948_AN004834.csv", index=True, header=True)
factors_4835.to_csv("meta_ST002948_AN004835.csv", index=True, header=True)

# --- Optional: concatenate both for cross-omics alignment ---
aligned_concat = pd.concat(
    [metabo_4834_aligned.add_prefix("HILIC_"),
     metabo_4835_aligned.add_prefix("REVER_")],
    axis=1
)
aligned_concat = aligned_concat.loc[:, (aligned_concat != "N.D.").any(axis=0)]
aligned_concat = aligned_concat.replace("N.D.", pd.NA)
aligned_concat.to_csv("aligned_ST002948_HILIC_REVER.csv", index=True, header=True)

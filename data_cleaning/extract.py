import mwtab
import pandas as pd
from mwtab.mwextract import write_metabolites_csv, extract_metabolites, generate_matchers, extract_metadata
import re
import json

#Extracting the peak table information from the first dataset.
with open("ST000355_AN000580.json") as f:
    data = json.load(f)

table_section = data.get("MS_METABOLITE_DATA",None)
if table_section is None:
    raise KeyError("Peak table section not found in JSON")

meta_data = data.get("SUBJECT_SAMPLE_FACTORS",None)
if meta_data is None:
    raise KeyError("Meta data section is not found in JSON")

meta_data = pd.DataFrame(meta_data)
meta_data.set_index("Sample ID",inplace=True)

metabolite_data = pd.DataFrame(table_section["Data"]).T
metabolites = metabolite_data.iloc[0].tolist()
metabolite_data = metabolite_data.iloc[1:].copy()
metabolite_data.index.name = "Sample ID"
metabolite_data.columns = metabolites

metabolite_data = metabolite_data[~metabolite_data.index.duplicated(keep="first")]
meta_data = meta_data[~meta_data.index.duplicated(keep="first")]

common_ids = metabolite_data.index.intersection(meta_data.index)

df_metabolites = metabolite_data.loc[common_ids]
df_metadata = meta_data.loc[common_ids]

df_metabolites.to_csv("ST000355_AN000580.csv",index=False,header=True)
df_metadata.to_csv("580_ref.csv")

factors_df = df_metadata["Factors"].apply(pd.Series)
additional_df = df_metadata["Additional sample data"].apply(pd.Series)

for col in ["Stage", "Diagnosis"]:
    factors_df[col] = pd.factorize(factors_df[col])[0]

for col in ['Sub-group', 'Clinical_Parameters', 'ER', 'PR', 'HER-2 neu', 'YOB',
       'Age (now)', 'Age (BOD-OD)', 'Race', 'Stage OB', 'BMI',
       'Year of Sample Collection', 'Prog/Relapse by 9/7/11']:
    additional_df[col] = pd.factorize(additional_df[col])[0]

factors_df = factors_df.add_prefix("F_")
additional_df  = additional_df.add_prefix("A_")

out = pd.concat([factors_df,additional_df],axis=1)

out.to_csv("meta_ST000355_AN00580.csv", index=False, header=True)

## Extracting the peak table information for the second dataset. Different formatting.
with open("ST000356_AN000583.json") as f:
    data = json.load(f)

table_section = data.get("MS_METABOLITE_DATA",None)
if table_section is None:
    raise KeyError("Peak table section not found in JSON")

meta_data = data.get("SUBJECT_SAMPLE_FACTORS",None)
if meta_data is None:
    raise KeyError("Meta data section is not found in JSON")


rows = []
for entry in meta_data:
    row = {
        "Sample ID": entry["Sample ID"],
        "Diagnosis": entry["Factors"]["Diagnosis"],
        "Stage": entry["Factors"]["stage"],
        "YOB": entry["Additional sample data"].get("YOB", "-"),
        "Race": entry["Additional sample data"].get("race", "-")
    }
    rows.append(row)

meta_df = pd.DataFrame(rows)


for col in ["Stage", "Diagnosis", "YOB", "Race"]:
    meta_df[col] = pd.factorize(meta_df[col])[0]

meta_df.set_index("Sample ID",inplace=True)

metabolite_data = pd.DataFrame(table_section["Data"]).T
metabolites = metabolite_data.iloc[0].tolist()
metabolite_data = metabolite_data.iloc[1:].copy()
metabolite_data.index.name = "Sample ID"
metabolite_data.columns = metabolites


print(metabolite_data)

common_ids = metabolite_data.index.intersection(meta_df.index)

pd.DataFrame(meta_data).to_csv("583_ref.csv")
df_metabolites = metabolite_data.loc[common_ids]
df_metadata = meta_df.loc[common_ids]

#The last five rows have missing experimental information
df_metadata = df_metadata.iloc[:-5]
df_metabolites = df_metabolites.iloc[:-5]

print(df_metadata)

df_metadata.to_csv("meta_ST000356_AN00583.csv", index=False, header=True)

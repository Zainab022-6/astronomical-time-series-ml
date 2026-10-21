import pandas as pd
import numpy as np
import lightkurve as lk
from astropy.timeseries import LombScargle
from scipy.stats import skew, kurtosis

def extract_features(target_id):
    print(f"Fetching {target_id}...")
    try:
        search = lk.search_lightcurve(target_id, mission="Kepler")
        if len(search) == 0:
            print(f"  [!] No Kepler data for {target_id}")
            return None
        
        # Download the first available quarter
        lc = search[0].download()
        clean = lc.remove_nans().remove_outliers(sigma=4).normalize()
        
        time = clean.time.value
        flux = clean.flux.value
        
        # 1. Summary Statistics
        mean_val = np.mean(flux)
        std_val = np.std(flux)
        amplitude = (np.percentile(flux, 95) - np.percentile(flux, 5)) / 2.0
        skew_val = float(skew(flux))
        kurt_val = float(kurtosis(flux))
        
        # 2. Lomb-Scargle Period Analysis
        # Search periods between 0.1 days (~2.4 hours) and 15 days
        ls = LombScargle(time, flux)
        freq, power = ls.autopower(minimum_frequency=1/15.0, maximum_frequency=1/0.1)
        best_freq = freq[np.argmax(power)]
        best_period = 1.0 / best_freq
        peak_power = float(np.max(power))
        
        return {
            "period_days": best_period,
            "peak_power": peak_power,
            "amplitude": amplitude,
            "flux_std": std_val,
            "skewness": skew_val,
            "kurtosis": kurt_val
        }
    except Exception as e:
        print(f"  [Error] {target_id}: {e}")
        return None

# Load targets and process
targets_df = pd.read_csv("sample_targets.csv")
dataset_rows = []

for _, row in targets_df.iterrows():
    feats = extract_features(row["target_id"])
    if feats:
        feats["target_id"] = row["target_id"]
        feats["label"] = row["label"]
        dataset_rows.append(feats)

result_df = pd.DataFrame(dataset_rows)
result_df.to_csv("extracted_features.csv", index=False)
print("\nFeature extraction complete! Saved to extracted_features.csv")
print(result_df.head())
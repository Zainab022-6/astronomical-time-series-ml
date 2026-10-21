import lightkurve as lk
import matplotlib.pyplot as plt

print("Searching MAST archive for target RR Lyrae...")
search_result = lk.search_lightcurve("RR Lyr", mission="Kepler")
print(f"Found {len(search_result)} light curve products.")

print("Downloading first light curve...")
lc = search_result[0].download()

# Clean and normalize
clean_lc = lc.remove_nans().remove_outliers(sigma=4).normalize()

# Compute Lomb-Scargle periodogram
pg = clean_lc.to_periodogram(method="lombscargle", minimum_period=0.2, maximum_period=1.5)
print(f"Detected dominant period: {pg.period_at_max_power.value:.4f} days (expected ~0.5668 days)")

# Save phase-folded plot
folded_lc = clean_lc.fold(period=pg.period_at_max_power)
ax = folded_lc.scatter()
ax.set_title(f"RR Lyrae (Folded at P = {pg.period_at_max_power.value:.4f} d)")
plt.savefig("rr_lyr_test.png", dpi=150)
print("Saved verification plot to rr_lyr_test.png")
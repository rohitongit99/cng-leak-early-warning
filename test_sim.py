from simulator import generate_stream
from risk import load, risk_from_df, level

bundle = load()
print("lo =", round(bundle["lo"], 4), " hi =", round(bundle["hi"], 4))

df = generate_stream(500, leaks=[(200, 100)], spikes=[50], seed=5)
r = risk_from_df(df, bundle)

print()
print("Normal data (rows 70-199): max risk =", round(r.loc[70:199].max(), 1))
print("Normal data (rows 70-199): average risk =", round(r.loc[70:199].mean(), 1))

print()
print("Spike ke aaspaas:")
print(r.loc[48:60].round(1).to_string())

print()
print("Leak ke andar:")
for i in [200, 210, 220, 240, 260, 290]:
    print(i, round(r.loc[i], 1), level(r.loc[i]))
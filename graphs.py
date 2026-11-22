import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Läs in datan
df = pd.read_csv("Entreprenadförsäkring training.csv")

# Beräkna medelvärdet för antalet skador
mean_val = df['AntalSkador'].mean()

# Beräkna observerade andelar
observed_counts = df['AntalSkador'].value_counts(normalize=True).sort_index()

# Sätt upp värden för x-axeln (0, 1, 2)
x_vals = np.array([0, 1, 2])
observed = [observed_counts.get(x, 0) for x in x_vals]

# Beräkna förväntade andelar enligt en teoretisk Poisson-fördelning
expected = poisson.pmf(x_vals, mean_val)

# Skapa figuren (en enda graf)
fig, ax = plt.subplots(figsize=(8, 6))

# Rita observerad data som staplar
ax.bar(x_vals, observed, color='#2c3e50', alpha=0.9, label='Faktisk data')

# Rita teoretisk Poisson som en kurva (linje med markörer) ovanpå
ax.plot(x_vals, expected, color='#e74c3c', marker='o', linestyle='-', 
        linewidth=2, markersize=8, label='Teoretisk Poisson-kurva')

# Formatering av grafen
ax.set_xlabel('Antal skador per företag', fontsize=12)
ax.set_ylabel('Andel av företagen (Logaritmisk)', fontsize=12)
ax.set_title('Logaritmisk skala med överlagrad Poisson-kurva', fontsize=14)
ax.set_xticks(x_vals)

# Aktivera logaritmisk y-axel
ax.set_yscale('log') 

# Lägg till legend och rutnät
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Justera layout och visa grafen
plt.tight_layout()
plt.savefig('poissonfördelning.png')

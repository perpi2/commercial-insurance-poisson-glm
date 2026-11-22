import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import patsy

# =========================================================
# 1. SETUP OCH DATAFÖRBEREDELSE
# =========================================================

def preprocess_data(filepath):
    """Laddar data, filtrerar bort ogiltig duration och logaritmerar omsättning."""
    df = pd.read_csv(filepath)
    df = df[df['Duration'] > 0].copy()
    df['Log_Omsattning'] = np.log(df['Omsattning'] + 1)
    return df

# Ladda och förbered tränings- och testdata
df_train = preprocess_data("Entreprenadförsäkring training.csv")
df_test = preprocess_data("Entreprenadförsäkring test.csv")

# =========================================================
# 2. MODELLTRÄNING
# =========================================================

# Definiera och träna modellen
formula = "AntalSkador ~ Log_Omsattning + C(Verksamhet) + C(GeografisktOmrade)"
poisson_model = smf.poisson(
    formula=formula,
    data=df_train,
    offset=np.log(df_train['Duration'])
).fit()

print(poisson_model.summary())

# =========================================================
# 3. PREDIKTION OCH UTVÄRDERING (KPI:ER)
# =========================================================

# Gör prediktioner med 95% konfidensintervall
predictions = poisson_model.get_prediction(exog=df_test, offset=np.log(df_test['Duration']))
pred_summary = predictions.summary_frame(alpha=0.05)

df_test['Pred_AntalSkador'] = pred_summary['predicted']
df_test['Pred_CI_Lower'] = pred_summary['ci_lower']
df_test['Pred_CI_Upper'] = pred_summary['ci_upper']

# --- Beräkna KPI:er ---
actual = df_test['AntalSkador']
pred = df_test['Pred_AntalSkador']

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
poisson_dev = mean_poisson_deviance(actual, pred)

total_actual = actual.sum()
total_pred = pred.sum()
total_duration = df_test['Duration'].sum()

actual_freq = total_actual / total_duration
pred_freq = total_pred / total_duration

# --- Beräkna totalt konfidensintervall (Delta-metoden) ---
_, X_test = patsy.dmatrices(formula, df_test, return_type='dataframe')
gradient = X_test.mul(df_test['Pred_AntalSkador'], axis=0).sum(axis=0)
cov_matrix = poisson_model.cov_params()

var_total = gradient.dot(cov_matrix).dot(gradient)
se_total = np.sqrt(var_total)

ci_lower_total = total_pred - 1.96 * se_total
ci_upper_total = total_pred + 1.96 * se_total

# --- Skriv ut resultat ---
print("\n" + "="*40)
print("--- UTVÄRDERING PÅ TESTDATA ---")
print("="*40)
print(f"Faktiskt antal skador:               {total_actual}")
print(f"Modellens förväntade skador totalt:  {total_pred:.2f} ± {1.96 * se_total:.2f}")
print(f"95% Konfidensintervall:              [{ci_lower_total:.2f}, {ci_upper_total:.2f}]")
print(f"Kvot (Förväntat / Faktiskt):         {total_pred / total_actual:.4f}")
print("-" * 40)
print(f"Faktisk skadefrekvens:               {actual_freq:.4f} (skador/år)")
print(f"Modellens förväntade skadefrekvens:  {pred_freq:.4f} (skador/år)")
print("-" * 40)
print(f"MAE (Mean Absolute Error):           {mae:.4f}")
print(f"RMSE (Root Mean Squared Error):      {rmse:.4f}")
print(f"Mean Poisson Deviance:               {poisson_dev:.4f}")
print("="*40 + "\n")

# =========================================================
# 4. SKAPA ENKLA GRAFER (KATEGORIER)
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Helper-funktion för bar charts
def plot_agg_bar(df, group_col, ax, title):
    agg_df = df.groupby(group_col)[['AntalSkador', 'Pred_AntalSkador']].sum().sort_values(by='AntalSkador', ascending=False)
    agg_df.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title(title)
    ax.set_ylabel('Totalt antal skador')
    ax.set_xlabel('')
    ax.legend(['Faktiska Skador', 'Förväntade Skador'])
    ax.tick_params(axis='x', rotation=45)

plot_agg_bar(df_test, 'Verksamhet', axes[0], 'Faktiska vs Förväntade Skador per Verksamhet')
plot_agg_bar(df_test, 'GeografisktOmrade', axes[1], 'Faktiska vs Förväntade Skador per Geografiskt Område')

plt.tight_layout()
plt.savefig('test_performance_no_year.png')
print("Kategorigrafer sparade som 'test_performance_no_year.png'.")

# =========================================================
# 5. SKAPA AVANCERAD GLM-VALIDERINGSGRAF (LIFT CHART)
# =========================================================

# 1. Binning
min_oms = df_test['Omsattning'].replace(0, 1).min() 
max_oms = df_test['Omsattning'].max()
bins = np.logspace(np.log10(min_oms), np.log10(max_oms), 35)

df_test['Oms_Bin'] = pd.cut(df_test['Omsattning'], bins=bins)

# 2. Aggregering
binned = df_test.groupby('Oms_Bin', observed=False).agg(
    Tot_Skador=('AntalSkador', 'sum'),
    Tot_Pred_Skador=('Pred_AntalSkador', 'sum'),
    Tot_Pred_CI_Lower=('Pred_CI_Lower', 'sum'), 
    Tot_Pred_CI_Upper=('Pred_CI_Upper', 'sum'), 
    Tot_Duration=('Duration', 'sum')
).reset_index()

binned = binned[binned['Tot_Duration'] > 0].copy()

# 3. Beräkna frekvenser
binned['Obs_Freq'] = binned['Tot_Skador'] / binned['Tot_Duration']
binned['Pred_Freq'] = binned['Tot_Pred_Skador'] / binned['Tot_Duration']
binned['Pred_Freq_Lower'] = binned['Tot_Pred_CI_Lower'] / binned['Tot_Duration']
binned['Pred_Freq_Upper'] = binned['Tot_Pred_CI_Upper'] / binned['Tot_Duration']
binned['Bin_Center'] = [np.sqrt(interval.left * interval.right) for interval in binned['Oms_Bin']]

# Dölj noll-värden för log-skalan
binned_plot = binned[binned['Obs_Freq'] > 0]

# 4. Rita grafen
fig, ax1 = plt.subplots(figsize=(14, 7))
bg_color = "#ffffff"
fig.patch.set_facecolor(bg_color)
ax1.set_facecolor(bg_color)

# Sekundär axel (Histogram / Exponering)
ax2 = ax1.twinx()
n, _, patches = ax2.hist(df_test['Omsattning'], bins=bins, weights=df_test['Duration'],
                         color='#9ecae1', edgecolor='black', linewidth=0.5, alpha=0.9, label='Exponering (Duration)')
ax2.set_ylim(0, max(n) * 3) 
ax2.set_yticks([]) 

# Primär axel (Modell vs Faktiskt)
line_pred, = ax1.plot(binned_plot['Bin_Center'], binned_plot['Pred_Freq'],
                      color='#d62728', linewidth=2.5, zorder=4, label='Modellens prediktion')

line_int, = ax1.plot(binned_plot['Bin_Center'], binned_plot['Pred_Freq_Upper'],
                     color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3, label='95% Konfidensintervall')
ax1.plot(binned_plot['Bin_Center'], binned_plot['Pred_Freq_Lower'],
         color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)

scatter_obs = ax1.scatter(binned_plot['Bin_Center'], binned_plot['Obs_Freq'],
                          color='#1f77b4', edgecolor='black', s=50, zorder=5, label='Faktiskt utfall')

# 5. Formatering
ax1.set_xscale('log')
ax1.set_yscale('log')

ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", " ")))
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

ax1.set_xlabel('Omsättning (SEK)', fontweight='bold', labelpad=15)
ax1.set_ylabel('Skadefrekvens (Skador per år)', fontweight='bold', labelpad=15)
plt.title('Poisson-regression: Förväntat vs Faktiskt utfall (Testdata)', fontweight='bold', pad=15)

ax1.grid(True, which='major', linestyle='--', alpha=0.6, color='gray')
ax1.grid(True, which='minor', linestyle=':', alpha=0.3, color='gray')

ax1.legend(handles=[line_pred, line_int, scatter_obs, patches[0]], 
           loc='upper left', frameon=True, edgecolor='black', facecolor=bg_color)

plt.tight_layout()
plt.savefig('advanced_glm_validation.png', dpi=300)
print("Avancerad valideringsgraf sparad som 'advanced_glm_validation.png'")

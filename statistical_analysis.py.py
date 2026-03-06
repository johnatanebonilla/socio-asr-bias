#!/usr/bin/env python3
"""
Corrected dialect assignments + all NB models + mixed-effects attempt.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, digamma
from scipy.stats import norm, mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/mnt/user-data/uploads/final_master_table_for_experiments.csv')
df.loc[(df['province']=='La Coruña') & (df['ccaa_normalized'].isna() | (df['ccaa_normalized']=='')), 'ccaa_normalized'] = 'Galicia'

# ============================================================
# FIX DIALECT ASSIGNMENTS using Johnatan's dictionaries
# ============================================================
MOUTON_CORRECT = {}
for g, provs in {
    'Northern': ['Álava','Burgos','Cantabria','La Rioja','Navarra','Soria',
        'Zaragoza','Huesca','Teruel','Palencia','Valladolid','León',
        'Zamora','Salamanca','Ávila','Segovia','Guadalajara','Madrid',
        'Cuenca','Toledo','Lugo','Orense','Pontevedra','La Coruña',
        'Asturias','Gerona','Barcelona','Lérida','Tarragona',
        'Castellón','Guipúzcoa','Vizcaya'],
    'Southern': ['Huelva','Sevilla','Cádiz','Córdoba','Málaga','Granada',
        'Jaén','Almería','Murcia','Badajoz','Cáceres',
        'Las Palmas','Santa Cruz de Tenerife','Valencia','Alicante',
        'Albacete','Ciudad Real']
}.items():
    for p in provs:
        MOUTON_CORRECT[p] = g

FO_CORRECT = {}
for g, provs in {
    'Central-Northern': ['Cantabria','Burgos','Palencia','Valladolid','León','Zamora',
        'Salamanca','Ávila','Segovia','Soria','Guadalajara',
        'Madrid','Toledo','Cuenca','Ciudad Real','Albacete'],
    'Western': ['Lugo','Orense','Pontevedra','La Coruña','Asturias',
        'Badajoz','Cáceres','Huelva'],
    'Southern': ['Sevilla','Cádiz','Córdoba','Málaga','Granada','Jaén',
        'Almería','Murcia','Las Palmas','Santa Cruz de Tenerife',
        'Valencia','Alicante'],
    'Eastern': ['Zaragoza','Huesca','Teruel','Gerona','Barcelona','Lérida',
        'Tarragona','Castellón','La Rioja','Navarra',
        'Guipúzcoa','Vizcaya','Álava']
}.items():
    for p in provs:
        FO_CORRECT[p] = g

# Apply corrections
df['mouton_corrected'] = df['province'].map(MOUTON_CORRECT)
df['fo_corrected'] = df['province'].map(FO_CORRECT)

# Report changes
m_changes = (df['dialect_mouton'] != df['mouton_corrected']) & df['mouton_corrected'].notna()
f_changes = (df['dialect_fernandez'] != df['fo_corrected']) & df['fo_corrected'].notna()
print(f"Mouton corrections: {m_changes.sum()} rows")
print(f"FO corrections: {f_changes.sum()} rows")

df['errors'] = df['S_clean'] + df['D_clean'] + df['I_clean']
df['log_offset'] = np.log(df['N_clean'].clip(lower=1))

# ============================================================
# NB2 fitting function (same as before)
# ============================================================
def fit_nb2(y, X, offset):
    n, p = X.shape
    def pois_ll(b):
        mu = np.exp(X @ b + offset)
        return -(y*np.log(mu) - mu - gammaln(y+1)).sum()
    r0 = minimize(pois_ll, np.zeros(p), method='L-BFGS-B', options={'maxiter':200})
    mu0 = np.exp(X @ r0.x + offset)
    a0 = max(((y-mu0)**2/mu0).sum()/n - 1, 0.1) / mu0.mean()
    p0 = np.concatenate([r0.x, [np.log(max(a0,0.01))]])
    
    def nb_ll(params):
        beta=params[:-1]; alpha=np.exp(params[-1]); ia=1.0/alpha
        mu=np.exp(X@beta+offset)
        return -(gammaln(y+ia)-gammaln(ia)-gammaln(y+1)+ia*np.log(ia/(ia+mu))+y*np.log(mu/(ia+mu))).sum()
    
    def nb_grad(params):
        beta=params[:-1]; alpha=np.exp(params[-1]); ia=1.0/alpha
        mu=np.exp(X@beta+offset)
        gb = X.T @ ((y-mu)/(1+alpha*mu))
        ga = ((-ia**2)*(digamma(y+ia)-digamma(ia)+np.log(ia/(ia+mu))+1-(y+ia)/(ia+mu))).sum()
        return -np.concatenate([gb, [ga*alpha]])
    
    res = minimize(nb_ll, p0, jac=nb_grad, method='L-BFGS-B', options={'maxiter':500,'ftol':1e-10})
    if not res.success:
        res = minimize(nb_ll, p0, method='L-BFGS-B', options={'maxiter':500})
    
    beta=res.x[:-1]; alpha=np.exp(res.x[-1])
    eps=1e-5; np_=len(res.x); H=np.zeros((np_,np_))
    for i in range(np_):
        for j in range(i,np_):
            ei=np.zeros(np_);ei[i]=eps;ej=np.zeros(np_);ej[j]=eps
            H[i,j]=(nb_ll(res.x+ei+ej)-nb_ll(res.x+ei-ej)-nb_ll(res.x-ei+ej)+nb_ll(res.x-ei-ej))/(4*eps*eps)
            H[j,i]=H[i,j]
    try: se=np.sqrt(np.diag(np.linalg.inv(H)).clip(min=0))
    except: se=np.full(np_,np.nan)
    se_b=se[:-1]; z=beta/se_b; pv=2*(1-norm.cdf(np.abs(z)))
    irr=np.exp(beta); lo=np.exp(beta-1.96*se_b); hi=np.exp(beta+1.96*se_b)
    p0n=np.zeros(np_);p0n[0]=np.log(y.sum()/np.exp(offset).sum());p0n[-1]=res.x[-1]
    ll_f=-nb_ll(res.x); ll_n=-nb_ll(p0n)
    return {'beta':beta,'se':se_b,'z':z,'p':pv,'irr':irr,'irr_lo':lo,'irr_hi':hi,
            'alpha':alpha,'pseudo_r2':1-(-ll_f)/(-ll_n) if ll_n!=0 else np.nan,'n':n,'converged':res.success}

def pr(r, names, title=""):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"  N={r['n']}, α={r['alpha']:.4f}, PseudoR²={r['pseudo_r2']:.4f}, Conv={r['converged']}")
    print(f"{'='*75}")
    print(f"{'Predictor':<30} {'IRR':>7} {'CI_lo':>7} {'CI_hi':>7} {'z':>7} {'p':>9}")
    print("-"*75)
    for i,nm in enumerate(names):
        s='***' if r['p'][i]<.001 else '**' if r['p'][i]<.01 else '*' if r['p'][i]<.05 else '·' if r['p'][i]<.10 else ''
        print(f"{nm:<30} {r['irr'][i]:>7.4f} {r['irr_lo'][i]:>7.4f} {r['irr_hi'][i]:>7.4f} {r['z'][i]:>7.3f} {r['p'][i]:>9.6f} {s}")

# ============================================================
# DESCRIPTIVE TABLES with corrected assignments
# ============================================================
print("\n" + "#"*75)
print("# CORRECTED DESCRIPTIVE TABLES (micro-average)")
print("#"*75)

for fname, col in [('García Mouton (corrected)', 'mouton_corrected'),
                    ('Fernández-Ordóñez (corrected)', 'fo_corrected')]:
    print(f"\n=== {fname} ===")
    print(f"{'Group':<25} {'Inf':>7} {'Int':>7} {'Gap':>8} {'N':>4}")
    for group in sorted(df[col].dropna().unique()):
        sub = df[df[col]==group]
        inf = sub[sub['role']=='Informant']
        intv = sub[sub['role']=='Interviewer']
        ie = inf['errors'].sum(); iw = inf['N_clean'].sum()
        te = intv['errors'].sum(); tw = intv['N_clean'].sum()
        inf_wer = ie/iw if iw>0 else 0
        int_wer = te/tw if tw>0 else 0
        n_rec = sub['id'].nunique()
        print(f"{group:<25} {inf_wer:>7.4f} {int_wer:>7.4f} {inf_wer-int_wer:>+8.4f} {n_rec:>4}")

# ============================================================
# CORRECTED AUXILIARY MODELS
# ============================================================
df_inf = df[df['role']=='Informant'].copy()
df_int = df[df['role']=='Interviewer'].copy()

print("\n" + "#"*75)
print("# CORRECTED MOUTON MODEL (Informant)")
print("#"*75)

df_m = df_inf[df_inf['mouton_corrected'].notna() & df_inf['snr_db'].notna()].copy()
df_m['snr_c'] = df_m['snr_db'].astype(float) - df_m['snr_db'].astype(float).mean()
df_m['is_male'] = (df_m['speaker_sex']=='Male').astype(float)
df_m['is_south'] = (df_m['mouton_corrected']=='Southern').astype(float)
df_m['age_71_85'] = (df_m['age_cohort']=='71-85').astype(float)
df_m['age_86plus'] = (df_m['age_cohort']=='86+').astype(float)
df_m['age_other'] = (~df_m['age_cohort'].isin(['50-70','71-85','86+'])).astype(float)

X = np.column_stack([np.ones(len(df_m)), df_m['is_south'].values, df_m['snr_c'].values,
                      df_m['is_male'].values, df_m['age_71_85'].values,
                      df_m['age_86plus'].values, df_m['age_other'].values])
y = df_m['errors'].values.astype(float); o = df_m['log_offset'].values.astype(float)
r_m_inf = fit_nb2(y, X, o)
pr(r_m_inf, ['const','Southern','snr_c','sex_male','age71-85','age86+','age_other'],
   "Mouton CORRECTED: Informant ~ Southern + SNR + Sex + Age")

# Mouton Interviewer
df_mi = df_int[df_int['mouton_corrected'].notna() & df_int['snr_db'].notna()].copy()
df_mi['snr_c'] = df_mi['snr_db'].astype(float) - df_mi['snr_db'].astype(float).mean()
df_mi['is_south'] = (df_mi['mouton_corrected']=='Southern').astype(float)

X_i = np.column_stack([np.ones(len(df_mi)), df_mi['is_south'].values, df_mi['snr_c'].values])
y_i = df_mi['errors'].values.astype(float); o_i = df_mi['log_offset'].values.astype(float)
r_m_int = fit_nb2(y_i, X_i, o_i)
pr(r_m_int, ['const','Southern','snr_c'], "Mouton CORRECTED: Interviewer ~ Southern + SNR")

# FO Informant
print("\n" + "#"*75)
print("# CORRECTED FO MODEL (Informant)")
print("#"*75)

df_f = df_inf[df_inf['fo_corrected'].notna() & df_inf['snr_db'].notna()].copy()
df_f['snr_c'] = df_f['snr_db'].astype(float) - df_f['snr_db'].astype(float).mean()
df_f['is_male'] = (df_f['speaker_sex']=='Male').astype(float)
df_f['is_west'] = (df_f['fo_corrected']=='Western').astype(float)
df_f['is_south'] = (df_f['fo_corrected']=='Southern').astype(float)
df_f['is_east'] = (df_f['fo_corrected']=='Eastern').astype(float)
df_f['age_71_85'] = (df_f['age_cohort']=='71-85').astype(float)
df_f['age_86plus'] = (df_f['age_cohort']=='86+').astype(float)
df_f['age_other'] = (~df_f['age_cohort'].isin(['50-70','71-85','86+'])).astype(float)

X_f = np.column_stack([np.ones(len(df_f)), df_f['is_west'].values, df_f['is_south'].values,
                        df_f['is_east'].values, df_f['snr_c'].values, df_f['is_male'].values,
                        df_f['age_71_85'].values, df_f['age_86plus'].values, df_f['age_other'].values])
y_f = df_f['errors'].values.astype(float); o_f = df_f['log_offset'].values.astype(float)
r_f_inf = fit_nb2(y_f, X_f, o_f)
pr(r_f_inf, ['const','Western','Southern','Eastern','snr_c','sex_male','age71-85','age86+','age_other'],
   "FO CORRECTED: Informant ~ FO + SNR + Sex + Age (ref: Centro-Norte)")

# FO Interviewer
df_fi = df_int[df_int['fo_corrected'].notna() & df_int['snr_db'].notna()].copy()
df_fi['snr_c'] = df_fi['snr_db'].astype(float) - df_fi['snr_db'].astype(float).mean()
df_fi['is_west'] = (df_fi['fo_corrected']=='Western').astype(float)
df_fi['is_south'] = (df_fi['fo_corrected']=='Southern').astype(float)
df_fi['is_east'] = (df_fi['fo_corrected']=='Eastern').astype(float)

X_fi = np.column_stack([np.ones(len(df_fi)), df_fi['is_west'].values, df_fi['is_south'].values,
                         df_fi['is_east'].values, df_fi['snr_c'].values])
y_fi = df_fi['errors'].values.astype(float); o_fi = df_fi['log_offset'].values.astype(float)
r_f_int = fit_nb2(y_fi, X_fi, o_fi)
pr(r_f_int, ['const','Western','Southern','Eastern','snr_c'],
   "FO CORRECTED: Interviewer ~ FO + SNR (ref: Centro-Norte)")

# ============================================================
# MIXED-EFFECTS: Cluster-robust SEs by recording
# ============================================================
print("\n" + "#"*75)
print("# CLUSTER-ROBUST SEs (by recording) for CCAA Informant model")
print("#"*75)

# Re-run CCAA model and compute cluster-robust (sandwich) SEs
ref_ccaa = 'Castile and Leon'
all_ccaa = sorted([c for c in df['ccaa_normalized'].unique() if c and c != ref_ccaa])

df_im = df_inf[df_inf['ccaa_normalized'].notna() & (df_inf['ccaa_normalized']!='') & df_inf['snr_db'].notna()].copy()
df_im['snr_c'] = df_im['snr_db'].astype(float) - df_im['snr_db'].astype(float).mean()
df_im['is_male'] = (df_im['speaker_sex']=='Male').astype(float)
df_im['age_71_85'] = (df_im['age_cohort']=='71-85').astype(float)
df_im['age_86plus'] = (df_im['age_cohort']=='86+').astype(float)
df_im['age_other'] = (~df_im['age_cohort'].isin(['50-70','71-85','86+'])).astype(float)
for c in all_ccaa:
    df_im[f'ccaa_{c}'] = (df_im['ccaa_normalized']==c).astype(float)

Xc = [f'ccaa_{c}' for c in all_ccaa] + ['snr_c','is_male','age_71_85','age_86plus','age_other']
X_full = np.column_stack([np.ones(len(df_im))] + [df_im[c].values for c in Xc])
y_full = df_im['errors'].values.astype(float)
o_full = df_im['log_offset'].values.astype(float)

# Get NB fit
r_ccaa = fit_nb2(y_full, X_full, o_full)
beta = r_ccaa['beta']
alpha = r_ccaa['alpha']

# Compute cluster-robust sandwich variance
mu = np.exp(X_full @ beta + o_full)
# Score residuals for NB: (y - mu) / (1 + alpha*mu) * X
w = (y_full - mu) / (1 + alpha * mu)
score_i = X_full * w[:, np.newaxis]  # N x p

# Bread: inverse of Hessian (already have from fit)
# Use numerical Hessian from the fit
n_params = len(beta)
eps = 1e-5
def nb_ll_beta(b):
    ia = 1.0/alpha; mu_ = np.exp(X_full @ b + o_full)
    return -(gammaln(y_full+ia)-gammaln(ia)-gammaln(y_full+1)+ia*np.log(ia/(ia+mu_))+y_full*np.log(mu_/(ia+mu_))).sum()

H = np.zeros((n_params, n_params))
for i in range(n_params):
    for j in range(i, n_params):
        ei=np.zeros(n_params);ei[i]=eps;ej=np.zeros(n_params);ej[j]=eps
        H[i,j]=(nb_ll_beta(beta+ei+ej)-nb_ll_beta(beta+ei-ej)-nb_ll_beta(beta-ei+ej)+nb_ll_beta(beta-ei-ej))/(4*eps*eps)
        H[j,i]=H[i,j]

try:
    bread = np.linalg.inv(H)
except:
    bread = np.eye(n_params)

# Meat: cluster sum of score outer products
recordings = df_im['id'].values
unique_recs = np.unique(recordings)
meat = np.zeros((n_params, n_params))
for rec in unique_recs:
    idx = recordings == rec
    s_cluster = score_i[idx].sum(axis=0)  # sum of scores within cluster
    meat += np.outer(s_cluster, s_cluster)

# Scale factor (small-sample correction)
n_clusters = len(unique_recs)
n_obs = len(y_full)
scale = n_clusters / (n_clusters - 1)

# Sandwich: bread @ meat @ bread
V_cluster = scale * bread @ meat @ bread
se_cluster = np.sqrt(np.diag(V_cluster).clip(min=0))

z_cluster = beta / se_cluster
p_cluster = 2 * (1 - norm.cdf(np.abs(z_cluster)))

pred_names = ['const'] + all_ccaa + ['snr_c','sex_male','age71-85','age86+','age_other']

print(f"\n  N={n_obs}, N_clusters(recordings)={n_clusters}")
print(f"  {'Predictor':<30} {'IRR':>7} {'SE_std':>8} {'SE_clust':>8} {'p_std':>9} {'p_clust':>9}")
print("-"*80)
for i, nm in enumerate(pred_names):
    s_std = '***' if r_ccaa['p'][i]<.001 else '**' if r_ccaa['p'][i]<.01 else '*' if r_ccaa['p'][i]<.05 else '·' if r_ccaa['p'][i]<.10 else ''
    s_cl = '***' if p_cluster[i]<.001 else '**' if p_cluster[i]<.01 else '*' if p_cluster[i]<.05 else '·' if p_cluster[i]<.10 else ''
    print(f"  {nm:<30} {r_ccaa['irr'][i]:>7.4f} {r_ccaa['se'][i]:>8.4f} {se_cluster[i]:>8.4f} {r_ccaa['p'][i]:>9.4f} {s_std:<3} {p_cluster[i]:>9.4f} {s_cl}")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*75}")
print("FINAL SUMMARY")
print(f"{'='*75}")
print(f"""
CORRECTED MOUTON (with age):
  Inf Southern:  IRR={r_m_inf['irr'][1]:.3f} [{r_m_inf['irr_lo'][1]:.3f}, {r_m_inf['irr_hi'][1]:.3f}], p={r_m_inf['p'][1]:.4f}
  Int Southern:  IRR={r_m_int['irr'][1]:.3f} [{r_m_int['irr_lo'][1]:.3f}, {r_m_int['irr_hi'][1]:.3f}], p={r_m_int['p'][1]:.4f}
  Sex (Mouton):  IRR={r_m_inf['irr'][3]:.3f}, p={r_m_inf['p'][3]:.4f}

CORRECTED FO (with age):
  Inf Southern:  IRR={r_f_inf['irr'][2]:.3f} [{r_f_inf['irr_lo'][2]:.3f}, {r_f_inf['irr_hi'][2]:.3f}], p={r_f_inf['p'][2]:.4f}
  Inf Western:   IRR={r_f_inf['irr'][1]:.3f} [{r_f_inf['irr_lo'][1]:.3f}, {r_f_inf['irr_hi'][1]:.3f}], p={r_f_inf['p'][1]:.4f}
  Int Southern:  IRR={r_f_int['irr'][2]:.3f} [{r_f_int['irr_lo'][2]:.3f}, {r_f_int['irr_hi'][2]:.3f}], p={r_f_int['p'][2]:.4f}
  Sex (FO):      IRR={r_f_inf['irr'][5]:.3f}, p={r_f_inf['p'][5]:.4f}

CLUSTER-ROBUST CCAA (key predictors):
  Andalusia:     p_standard={r_ccaa['p'][1]:.4f}, p_cluster={p_cluster[1]:.4f}
  Extremadura:   p_standard={r_ccaa['p'][10]:.4f}, p_cluster={p_cluster[10]:.4f}
  Galicia:       p_standard={r_ccaa['p'][11]:.4f}, p_cluster={p_cluster[11]:.4f}
  Murcia:        p_standard={r_ccaa['p'][14]:.4f}, p_cluster={p_cluster[14]:.4f}
  Sex:           p_standard={r_ccaa['p'][-4]:.4f}, p_cluster={p_cluster[-4]:.4f}
  SNR:           p_standard={r_ccaa['p'][-5]:.4f}, p_cluster={p_cluster[-5]:.4f}
""")

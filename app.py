# app.py (updated with detailed compute_matrices and proses route)
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import numpy as np
import os
import re

# ---------- Config ----------
APP_DB = 'spk_results.db'
EXCEL_FILES = ['data_hasil_gabungan.xlsx', 'data_hasil_gabungan.xls']
DEFAULT_ALPHA = 0.5
WEIGHT_SUM_TOL = 0.001

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{APP_DB}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------- DB Model ----------
class Production(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    produksi = db.Column(db.String(256))
    provinsi = db.Column(db.String(128))
    volume = db.Column(db.Float)
    nilai = db.Column(db.Float)
    harga = db.Column(db.Float)
    saw_score = db.Column(db.Float)
    topsis_score = db.Column(db.Float)
    composite = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ---------- Utilities ----------
def find_excel_file():
    for name in EXCEL_FILES:
        if os.path.exists(name):
            return name
    return None

def to_numeric_safe(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == '':
        return None
    s = s.replace(' ', '')
    if s.count(',') > 1 and s.count('.') == 0:
        s = s.replace(',', '')
    elif s.count('.') > 1 and s.count(',') == 0:
        s = s.replace('.', '')
    if s.count(',') == 1 and s.count('.') == 0:
        s = s.replace(',', '.')
    cleaned = re.sub(r'[^\d.\-]', '', s)
    try:
        return float(cleaned) if cleaned != '' else None
    except:
        return None

def load_and_prepare_dataframe():
    fname = find_excel_file()
    if not fname:
        raise FileNotFoundError("File Excel data_hasil_gabungan.xlsx tidak ditemukan di folder proyek.")
    df = pd.read_excel(fname, header=0, dtype=str)
    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols

    mapping = {}
    for c in cols:
        if 'provinsi' in c:
            mapping['provinsi'] = c
        elif 'produksi' in c or 'alternatif' in c or 'nama' in c:
            mapping['produksi'] = c
        elif 'volume' in c or 'ekor' in c:
            mapping['volume'] = c
        elif ('nilai' in c) or ('rp' in c and ('juta' in c or 'nilai' in c)):
            mapping['nilai'] = c
        elif 'harga' in c or 'tertimbang' in c:
            mapping['harga'] = c

    if len(cols) >= 4 and (set(['provinsi','volume','nilai','harga']) - set(mapping.keys())):
        mapping.setdefault('provinsi', cols[0])
        mapping.setdefault('volume', cols[1])
        mapping.setdefault('nilai', cols[2])
        mapping.setdefault('harga', cols[3])
        if 'produksi' not in mapping:
            df['produksi'] = df[mapping['provinsi']].apply(lambda v: 'Produksi')
            mapping['produksi'] = 'produksi'

    for key in ['produksi','provinsi','volume','nilai','harga']:
        if key not in mapping:
            if key == 'produksi':
                if 'produksi' not in df.columns:
                    df['produksi'] = df.index.astype(str).apply(lambda i: f'Alt-{i+1}')
                mapping['produksi'] = 'produksi'
            else:
                raise ValueError(f"Tidak bisa menemukan kolom untuk '{key}' di Excel. Kolom yang tersedia: {cols}")

    out = pd.DataFrame()
    out['produksi'] = df[mapping['produksi']].astype(str).str.strip()
    out['provinsi'] = df[mapping['provinsi']].astype(str).str.strip()
    out['volume'] = df[mapping['volume']].apply(to_numeric_safe).astype(float)
    out['nilai'] = df[mapping['nilai']].apply(to_numeric_safe).astype(float)
    out['harga'] = df[mapping['harga']].apply(to_numeric_safe).astype(float)
    out = out.dropna(subset=['volume','nilai','harga']).reset_index(drop=True)
    return out

# ---------- Calculation helpers ----------
def compute_saw_basic(vals, weights):
    # vals: numpy array shape (n,3)
    maxs = vals.max(axis=0)
    maxs[maxs==0] = 1.0
    saw_norm = vals / maxs  # r_ij for SAW
    saw_scores = (saw_norm * weights).sum(axis=1)
    return saw_norm, saw_scores

def compute_topsis_basic(vals, weights):
    denom = np.sqrt((vals**2).sum(axis=0))
    denom[denom==0] = 1.0
    r = vals / denom
    v = r * weights
    v_pos = v.max(axis=0)
    v_neg = v.min(axis=0)
    d_pos = np.sqrt(((v - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - v_neg) ** 2).sum(axis=1))
    topsis_scores = d_neg / (d_pos + d_neg + 1e-12)
    return r, v, v_pos, v_neg, d_pos, d_neg, topsis_scores

def compute_matrices_for_df(df, weights, alpha=DEFAULT_ALPHA):
    """
    df: pandas DataFrame with columns ['produksi','provinsi','volume','nilai','harga']
    weights: list-like [w1,w2,w3]
    returns dict of matrices/lists ready to pass to template
    """
    # ensure order
    cols = ['volume','nilai','harga']
    vals = df[cols].values.astype(float)  # shape (n,3)
    w = np.array(weights).astype(float)

    # SAW
    saw_norm, saw_scores = compute_saw_basic(vals, w)

    # TOPSIS
    r, v, v_pos, v_neg, d_pos, d_neg, topsis_scores = compute_topsis_basic(vals, w)

    # Composite
    composite = alpha * saw_scores + (1.0 - alpha) * topsis_scores

    # prepare headers and rows for template (lists)
    index = df['produksi'].tolist()
    provs = df['provinsi'].tolist()
    headers = cols  # ['volume','nilai','harga']

    # convert to python lists rounded for display
    saw_norm_list = saw_norm.tolist()
    saw_scores_list = saw_scores.tolist()
    r_list = r.tolist()
    v_list = v.tolist()
    v_pos_list = v_pos.tolist()
    v_neg_list = v_neg.tolist()
    d_pos_list = d_pos.tolist()
    d_neg_list = d_neg.tolist()
    topsis_scores_list = topsis_scores.tolist()
    composite_list = composite.tolist()

    return {
        'index': index,
        'provinsi': provs,
        'headers': headers,
        'saw_norm': saw_norm_list,
        'saw_scores': saw_scores_list,
        'topsis_r': r_list,
        'topsis_v': v_list,
        'v_pos': v_pos_list,
        'v_neg': v_neg_list,
        'd_pos': d_pos_list,
        'd_neg': d_neg_list,
        'topsis_scores': topsis_scores_list,
        'composite': composite_list
    }

# ---------- existing compute & persist (unchanged) ----------
def compute_saw(df, weights):
    vals = df[['volume','nilai','harga']].values.astype(float)
    maxs = vals.max(axis=0)
    maxs[maxs==0] = 1.0
    norm = vals / maxs
    w = np.array(weights)
    saw_scores = (norm * w).sum(axis=1)
    return saw_scores, norm

def compute_topsis(df, weights):
    vals = df[['volume','nilai','harga']].values.astype(float)
    denom = np.sqrt((vals**2).sum(axis=0))
    denom[denom==0] = 1.0
    r = vals / denom
    w = np.array(weights)
    v = r * w
    v_pos = v.max(axis=0)
    v_neg = v.min(axis=0)
    d_pos = np.sqrt(((v - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - v_neg) ** 2).sum(axis=1))
    topsis_scores = d_neg / (d_pos + d_neg + 1e-12)
    return topsis_scores, v, v_pos, v_neg, r

def compute_and_persist(weights=None, alpha=DEFAULT_ALPHA):
    if weights is None:
        weights = [0.4, 0.4, 0.2]
    df = load_and_prepare_dataframe()
    saw_scores, saw_norm = compute_saw(df, weights)
    topsis_scores, v, v_pos, v_neg, r = compute_topsis(df, weights)
    composite = alpha * saw_scores + (1.0 - alpha) * topsis_scores

    Production.query.delete()
    db.session.commit()

    now = datetime.utcnow()
    rows = []
    for i, row in df.iterrows():
        p = Production(
            produksi = row['produksi'],
            provinsi = row['provinsi'],
            volume = float(row['volume']),
            nilai = float(row['nilai']),
            harga = float(row['harga']),
            saw_score = float(saw_scores[i]),
            topsis_score = float(topsis_scores[i]),
            composite = float(composite[i]),
            created_at = now
        )
        rows.append(p)
    db.session.bulk_save_objects(rows)
    db.session.commit()
    return len(rows)

def compute_and_persist_custom(weights, alpha=DEFAULT_ALPHA):
    weights = [float(w) for w in weights]
    return compute_and_persist(weights=weights, alpha=alpha)

# ---------- Routes ----------
@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    uploaded_template = '/mnt/data/template spk.docx'
    return render_template('dashboard.html', uploaded_template=uploaded_template)

@app.route('/index')
def index():
    try:
        df = load_and_prepare_dataframe()
        prov_list = sorted(df['provinsi'].unique().tolist())
    except Exception as e:
        prov_list = []
        flash(str(e), 'error')
    weights = session.get('weights', [0.4, 0.4, 0.2])
    return render_template('index.html', prov_list=prov_list, weights=weights)

@app.route('/compute', methods=['POST'])
def compute():
    prov_selected = request.form.getlist('provinsi')
    prov_selected = [p.strip() for p in prov_selected if p.strip() != '']
    if len(prov_selected) == 0:
        flash('Silakan pilih minimal satu provinsi (atau Pilih Semua).', 'error')
        return redirect(url_for('index'))

    try:
        w1 = float(request.form.get('w1', '0.0'))
        w2 = float(request.form.get('w2', '0.0'))
        w3 = float(request.form.get('w3', '0.0'))
    except ValueError:
        flash('Bobot tidak valid. Pastikan Anda memasukkan angka desimal, mis. 0.4', 'error')
        return redirect(url_for('index'))

    total = w1 + w2 + w3
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        flash(f'Bobot harus berjumlah 1. Saat ini jumlah = {total:.4f}. Silakan sesuaikan sehingga total = 1.', 'error')
        return redirect(url_for('index'))

    w1, w2, w3 = w1/total, w2/total, w3/total

    session['provinsi'] = prov_selected
    session['weights'] = [w1, w2, w3]

    try:
        compute_and_persist_custom(weights=[w1, w2, w3])
        flash('Perhitungan SAW+TOPSIS selesai dan tersimpan. Silakan lanjutkan ke penjelasan.', 'success')
    except Exception as e:
        flash(f'Gagal menghitung: {e}', 'error')
        return redirect(url_for('index'))

    return redirect(url_for('proses'))

@app.route('/proses')
def proses():
    # load full df then filter by selected provinsi(s)
    df = load_and_prepare_dataframe()
    provs = session.get('provinsi', [])
    if provs and len(provs) > 0:
        # filter rows whose provinsi contains any of selected
        filt = np.zeros(len(df), dtype=bool)
        for p in provs:
            filt = filt | df['provinsi'].str.contains(p, case=False, na=False)
        df_sel = df[filt].reset_index(drop=True)
    else:
        df_sel = df.copy()

    weights = session.get('weights', [0.4, 0.4, 0.2])
    # compute matrices for selected df
    matrices = compute_matrices_for_df(df_sel, weights, alpha=DEFAULT_ALPHA)

    # also pass simple formulas for display
    formulas = {
        'saw_norm': 'r_{ij} = x_{ij} / max_j(x_{ij})',
        'saw_score': 'S_i = Σ_j (w_j * r_{ij})',
        'topsis_norm': 'r_{ij} = x_{ij} / sqrt(Σ_i x_{ij}^2)',
        'topsis_weighted': 'v_{ij} = w_j * r_{ij}',
        'topsis_distance': "D_i^+ = sqrt(Σ_j (v_{ij} - v_j^+)^2),  D_i^- = sqrt(Σ_j (v_{ij} - v_j^-)^2)",
        'topsis_score': 'C_i = D_i^- / (D_i^+ + D_i^-)',
        'hybrid': 'Composite = α * SAW + (1-α) * TOPSIS'
    }

    return render_template('proses.html',
                           matrices=matrices,
                           formulas=formulas,
                           weights=weights,
                           provs=provs)

@app.route('/results')
def results():
    provs = session.get('provinsi', None)
    if Production.query.count() == 0:
        saved_weights = session.get('weights', [0.4,0.4,0.2])
        compute_and_persist_custom(weights=saved_weights)

    q = Production.query
    if provs and len(provs) > 0:
        from sqlalchemy import or_
        filters = [Production.provinsi.ilike(f"%{p}%") for p in provs]
        q = q.filter(or_(*filters))
        current_label = ', '.join(provs)
    else:
        current_label = 'SEMUA'

    items = q.order_by(Production.composite.desc()).all()
    ready_to_show = True
    return render_template('results.html', items=items, provinsi=current_label, ready=ready_to_show)

@app.route('/recompute', methods=['POST'])
def recompute():
    weights = session.get('weights', [0.4,0.4,0.2])
    compute_and_persist_custom(weights=weights)
    flash('Recompute selesai dengan bobot saat ini.', 'success')
    return redirect(url_for('results'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

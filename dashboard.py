import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import poisson, norm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

.main-header {
    background: linear-gradient(135deg, #0f1729 0%, #1a2540 50%, #0f1729 100%);
    border: 1px solid #2a3a5c;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    color: #63b3ed;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: #718096;
    margin: 8px 0 0;
    font-size: 0.9rem;
}

.badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-right: 8px;
}

.metric-card {
    background: #111827;
    border: 1px solid #1f2d47;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}

.metric-card:hover {
    border-color: #3d5a8a;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #63b3ed;
}

.metric-label {
    font-size: 0.78rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #a0aec0;
    border-left: 3px solid #63b3ed;
    padding-left: 12px;
    margin: 28px 0 16px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-box {
    background: linear-gradient(135deg, #0d1f3c, #111827);
    border: 1px solid #2d4a7a;
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
}

.result-box .label {
    font-size: 0.8rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-box .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: #68d391;
    font-weight: 700;
    margin-top: 4px;
}

.info-pill {
    background: rgba(104,211,145,0.1);
    border: 1px solid rgba(104,211,145,0.25);
    color: #68d391;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    display: inline-block;
    margin: 4px 4px 0 0;
}

.warn-pill {
    background: rgba(252,211,77,0.1);
    border: 1px solid rgba(252,211,77,0.25);
    color: #fcd34d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    display: inline-block;
}

.sidebar-section {
    background: #111827;
    border: 1px solid #1f2d47;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
}

[data-testid="stSidebar"] {
    background: #080d18 !important;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #a0aec0;
}

.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #718096;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    padding: 8px 16px;
}

.stTabs [aria-selected="true"] {
    background: #1a2f4e !important;
    color: #63b3ed !important;
}

.stSlider > div > div > div > div {
    background: #63b3ed !important;
}

.stSelectbox > div > div {
    background: #111827 !important;
    border-color: #2d4a7a !important;
    color: #e8eaf0 !important;
}

div[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1f2d47;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA HISTORIS (Produk: Minuman Kemasan)
# ─────────────────────────────────────────────
np.random.seed(42)
n_periods = 52  # 52 minggu

# Simulasi demand historis mingguan produk minuman
demand_history = np.array([
    45,52,38,61,55,42,70,48,53,39,
    66,58,44,71,49,57,41,63,50,55,
    47,68,43,59,52,46,74,51,56,40,
    65,54,48,72,43,60,55,49,67,53,
    41,70,46,58,52,44,69,50,57,42,
    65,56
])

demand_df = pd.DataFrame({
    'Minggu': range(1, n_periods+1),
    'Demand': demand_history
})

# Parameter produk
HARGA_JUAL = 15000      # Rp per unit
HARGA_BELI = 8000       # Rp per unit
BIAYA_SIMPAN = 500      # Rp per unit per periode
BIAYA_KEKURANGAN = 3000 # Rp per unit (opportunity cost + penalty)

ORDER_ALTERNATIVES = [40, 50, 60, 70, 80]  # pilihan order quantity

# State demand
DEMAND_LOW = 42
DEMAND_NORMAL = 55
DEMAND_HIGH = 70

# ─────────────────────────────────────────────
# HELPER: PAYOFF CALCULATION
# ─────────────────────────────────────────────
def hitung_payoff(order_qty, demand):
    """Hitung profit dari kombinasi order & demand."""
    terjual = min(order_qty, demand)
    sisa = max(0, order_qty - demand)
    kekurangan = max(0, demand - order_qty)
    
    revenue = terjual * HARGA_JUAL
    cogs = order_qty * HARGA_BELI
    biaya_simpan = sisa * BIAYA_SIMPAN
    biaya_kurang = kekurangan * BIAYA_KEKURANGAN
    
    profit = revenue - cogs - biaya_simpan - biaya_kurang
    return profit

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Space Mono,monospace; color:#63b3ed; font-size:1rem; padding:10px 0 16px;'>
    📦 PARAM PRODUK
    </div>
    """, unsafe_allow_html=True)
    
    harga_jual = st.slider("Harga Jual (Rp)", 10000, 25000, HARGA_JUAL, 500, format="Rp%d")
    harga_beli = st.slider("Harga Beli / HPP (Rp)", 4000, 15000, HARGA_BELI, 500, format="Rp%d")
    biaya_simpan = st.slider("Biaya Simpan/unit (Rp)", 100, 2000, BIAYA_SIMPAN, 100, format="Rp%d")
    biaya_kurang = st.slider("Biaya Kekurangan/unit (Rp)", 500, 8000, BIAYA_KEKURANGAN, 500, format="Rp%d")
    
    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace; color:#63b3ed; font-size:0.85rem; padding:6px 0 12px;'>
    🎲 MONTE CARLO
    </div>
    """, unsafe_allow_html=True)
    n_simulasi = st.slider("Jumlah Simulasi", 100, 2000, 500, 100)
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#4a5568; text-align:center; padding-top:8px;'>
    Data: Demand Minuman Kemasan<br>52 Minggu Historis
    </div>
    """, unsafe_allow_html=True)

# Update parameter dari sidebar
HARGA_JUAL = harga_jual
HARGA_BELI = harga_beli
BIAYA_SIMPAN = biaya_simpan
BIAYA_KEKURANGAN = biaya_kurang

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📦 Decision Analysis Dashboard</h1>
    <p>Inventory Order Quantity · Metode Pengambilan Keputusan di Bawah Kondisi Ketidakpastian</p>
    <div style="margin-top:14px">
        <span class="badge">CERTAINTY</span>
        <span class="badge">RISK</span>
        <span class="badge">UNCERTAINTY</span>
        <span class="badge">PROB. MODELING</span>
        <span class="badge">UTILITY</span>
        <span class="badge">MONTE CARLO</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RINGKASAN DATA
# ─────────────────────────────────────────────
mean_d = demand_history.mean()
std_d  = demand_history.std()
med_d  = np.median(demand_history)
cv     = std_d / mean_d * 100

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, lbl in zip(
    [c1,c2,c3,c4,c5],
    [f"{mean_d:.1f}", f"{std_d:.1f}", f"{med_d:.0f}", f"{demand_history.min()}", f"{demand_history.max()}"],
    ["Rata-rata Demand", "Std. Deviasi", "Median", "Min Demand", "Max Demand"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Historis",
    "1️⃣ Certainty",
    "2️⃣ Risk (EV)",
    "3️⃣ Uncertainty",
    "4️⃣ Prob. Modeling",
    "5️⃣ Utility",
    "6️⃣ Monte Carlo"
])

# ══════════════════════════════════════════════
# TAB 0: DATA HISTORIS
# ══════════════════════════════════════════════
with tab0:
    st.markdown('<div class="section-header">Data Demand Historis (52 Minggu)</div>', unsafe_allow_html=True)
    
    col_l, col_r = st.columns([2,1])
    
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=demand_df['Minggu'], y=demand_df['Demand'],
            mode='lines+markers',
            line=dict(color='#63b3ed', width=2),
            marker=dict(size=5, color='#63b3ed'),
            fill='tozeroy',
            fillcolor='rgba(99,179,237,0.07)',
            name='Demand'
        ))
        fig.add_hline(y=mean_d, line_dash='dash', line_color='#f6ad55',
                      annotation_text=f"Mean = {mean_d:.1f}")
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#111827',
            plot_bgcolor='#111827',
            margin=dict(l=10,r=10,t=10,b=10),
            height=320,
            xaxis=dict(gridcolor='#1f2d47', title='Minggu ke-'),
            yaxis=dict(gridcolor='#1f2d47', title='Unit'),
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_r:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=demand_history, nbinsx=12,
            marker_color='#63b3ed', opacity=0.8,
            name='Freq'
        ))
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='#111827',
            plot_bgcolor='#111827',
            margin=dict(l=10,r=10,t=20,b=10),
            height=320,
            title=dict(text='Distribusi Demand', font=dict(size=12, color='#a0aec0')),
            xaxis=dict(gridcolor='#1f2d47', title='Unit'),
            yaxis=dict(gridcolor='#1f2d47', title='Frekuensi'),
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown('<div class="section-header">Statistik Deskriptif</div>', unsafe_allow_html=True)
    desc = pd.DataFrame({
        'Metrik': ['Mean','Median','Std Dev','Variance','Min','Max','Skewness','Kurtosis','CV (%)'],
        'Nilai': [
            f"{mean_d:.2f}", f"{med_d:.2f}", f"{std_d:.2f}",
            f"{std_d**2:.2f}", f"{demand_history.min()}", f"{demand_history.max()}",
            f"{stats.skew(demand_history):.3f}", f"{stats.kurtosis(demand_history):.3f}",
            f"{cv:.1f}%"
        ]
    })
    st.dataframe(desc, hide_index=True, use_container_width=False)

# ══════════════════════════════════════════════
# TAB 1: CERTAINTY
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Metode 1: Certainty — Demand = Rata-rata Historis</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="result-box">
        <div class="label">Asumsi</div>
        <div style="color:#a0aec0; margin-top:6px; font-size:0.9rem;">
        Demand diasumsikan <b style="color:#63b3ed;">pasti</b> = rata-rata historis = <b style="color:#f6ad55;">{mean_d:.0f} unit</b>.
        Kita hanya menghitung payoff untuk setiap alternatif order, lalu pilih yang terbaik.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    demand_pasti = round(mean_d)
    cert_data = []
    for q in ORDER_ALTERNATIVES:
        profit = hitung_payoff(q, demand_pasti)
        terjual = min(q, demand_pasti)
        sisa = max(0, q - demand_pasti)
        kurang = max(0, demand_pasti - q)
        cert_data.append({
            'Order Qty': q,
            'Demand (Pasti)': demand_pasti,
            'Unit Terjual': terjual,
            'Sisa Stok': sisa,
            'Kekurangan': kurang,
            'Profit (Rp)': profit
        })
    
    cert_df = pd.DataFrame(cert_data)
    best_idx = cert_df['Profit (Rp)'].idxmax()
    best_q = cert_df.loc[best_idx, 'Order Qty']
    best_p = cert_df.loc[best_idx, 'Profit (Rp)']
    
    col_a, col_b = st.columns([3,2])
    with col_a:
        fig = go.Figure()
        colors = ['#68d391' if i == best_idx else '#63b3ed' for i in range(len(ORDER_ALTERNATIVES))]
        fig.add_trace(go.Bar(
            x=[str(q) for q in ORDER_ALTERNATIVES],
            y=cert_df['Profit (Rp)'],
            marker_color=colors,
            text=[f"Rp{p:,.0f}" for p in cert_df['Profit (Rp)']],
            textposition='outside',
            textfont=dict(size=11, color='#e8eaf0')
        ))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827',
            height=320, margin=dict(l=10,r=10,t=20,b=10),
            xaxis=dict(title='Order Quantity', gridcolor='#1f2d47'),
            yaxis=dict(title='Profit (Rp)', gridcolor='#1f2d47'),
            title=dict(text='Payoff per Alternatif (Certainty)', font=dict(color='#a0aec0', size=12))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown(f"""
        <div class="result-box" style="margin-top:16px">
            <div class="label">✅ Keputusan Optimal</div>
            <div class="value">Order = {best_q} unit</div>
            <div style="color:#a0aec0; margin-top:8px; font-size:0.85rem;">Profit: Rp{best_p:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Detail Payoff:**")
        st.dataframe(
            cert_df[['Order Qty','Unit Terjual','Sisa Stok','Kekurangan','Profit (Rp)']],
            hide_index=True, use_container_width=True
        )

# ══════════════════════════════════════════════
# TAB 2: RISK (EV)
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Metode 2: Risk — Expected Value (EV)</div>', unsafe_allow_html=True)
    
    # Hitung probabilitas empiris dari data historis
    low_count  = np.sum(demand_history <= 46)
    norm_count = np.sum((demand_history > 46) & (demand_history <= 62))
    high_count = np.sum(demand_history > 62)
    total = len(demand_history)
    
    p_low  = low_count / total
    p_norm = norm_count / total
    p_high = high_count / total
    
    st.markdown(f"""
    <div class="result-box">
        <div class="label">Probabilitas Empiris (dari {total} observasi)</div>
        <div style="margin-top:10px; display:flex; gap:16px; flex-wrap:wrap;">
            <span class="warn-pill">Rendah (≤46) : {p_low:.2%} ({low_count}x)</span>
            <span class="info-pill">Normal (47-62) : {p_norm:.2%} ({norm_count}x)</span>
            <span style="background:rgba(252,129,129,0.1);border:1px solid rgba(252,129,129,0.3);color:#fc8181;border-radius:20px;padding:4px 14px;font-size:0.8rem;display:inline-block;">Tinggi (>62) : {p_high:.2%} ({high_count}x)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Payoff matrix + EV
    states = [DEMAND_LOW, DEMAND_NORMAL, DEMAND_HIGH]
    probs  = [p_low, p_norm, p_high]
    state_labels = [f"Rendah ({DEMAND_LOW})", f"Normal ({DEMAND_NORMAL})", f"Tinggi ({DEMAND_HIGH})"]
    
    risk_data = []
    for q in ORDER_ALTERNATIVES:
        row = {'Order Qty': q}
        ev = 0
        for s, p, lbl in zip(states, probs, state_labels):
            payoff = hitung_payoff(q, s)
            row[lbl] = payoff
            ev += p * payoff
        row['EV (Rp)'] = ev
        risk_data.append(row)
    
    risk_df = pd.DataFrame(risk_data)
    best_ev_idx = risk_df['EV (Rp)'].idxmax()
    best_ev_q   = risk_df.loc[best_ev_idx, 'Order Qty']
    best_ev_val = risk_df.loc[best_ev_idx, 'EV (Rp)']
    
    col_a, col_b = st.columns([3,2])
    with col_a:
        fig = go.Figure()
        for lbl, color in zip(state_labels, ['#fc8181','#63b3ed','#68d391']):
            fig.add_trace(go.Bar(
                name=lbl,
                x=[str(q) for q in ORDER_ALTERNATIVES],
                y=risk_df[lbl],
                marker_color=color,
                opacity=0.7
            ))
        fig.add_trace(go.Scatter(
            name='EV',
            x=[str(q) for q in ORDER_ALTERNATIVES],
            y=risk_df['EV (Rp)'],
            mode='lines+markers',
            line=dict(color='#fcd34d', width=3),
            marker=dict(size=8, color='#fcd34d')
        ))
        fig.update_layout(
            barmode='group', template='plotly_dark',
            paper_bgcolor='#111827', plot_bgcolor='#111827',
            height=330, margin=dict(l=10,r=10,t=30,b=10),
            xaxis=dict(title='Order Qty', gridcolor='#1f2d47'),
            yaxis=dict(title='Payoff (Rp)', gridcolor='#1f2d47'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            title=dict(text='Payoff per State + Expected Value', font=dict(color='#a0aec0', size=12))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown(f"""
        <div class="result-box" style="margin-top:16px">
            <div class="label">✅ Keputusan: Max EV</div>
            <div class="value">Order = {best_ev_q} unit</div>
            <div style="color:#a0aec0; margin-top:8px; font-size:0.85rem;">EV = Rp{best_ev_val:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("**Payoff Matrix:**")
    fmt_df = risk_df.copy()
    for col in [c for c in fmt_df.columns if '(Rp)' in c or 'Rp' in c or 'Normal' in c or 'Rendah' in c or 'Tinggi' in c]:
        if fmt_df[col].dtype != object:
            fmt_df[col] = fmt_df[col].apply(lambda x: f"Rp{x:,.0f}")
    st.dataframe(fmt_df, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3: UNCERTAINTY
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Metode 3: Uncertainty — Tanpa Probabilitas</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-box">
        <div class="label">Kriteria yang Digunakan</div>
        <div style="color:#a0aec0; margin-top:8px; font-size:0.85rem; line-height:1.8">
        <b style="color:#63b3ed;">Maximax</b> — Optimis: pilih max dari payoff tertinggi tiap alternatif<br>
        <b style="color:#68d391;">Maximin</b> — Pesimis: pilih max dari payoff terendah (worst-case)<br>
        <b style="color:#f6ad55;">Minimax Regret</b> — Minimalkan penyesalan terbesar<br>
        <b style="color:#fc8181;">Equally Likely (Laplace)</b> — Semua state sama probable, pilih avg tertinggi
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    states = [DEMAND_LOW, DEMAND_NORMAL, DEMAND_HIGH]
    state_labels = [f"Rendah ({DEMAND_LOW})", f"Normal ({DEMAND_NORMAL})", f"Tinggi ({DEMAND_HIGH})"]
    
    # Payoff matrix
    payoff_matrix = np.array([[hitung_payoff(q, s) for s in states] for q in ORDER_ALTERNATIVES])
    
    # Maximax
    maximax_vals = payoff_matrix.max(axis=1)
    maximax_best = ORDER_ALTERNATIVES[maximax_vals.argmax()]
    
    # Maximin
    maximin_vals = payoff_matrix.min(axis=1)
    maximin_best = ORDER_ALTERNATIVES[maximin_vals.argmax()]
    
    # Laplace
    laplace_vals = payoff_matrix.mean(axis=1)
    laplace_best = ORDER_ALTERNATIVES[laplace_vals.argmax()]
    
    # Minimax Regret
    max_per_state = payoff_matrix.max(axis=0)
    regret_matrix = max_per_state - payoff_matrix
    max_regret = regret_matrix.max(axis=1)
    minimax_best = ORDER_ALTERNATIVES[max_regret.argmin()]
    
    col_a, col_b = st.columns([2,2])
    with col_a:
        fig = go.Figure()
        x_labels = [str(q) for q in ORDER_ALTERNATIVES]
        fig.add_trace(go.Scatter(x=x_labels, y=maximax_vals, name='Maximax', 
                                  line=dict(color='#63b3ed', width=2), mode='lines+markers'))
        fig.add_trace(go.Scatter(x=x_labels, y=maximin_vals, name='Maximin',
                                  line=dict(color='#68d391', width=2), mode='lines+markers'))
        fig.add_trace(go.Scatter(x=x_labels, y=laplace_vals, name='Laplace',
                                  line=dict(color='#f6ad55', width=2), mode='lines+markers'))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827',
            height=300, margin=dict(l=10,r=10,t=30,b=10),
            xaxis=dict(title='Order Qty', gridcolor='#1f2d47'),
            yaxis=dict(title='Payoff (Rp)', gridcolor='#1f2d47'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            title=dict(text='Perbandingan Kriteria', font=dict(color='#a0aec0', size=12))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=x_labels, y=max_regret,
            marker_color='#fc8181', opacity=0.8,
            text=[f"Rp{r:,.0f}" for r in max_regret],
            textposition='outside', textfont=dict(size=10)
        ))
        fig2.update_layout(
            template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827',
            height=300, margin=dict(l=10,r=10,t=30,b=10),
            xaxis=dict(title='Order Qty', gridcolor='#1f2d47'),
            yaxis=dict(title='Max Regret (Rp)', gridcolor='#1f2d47'),
            title=dict(text='Minimax Regret', font=dict(color='#a0aec0', size=12))
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Summary
    c1,c2,c3,c4 = st.columns(4)
    for col, name, val, color in zip(
        [c1,c2,c3,c4],
        ["Maximax","Maximin","Laplace","Minimax Regret"],
        [maximax_best, maximin_best, laplace_best, minimax_best],
        ["#63b3ed","#68d391","#f6ad55","#fc8181"]
    ):
        col.markdown(f"""
        <div class="result-box">
            <div class="label">{name}</div>
            <div class="value" style="color:{color};">Order = {val}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4: PROBABILISTIC MODELING
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Metode 4: Probabilistic Modeling — Fit Distribusi</div>', unsafe_allow_html=True)
    
    # Fit distribusi
    mu_norm, std_norm = norm.fit(demand_history)
    lambda_pois = demand_history.mean()
    
    # Goodness of fit
    ks_norm = stats.kstest(demand_history, 'norm', args=(mu_norm, std_norm))
    
    # Pilihan distribusi
    dist_choice = st.radio("Pilih Distribusi:", ["Normal", "Poisson"], horizontal=True)
    
    col_a, col_b = st.columns([3,2])
    with col_a:
        x_range = np.linspace(demand_history.min()-5, demand_history.max()+5, 200)
        
        fig = go.Figure()
        # Histogram normalized
        fig.add_trace(go.Histogram(
            x=demand_history, nbinsx=14,
            histnorm='probability density',
            marker_color='rgba(99,179,237,0.4)',
            marker_line=dict(color='#63b3ed', width=1),
            name='Data Historis'
        ))
        
        if dist_choice == "Normal":
            pdf_vals = norm.pdf(x_range, mu_norm, std_norm)
            fig.add_trace(go.Scatter(x=x_range, y=pdf_vals, mode='lines',
                line=dict(color='#68d391', width=3), name=f'Normal(μ={mu_norm:.1f}, σ={std_norm:.1f})'))
        else:
            x_pois = np.arange(demand_history.min(), demand_history.max()+1)
            pmf_pois = poisson.pmf(x_pois, lambda_pois)
            fig.add_trace(go.Bar(x=x_pois, y=pmf_pois, opacity=0.5,
                marker_color='#f6ad55', name=f'Poisson(λ={lambda_pois:.1f})'))
        
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827',
            height=320, margin=dict(l=10,r=10,t=20,b=10),
            xaxis=dict(title='Demand (unit)', gridcolor='#1f2d47'),
            yaxis=dict(title='Density', gridcolor='#1f2d47'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            title=dict(text=f'Fit Distribusi {dist_choice}', font=dict(color='#a0aec0', size=12))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown(f"""
        <div class="result-box">
            <div class="label">Parameter Distribusi Normal</div>
            <div style="color:#e8eaf0; margin-top:8px; font-size:0.9rem; font-family:Space Mono,monospace;">
            μ = {mu_norm:.2f}<br>
            σ = {std_norm:.2f}<br>
            KS-stat = {ks_norm.statistic:.4f}<br>
            p-value = {ks_norm.pvalue:.4f}
            </div>
        </div>
        <div class="result-box">
            <div class="label">Parameter Distribusi Poisson</div>
            <div style="color:#e8eaf0; margin-top:8px; font-size:0.9rem; font-family:Space Mono,monospace;">
            λ = {lambda_pois:.2f}<br>
            E[X] = {lambda_pois:.2f}<br>
            Var[X] = {lambda_pois:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        note_color = "#68d391" if ks_norm.pvalue > 0.05 else "#fc8181"
        note_txt = "✅ Distribusi Normal diterima (p > 0.05)" if ks_norm.pvalue > 0.05 else "⚠️ Distribusi Normal ditolak (p ≤ 0.05)"
        st.markdown(f'<span style="color:{note_color}; font-size:0.82rem;">{note_txt}</span>', unsafe_allow_html=True)
    
    # EV dengan distribusi
    st.markdown('<div class="section-header">Expected Profit dengan Sampling Distribusi</div>', unsafe_allow_html=True)
    
    n_samples = 1000
    if dist_choice == "Normal":
        demand_samples = norm.rvs(mu_norm, std_norm, size=n_samples)
        demand_samples = np.clip(demand_samples, 0, None).round().astype(int)
    else:
        demand_samples = poisson.rvs(lambda_pois, size=n_samples)
    
    prob_data = []
    for q in ORDER_ALTERNATIVES:
        profits = [hitung_payoff(q, int(d)) for d in demand_samples]
        prob_data.append({
            'Order Qty': q,
            'EV (Rp)': np.mean(profits),
            'Std Dev': np.std(profits),
            'P5 (Rp)': np.percentile(profits, 5),
            'P95 (Rp)': np.percentile(profits, 95)
        })
    
    prob_df = pd.DataFrame(prob_data)
    best_prob = prob_df.loc[prob_df['EV (Rp)'].idxmax(), 'Order Qty']
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=[str(q) for q in ORDER_ALTERNATIVES],
        y=prob_df['EV (Rp)'],
        error_y=dict(type='data', array=prob_df['Std Dev'], visible=True, color='rgba(99,179,237,0.4)'),
        mode='lines+markers',
        line=dict(color='#63b3ed', width=3),
        marker=dict(size=10),
        name='EV ± StdDev'
    ))
    fig3.add_trace(go.Scatter(
        x=[str(q) for q in ORDER_ALTERNATIVES] + [str(q) for q in ORDER_ALTERNATIVES][::-1],
        y=list(prob_df['P95 (Rp)']) + list(prob_df['P5 (Rp)'])[::-1],
        fill='toself', fillcolor='rgba(99,179,237,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='P5–P95'
    ))
    fig3.update_layout(
        template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827',
        height=280, margin=dict(l=10,r=10,t=20,b=10),
        xaxis=dict(title='Order Qty', gridcolor='#1f2d47'),
        yaxis=dict(title='Profit (Rp)', gridcolor='#1f2d47'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        title=dict(text=f'EV + Confidence Interval ({dist_choice})', font=dict(color='#a0aec0', size=12))
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(f'<div class="result-box"><div class="label">✅ Keputusan Optimal</div><div class="value">Order = {best_prob} unit</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5: UTILITY
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Metode 5: Utility — Risk Preference</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-box">
        <div class="label">Konsep</div>
        <div style="color:#a0aec0; margin-top:6px; font-size:0.85rem; line-height:1.8">
        <b style="color:#68d391;">Risk-Averse:</b> Utility = √(payoff) — diminishing marginal utility<br>
        <b style="color:#63b3ed;">Risk-Neutral:</b> Utility = payoff (linear)<br>
        <b style="color:#fc8181;">Risk-Seeking:</b> Utility = payoff² — increasing marginal utility
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    states = [DEMAND_LOW, DEMAND_NORMAL, DEMAND_HIGH]
    probs  = [p_low, p_norm, p_high]
    
    # Shift payoffs ke positif untuk utility
    raw_payoffs = np.array([[hitung_payoff(q, s) for s in states] for q in ORDER_ALTERNATIVES])
    shift = abs(raw_payoffs.min()) + 1
    shifted = raw_payoffs + shift
    
    def utility_averse(x):  return np.sqrt(x)
    def utility_neutral(x): return x
    def utility_seeking(x): return x ** 1.5
    
    util_data = []
    for i, q in enumerate(ORDER_ALTERNATIVES):
        eu_averse  = sum(utility_averse(shifted[i,j])  * probs[j] for j in range(3))
        eu_neutral = sum(utility_neutral(shifted[i,j]) * probs[j] for j in range(3))
        eu_seeking = sum(utility_seeking(shifted[i,j]) * probs[j] for j in range(3))
        util_data.append({
            'Order Qty': q,
            'EU Risk-Averse': eu_averse,
            'EU Risk-Neutral': eu_neutral,
            'EU Risk-Seeking': eu_seeking
        })
    
    util_df = pd.DataFrame(util_data)
    
    # Normalize untuk perbandingan
    for col in ['EU Risk-Averse','EU Risk-Neutral','EU Risk-Seeking']:
        mn, mx = util_df[col].min(), util_df[col].max()
        util_df[f'{col} (norm)'] = (util_df[col] - mn) / (mx - mn)
    
    best_averse  = ORDER_ALTERNATIVES[util_df['EU Risk-Averse'].idxmax()]
    best_neutral = ORDER_ALTERNATIVES[util_df['EU Risk-Neutral'].idxmax()]
    best_seeking = ORDER_ALTERNATIVES[util_df['EU Risk-Seeking'].idxmax()]
    
    col_a, col_b = st.columns([3,2])
    with col_a:
        fig = go.Figure()
        x = [str(q) for q in ORDER_ALTERNATIVES]
        for col, color, name in [
            ('EU Risk-Averse (norm)', '#68d391', '🟢 Risk-Averse (√x)'),
            ('EU Risk-Neutral (norm)', '#63b3ed', '🔵 Risk-Neutral (x)'),
            ('EU Risk-Seeking (norm)', '#fc8181', '🔴 Risk-Seeking (x^1.5)')
        ]:
            fig.add_trace(go.Scatter(
                x=x, y=util_df[col], mode='lines+markers',
                line=dict(color=color, width=2.5),
                marker=dict(size=8), name=name
            ))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827',
            height=320, margin=dict(l=10,r=10,t=20,b=10),
            xaxis=dict(title='Order Qty', gridcolor='#1f2d47'),
            yaxis=dict(title='Expected Utility (normalized)', gridcolor='#1f2d47'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            title=dict(text='Expected Utility per Preferensi Risiko', font=dict(color='#a0aec0', size=12))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        for name, val, color in [
            ("🟢 Risk-Averse", best_averse, "#68d391"),
            ("🔵 Risk-Neutral", best_neutral, "#63b3ed"),
            ("🔴 Risk-Seeking", best_seeking, "#fc8181")
        ]:
            st.markdown(f"""
            <div class="result-box">
                <div class="label">{name}</div>
                <div class="value" style="color:{color};">Order = {val} unit</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Utility curve viz
    st.markdown('<div class="section-header">Kurva Fungsi Utility</div>', unsafe_allow_html=True)
    x_util = np.linspace(1, 100, 200)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_util, y=np.sqrt(x_util),    name='Risk-Averse (√x)',    line=dict(color='#68d391',width=2.5)))
    fig2.add_trace(go.Scatter(x=x_util, y=x_util,              name='Risk-Neutral (x)',     line=dict(color='#63b3ed',width=2.5, dash='dash')))
    fig2.add_trace(go.Scatter(x=x_util, y=(x_util**1.5)/100,   name='Risk-Seeking (x^1.5)', line=dict(color='#fc8181',width=2.5)))
    fig2.update_layout(
        template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827',
        height=250, margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(title='Payoff', gridcolor='#1f2d47'),
        yaxis=dict(title='Utility', gridcolor='#1f2d47'),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 6: MONTE CARLO
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Metode 6: Monte Carlo Simulation</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="result-box">
        <div class="label">Konfigurasi Simulasi</div>
        <div style="color:#a0aec0; margin-top:6px; font-size:0.85rem;">
        Demand di-sample dari distribusi Normal(μ={mu_norm:.1f}, σ={std_norm:.1f}).
        Jalankan <b style="color:#63b3ed;">{n_simulasi:,} siklus order</b> untuk setiap alternatif.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    np.random.seed(99)
    mc_results = {}
    for q in ORDER_ALTERNATIVES:
        demands = norm.rvs(mu_norm, std_norm, size=n_simulasi)
        demands = np.clip(demands, 0, None).round().astype(int)
        profits = np.array([hitung_payoff(q, int(d)) for d in demands])
        mc_results[q] = profits
    
    mc_summary = pd.DataFrame({
        'Order Qty': ORDER_ALTERNATIVES,
        'Mean Profit': [mc_results[q].mean() for q in ORDER_ALTERNATIVES],
        'Std Dev': [mc_results[q].std() for q in ORDER_ALTERNATIVES],
        'P5': [np.percentile(mc_results[q], 5) for q in ORDER_ALTERNATIVES],
        'P25': [np.percentile(mc_results[q], 25) for q in ORDER_ALTERNATIVES],
        'P75': [np.percentile(mc_results[q], 75) for q in ORDER_ALTERNATIVES],
        'P95': [np.percentile(mc_results[q], 95) for q in ORDER_ALTERNATIVES],
        'P(Profit>0)': [np.mean(mc_results[q] > 0) for q in ORDER_ALTERNATIVES]
    })
    
    best_mc = mc_summary.loc[mc_summary['Mean Profit'].idxmax(), 'Order Qty']
    
    col_a, col_b = st.columns([3,2])
    with col_a:
        fig = go.Figure()
        colors_mc = ['#63b3ed','#68d391','#f6ad55','#fc8181','#b794f4']
        for i, q in enumerate(ORDER_ALTERNATIVES):
            fig.add_trace(go.Histogram(
                x=mc_results[q], nbinsx=40,
                name=f'Order {q}',
                marker_color=colors_mc[i],
                opacity=0.6,
                histnorm='probability density'
            ))
        fig.update_layout(
            barmode='overlay',
            template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827',
            height=320, margin=dict(l=10,r=10,t=30,b=10),
            xaxis=dict(title='Profit (Rp)', gridcolor='#1f2d47'),
            yaxis=dict(title='Density', gridcolor='#1f2d47'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            title=dict(text=f'Distribusi Profit dari {n_simulasi:,} Simulasi', font=dict(color='#a0aec0', size=12))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        # Box plot
        fig2 = go.Figure()
        for i, q in enumerate(ORDER_ALTERNATIVES):
            fig2.add_trace(go.Box(
                y=mc_results[q], name=str(q),
                marker_color=colors_mc[i],
                line=dict(color=colors_mc[i])
            ))
        fig2.update_layout(
            template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827',
            height=320, margin=dict(l=10,r=10,t=20,b=10),
            xaxis=dict(title='Order Qty', gridcolor='#1f2d47'),
            yaxis=dict(title='Profit (Rp)', gridcolor='#1f2d47'),
            showlegend=False,
            title=dict(text='Box Plot Profit', font=dict(color='#a0aec0', size=12))
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Convergence plot (running mean for best alternative)
    st.markdown('<div class="section-header">Konvergensi Running Mean</div>', unsafe_allow_html=True)
    
    fig3 = go.Figure()
    for i, q in enumerate(ORDER_ALTERNATIVES):
        running_mean = np.cumsum(mc_results[q]) / np.arange(1, n_simulasi+1)
        fig3.add_trace(go.Scatter(
            x=np.arange(1, n_simulasi+1), y=running_mean,
            mode='lines', name=f'Order {q}',
            line=dict(color=colors_mc[i], width=1.5)
        ))
    fig3.update_layout(
        template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827',
        height=250, margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(title='Simulasi ke-', gridcolor='#1f2d47'),
        yaxis=dict(title='Running Mean Profit (Rp)', gridcolor='#1f2d47'),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Summary table
    st.markdown('<div class="section-header">Ringkasan Monte Carlo</div>', unsafe_allow_html=True)
    fmt_mc = mc_summary.copy()
    for col in ['Mean Profit','Std Dev','P5','P25','P75','P95']:
        fmt_mc[col] = fmt_mc[col].apply(lambda x: f"Rp{x:,.0f}")
    fmt_mc['P(Profit>0)'] = fmt_mc['P(Profit>0)'].apply(lambda x: f"{x:.1%}")
    st.dataframe(fmt_mc, hide_index=True, use_container_width=True)
    
    st.markdown(f'<div class="result-box"><div class="label">✅ Keputusan Optimal (Max Mean Profit)</div><div class="value">Order = {best_mc} unit</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# FOOTER: REKAPITULASI
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-header">📋 Rekapitulasi Keputusan Semua Metode</div>', unsafe_allow_html=True)

rekap = pd.DataFrame({
    'Metode': ['Certainty','Risk (EV)','Maximax','Maximin','Laplace','Minimax Regret','Prob. Normal','Utility-Averse','Utility-Neutral','Utility-Seeking','Monte Carlo'],
    'Keputusan Order': [best_q, best_ev_q, maximax_best, maximin_best, laplace_best, minimax_best, best_prob, best_averse, best_neutral, best_seeking, best_mc],
    'Deskripsi': [
        'Demand dianggap pasti = rata-rata',
        'Maksimalkan expected value dengan prob. empiris',
        'Optimis total — pilih max dari max',
        'Pesimis — pilih max dari worst case',
        'Semua state equally likely',
        'Minimalkan penyesalan terbesar',
        'EV dari sampling distribusi Normal',
        'Risk-Averse: utility = √(payoff)',
        'Risk-Neutral: utility = payoff',
        'Risk-Seeking: utility = payoff^1.5',
        f'Simulasi {n_simulasi:,} siklus, max mean profit'
    ]
})

st.dataframe(rekap, hide_index=True, use_container_width=True)

st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:0.75rem; padding:20px 0 8px; font-family:Space Mono,monospace;'>
Decision Analysis Dashboard · Data: Minuman Kemasan · 52 Periode Historis
</div>
""", unsafe_allow_html=True)

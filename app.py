import os
from datetime import date, datetime, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import folium
from streamlit.components.v1 import html as st_html

# ================ APP CONFIG ================
st.set_page_config(
    page_title="Pendataan SLOG – Dashboard",
    page_icon="📊",
    layout="wide"
)

load_dotenv()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
DB_PATH = "data.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)

# ================ DB BOOTSTRAP ================
DDL = """
CREATE TABLE IF NOT EXISTS targets (
    id INTEGER PRIMARY KEY CHECK (id=1),
    internal_target INTEGER NOT NULL,
    eksternal_offline_target INTEGER NOT NULL,
    eksternal_online_target INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS aggregates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_date TEXT NOT NULL,     -- YYYY-MM-DD
    source TEXT NOT NULL,         -- internal | eksternal_offline | eksternal_online
    n INTEGER NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL      -- UTC ISO
);
CREATE TABLE IF NOT EXISTS map_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    label TEXT,
    created_at TEXT NOT NULL      -- UTC ISO
);
"""
with engine.begin() as conn:
    for stmt in DDL.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(text(s))
    # seed target jika kosong
    cur = conn.execute(text("SELECT COUNT(*) FROM targets"))
    if cur.scalar_one() == 0:
        conn.execute(text("""
            INSERT INTO targets (id, internal_target, eksternal_offline_target, eksternal_online_target, updated_at)
            VALUES (1, 250, 50, 200, :ts)
        """), {"ts": datetime.utcnow().isoformat()})

# ================ HELPERS ================
@st.cache_data(ttl=20)
def load_targets():
    with engine.begin() as conn:
        return conn.execute(text("SELECT * FROM targets WHERE id=1")).mappings().first()

@st.cache_data(ttl=20)
def load_aggregates():
    with engine.begin() as conn:
        return pd.read_sql("SELECT * FROM aggregates", conn)

@st.cache_data(ttl=20)
def load_map_points():
    with engine.begin() as conn:
        return pd.read_sql("SELECT * FROM map_points", conn)

def invalidate():
    load_targets.clear()
    load_aggregates.clear()
    load_map_points.clear()

def totals_from(df: pd.DataFrame):
    base = {"internal": 0, "eksternal_offline": 0, "eksternal_online": 0}
    if df.empty:
        return base
    g = df.groupby("source")["n"].sum()
    for k in base.keys():
        base[k] = int(g.get(k, 0))
    return base

def pie_progress(title, achieved, target):
    remain = max(target - achieved, 0)
    data = pd.DataFrame({"Label": ["Tercapai", "Sisa"], "Jumlah": [achieved, remain]})
    fig = px.pie(data, values="Jumlah", names="Label", hole=0.5, title=title)
    fig.update_traces(textinfo="value+percent")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def card_metric(title, value, target):
    delta = target - value
    st.metric(title, value=value, delta=f"Sisa {delta}" if delta >= 0 else f"Lebih {abs(delta)}")

def parse_wib(ts_iso: str) -> str:
    """Konversi UTC ISO ke WIB (UTC+7) dan format cantik."""
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", ""))
    except Exception:
        return ts_iso
    dt_wib = dt + timedelta(hours=7)
    return dt_wib.strftime("%Y-%m-%d %H:%M WIB")

def build_map_html(df_pts: pd.DataFrame) -> str:
    """Bangun folium map dan kembalikan HTML string."""
    if df_pts.empty:
        # map default Jakarta jika kosong
        center = [-6.2, 106.816666]  # Jakarta
        m = folium.Map(location=center, zoom_start=11)
        folium.Marker(center, popup="Belum ada titik").add_to(m)
    else:
        lat0 = df_pts["lat"].mean()
        lon0 = df_pts["lon"].mean()
        m = folium.Map(location=[lat0, lon0], zoom_start=12)
        for _, r in df_pts.iterrows():
            popup = r.get("label") or f"{r['lat']}, {r['lon']}"
            folium.Marker([r["lat"], r["lon"]], popup=str(popup)).add_to(m)
    return m._repr_html_()

def tolerant_read_csv(file) -> pd.DataFrame:
    """Baca CSV toleran separator/quote, auto-parse titik->lat/lon bila perlu."""
    # coba sniff/auto sep
    df = pd.read_csv(file, engine="python", quotechar='"', sep=None)
    # Kalau ada kolom lat/lon langsung pakai
    cols = [c.lower().strip() for c in df.columns]
    if "lat" in cols and "lon" in cols:
        df = df.rename(columns={df.columns[cols.index("lat")]: "lat",
                                df.columns[cols.index("lon")]: "lon"})
        return df[["lat", "lon"]].copy()

    # Kalau ada 'titik' → parse angka
    if "titik" in cols:
        titik_col = df.columns[cols.index("titik")]
        s = df[titik_col].astype(str)
    else:
        # gabungkan semua kolom jadi satu string
        s = df.astype(str).agg(" ".join, axis=1)

    import re
    def parse_lat_lon_auto(text):
        if pd.isna(text):
            return None, None
        text = str(text)
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        return None, None

    parsed = s.apply(parse_lat_lon_auto)
    lat = [p[0] for p in parsed]
    lon = [p[1] for p in parsed]
    out = pd.DataFrame({"lat": lat, "lon": lon})
    out = out.dropna().reset_index(drop=True)
    return out

# ================ AUTH STATE ================
if "authed" not in st.session_state:
    st.session_state.authed = False

# ================ SIDEBAR NAV ================
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio("Pilih tampilan", ["Dashboard", "Kelola Data (Admin)"], index=0)

if page == "Kelola Data (Admin)" and not st.session_state.authed:
    st.sidebar.subheader("Login Admin")
    pwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Masuk"):
        if pwd == ADMIN_PASSWORD:
            st.session_state.authed = True
            st.sidebar.success("Login berhasil ✅")
        else:
            st.sidebar.error("Password salah")
if page == "Kelola Data (Admin)" and st.session_state.authed:
    st.sidebar.success("Mode Admin aktif")

st.sidebar.markdown("---")
st.sidebar.caption("© Firstat • SLOG Pendataan")

# ================ LOAD DATA ================
targets = load_targets()
df = load_aggregates()
df_pts = load_map_points()

tot = totals_from(df)
total_target = int(targets["internal_target"]) + int(targets["eksternal_offline_target"]) + int(targets["eksternal_online_target"])
total_achieved = tot["internal"] + tot["eksternal_offline"] + tot["eksternal_online"]

# cari waktu update terakhir (aggregates)
last_ts = None
if not df.empty:
    try:
        last_ts = df["created_at"].max()
    except Exception:
        last_ts = None

# ================ HEADER =================
st.title("📊 Dashboard Progres Pendataan SLOG")
st.caption("By Firstat")

# ================ DASHBOARD =================
if page == "Dashboard":
    # ---- PROGRES KESELURUHAN (headline) ----
    st.markdown("## 🔹 Progres Keseluruhan")
    overall_pct = 0 if total_target <= 0 else min(total_achieved / total_target, 1.0)
    cA, cB = st.columns([1, 3])
    with cA:
        st.metric("TOTAL Terkumpul", value=total_achieved, delta=f"Target {total_target}")
    with cB:
        st.progress(overall_pct, text=f"{total_achieved} / {total_target} responden ({overall_pct*100:.1f}%)")

    # info waktu update terakhir
    if last_ts:
        st.caption(f"Data terakhir diperbarui: {parse_wib(last_ts)}")

    # ---- METRIK PER KANAL ----
    st.markdown("## 📈 Ringkasan Per Survei")
    m1, m2, m3 = st.columns(3)
    with m1:
        card_metric("Internal", tot["internal"], int(targets["internal_target"]))
    with m2:
        card_metric("Eksternal Offline", tot["eksternal_offline"], int(targets["eksternal_offline_target"]))
    with m3:
        card_metric("Eksternal Online", tot["eksternal_online"], int(targets["eksternal_online_target"]))

    # ---- PIE PROGRESS ----
    st.markdown("## 🥧 Distribusi Pencapaian")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(pie_progress("Internal", tot["internal"], int(targets["internal_target"])), use_container_width=True)
    with c2:
        st.plotly_chart(pie_progress("Eksternal Offline", tot["eksternal_offline"], int(targets["eksternal_offline_target"])), use_container_width=True)
    with c3:
        st.plotly_chart(pie_progress("Eksternal Online", tot["eksternal_online"], int(targets["eksternal_online_target"])), use_container_width=True)

    # ---- MAP (di bawah chart) ----
    st.markdown("## 🗺️ Peta Titik Responden Eksternal Offline")
    if df_pts.empty:
        st.info("Belum ada data peta. Admin bisa unggah CSV pada menu **Kelola Data (Admin) → 🗺️ Data Peta**.")
    else:
        map_html = build_map_html(df_pts)
        st_html(map_html, height=520, scrolling=False)

# ================ ADMIN =================
if page == "Kelola Data (Admin)":
    if not st.session_state.authed:
        st.warning("Masuk sebagai Admin untuk mengelola data dan target.")
    else:
        tab1, tab2, tab3 = st.tabs(["➕ Input Agregat", "🎯 Ubah Target", "🗺️ Data Peta"])

        # ---- INPUT AGREGAT ----
        with tab1:
            st.subheader("Tambah Data Agregat")
            with st.form("form_agg"):
                colA, colB = st.columns(2)
                with colA:
                    tgl = st.date_input("Tanggal", date.today())
                    sumber = st.selectbox("Sumber", ["internal", "eksternal_offline", "eksternal_online"])
                with colB:
                    n = st.number_input("Jumlah responden", min_value=0, step=1, value=0)
                    note = st.text_input("Catatan (opsional)", value="")
                ok = st.form_submit_button("Simpan")
            if ok:
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO aggregates (input_date, source, n, note, created_at)
                        VALUES (:d, :s, :n, :note, :ts)
                    """), {"d": tgl.strftime("%Y-%m-%d"), "s": sumber, "n": int(n),
                            "note": note.strip(), "ts": datetime.utcnow().isoformat()})
                invalidate()
                st.success("Agregat tersimpan ✅")

            st.divider()
            st.subheader("🧾 Riwayat Input (10 terbaru)")
            df_now = load_aggregates()
            if df_now.empty:
                st.info("Belum ada data.")
            else:
                last_10 = df_now.sort_values("created_at", ascending=False).head(10).copy()
                last_10["Waktu (WIB)"] = last_10["created_at"].apply(parse_wib)
                last_10.rename(columns={
                    "input_date": "Tanggal", "source": "Sumber", "n": "Jumlah", "note": "Catatan"
                }, inplace=True)
                st.dataframe(last_10[["Tanggal", "Sumber", "Jumlah", "Catatan", "Waktu (WIB)"]],
                             use_container_width=True, hide_index=True)
                csv = df_now.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Unduh Semua Data (CSV)", data=csv, file_name="aggregates.csv", mime="text/csv")

        # ---- UBAH TARGET ----
        with tab2:
            st.subheader("Atur Target")
            cur = load_targets()
            with st.form("form_target"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    t_int = st.number_input("Target Internal", min_value=0, step=1, value=int(cur["internal_target"]))
                with c2:
                    t_off = st.number_input("Target Eksternal Offline", min_value=0, step=1, value=int(cur["eksternal_offline_target"]))
                with c3:
                    t_on = st.number_input("Target Eksternal Online", min_value=0, step=1, value=int(cur["eksternal_online_target"]))
                up = st.form_submit_button("Simpan Target")
            if up:
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE targets
                        SET internal_target=:a, eksternal_offline_target=:b, eksternal_online_target=:c, updated_at=:ts
                        WHERE id=1
                    """), {"a": int(t_int), "b": int(t_off), "c": int(t_on), "ts": datetime.utcnow().isoformat()})
                invalidate()
                st.success("Target diperbarui ✅")

        # ---- DATA PETA ----
        with tab3:
            st.subheader("Unggah CSV Titik (lat/lon atau kolom 'titik')")
            replace = st.checkbox("Ganti semua titik lama (replace)", value=True)
            file = st.file_uploader("Pilih file CSV", type=["csv"])
            if file is not None:
                try:
                    pts = tolerant_read_csv(file)
                    pts = pts.dropna(subset=["lat", "lon"])
                    st.success(f"Terbaca {len(pts)} titik.")
                    st.dataframe(pts.head(10), use_container_width=True, hide_index=True)
                    if st.button("Simpan ke Peta"):
                        with engine.begin() as conn:
                            if replace:
                                conn.execute(text("DELETE FROM map_points"))
                            # batch insert
                            now = datetime.utcnow().isoformat()
                            rows = [{"lat": float(r.lat), "lon": float(r.lon), "label": None, "created_at": now}
                                    for r in pts.itertuples(index=False)]
                            if rows:
                                conn.execute(text("""
                                    INSERT INTO map_points (lat, lon, label, created_at)
                                    VALUES (:lat, :lon, :label, :created_at)
                                """), rows)
                        invalidate()
                        st.success("Titik peta berhasil disimpan ✅")
                except Exception as e:
                    st.error(f"Gagal membaca CSV: {e}")

        st.markdown("---")
        if st.button("🚪 Keluar dari Mode Admin"):
            st.session_state.authed = False
            st.success("Logout berhasil. Kembali ke Dashboard.")

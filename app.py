import os
from datetime import date, datetime, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import folium
from streamlit.components.v1 import html as st_html
import re

# ================ APP CONFIG ================
st.set_page_config(
    page_title="Pendataan SLOG – Dashboard",
    page_icon="chart",
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
    input_date TEXT NOT NULL, -- YYYY-MM-DD (tanggal cek)
    source TEXT NOT NULL, -- internal | eksternal_offline | eksternal_online
    total_n INTEGER NOT NULL, -- TOTAL akumulasi sampai tanggal ini
    note TEXT,
    created_at TEXT NOT NULL -- UTC ISO
);
CREATE TABLE IF NOT EXISTS map_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    label TEXT,
    created_at TEXT NOT NULL -- UTC ISO
);
"""
with engine.begin() as conn:
    for stmt in DDL.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(text(s))
    # Seed target
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
        return pd.read_sql("SELECT * FROM aggregates ORDER BY input_date, source", conn)

@st.cache_data(ttl=20)
def load_map_points():
    with engine.begin() as conn:
        return pd.read_sql("SELECT * FROM map_points", conn)

def invalidate():
    load_targets.clear()
    load_aggregates.clear()
    load_map_points.clear()

def get_latest_totals(df: pd.DataFrame):
    """Ambil TOTAL terbaru per sumber (berdasarkan input_date terbaru per sumber)"""
    if df.empty:
        return {"internal": 0, "eksternal_offline": 0, "eksternal_online": 0}
    # Urutkan & ambil yang terbaru per sumber
    latest = df.sort_values("input_date").groupby("source").tail(1)
    totals = {}
    for src in ["internal", "eksternal_offline", "eksternal_online"]:
        val = latest[latest["source"] == src]["total_n"].iloc[0] if src in latest["source"].values else 0
        totals[src] = int(val)
    return totals

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
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", ""))
    except Exception:
        return ts_iso
    dt_wib = dt + timedelta(hours=7)
    return dt_wib.strftime("%Y-%m-%d %H:%M WIB")

def build_map_html(df_pts: pd.DataFrame) -> str:
    if df_pts.empty:
        center = [-6.2, 106.816666]
        m = folium.Map(location=center, zoom_start=11)
        folium.Marker(center, popup="Belum ada titik").add_to(m)
    else:
        lat0 = df_pts["lat"].mean()
        lon0 = df_pts["lon"].mean()
        m = folium.Map(location=[lat0, lon0], zoom_start=12)
        for _, r in df_pts.iterrows():
            popup = r.get("label") or f"{r['lat']:.6f}, {r['lon']:.6f}"
            folium.Marker([r["lat"], r["lon"]], popup=str(popup)).add_to(m)
    return m._repr_html_()

def tolerant_read_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, engine="python", quotechar='"', sep=None)
    cols = [c.lower().strip() for c in df.columns]
    if "lat" in cols and "lon" in cols:
        df = df.rename(columns={
            df.columns[cols.index("lat")]: "lat",
            df.columns[cols.index("lon")]: "lon"
        })
        return df[["lat", "lon"]].copy()
    if "titik" in cols:
        titik_col = df.columns[cols.index("titik")]
        s = df[titik_col].astype(str)
    else:
        s = df.astype(str).agg(" ".join, axis=1)

    def parse_lat_lon_auto(text):
        if pd.isna(text): return None, None
        text = str(text)
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        return None, None
    parsed = s.apply(parse_lat_lon_auto)
    lat = [p[0] for p in parsed]
    lon = [p[1] for p in parsed]
    out = pd.DataFrame({"lat": lat, "lon": lon})
    return out.dropna().reset_index(drop=True)

# ================ AUTH STATE ================
if "authed" not in st.session_state:
    st.session_state.authed = False

# ================ SIDEBAR NAV ================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih tampilan", ["Dashboard", "Kelola Data (Admin)"], index=0)

if page == "Kelola Data (Admin)" and not st.session_state.authed:
    st.sidebar.subheader("Login Admin")
    pwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Masuk"):
        if pwd == ADMIN_PASSWORD:
            st.session_state.authed = True
            st.sidebar.success("Login berhasil")
        else:
            st.sidebar.error("Password salah")

if page == "Kelola Data (Admin)" and st.session_state.authed:
    st.sidebar.success("Mode Admin aktif")

st.sidebar.markdown("---")
st.sidebar.caption("© Firstat • SLOG Pendataan")

# ================ LOAD DATA ================
targets = load_targets()
df_agg = load_aggregates()
df_pts = load_map_points()
tot = get_latest_totals(df_agg)  # TOTAL terbaru per sumber
total_target = int(targets["internal_target"]) + int(targets["eksternal_offline_target"]) + int(targets["eksternal_online_target"])
total_achieved = sum(tot.values())

last_update = None
if not df_agg.empty:
    last_update = df_agg["created_at"].max()

# ================ HEADER =================
st.title("Dashboard Progres Pendataan SLOG")
st.caption("By Firstat • Data diambil dari **total akumulasi terbaru per kanal**")

# ================ DASHBOARD =================
if page == "Dashboard":
    st.markdown("## Progres Keseluruhan")
    overall_pct = 0 if total_target <= 0 else min(total_achieved / total_target, 1.0)
    cA, cB = st.columns([1, 3])
    with cA:
        st.metric("TOTAL Terkumpul", value=total_achieved, delta=f"Target {total_target}")
    with cB:
        st.progress(overall_pct, text=f"{total_achieved} / {total_target} ({overall_pct*100:.1f}%)")

    if last_update:
        st.caption(f"Terakhir diupdate: **{parse_wib(last_update)}**")

    st.markdown("## Ringkasan Per Kanal")
    m1, m2, m3 = st.columns(3)
    with m1:
        card_metric("Internal", tot["internal"], int(targets["internal_target"]))
    with m2:
        card_metric("Eksternal Offline", tot["eksternal_offline"], int(targets["eksternal_offline_target"]))
    with m3:
        card_metric("Eksternal Online", tot["eksternal_online"], int(targets["eksternal_online_target"]))

    st.markdown("## Distribusi Pencapaian")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(pie_progress("Internal", tot["internal"], int(targets["internal_target"])), use_container_width=True)
    with c2:
        st.plotly_chart(pie_progress("Eksternal Offline", tot["eksternal_offline"], int(targets["eksternal_offline_target"])), use_container_width=True)
    with c3:
        st.plotly_chart(pie_progress("Eksternal Online", tot["eksternal_online"], int(targets["eksternal_online_target"])), use_container_width=True)

    st.markdown("## Peta Titik Responden Eksternal Offline")
    if df_pts.empty:
        st.info("Belum ada data peta. Admin bisa unggah CSV di menu **Kelola Data (Admin) → Data Peta**.")
    else:
        map_html = build_map_html(df_pts)
        st_html(map_html, height=520, scrolling=False)

# ================ ADMIN =================
if page == "Kelola Data (Admin)":
    if not st.session_state.authed:
        st.warning("Masuk sebagai Admin untuk mengelola data.")
    else:
        tab1, tab2, tab3 = st.tabs(["Input Total Akumulasi", "Ubah Target", "Data Peta"])

        # ---- INPUT TOTAL AKUMULASI ----
        with tab1:
            st.subheader("Update Total Akumulasi per Tanggal Cek")
            st.caption("Masukkan **total responden sampai hari ini** (bukan tambahan harian).")

            with st.form("form_agg"):
                colA, colB = st.columns(2)
                with colA:
                    tgl = st.date_input("Tanggal Cek", date.today())
                    sumber = st.selectbox("Kanal", ["internal", "eksternal_offline", "eksternal_online"])
                with colB:
                    total_n = st.number_input("Total Responden (Akumulasi)", min_value=0, step=1, value=0)
                    note = st.text_input("Catatan (opsional)", placeholder="Contoh: data dari lapangan")
                submit = st.form_submit_button("Simpan Total", type="primary")

            if submit:
                input_date_str = tgl.strftime("%Y-%m-%d")
                now_ts = datetime.utcnow().isoformat()

                with engine.begin() as conn:
                    # Cek apakah sudah ada entri
                    existing = conn.execute(
                        text("SELECT id, total_n FROM aggregates WHERE input_date = :d AND source = :s"),
                        {"d": input_date_str, "s": sumber}
                    ).fetchone()

                    if existing:
                        old_total = existing[1]
                        conn.execute(
                            text("""
                                UPDATE aggregates
                                SET total_n = :n, note = :note, created_at = :ts
                                WHERE id = :id
                            """),
                            {"n": int(total_n), "note": note.strip(), "ts": now_ts, "id": existing[0]}
                        )
                        st.success(f"Total **{sumber}** pada **{input_date_str}** diperbarui: {old_total} → {total_n}")
                    else:
                        conn.execute(
                            text("""
                                INSERT INTO aggregates (input_date, source, total_n, note, created_at)
                                VALUES (:d, :s, :n, :note, :ts)
                            """),
                            {"d": input_date_str, "s": sumber, "n": int(total_n),
                             "note": note.strip(), "ts": now_ts}
                        )
                        st.success(f"Total **{sumber}** pada **{input_date_str}** disimpan: {total_n}")

                invalidate()

            st.divider()
            st.subheader("Riwayat Update Total (10 Terbaru)")
            df_now = load_aggregates()
            if df_now.empty:
                st.info("Belum ada data.")
            else:
                last_10 = df_now.sort_values("created_at", ascending=False).head(10).copy()
                last_10["Waktu (WIB)"] = last_10["created_at"].apply(parse_wib)
                last_10.rename(columns={
                    "input_date": "Tanggal Cek",
                    "source": "Kanal",
                    "total_n": "Total Responden",
                    "note": "Catatan"
                }, inplace=True)
                st.dataframe(last_10[["Tanggal Cek", "Kanal", "Total Responden", "Catatan", "Waktu (WIB)"]],
                             use_container_width=True, hide_index=True)
                csv = df_now.to_csv(index=False).encode("utf-8")
                st.download_button("Unduh Semua Data (CSV)", data=csv, file_name="slog_totals.csv", mime="text/csv")

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
                st.success("Target diperbarui")

        # ---- DATA PETA ----
        with tab3:
            st.subheader("Unggah CSV Titik (lat/lon atau kolom 'titik')")
            replace = st.checkbox("Ganti semua titik lama (replace)", value=True)
            file = st.file_uploader("Pilih file CSV", type=["csv"])

            if file is not None:
                try:
                    pts = tolerant_read_csv(file)
                    pts = pts.dropna(subset=["lat", "lon"]).reset_index(drop=True)
                    if pts.empty:
                        st.warning("Tidak ada titik valid.")
                    else:
                        st.success(f"Berhasil baca **{len(pts)}** titik.")
                        st.dataframe(pts.head(10), use_container_width=True, hide_index=True)

                        preview = f"Akan **{'menggantikan semua' if replace else 'menambahkan'}** {len(pts)} titik."
                        if replace and not df_pts.empty:
                            preview += f" (hapus {len(df_pts)} titik lama)"
                        st.info(preview)

                        if st.button("Simpan ke Peta", type="primary"):
                            with engine.begin() as conn:
                                if replace:
                                    conn.execute(text("DELETE FROM map_points"))
                                now = datetime.utcnow().isoformat()
                                rows = [
                                    {"lat": float(r.lat), "lon": float(r.lon), "label": None, "created_at": now}
                                    for r in pts.itertuples(index=False)
                                ]
                                if rows:
                                    conn.execute(text("""
                                        INSERT INTO map_points (lat, lon, label, created_at)
                                        VALUES (:lat, :lon, :label, :created_at)
                                    """), rows)
                                    action = "diganti" if replace else "ditambahkan"
                                    st.success(f"{len(rows)} titik berhasil **{action}**")
                            invalidate()
                except Exception as e:
                    st.error(f"Gagal baca CSV: {e}")

        if st.button("Keluar dari Mode Admin"):
            st.session_state.authed = False
            st.rerun()

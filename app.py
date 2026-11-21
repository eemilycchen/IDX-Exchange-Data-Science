"""Home sale price estimator: a demo of our team's XGBoost model.

Build the model files first (once):
    uv run python build_app_model.py
Then run:
    uv run streamlit run app.py

Two ways to value a home:
- Pick an April 2026 home: every feature comes pre-computed from that sale's MLS record.
- Enter a home: you type the key inputs; location fields come from the nearest recorded sale
  and anything else is left blank for the model's training-median fill.
The "Where each input comes from" section lists which features are typed and which are pre-computed.
"""
import os
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
# The arm64 XGBoost wheel needs libomp; scikit-learn ships one, so load it before the model unpickles XGBoost
import sklearn.utils._openmp_helpers
import streamlit as st
from sklearn.neighbors import BallTree

# APP_ARTIFACTS lets you point the app at another build folder
ARTIFACTS = Path(os.environ.get("APP_ARTIFACTS", Path(__file__).parent / "app_artifacts"))
MODEL_PATH = ARTIFACTS / "model.joblib"
DATA_PATH = ARTIFACTS / "app_data.joblib"
EARTH_KM = 6371.0
# Same similar-comp rules as idx_pipeline.add_similar_comps (used in training)
COMP_K = 10
COMP_WINDOW_DAYS = 180
COMP_AREA_TOL = 0.25
COMP_YEAR_TOL = 15
COMP_CANDIDATES = 300
SIM_COMPS = ["sim_ppsf", "sim_ppsf_iqr", "sim_count", "sim_value"]
YES_NO = {"Yes": 1.0, "No": 0.0, "Not reported": np.nan}
BINARY_LABELS = {
    "PoolPrivateYN": "Private pool",
    "FireplaceYN": "Fireplace",
    "ViewYN": "View",
    "NewConstructionYN": "New construction",
    "AttachedGarageYN": "Attached garage",
}

st.set_page_config(page_title="Home Price Estimator", page_icon="🏠", layout="wide")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error(f"Model files not found in `{ARTIFACTS}`. Run `uv run python build_app_model.py` first.")
    st.stop()


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    data = joblib.load(DATA_PATH)
    pool = data["pool"]
    tree = BallTree(np.radians(pool[["Latitude", "Longitude"]].to_numpy(dtype=float)), metric="haversine")
    return model, data, tree


model, data, pool_tree = load_artifacts()
NUM = data["num"]
CAT = data["cat"]
FEATURES = NUM + CAT
INFO = data["info"]
BANDS = data["bands"]
POOL = data["pool"]
DEMO = data["demo"]
USES_COMPS = "sim_ppsf" in NUM
USES_MONTH = "month_idx" in NUM
ESTIMATE_DATE = pd.Timestamp(INFO["estimate_date"])
FIRST_MONTH = pd.Period(INFO["first_month"], freq="M")


# ============================================================
# Helpers
# ============================================================

def money(x):
    if x is None or pd.isna(x):
        return "n/a"
    return f"${x:,.0f}"


def number(x, fmt="{:,.0f}"):
    if x is None or pd.isna(x):
        return "n/a"
    return fmt.format(x)


def text(x):
    if x is None or pd.isna(x) or str(x).strip() == "":
        return "n/a"
    return str(x)


def yes_no(x):
    if x is None or pd.isna(x):
        return "Not reported"
    return "Yes" if x >= 0.5 else "No"


def price_band(price):
    labels = data["band_labels"]
    idx = int(np.searchsorted(data["band_edges"], price, side="right")) - 1
    return labels[min(max(idx, 0), len(labels) - 1)]


def find_comps(lat, lon, area, year, as_of, listing_id="", property_id=""):
    # Nearest similar sales: closed in the 180 days before as_of, size within 25%, built within 15 years.
    # Same rules as the training feature; a home's own sale and earlier sales of the same house are excluded.
    if pd.isna(area) or pd.isna(year) or pd.isna(lat) or pd.isna(lon):
        return POOL.iloc[0:0].assign(km=[])
    k = min(COMP_CANDIDATES, len(POOL))
    dist, idx = pool_tree.query(np.radians([[lat, lon]]), k=k)
    cand = POOL.iloc[idx[0]].copy()
    cand["km"] = dist[0] * EARTH_KM
    ok = (cand["CloseDate"] < as_of) & (cand["CloseDate"] >= as_of - pd.Timedelta(days=COMP_WINDOW_DAYS))
    ok &= (cand["ListingId"] != str(listing_id)) & (cand["property_id"] != str(property_id))
    ok &= (cand["LivingArea"] / area - 1).abs() <= COMP_AREA_TOL
    ok &= (cand["YearBuilt"] - year).abs() <= COMP_YEAR_TOL
    return cand.loc[ok].head(COMP_K)


def comp_features(comps, area):
    if comps.empty:
        return {"sim_ppsf": np.nan, "sim_ppsf_iqr": np.nan, "sim_count": 0.0, "sim_value": np.nan}
    ppsf = comps["price_per_sqft"].to_numpy(dtype=float)
    med = float(np.median(ppsf))
    q75, q25 = np.percentile(ppsf, [75, 25])
    return {"sim_ppsf": med, "sim_ppsf_iqr": float(q75 - q25), "sim_count": float(len(ppsf)),
            "sim_value": med * area}


def nearest_sale(lat, lon):
    _, idx = pool_tree.query(np.radians([[lat, lon]]), k=1)
    return POOL.iloc[idx[0][0]]


def predict(row):
    X = pd.DataFrame([row])[FEATURES]
    for c in CAT:
        X[c] = X[c].astype("object").where(X[c].notna(), np.nan)
    # The model was trained on log(ClosePrice), so exp converts back to dollars
    price = float(np.exp(model.predict(X)[0]))
    band = price_band(price)
    stats = BANDS.loc[band]
    return {
        "price": price,
        "low": price * float(stats["ratio_q10"]),
        "high": price * float(stats["ratio_q90"]),
        "band": band,
        "band_stats": stats,
    }


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Home price estimator")
    mode = st.radio("How do you want to value a home?",
                    ["Pick an April 2026 home", "Enter a home"])
    st.divider()
    show_actual = st.toggle(
        "Show actual sale price", value=False,
        help="April 2026 is the team's held-out test month. Leave this off until the final model is agreed, "
             "so nobody tunes on test results.")
    st.caption(f"Model: {INFO['config_name']} ({INFO['config']}), XGBoost, "
               f"trained on {INFO['n_train']:,} sales through {INFO['trained_through']}.")
    vm = INFO["validation_metrics"]
    st.caption(f"Validation ({INFO['validation_month']}): MdAPE {vm['MdAPE_%']:.2f}%, "
               f"MAPE {vm['MAPE_%']:.2f}%, R² {vm['R2']:.3f}.")

st.title("Home sale price estimate")
st.caption("A portfolio demo of our IDX Exchange internship model. It's separate from the valuation tool "
           "on idx-search.com. Predictions run locally from the saved pipeline.")


# ============================================================
# Build the feature row
# ============================================================

subject = None

if mode == "Pick an April 2026 home":
    left, mid, right = st.columns(3)
    with left:
        county = st.selectbox("County", ["All"] + sorted(DEMO["CountyOrParish"].dropna().unique()))
    homes = DEMO if county == "All" else DEMO[DEMO["CountyOrParish"] == county]
    with mid:
        city = st.selectbox("City", ["All"] + sorted(homes["City"].dropna().unique()))
    homes = homes if city == "All" else homes[homes["City"] == city]
    with right:
        search = st.text_input("Search address", placeholder="e.g. Oak")
    if search.strip():
        homes = homes[homes["UnparsedAddress"].astype(str).str.contains(search.strip(), case=False, na=False)]
    homes = homes.sort_values("UnparsedAddress").head(500)
    if homes.empty:
        st.info("No April 2026 homes match those filters.")
        st.stop()

    def label(i):
        h = DEMO.loc[i]
        return (f"{h['UnparsedAddress']}, {h['City']} · {number(h['BedroomsTotal'])} bd / "
                f"{number(h['BathroomsTotalInteger'])} ba · {number(h['LivingArea'])} sq ft")

    st.caption(f"{len(homes):,} homes shown (max 500). Narrow by county, city or address.")
    choice = st.selectbox("Home", homes.index, format_func=label)
    home = DEMO.loc[choice]
    row = home[FEATURES].to_dict()
    as_of = home["CloseDate"]
    comps = find_comps(home["Latitude"], home["Longitude"], home["LivingArea"], home["YearBuilt"],
                       as_of, home["ListingId"], home["property_id"])
    subject = {
        "address": f"{home['UnparsedAddress']}, {home['City']}",
        "row": row,
        "display": home,
        "comps": comps,
        "actual": home["ClosePrice"],
        "as_of": as_of,
        "sources": {f: "Pre-computed from the April MLS record" for f in FEATURES},
    }

else:
    with st.form("manual"):
        st.subheader("Key inputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            living_area = st.number_input("Living area (sq ft)", 300, 15000, 1600, step=50)
            lot = st.number_input("Lot size (sq ft, 0 = unknown)", 0, 4_356_000, 6000, step=250)
            year_built = st.number_input("Year built", 1850, 2026, 1975)
        with c2:
            beds = st.number_input("Bedrooms", 1, 20, 3)
            baths = st.number_input("Bathrooms", 1, 20, 2)
            garage = st.number_input("Garage spaces", 0, 20, 2)
        with c3:
            loc_mode = st.radio("Location", ["ZIP code", "Latitude / longitude"], horizontal=True)
            zip_code = st.text_input("ZIP code", "90034", max_chars=5)
            lat_in = st.number_input("Latitude", 32.4, 42.1, 34.03, format="%.5f")
            lon_in = st.number_input("Longitude", -124.5, -114.1, -118.40, format="%.5f")

        st.subheader("Optional details")
        d1, d2, d3 = st.columns(3)
        binary = {}
        with d1:
            for col in ["PoolPrivateYN", "FireplaceYN", "ViewYN"]:
                binary[col] = YES_NO[st.selectbox(BINARY_LABELS[col], list(YES_NO), index=2, key=col)]
        with d2:
            hoa = st.selectbox("HOA", ["Unknown", "No HOA", "Has HOA"])
            fee = st.number_input("Monthly HOA fee ($, if Has HOA)", 0.0, 10000.0, 0.0, step=25.0)
        with d3:
            floors = st.multiselect("Flooring materials", data["flooring_types"],
                                    help="Leave empty if not known.")
        submitted = st.form_submit_button("Estimate price", type="primary")

    if submitted:
        st.session_state.manual_inputs = dict(
            living_area=living_area, lot=lot, year_built=year_built, beds=beds, baths=baths,
            garage=garage, loc_mode=loc_mode, zip_code=zip_code.strip(), lat=lat_in, lon=lon_in,
            binary=binary, hoa=hoa, fee=fee, floors=floors)

    m = st.session_state.get("manual_inputs")
    if m is None:
        st.info("Fill in the form and press **Estimate price**.")
        st.stop()

    # Location: a ZIP uses the median coordinates of recent sales in that ZIP
    if m["loc_mode"] == "ZIP code":
        in_zip = POOL[POOL["PostalCode"] == m["zip_code"]]
        if in_zip.empty:
            st.error(f"No recorded sales in ZIP {m['zip_code']} since {POOL['CloseDate'].min():%b %Y}. "
                     "Try a nearby ZIP or enter coordinates.")
            st.stop()
        lat = float(in_zip["Latitude"].median())
        lon = float(in_zip["Longitude"].median())
    else:
        lat = m["lat"]
        lon = m["lon"]
    near = nearest_sale(lat, lon)
    zip_value = m["zip_code"] if m["loc_mode"] == "ZIP code" else near["PostalCode"]

    if m["hoa"] == "Has HOA":
        fee_monthly = m["fee"]
    elif m["hoa"] == "No HOA":
        fee_monthly = 0.0
    else:
        fee_monthly = np.nan

    row = {f: np.nan for f in FEATURES}
    row.update({
        "LivingArea": m["living_area"],
        "LotSizeSquareFeet": m["lot"] if m["lot"] > 0 else np.nan,
        "BedroomsTotal": m["beds"],
        "BathroomsTotalInteger": m["baths"],
        "GarageSpaces": m["garage"],
        "YearBuilt": m["year_built"],
        "PropertyAge": max(ESTIMATE_DATE.year - m["year_built"], 0),
        "Latitude": lat,
        "Longitude": lon,
        "Latitude_missing": 0,
        "City": near["City"],
        "CountyOrParish": near["CountyOrParish"],
        "PostalCode": zip_value,
        "ElementarySchoolDistrict": near["ElementarySchoolDistrict"],
        "HighSchoolDistrict": near["HighSchoolDistrict"],
        "fee_monthly": fee_monthly,
        "HasHOA": int(fee_monthly > 0) if pd.notna(fee_monthly) else 0,
        "AssociationFee_missing": int(pd.isna(fee_monthly)),
        "Flooring_missing": int(not m["floors"]),
    })
    for f in data["flooring_types"]:
        row[f"has_{f.lower()}"] = int(f in m["floors"])
    for col in BINARY_LABELS:
        value = m["binary"].get(col, np.nan)
        row[col] = value
        row[f"{col}_missing"] = int(pd.isna(value))

    comps = find_comps(lat, lon, m["living_area"], m["year_built"], ESTIMATE_DATE)
    if USES_COMPS:
        row.update(comp_features(comps, m["living_area"]))
    if USES_MONTH:
        row["month_idx"] = float((ESTIMATE_DATE.to_period("M") - FIRST_MONTH).n)

    # Where each feature came from, for the documentation table
    typed = {"LivingArea", "LotSizeSquareFeet", "BedroomsTotal", "BathroomsTotalInteger", "GarageSpaces",
             "YearBuilt", "PoolPrivateYN", "FireplaceYN", "ViewYN", "fee_monthly"}
    typed |= {f"has_{f.lower()}" for f in data["flooring_types"]}
    derived = {"PropertyAge", "Latitude_missing", "HasHOA", "AssociationFee_missing", "Flooring_missing"}
    derived |= {f"{c}_missing" for c in BINARY_LABELS}
    looked_up = {"City", "CountyOrParish", "ElementarySchoolDistrict", "HighSchoolDistrict"}
    sources = {}
    for f in FEATURES:
        if f in typed:
            sources[f] = "Entered by you"
        elif f in derived:
            sources[f] = "Derived from your inputs"
        elif f in ("Latitude", "Longitude"):
            sources[f] = "Entered by you" if m["loc_mode"] != "ZIP code" else "Median of recent sales in your ZIP"
        elif f == "PostalCode":
            sources[f] = "Entered by you" if m["loc_mode"] == "ZIP code" else "Nearest recorded sale"
        elif f in looked_up:
            sources[f] = "Nearest recorded sale"
        elif f in SIM_COMPS:
            sources[f] = "Computed from similar recent sales"
        elif f == "month_idx":
            sources[f] = f"Estimate date ({ESTIMATE_DATE:%b %Y})"
        else:
            sources[f] = "Blank, filled with the training median"

    display = pd.Series({
        "LivingArea": m["living_area"], "LotSizeSquareFeet": row["LotSizeSquareFeet"],
        "BedroomsTotal": m["beds"], "BathroomsTotalInteger": m["baths"], "GarageSpaces": m["garage"],
        "YearBuilt": m["year_built"], "fee_monthly": fee_monthly, "City": near["City"],
        "CountyOrParish": near["CountyOrParish"], "PostalCode": zip_value,
        "ElementarySchoolDistrict": near["ElementarySchoolDistrict"],
        "HighSchoolDistrict": near["HighSchoolDistrict"],
        "Flooring": ", ".join(m["floors"]) if m["floors"] else None,
        "Latitude": lat, "Longitude": lon, **m["binary"],
    })
    subject = {
        "address": f"Entered home near {near['City']} {zip_value}",
        "row": row,
        "display": display,
        "comps": comps,
        "actual": np.nan,
        "as_of": ESTIMATE_DATE,
        "sources": sources,
    }


# ============================================================
# Prediction
# ============================================================

try:
    result = predict(subject["row"])
except Exception as exc:
    st.error("Prediction failed. The saved model and the app's feature list may not match.")
    st.exception(exc)
    st.stop()

st.subheader(subject["address"])
k1, k2, k3 = st.columns(3)
k1.metric("Estimated sale price", money(result["price"]))
k2.metric("Likely range (80%)", f"{money(result['low'])} – {money(result['high'])}")
if show_actual and pd.notna(subject["actual"]):
    error = (result["price"] - subject["actual"]) / subject["actual"]
    k3.metric("Actual sale price", money(subject["actual"]), f"estimate {error:+.1%} vs actual",
              delta_color="off")
else:
    k3.metric("Price band", result["band"])

stats = result["band_stats"]
range_note = "this band's" if stats["interval_from"] == "band" else "all homes' (this band has too few)"
st.caption(
    f"The range uses {range_note} errors on {INFO['validation_month']} validation sales: 8 in 10 sold between "
    f"these two prices. In the {result['band']} band the typical miss was {stats['MdAPE_%']:.1f}% "
    f"(MdAPE, n={int(stats['n']):,}), and the median estimate was {stats['median_bias_%']:+.1f}% vs actual."
)


# ============================================================
# Details, map and price-band chart
# ============================================================

d = subject["display"]
st.markdown("#### Home details")
c1, c2, c3 = st.columns(3)
with c1:
    st.write("Living area:", number(d.get("LivingArea")), "sq ft")
    st.write("Lot size:", number(d.get("LotSizeSquareFeet")), "sq ft")
    st.write("Bedrooms:", number(d.get("BedroomsTotal")))
    st.write("Bathrooms:", number(d.get("BathroomsTotalInteger")))
    st.write("Garage spaces:", number(d.get("GarageSpaces")))
with c2:
    st.write("Year built:", number(d.get("YearBuilt"), "{:.0f}"))
    st.write("Private pool:", yes_no(d.get("PoolPrivateYN")))
    st.write("Fireplace:", yes_no(d.get("FireplaceYN")))
    st.write("View:", yes_no(d.get("ViewYN")))
    st.write("Monthly HOA fee:", money(d.get("fee_monthly")))
with c3:
    st.write("City:", text(d.get("City")), text(d.get("PostalCode")))
    st.write("County:", text(d.get("CountyOrParish")))
    st.write("Elementary district:", text(d.get("ElementarySchoolDistrict")))
    st.write("High school district:", text(d.get("HighSchoolDistrict")))
    flooring_text = text(d.get("Flooring"))
    st.write("Flooring:", "Not reported" if flooring_text == "n/a" else flooring_text)

comps = subject["comps"]
map_col, chart_col = st.columns([3, 2])
with map_col:
    st.markdown("#### Location and similar sales")
    points = pd.DataFrame([{
        "Latitude": d["Latitude"], "Longitude": d["Longitude"], "label": "This home",
        "color": [214, 39, 40], "radius": 45,
    }])
    if not comps.empty:
        cp = comps.assign(
            label=comps["UnparsedAddress"].astype(str) + " · " + comps["ClosePrice"].map(money),
            color=[[31, 119, 180]] * len(comps),
            radius=30,
        )[["Latitude", "Longitude", "label", "color", "radius"]]
        points = pd.concat([points, cp], ignore_index=True)
    layer = pdk.Layer("ScatterplotLayer", data=points, get_position=["Longitude", "Latitude"],
                      get_fill_color="color", get_radius="radius", radius_min_pixels=5, pickable=True)
    view = pdk.ViewState(latitude=float(d["Latitude"]), longitude=float(d["Longitude"]), zoom=13)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{label}"}),
                    height=340)
    st.caption("Red: this home. Blue: the similar sales below.")

with chart_col:
    st.markdown("#### Error by price band")
    band_df = BANDS.drop(index="All").reset_index(names="band")
    band_df = band_df[band_df["n"].astype(float) > 0].copy()
    band_df["MdAPE_%"] = band_df["MdAPE_%"].astype(float)
    band_df["n"] = band_df["n"].astype(int)
    band_df["this home"] = band_df["band"] == result["band"]
    chart = alt.Chart(band_df).mark_bar().encode(
        x=alt.X("band:N", sort=data["band_labels"], title="Predicted price band"),
        y=alt.Y("MdAPE_%:Q", title="Typical error, MdAPE (%)"),
        color=alt.condition("datum['this home']", alt.value("#d62728"), alt.value("#9ecae1")),
        tooltip=["band", "n", "MdAPE_%", "MAPE_%", "median_bias_%", "within_10%"],
    ).properties(height=300)
    st.altair_chart(chart, width="stretch")
    st.caption(f"{INFO['validation_month']} validation homes, grouped by predicted price. "
               "Red is this home's band.")


# ============================================================
# Comparable properties
# ============================================================

st.markdown("#### Comparable properties")
if comps.empty:
    st.info("No similar sales (size within 25%, built within 15 years) in the 180 days before "
            f"{subject['as_of']:%b %d, %Y}. Without them, the estimate leans more on location and size.")
else:
    st.caption(f"The {len(comps)} nearest sales that closed in the 180 days before {subject['as_of']:%b %d, %Y}, "
               f"with living area within 25% and year built within 15 years. Median "
               f"{money(comps['price_per_sqft'].median())}/sq ft.")
    top = comps.head(3)
    tabs = st.tabs([f"Comp {i + 1}" for i in range(len(top))])
    for tab, (_, c) in zip(tabs, top.iterrows()):
        with tab:
            st.write(f"**{c['UnparsedAddress']}, {c['City']}** · {c['km']:.2f} km away")
            t1, t2, t3 = st.columns(3)
            t1.metric("Sold for", money(c["ClosePrice"]))
            t2.metric("Price per sq ft", money(c["price_per_sqft"]))
            t3.metric("Closed", f"{c['CloseDate']:%b %d, %Y}")
            st.write(f"{number(c['BedroomsTotal'])} bd / {number(c['BathroomsTotalInteger'])} ba · "
                     f"{number(c['LivingArea'])} sq ft · built {number(c['YearBuilt'], '{:.0f}')} · "
                     f"lot {number(c['LotSizeSquareFeet'])} sq ft")
    with st.expander("All similar sales"):
        table = comps[["UnparsedAddress", "City", "CloseDate", "ClosePrice", "price_per_sqft", "LivingArea",
                       "BedroomsTotal", "BathroomsTotalInteger", "YearBuilt", "km"]].rename(columns={
            "UnparsedAddress": "Address", "CloseDate": "Closed", "ClosePrice": "Price",
            "price_per_sqft": "$/sq ft", "LivingArea": "Sq ft", "BedroomsTotal": "Beds",
            "BathroomsTotalInteger": "Baths", "YearBuilt": "Built", "km": "km away"})
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Price": st.column_config.NumberColumn(format="$%d"),
            "$/sq ft": st.column_config.NumberColumn(format="$%d"),
            "Closed": st.column_config.DateColumn(format="MMM D, YYYY"),
            "km away": st.column_config.NumberColumn(format="%.2f"),
        })


# ============================================================
# Where each input comes from + export
# ============================================================

sources = pd.DataFrame({
    "feature": FEATURES,
    "value": [subject["row"].get(f) for f in FEATURES],
    "source": [subject["sources"][f] for f in FEATURES],
})
sources["value"] = sources["value"].map(lambda v: "blank" if v is None or (isinstance(v, float) and np.isnan(v))
                                        else (f"{v:,.4g}" if isinstance(v, (int, float, np.number)) else str(v)))

with st.expander("Where each input comes from"):
    st.write(f"The model uses {len(FEATURES)} features. In **Pick a home** mode all of them come from the "
             "sale's MLS record, with the same feature code used in training. In **Enter a home** mode you type "
             "the key inputs, location fields come from the nearest recorded sale, and anything left blank is "
             "filled with the training median inside the saved pipeline (and its missing flag is set).")
    st.dataframe(sources, hide_index=True, width="stretch")

export = pd.DataFrame([{
    "home": subject["address"],
    "estimate": round(result["price"]),
    "range_low_80": round(result["low"]),
    "range_high_80": round(result["high"]),
    "price_band": result["band"],
    "band_validation_MdAPE_%": float(stats["MdAPE_%"]),
    "model": f"{INFO['config']} XGBoost, trained through {INFO['trained_through']}",
    **{f: subject["row"].get(f) for f in FEATURES},
}])
report_lines = [
    f"# Price estimate: {subject['address']}",
    "",
    f"- Estimate: {money(result['price'])}",
    f"- Likely range (80%): {money(result['low'])} – {money(result['high'])}",
    f"- Price band: {result['band']} (validation MdAPE {stats['MdAPE_%']:.1f}%, n={int(stats['n']):,})",
    f"- Model: {INFO['config_name']}, XGBoost, {INFO['n_train']:,} sales through {INFO['trained_through']}",
    "",
    "## Similar sales",
]
for _, c in comps.iterrows():
    report_lines.append(f"- {c['UnparsedAddress']}, {c['City']}: {money(c['ClosePrice'])} on "
                        f"{c['CloseDate']:%Y-%m-%d}, {money(c['price_per_sqft'])}/sq ft, {c['km']:.2f} km")
if comps.empty:
    report_lines.append("- None found")
report_lines += ["", "## Inputs", "", "| feature | value | source |", "|---|---|---|"]
for _, s in sources.iterrows():
    report_lines.append(f"| {s['feature']} | {s['value']} | {s['source']} |")

e1, e2, e3 = st.columns([1, 1, 3])
e1.download_button("Download CSV", export.to_csv(index=False), "price_estimate.csv", "text/csv")
e2.download_button("Download report", "\n".join(report_lines), "price_estimate.md", "text/markdown")

with st.expander("About this model"):
    st.markdown(f"""
- **Model:** {INFO['config_name']} ({INFO['config']}). XGBoost on log(sale price), `max_depth=7`, seed 42,
  {INFO['n_trees']:,} trees (early stopping on 10% of training rows).
- **Data:** CRMLS California single-family sales, {INFO['first_month']} to {INFO['trained_through']},
  priced {money(INFO['price_limits'][0])} to {money(INFO['price_limits'][1])}. Cleaning and features come
  from the team's `unified_preprocessing.py` and `unified_feature_engineering.py`.
- **Validation:** the price ranges come from a model trained on sales before {INFO['validation_month']} and scored
  on {INFO['validation_month']} sales. The app's model is then refit through {INFO['trained_through']}.
- **Test month:** {INFO['demo_month']} homes are held out. Nothing here tunes on them.
- **Limits:** the data has no interior condition or renovation quality, which drives much of the error on
  high-end homes. School districts are the MLS-reported ones, and they're missing for some homes.
- Built {INFO['built']}.
""")

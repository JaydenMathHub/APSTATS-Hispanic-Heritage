import ast
import io
import math
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as PDFImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📊", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Expected years lived by a newborn under prevailing age-specific mortality rates.", "nonnegative"),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Unemployed people as a percentage of the labor force using national estimates.", "percent"),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "GDP divided by midyear population; not average wealth or household income.", "nonnegative"),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; a rate, not a total birth count.", "nonnegative"),
    "Net migration": ("SM.POP.NETM", "people over the period", "Net number of migrants over the period; not emigration alone.", "any"),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "victims per 100,000 people", "Intentional homicide victims per 100,000 people.", "nonnegative"),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index.", "any"),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP.", "any"),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, valued in current US dollars.", "nonnegative"),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP.", "any"),
}
API_ROOT = "https://api.worldbank.org/v2"
CONTEXTS = {
    "Business": ("Revenue", "currency", "Time", "months", "A simplified revenue-over-time model."),
    "Engineering": ("Position", "m", "Time", "seconds", "Position along one axis."),
    "Medical": ("Concentration", "mg/L", "Time", "hours", "Illustrative concentration curve; not medical advice."),
    "Sports science": ("Speed", "m/s", "Time", "seconds", "Illustrative performance measure over time."),
    "Biology": ("Population", "organisms", "Time", "days", "Illustrative population or cell colony."),
    "Bioengineering": ("Signal", "signal units", "Time", "seconds", "Illustrative sensor signal."),
    "Economic forecasting": ("Index", "index points", "Time", "years", "Illustrative economic indicator; extrapolation is uncertain."),
    "Weather": ("Temperature", "°C", "Time", "hours", "Illustrative temperature-over-time model."),
    "Custom": ("y", "units", "x", "x units", "Describe your own application."),
}
SUGGESTIONS = {
    "Polynomial": "0.08*x**3 - 0.6*x**2 + 2*x + 5",
    "Rational": "(2*x + 3)/(x + 1)",
    "Exponential": "3*exp(0.15*x)",
    "Logarithmic": "4*log(x + 2)",
    "Trigonometric": "10 + 3*sin(2*pi*x/24)",
}
SAFE_FUNCTIONS = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
                  "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi, "e": np.e}
ALLOWED_AST = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Call, ast.Mod)

def api_url(iso3, indicator, start, end):
    return f"{API_ROOT}/country/{iso3}/indicator/{indicator}?date={start}:{end}&format=json&per_page=200"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(iso3, indicator, start, end):
    url = api_url(iso3, indicator, start, end)
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Hispanic-Heritage-Data-Explorer/1.0"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["Year", "Value"]), url, "The API returned an empty or unexpected response."
        rows = [{"Year": int(r["date"]), "Value": r.get("value")} for r in payload[1] if r.get("date")]
        df = pd.DataFrame(rows)
        if df.empty:
            return df, url, "No observations were returned for this country and indicator."
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Value"]).sort_values("Year").reset_index(drop=True)
        if df.empty:
            return df, url, "The API returned no non-missing observations in the requested period."
        return df, url, None
    except requests.Timeout:
        return pd.DataFrame(columns=["Year", "Value"]), url, "The World Bank API request timed out. Please retry later."
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return pd.DataFrame(columns=["Year", "Value"]), url, f"Unable to retrieve valid data: {exc}"

def validate_data(df, constraint):
    errors = []
    if not {"Year", "Value"}.issubset(df.columns):
        return ["The working table must have Year and Value columns."]
    years = pd.to_numeric(df["Year"], errors="coerce")
    vals = pd.to_numeric(df["Value"], errors="coerce")
    if years.isna().any() or not np.equal(years.dropna(), np.floor(years.dropna())).all():
        errors.append("Every year must be a whole-number integer.")
    if years.duplicated().any():
        errors.append("Duplicate years are not allowed.")
    if vals.isna().any() or not np.isfinite(vals.dropna()).all():
        errors.append("Every value must be a finite numeric value.")
    if constraint == "nonnegative" and (vals < 0).any():
        errors.append("This indicator must be nonnegative.")
    if constraint == "percent" and ((vals < 0) | (vals > 100)).any():
        errors.append("Unemployment percentages must be between 0 and 100.")
    if len(df) == 0:
        errors.append("At least one row is required.")
    return errors

def fit_model(df, degree):
    df = df.sort_values("Year").copy()
    if df["Year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct years.")
    base = int(df["Year"].min())
    t = (df["Year"].to_numpy(dtype=float) - base).reshape(-1, 1)
    y = df["Value"].to_numpy(dtype=float)
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
    model.fit(t, y)
    fitted = model.predict(t)
    return model, base, df, fitted, float(r2_score(y, fitted)), float(mean_absolute_error(y, fitted)), y - fitted

def predict(model, base, year):
    return float(model.predict(np.array([[float(year - base)]], dtype=float))[0])

def predict_continuous(model, base, x):
    values = np.asarray(x, dtype=float)
    flat = (values.reshape(-1) - float(base)).reshape(-1, 1)
    result = model.predict(flat).reshape(values.shape)
    return float(result) if result.shape == () else result

def equation(model, base, degree):
    lr = model.named_steps["linearregression"]
    coefs = np.zeros(degree + 1)
    coefs[0] = lr.intercept_
    coefs[1:] = lr.coef_
    terms = [f"{coefs[0]:.8g}"]
    for p, c in enumerate(coefs[1:], 1):
        if abs(c) > 1e-12:
            terms.append(f"{c:+.8g}·t" + (f"^{p}" if p > 1 else ""))
    return "y = " + " ".join(terms) + f", t = year − {base}"

def safe_expression(expr, x):
    expr = expr.strip().replace("^", "**")
    if not expr or any(ch in expr for ch in ("__", "[", "]", "{", "}", ";", "=")):
        raise ValueError("Enter a mathematical expression using x, numbers, arithmetic, parentheses, and supported functions.")
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST):
            raise ValueError("Unsupported expression syntax.")
        if isinstance(node, ast.Name) and node.id not in {"x", *SAFE_FUNCTIONS.keys()}:
            raise ValueError(f"Unsupported name: {node.id}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS or node.func.id in {"pi", "e"}):
            raise ValueError("Only sin, cos, tan, exp, log, sqrt, and abs may be called.")
    return eval(compile(tree, "<math-expression>", "eval"), {"__builtins__": {}}, {"x": x, **SAFE_FUNCTIONS})

def scalar(fn, x):
    try:
        v = float(np.asarray(fn(float(x))).reshape(-1)[0])
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")

def derivative(fn, x, order=1):
    h = max(1e-4, abs(x)*1e-5)
    if order == 1:
        return (scalar(fn, x+h) - scalar(fn, x-h))/(2*h)
    return (scalar(fn, x+h) - 2*scalar(fn, x) + scalar(fn, x-h))/(h*h)

def pdf_bytes(summary, png, data, results):
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=landscape(letter), rightMargin=25, leftMargin=25, topMargin=25, bottomMargin=25)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]), Spacer(1, 8)]
    for k, v in summary.items():
        story.append(Paragraph(f"<b>{k}:</b> {str(v)}", styles["BodyText"]))
    story.extend([Spacer(1, 8), PDFImage(io.BytesIO(png), width=650, height=320), Spacer(1, 8),
                  Paragraph("Working data preview", styles["Heading2"])])
    rows = [["Year", "Value"]] + [[str(int(r.Year)), f"{float(r.Value):.7g}"] for r in data.head(18).itertuples()]
    tbl = Table(rows, repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightblue),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8)]))
    story.append(tbl)
    if not results.empty:
        story.extend([Spacer(1, 8), Paragraph("Calculated-year results", styles["Heading2"])])
        result_rows = [list(results.columns)] + results.astype(str).values.tolist()
        tbl2 = Table(result_rows[:25], repeatRows=1)
        tbl2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightblue),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(tbl2)
    doc.build(story)
    return out.getvalue()

st.title("📊 Hispanic Heritage Historical Data Explorer")
st.caption("Hispanic Heritage Data Lab · Official World Bank indicators, polynomial modeling, and a classroom function studio")
with st.expander("Project question and model settings", expanded=True):
    question = st.text_input("Research question", "How has life expectancy changed over time in Mexico, and how does it compare with another Latin American country?")
    pathway = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    country = st.selectbox("Featured country", list(COUNTRIES))
    indicator_name = st.selectbox("Indicator", list(INDICATORS))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    comparison_name = st.selectbox("Comparison country", ["None"] + [c for c in COUNTRIES if c != country])
    printer_mode = st.checkbox("Printer-friendly view", False)

code, units, definition, constraint = INDICATORS[indicator_name]
this_year = date.today().year
requested_start, requested_end = this_year - 70, this_year - 1
source, source_url, error = fetch_data(COUNTRIES[country], code, requested_start, requested_end)
if error:
    st.error(error)
    st.markdown(f"[Open the World Bank API request]({source_url})")
    st.info("No random or substitute data are used. Modeling is paused until authentic observations are available.")
    st.stop()
st.markdown(f"**Definition:** {definition}  \n**Units:** {units}  \n**Indicator code:** `{code}`  \n**Source:** [World Bank API]({source_url})  \n**Retrieved:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}")
st.caption(f"Requested period: {requested_start}–{requested_end}. Actual coverage: {source.Year.min()}–{source.Year.max()}, {len(source)} non-missing observations. Brazil is a Latin American comparison, not a Spanish-speaking country.")

signature = f"{COUNTRIES[country]}|{code}|{requested_start}|{requested_end}"
if st.session_state.get("source_signature") != signature:
    st.session_state.source_signature = signature
    st.session_state.source_data = source.copy()
    st.session_state.working_data = source.copy()
    st.session_state.edited = False
st.subheader("1. Source data and editable working table")
st.write("The retrieved source data remain separate from the editable working table. Edits are treated as a user-modified scenario.")
edited = st.data_editor(st.session_state.working_data, num_rows="dynamic", use_container_width=True, key=f"editor_{signature}")
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Reset to source data"):
        st.session_state.working_data = st.session_state.source_data.copy()
        st.session_state.edited = False
        st.rerun()
with c2:
    st.session_state.working_data = edited.copy()
    st.session_state.edited = not edited.reset_index(drop=True).equals(source.reset_index(drop=True))
    if st.session_state.edited:
        st.warning("User-modified scenario: model and exports use the edited working data.")
    else:
        st.success("Working data matches the retrieved source observations.")
errors = validate_data(edited, constraint)
if errors:
    for e in errors:
        st.error(e)
    st.stop()
working = edited.copy()
working["Year"] = pd.to_numeric(working.Year).astype(int)
working["Value"] = pd.to_numeric(working.Value).astype(float)
working = working.sort_values("Year").reset_index(drop=True)
if working.Year.nunique() < degree + 1:
    st.error(f"Degree {degree} requires at least {degree+1} distinct years; current count is {working.Year.nunique()}.")
    st.stop()
model, base, sorted_data, fitted, r2, mae, residuals = fit_model(working, degree)
eq = equation(model, base, degree)
fit_data = sorted_data.copy()
fit_data["Fitted value"] = fitted
fit_data["Residual (actual − fitted)"] = residuals
xmin, xmax = int(working.Year.min()), int(working.Year.max())
summary = {
    "Country": country, "Indicator": indicator_name, "Units": units, "Degree": degree, "Base year": base,
    "Equation": eq, "In-sample R²": f"{r2:.6f}", "In-sample MAE": f"{mae:.7g} {units}",
    "Observations": len(working), "Historical fitted range": f"{xmin}–{xmax}",
    "Working-data mean": f"{working.Value.mean():.7g} {units}",
    "Data status": "User-modified scenario" if st.session_state.edited else "Source data",
}
st.subheader("2. Fit summary")
a1, a2, a3, a4 = st.columns(4)
a1.metric("Observations", len(working)); a2.metric("In-sample R²", f"{r2:.4f}")
a3.metric("In-sample MAE", f"{mae:.4g}"); a4.metric("Mean", f"{working.Value.mean():.5g}")
st.markdown(f"**Centered-time equation:** `{eq}`")
st.write("The model uses `t = year − base year`. R² and MAE are in-sample fit metrics, not causal evidence or prediction intervals.")
if len(working) >= max(12, degree+5):
    hold_n = min(5, max(2, len(working)//5))
    train, test = working.iloc[:-hold_n], working.iloc[-hold_n:]
    if train.Year.nunique() >= degree+1:
        hm, hb, _, _, _, _, _ = fit_model(train, degree)
        hold_preds = [predict(hm, hb, int(y)) for y in test.Year]
        st.write(f"**Chronological holdout MAE:** {mean_absolute_error(test.Value, hold_preds):.7g} {units}, evaluated on the final {len(test)} years and separate from in-sample metrics.")
st.caption("Residual = actual − fitted. Polynomial extrapolations can be implausible; MAE is not a validated prediction interval.")

st.subheader("3. Single-year and batch estimates")
left, right = st.columns(2)
with left:
    year_one = st.number_input("Single year (1950–2100)", 1950, 2100, int(np.clip(xmax, 1950, 2100)), 1)
    do_calc = st.button("Calculate Value", type="primary")
with right:
    batch_type = st.radio("Batch format", ["Year range", "Custom years"], horizontal=True)
    if batch_type == "Year range":
        r1, r2 = st.columns(2)
        with r1: year_start = st.number_input("Range start", 1950, 2100, int(np.clip(xmin, 1950, 2100)), 1)
        with r2: year_end = st.number_input("Range end", 1950, 2100, int(np.clip(xmax, 1950, 2100)), 1)
        range_step = st.slider("Range step", 1, 10, 1)
        custom_year_text = ""
    else:
        custom_year_text = st.text_input("Comma-separated custom years", f"{xmin}, {xmax}, 2050")
        year_start, year_end, range_step = 1950, 1950, 1

data_hash = hash(pd.util.hash_pandas_object(working[["Year", "Value"]], index=True).values.tobytes())
model_key = f"{country}|{indicator_name}|{degree}|{data_hash}"
if st.session_state.get("model_key") != model_key:
    st.session_state.model_key = model_key
    st.session_state.single_result = None
if do_calc:
    st.session_state.single_result = {"year": int(year_one), "value": predict(model, base, int(year_one)), "key": model_key}
single = st.session_state.get("single_result")
if single and single["key"] == model_key:
    label = "within fitted historical range" if xmin <= single["year"] <= xmax else "outside fitted historical range (extrapolation)"
    st.success(f"{single['year']}: {single['value']:.8g} {units} — {label}. This is an estimate, not an observed value.")
batch_years, invalid_tokens = [], []
if batch_type == "Year range":
    if year_start > year_end:
        st.error("Range start must be less than or equal to range end.")
    else:
        batch_years = list(range(int(year_start), int(year_end)+1, int(range_step)))
else:
    for token in custom_year_text.split(","):
        token = token.strip()
        if not token: continue
        try:
            v = float(token)
            if not v.is_integer() or not 1950 <= int(v) <= 2100: invalid_tokens.append(token)
            else: batch_years.append(int(v))
        except ValueError: invalid_tokens.append(token)
    batch_years = sorted(set(batch_years))
if invalid_tokens:
    st.warning("Ignored invalid year token(s): " + ", ".join(invalid_tokens) + ". Use whole years from 1950 to 2100.")
batch_results = pd.DataFrame()
if batch_years:
    batch_results = pd.DataFrame({"Year": batch_years, "Estimate": [predict(model, base, y) for y in batch_years]})
    batch_results["Units"] = units
    batch_results["Classification"] = batch_results.Year.apply(lambda y: "within fitted historical range" if xmin <= y <= xmax else "outside fitted historical range (extrapolation)")
    st.dataframe(batch_results, use_container_width=True)
else:
    st.info("No valid batch years to display.")

comparison = pd.DataFrame()
if comparison_name != "None":
    comparison, comp_url, comp_error = fetch_data(COUNTRIES[comparison_name], code, requested_start, requested_end)
    if comp_error:
        st.warning(f"Comparison data unavailable: {comp_error}")
    else:
        st.caption(f"Comparison source: [{comparison_name} World Bank API]({comp_url}). Coverage may differ due to missing observations.")

st.subheader("4. Historical plot")
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.scatter(working.Year, working.Value, label="Observed / working data", s=28, zorder=3)
curve_x = np.linspace(xmin, xmax+horizon, max(250, (xmax+horizon-xmin+1)*3))
curve_y = predict_continuous(model, base, curve_x)
hist_mask = curve_x <= xmax
ax.plot(curve_x[hist_mask], curve_y[hist_mask], label="Fitted curve (historical)", linewidth=2)
if (~hist_mask).any(): ax.plot(curve_x[~hist_mask], curve_y[~hist_mask], "--", label="Extrapolation (uncertain)", linewidth=2)
samples = np.arange(xmin, xmax+1, sample_step)
ax.scatter(samples, [predict(model, base, int(y)) for y in samples], marker="x", label=f"Model samples every {sample_step} year(s)")
ax.axvline(xmax, color="gray", linestyle=":", label=f"Historical boundary ({xmax})")
if single and single["key"] == model_key:
    ax.scatter([single["year"]], [single["value"]], marker="*", s=130, label="Single-year estimate")
if batch_years: ax.scatter(batch_years, [predict(model, base, y) for y in batch_years], marker="D", s=28, label="Batch estimates")
if not comparison.empty: ax.scatter(comparison.Year, comparison.Value, marker="s", alpha=.6, label=f"{comparison_name} observations")
ax.set_title(f"{country} — {indicator_name}"); ax.set_xlabel("Year"); ax.set_ylabel(units)
ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
st.pyplot(fig, use_container_width=True)
png_buffer = io.BytesIO(); fig.savefig(png_buffer, format="png", dpi=150, bbox_inches="tight"); chart_png = png_buffer.getvalue()
plt.close(fig)

st.subheader("5. Mathematics pathway")
if pathway == "Honors Algebra 2":
    st.markdown(f"- **Observed domain:** {xmin}–{xmax}; the polynomial formula itself is defined for all real inputs.\n- **Observed value range:** {working.Value.min():.6g} to {working.Value.max():.6g} {units}.\n- **Average rate of change:** `(f(b) − f(a))/(b − a)`, in {units} per year.\n- Use the graph to describe trends; do not infer causes from the curve.")
elif pathway == "Precalculus":
    st.markdown(f"- Degree {degree} affects shape and permits at most {degree-1} turning points.\n- Concavity can be discussed by observing how average rates change across intervals.\n- Coefficients affect vertical scale and curve shape; extrapolation is not a validated forecast.")
else:
    st.markdown("The Function Studio below includes numerical first/second derivatives, tangent and secant lines, and approximate critical/inflection candidates. Numerical results are not symbolic proofs.")

st.subheader("6. Function studio")
context = st.selectbox("Application context", list(CONTEXTS), index=1)
default_y, default_y_unit, default_x, default_x_unit, default_desc = CONTEXTS[context]
c1, c2, c3 = st.columns(3)
with c1:
    y_name = st.text_input("Dependent variable name", default_y)
    y_unit = st.text_input("Dependent variable units", default_y_unit)
with c2:
    x_name = st.text_input("Independent variable name", default_x)
    x_unit = st.text_input("Independent variable units", default_x_unit)
with c3:
    context_desc = st.text_input("Context description", default_desc)
function_mode = st.radio("Choose function source", ["Use data curve of best fit", "Suggested function", "Enter my own function"], horizontal=True)
expression = None
if function_mode == "Suggested function":
    family = st.selectbox("Function family", list(SUGGESTIONS))
    expression = st.text_input("Suggested expression (editable)", SUGGESTIONS[family])
elif function_mode == "Enter my own function":
    expression = st.text_input("Expression in x", "0.08*x**3 - 0.6*x**2 + 2*x + 5")

if function_mode == "Use data curve of best fit":
    def fn(x):
        arr = np.asarray(x, dtype=float)
        return predict_continuous(model, base, arr)
    x_label, x_units = "Year", "calendar year"
    x_lo, x_hi = float(xmin), float(xmax+horizon)
    default_point = float(np.clip(xmax, x_lo, x_hi))
else:
    try:
        check = safe_expression(expression, np.array([1.0, 2.0]))
        if not np.all(np.isfinite(np.asarray(check, dtype=float))):
            raise ValueError("The expression returned a non-finite test value.")
        fn = lambda x: safe_expression(expression, x)
        x_label, x_units = x_name, x_unit
        q1, q2 = st.columns(2)
        with q1: x_lo = float(st.number_input("Analysis interval minimum", value=0.0, step=1.0))
        with q2: x_hi = float(st.number_input("Analysis interval maximum", value=10.0, step=1.0))
        if x_hi <= x_lo:
            st.error("Analysis maximum must be greater than minimum."); st.stop()
        default_point = (x_lo+x_hi)/2
    except Exception as exc:
        st.error(f"Could not interpret function: {exc}"); st.stop()
if x_hi <= x_lo:
    st.error("The analysis interval must have positive width."); st.stop()
st.caption("Custom syntax: +, -, *, /, ** (or ^), parentheses, x, sin, cos, tan, exp, log, sqrt, abs, pi, e. Use explicit multiplication such as 2*x.")
point = st.slider("Current input value", min_value=float(x_lo), max_value=float(x_hi), value=float(np.clip(default_point, x_lo, x_hi)))
c1, c2 = st.columns(2)
with c1: a = st.slider("Average-rate interval start (a)", min_value=float(x_lo), max_value=float(x_hi), value=float(x_lo))
with c2: b = st.slider("Average-rate interval end (b)", min_value=float(x_lo), max_value=float(x_hi), value=float(x_hi))
show_f = st.checkbox("Show f(x)", True)
show_d1 = st.checkbox("Show f′(x)", pathway == "Calculus")
show_d2 = st.checkbox("Show f″(x)", pathway == "Calculus")
play = st.toggle("Play animation (advances on rerun)", False)
speed = st.select_slider("Animation speed", options=["Slow", "Normal", "Fast"], value="Normal")
if st.button("Reset point"): st.session_state["reset_point"] = True; st.rerun()
if st.session_state.pop("reset_point", False): point = float(np.clip(default_point, x_lo, x_hi))
value = scalar(fn, point); d1 = derivative(fn, point, 1); d2 = derivative(fn, point, 2)
fa, fb = scalar(fn, a), scalar(fn, b)
avg = (fb-fa)/(b-a) if b != a and np.isfinite(fa) and np.isfinite(fb) else float("nan")
st.markdown(f"**Function:** `{eq if function_mode == 'Use data curve of best fit' else expression}`  \n**Context:** {context_desc}")
if np.isfinite(value):
    st.write(f"At {x_label} = {point:.5g} {x_units}, **f(x) ≈ {value:.7g} {y_unit}**.")
    st.write(f"**f′(x) ≈ {d1:.7g} {y_unit} per {x_units}; f″(x) ≈ {d2:.7g} {y_unit} per ({x_units})².**")
else:
    st.warning("The function is undefined or non-finite at this point. Try a point in its domain.")
if np.isfinite(avg):
    st.write(f"**Average rate of change on [{a:.5g}, {b:.5g}]:** {avg:.7g} {y_unit} per {x_units}.")
else:
    st.info("Average rate unavailable if the endpoints are equal or either endpoint is outside the function's finite domain.")
if context == "Engineering" and y_name.lower() == "position":
    st.info("If f represents position, f′ is velocity and f″ is acceleration. Their units are distance/time and distance/time².")
elif context in {"Business", "Economic forecasting"}:
    st.info("Positive f′ means the modeled quantity is increasing; negative f′ means decreasing. f″ describes whether the rate itself is increasing or decreasing. This is not a causal explanation.")
else:
    st.info(f"f is modeled {y_name}; f′ describes its instantaneous rate of change and f″ describes how that rate changes. Units follow your labels.")

xs = np.linspace(x_lo, x_hi, 800)
with np.errstate(all="ignore"):
    ys = np.asarray(fn(xs), dtype=float)
if ys.shape == (): ys = np.full_like(xs, float(ys))
ys[~np.isfinite(ys)] = np.nan
dys = np.array([derivative(fn, float(z), 1) for z in xs])
ddys = np.array([derivative(fn, float(z), 2) for z in xs])
fig2, ax2 = plt.subplots(figsize=(10, 5))
if show_f: ax2.plot(xs, ys, label="f(x)")
if show_d1: ax2.plot(xs, dys, label="f′(x), numerical")
if show_d2: ax2.plot(xs, ddys, label="f″(x), numerical")
if np.isfinite(value):
    ax2.scatter([point], [value], s=55, label="Current point", zorder=4)
    if np.isfinite(d1):
        tx = np.linspace(max(x_lo, point-(x_hi-x_lo)*.12), min(x_hi, point+(x_hi-x_lo)*.12), 40)
        ax2.plot(tx, value+d1*(tx-point), "--", label="Tangent line")
if np.isfinite(fa) and np.isfinite(fb) and a != b:
    ax2.plot([a,b], [fa,fb], ":", marker="o", label="Secant line")
ax2.set_xlabel(f"{x_label} ({x_units})"); ax2.set_ylabel("Function/derivative values (different units)")
ax2.set_title(f"Function analysis — {context}"); ax2.grid(alpha=.25); ax2.legend(fontsize=8); fig2.tight_layout()
st.pyplot(fig2, use_container_width=True); plt.close(fig2)

if pathway == "Calculus":
    st.markdown("#### Numerical critical points and concavity changes")
    finite = np.isfinite(ys) & np.isfinite(dys) & np.isfinite(ddys)
    extrema, inflections = [], []
    for i in range(1, len(xs)-1):
        if finite[i-1] and finite[i] and finite[i+1]:
            if dys[i-1]*dys[i+1] < 0:
                x0 = float(xs[i]); dl = derivative(fn, x0-1e-3); dr = derivative(fn, x0+1e-3)
                kind = "local minimum candidate" if dl < 0 < dr else ("local maximum candidate" if dl > 0 > dr else "unclassified stationary candidate")
                extrema.append({"x (approx.)": x0, "f(x)": scalar(fn, x0), "classification": kind})
            if ddys[i-1]*ddys[i+1] < 0:
                x0 = float(xs[i]); inflections.append({"x (approx.)": x0, "f(x)": scalar(fn, x0), "note": "second derivative changes sign on grid"})
    st.write("Candidate points are numerical approximations; zeros are classified only when nearby derivative signs change.")
    if extrema: st.dataframe(pd.DataFrame(extrema).round(6), use_container_width=True)
    else: st.write("No local-extremum candidates detected on this interval.")
    if inflections: st.dataframe(pd.DataFrame(inflections).round(6), use_container_width=True)
    else: st.write("No concavity-change candidates detected on this interval.")
    finite_y = ys[np.isfinite(ys)]
    if len(finite_y): st.write(f"Sampled closed-interval extrema: min ≈ {np.min(finite_y):.7g}; max ≈ {np.max(finite_y):.7g}. Grid-based estimates, not symbolic proofs.")
    st.caption("Finite differences can miss roots or behave poorly near singularities. Domain restrictions and discontinuities are not guaranteed to be fully detected numerically.")

st.subheader("7. Contextual visual simulation")
visual = st.selectbox("Visual simulation", ["Walking person with prosthetic leg", "Moving object / projectile", "Tank or drug concentration", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"])
finite_y = ys[np.isfinite(ys)]
level = .5 if not np.isfinite(value) or not len(finite_y) or np.ptp(finite_y) == 0 else float(np.clip((value-np.min(finite_y))/np.ptp(finite_y), 0, 1))
st.caption("The visual maps the current function value linearly to 0–100% of the displayed finite range. This is a visual scale, not a physical unit conversion.")
if visual == "Walking person with prosthetic leg":
    angle = -30 + 60*level
    st.markdown(f"🧍 **Illustrative gait phase:** {angle:.1f}° · modeled value {value:.5g} {y_unit}" if np.isfinite(value) else "🧍 Function undefined at this input.")
    svg = '<svg viewBox="0 0 420 180" width="100%" xmlns="http://www.w3.org/2000/svg"><line x1="20" y1="150" x2="400" y2="150" stroke="#888" stroke-width="2"/><circle cx="205" cy="30" r="14" fill="#e5a66a"/><line x1="205" y1="44" x2="205" y2="95" stroke="#2878b5" stroke-width="8" stroke-linecap="round"/><line x1="205" y1="58" x2="175" y2="83" stroke="#2878b5" stroke-width="6" stroke-linecap="round"/><line x1="205" y1="58" x2="235" y2="80" stroke="#2878b5" stroke-width="6" stroke-linecap="round"/><line x1="205" y1="95" x2="185" y2="125" stroke="#2878b5" stroke-width="7" stroke-linecap="round"/><line x1="185" y1="125" x2="180" y2="145" stroke="#555" stroke-width="7" stroke-linecap="round"/><line x1="205" y1="95" x2="235" y2="125" stroke="#2878b5" stroke-width="7" stroke-linecap="round"/><line x1="235" y1="125" x2="245" y2="145" stroke="#d97706" stroke-width="8" stroke-linecap="round"/><rect x="235" y="139" width="24" height="7" rx="3" fill="#555"/><text x="20" y="22" font-size="12" fill="currentColor">Illustrative motion, not a clinical gait model</text></svg>'
    st.markdown(svg, unsafe_allow_html=True)
else:
    labels = {"Moving object / projectile":"🎯 Object position", "Tank or drug concentration":"🧪 Tank/concentration level",
              "Population / cell colony":"🦠 Population display", "Thermometer / weather":"🌡️ Weather indicator",
              "Business / economic indicator":"📈 Business/economic indicator"}
    st.write(f"{labels[visual]}: {level*100:.1f}% of visual scale.")
    st.progress(level)

st.subheader("8. Residuals and downloads")
with st.expander("View fitted values and residuals"):
    st.dataframe(fit_data, use_container_width=True)
st.download_button("Download CSV", fit_data.to_csv(index=False).encode("utf-8"), "hispanic_heritage_fit_residuals.csv", "text/csv")
xlsx = io.BytesIO()
with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
    pd.DataFrame([{"Research question": question, **summary, "Source URL": source_url, "Retrieved": date.today().isoformat()}]).to_excel(writer, sheet_name="Summary", index=False)
    working.to_excel(writer, sheet_name="Working Data", index=False)
    (batch_results if not batch_results.empty else pd.DataFrame(columns=["Year","Estimate","Units","Classification"])).to_excel(writer, sheet_name="Calculated Years", index=False)
    fit_data.to_excel(writer, sheet_name="Fit and Residuals", index=False)
st.download_button("Download Excel workbook (.xlsx)", xlsx.getvalue(), "hispanic_heritage_explorer.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
pdf_summary = {**summary, "Research question": question, "Source URL": source_url, "Retrieved date": date.today().isoformat(),
               "Limitations": "Descriptive polynomial fit; extrapolation uncertain; MAE is not a prediction interval."}
st.download_button("Download PDF report", pdf_bytes(pdf_summary, chart_png, working, batch_results), "hispanic_heritage_report.pdf", "application/pdf")
if printer_mode:
    st.markdown("---")
    st.markdown("### Printer-friendly report")
    st.write(f"**Question:** {question}")
    st.write(f"**Finding:** {country}, {indicator_name}; {len(working)} observations over {xmin}–{xmax}; degree {degree}; in-sample R² {r2:.4f}; MAE {mae:.5g} {units}.")
    st.write(f"**Equation:** {eq}")
    st.write("**Limitations:** No causal claim is inferred. Polynomial extrapolations may be implausible.")
    st.dataframe(working.head(20), use_container_width=True)
st.caption("Source: World Bank Indicators API. Brazil is included as a Latin American comparison. This is a classroom tool, not an official forecast or medical decision tool.")

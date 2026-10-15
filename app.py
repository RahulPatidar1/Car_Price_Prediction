"""
AURUM  |  Luxury Car Valuation Studio
------------------------------------------------------------
Run with:   streamlit run app.py

Keep these files in the SAME folder as app.py:
    car_price_model.pkl   scaler.pkl   columns.pkl

Optional: drop your own premium photos into a folder called  car_images/
(jpg / png / webp). They are used for the hero slider first.
"""
import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path
from string import Template
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(
    page_title="AURUM · Luxury Car Valuation",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
REF_YEAR = 2026                      # the notebook computed Car_Age_Years = 2026 - Year
POLY_COLS = ["Kms_Driven", "Engine_CC", "Power_BHP", "Car_Age_Years"]
PRICE_FLOOR = 5.0                    # lakh; a linear model can dip below zero on odd inputs

# ----------------------------------------------------------------------------
# Catalogue: brand -> model -> (body, fuel, engine cc, power bhp, mileage, seats)
# These are typical values used to auto-fill the form (all editable by the user).
# ----------------------------------------------------------------------------
CATALOG = {
    "Audi": {
        "A4": ("Sedan", "Petrol", 1984, 187.0, 14.0, 5),
        "A6": ("Sedan", "Petrol", 1984, 241.0, 13.0, 5),
        "Q3": ("SUV", "Petrol", 1984, 187.0, 14.5, 5),
        "Q5": ("SUV", "Diesel", 1968, 190.0, 15.0, 5),
        "Q7": ("SUV", "Diesel", 2967, 250.5, 15.1, 7),
    },
    "BMW": {
        "3 Series": ("Sedan", "Petrol", 1998, 255.2, 12.0, 5),
        "5 Series": ("Sedan", "Petrol", 1998, 250.0, 12.0, 5),
        "7 Series": ("Sedan", "Petrol", 2998, 335.0, 10.0, 5),
        "X1": ("SUV", "Diesel", 1995, 190.0, 16.0, 5),
        "X5": ("SUV", "Petrol", 2998, 325.0, 10.8, 5),
    },
    "Bentley": {"Continental GT": ("Coupe", "Petrol", 5998, 626.0, 7.0, 4)},
    "Ferrari": {"Portofino": ("Coupe", "Petrol", 3855, 581.9, 8.5, 4)},
    "Jaguar": {
        "F-Pace": ("SUV", "Diesel", 1999, 177.0, 14.5, 5),
        "XF": ("Sedan", "Diesel", 1999, 177.0, 15.0, 5),
    },
    "Jeep": {"Grand Cherokee": ("SUV", "Diesel", 2987, 247.0, 11.0, 5)},
    "Lamborghini": {"Urus": ("SUV", "Petrol", 3996, 638.9, 7.0, 5)},
    "Land Rover": {
        "Discovery Sport": ("SUV", "Diesel", 1999, 176.4, 16.2, 7),
        "Range Rover": ("SUV", "Diesel", 2993, 335.0, 12.0, 5),
        "Range Rover Evoque": ("SUV", "Diesel", 1999, 177.0, 15.0, 5),
    },
    "Lexus": {
        "ES 300h": ("Sedan", "Hybrid", 2487, 214.0, 17.0, 5),
        "LX 570": ("SUV", "Petrol", 5663, 383.0, 7.0, 7),
        "NX 300h": ("SUV", "Hybrid", 2494, 194.0, 17.0, 5),
    },
    "Maserati": {"Ghibli": ("Sedan", "Petrol", 2979, 345.0, 9.0, 5)},
    "Mercedes-Benz": {
        "C-Class": ("Sedan", "Petrol", 1991, 194.0, 14.0, 5),
        "E-Class": ("Sedan", "Diesel", 1950, 191.0, 15.0, 5),
        "S-Class": ("Sedan", "Petrol", 2999, 362.0, 10.0, 5),
        "GLC": ("SUV", "Diesel", 1950, 191.0, 15.0, 5),
        "GLE": ("SUV", "Diesel", 2925, 330.0, 12.0, 5),
    },
    "Mini": {"Cooper S": ("Hatchback", "Petrol", 1998, 189.0, 14.0, 4)},
    "Porsche": {
        "911": ("Coupe", "Petrol", 2981, 379.0, 9.0, 4),
        "Cayenne": ("SUV", "Petrol", 2995, 335.0, 10.0, 5),
        "Macan": ("SUV", "Petrol", 1984, 248.0, 11.0, 5),
    },
    "Rolls-Royce": {"Ghost": ("Sedan", "Petrol", 6592, 563.0, 6.0, 5)},
    "Volvo": {
        "XC60": ("SUV", "Diesel", 1969, 187.0, 16.0, 5),
        "XC90": ("SUV", "Diesel", 1969, 232.0, 14.0, 7),
    },
}
BODIES = ["Coupe", "Hatchback", "SUV", "Sedan"]
FUELS = ["Petrol", "Diesel", "Hybrid"]
OWNERS = ["First", "Second", "Third"]
TRANS = ["Automatic", "Manual"]
SEATS = [4, 5, 7]
YEARS = list(range(2025, 2011, -1))
N_MODELS = sum(len(v) for v in CATALOG.values())

# ----------------------------------------------------------------------------
# Model loading + prediction (reproduces the notebook pipeline exactly)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    def find(name):
        for folder in (BASE_DIR, Path.cwd()):
            if (folder / name).exists():
                return folder / name
        raise FileNotFoundError(name)

    return (
        joblib.load(find("car_price_model.pkl")),
        joblib.load(find("scaler.pkl")),
        joblib.load(find("columns.pkl")),
    )


try:
    load_artifacts()
except Exception as exc:  # friendly message instead of a stack trace
    st.error(
        "Could not load the model files. Place **car_price_model.pkl**, "
        "**scaler.pkl** and **columns.pkl** in the same folder as app.py.\n\n"
        f"Details: {exc}"
    )
    st.stop()


def predict_many(rows):
    """rows: list of dicts -> numpy array of prices in Lakh INR."""
    model, scaler, cols = load_artifacts()
    recs = []
    for p in rows:
        r = dict.fromkeys(cols, 0)
        r.update(
            Year=p["year"],
            Car_Age_Years=p.get("age", REF_YEAR - p["year"]),
            Kms_Driven=p["kms"],
            Engine_CC=p["cc"],
            Power_BHP=p["bhp"],
            Mileage_kmpl=p["mpg"],
            Seats=p["seats"],
        )
        for key in (
            f"Brand_{p['brand']}",
            f"Model_{p['model']}",
            f"Body_Type_{p['body']}",
            f"Fuel_Type_{p['fuel']}",
            f"Transmission_{p['trans']}",
            f"Owner_Type_{p['owner']}",
        ):
            if key in r:          # baselines (Audi, 3 Series, Coupe, Diesel...) have no column
                r[key] = 1
        recs.append(r)
    X = pd.DataFrame(recs)[cols]
    pf = PolynomialFeatures(degree=2)
    P = pd.DataFrame(
        pf.fit_transform(X[POLY_COLS]),
        columns=pf.get_feature_names_out(POLY_COLS),
        index=X.index,
    )
    X2 = pd.concat([X.drop(columns=POLY_COLS), P], axis=1)
    out = model.predict(scaler.transform(X2))
    return np.maximum(out, PRICE_FLOOR)


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def fmt_price(lakh):
    return f"₹ {lakh / 100:,.2f} Cr" if lakh >= 100 else f"₹ {lakh:,.2f} L"


def fmt_short(lakh):
    return f"{lakh / 100:.2f} Cr" if lakh >= 100 else f"{lakh:.1f} L"


def inr_full(lakh):
    s = str(int(round(lakh * 100000)))
    if len(s) > 3:
        head, tail, groups = s[:-3], s[-3:], []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    return "₹ " + s


TIER_BOUNDS = [7.5, 32.47, 65.47, 150.0, 450.0]   # quartiles from the notebook's describe()
TIER_NAMES = ["Entry Luxury", "Prestige", "Elite", "Ultra Luxury"]


def tier_info(lakh):
    idx = 3
    for i in range(4):
        if lakh < TIER_BOUNDS[i + 1]:
            idx = i
            break
    lo, hi = TIER_BOUNDS[idx], TIER_BOUNDS[idx + 1]
    frac = (np.log(max(lakh, lo)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return TIER_NAMES[idx], (idx + min(max(frac, 0), 1)) / 4 * 100


def md(s):
    """Render an HTML snippet safely through Markdown (no indentation / blank lines)."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Built-in "studio render" artwork (works offline; used until / unless a photo loads)
# ----------------------------------------------------------------------------
SHAPES = {
    "coupe": dict(
        body="M335 650 C322 600 330 568 372 552 C430 534 520 520 600 492 C670 440 760 392 880 386 "
             "C985 384 1060 424 1115 468 C1170 496 1240 510 1285 536 C1308 552 1312 596 1300 626 "
             "L1290 650 Z",
        glass="M632 488 C690 446 775 410 880 405 C972 403 1038 436 1090 472 L1090 478 L632 490 Z",
        belt="M372 556 C520 524 700 500 900 494 C1060 490 1190 505 1285 538",
        wheels=(490, 1110, 626, 66, 78), head=(1262, 540), tail=(340, 568),
    ),
    "sedan": dict(
        body="M318 652 C306 610 312 570 340 556 C420 538 560 526 650 506 C720 450 800 410 900 406 "
             "C1000 404 1065 436 1120 480 C1180 504 1262 514 1300 540 C1322 556 1326 600 1316 628 "
             "L1306 652 Z",
        glass="M662 500 C725 452 802 418 900 414 C990 412 1050 440 1100 478 L1100 484 L662 502 Z",
        belt="M345 560 C560 530 760 510 960 504 C1120 500 1230 510 1300 542",
        wheels=(480, 1130, 626, 66, 78), head=(1270, 546), tail=(324, 574),
    ),
    "suv": dict(
        body="M345 652 C335 600 338 520 352 470 C362 430 400 392 470 350 L960 344 C1030 344 1070 380 "
             "1110 440 C1150 485 1235 500 1265 530 C1290 550 1295 600 1285 628 L1275 652 Z",
        glass="M470 428 C468 380 495 358 540 354 L955 348 C1012 348 1050 375 1090 430 Z",
        belt="M360 505 C600 492 900 488 1150 498 C1210 505 1250 520 1268 534",
        wheels=(510, 1110, 618, 72, 86), head=(1244, 536), tail=(348, 520),
    ),
}


def car_art(kind, accent):
    s = SHAPES[kind]
    w1, w2, wy, tr, wl = s["wheels"]

    def wheel(cx):
        spokes = "".join(
            f'<line x1="{cx}" y1="{wy}" x2="{cx + 40 * np.cos(a):.1f}" y2="{wy + 40 * np.sin(a):.1f}" '
            f'stroke="{accent}" stroke-opacity=".65" stroke-width="5" stroke-linecap="round"/>'
            for a in np.linspace(0, 2 * np.pi, 6, endpoint=False) + 0.3
        )
        return (
            f'<circle cx="{cx}" cy="{wy}" r="{wl}" fill="#030304"/>'
            f'<circle cx="{cx}" cy="{wy}" r="{tr}" fill="#0c0c0f" stroke="#2c2c33" stroke-width="3"/>'
            f'<circle cx="{cx}" cy="{wy}" r="{int(tr * .68)}" fill="#17171d" stroke="{accent}" '
            f'stroke-opacity=".75" stroke-width="2"/>{spokes}'
            f'<circle cx="{cx}" cy="{wy}" r="9" fill="{accent}"/>'
        )

    hx, hy = s["head"]
    tx, ty = s["tail"]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">
<defs>
<radialGradient id="g" cx="62%" cy="40%" r="75%"><stop offset="0" stop-color="{accent}" stop-opacity=".42"/><stop offset=".45" stop-color="#1a191e"/><stop offset="1" stop-color="#040405"/></radialGradient>
<linearGradient id="b" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#555b69"/><stop offset=".3" stop-color="#20232c"/><stop offset="1" stop-color="#07080b"/></linearGradient>
<linearGradient id="f" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#19181d"/><stop offset="1" stop-color="#040405"/></linearGradient>
<linearGradient id="r" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{accent}" stop-opacity="0"/><stop offset=".5" stop-color="{accent}"/><stop offset="1" stop-color="{accent}" stop-opacity="0"/></linearGradient>
<linearGradient id="gl" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1b2030"/><stop offset="1" stop-color="#05060a"/></linearGradient>
<filter id="bl" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="22"/></filter>
<filter id="bs" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6"/></filter>
<g id="car">
<path d="{s['body']}" fill="url(#b)" stroke="{accent}" stroke-opacity=".35" stroke-width="2"/>
<path d="{s['glass']}" fill="url(#gl)" stroke="{accent}" stroke-opacity=".4" stroke-width="2"/>
<path d="{s['belt']}" fill="none" stroke="url(#r)" stroke-width="3.5"/>
<path d="M{w1 + 80} 640 L{w2 - 80} 640" stroke="{accent}" stroke-opacity=".28" stroke-width="3"/>
{wheel(w1)}{wheel(w2)}
<ellipse cx="{hx + 30}" cy="{hy + 14}" rx="70" ry="26" fill="#fff" opacity=".4" filter="url(#bs)"/>
<path d="M{hx} {hy} C{hx + 28} {hy + 6} {hx + 44} {hy + 16} {hx + 50} {hy + 26} C{hx + 28} {hy + 26} {hx + 10} {hy + 18} {hx} {hy} Z" fill="#fffbe8"/>
<ellipse cx="{tx}" cy="{ty}" rx="22" ry="16" fill="#ff2a2a" opacity=".5" filter="url(#bs)"/>
<path d="M{tx} {ty + 10} C{tx - 2} {ty - 4} {tx + 6} {ty - 12} {tx + 18} {ty - 14} L{tx + 18} {ty - 4} Z" fill="#ff3b3b"/>
</g>
</defs>
<rect width="1600" height="900" fill="url(#g)"/>
<path d="M0 330 L1600 200" stroke="{accent}" stroke-opacity=".10" stroke-width="130" filter="url(#bl)"/>
<rect y="690" width="1600" height="210" fill="url(#f)"/>
<rect y="689" width="1600" height="2" fill="url(#r)" opacity=".75"/>
<ellipse cx="800" cy="706" rx="640" ry="30" fill="#000" opacity=".75" filter="url(#bl)"/>
<use href="#car" transform="matrix(1 0 0 -1 0 1380)" opacity=".12"/>
<use href="#car"/>
</svg>"""
    return "data:image/svg+xml;charset=utf-8," + quote(svg, safe="")


# Hero slides. Captions are deliberately generic so they always match the picture.
# Each slide lists candidate premium photos (tried in order); the studio render is the fallback.
UNS = "https://images.unsplash.com/photo-{}?auto=format&fit=crop&w=1800&q=80"
SLIDES = [
    dict(label="Supercar Soul", tag="Raw power, sculpted by obsession.", kind="coupe", accent="#e0b84a",
         ids=["1583121274602-3e2820c69888", "1614162692292-7ac56d7f7f1e"]),
    dict(label="Grand Touring", tag="Continents crossed in effortless silence.", kind="coupe", accent="#c9a15c",
         ids=["1618843479313-40f8afb4b4d8", "1544636331-e26879cd4d9b"]),
    dict(label="Executive Class", tag="Where the boardroom meets the open road.", kind="sedan", accent="#d8c27a",
         ids=["1555215695-3004980ad54e", "1617814076367-b759c7d7e738"]),
    dict(label="Luxury SUV", tag="Command presence, cloud-soft comfort.", kind="suv", accent="#e6c35c",
         ids=["1606016159991-dfe4f2746ad5", "1519641471654-76ce0107ad1b"]),
    dict(label="Performance Coupe", tag="Precision engineering you can feel.", kind="coupe", accent="#d4af37",
         ids=["1580273916550-e323be2ae537", "1552519507-da3b142c6e3d"]),
    dict(label="Timeless Elegance", tag="Craftsmanship that outlives trends.", kind="sedan", accent="#f0d78c",
         ids=["1563720223185-11003d516935", "1494905998402-395d579af36f"]),
]


@st.cache_data(show_spinner=False)
def local_photos():
    folder = BASE_DIR / "car_images"
    mimes = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    out = []
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in mimes:
                out.append(f"data:{mimes[f.suffix.lower()]};base64," + base64.b64encode(f.read_bytes()).decode())
    return out


HERO_TEMPLATE = Template(r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="color-scheme" content="dark">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#07070a;font-family:'Inter',system-ui,-apple-system,Segoe UI,sans-serif;color:#f4ecd8;overflow:hidden}
.hero{position:relative;height:100%;border-radius:26px;overflow:hidden;border:1px solid rgba(212,175,55,.3);box-shadow:inset 0 0 0 1px rgba(255,255,255,.03)}
.slide{position:absolute;inset:0;opacity:0;transition:opacity 1.2s ease}
.slide.active{opacity:1;z-index:1}
.bg{position:absolute;inset:0;background-size:cover;background-position:center;transform:scale(1.04);transition:opacity 1s ease}
.slide.active .bg{animation:kb 10s ease-out forwards}
@keyframes kb{from{transform:scale(1.04)}to{transform:scale(1.13)}}
.bg.photo{opacity:0}.bg.photo.on{opacity:1}
.shade{position:absolute;inset:0;background:linear-gradient(90deg,rgba(7,7,10,.93) 0%,rgba(7,7,10,.62) 36%,rgba(7,7,10,.06) 72%),linear-gradient(0deg,rgba(7,7,10,.88) 0%,rgba(7,7,10,0) 38%),linear-gradient(180deg,rgba(7,7,10,.55) 0%,rgba(7,7,10,0) 22%)}
.top{position:absolute;z-index:5;top:26px;left:44px;right:44px;display:flex;justify-content:space-between;align-items:center}
.logo{font-family:'Playfair Display',Georgia,serif;font-size:22px;letter-spacing:.42em;color:#f4e2a1}
.logo b{color:#d4af37;font-weight:600}
.pill{border:1px solid rgba(212,175,55,.45);color:#ecdca8;padding:8px 16px;border-radius:999px;font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;background:rgba(0,0,0,.4);backdrop-filter:blur(6px)}
.headline{position:absolute;z-index:5;left:56px;top:48%;transform:translateY(-55%);max-width:660px}
.eyebrow{font-size:11.5px;letter-spacing:.38em;color:#d4af37;text-transform:uppercase;margin-bottom:18px}
.eyebrow:before{content:"";display:inline-block;width:38px;height:1px;background:#d4af37;vertical-align:middle;margin-right:14px}
h1{font-family:'Playfair Display',Georgia,serif;font-weight:600;font-size:clamp(34px,5.4vw,70px);line-height:1.04;background:linear-gradient(180deg,#fff8de 0%,#e6c35c 55%,#b8902a 100%);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{margin-top:20px;color:#d2c9b2;font-size:clamp(13px,1.35vw,17px);line-height:1.65;max-width:500px;font-weight:300}
.cta{margin-top:26px;display:inline-flex;align-items:center;gap:12px;font-size:11.5px;letter-spacing:.24em;text-transform:uppercase;color:#07070a;background:linear-gradient(135deg,#f4e2a1,#d4af37);padding:13px 24px;border-radius:999px;font-weight:500;box-shadow:0 10px 30px rgba(212,175,55,.35)}
.caption{position:absolute;z-index:5;left:56px;bottom:40px;animation:fade .9s ease}
@keyframes fade{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.caption .n{font-family:'Playfair Display',Georgia,serif;font-size:30px;color:#d4af37}
.caption .t{font-size:13px;letter-spacing:.26em;text-transform:uppercase;color:#f4ecd8;margin-top:4px}
.caption .d{font-size:13px;color:#a79f8a;margin-top:4px;font-weight:300}
.thumbs{position:absolute;z-index:5;right:44px;bottom:40px;display:flex;gap:10px;align-items:flex-end}
.th{width:84px;height:56px;border-radius:10px;border:1px solid rgba(255,255,255,.16);background-size:cover;background-position:center;opacity:.5;cursor:pointer;transition:all .35s ease}
.th:hover{opacity:.85}
.th.on{opacity:1;border-color:#d4af37;transform:translateY(-6px);box-shadow:0 10px 26px rgba(212,175,55,.38)}
.arrows{position:absolute;z-index:5;right:44px;top:50%;transform:translateY(-50%);display:flex;flex-direction:column;gap:12px}
.ar{width:46px;height:46px;border-radius:50%;border:1px solid rgba(212,175,55,.5);background:rgba(0,0,0,.45);color:#f4e2a1;font-size:18px;cursor:pointer;backdrop-filter:blur(6px);transition:all .25s}
.ar:hover{background:#d4af37;color:#07070a}
.bar{position:absolute;z-index:6;left:0;bottom:0;height:3px;width:0;background:linear-gradient(90deg,#d4af37,#fff1c1)}
@keyframes fill{from{width:0}to{width:100%}}
@media(max-width:820px){.thumbs{display:none}.arrows{display:none}.pill{display:none}.headline{left:26px;right:26px}.caption{left:26px}.top{left:26px}}
</style></head><body>
<div class="hero" id="hero">
  <div id="stage"></div>
  <div class="top"><div class="logo">AU<b>R</b>UM</div><div class="pill">AI Valuation · Lakh &amp; Crore</div></div>
  <div class="headline">
    <div class="eyebrow">Luxury Car Valuation Studio</div>
    <h1>Know what your prestige is truly worth.</h1>
    <p class="sub">A machine-learning valuation engine trained on premium marques &mdash; from Audi to Rolls-Royce &mdash; delivering an instant, data-driven price for your automobile.</p>
    <div class="cta">Begin your valuation &darr;</div>
  </div>
  <div class="arrows"><button class="ar" id="prev">&#8593;</button><button class="ar" id="next">&#8595;</button></div>
  <div class="caption" id="cap"><div class="n" id="capN"></div><div class="t" id="capT"></div><div class="d" id="capD"></div></div>
  <div class="thumbs" id="thumbs"></div>
  <div class="bar" id="bar"></div>
</div>
<script>
const S=$SLIDES;
const stage=document.getElementById('stage'),thumbs=document.getElementById('thumbs'),bar=document.getElementById('bar');
const cap=document.getElementById('cap'),capN=document.getElementById('capN'),capT=document.getElementById('capT'),capD=document.getElementById('capD');
const els=[],ths=[];let cur=0,timer=null,paused=false;const DUR=6500;
function load(i,k){
  const s=S[i];if(k>=s.urls.length)return;
  const im=new Image();im.referrerPolicy='no-referrer';
  im.onload=()=>{
    if(im.naturalWidth<500){load(i,k+1);return;}
    const ph=els[i].querySelector('.photo');ph.style.backgroundImage='url("'+im.src+'")';ph.classList.add('on');
    ths[i].style.backgroundImage='url("'+im.src+'")';
  };
  im.onerror=()=>load(i,k+1);
  im.src=s.urls[k];
}
S.forEach((s,i)=>{
  const d=document.createElement('div');d.className='slide'+(i?'':' active');
  d.innerHTML='<div class="bg" style="background-image:url(\''+s.fb+'\')"></div><div class="bg photo"></div><div class="shade"></div>';
  stage.appendChild(d);els.push(d);
  const t=document.createElement('div');t.className='th'+(i?'':' on');t.style.backgroundImage='url(\''+s.fb+'\')';
  t.onclick=()=>{go(i);restart()};thumbs.appendChild(t);ths.push(t);
  load(i,0);
});
function caption(){
  const s=S[cur];capN.textContent=String(cur+1).padStart(2,'0')+' / '+String(S.length).padStart(2,'0');
  capT.textContent=s.label;capD.textContent=s.tag;
  cap.style.animation='none';void cap.offsetWidth;cap.style.animation='fade .9s ease';
}
function go(n){
  els[cur].classList.remove('active');ths[cur].classList.remove('on');
  cur=(n+S.length)%S.length;
  els[cur].classList.add('active');ths[cur].classList.add('on');caption();
}
function restart(){
  clearInterval(timer);bar.style.animation='none';void bar.offsetWidth;
  bar.style.animation='fill '+DUR+'ms linear forwards';
  timer=setInterval(()=>{if(!paused){go(cur+1);bar.style.animation='none';void bar.offsetWidth;bar.style.animation='fill '+DUR+'ms linear forwards';}},DUR);
}
document.getElementById('next').onclick=()=>{go(cur+1);restart()};
document.getElementById('prev').onclick=()=>{go(cur-1);restart()};
const hero=document.getElementById('hero');
hero.addEventListener('mouseenter',()=>{paused=true;bar.style.animationPlayState='paused'});
hero.addEventListener('mouseleave',()=>{paused=false;restart()});
let x0=null;
hero.addEventListener('touchstart',e=>{x0=e.touches[0].clientX},{passive:true});
hero.addEventListener('touchend',e=>{if(x0===null)return;const dx=e.changedTouches[0].clientX-x0;if(Math.abs(dx)>40){go(cur+(dx<0?1:-1));restart()}x0=null});
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowDown'){go(cur+1);restart()}if(e.key==='ArrowLeft'||e.key==='ArrowUp'){go(cur-1);restart()}});
caption();restart();
</script></body></html>""")


@st.cache_data(show_spinner=False)
def hero_html():
    locals_ = local_photos()
    payload = []
    for i, s in enumerate(SLIDES):
        urls = [UNS.format(x) for x in s["ids"]]
        if i < len(locals_):
            urls.insert(0, locals_[i])
        payload.append(dict(label=s["label"], tag=s["tag"], fb=car_art(s["kind"], s["accent"]), urls=urls))
    return HERO_TEMPLATE.substitute(SLIDES=json.dumps(payload))


# ----------------------------------------------------------------------------
# Forecast chart (rendered in its own HTML component -> never sanitised)
# ----------------------------------------------------------------------------
def forecast_html(vals, years):
    W, H, L, R, T, B = 900, 330, 84, 40, 50, 58
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.3, hi * 0.08)
    vmin, vmax = max(0.0, lo - pad), hi + pad
    n = len(vals)
    X = lambda i: L + (W - L - R) * i / (n - 1)
    Y = lambda v: T + (H - T - B) * (1 - (v - vmin) / (vmax - vmin))
    grid, labels = "", ""
    for k in range(4):
        v = vmin + (vmax - vmin) * k / 3
        y = Y(v)
        grid += f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d4af37" stroke-opacity=".12"/>'
        labels += f'<text x="{L - 12}" y="{y + 4:.1f}" text-anchor="end" fill="#8d8675" font-size="12">{fmt_short(v)}</text>'
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
    area = f"M{X(0):.1f},{Y(vmin):.1f} " + " ".join(f"L{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals)) + f" L{X(n - 1):.1f},{Y(vmin):.1f} Z"
    dots = ""
    for i, v in enumerate(vals):
        dots += (
            f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="9" fill="#d4af37" opacity=".18"/>'
            f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5" fill="#f4e2a1" stroke="#d4af37" stroke-width="2"/>'
            f'<text x="{X(i):.1f}" y="{Y(v) - 18:.1f}" text-anchor="middle" fill="#f4ecd8" font-size="13" font-weight="600">{fmt_short(v)}</text>'
            f'<text x="{X(i):.1f}" y="{H - 30}" text-anchor="middle" fill="#cfc6b0" font-size="12.5">{"Today" if i == 0 else f"+{i} yr"}</text>'
            f'<text x="{X(i):.1f}" y="{H - 13}" text-anchor="middle" fill="#7d7766" font-size="11">{years[i]}</text>'
        )
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="color-scheme" content="dark">
<style>html,body{{margin:0;background:#111116;font-family:Inter,system-ui,Segoe UI,sans-serif}}svg{{display:block;width:100%;height:auto}}</style></head><body>
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,system-ui,Segoe UI,sans-serif">
<defs>
<linearGradient id="ar" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d4af37" stop-opacity=".35"/><stop offset="1" stop-color="#d4af37" stop-opacity="0"/></linearGradient>
<linearGradient id="ln" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#f4e2a1"/><stop offset="1" stop-color="#c9a02f"/></linearGradient>
</defs>
{grid}{labels}
<path d="{area}" fill="url(#ar)"/>
<polyline points="{pts}" fill="none" stroke="url(#ln)" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>
{dots}
</svg></body></html>"""


# ----------------------------------------------------------------------------
# Certificate (downloadable HTML)
# ----------------------------------------------------------------------------
CERT_TEMPLATE = Template("""<!doctype html><html><head><meta charset="utf-8"><title>AURUM Valuation Certificate</title>
<style>
body{margin:0;background:#0b0b0e;font-family:Georgia,'Times New Roman',serif;color:#2a2418;padding:40px}
.c{max-width:820px;margin:auto;background:linear-gradient(160deg,#fffaf0,#f3e7c6);border:2px solid #b8902a;outline:8px solid #fffaf0;outline-offset:-14px;padding:60px 56px;text-align:center;box-shadow:0 30px 80px rgba(0,0,0,.5)}
.logo{letter-spacing:.5em;color:#9a7416;font-size:22px}.t{margin:26px 0 6px;font-size:13px;letter-spacing:.35em;text-transform:uppercase;color:#7a6a45}
h1{font-size:44px;margin:6px 0 4px;font-weight:600}.p{font-size:58px;color:#9a7416;margin:26px 0 4px}.f{color:#6b5b36}
table{width:100%;margin:34px 0;border-collapse:collapse;text-align:left;font-family:Arial,sans-serif;font-size:14px}
td{padding:9px 6px;border-bottom:1px solid rgba(154,116,22,.28)}td:first-child{color:#7a6a45;text-transform:uppercase;letter-spacing:.12em;font-size:11.5px;width:42%}
.s{font-size:11.5px;color:#7a6a45;font-family:Arial,sans-serif;line-height:1.6}
</style></head><body><div class="c">
<div class="logo">AURUM</div><div class="t">Valuation Certificate</div>
<h1>$car</h1><div class="f">$tier</div>
<div class="p">$price</div><div class="f">$full</div>
<table>$rows</table>
<div class="s">Certificate ID $cid &middot; Issued $date<br>Indicative machine-learning estimate. Not a formal appraisal or an offer to purchase.</div>
</div></body></html>""")


def certificate(p, price):
    tier, _ = tier_info(price)
    rows_data = [
        ("Model year", p["year"]), ("Kilometres driven", f"{p['kms']:,} km"), ("Ownership", f"{p['owner']} owner"),
        ("Body", p["body"]), ("Fuel", p["fuel"]), ("Transmission", p["trans"]),
        ("Engine", f"{p['cc']:,} cc"), ("Power", f"{p['bhp']:.1f} bhp"),
        ("Claimed mileage", f"{p['mpg']:.1f} kmpl"), ("Seats", p["seats"]),
    ]
    rows = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows_data)
    cid = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10].upper()
    return CERT_TEMPLATE.substitute(
        car=f"{p['year']} {p['brand']} {p['model']}", tier=tier, price=fmt_price(price), full=f"≈ {inr_full(price)}",
        rows=rows, cid=cid, date=datetime.now().strftime("%d %B %Y"),
    ).encode("utf-8")


# ----------------------------------------------------------------------------
# Session state + callbacks
# ----------------------------------------------------------------------------
def apply_preset():
    body, fuel, cc, bhp, mpg, seats = CATALOG[st.session_state.brand][st.session_state.model]
    st.session_state.update(body=body, fuel=fuel, cc=cc, bhp=bhp, mpg=mpg, seats=seats)


def on_brand():
    st.session_state.model = next(iter(CATALOG[st.session_state.brand]))
    apply_preset()


def on_b_brand():
    st.session_state.b_model = next(iter(CATALOG[st.session_state.b_brand]))


if "brand" not in st.session_state:
    st.session_state.update(
        brand="Audi", model="Q7", year=2024, kms=17000, owner="First", trans="Automatic", annual=10000,
        b_brand="BMW", b_model="X5", b_year=2023, b_kms=22800, b_owner="First",
    )
    apply_preset()

# ----------------------------------------------------------------------------
# Global styling
# ----------------------------------------------------------------------------
md("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap');
:root{--gold:#d4af37;--gold2:#f4e2a1;--text:#efe8d6;--muted:#9a937f;--line:rgba(212,175,55,.22)}
.stApp{background:radial-gradient(1100px 560px at 85% -8%,rgba(212,175,55,.13),transparent 60%),radial-gradient(900px 520px at -10% 32%,rgba(140,100,20,.12),transparent 60%),#07070a;font-family:'Inter',system-ui,sans-serif}
[data-testid="stHeader"]{background:transparent;height:0}
[data-testid="stToolbar"],[data-testid="stDecoration"],footer,#MainMenu{display:none!important}
.block-container{max-width:1280px;padding:1.2rem 1.4rem 4rem}
.stApp iframe{border:0;border-radius:26px}
h1,h2,h3,.serif{font-family:'Playfair Display',Georgia,serif!important}
.stMarkdown,.stMarkdown p,.stMarkdown li{color:var(--text)}
[data-testid="stWidgetLabel"] p,label{color:#cdc4ad!important;font-size:.72rem!important;letter-spacing:.16em;text-transform:uppercase;font-weight:500}
div[data-baseweb="select"]>div,div[data-baseweb="input"],div[data-baseweb="base-input"]{background:#17171e!important;border:1px solid rgba(212,175,55,.26)!important;border-radius:12px!important;color:var(--text)!important}
div[data-baseweb="select"] *,input{color:var(--text)!important;-webkit-text-fill-color:var(--text)!important}
div[data-baseweb="select"]>div:hover,div[data-baseweb="input"]:focus-within{border-color:var(--gold)!important;box-shadow:0 0 0 3px rgba(212,175,55,.14)}
div[data-baseweb="popover"] ul,div[data-baseweb="popover"]>div,div[data-baseweb="menu"]{background:#17171e!important}
li[role="option"]{color:var(--text)!important}li[role="option"]:hover,li[aria-selected="true"]{background:rgba(212,175,55,.16)!important}
[data-testid="stNumberInputStepUp"],[data-testid="stNumberInputStepDown"]{background:#20202a!important;color:var(--gold)!important}
button[data-baseweb="tab"]{background:transparent!important;padding:.7rem 1.3rem}
button[data-baseweb="tab"] p{color:#a79f8a!important;font-size:.82rem;letter-spacing:.2em;text-transform:uppercase;font-weight:500}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--gold2)!important}
[data-baseweb="tab-highlight"]{background:var(--gold)!important;height:2px!important}
[data-baseweb="tab-border"]{background:rgba(212,175,55,.15)!important}
[class*="st-key-card"]{background:#111116;border:1px solid var(--line);border-radius:22px;padding:1.5rem 1.6rem 1.2rem;box-shadow:0 24px 60px rgba(0,0,0,.45)}
.card-title{font-family:'Playfair Display',Georgia,serif;font-size:1.05rem;color:var(--gold2);margin:.3rem 0 .7rem;display:flex;align-items:center;gap:.7rem}
.card-title:after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent)}
.stDownloadButton button{width:100%;background:linear-gradient(135deg,#f4e2a1,#d4af37)!important;color:#0a0a0c!important;border:0!important;border-radius:999px!important;padding:.75rem 1rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;font-size:.78rem;box-shadow:0 12px 30px rgba(212,175,55,.28)}
.stDownloadButton button:hover{filter:brightness(1.08);transform:translateY(-1px)}
.stDownloadButton button p{color:#0a0a0c!important;font-weight:600}
.eyebrow{font-size:.72rem;letter-spacing:.38em;color:var(--gold);text-transform:uppercase}
.sec-h{font-family:'Playfair Display',Georgia,serif;font-size:2.1rem;font-weight:600;margin:.35rem 0 1.1rem;background:linear-gradient(180deg,#fff8de,#d9b445);-webkit-background-clip:text;background-clip:text;color:transparent}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:1.3rem 0 2.6rem}
.stat{background:linear-gradient(160deg,#15151b,#0e0e12);border:1px solid var(--line);border-radius:18px;padding:1.2rem 1.3rem;position:relative;overflow:hidden}
.stat:before{content:"";position:absolute;right:-30px;top:-30px;width:110px;height:110px;border-radius:50%;background:radial-gradient(circle,rgba(212,175,55,.22),transparent 70%)}
.stat .v{font-family:'Playfair Display',Georgia,serif;font-size:2.3rem;color:var(--gold2);line-height:1}
.stat .l{margin-top:.5rem;font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted)}
.result{position:relative;background:radial-gradient(520px 260px at 80% 0%,rgba(212,175,55,.2),transparent 70%),linear-gradient(160deg,#17171e,#0d0d11);border:1px solid rgba(212,175,55,.38);border-radius:24px;padding:1.8rem 1.9rem;box-shadow:0 30px 70px rgba(0,0,0,.5),0 0 60px rgba(212,175,55,.08);animation:rise .6s ease}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.r-eye{font-size:.7rem;letter-spacing:.34em;color:var(--gold);text-transform:uppercase}
.r-car{font-family:'Playfair Display',Georgia,serif;font-size:1.55rem;color:var(--text);margin-top:.5rem}
.r-price{font-family:'Playfair Display',Georgia,serif;font-size:3.7rem;font-weight:700;line-height:1.1;margin-top:.7rem;background:linear-gradient(180deg,#fff6d0,#e0b83f 60%,#b78e25);-webkit-background-clip:text;background-clip:text;color:transparent;text-shadow:0 0 40px rgba(212,175,55,.25)}
.r-full{color:#cdc4ad;font-size:1rem;margin-top:.1rem}
.r-range{margin-top:.9rem;display:inline-block;font-size:.78rem;color:#bdb49c;border:1px solid var(--line);padding:.35rem .8rem;border-radius:999px;background:rgba(0,0,0,.25)}
.tier{margin-top:1.4rem;display:flex;justify-content:space-between;align-items:center;font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted)}
.badge{color:#0a0a0c;background:linear-gradient(135deg,#f4e2a1,#d4af37);padding:.28rem .8rem;border-radius:999px;font-weight:600;letter-spacing:.16em}
.meter{position:relative;height:8px;border-radius:99px;margin-top:.8rem;background:linear-gradient(90deg,#3a3320 0%,#6d5a22 35%,#b8902a 70%,#f4e2a1 100%)}
.dot{position:absolute;top:50%;width:18px;height:18px;border-radius:50%;background:#fff6d0;border:3px solid #d4af37;transform:translate(-50%,-50%);box-shadow:0 0 18px rgba(212,175,55,.8)}
.mlab{display:flex;justify-content:space-between;margin-top:.55rem;font-size:.64rem;letter-spacing:.14em;color:#7d7766;text-transform:uppercase}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1.3rem}
.chip{font-size:.76rem;color:#d8cfb8;border:1px solid rgba(212,175,55,.2);background:rgba(255,255,255,.03);padding:.38rem .8rem;border-radius:999px}
.warn{margin-top:1rem;font-size:.78rem;color:#c9a95a}
.cmp{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.cmp .result{padding:1.5rem 1.6rem}.cmp .r-price{font-size:2.9rem}
.win{outline:1px solid rgba(244,226,161,.7);box-shadow:0 0 50px rgba(212,175,55,.2)}
.delta{margin:1.2rem 0;text-align:center;font-family:'Playfair Display',Georgia,serif;font-size:1.35rem;color:var(--gold2)}
.spec{width:100%;border-collapse:collapse;margin-top:.4rem}
.spec td{padding:.7rem .4rem;border-bottom:1px solid rgba(212,175,55,.12);font-size:.92rem;color:var(--text)}
.spec td:first-child{color:var(--muted);font-size:.7rem;letter-spacing:.18em;text-transform:uppercase;width:30%}
.spec td:not(:first-child){text-align:center}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:.6rem 0 1.6rem}
.step{background:#15151b;border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.1rem}
.step b{font-family:'Playfair Display',Georgia,serif;color:var(--gold);font-size:1.6rem;display:block}
.step span{font-size:.86rem;color:#cfc6b0;line-height:1.5}
.tags{display:flex;flex-wrap:wrap;gap:.55rem;margin:.4rem 0 1rem}
.tag{font-size:.8rem;letter-spacing:.06em;color:#e8dcb5;border:1px solid rgba(212,175,55,.3);padding:.42rem .95rem;border-radius:999px;background:rgba(212,175,55,.06)}
.note{color:#9a937f;font-size:.82rem;line-height:1.7}
.foot{margin-top:3rem;text-align:center;color:#6f6a5b;font-size:.72rem;letter-spacing:.24em;text-transform:uppercase}
@media(max-width:900px){.stats,.steps{grid-template-columns:1fr 1fr}.cmp{grid-template-columns:1fr}.r-price{font-size:2.8rem}}
</style>
""")

# ----------------------------------------------------------------------------
# HERO + STATS
# ----------------------------------------------------------------------------
components.html(hero_html(), height=640, scrolling=False)

md(f"""
<div class="stats">
<div class="stat"><div class="v">0.83</div><div class="l">R² score · hold-out</div></div>
<div class="stat"><div class="v">500</div><div class="l">Luxury listings learned</div></div>
<div class="stat"><div class="v">{len(CATALOG)}</div><div class="l">Premium marques</div></div>
<div class="stat"><div class="v">{N_MODELS}</div><div class="l">Models valued</div></div>
</div>
<div class="eyebrow">The Studio</div>
<div class="sec-h">Configure your automobile</div>
""")

tab1, tab2, tab3 = st.tabs(["Valuation Studio", "Head-to-Head", "The Engine"])

# ----------------------------------------------------------------------------
# TAB 1 — VALUATION STUDIO
# ----------------------------------------------------------------------------
with tab1:
    left, right = st.columns([1.12, 1], gap="large")

    with left:
        with st.container(key="card_cfg"):
            md('<div class="card-title">I · Marque &amp; model</div>')
            c1, c2 = st.columns(2)
            c1.selectbox("Marque", list(CATALOG), key="brand", on_change=on_brand)
            c2.selectbox("Model", list(CATALOG[st.session_state.brand]), key="model", on_change=apply_preset)

            md('<div class="card-title">II · Condition</div>')
            c1, c2, c3 = st.columns(3)
            c1.selectbox("Model year", YEARS, key="year")
            c2.number_input("Kms driven", min_value=500, max_value=250000, step=1000, key="kms")
            c3.selectbox("Ownership", OWNERS, key="owner")

            md('<div class="card-title">III · Specification</div>')
            c1, c2, c3 = st.columns(3)
            c1.selectbox("Body", BODIES, key="body")
            c2.selectbox("Fuel", FUELS, key="fuel")
            c3.selectbox("Transmission", TRANS, key="trans")
            c1, c2 = st.columns(2)
            c1.number_input("Engine (cc)", min_value=1000, max_value=7000, step=1, key="cc")
            c2.number_input("Power (bhp)", min_value=100.0, max_value=700.0, step=0.5, key="bhp")
            c1, c2 = st.columns(2)
            c1.number_input("Mileage (kmpl)", min_value=4.0, max_value=25.0, step=0.1, key="mpg")
            c2.selectbox("Seats", SEATS, key="seats")

    S_ = st.session_state
    params = dict(brand=S_.brand, model=S_.model, body=S_.body, fuel=S_.fuel, trans=S_.trans, owner=S_.owner,
                  year=int(S_.year), kms=int(S_.kms), cc=int(S_.cc), bhp=float(S_.bhp), mpg=float(S_.mpg),
                  seats=int(S_.seats))
    price = float(predict_many([params])[0])
    tier_name, pos = tier_info(price)

    with right:
        floored = price <= PRICE_FLOOR + 1e-9
        md(f"""
        <div class="result">
        <div class="r-eye">◆ Estimated market value</div>
        <div class="r-car">{params['year']} {params['brand']} {params['model']}</div>
        <div class="r-price">{fmt_price(price)}</div>
        <div class="r-full">≈ {inr_full(price)}</div>
        <div class="r-range">Indicative range (±10%) &nbsp;·&nbsp; {fmt_price(price * .9)} — {fmt_price(price * 1.1)}</div>
        <div class="tier"><span>Market tier</span><span class="badge">{tier_name}</span></div>
        <div class="meter"><div class="dot" style="left:{pos:.1f}%"></div></div>
        <div class="mlab"><span>Entry</span><span>Prestige</span><span>Elite</span><span>Ultra</span></div>
        <div class="chips">
        <span class="chip">{REF_YEAR - params['year']} yrs old</span>
        <span class="chip">{params['kms']:,} km</span>
        <span class="chip">{params['owner']} owner</span>
        <span class="chip">{params['fuel']} · {params['trans']}</span>
        <span class="chip">{params['bhp']:.0f} bhp</span>
        <span class="chip">{params['cc']:,} cc</span>
        </div>
        {'<div class="warn">This combination sits outside what the model has seen; the value is shown at the minimum floor.</div>' if floored else ''}
        </div>
        """)
        st.write("")
        st.download_button(
            "Download valuation certificate",
            data=certificate(params, price),
            file_name=f"AURUM_{params['brand']}_{params['model']}_valuation.html".replace(" ", "_"),
            mime="text/html",
            key="dl_cert",
        )

    # ---- 5-year outlook
    st.write("")
    with st.container(key="card_forecast"):
        md('<div class="card-title">IV · Five-year value outlook</div>')
        fc1, fc2 = st.columns([1, 3])
        fc1.number_input("Driving per year (km)", min_value=2000, max_value=40000, step=1000, key="annual")
        rows = []
        for t in range(6):
            q = dict(params)
            q["year"] = params["year"] - t               # same car, t model-years older
            q["age"] = (REF_YEAR - params["year"]) + t
            q["kms"] = params["kms"] + t * int(S_.annual)
            rows.append(q)
        vals = [float(v) for v in predict_many(rows)]
        years = [REF_YEAR + t for t in range(6)]
        retention = vals[-1] / vals[0] * 100 if vals[0] else 0
        fc1.markdown(
            f"<div class='note' style='margin-top:1rem'>Projected value after 5 years<br>"
            f"<span class='serif' style='font-size:2rem;color:#f4e2a1'>{retention:.0f}%</span> of today's value</div>",
            unsafe_allow_html=True,
        )
        with fc2:
            components.html(forecast_html(vals, years), height=345, scrolling=False)
        md('<div class="note">Outlook ages your car through the model (one year older and the chosen kilometres added each year). '
           'It reflects patterns in the training data, not a market guarantee.</div>')

# ----------------------------------------------------------------------------
# TAB 2 — HEAD-TO-HEAD
# ----------------------------------------------------------------------------
with tab2:
    md('<div class="note" style="margin-bottom:1rem">Car A is your configuration from the Valuation Studio. '
       'Choose a rival below; its specification is auto-filled with typical values for that model.</div>')
    with st.container(key="card_cmp"):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.selectbox("Rival marque", list(CATALOG), key="b_brand", on_change=on_b_brand)
        c2.selectbox("Rival model", list(CATALOG[st.session_state.b_brand]), key="b_model")
        c3.selectbox("Model year", YEARS, key="b_year")
        c4.number_input("Kms driven", min_value=500, max_value=250000, step=1000, key="b_kms")
        c5.selectbox("Ownership", OWNERS, key="b_owner")

    body_b, fuel_b, cc_b, bhp_b, mpg_b, seats_b = CATALOG[S_.b_brand][S_.b_model]
    pb = dict(brand=S_.b_brand, model=S_.b_model, body=body_b, fuel=fuel_b, trans="Automatic", owner=S_.b_owner,
              year=int(S_.b_year), kms=int(S_.b_kms), cc=cc_b, bhp=bhp_b, mpg=mpg_b, seats=seats_b)
    price_b = float(predict_many([pb])[0])
    diff = price - price_b
    win_a, win_b = ("win" if price >= price_b else ""), ("win" if price_b > price else "")

    def mini(p, v, cls, tag):
        t_name, ps = tier_info(v)
        return f"""
        <div class="result {cls}">
        <div class="r-eye">◆ {tag}</div>
        <div class="r-car">{p['year']} {p['brand']} {p['model']}</div>
        <div class="r-price">{fmt_price(v)}</div>
        <div class="r-full">≈ {inr_full(v)}</div>
        <div class="tier"><span>Market tier</span><span class="badge">{t_name}</span></div>
        <div class="meter"><div class="dot" style="left:{ps:.1f}%"></div></div>
        </div>"""

    st.write("")
    md(f'<div class="cmp">{mini(params, price, win_a, "Car A · Your car")}{mini(pb, price_b, win_b, "Car B · Rival")}</div>')
    verdict = ("Car A is valued higher by " if diff >= 0 else "Car B is valued higher by ") + fmt_price(abs(diff))
    md(f'<div class="delta">{verdict}</div>')
    md(f"""
    <table class="spec">
    <tr><td>Power</td><td>{params['bhp']:.0f} bhp</td><td>{pb['bhp']:.0f} bhp</td></tr>
    <tr><td>Engine</td><td>{params['cc']:,} cc</td><td>{pb['cc']:,} cc</td></tr>
    <tr><td>Mileage</td><td>{params['mpg']:.1f} kmpl</td><td>{pb['mpg']:.1f} kmpl</td></tr>
    <tr><td>Fuel</td><td>{params['fuel']}</td><td>{pb['fuel']}</td></tr>
    <tr><td>Seats</td><td>{params['seats']}</td><td>{pb['seats']}</td></tr>
    <tr><td>Age</td><td>{REF_YEAR - params['year']} yrs</td><td>{REF_YEAR - pb['year']} yrs</td></tr>
    <tr><td>Driven</td><td>{params['kms']:,} km</td><td>{pb['kms']:,} km</td></tr>
    </table>
    """)

# ----------------------------------------------------------------------------
# TAB 3 — THE ENGINE
# ----------------------------------------------------------------------------
with tab3:
    tags = "".join(f'<span class="tag">{b}</span>' for b in CATALOG)
    md(f"""
    <div class="eyebrow" style="margin-top:.6rem">How it works</div>
    <div class="sec-h" style="font-size:1.6rem">From raw listing to instant valuation</div>
    <div class="steps">
    <div class="step"><b>01</b><span>One-hot encoding of marque, model, body, fuel, transmission and ownership.</span></div>
    <div class="step"><b>02</b><span>Degree-2 polynomial features on kilometres, engine size, power and age to capture non-linear value curves.</span></div>
    <div class="step"><b>03</b><span>Standard scaling so every signal contributes on an equal footing.</span></div>
    <div class="step"><b>04</b><span>Linear regression predicts the selling price in Lakh INR, shown here with Crore conversion.</span></div>
    </div>
    <div class="eyebrow">Marques covered</div>
    <div class="tags">{tags}</div>
    <div class="note">Trained on 500 premium used-car listings (2012–2025 model years). Prices are indicative estimates
    from a statistical model and are not a formal appraisal. Specifications auto-filled for each model are typical
    values and can be edited at any time.</div>
    """)

md('<div class="foot">◆ &nbsp; AURUM · Luxury Car Valuation Studio &nbsp; ◆</div>')

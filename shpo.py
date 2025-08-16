"""
Shopify Insights-Fetcher – FastAPI application
(Zero official Shopify API usage; pure web-scraping + heuristics)

Features
- POST /fetch: Accepts {"website_url": "https://brand.com", "include_competitors": false}
  Scrapes brand site (Shopify-based) and returns a structured JSON object with:
  * product_catalog (via /products.json, paginated)
  * hero_products (from homepage product links/cards)
  * policies: privacy_policy, returns_refund, shipping, terms
  * faqs (multiple patterns supported)
  * social_handles (instagram, facebook, tiktok, youtube, twitter/x, linkedin, pinterest)
  * contacts (emails, phones, addresses if available)
  * brand_context (about text)
  * important_links (order tracking, contact us, blog, careers, store locator, etc.)
  * meta (is_shopify, theme hints, storefront name, timestamp)

- Optional competitor analysis: if include_competitors=True, performs a light-weight
  DuckDuckGo HTML search to find likely competitor Shopify stores and fetches their insights too.

- Persists results into a local SQLite DB (shopify_insights.db) using SQLAlchemy.

Run
  pip install fastapi uvicorn[standard] requests beautifulsoup4 lxml html5lib tldextract sqlalchemy python-dateutil
  uvicorn shopify_insights_app:app --reload --port 8000

Sample
  curl -X POST http://127.0.0.1:8000/fetch \
       -H "Content-Type: application/json" \
       -d '{"website_url":"https://memy.co.in","include_competitors":false}'

Notes
- Be respectful: set a modest crawl budget and timeouts. This is an example and not
  intended for high-volume scraping. Always check a site's robots.txt and terms.
- FAQ parsing is heuristic-driven. It attempts multiple patterns (details/summary,
  accordions, schema.org FAQPage JSON-LD, and per-question pages).
- "Hero products" are approximated by collecting /products/ links on the homepage
  that appear in prominent sections (cards, featured blocks) – still heuristic.
- Competitor discovery is best-effort via DuckDuckGo HTML (no API keys). Results vary.
"""

from __future__ import annotations
import re
import json
import time
import math
import html
import tldextract
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# ---------------------- DB setup ----------------------
ENGINE = create_engine("sqlite:///shopify_insights.db", echo=False, future=True)
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)
Base = declarative_base()

class Brand(Base):
    __tablename__ = "brands"
    id = Column(Integer, primary_key=True)
    domain = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)
    is_shopify = Column(Boolean, default=False)
    storefront_title = Column(String)
    about_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    products = relationship("Product", cascade="all, delete-orphan")
    policies = relationship("Policy", cascade="all, delete-orphan")
    faqs = relationship("FAQ", cascade="all, delete-orphan")
    socials = relationship("SocialHandle", cascade="all, delete-orphan")
    contacts = relationship("ContactDetail", cascade="all, delete-orphan")
    links = relationship("ImportantLink", cascade="all, delete-orphan")

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    brand_id = Column(Integer, ForeignKey("brands.id"))
    handle = Column(String, index=True)
    title = Column(String)
    url = Column(String)
    vendor = Column(String)
    product_type = Column(String)
    tags = Column(Text)
    price_min = Column(String)
    price_max = Column(String)
    images = Column(Text)  # JSON list
    is_hero = Column(Boolean, default=False)

class Policy(Base):
    __tablename__ = "policies"
    id = Column(Integer, primary_key=True)
    brand_id = Column(Integer, ForeignKey("brands.id"))
    kind = Column(String)  # privacy, returns_refund, shipping, terms, other
    url = Column(String)
    content = Column(Text)

class FAQ(Base):
    __tablename__ = "faqs"
    id = Column(Integer, primary_key=True)
    brand_id = Column(Integer, ForeignKey("brands.id"))
    question = Column(Text)
    answer = Column(Text)

class SocialHandle(Base):
    __tablename__ = "socials"
    id = Column(Integer, primary_key=True)
    brand_id = Column(Integer, ForeignKey("brands.id"))
    platform = Column(String)
    handle = Column(String)
    url = Column(String)

class ContactDetail(Base):
    __tablename__ = "contacts"
    id = Column(Integer, primary_key=True)
    brand_id = Column(Integer, ForeignKey("brands.id"))
    kind = Column(String)  # email, phone, address, whatsapp
    value = Column(String)
    url = Column(String, nullable=True)

class ImportantLink(Base):
    __tablename__ = "links"
    id = Column(Integer, primary_key=True)
    brand_id = Column(Integer, ForeignKey("brands.id"))
    label = Column(String)
    url = Column(String)

from pydantic import BaseModel

class BrandRequest(BaseModel):
    domain: str


class CrawlLog(Base):
    __tablename__ = "crawl_logs"
    id = Column(Integer, primary_key=True)
    brand_id = Column(Integer, ForeignKey("brands.id"), nullable=True)
    domain = Column(String)
    status = Column(String)
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(ENGINE)

# ---------------------- HTTP utils ----------------------
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def normalize_url(url: str) -> str:
    if not url.startswith("http"):
        url = "https://" + url
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return base.rstrip("/")


def fetch(url: str, timeout: int = 15) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        return resp
    except requests.RequestException:
        return None

# ---------------------- Shopify detectors ----------------------
SHOPIFY_HINT_HOSTS = ["cdn.shopify.com", "myshopify.com"]


def looks_like_shopify(html_text: str, response: requests.Response) -> bool:
    if not response:
        return False
    if any(h in (response.headers.get("Server", "") or "").lower() for h in ["shopify"]):
        return True
    if "shopify" in (response.headers.get("X-Powered-By", "") or "").lower():
        return True
    if any(host in (response.text or "") for host in SHOPIFY_HINT_HOSTS):
        return True
    # meta generator
    if re.search(r"<meta[^>]+generator\"?[^>]+Shopify", html_text, re.I):
        return True
    return False

# ---------------------- Parsers ----------------------

PRODUCTS_PAGE_LIMIT = 250


def fetch_products(base: str, limit: int = PRODUCTS_PAGE_LIMIT, max_pages: int = 10) -> List[Dict[str, Any]]:
    products = []
    for page in range(1, max_pages + 1):
        url = f"{base}/products.json?limit={limit}&page={page}"
        resp = fetch(url)
        if not resp or resp.status_code >= 400:
            break
        try:
            data = resp.json()
        except Exception:
            break
        page_products = data.get("products", [])
        if not page_products:
            break
        products.extend(page_products)
        if len(page_products) < limit:
            break
    return products


def extract_home_links(base: str, html_text: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html_text, "lxml")
    anchors = soup.find_all("a", href=True)
    links = [urljoin(base + "/", a["href"]) for a in anchors]

    # Collect links by label hints
    def find_first_url(keywords: List[str]) -> Optional[str]:
        for a in anchors:
            t = (a.get_text(strip=True) or "").lower()
            if any(k in t for k in keywords):
                return urljoin(base + "/", a["href"]) 
        return None

    important = {
        "contact": find_first_url(["contact", "support", "help"]),
        "blog": find_first_url(["blog"]),
        "track_order": find_first_url(["track", "order tracking", "track order", "track your order"]),
        "about": find_first_url(["about", "our story", "who we are"]),
        "faq": find_first_url(["faq", "faqs", "frequently asked questions", "help center", "help centre"]),
        "privacy": find_first_url(["privacy"]),
        "returns": find_first_url(["return", "refund"]),
        "shipping": find_first_url(["shipping", "delivery"]),
        "terms": find_first_url(["terms", "terms of service", "tos"]),
        "careers": find_first_url(["careers", "jobs"]),
        "store_locator": find_first_url(["store locator", "stores"]),
    }

    # Hero products: grab prominent product links from homepage
    product_link_re = re.compile(r"/products/", re.I)
    product_links = []
    for a in anchors:
        href = a["href"]
        if product_link_re.search(href):
            product_links.append(urljoin(base + "/", href))
    # Deduplicate while keeping order
    seen = set()
    hero_products = []
    for link in product_links:
        if link not in seen:
            hero_products.append(link)
            seen.add(link)
    return {"links": links, "important": important, "hero_products": hero_products}


SOCIAL_PATTERNS = {
    "instagram": re.compile(r"instagram\.com/([A-Za-z0-9_.-]+)", re.I),
    "facebook": re.compile(r"facebook\.com/(?!share|sharer|dialog)([A-Za-z0-9_.-]+)", re.I),
    "tiktok": re.compile(r"tiktok\.com/@([A-Za-z0-9_.-]+)", re.I),
    "youtube": re.compile(r"youtube\.com/(?:c/|channel/|@)?([A-Za-z0-9_.-]+)", re.I),
    "twitter": re.compile(r"(?:twitter|x)\.com/([A-Za-z0-9_.-]+)", re.I),
    "linkedin": re.compile(r"linkedin\.com/company/([A-Za-z0-9_.-]+)", re.I),
    "pinterest": re.compile(r"pinterest\.com/([A-Za-z0-9_.-]+)", re.I),
}


def extract_socials(all_links: List[str]) -> List[Dict[str, str]]:
    out = []
    added = set()
    for link in all_links:
        for platform, regex in SOCIAL_PATTERNS.items():
            m = regex.search(link)
            if m:
                handle = m.group(1)
                key = (platform, handle)
                if key in added:
                    continue
                out.append({"platform": platform, "handle": handle, "url": link})
                added.add(key)
    return out


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")


def extract_contacts(base: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    out = []
    # Emails
    for m in set(EMAIL_RE.findall(html_text or "")):
        out.append({"kind": "email", "value": m, "url": None})
    for a in soup.select('a[href^="mailto:"]'):
        val = a.get('href', '').replace('mailto:', '')
        if val:
            out.append({"kind": "email", "value": val, "url": None})
    # Phones
    for a in soup.select('a[href^="tel:"]'):
        val = a.get('href', '').replace('tel:', '')
        if val:
            out.append({"kind": "phone", "value": val, "url": None})
    for m in set(PHONE_RE.findall(html_text or "")):
        out.append({"kind": "phone", "value": m.strip(), "url": None})
    # WhatsApp
    for a in soup.select('a[href*="wa.me"], a[href*="api.whatsapp.com"]'):
        out.append({"kind": "whatsapp", "value": a.get('href'), "url": a.get('href')})
    return out


def fetch_text_page(url: str) -> str:
    resp = fetch(url)
    if not resp or resp.status_code >= 400:
        return ""
    soup = BeautifulSoup(resp.text, "lxml")
    # Try to remove nav/footer/scripts
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    text = "\n".join(t.strip() for t in soup.get_text("\n").splitlines() if t.strip())
    return text[:50000]


def parse_policies(base: str, home: Dict[str, Any]) -> List[Dict[str, str]]:
    policies = []
    mapping = {
        "privacy": home['important'].get('privacy') or f"{base}/policies/privacy-policy",
        "returns_refund": home['important'].get('returns') or f"{base}/policies/refund-policy",
        "shipping": home['important'].get('shipping') or f"{base}/policies/shipping-policy",
        "terms": home['important'].get('terms') or f"{base}/policies/terms-of-service",
    }
    for kind, url in mapping.items():
        if not url:
            continue
        content = fetch_text_page(url)
        if content:
            policies.append({"kind": kind, "url": url, "content": content})
    return policies


def parse_about(base: str, home: Dict[str, Any]) -> str:
    about_url = home['important'].get('about') or f"{base}/pages/about-us"
    return fetch_text_page(about_url)


FAQ_KEYWORDS = ["faq", "faqs", "frequently asked questions", "help center", "help centre", "support"]


def discover_faq_urls(base: str, home_html: str, home: Dict[str, Any]) -> List[str]:
    urls = []
    if home['important'].get('faq'):
        urls.append(home['important']['faq'])
    soup = BeautifulSoup(home_html, "lxml")
    for a in soup.find_all('a', href=True):
        text = (a.get_text(strip=True) or "").lower()
        if any(k in text for k in FAQ_KEYWORDS):
            urls.append(urljoin(base + "/", a['href']))
    # Deduplicate
    out = []
    seen = set()
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out[:5]


def parse_faq_page(url: str) -> List[Dict[str, str]]:
    resp = fetch(url)
    if not resp or resp.status_code >= 400:
        return []
    soup = BeautifulSoup(resp.text, "lxml")

    # 1) JSON-LD FAQPage
    faqs: List[Dict[str, str]] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
        except Exception:
            continue
        items = []
        if isinstance(data, list):
            items = data
        else:
            items = [data]
        for item in items:
            if item.get("@type") == "FAQPage" and "mainEntity" in item:
                for q in item.get("mainEntity", []):
                    if q.get("@type") == "Question":
                        question = html.unescape(q.get("name", "")).strip()
                        answers = q.get("acceptedAnswer") or q.get("acceptedAnswers")
                        if isinstance(answers, list) and answers:
                            ans = answers[0]
                        else:
                            ans = answers
                        answer = ""
                        if isinstance(ans, dict):
                            answer = html.unescape(ans.get("text", "")).strip()
                        if question and answer:
                            faqs.append({"question": question, "answer": answer})

    if faqs:
        return faqs

    # 2) <details><summary>
    for det in soup.find_all("details"):
        q = det.find("summary")
        if q:
            question = q.get_text(strip=True)
            answer = det.get_text("\n", strip=True)
            answer = answer.replace(question, "", 1).strip()
            if question and answer:
                faqs.append({"question": question, "answer": answer})

    if faqs:
        return faqs

    # 3) Accordion patterns
    for h in soup.find_all(["h2", "h3", "h4"]):
        q = h.get_text(strip=True)
        nxt = h.find_next_sibling()
        if q and nxt:
            a = nxt.get_text("\n", strip=True)
            if len(q) < 140 and len(a) > 0:
                faqs.append({"question": q, "answer": a})
    if faqs:
        return faqs

    # 4) Per-question pages heuristic: collect question links and fetch
    question_links = []
    for a in soup.find_all('a', href=True):
        t = (a.get_text(strip=True) or "").lower()
        if any(word in t for word in ["how", "what", "when", "where", "why", "can", "do you", "is there"]):
            href = urljoin(url, a['href'])
            if urlparse(href).netloc == urlparse(url).netloc:
                question_links.append(href)
    question_links = list(dict.fromkeys(question_links))[:8]

    for qurl in question_links:
        txt = fetch_text_page(qurl)
        if not txt:
            continue
        lines = txt.splitlines()
        if not lines:
            continue
        question = lines[0][:200]
        answer = "\n".join(lines[1:])[:1000]
        if question and answer:
            faqs.append({"question": question, "answer": answer})

    return faqs


def extract_storefront_title(home_html: str) -> str:
    soup = BeautifulSoup(home_html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()[:200]
    og = soup.find("meta", property="og:site_name")
    if og and og.get("content"):
        return og["content"].strip()[:200]
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)[:200]
    return ""


from typing import Dict, Any, Tuple

def price_from_variants(prod: Dict[str, Any]) -> Tuple[str, str]:
    variants = prod.get("variants", [])
    if not variants:
        return ("0", "0")
    prices = [float(v.get("price", 0)) for v in variants]
    return (str(min(prices)), str(max(prices)))


# ---------------------- Competitor discovery (best-effort) ----------------------
DUCK_URL = "https://duckduckgo.com/html/"


def discover_competitors(brand_name: str, domain: str, limit: int = 3) -> List[str]:
    if not brand_name:
        ext = tldextract.extract(domain)
        brand_name = ext.domain
    q = f"similar brands to {brand_name} site:.com OR site:.co OR site:.in shopify"
    try:
        resp = requests.get(DUCK_URL, params={"q": q}, headers=DEFAULT_HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        out = []
        for a in soup.select("a.result__a"):
            href = a.get("href")
            if not href:
                continue
            netloc = urlparse(href).netloc
            if not netloc or netloc.endswith("duckduckgo.com"):
                continue
            base = normalize_url(href)
            if base not in out and domain not in base:
                out.append(base)
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []

# ---------------------- Orchestration ----------------------

class FetchRequest(BaseModel):
    website_url: HttpUrl
    include_competitors: bool = False


app = FastAPI(title="Shopify Insights-Fetcher", version="1.0.0")


@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}


import tldextract

@app.post("/brand")
def get_brand(req: BrandRequest):
    extracted = tldextract.extract(req.domain)
    brand_name = extracted.domain.capitalize()
    return {"brand": brand_name}


def serialize_brand(b: Brand) -> Dict[str, Any]:
    return {
        "domain": b.domain,
        "name": b.name,
        "meta": {
            "is_shopify": b.is_shopify,
            "storefront_title": b.storefront_title,
            "timestamp": b.timestamp.isoformat() if b.timestamp else None,
        },
        "brand_context": b.about_text,
        "product_catalog": [
            {
                "handle": p.handle, "title": p.title, "url": p.url, "vendor": p.vendor,
                "product_type": p.product_type, "tags": json.loads(p.tags or "[]"),
                "price_min": p.price_min, "price_max": p.price_max,
                "images": json.loads(p.images or "[]"),
                "is_hero": p.is_hero,
            } for p in b.products
        ],
        "policies": [
            {"kind": p.kind, "url": p.url, "content": p.content} for p in b.policies
        ],
        "faqs": [
            {"question": f.question, "answer": f.answer} for f in b.faqs
        ],
        "social_handles": [
            {"platform": s.platform, "handle": s.handle, "url": s.url} for s in b.socials
        ],
        "contacts": [
            {"kind": c.kind, "value": c.value, "url": c.url} for c in b.contacts
        ],
        "important_links": [
            {"label": l.label, "url": l.url} for l in b.links
        ],
    }

import json
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
from urllib.parse import urljoin

import requests
import tldextract
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)

# -------------------- FastAPI app --------------------
app = FastAPI(title="Shopify Store Insights-Fetcher (Live)")

# -------------------- Request models --------------------
class FetchRequest(BaseModel):
    website_url: str
    include_competitors: Optional[bool] = False

# -------------------- Helper functions --------------------
def normalize_url(url: str) -> str:
    return url.rstrip("/")

def fetch(url: str, timeout: int = 10):
    """Fetch a URL and return response or None."""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        return resp
    except Exception as e:
        logging.error(f"Fetch failed for {url}: {e}")
        return None

def looks_like_shopify(home_html: str, resp: Any) -> bool:
    """Simple Shopify detection based on /products.json route or meta tags."""
    if "/products.json" in home_html:
        return True
    if 'cdn.shopify.com' in home_html:
        return True
    return False

def fetch_products(base: str) -> list:
    """Try to fetch the products.json from a Shopify store."""
    try:
        url = urljoin(base + "/", "/products.json?limit=250")
        resp = fetch(url)
        if resp and resp.status_code == 200:
            data = resp.json()
            return data.get("products", [])
        else:
            return []
    except Exception as e:
        logging.error(f"Failed to fetch products.json for {base}: {e}")
        return []

# -------------------- Minimal HTML extraction helpers --------------------
def extract_storefront_title(html: str) -> str:
    import re
    match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def extract_home_links(base: str, html: str) -> Dict[str, Any]:
    """Stub: Hero products and links"""
    return {"hero_products": [], "links": [], "important": {}}

def parse_policies(base: str, home: Dict[str, Any]) -> list:
    return []

def discover_faq_urls(base: str, html: str, home: Dict[str, Any]) -> list:
    return []

def parse_faq_page(url: str) -> list:
    return []

def extract_socials(links: list) -> list:
    return []

def extract_contacts(base: str, html: str) -> list:
    return []

def parse_about(base: str, home: Dict[str, Any]) -> str:
    return ""

def discover_competitors(title: str, base: str) -> list:
    return []

# -------------------- Fetch Endpoint --------------------
@app.post("/fetch")
def fetch_brand(req: FetchRequest):
    base = normalize_url(str(req.website_url))
    logging.info(f"Fetching store: {base}")

    try:
        # Fetch homepage
        home_resp = fetch(base)
        if not home_resp or home_resp.status_code >= 400:
            logging.error(f"Failed to fetch homepage {base}")
            raise HTTPException(status_code=401, detail="Website not found or unreachable")

        home_html = home_resp.text
        is_shop = looks_like_shopify(home_html, home_resp)

        storefront_title = extract_storefront_title(home_html)
        home = extract_home_links(base, home_html)

        # Products
        product_json = fetch_products(base) or []
        prod_map = {}
        for prod in product_json:
            handle = prod.get("handle")
            url = urljoin(base + "/", f"/products/{handle}") if handle else base
            price_min = prod.get("variants", [{}])[0].get("price", "")
            price_max = prod.get("variants", [{}])[-1].get("price", "")
            prod_map[url] = {
                "handle": handle,
                "title": prod.get("title"),
                "url": url,
                "vendor": (prod.get("vendor") or "")[:200],
                "product_type": (prod.get("product_type") or "")[:200],
                "tags": prod.get("tags") or [],
                "price_min": price_min,
                "price_max": price_max,
                "images": [img.get("src") for img in prod.get("images", []) if img.get("src")],
                "is_hero": False,
            }

        # Hero products
        for link in home.get("hero_products", [])[:20]:
            if link in prod_map:
                prod_map[link]["is_hero"] = True
            else:
                prod_map[link] = {
                    "handle": link.split("/products/")[-1],
                    "title": None,
                    "url": link,
                    "vendor": "",
                    "product_type": "",
                    "tags": [],
                    "price_min": "",
                    "price_max": "",
                    "images": [],
                    "is_hero": True,
                }

        # Build JSON response
        brand = {
            "domain": base,
            "name": tldextract.extract(base).domain.capitalize(),
            "meta": {
                "is_shopify": bool(is_shop),
                "storefront_title": storefront_title,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "brand_context": parse_about(base, home) or "",
            "product_catalog": list(prod_map.values()),
            "policies": parse_policies(base, home),
            "faqs": [],
            "social_handles": extract_socials(home.get("links", [])),
            "contacts": extract_contacts(base, home_html),
            "important_links": [],
        }

        logging.info(f"Scrape succeeded for {base}")
        return brand

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Internal error fetching {base}: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

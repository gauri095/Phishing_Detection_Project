from urllib.parse import urlparse
import tldextract
import re

SUSPICIOUS_TOKENS = [
    "login", "signin", "bank", "secure", "update", "verify", "confirm",
    "account", "paypal", "password", "click", "urgent", "immediately",
    "ssn", "social security", "credential", "credentials"
]

URL_REGEX = re.compile(r"https?://[^\s]+")
WORD_RE = re.compile(r"\b\w[\w']*\b")

def is_ip(host: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\.\d+\.\d+$", host.split(":")[0]))

def count_subdomains(host: str) -> int:
    parts = host.split(".")
    if len(parts) <= 2:
        return 0
    return len(parts) - 2

def extract_basic(url: str):
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc
    path = parsed.path or ""
    query = parsed.query or ""
    ext = tldextract.extract(host)
    registered = ext.registered_domain or host

    features = {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "has_ip": int(is_ip(host)),
        "num_subdomains": count_subdomains(host),
        "num_dots": host.count("."),
        "num_hyphens": host.count("-"),
        "has_at": int("@" in url),
        "has_query": int(bool(query)),
        "num_query_params": len(query.split("&")) if query else 0,
        "registered_domain": registered
    }

    text = (path + " " + query).lower()
    features["suspicious_tokens"] = sum(text.count(tok) for tok in SUSPICIOUS_TOKENS)
    return features

def extract_features(url: str):
    return extract_basic(url)

# ---------------------
# Message / text features (fallback)
# ---------------------
def count_urls_in_text(text: str) -> int:
    return len(URL_REGEX.findall(text))

def uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)

def token_stats(text: str):
    toks = WORD_RE.findall(text.lower())
    return toks

def extract_text_features(text: str):
    t = text or ""
    length = len(t)
    words = token_stats(t)
    num_words = len(words)
    unique_words = len(set(words))
    num_urls = count_urls_in_text(t)
    num_digits = sum(ch.isdigit() for ch in t)
    num_exclamations = t.count("!")
    up_ratio = uppercase_ratio(t)
    susp_count = sum(1 for tok in SUSPICIOUS_TOKENS if tok in t.lower())
    avg_word_len = (sum(len(w) for w in words) / num_words) if num_words else 0.0
    flesch_proxy = 206.835 - 1.015 * (num_words / (1 if num_words==0 else num_words)) - 84.6 * (sum(len(w) for w in words) / (num_words if num_words else 1))
    return {
        "text_length": length,
        "num_words": num_words,
        "unique_words": unique_words,
        "num_urls": num_urls,
        "num_digits": num_digits,
        "num_exclamations": num_exclamations,
        "uppercase_ratio": round(up_ratio, 4),
        "suspicious_token_count": susp_count,
        "avg_word_len": round(avg_word_len, 2),
        "flesch_proxy": round(flesch_proxy, 2),
    }
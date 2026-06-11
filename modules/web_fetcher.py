"""Web content fetcher: fetch URLs, extract text, discover and download data files."""

import urllib.request
import urllib.error
import urllib.parse
import re
import os
import io
import tempfile
from html.parser import HTMLParser
from http.cookiejar import CookieJar
import chardet

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


_COOKIE_JAR = CookieJar()


def set_custom_cookies(cookie_string):
    """Accept a raw cookie string (from browser DevTools) and add to the cookie jar."""
    import http.cookiejar
    for cookie in cookie_string.split(";"):
        cookie = cookie.strip()
        if "=" in cookie:
            name, value = cookie.split("=", 1)
    # Actually, use a simpler approach: set Cookie header directly
    global _CUSTOM_COOKIES
    _CUSTOM_COOKIES = cookie_string


_CUSTOM_COOKIES = ""


def set_cookie_header(cookie_string):
    """Set a custom Cookie header to use for all subsequent requests."""
    global _CUSTOM_COOKIES
    _CUSTOM_COOKIES = cookie_string.strip()

_COOKIE_OPENER = None


def _get_opener():
    global _COOKIE_OPENER
    if _COOKIE_OPENER is None:
        _COOKIE_OPENER = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(_COOKIE_JAR)
        )
    return _COOKIE_OPENER


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "code", "pre"):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "code", "pre"):
            self.skip = False
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th"):
            self.text.append("\n")

    def handle_data(self, data):
        if not self.skip:
            d = data.strip()
            if d:
                self.text.append(d + " ")

    def get_text(self):
        raw = "".join(self.text)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r" {2,}", " ", raw)
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return "\n".join(lines)


class LinkFinder(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag in ("a", "link", "iframe", "embed", "source"):
            for name, value in attrs:
                if name in ("href", "src", "data-url", "data-file"):
                    self.links.append(value)


def _make_request(url, timeout=30):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    if _CUSTOM_COOKIES:
        headers["Cookie"] = _CUSTOM_COOKIES
    req = urllib.request.Request(url, headers=headers)
    opener = _get_opener()
    with opener.open(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        data = resp.read()
        detected = chardet.detect(data)
        chardet_enc = detected.get("encoding", "utf-8")
        confidence = detected.get("confidence", 0)
        header_enc = None
        for part in content_type.split(";"):
            if "charset" in part.lower():
                header_enc = part.split("=")[-1].strip().strip('"').strip("'")
                break
        candidates = []
        if chardet_enc and confidence > 0.7:
            candidates.append(chardet_enc)
        if header_enc:
            candidates.append(header_enc)
        candidates.append("utf-8")
        text = None
        for enc in candidates:
            try:
                text = data.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if text is None:
            text = data.decode("utf-8", errors="replace")
        return text, content_type, data


DATA_EXTENSIONS = (".csv", ".xlsx", ".xls", ".tsv", ".json", ".txt", ".pdf", ".zip")

# Canvas-specific patterns for file downloads
CANVAS_FILE_PATTERNS = [
    "/files/", "/download?download_frd=1", "/download?verifier=",
    "/preview?", "/api/v1/files/", "/courses/",
]

# Submission-related URL markers to exclude
SUBMISSION_MARKERS = [
    "/submissions/", "submission_draft", "submission%2F",
    "/assignments/", "file_upload", "uploaded_file",
    "download_frd=1&verifier=",
]


def _is_data_url(url, check_submission=True):
    url_lower = url.lower()
    # Exclude submission-related URLs
    if check_submission:
        for marker in SUBMISSION_MARKERS:
            if marker.lower() in url_lower:
                return False
    url_path = url_lower.split("?")[0]
    if any(url_path.endswith(ext) for ext in DATA_EXTENSIONS):
        return True
    # Canvas file patterns
    for pattern in CANVAS_FILE_PATTERNS:
        if pattern.lower() in url_lower:
            if "download" in url_lower or "/files/" in url_lower:
                return True
    return False


def _find_urls_in_html(html, base_url):
    """Aggressive URL finder: HTMLParser + regex patterns."""
    found = {}

    # Method 1: HTMLParser (a, link, iframe, embed, source tags)
    finder = LinkFinder()
    finder.feed(html)
    for href in finder.links:
        if href.startswith("#") or href.startswith("javascript:"):
            continue
        full = urllib.parse.urljoin(base_url, href)
        if full not in found:
            found[full] = _is_data_url(full)

    # Method 2: Regex for data file URLs anywhere in raw HTML
    data_pattern = re.compile(
        r"""https?://[^\s"'<>()]+\.(?:csv|xlsx|xls|pdf|zip|tsv|json|txt)""",
        re.IGNORECASE,
    )
    for match in data_pattern.finditer(html):
        full = match.group(0)
        if full not in found:
            found[full] = True

    # Method 3: Common LMS patterns (/files/XXX, /api/v1/files/XXX)
    lms_patterns = [
        r'(/files/\d+/download[^\s"\'<>()]*)',
        r'(/api/v1/files/\d+[^\s"\'<>()]*)',
        r'(/courses/\d+/files/\d+/download[^\s"\'<>()]*)',
    ]
    for pat in lms_patterns:
        for match in re.finditer(pat, html, re.IGNORECASE):
            full = urllib.parse.urljoin(base_url, match.group(1))
            if full not in found:
                found[full] = _is_data_url(full)

    return found


def _detect_login_page(html_text):
    indicators = [
        'type="password"', 'name="password"', 'id="password"',
        'login-form', 'signin-form', 'Please log in', 'Please sign in',
        'authentication required', 'access denied', 'unauthorized',
    ]
    return [i for i in indicators if i.lower() in html_text.lower()]


def fetch_page(url):
    result = {"success": False, "url": url, "text": "", "title": "",
              "links": [], "data_files": [], "error": "",
              "login_wall": False, "login_indicators": []}

    try:
        html_text, content_type, raw_bytes = _make_request(url)
    except urllib.error.HTTPError as e:
        result["error"] = "HTTP {}: {}".format(e.code, e.reason)
        return result
    except urllib.error.URLError as e:
        result["error"] = "Connection failed: {}".format(e.reason)
        return result
    except Exception as e:
        result["error"] = str(e)
        return result

    # Title
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if title_match:
        result["title"] = title_match.group(1).strip()

    # Text extraction
    if HAS_BS4:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "code", "pre"]):
            tag.decompose()
        result["text"] = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in result["text"].split("\n") if l.strip()]
        result["text"] = "\n".join(lines)
    else:
        extractor = TextExtractor()
        extractor.feed(html_text)
        result["text"] = extractor.get_text()

    if len(result["text"]) > 50000:
        result["text"] = result["text"][:50000] + "\n\n[Content truncated at 50,000 characters...]"

    # Login detection
    login_indicators = _detect_login_page(html_text)
    if login_indicators:
        result["login_wall"] = True
        result["login_indicators"] = login_indicators

    # Link extraction
    url_map = _find_urls_in_html(html_text, url)
    for full_url, is_data in url_map.items():
        link_info = {"url": full_url, "is_data": is_data, "text": full_url[:100]}
        result["links"].append(link_info)
        if is_data:
            result["data_files"].append(link_info)

    result["success"] = True
    return result


def download_data_file(url):
    try:
        text, content_type, raw_bytes = _make_request(url)
        parsed = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename:
            filename = "downloaded_data"
        return raw_bytes, filename, None
    except Exception as e:
        return None, "", str(e)


def try_load_as_dataframe(file_bytes, filename):
    import pandas as pd
    import io as io_mod

    fn_lower = filename.split("?")[0].lower()
    if fn_lower.endswith(".pdf"):
        return None
    try:
        if any(fn_lower.endswith(ext) for ext in [".csv", ".txt", ".tsv"]):
            return pd.read_csv(io_mod.BytesIO(file_bytes))
        elif any(fn_lower.endswith(ext) for ext in [".xlsx", ".xls"]):
            return pd.read_excel(io_mod.BytesIO(file_bytes))
        elif fn_lower.endswith(".json"):
            return pd.read_json(io_mod.BytesIO(file_bytes))
        else:
            try:
                return pd.read_csv(io_mod.BytesIO(file_bytes))
            except Exception:
                try:
                    return pd.read_excel(io_mod.BytesIO(file_bytes))
                except Exception:
                    return None
    except Exception:
        return None

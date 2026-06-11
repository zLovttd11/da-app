"""Web content fetcher: fetch URLs, extract text, discover and download data files."""

import urllib.request
import urllib.error
import urllib.parse
from http.cookiejar import CookieJar
import re
import os
import io
import tempfile
from html.parser import HTMLParser

# Try to use BeautifulSoup if available, fallback to basic parser
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class TextExtractor(HTMLParser):
    """Extract visible text from HTML, stripping tags and scripts."""
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript', 'code', 'pre'):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript', 'code', 'pre'):
            self.skip = False
        if tag in ('p', 'br', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr', 'td', 'th'):
            self.text.append('\n')

    def handle_data(self, data):
        if not self.skip:
            d = data.strip()
            if d:
                self.text.append(d + ' ')

    def get_text(self):
        raw = ''.join(self.text)
        # Collapse whitespace
        raw = re.sub(r'\n{3,}', '\n\n', raw)
        raw = re.sub(r' {2,}', ' ', raw)
        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        return '\n'.join(lines)


class LinkFinder(HTMLParser):
    """Find all link hrefs in HTML."""
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for name, value in attrs:
                if name == 'href':
                    self.links.append(value)


# Shared cookie jar for session persistence across requests
_COOKIE_JAR = CookieJar()
_COOKIE_OPENER = None

def _get_opener():
    """Get or create a cookie-aware opener."""
    global _COOKIE_OPENER
    if _COOKIE_OPENER is None:
        _COOKIE_OPENER = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(_COOKIE_JAR)
        )
    return _COOKIE_OPENER


def _make_request(url, timeout=30):
    """Make an HTTP request with common headers and cookie support."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    req = urllib.request.Request(url, headers=headers)
    opener = _get_opener()
    with opener.open(req, timeout=timeout) as resp:
        content_type = resp.headers.get('Content-Type', '')
        data = resp.read()
        # Try to detect encoding
        encoding = 'utf-8'
        for part in content_type.split(';'):
            if 'charset' in part.lower():
                encoding = part.split('=')[-1].strip()
                break
        if encoding.lower() not in ('utf-8', 'utf8'):
            try:
                text = data.decode(encoding)
            except Exception:
                text = data.decode('utf-8', errors='replace')
        else:
            text = data.decode('utf-8', errors='replace')
        return text, content_type, data


def fetch_page(url):
    """Fetch a URL and return structured result.

    Returns dict with:
    - success: bool
    - url: final URL
    - text: extracted readable text
    - title: page title
    - links: list of dicts {url, text, is_data}
    - data_files: list of downloadable data file info
    - error: if failed
    """
    result = {'success': False, 'url': url, 'text': '', 'title': '', 'links': [], 'data_files': [], 'error': ''}

    try:
        html_text, content_type, raw_bytes = _make_request(url)
    except urllib.error.HTTPError as e:
        result['error'] = f'HTTP {e.code}: {e.reason}'
        return result
    except urllib.error.URLError as e:
        result['error'] = f'Connection failed: {e.reason}'
        return result
    except Exception as e:
        result['error'] = str(e)
        return result

    # Extract title
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html_text, re.IGNORECASE | re.DOTALL)
    if title_match:
        result['title'] = title_match.group(1).strip()

    # Extract text
    if HAS_BS4:
        soup = BeautifulSoup(html_text, 'html.parser')
        for tag in soup(['script', 'style', 'noscript', 'code', 'pre']):
            tag.decompose()
        result['text'] = soup.get_text(separator='\n', strip=True)
        # Clean up
        lines = [l.strip() for l in result['text'].split('\n') if l.strip()]
        result['text'] = '\n'.join(lines)
    else:
        extractor = TextExtractor()
        extractor.feed(html_text)
        result['text'] = extractor.get_text()

    # Find links
    finder = LinkFinder()
    finder.feed(html_text)
    parsed_base = urllib.parse.urlparse(url)

    for href in finder.links:
        if href.startswith('#') or href.startswith('javascript:'):
            continue
        # Resolve relative URLs
        full_url = urllib.parse.urljoin(url, href)
        # Check if it's a data file
        fn_lower = full_url.lower()
        is_data = any(fn_lower.endswith(ext) for ext in
                      ['.csv', '.xlsx', '.xls', '.tsv', '.json', '.txt'])
        link_info = {'url': full_url, 'is_data': is_data, 'text': href[:100]}
        result['links'].append(link_info)
        if is_data:
            result['data_files'].append(link_info)

    result['success'] = True

    # Truncate text if too long
    if len(result['text']) > 50000:
        result['text'] = result['text'][:50000] + '\n\n[Content truncated at 50,000 characters...]'

    return result


def download_data_file(url):
    """Download a data file from URL, return (bytes, filename, error)."""
    try:
        text, content_type, raw_bytes = _make_request(url)
        # Get filename from URL
        parsed = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename:
            filename = 'downloaded_data'
        return raw_bytes, filename, None
    except Exception as e:
        return None, '', str(e)


def try_load_as_dataframe(file_bytes, filename):
    """Try loading bytes as a pandas DataFrame."""
    import pandas as pd
    import io as io_mod

    fn_lower = filename.lower()
    try:
        if fn_lower.endswith('.csv') or fn_lower.endswith('.txt') or fn_lower.endswith('.tsv'):
            return pd.read_csv(io_mod.BytesIO(file_bytes))
        elif fn_lower.endswith('.xlsx') or fn_lower.endswith('.xls'):
            return pd.read_excel(io_mod.BytesIO(file_bytes))
        elif fn_lower.endswith('.json'):
            return pd.read_json(io_mod.BytesIO(file_bytes))
        else:
            # Try CSV first
            try:
                return pd.read_csv(io_mod.BytesIO(file_bytes))
            except Exception:
                pass
            # Try Excel
            try:
                return pd.read_excel(io_mod.BytesIO(file_bytes))
            except Exception:
                pass
            return None
    except Exception:
        return None

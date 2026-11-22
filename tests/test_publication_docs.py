"""Offline checks for publication links, metadata, and cited result values."""
import csv
import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOI = "10.1145/3858063.3858809"
BASE = "https://jsz-05.github.io/same-market-different-pixels/"
TITLE = "Same Market, Different Pixels: Can Invariance Training Repair Financial Vision Models?"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links, self.ids, self.meta, self.canonicals = [], set(), {}, []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag in {"a", "link", "img"}:
            value = attrs.get("href", attrs.get("src"))
            if value:
                self.links.append(value)
        if tag == "meta":
            self.meta[attrs.get("name", attrs.get("property"))] = attrs.get("content")
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonicals.append(attrs["href"])


class PublicationDocsTests(unittest.TestCase):
    def test_local_links_and_fragments(self):
        for source in DOCS.glob("*.html"):
            for link in Page(source.read_text()).links:
                if link.startswith(BASE):
                    link = link[len(BASE):] or "index.html"
                parts = urlsplit(link)
                if parts.scheme or parts.netloc:
                    continue
                target = source.parent / unquote(parts.path) if parts.path else source
                if target.is_dir():
                    target = target / "index.html"
                self.assertTrue(target.is_file(), (source.name, link))
                if parts.fragment and target.suffix == ".html":
                    self.assertIn(parts.fragment, Page(target.read_text()).ids)

    def test_article_metadata(self):
        text = (DOCS / "index.html").read_text()
        page = Page(text)
        self.assertEqual(page.meta["citation_doi"], DOI)
        self.assertEqual(page.meta["citation_title"], TITLE)
        self.assertEqual(page.meta["citation_author"], "Jeffrey Zhou")
        self.assertEqual(page.canonicals, [BASE])
        block = re.search(r'<script type="application/ld\+json">(.*?)</script>', text, re.S)
        metadata = json.loads(block.group(1))
        self.assertEqual(metadata["@id"], "https://doi.org/" + DOI)
        self.assertEqual(metadata["headline"], TITLE)

    def test_citation_files_agree(self):
        bib = (DOCS / "paper.bib").read_text().strip()
        self.assertIn(bib, (ROOT / "README.md").read_text())
        cff = (ROOT / "CITATION.cff").read_text()
        self.assertIn("preferred-citation:", cff)
        for name in ["index.html", "evidence.html", "faq.html", "llms.txt", "paper.bib"]:
            self.assertIn(DOI, (DOCS / name).read_text())
        self.assertIn(DOI, cff)

    def test_sitemap_and_agent_links(self):
        xml = ElementTree.parse(DOCS / "sitemap.xml")
        locations = [e.text for e in xml.findall(".//{*}loc")]
        self.assertEqual(set(locations), {BASE, BASE + "evidence.html", BASE + "faq.html"})
        for link in re.findall(r'\]\(([^)]+)\)', (DOCS / "llms.txt").read_text()):
            if link.startswith(BASE):
                self.assertTrue((DOCS / (link[len(BASE):] or "index.html")).is_file())

    def test_external_selection_and_quoted_values(self):
        with (ROOT / "results/extension_analysis/extension_summary.csv").open() as f:
            rows = list(csv.DictReader(f))
        for method, expected in [
            ("canonical_raw", ("0.543", "0.658", "15.3")),
            ("rand_cons_raw", ("0.347", "0.493", "19.8")),
        ]:
            subset = [r for r in rows if r["family"] == "external_renderer" and r["method"] == method]
            selected = max(subset, key=lambda r: float(r["mean_absolute_shift"]))
            actual = (f'{float(selected["mean_absolute_shift"]):.3f}',
                      f'{float(selected["mae_log_rv"]):.3f}',
                      f'{100 * float(selected["decision_flip_rate"]):.1f}')
            self.assertEqual(actual, expected)
            for value in expected:
                self.assertIn(value, (DOCS / "evidence.html").read_text())


if __name__ == "__main__":
    unittest.main()

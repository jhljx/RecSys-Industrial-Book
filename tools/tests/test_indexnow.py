from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import notify_indexnow as indexnow


class IndexNowTests(unittest.TestCase):
    def test_key_is_in_project_root(self):
        self.assertEqual(indexnow.KEY + "\n", (ROOT / (indexnow.KEY + ".txt")).read_text())
        self.assertEqual(indexnow.BASE + indexnow.KEY + ".txt", indexnow.KEY_LOCATION)

    def test_changed_urls_use_sitemap_and_skip_unpublished_assets(self):
        before = b'<urlset><url><loc>https://jhljx.github.io/RecSys-Industrial-Book/knowledge/concepts/old.html</loc></url></urlset>'
        after = b'<urlset><url><loc>https://jhljx.github.io/RecSys-Industrial-Book/knowledge/concepts/new.html</loc></url><url><loc>https://jhljx.github.io/RecSys-Industrial-Book/knowledge-map/</loc></url></urlset>'
        def blob(commit, path):
            return before if commit == "old" else after
        with patch.object(indexnow, "git_bytes", side_effect=blob), patch.object(indexnow.subprocess, "check_output", return_value=b"knowledge/concepts/new.md\0knowledge-map/index.html\0secret.tex\0"):
            urls = indexnow.changed_urls("old", "new")
        self.assertEqual([indexnow.BASE + "knowledge-map/", indexnow.BASE + "knowledge/concepts/new.html", indexnow.BASE + "knowledge/concepts/old.html"], sorted(urls))

    def test_live_gate_prevents_post_on_mismatch(self):
        with patch.object(indexnow, "check_live", side_effect=ValueError("not deployed")), patch.object(indexnow, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, "not deployed"):
                indexnow.notify([indexnow.BASE + "knowledge-map/"], "new")
            request.assert_not_called()

    def test_payload_includes_scoped_key_location(self):
        urls = [indexnow.BASE + "knowledge-map/"]
        class Response:
            status = 202
            def __enter__(self): return self
            def __exit__(self, *_): return False
        with patch.object(indexnow, "git_bytes", return_value=b'<urlset/>'), patch.object(indexnow, "check_live") as gate, patch.object(indexnow, "urlopen", return_value=Response()) as request:
            self.assertEqual(0, indexnow.notify(urls, "new"))
        gate.assert_called_once()
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual("jhljx.github.io", payload["host"])
        self.assertEqual(indexnow.KEY_LOCATION, payload["keyLocation"])
        self.assertEqual(urls, payload["urlList"])


if __name__ == "__main__":
    unittest.main()

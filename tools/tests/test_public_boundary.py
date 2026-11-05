"""Public-source boundary regression tests."""
from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import check_public_boundary as boundary  # noqa: E402


class PublicSourceBoundaryTest(unittest.TestCase):
    def test_private_authoring_reference_is_rejected_in_chapter_code(self) -> None:
        self.assertRegex("见 chapters/ch24" + ".tex §24.1", boundary.PRIVATE_SOURCE_REF)
        self.assertRegex("见 ch24" + ".tex §24.1", boundary.PRIVATE_SOURCE_REF)
        self.assertNotRegex("见第 24 章发布 PDF §24.1", boundary.PRIVATE_SOURCE_REF)

    def test_personal_state_and_generated_text_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for relative, content in ((".recsys-workspace/progress.jsonl", "private learning state"),
                                      ("knowledge-map/generated.html", "see chapters/ch24" + ".tex"),
                                      ("knowledge-map/data.js", "see ch24" + ".tex")):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                with patch.object(boundary, "tracked_files", return_value=[path]), \
                     patch.object(sys, "argv", ["check_public_boundary", "--root", str(root)]), \
                     contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit):
                    boundary.main()
                self.assertIn(relative, output.getvalue())


if __name__ == "__main__":
    unittest.main()

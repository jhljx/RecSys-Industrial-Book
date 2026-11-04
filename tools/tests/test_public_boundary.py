"""Public-source boundary regression tests."""
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import check_public_boundary as boundary  # noqa: E402


class PublicSourceBoundaryTest(unittest.TestCase):
    def test_private_authoring_reference_is_rejected_in_chapter_code(self) -> None:
        self.assertRegex("见 chapters/ch24" + ".tex §24.1", boundary.PRIVATE_SOURCE_REF)
        self.assertRegex("见 ch24" + ".tex §24.1", boundary.PRIVATE_SOURCE_REF)
        self.assertNotRegex("见第 24 章发布 PDF §24.1", boundary.PRIVATE_SOURCE_REF)


if __name__ == "__main__":
    unittest.main()

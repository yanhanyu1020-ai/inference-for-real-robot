import unittest
from pathlib import Path


class WrapperTests(unittest.TestCase):
    def test_wrappers_do_not_enable_nounset_before_ros_setup(self):
        bin_dir = Path(__file__).resolve().parents[1] / "bin"
        for name in ("p450_record", "p450_capture"):
            with self.subTest(name=name):
                wrapper = (bin_dir / name).read_text(encoding="utf-8")
                self.assertIn("set -eo pipefail", wrapper)
                self.assertNotIn("set -euo pipefail", wrapper)


if __name__ == "__main__":
    unittest.main()

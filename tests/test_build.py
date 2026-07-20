import unittest
from enum import IntEnum
from types import SimpleNamespace

from sam2_trt.build import _network_flags


class TensorRtNetworkFlagsTest(unittest.TestCase):
    def test_tensorrt_11_uses_strong_typing_without_explicit_batch(self):
        class Flags(IntEnum):
            STRONGLY_TYPED = 0

        fake = SimpleNamespace(NetworkDefinitionCreationFlag=Flags)
        self.assertEqual(_network_flags(fake), 1)

    def test_tensorrt_10_combines_explicit_batch_and_strong_typing(self):
        class Flags(IntEnum):
            EXPLICIT_BATCH = 0
            STRONGLY_TYPED = 1

        fake = SimpleNamespace(NetworkDefinitionCreationFlag=Flags)
        self.assertEqual(_network_flags(fake), 3)


if __name__ == "__main__":
    unittest.main()

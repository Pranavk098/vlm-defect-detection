"""Tests for vlm_defect.model.load_model_and_processor().

peft and transformers.LlavaForConditionalGeneration are stubbed out via
sys.modules so these tests run without a working GPU, downloaded weights,
or a compatible torchvision build.  The goal is to verify the branching
logic around lora.enable, not the correctness of the underlying libraries.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub peft before vlm_defect.model is ever imported.
#
# peft → transformers.PreTrainedModel → torchvision, which has a version
# conflict on this system (RuntimeError: operator torchvision::nms does not
# exist).  Injecting a MagicMock satisfies the top-level import in model.py
# without touching the real library.
# ---------------------------------------------------------------------------
if not isinstance(sys.modules.get("peft"), MagicMock):
    sys.modules["peft"] = MagicMock()

# transformers uses lazy __getattr__ for heavy classes; accessing
# AutoProcessor or LlavaForConditionalGeneration triggers a torchvision
# import that fails on this system (version mismatch).  Force-set them on
# the already-loaded module object so the subsequent `from transformers
# import ...` in model.py finds them in __dict__ instead of going through
# __getattr__.
import transformers as _transformers  # already imported by vlm_defect.evaluate

for _attr in ("AutoProcessor", "LlavaForConditionalGeneration", "BitsAndBytesConfig"):
    # transformers raises ModuleNotFoundError (not AttributeError) from its
    # lazy __getattr__ when the underlying module can't be loaded, so we
    # cannot use getattr(obj, attr, default) — we need an explicit try/except.
    try:
        getattr(_transformers, _attr)
    except Exception:
        setattr(_transformers, _attr, MagicMock())

# Remove any broken cached import so the module reloads cleanly with stubs.
sys.modules.pop("vlm_defect.model", None)

# Now import the module under test.
from vlm_defect.model import load_model_and_processor  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(lora_enable: bool = True, include_enable_key: bool = True) -> dict:
    """Return a minimal valid config dict."""
    lora_section: dict = {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    }
    if include_enable_key:
        lora_section["enable"] = lora_enable

    return {
        "model":        {"name_or_path": "fake/model-id"},
        "quantization": {"bits": 4, "quant_type": "nf4", "double_quant": True},
        "lora":         lora_section,
    }


# Patch targets — attributes in vlm_defect.model's namespace
_PATCH_LLAVA   = "vlm_defect.model.LlavaForConditionalGeneration.from_pretrained"
_PATCH_PROC    = "vlm_defect.model.AutoProcessor.from_pretrained"
_PATCH_BNB     = "vlm_defect.model.BitsAndBytesConfig"
_PATCH_PREPARE = "vlm_defect.model.prepare_model_for_kbit_training"
_PATCH_PEFT    = "vlm_defect.model.get_peft_model"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadModelAndProcessor:

    @patch(_PATCH_PEFT)
    @patch(_PATCH_PREPARE)
    @patch(_PATCH_BNB)
    @patch(_PATCH_PROC)
    @patch(_PATCH_LLAVA)
    def test_lora_enabled_calls_get_peft_model(
        self, mock_llava, mock_proc, mock_bnb, mock_prepare, mock_get_peft
    ):
        """When lora.enable=True, get_peft_model must be called exactly once."""
        fake_model = MagicMock()
        mock_llava.return_value    = fake_model
        mock_prepare.return_value  = fake_model
        mock_get_peft.return_value = fake_model

        load_model_and_processor(_cfg(lora_enable=True))

        mock_get_peft.assert_called_once()

    @patch(_PATCH_PEFT)
    @patch(_PATCH_PREPARE)
    @patch(_PATCH_BNB)
    @patch(_PATCH_PROC)
    @patch(_PATCH_LLAVA)
    def test_lora_disabled_skips_get_peft_model(
        self, mock_llava, mock_proc, mock_bnb, mock_prepare, mock_get_peft
    ):
        """When lora.enable=False, get_peft_model must NOT be called."""
        fake_model = MagicMock()
        mock_llava.return_value   = fake_model
        mock_prepare.return_value = fake_model

        load_model_and_processor(_cfg(lora_enable=False))

        mock_get_peft.assert_not_called()

    @patch(_PATCH_PEFT)
    @patch(_PATCH_PREPARE)
    @patch(_PATCH_BNB)
    @patch(_PATCH_PROC)
    @patch(_PATCH_LLAVA)
    def test_lora_enable_key_absent_defaults_to_enabled(
        self, mock_llava, mock_proc, mock_bnb, mock_prepare, mock_get_peft
    ):
        """cfg without 'enable' key should behave as enable=True."""
        fake_model = MagicMock()
        mock_llava.return_value    = fake_model
        mock_prepare.return_value  = fake_model
        mock_get_peft.return_value = fake_model

        load_model_and_processor(_cfg(include_enable_key=False))

        mock_get_peft.assert_called_once()

    @patch(_PATCH_PEFT)
    @patch(_PATCH_PREPARE)
    @patch(_PATCH_BNB)
    @patch(_PATCH_PROC)
    @patch(_PATCH_LLAVA)
    def test_returns_model_and_processor_tuple(
        self, mock_llava, mock_proc, mock_bnb, mock_prepare, mock_get_peft
    ):
        """Return value must be a 2-tuple (model, processor)."""
        fake_model     = MagicMock()
        fake_processor = MagicMock()
        mock_llava.return_value    = fake_model
        mock_prepare.return_value  = fake_model
        mock_get_peft.return_value = fake_model
        mock_proc.return_value     = fake_processor

        result = load_model_and_processor(_cfg())

        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch(_PATCH_PEFT)
    @patch(_PATCH_PREPARE)
    @patch(_PATCH_BNB)
    @patch(_PATCH_PROC)
    @patch(_PATCH_LLAVA)
    def test_processor_tokenizer_padding_side_set_to_right(
        self, mock_llava, mock_proc, mock_bnb, mock_prepare, mock_get_peft
    ):
        """The processor's tokenizer.padding_side must be set to 'right'."""
        fake_model     = MagicMock()
        fake_processor = MagicMock()
        mock_llava.return_value    = fake_model
        mock_prepare.return_value  = fake_model
        mock_get_peft.return_value = fake_model
        mock_proc.return_value     = fake_processor

        load_model_and_processor(_cfg())

        assert fake_processor.tokenizer.padding_side == "right"

    @patch(_PATCH_PEFT)
    @patch(_PATCH_PREPARE)
    @patch(_PATCH_BNB)
    @patch(_PATCH_PROC)
    @patch(_PATCH_LLAVA)
    def test_base_model_loaded_from_config_name(
        self, mock_llava, mock_proc, mock_bnb, mock_prepare, mock_get_peft
    ):
        """from_pretrained must receive the model name from the config."""
        fake_model = MagicMock()
        mock_llava.return_value    = fake_model
        mock_prepare.return_value  = fake_model
        mock_get_peft.return_value = fake_model

        cfg = _cfg()
        load_model_and_processor(cfg)

        mock_llava.assert_called_once()
        assert mock_llava.call_args[0][0] == cfg["model"]["name_or_path"]

    @patch(_PATCH_PEFT)
    @patch(_PATCH_PREPARE)
    @patch(_PATCH_BNB)
    @patch(_PATCH_PROC)
    @patch(_PATCH_LLAVA)
    def test_prepare_for_kbit_training_always_called(
        self, mock_llava, mock_proc, mock_bnb, mock_prepare, mock_get_peft
    ):
        """prepare_model_for_kbit_training is called regardless of lora.enable."""
        fake_model = MagicMock()
        mock_llava.return_value    = fake_model
        mock_prepare.return_value  = fake_model
        mock_get_peft.return_value = fake_model

        for enable in (True, False):
            mock_prepare.reset_mock()
            load_model_and_processor(_cfg(lora_enable=enable))
            mock_prepare.assert_called_once()

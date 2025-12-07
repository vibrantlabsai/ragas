import json

import pytest

from ragas.prompt.base import BasePrompt


class DummyPrompt(BasePrompt):
    async def generate(self, llm, data, temperature=None, stop=None, callbacks=[]):
        return "dummy"

    def generate_multiple(
        self, llm, data, n=1, temperature=None, stop=None, callbacks=[]
    ):
        return ["dummy"] * n


class TestBasePromptSaveLoad:
    def test_save_basic(self, tmp_path):
        prompt = DummyPrompt(name="test_prompt", language="english")
        file_path = tmp_path / "test_prompt.json"

        prompt.save(str(file_path))

        assert file_path.exists()
        with open(file_path, "r") as f:
            data = json.load(f)

        assert "ragas_version" in data
        assert data["language"] == "english"
        assert data["original_hash"] is None

    def test_save_with_language(self, tmp_path):
        prompt = DummyPrompt(name="test_prompt", language="french")
        file_path = tmp_path / "test_french.json"

        prompt.save(str(file_path))

        with open(file_path, "r") as f:
            data = json.load(f)

        assert data["language"] == "french"

    def test_save_with_hash(self, tmp_path):
        prompt = DummyPrompt(
            name="test_prompt", language="english", original_hash="test_hash"
        )
        file_path = tmp_path / "test_hash.json"

        prompt.save(str(file_path))

        with open(file_path, "r") as f:
            data = json.load(f)

        assert data["original_hash"] == "test_hash"

    def test_save_file_already_exists(self, tmp_path):
        prompt = DummyPrompt(name="test_prompt")
        file_path = tmp_path / "existing.json"

        file_path.write_text("{}")

        with pytest.raises(FileExistsError, match="already exists"):
            prompt.save(str(file_path))

    def test_load_basic(self, tmp_path):
        original = DummyPrompt(name="test_prompt", language="spanish")
        file_path = tmp_path / "test_load.json"

        original.save(str(file_path))
        loaded = DummyPrompt.load(str(file_path))

        assert loaded.language == "spanish"
        assert loaded.original_hash is None

    def test_load_with_hash(self, tmp_path):
        original = DummyPrompt(
            name="test_prompt", language="german", original_hash="hash123"
        )
        file_path = tmp_path / "test_hash_load.json"

        original.save(str(file_path))
        loaded = DummyPrompt.load(str(file_path))

        assert loaded.language == "german"
        assert loaded.original_hash == "hash123"

    def test_load_nonexistent_file(self, tmp_path):
        file_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            DummyPrompt.load(str(file_path))

    def test_round_trip(self, tmp_path):
        original = DummyPrompt(
            name="test_prompt", language="japanese", original_hash="original_hash"
        )
        file_path = tmp_path / "round_trip.json"

        original.save(str(file_path))
        loaded = DummyPrompt.load(str(file_path))

        assert loaded.language == original.language
        assert loaded.original_hash == original.original_hash

    def test_load_version_mismatch_warning(self, tmp_path, caplog):
        file_path = tmp_path / "version_test.json"

        data = {
            "ragas_version": "0.0.1",
            "language": "english",
            "original_hash": None,
        }

        with open(file_path, "w") as f:
            json.dump(data, f)

        DummyPrompt.load(str(file_path))

        assert any("incompatibilities" in record.message for record in caplog.records)

    def test_save_unicode_language(self, tmp_path):
        prompt = DummyPrompt(name="test_prompt", language="日本語")
        file_path = tmp_path / "unicode.json"

        prompt.save(str(file_path))

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["language"] == "日本語"

        loaded = DummyPrompt.load(str(file_path))
        assert loaded.language == "日本語"

    def test_load_missing_fields(self, tmp_path):
        file_path = tmp_path / "minimal.json"

        data = {
            "ragas_version": "0.3.0",
        }

        with open(file_path, "w") as f:
            json.dump(data, f)

        loaded = DummyPrompt.load(str(file_path))

        assert loaded.language == "english"
        assert loaded.original_hash is None

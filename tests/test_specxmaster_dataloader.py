import json
import zipfile

import pytest

from spectune import SpecXMasterDataLoader, SpecXMasterDataLoaderConfig


@pytest.fixture
def raw_dir(tmp_path):
    return tmp_path / "raw"


@pytest.fixture
def processed_dir(tmp_path):
    return tmp_path / "processed"


def _config(raw_dir, processed_dir, **overrides) -> SpecXMasterDataLoaderConfig:
    return SpecXMasterDataLoaderConfig(raw_dir=str(raw_dir), processed_dir=str(processed_dir), **overrides)


def _text_message(role: str, text: str, **overrides) -> dict:
    message = {
        "msg_id": f"msg-{role}-{abs(hash(text)) % 10_000}",
        "chat_id": "chat-1",
        "role": role,
        "timestamp": "2026-01-01 00:00:00",
        "content": [{"type": "text", "text": text}],
    }
    message.update(overrides)
    return message


def _conversation(conv_id: str, user_texts: list[str], *, modality: str = "nmr", **overrides) -> dict:
    """Build a conversation with alternating user/assistant turns.

    Each assistant reply also carries a non-text ``thinking`` block, mirroring
    real exports, to exercise the "only user text counts" filtering.
    """
    messages = []
    for i, text in enumerate(user_texts):
        messages.append(_text_message("user", text))
        messages.append(
            {
                "msg_id": f"reply-{conv_id}-{i}",
                "chat_id": "chat-1",
                "role": "assistant",
                "timestamp": "2026-01-01 00:00:01",
                "content": [{"type": "thinking", "text": "reasoning..."}, {"type": "text", "text": "ok"}],
            }
        )
    conversation = {
        "id": conv_id,
        "user_id": "user-1",
        "title": user_texts[0] if user_texts else "",
        "modality": modality,
        "agent": None,
        "created_at": "2026-01-01 00:00:00",
        "messages": messages,
    }
    conversation.update(overrides)
    return conversation


def _write_zip(zip_path, conversations: dict[str, dict], *, extra_files: dict[str, str] | None = None) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for folder, conversation in conversations.items():
            archive.writestr(f"{folder}/conversation.json", json.dumps(conversation, ensure_ascii=False))
            archive.writestr(f"{folder}/conversation.txt", "human-readable transcript, ignored by the loader")
        for name, content in (extra_files or {}).items():
            archive.writestr(name, content)


class TestPreprocessExtractsQueries:
    def test_one_record_per_user_turn(self, raw_dir, processed_dir):
        conv = _conversation("conv-1", ["first question", "follow-up question"])
        _write_zip(raw_dir / "logs.zip", {"conv-1_folder": conv})
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))

        summary = loader.preprocess()
        dataset = loader.load()

        assert summary["status"] == "built"
        assert summary["conversations_kept"] == 1
        assert summary["queries"] == 2
        assert len(dataset) == 2

        first, second = dataset[0], dataset[1]
        assert first["sample_id"] == "SpecXMaster:conv-1:0"
        assert first["query"] == "first question"
        assert first["modality"] == "nmr"
        assert first["provenance"]["turn_index"] == 0
        assert first["provenance"]["num_turns"] == 2
        assert first["provenance"]["conversation_id"] == "conv-1"
        assert first["provenance"]["source_zip"] == "logs.zip"

        assert second["sample_id"] == "SpecXMaster:conv-1:1"
        assert second["query"] == "follow-up question"
        assert second["provenance"]["turn_index"] == 1

    def test_modality_is_read_from_data_not_hardcoded(self, raw_dir, processed_dir):
        conv = _conversation("conv-ms", ["some MS query"], modality="ms")
        _write_zip(raw_dir / "logs.zip", {"conv-ms_folder": conv})
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))

        dataset = loader.load()

        assert dataset[0]["modality"] == "ms"

    def test_only_user_text_content_is_kept(self, raw_dir, processed_dir):
        conv = _conversation("conv-2", ["only this counts"])
        # Add a non-text content block to the user turn; it must be ignored.
        conv["messages"][0]["content"].append({"type": "image", "url": "http://example.com/spectrum.png"})
        _write_zip(raw_dir / "logs.zip", {"conv-2_folder": conv})
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))

        dataset = loader.load()

        assert len(dataset) == 1
        assert dataset[0]["query"] == "only this counts"

    def test_blank_user_text_is_skipped(self, raw_dir, processed_dir):
        conv = _conversation("conv-3", ["real query", "   "])
        _write_zip(raw_dir / "logs.zip", {"conv-3_folder": conv})
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))

        dataset = loader.load()

        assert len(dataset) == 1
        assert dataset[0]["query"] == "real query"

    def test_conversation_txt_sibling_is_ignored(self, raw_dir, processed_dir):
        # _write_zip always writes a conversation.txt sibling; this just asserts
        # its presence doesn't produce extra records or break parsing.
        conv = _conversation("conv-4", ["hello"])
        _write_zip(raw_dir / "logs.zip", {"conv-4_folder": conv})
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))

        dataset = loader.load()

        assert len(dataset) == 1


class TestDeduplicationAndRobustness:
    def test_deduplicates_conversation_across_zip_shards(self, raw_dir, processed_dir):
        conv = _conversation("conv-dup", ["duplicated across shards"])
        _write_zip(raw_dir / "logs-1.zip", {"conv-dup_folder": conv})
        _write_zip(raw_dir / "logs-2.zip", {"conv-dup_folder_again": conv})
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))

        summary = loader.preprocess()
        dataset = loader.load()

        assert summary["conversations_seen"] == 2
        assert summary["conversations_duplicate"] == 1
        assert summary["conversations_kept"] == 1
        assert len(dataset) == 1

    def test_skips_unparseable_conversation_json(self, raw_dir, processed_dir):
        good = _conversation("conv-good", ["a valid query"])
        extra = {"bad_folder/conversation.json": "{not json"}
        _write_zip(raw_dir / "logs.zip", {"good_folder": good}, extra_files=extra)
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))

        summary = loader.preprocess()
        dataset = loader.load()

        assert summary["conversations_unparseable"] == 1
        assert summary["conversations_kept"] == 1
        assert len(dataset) == 1

    def test_missing_raw_dir_or_no_zips_raises_file_not_found(self, raw_dir, processed_dir):
        raw_dir.mkdir(parents=True)
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))

        with pytest.raises(FileNotFoundError):
            loader.preprocess()


class TestPreprocessCachingAndErrors:
    def test_preprocess_reuses_cache_by_default(self, raw_dir, processed_dir):
        _write_zip(raw_dir / "logs.zip", {"conv_folder": _conversation("conv-1", ["q"])})
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))
        loader.preprocess()

        # Remove the raw zip; a cache hit should not need to read it again.
        (raw_dir / "logs.zip").unlink()
        summary = loader.preprocess()

        assert summary["status"] == "cached"

    def test_preprocess_overwrite_rebuilds(self, raw_dir, processed_dir):
        _write_zip(raw_dir / "logs.zip", {"conv_folder": _conversation("conv-1", ["q1"])})
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))
        loader.preprocess()

        _write_zip(raw_dir / "logs.zip", {"conv_folder": _conversation("conv-1", ["q1", "q2"])})
        summary = loader.preprocess(overwrite=True)

        assert summary["status"] == "built"
        assert summary["queries"] == 2

    def test_load_auto_preprocesses_when_cache_missing(self, raw_dir, processed_dir):
        _write_zip(raw_dir / "logs.zip", {"conv_folder": _conversation("conv-1", ["hi"])})
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))

        dataset = loader.load()

        assert len(dataset) == 1


class TestConvenienceMethods:
    def test_load_queries_alias(self, raw_dir, processed_dir):
        _write_zip(raw_dir / "logs.zip", {"conv_folder": _conversation("conv-1", ["hi"])})
        loader = SpecXMasterDataLoader(_config(raw_dir, processed_dir))

        dataset = loader.load_queries()

        assert len(dataset) == 1
        for sample in dataset:
            assert set(sample) == {"sample_id", "query", "modality", "provenance"}

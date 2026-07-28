import pytest

from spectune.dataloader import Dataset, InMemoryDataset, JsonlDataset, Subset, write_jsonl


def _records(n: int) -> list[dict]:
    return [{"id": i, "value": i * i} for i in range(n)]


class TestInMemoryDataset:
    def test_len_and_getitem(self):
        dataset = InMemoryDataset(_records(5))

        assert len(dataset) == 5
        assert dataset[0] == {"id": 0, "value": 0}
        assert dataset[4] == {"id": 4, "value": 16}

    def test_negative_and_out_of_range_index(self):
        dataset = InMemoryDataset(_records(3))

        assert dataset[-1] == {"id": 2, "value": 4}
        with pytest.raises(IndexError):
            dataset[3]
        with pytest.raises(IndexError):
            dataset[-4]

    def test_slice(self):
        dataset = InMemoryDataset(_records(5))

        assert dataset[1:3] == [{"id": 1, "value": 1}, {"id": 2, "value": 4}]

    def test_iterable_twice(self):
        dataset = InMemoryDataset(_records(3))

        assert list(dataset) == list(dataset)
        assert len(list(dataset)) == 3

    def test_is_a_dataset(self):
        assert isinstance(InMemoryDataset([]), Dataset)


class TestJsonlDataset:
    @pytest.fixture
    def jsonl_path(self, tmp_path):
        path = tmp_path / "records.jsonl"
        count = write_jsonl(path, _records(4))
        assert count == 4
        return path

    def test_len_and_getitem(self, jsonl_path):
        dataset = JsonlDataset(jsonl_path)

        assert len(dataset) == 4
        assert dataset[0] == {"id": 0, "value": 0}
        assert dataset[3] == {"id": 3, "value": 9}

    def test_negative_and_out_of_range_index(self, jsonl_path):
        dataset = JsonlDataset(jsonl_path)

        assert dataset[-1] == {"id": 3, "value": 9}
        with pytest.raises(IndexError):
            dataset[4]

    def test_slice(self, jsonl_path):
        dataset = JsonlDataset(jsonl_path)

        assert dataset[1:3] == [{"id": 1, "value": 1}, {"id": 2, "value": 4}]

    def test_repeatable_iteration(self, jsonl_path):
        dataset = JsonlDataset(jsonl_path)

        first_pass = list(dataset)
        second_pass = list(dataset)

        assert first_pass == second_pass == _records(4)

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "with_blanks.jsonl"
        path.write_text('{"id": 0}\n\n{"id": 1}\n\n', encoding="utf-8")

        dataset = JsonlDataset(path)

        assert len(dataset) == 2
        assert list(dataset) == [{"id": 0}, {"id": 1}]


class TestComposableViews:
    def test_take(self):
        dataset = InMemoryDataset(_records(10))

        subset = dataset.take(3)

        assert isinstance(subset, Subset)
        assert len(subset) == 3
        assert list(subset) == _records(3)

    def test_take_more_than_available(self):
        dataset = InMemoryDataset(_records(2))

        assert len(dataset.take(10)) == 2

    def test_select(self):
        dataset = InMemoryDataset(_records(5))

        subset = dataset.select([4, 0, 2])

        assert [record["id"] for record in subset] == [4, 0, 2]

    def test_shuffle_is_deterministic_with_seed(self):
        dataset = InMemoryDataset(_records(20))

        shuffled_a = list(dataset.shuffle(seed=7))
        shuffled_b = list(dataset.shuffle(seed=7))

        assert shuffled_a == shuffled_b
        assert {record["id"] for record in shuffled_a} == {record["id"] for record in dataset}

    def test_shuffle_changes_order_with_high_probability(self):
        dataset = InMemoryDataset(_records(20))

        assert [r["id"] for r in dataset.shuffle(seed=1)] != [r["id"] for r in dataset]

    def test_filter_materializes_matches(self):
        dataset = InMemoryDataset(_records(10))

        evens = dataset.filter(lambda record: record["id"] % 2 == 0)

        assert isinstance(evens, InMemoryDataset)
        assert [record["id"] for record in evens] == [0, 2, 4, 6, 8]

    def test_map_materializes_transformed_records(self):
        dataset = InMemoryDataset(_records(3))

        doubled = dataset.map(lambda record: {**record, "value": record["value"] * 2})

        assert [record["value"] for record in doubled] == [0, 2, 8]

    def test_head_and_to_list(self):
        dataset = InMemoryDataset(_records(5))

        assert dataset.head(2) == _records(2)
        assert dataset.to_list() == _records(5)

    def test_save_jsonl_round_trips(self, tmp_path):
        dataset = InMemoryDataset(_records(3))
        out_path = tmp_path / "out.jsonl"

        count = dataset.save_jsonl(out_path)

        assert count == 3
        assert list(JsonlDataset(out_path)) == _records(3)

    def test_subset_over_jsonl_dataset(self, tmp_path):
        path = tmp_path / "records.jsonl"
        write_jsonl(path, _records(6))
        dataset = JsonlDataset(path)

        subset = dataset.take(3).select([2, 0])

        assert [record["id"] for record in subset] == [2, 0]

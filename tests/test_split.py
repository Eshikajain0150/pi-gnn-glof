from pignn_glof.data.loader import split_by_region


class DummyDataset:
    def __init__(self):
        self.records = [
            {"region_id": f"region-{region:02d}", "path": f"{region}-{step}.npz"}
            for region in range(10)
            for step in range(2)
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def test_region_split_is_disjoint_and_deterministic():
    dataset = DummyDataset()
    first = split_by_region(dataset, seed=42)
    second = split_by_region(dataset, seed=42)
    assert [subset.indices for subset in first] == [subset.indices for subset in second]

    region_sets = []
    for subset in first:
        region_sets.append({dataset.records[index]["region_id"] for index in subset.indices})
    train, validation, test = region_sets
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == {f"region-{region:02d}" for region in range(10)}

from src.Dataloader.nbeDataset import NbeDataset
ds = NbeDataset("dataset/bin/toy_all_model/test", node_id=10, preload=False)
s = ds[0]
print(s["inputs"].shape, s["targets"].shape)
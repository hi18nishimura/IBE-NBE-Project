from src.Dataloader.nbeRelativeDataset import NbeDataset

dataset = NbeDataset("/workspace/dataset/bin/toy_all_model/valid", node_id=1)
print(dataset[0])
# dataset = NbeDataset("/workspace/dataset/bin/toy_all_model/train", node_id=3)
# dataset = NbeDataset("/workspace/dataset/bin/toy_all_model/train", node_id=4)
# dataset = NbeDataset("/workspace/dataset/bin/toy_all_model/train", node_id=5)
# dataset = NbeDataset("/workspace/dataset/bin/toy_all_model/valid", node_id=6)

# from src.Dataloader.nbeDataset import NbeDataset
# ds = NbeDataset(data_dir="/workspace/dataset/bin/toy_all_model/valid", node_id=1, preload=True, glob='*.feather')
# # None 入りの要素があるか
# print('has_none:', any((item is None) or (isinstance(item, dict) and (item.get('inputs') is None or item.get('targets') is None)) for item in ds._data_cache))
# # 先頭数個を表示
# for i,item in enumerate(ds._data_cache[:8]):
#     print(i, type(item), getattr(item,'keys',lambda:None)() )

dataset    = datasets.ImageFolder(root='data/NWPU-RESISC45', transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
X = torch.cat([x for x, _ in tqdm(dataloader, total=len(dataloader))], dim=0)

X.mean(axis=(0, 2, 3)) # 0.3680, 0.3810, 0.3436
X.std(axis=(0, 2, 3))  # 0.2035, 0.1854, 0.1849
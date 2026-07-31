# Phase 4 v2 Classification -- 18-Combo Comparison

All 18 (architecture, tissue_type) v2 combos (9 architectures x 2 tissue types), 12-trial Optuna search each. Sorted by overall validation F1, descending.

| Rank | Architecture | Tissue | F1 | Balanced Acc. | AUC | India AUC | Italy AUC | India/Italy AUC Gap |
|---|---|---|---|---|---|---|---|---|
| 1 | EfficientNet-B0 | forniceal_palpebral | 0.933 | 0.941 | 0.887 | 0.750 | 0.942 | 0.192 |
| 2 | ConvNeXt-Tiny | palpebral | 0.903 | 0.921 | 0.861 | 1.000 | 0.883 | 0.117 |
| 3 | ViT-L/16 | palpebral | 0.903 | 0.921 | 0.944 | 0.825 | 0.983 | 0.158 |
| 4 | EfficientNet-B0 | palpebral | 0.897 | 0.912 | 0.914 | 0.625 | 1.000 | 0.375 |
| 5 | ViT-B/16 | palpebral | 0.875 | 0.895 | 0.857 | 0.900 | 0.800 | 0.100 |
| 6 | RegNetY-400MF | forniceal_palpebral | 0.875 | 0.882 | 0.874 | 0.700 | 0.923 | 0.223 |
| 7 | RegNetY-400MF | palpebral | 0.875 | 0.895 | 0.891 | 0.675 | 0.917 | 0.242 |
| 8 | MobileNetV3-Small | palpebral | 0.839 | 0.859 | 0.857 | 0.950 | 0.967 | 0.017 |
| 9 | ViT-L/16 | forniceal_palpebral | 0.839 | 0.847 | 0.870 | 0.700 | 0.962 | 0.262 |
| 10 | DenseNet121 | forniceal_palpebral | 0.815 | 0.834 | 0.857 | 0.750 | 0.981 | 0.231 |
| 11 | Swin-Tiny | palpebral | 0.812 | 0.833 | 0.823 | 0.900 | 0.850 | 0.050 |
| 12 | DenseNet121 | palpebral | 0.812 | 0.833 | 0.797 | 0.525 | 1.000 | 0.475 |
| 13 | ResNet18 | palpebral | 0.812 | 0.833 | 0.838 | 0.650 | 0.967 | 0.317 |
| 14 | ViT-B/16 | forniceal_palpebral | 0.800 | 0.828 | 0.861 | 0.675 | 1.000 | 0.325 |
| 15 | Swin-Tiny | forniceal_palpebral | 0.800 | 0.811 | 0.811 | 0.600 | 0.981 | 0.381 |
| 16 | ConvNeXt-Tiny | forniceal_palpebral | 0.788 | 0.788 | 0.786 | 0.600 | 0.923 | 0.323 |
| 17 | MobileNetV3-Small | forniceal_palpebral | 0.778 | 0.765 | 0.815 | 0.700 | 0.865 | 0.165 |
| 18 | ResNet18 | forniceal_palpebral | 0.765 | 0.758 | 0.714 | 0.375 | 0.942 | 0.567 |

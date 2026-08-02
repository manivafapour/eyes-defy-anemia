# CNN Batch -- Model Comparison (clean data, _v2_clean)

12 combos compared, sorted by overall validation F1 (descending).

| rank | Model | F1 | Accuracy | Recall/Sensitivity | Precision | Specificity | Balanced Accuracy | AUC | India/Italy AUC Gap |
|---|---|---|---|---|---|---|---|---|---|
| 1 | convnext_tiny_palpebral_v2_clean | 0.9333 | 0.9394 | 1.0000 | 0.8750 | 0.8947 | 0.9474 | 0.9398 | 0.1000 |
| 2 | efficientnet_b0_forniceal_palpebral_v2_clean | 0.9032 | 0.9032 | 1.0000 | 0.8235 | 0.8235 | 0.9118 | 0.8824 | 0.3365 |
| 3 | regnet_y_400mf_forniceal_palpebral_v2_clean | 0.8966 | 0.9032 | 0.9286 | 0.8667 | 0.8824 | 0.9055 | 0.8739 | 0.4500 |
| 4 | regnet_y_400mf_palpebral_v2_clean | 0.8750 | 0.8788 | 1.0000 | 0.7778 | 0.7895 | 0.8947 | 0.9173 | 0.2417 |
| 5 | efficientnet_b0_palpebral_v2_clean | 0.8667 | 0.8788 | 0.9286 | 0.8125 | 0.8421 | 0.8853 | 0.9323 | 0.2167 |
| 6 | densenet121_forniceal_palpebral_v2_clean | 0.8485 | 0.8387 | 1.0000 | 0.7368 | 0.7059 | 0.8529 | 0.8782 | 0.3058 |
| 7 | resnet18_palpebral_v2_clean | 0.8387 | 0.8485 | 0.9286 | 0.7647 | 0.7895 | 0.8590 | 0.8910 | 0.2167 |
| 8 | densenet121_palpebral_v2_clean | 0.8387 | 0.8485 | 0.9286 | 0.7647 | 0.7895 | 0.8590 | 0.8872 | 0.3583 |
| 9 | convnext_tiny_forniceal_palpebral_v2_clean | 0.7778 | 0.7419 | 1.0000 | 0.6364 | 0.5294 | 0.7647 | 0.7437 | 0.2904 |
| 10 | mobilenet_v3_small_palpebral_v2_clean | 0.7742 | 0.7879 | 0.8571 | 0.7059 | 0.7368 | 0.7970 | 0.8759 | 0.1083 |
| 11 | resnet18_forniceal_palpebral_v2_clean | 0.7692 | 0.8065 | 0.7143 | 0.8333 | 0.8824 | 0.7983 | 0.7731 | 0.2750 |
| 12 | mobilenet_v3_small_forniceal_palpebral_v2_clean | 0.7568 | 0.7097 | 1.0000 | 0.6087 | 0.4706 | 0.7353 | 0.7941 | 0.0192 |

## Top performer

**convnext_tiny_palpebral_v2_clean** -- F1=0.9333, Balanced Accuracy=0.9474, AUC=0.9398, India/Italy AUC gap=0.1000.

## Best confound-handling (smallest India/Italy AUC gap)

**mobilenet_v3_small_forniceal_palpebral_v2_clean** -- gap=0.0192, F1=0.7568.


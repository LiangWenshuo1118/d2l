
## fashion_nist CNN.py

```
epoch 1, average loss 0.861013
epoch 2, average loss 0.493013
epoch 3, average loss 0.429167
epoch 4, average loss 0.395374
epoch 5, average loss 0.373267
epoch 6, average loss 0.354991
epoch 7, average loss 0.340585
epoch 8, average loss 0.329741
epoch 9, average loss 0.320100
epoch 10, average loss 0.311130
epoch 11, average loss 0.303879
epoch 12, average loss 0.298415
epoch 13, average loss 0.293386
epoch 14, average loss 0.285870
epoch 15, average loss 0.282882
epoch 16, average loss 0.279096
epoch 17, average loss 0.274544
epoch 18, average loss 0.269048
epoch 19, average loss 0.264603
epoch 20, average loss 0.262767
epoch 21, average loss 0.258248
epoch 22, average loss 0.253545
epoch 23, average loss 0.253597
epoch 24, average loss 0.247592
epoch 25, average loss 0.247737
epoch 26, average loss 0.242023
epoch 27, average loss 0.239434
epoch 28, average loss 0.239215
epoch 29, average loss 0.237451
epoch 30, average loss 0.231885
epoch 31, average loss 0.231349
epoch 32, average loss 0.230741
epoch 33, average loss 0.227524
epoch 34, average loss 0.226797
epoch 35, average loss 0.222408
epoch 36, average loss 0.220237
epoch 37, average loss 0.221013
epoch 38, average loss 0.218296
epoch 39, average loss 0.217474
epoch 40, average loss 0.213721
epoch 41, average loss 0.214820
epoch 42, average loss 0.212414
epoch 43, average loss 0.210354
epoch 44, average loss 0.208950
epoch 45, average loss 0.205563
epoch 46, average loss 0.205414
epoch 47, average loss 0.204687
epoch 48, average loss 0.202632
epoch 49, average loss 0.201667
epoch 50, average loss 0.198938
Accuracy on the test set: 89.98%
```

## fashion_nist VGG_like.py

- using mps

- using dropout (no)
```
Using mps device
Epoch 1: Train Loss 2.0595, Test Loss 1.5895, Test Accuracy 49.81%
Epoch 2: Train Loss 0.7010, Test Loss 0.7166, Test Accuracy 73.53%
Epoch 3: Train Loss 0.4825, Test Loss 0.7495, Test Accuracy 68.37%
Epoch 4: Train Loss 0.4165, Test Loss 0.4824, Test Accuracy 81.34%
Epoch 5: Train Loss 0.3746, Test Loss 0.3886, Test Accuracy 86.10%
Epoch 6: Train Loss 0.3428, Test Loss 0.3772, Test Accuracy 86.32%
Epoch 7: Train Loss 0.3222, Test Loss 0.3938, Test Accuracy 85.60%
Epoch 8: Train Loss 0.3005, Test Loss 0.3509, Test Accuracy 87.39%
Epoch 9: Train Loss 0.2853, Test Loss 0.3347, Test Accuracy 87.94%
Epoch 10: Train Loss 0.2712, Test Loss 0.3528, Test Accuracy 87.15%
Epoch 11: Train Loss 0.2594, Test Loss 0.4091, Test Accuracy 85.68%
Epoch 12: Train Loss 0.2489, Test Loss 0.3296, Test Accuracy 88.49%
Epoch 13: Train Loss 0.2386, Test Loss 0.2924, Test Accuracy 89.43%
Epoch 14: Train Loss 0.2285, Test Loss 0.2967, Test Accuracy 89.01%
Epoch 15: Train Loss 0.2203, Test Loss 0.3502, Test Accuracy 87.41%
Epoch 16: Train Loss 0.2121, Test Loss 0.2858, Test Accuracy 89.58%
Epoch 17: Train Loss 0.2055, Test Loss 0.2920, Test Accuracy 90.11%
Epoch 18: Train Loss 0.1965, Test Loss 0.2843, Test Accuracy 89.59%
Epoch 19: Train Loss 0.1897, Test Loss 0.2869, Test Accuracy 89.70%
Epoch 20: Train Loss 0.1840, Test Loss 0.2708, Test Accuracy 90.53%
Epoch 21: Train Loss 0.1770, Test Loss 0.3522, Test Accuracy 87.56%
Epoch 22: Train Loss 0.1707, Test Loss 0.3297, Test Accuracy 88.59%
Epoch 23: Train Loss 0.1645, Test Loss 0.2875, Test Accuracy 90.11%
Epoch 24: Train Loss 0.1590, Test Loss 0.2717, Test Accuracy 90.80%
Epoch 25: Train Loss 0.1500, Test Loss 0.2960, Test Accuracy 90.03%
Epoch 26: Train Loss 0.1448, Test Loss 0.3333, Test Accuracy 89.02%
Epoch 27: Train Loss 0.1393, Test Loss 0.2997, Test Accuracy 90.86%
Epoch 28: Train Loss 0.1327, Test Loss 0.3301, Test Accuracy 88.92%
Epoch 29: Train Loss 0.1266, Test Loss 0.2954, Test Accuracy 90.72%
Epoch 30: Train Loss 0.1205, Test Loss 0.3341, Test Accuracy 89.19%
Epoch 31: Train Loss 0.1158, Test Loss 0.3203, Test Accuracy 90.23%
Epoch 32: Train Loss 0.1082, Test Loss 0.3452, Test Accuracy 89.76%
Epoch 33: Train Loss 0.1060, Test Loss 0.3050, Test Accuracy 90.90%
Epoch 34: Train Loss 0.0983, Test Loss 0.3534, Test Accuracy 89.91%
Epoch 35: Train Loss 0.0963, Test Loss 0.3432, Test Accuracy 90.59%
Epoch 36: Train Loss 0.0909, Test Loss 0.4445, Test Accuracy 87.12%
Epoch 37: Train Loss 0.0849, Test Loss 0.4040, Test Accuracy 89.05%
Epoch 38: Train Loss 0.0797, Test Loss 0.4340, Test Accuracy 88.92%
Epoch 39: Train Loss 0.0758, Test Loss 0.3924, Test Accuracy 90.78%
Epoch 40: Train Loss 0.0759, Test Loss 0.6448, Test Accuracy 86.78%
Epoch 41: Train Loss 0.0675, Test Loss 0.4011, Test Accuracy 90.69%
Epoch 42: Train Loss 0.0677, Test Loss 0.3961, Test Accuracy 90.75%
Epoch 43: Train Loss 0.0563, Test Loss 0.4427, Test Accuracy 90.19%
Epoch 44: Train Loss 0.0561, Test Loss 0.5779, Test Accuracy 89.18%
Epoch 45: Train Loss 0.0527, Test Loss 0.4632, Test Accuracy 90.18%
Epoch 46: Train Loss 0.0526, Test Loss 0.4728, Test Accuracy 89.97%
Epoch 47: Train Loss 0.0442, Test Loss 0.5376, Test Accuracy 89.52%
Epoch 48: Train Loss 0.0475, Test Loss 0.4596, Test Accuracy 90.69%
Epoch 49: Train Loss 0.0393, Test Loss 0.5546, Test Accuracy 90.00%
Epoch 50: Train Loss 0.0364, Test Loss 0.5747, Test Accuracy 88.55%
```

- using dropout (yes)

```
Using mps device
Epoch 1: Train Loss 1.5869, Test Loss 0.8170, Test Accuracy 70.20%
Epoch 2: Train Loss 0.7859, Test Loss 0.5706, Test Accuracy 79.49%
Epoch 3: Train Loss 0.6165, Test Loss 0.5006, Test Accuracy 81.46%
Epoch 4: Train Loss 0.5447, Test Loss 0.4684, Test Accuracy 82.30%
Epoch 5: Train Loss 0.5035, Test Loss 0.4687, Test Accuracy 80.70%
Epoch 6: Train Loss 0.4718, Test Loss 0.3876, Test Accuracy 85.47%
Epoch 7: Train Loss 0.4460, Test Loss 0.3694, Test Accuracy 86.48%
Epoch 8: Train Loss 0.4230, Test Loss 0.3702, Test Accuracy 86.01%
Epoch 9: Train Loss 0.4055, Test Loss 0.3825, Test Accuracy 85.31%
Epoch 10: Train Loss 0.3945, Test Loss 0.3451, Test Accuracy 87.04%
Epoch 11: Train Loss 0.3792, Test Loss 0.3359, Test Accuracy 87.54%
Epoch 12: Train Loss 0.3738, Test Loss 0.3204, Test Accuracy 88.14%
Epoch 13: Train Loss 0.3604, Test Loss 0.3242, Test Accuracy 87.88%
Epoch 14: Train Loss 0.3509, Test Loss 0.2964, Test Accuracy 88.94%
Epoch 15: Train Loss 0.3431, Test Loss 0.2957, Test Accuracy 89.03%
Epoch 16: Train Loss 0.3360, Test Loss 0.2934, Test Accuracy 88.96%
Epoch 17: Train Loss 0.3273, Test Loss 0.2839, Test Accuracy 89.52%
Epoch 18: Train Loss 0.3202, Test Loss 0.3405, Test Accuracy 87.40%
Epoch 19: Train Loss 0.3139, Test Loss 0.3100, Test Accuracy 88.25%
Epoch 20: Train Loss 0.3084, Test Loss 0.2744, Test Accuracy 89.78%
Epoch 21: Train Loss 0.3017, Test Loss 0.2699, Test Accuracy 90.22%
Epoch 22: Train Loss 0.2999, Test Loss 0.2650, Test Accuracy 90.29%
Epoch 23: Train Loss 0.2935, Test Loss 0.3015, Test Accuracy 88.51%
Epoch 24: Train Loss 0.2932, Test Loss 0.2588, Test Accuracy 90.41%
Epoch 25: Train Loss 0.2858, Test Loss 0.2799, Test Accuracy 89.40%
Epoch 26: Train Loss 0.2795, Test Loss 0.2531, Test Accuracy 90.71%
Epoch 27: Train Loss 0.2792, Test Loss 0.2698, Test Accuracy 89.81%
Epoch 28: Train Loss 0.2735, Test Loss 0.2474, Test Accuracy 90.89%
Epoch 29: Train Loss 0.2709, Test Loss 0.2437, Test Accuracy 90.76%
Epoch 30: Train Loss 0.2675, Test Loss 0.2419, Test Accuracy 90.95%
Epoch 31: Train Loss 0.2642, Test Loss 0.2547, Test Accuracy 90.54%
Epoch 32: Train Loss 0.2597, Test Loss 0.2405, Test Accuracy 91.22%
Epoch 33: Train Loss 0.2586, Test Loss 0.2330, Test Accuracy 91.54%
Epoch 34: Train Loss 0.2580, Test Loss 0.2647, Test Accuracy 90.16%
Epoch 35: Train Loss 0.2542, Test Loss 0.2441, Test Accuracy 91.20%
Epoch 36: Train Loss 0.2517, Test Loss 0.2300, Test Accuracy 91.62%
Epoch 37: Train Loss 0.2465, Test Loss 0.2368, Test Accuracy 91.40%
Epoch 38: Train Loss 0.2457, Test Loss 0.2288, Test Accuracy 91.75%
Epoch 39: Train Loss 0.2418, Test Loss 0.2373, Test Accuracy 91.01%
Epoch 40: Train Loss 0.2421, Test Loss 0.2338, Test Accuracy 91.39%
Epoch 41: Train Loss 0.2378, Test Loss 0.2368, Test Accuracy 91.49%
Epoch 42: Train Loss 0.2335, Test Loss 0.2373, Test Accuracy 91.25%
Epoch 43: Train Loss 0.2355, Test Loss 0.2407, Test Accuracy 91.18%
Epoch 44: Train Loss 0.2315, Test Loss 0.2293, Test Accuracy 91.65%
Epoch 45: Train Loss 0.2303, Test Loss 0.2388, Test Accuracy 90.89%
Epoch 46: Train Loss 0.2288, Test Loss 0.2215, Test Accuracy 91.93%
Epoch 47: Train Loss 0.2255, Test Loss 0.2214, Test Accuracy 92.15%
Epoch 48: Train Loss 0.2260, Test Loss 0.2436, Test Accuracy 91.02%
Epoch 49: Train Loss 0.2255, Test Loss 0.2177, Test Accuracy 91.98%
Epoch 50: Train Loss 0.2226, Test Loss 0.2208, Test Accuracy 91.69%
```


## 实验记录

为了便于比较，首先采用了不变的learning rate = 1e - 5和batch size = 4

| 主干网络   | 分类数量 | 准确率 |
| ---------- | -------- | ------ |
| SimpleNet2 | 2        | 94.25  |
| SimpleNet2 | 3        | 87.83  |
| SimpleNet2 | 4        | 80.25  |
| SimpleNet2 | 5        | 71.84  |
| SimpleNet2 | 6        | 67.4   |
| SimpleNet2 | 7        | 65.73  |
| SimpleNet2 | 8        | 64.32  |
| SimpleNet2 | 9        | 64.11  |
| SimpleNet2 | 10       | 65.04  |
| SimpleNet2 | 11       | 65.55  |
| SimpleNet2 | 12       | 61.33  |
| SimpleNet2 | 13       | 61.46  |
| SimpleNet2 | 14       | 58.07  |
| SimpleNet2 | 15       | 58.27  |
| SimpleNet2 | 20       | 53.75  |
| SimpleNet2 | 25       | 53.68  |
| SimpleNet2 | 30       | 47.57  |
| ResNet18   | 2        |        |
| ResNet18   | 3        |        |
| ResNet18   | 4        |        |
| ResNet18   | 5        |        |
| ResNet18   | 6        |        |
| ResNet18   | 7        |        |
| ResNet18   | 8        |        |
| ResNet18   | 9        |        |
|            |          |        |
|            |          |        |
|            |          |        |
|            |          |        |

存在问题，对于resnet等网络而言，收敛过慢，同时准确率不高，因此采用了学习率衰减和增大batch size的方法重新进行训练

| 主干网络   | 分类数量 | 准确率 |
| ---------- | -------- | ------ |
| SimpleNet2 | 2        |        |
| SimpleNet2 | 3        |        |
| SimpleNet2 | 4        |        |
| SimpleNet2 | 5        |        |
| SimpleNet2 | 6        |        |
| SimpleNet2 | 7        |        |
| SimpleNet2 | 8        |        |
| SimpleNet2 | 9        |        |
| SimpleNet2 | 10       |        |
| SimpleNet2 | 11       |        |
| SimpleNet2 | 12       |        |
| SimpleNet2 | 13       |        |
| SimpleNet2 | 14       |        |
| SimpleNet2 | 15       |        |
|            |          |        |
|            |          |        |
|            |          |        |
|            |          |        |
|            |          |        |
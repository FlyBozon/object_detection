import pandas as pd
from torch.utils.tensorboard import SummaryWriter

df = pd.read_csv("colab_output2/yolo_output/training_run/results.csv")

df.columns = df.columns.str.strip()

writer = SummaryWriter("runs/yolo_tensorboard")

for epoch, row in df.iterrows():
    writer.add_scalar("mAP50", row['metrics/mAP50(B)'], epoch)
    writer.add_scalar("mAP50-95", row['metrics/mAP50-95(B)'], epoch)
    writer.add_scalar("Precision", row['metrics/precision(B)'], epoch)
    writer.add_scalar("Recall", row['metrics/recall(B)'], epoch)

writer.close()

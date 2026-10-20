import json, numpy as np
from LineFeat import LineMatchPipeline

# 1. 直接读取直线的真值坐标 (N0, 4) 与 (N1, 4) —— 外部输入，无需任何检测器
with open("data/test_samples/pair1_small_rotation/mode_a_pure_permuted/lines0.json") as f:
    lines0 = np.array(json.load(f), dtype=np.float32)

with open("data/test_samples/pair1_small_rotation/mode_a_pure_permuted/lines1.json") as f:
    lines1 = np.array(json.load(f), dtype=np.float32)

# 2. 初始化纯匹配模型
pipeline = LineMatchPipeline("checkpoints/linefeat/best_model.pth")

# 3. 传入 lines0 与 lines1，内部检测器 100% 跳过！纯推断匹配
result = pipeline.match(
    image0="data/test_samples/pair1_small_rotation/image0.png",
    image1="data/test_samples/pair1_small_rotation/image1.png",
    lines0=lines0,
    lines1=lines1,
    threshold=0.70
)

# 4. 获取预测的匹配对索引与得分
for rank, (i, j, score) in enumerate(result.matches):
    print(f"Match #{rank+1}: Line0[{i}] <--> Line1[{j}] | Score: {score:.3f}")

# 5. 保存结果连线图
result.save_visualization("outputs/pure_matching/result.jpg")
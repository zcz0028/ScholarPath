# ScholarPath 第6.3步：B1.3 保底融合输出

## 1. 当前目的

B1.2已经证明：

```text
查询改写 + Raw-anchor聚合可以增加Top100召回
但直接作为最终Top50输出时，FP增加导致Macro F1仍低于B0
```

因此第6.3步做：

```text
B1.3 = B0原始检索保底 + B1.2高置信补充
```

核心思想：

```text
先保住B0比较稳定的前排结果
再从B1.2候选池里插入少量高置信补充论文
避免keyword_core候选大量挤掉raw结果
```

## 2. 本步骤是否请求OpenAlex？

不请求。本步骤只读取已有文件：

```text
outputs/b0_openalex/predictions_top100.jsonl
outputs/b1_2_raw_anchor_q3_p40/predictions_top100.jsonl
```

然后生成融合预测结果并自动评测。

因此：

```text
API调用 = 0
Token = 0
```

## 3. 新增文件

```text
src/scholarpath/fusion/
├── __init__.py
└── fallback_fusion.py

scripts/
└── fuse_b0_b1_predictions.py

tests/
└── test_fallback_fusion.py

README_STEP6_3.md
```

## 4. 推荐先跑策略A：Raw Top45 + Supplement Top5

```powershell
python scripts/fuse_b0_b1_predictions.py --b0-dir outputs/b0_openalex --b1-dir outputs/b1_2_raw_anchor_q3_p40 --output-dir outputs/b1_3_fusion_45_5 --base-keep-top 45 --supplement-slots 5
```

含义：

```text
Top50前45个位置优先保留B0
最后5个位置从B1.2中选高置信补充
```

## 5. 再跑策略B：Raw Top40 + Supplement Top10

```powershell
python scripts/fuse_b0_b1_predictions.py --b0-dir outputs/b0_openalex --b1-dir outputs/b1_2_raw_anchor_q3_p40 --output-dir outputs/b1_3_fusion_40_10 --base-keep-top 40 --supplement-slots 10
```

## 6. 查看对比结果

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B1.2':'outputs/b1_2_raw_anchor_q3_p40','B1.3_45_5':'outputs/b1_3_fusion_45_5','B1.3_40_10':'outputs/b1_3_fusion_40_10'}; [print(name,'MacroF1=',(r:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top50_strict'])['macro']['f1'],'TP=',r['counts']['tp'],'FP=',r['counts']['fp'],'FN=',r['counts']['fn']) for name,path in runs.items()]"
```

## 7. 判断标准

B0参考：

```text
Top50 strict Macro F1 = 0.02183
TP = 31
FP = 2198
```

B1.2参考：

```text
Top50 strict Macro F1 = 0.02053
TP = 36
FP = 2356
```

B1.3通过目标：

```text
Top50 strict Macro F1 >= B0
或者
Top50 TP > B0 且 Macro F1接近B0
```

如果45_5超过B0，说明保守补充有效；如果40_10召回更高但F1下降，说明补充过多。

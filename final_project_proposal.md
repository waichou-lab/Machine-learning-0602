# 期末報告構想：基於 Criss-Cross Attention 的高效語意分割方法研究

## 一、報告主題

**基於 Criss-Cross Attention 的高效語意分割方法研究**  
（暫定，可微調）

## 二、研究動機與背景

語意分割需要對每個像素進行分類，因此必須同時捕捉局部紋理細節與全局上下文資訊。傳統 CNN 受限於局部感受野，難以直接建模長距離依賴。Non-local Network 雖然引入了自注意力機制，但其時間與記憶體複雜度為 \(O(N^2)\)（\(N\) 為像素數），難以應用於高解析度影像。

CCNet（Criss-Cross Attention Network，ICCV 2019）提出了一種 criss-cross 注意力機制，將複雜度降為 \(O(N)\)，同時透過兩次遞迴模組（RCCA）仍能捕捉全圖的上下文資訊。本報告將以此為基礎，探討其相較於 Non-local、GCNet 等方法的效率與精度權衡，並嘗試結合卷積的多尺度特性進一步優化邊緣細節。

## 三、研究問題

1. CCNet 的 criss-cross attention 在語意分割任務上，能否在維持（或提升）mIoU 的同時，顯著降低記憶體與時間成本？
2. 與 Non-local Network、GCNet（全局上下文+SENet）以及輕量化的 SENet/ECA-Net 相比，CCNet 的效率與精度權衡表現如何？
3. 能否將 CCNet 與卷積的多尺度注意力（如 CBAM 的空間部分）融合，提升小物體與邊緣區域的分割精準度？

## 四、預期貢獻

- **定量比較**：在 Cityscapes 或 ADE20K 資料集上，複現並公平比較 CCNet、Non-local、GCNet、SENet（僅通道注意力）的 mIoU、參數量、FLOPs、推論時間。
- **提出改進模組**：設計一個混合模組，結合 criss-cross 注意力與局部卷積注意力，期望在邊緣細節上獲得提升。
- **實務部署建議**：分析各方法在即時分割系統（如自動駕駛）中的可行性，給出延遲、記憶體與準確率的取捨指南。

## 五、研究方法與步驟

| 階段 | 工作內容 | 預計時程 |
|------|----------|----------|
| 1 | 環境建置（PyTorch）、下載 Cityscapes 資料集、準備基準模型（如 ResNet-101 + FCN/DeepLab） | 第 1 週 |
| 2 | 複現 CCNet、Non-local、GCNet、SENet 嵌入分割頭，訓練並記錄 mIoU、參數量、FLOPs | 第 2-3 週 |
| 3 | 設計混合注意力模組（criss-cross + 局部卷積），在驗證集上進行消融實驗 | 第 4 週 |
| 4 | 評估推論速度（GPU / CPU / 邊緣裝置模擬），繪製效率-精度曲線 | 第 5 週 |
| 5 | 撰寫報告，整理圖表與結論 | 第 6 週 |

## 六、評估指標

- **精度**：mIoU（mean Intersection over Union）
- **參數量**：Model size（M）
- **計算量**：FLOPs（G）
- **速度**：FPS（frames per second）在 NVIDIA GPU 或 NVIDIA Jetson 等邊緣裝置上的實測

## 七、潛在風險與應對

| 風險 | 應對方案 |
|------|----------|
| 訓練時間過長（高解析度影像） | 先降低輸入解析度（如 512×1024）進行初期實驗，確認趨勢後再放大 |
| 混合模組效果不如預期 | 保留原始 CCNet 作為基線，分析改進無效的原因（可能過度稀疏或訓練不穩定） |
| 缺乏邊緣裝置實測環境 | 使用 NVIDIA Jetson Nano 模擬或僅提供理論 FLOPs 比較 |

## 八、參考文獻（主要）

- Huang, Z., Wang, X., Huang, L., Huang, C., Wei, Y., & Liu, W. (2019). CCNet: Criss-cross attention for semantic segmentation. *ICCV*.
- Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local neural networks. *CVPR*.
- Cao, Y., Xu, J., Lin, S., Wei, F., & Hu, H. (2019). GCNet: Non-local networks meet squeeze-excitation networks and beyond. *ICCV Workshop*.
- Chen, L. C., et al. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *ECCV*（DeepLabV3+）.

## 九、備註

本報告將以公開程式碼為基礎，並註明原作者。若時間允許，可進一步擴充至 ADE20K 資料集或探索更輕量的線性注意力變體（如 Efficient Attention）。

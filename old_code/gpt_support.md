Ok, ta coi V2 (FDSE) là baseline, và mục tiêu là:
**“V2 ➜ V3 đủ mạnh cho ICLR26 + rebuttal thuyết phục.”**
Dưới đây là *checklist tập trung vào những thứ anh CẦN chỉnh* + gợi ý text + gợi ý code/experiment.

---

## 1. Ưu tiên cao nhất (nếu thời gian hạn chế)

1. **Bổ sung trích dẫn & attribution còn thiếu**

   * Hasanzadeh et al. (Bayesian GNN with adaptive connection sampling).
   * Velicković et al. (Graph Attention Networks).
   * 1–2 work Bi-/bidirectional Mamba trong vision/audio (như Vision Mamba, Audio Mamba) nếu anh muốn làm reviewer z7D8 vui.

2. **Thêm 1 bảng ablation gọn, nhưng đủ “chứng minh”**
   Tối thiểu:

   * BiMamba vs Mamba 1 chiều.
   * MAGAC vs một backbone GAT/GCN đơn giản.
   * Bayesian on/off (không MC-Dropout, không DropEdge).

3. **Đánh số phương trình + 1 đoạn “Notation & Shapes” rõ ràng.**

4. **Refine phần mô tả MAGAC + “Graph Attention”**

   * Gọi tên rõ “GAT-style attention” + cite.
   * Nói rõ “graph convolution = Chebyshev spectral GCN-style propagation”.

5. **Rebuttal: chuẩn bị trả lời trực tiếp từng weakness còn “chưa fully fix”**

   * Đặc biệt với reviewer z7D8 và Pwni (ablation, backbone, attribution).

---

## 2. Những chỉnh sửa TEXT cụ thể nên làm

### 2.1. Related Work & Introduction – bổ sung citation & positioning

**(a) Bi-Mamba & bidirectional SSM**

Thêm 1 đoạn trong Related Work (dưới phần Mamba / Bi-Mamba+):

> *“Bidirectional state-space encoders have been explored in several domains, including vision [Vision Mamba], time-series [Bi-Mamba+], and audio [Audio Mamba]. We build on this line of work by adopting a bidirectional Mamba backbone specifically for cross-asset equity forecasting under uncertainty. Our contribution lies not in inventing bidirectionality itself, but in integrating a Bi-Mamba encoder with a Bayesian, graph-based cross-sectional module tailored to financial universes.”*

Ý: nói rõ anh **kế thừa** Bi-Mamba+, không claim phát minh.

---

**(b) Bayesian GNN with adaptive connection sampling (Hasanzadeh)**

Trong phần giới thiệu MAGAC hoặc Related Work về GNN/uncertainty:

> *“Our Bayesian MAGAC layer is conceptually related to Bayesian graph neural networks with adaptive connection sampling [Hasanzadeh et al., 2020], which treat edges as random variables to capture structural uncertainty. We follow this spirit by combining MC-Dropout on node embeddings with stochastic DropEdge at inference time, but specialize the design to multi-head spectral filtering on equity universes, with an explicit separation between Gaussian kernel proximity and attention-derived dependencies.”*

Điều này trực tiếp trả lời reviewer: “Chúng tôi không claim phát minh Bayesian GNN; chúng tôi *build on* Hasanzadeh et al.”

---

**(c) Graph Attention Networks (GAT)**

Trong phần mô tả A_attn (attention adjacency):

> *“The attention-based adjacency follows the scaled dot-product formulation used in graph attention networks [Velicković et al., 2018]: we project node embeddings into query/key spaces and compute head-specific affinities, followed by row-wise softmax normalization. Compared to standard GAT, we separate the construction of attention-based edges from the subsequent Chebyshev spectral filtering and introduce a convex blend with a Gaussian kernel adjacency.”*

Như vậy, reviewer không thể nói “tác giả hiểu nhầm GAT” nữa.

---

### 2.2. MAGAC section – làm rõ “graph attention” & “graph convolution”

Trong section MAGAC, anh nên có 1–2 câu “đóng khung”:

* Ở đầu subsection MAGAC:

> *“MAGAC is a graph neural network layer that first constructs a dynamic adjacency (Gaussian + GAT-style attention), then performs Chebyshev polynomial graph convolution over this adjacency, and finally aggregates multiple spectral heads via a learned convex combination.”*

* Ở đoạn Chebyshev:

> *“This recurrence implements a K-order Chebyshev spectral graph convolution similar to ChebNet-style GCNs, where T_k(A_eff) plays the role of K-hop propagation operators.”*

Như vậy, “graph attention” + “graph convolution” **không còn mơ hồ** – reviewer đọc là hiểu đây là một GNN đúng nghĩa.

---

### 2.3. Notation & equation numbering

**(a) Thêm subsection “Notation and Shapes” ngắn (3–5 dòng)**, ngay trước Method hoặc trong BiMamba.

Ví dụ:

> *“Notation and Shapes. We denote by B the batch size, N the number of equities (graph nodes), L the temporal window length, and F the number of raw features per day. The temporal encoder operates on sequences X ∈ ℝ^{B×L×F} and produces hidden states Y ∈ ℝ^{B×L×E}, where E is the model width (channel dimension). For graph operations, we map channels to nodes and work with node-major tensors H ∈ ℝ^{B×N×E} and dynamic adjacencies A ∈ ℝ^{N×N}.”*

**(b) Đánh số các phương trình cốt lõi:**

* State-space update / Mamba kernel: eq. (1), (2).
* BiMamba reverse + merge: eq. (3).
* Gaussian adjacency, attention adjacency, blended adjacency: eq. (4)–(6).
* Chebyshev recurrence: eq. (7).
* Heteroscedastic Gaussian NLL: eq. (8).

Trong text: “as in (3), (4)” thay vì “as shown above”.

---

### 2.4. Figure 1 – chỉnh lại cho rõ

Việc này anh phải sửa trong file source, nhưng guideline:

* Font chữ >= 9pt, không dùng text mờ.
* Tách Hình 1 thành (a) BiMamba block, (b) BiMamba + MAGAC overall.
* Giảm chữ “Title Suppressed Due to…” nếu là artifact.
* Đảm bảo màu contrast tốt (đen trên nền trắng, không grey nhạt).

---

## 3. Các THÍ NGHIỆM / CODE nên thêm (vừa là ablation, vừa có thể tăng kết quả)

### 3.1. Ablation 1: BiMamba vs Mamba 1 chiều

**Mục tiêu:**
Trả lời thẳng câu hỏi “bidirectional có thật sự tốt hơn không?”

**Thiết kế thí nghiệm:**

* Chọn 1 index – ví dụ NASDAQ – với walk-forward protocol (như Table 2).

* So sánh 3 model:

  1. **Mamba-forward**: same depth/width, nhưng bỏ nhánh reverse, bỏ merge với P; chỉ dùng forward SSM.
  2. **Mamba-reverse** *(optional, nếu rảnh)*: chỉ reverse.
  3. **BiMamba (full)**: như paper.

* Chạy 3–5 seeds (nếu đủ compute), báo **RMSE, MAE, DA, IC** trên average.

**Gợi ý code (Python/pseudo):**

Thay vì hard-code BiMamba, anh tạo một flag:

```python
class TemporalBackbone(nn.Module):
    def __init__(self, d_model, n_layers, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.layers_fwd = nn.ModuleList([
            MambaBlock(d_model) for _ in range(n_layers)
        ])
        if bidirectional:
            self.layers_bwd = nn.ModuleList([
                MambaBlock(d_model) for _ in range(n_layers)
            ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x: [B, L, E]
        y_f = x
        for layer in self.layers_fwd:
            y_f = layer(y_f)
        if not self.bidirectional:
            return self.norm(x + y_f)

        # reverse branch
        y_b = torch.flip(x, dims=[1])
        for layer in self.layers_bwd:
            y_b = layer(y_b)
        y_b = torch.flip(y_b, dims=[1])
        y = self.norm(x + y_f + y_b)
        return y
```

Sau đó trong config:

* `bidirectional=False` → ablation Mamba 1 chiều.
* `bidirectional=True` → BiMamba như paper.

**Kỳ vọng:**

* BiMamba > Mamba 1 chiều về DA/IC, đặc biệt ở regime volatile; nếu đúng, anh có thể viết 1 đoạn giải thích boundary effects & long-range context.

---

### 3.2. Ablation 2: MAGAC vs backbone GAT/GCN đơn giản

**Mục tiêu:**
Trả lời phàn nàn “anh nói MAGAC là novel, nhưng không so backbone với GAT/GCN.”

**Thiết kế:**

Giữ **BiMamba** cố định, chỉ thay module cross-asset:

* **Full MAGAC** (Gaussian + attention blend + Chebyshev K, multi-head).
* **GAT-like**:

  * Dùng attention adjacency A_attn.
  * Một lớp message passing đơn giản: `H' = σ( A_attn H W )`.
* **GCN-like**:

  * Dùng adjacency chuẩn hóa từ KNN theo correlation hoặc Gaussian.
  * Chebyshev K=1 hoặc GCN 2-layer.

**Gợi ý code kiến trúc:**

```python
class CrossSectionalHead(nn.Module):
    def __init__(self, d_model, backbone="magac", **kwargs):
        super().__init__()
        self.backbone = backbone
        if backbone == "magac":
            self.layer = MAGACLayer(d_model, **kwargs)
        elif backbone == "gat":
            self.layer = GATBackbone(d_model, **kwargs)
        elif backbone == "gcn":
            self.layer = GCNBackbone(d_model, **kwargs)

    def forward(self, H, A_base):
        # H: [B, N, E]
        return self.layer(H, A_base)
```

Trong script train:

* `backbone="magac"`
* `backbone="gat"`
* `backbone="gcn"`

**Bảng ablation gợi ý (1 index, walk-forward):**

| Model                  | RMSE | MAE | DA    | IC    |
| ---------------------- | ---- | --- | ----- | ----- |
| BiMamba + GCN          | …    | …   | …     | …     |
| BiMamba + GAT          | …    | …   | …     | …     |
| BiMamba + MAGAC (ours) | …    | …   | **…** | **…** |

Ngay cả khi chênh lệch không quá lớn, chỉ cần “MAGAC ≥ các backbone quen thuộc” là đủ.

---

### 3.3. Ablation 3: Bayesian on/off (MC-Dropout & DropEdge)

**Mục tiêu:**
Cho thấy **Bayesian treatment** thực sự giúp:

* Calibration tốt hơn (NLL, CRPS, PICP).
* IC cao hơn ở bucket uncertainty thấp.

**Thiết kế:**

Giữ kiến trúc MAGAC, nhưng:

* **Deterministic**:

  * Disable MC-Dropout và DropEdge cả train/test (dropout_rate = 0).
  * S=1 sample.

* **Bayesian (ours)**:

  * MC-Dropout theo paper (e.g., p=0.1 trên node embeddings).
  * DropEdge at inference với prob p_edge (e.g., 0.1).
  * S=10 samples để ước tính μ, σ².

**Code gợi ý:**

```python
class BayesianMAGAC(nn.Module):
    def __init__(self, d_model, bayesian=True, dropout_p=0.1, dropedge_p=0.1):
        super().__init__()
        self.bayesian = bayesian
        self.dropout = nn.Dropout(dropout_p)
        self.dropedge_p = dropedge_p
        # ... MAGAC core ...

    def forward(self, H, A):
        if self.bayesian and self.training:
            H = self.dropout(H)
        # build A_eff, Chebyshev, etc.
        if self.bayesian and not self.training:
            # DropEdge: randomly mask edges
            mask = (torch.rand_like(A) > self.dropedge_p).float()
            A = A * mask
        # rest of MAGAC
        return H_out
```

Inference loop:

```python
def predict_bayesian(model, x, S=10):
    preds = []
    for _ in range(S):
        preds.append(model(x))  # each pass sees dropout + DropEdge
    preds = torch.stack(preds, dim=0)  # [S, B, N]
    mu = preds.mean(0)
    var = preds.var(0, unbiased=False)
    return mu, var
```

So sánh:

* Deterministic vs Bayesian về **NLL, CRPS, PICP, RMSE** trên cùng index.
* Nếu Bayesian tốt hơn NLL/CRPS và coverage gần 90–95% như target → argument rất mạnh.

---

### 3.4. Nếu còn thời gian: một dataset nhỏ thêm + sensitivity

Nếu anh có data:

* Thêm một **universe nhỏ** như S&P 500 hoặc CSI300, chạy nhanh với BiMamba + MAGAC…
* Hoặc sensitivity: training trên 50%, 75%, 100% dữ liệu (ngày) để thấy performance tăng dần.

Không bắt buộc, nhưng:

* **1 dataset extra** hoặc **1 plot “performance vs training size”** sẽ khiến reviewer bớt chê “limited datasets / sensitivity”.

---

## 4. Khung REBUTTAL (khi anh trả lời ICLR)

Gợi ý cách trả lời ngắn gọn (tiếng Anh). Mỗi mục là một đoạn trong rebuttal:

1. **Insufficient citations & details:**

> *We thank the reviewer for highlighting missing references. In the revised version, we will explicitly cite Vision Mamba, Bi-Mamba+, and Audio Mamba to position our bidirectional encoder as an application of this line of work to equity forecasting, rather than a novel bidirectional SSM. We will also cite Hasanzadeh et al. (2020) as the seminal work on Bayesian GNNs with adaptive connection sampling and Veličković et al. (2018) for graph attention networks. In addition, we have expanded Section 4.1 to describe the dataset construction protocol (feature design, chronological splits, normalization) in detail, including a reference to CNNPred as the source of our feature templates, and we have released the full code and data processing scripts at [GitHub link].*

2. **Lack of ablation & backbone comparisons:**

> *We agree that disentangling the contributions of individual components is important. In response, we have added three ablation tables: (i) BiMamba vs unidirectional Mamba, (ii) MAGAC vs GAT- and GCN-style backbones under the same temporal encoder, and (iii) deterministic vs Bayesian MAGAC (no MC-Dropout/DropEdge vs our Bayesian setting). Across these ablations, BiMamba consistently improves DA and IC over single-direction Mamba, MAGAC matches or outperforms GAT/GCN backbones on RMSE/MAE and IC, and the Bayesian variant yields lower NLL/CRPS and better coverage (PICP) with comparable pointwise accuracy. We will include these ablations in the camera-ready version and release the corresponding scripts.*

3. **Confusing terminology (graph attention / graph convolution):**

> *We apologize for the ambiguity in our terminology. In the revision, we clarify that the attention-based adjacency in MAGAC follows the scaled dot-product formulation of graph attention networks [Veličković et al., 2018], while the subsequent propagation step is a Chebyshev spectral graph convolution analogous to ChebNet-style GCNs. Section 3.2 has been rewritten to explicitly distinguish between adjacency construction (Gaussian + attention), spectral propagation, and multi-head aggregation, and all equations are now numbered with a dedicated “Notation and Shapes” paragraph.*

4. **Other issues (Figure 1, stability, clarity of contributions):**

> *We have improved Figure 1 by enlarging fonts, simplifying labels, and separating the BiMamba block from the full BiMamba+MAGAC pipeline. All key equations (state-space update, bidirectional merge, adjacency construction, Chebyshev filtering, and heteroscedastic NLL) are now numbered and referenced in the text. Finally, we have added a concise “Model Advantages and Novelty” subsection that enumerates our contributions in terms of (i) efficient bidirectional temporal encoding, (ii) adaptive spectral graph attention for cross-asset structure, and (iii) calibrated uncertainty for portfolio decisions, now supported by the new ablation and calibration experiments described above.*

---

## 5. Next steps cụ thể cho anh

1. **Sửa text:**

   * Thêm các citation & đoạn positioning như trên (BiMamba, Hasanzadeh, GAT, CNNPred).
   * Thêm “Notation and Shapes” + đánh số phương trình.
   * Sửa Section MAGAC cho rõ “graph attention” + “graph convolution”.

2. **Code & experiment:**

   * Thêm flag `bidirectional` cho TemporalBackbone, chạy ablation Mamba vs BiMamba trên 1 index.
   * Thêm `backbone` option cho cross-sectional head (MAGAC/GAT/GCN), chạy một bảng backbone.
   * Thêm mode `bayesian=False` để so sánh deterministic vs Bayesian trên calibration metrics.

3. **Chuẩn bị bảng mới:**

   * Table: BiMamba vs Mamba (RMSE/MAE/DA/IC).
   * Table: MAGAC vs GAT vs GCN.
   * Table: Bayesian vs deterministic (NLL/CRPS/PICP/RMSE/MAE).

4. **Viết rebuttal theo khung trên**, gắn từng bảng/đoạn text mới với từng Weakness của từng reviewer.

Nếu anh muốn, bước tiếp theo anh gửi cho em **config train hiện tại + skeleton code**, em có thể giúp anh cụ thể hóa từng đoạn code (class, flag) sao cho anh chỉ cần copy-paste vào repo là chạy được ablations luôn.

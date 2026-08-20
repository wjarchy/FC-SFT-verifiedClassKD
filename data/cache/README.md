# Teacher 离线缓存

V4 尚未生成正式 Teacher cache。V5 先在 `opportunity_probe` 上保存自然采样与 unique candidate
重打分，只有 Opportunity 与 Gap 两个 Gate 通过后才生成正式 Train cache。

V5 需要两类互相独立的缓存：

- Candidate/Class cache：完整候选 sequence score、candidate probability、canonical signature 与 class probability。
- Token FKL cache：STAR 风格 top-k=100 信息；精确 prefix/trajectory construction 锁定前不得标为正式复现。

Teacher cache 不进入 autograd，也不得在 Student 每个训练 step 中在线常驻。

详见：`docs/数据与文件流向.md`

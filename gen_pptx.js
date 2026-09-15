const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "YOLO26 Vehicle Detection Project";
pres.title = "基于YOLO26的车流量检测系统设计与实现";

// Colors
const NAVY = "1E2761";
const ICE = "CADCFC";
const WHITE = "FFFFFF";
const GREEN = "00FF00";
const DARK = "0D1B2A";
const MID = "1B3A5C";

// Helper: add background
function addBg(slide) {
  slide.background = { color: NAVY };
}

// Helper: add accent bar at top
function addTopBar(slide) {
  slide.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: "100%", h: 0.08, fill: { color: ICE } });
}

// Helper: slide title
function addTitle(slide, text) {
  addTopBar(slide);
  slide.addText(text, { x: 0.5, y: 0.25, w: 9, h: 0.6, fontSize: 28, fontFace: "Arial", color: ICE, bold: true });
  // underline
  slide.addShape(pres.ShapeType.rect, { x: 0.5, y: 0.85, w: 2.5, h: 0.04, fill: { color: GREEN } });
}

// Helper: card shape
function addCard(slide, x, y, w, h, opts) {
  const o = Object.assign({ x, y, w, h, fill: { color: DARK }, rectRadius: 0.1, shadow: { type: "outer", blur: 6, offset: 2, color: "000000", opacity: 0.4 } }, opts || {});
  return slide.addShape(pres.ShapeType.roundRect, o);
}

// Helper: bullet list in card
function addBullets(slide, x, y, w, h, items, opts) {
  const textRows = items.map(item => ({ text: item, options: { fontSize: 14, fontFace: "Arial", color: WHITE, bullet: { code: "25B6", color: GREEN }, paraSpaceAfter: 6 } }));
  slide.addText(textRows, { x, y, w, h, valign: "top", ...opts });
}

// ============ SLIDE 1: Title ============
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };
  // Large accent shape
  slide.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: "100%", h: 0.12, fill: { color: ICE } });
  slide.addShape(pres.ShapeType.rect, { x: 0, y: 4.38, w: "100%", h: 0.12, fill: { color: ICE } });
  // Decorative side bar
  slide.addShape(pres.ShapeType.rect, { x: 0, y: 0.12, w: 0.08, h: 4.26, fill: { color: GREEN } });

  slide.addText("基于YOLO26的车流量检测系统\n设计与实现", {
    x: 0.8, y: 0.8, w: 8.4, h: 1.8,
    fontSize: 36, fontFace: "Arial", color: WHITE, bold: true, align: "center", lineSpacingMultiple: 1.3
  });

  slide.addShape(pres.ShapeType.rect, { x: 3.5, y: 2.7, w: 3, h: 0.04, fill: { color: GREEN } });

  slide.addText("《深度学习实践》课程项目答辩", {
    x: 0.8, y: 2.95, w: 8.4, h: 0.6,
    fontSize: 22, fontFace: "Arial", color: ICE, align: "center"
  });

  slide.addText("答辩人：XXX  |  指导教师：XXX  |  2026年6月", {
    x: 0.8, y: 3.65, w: 8.4, h: 0.5,
    fontSize: 14, fontFace: "Arial", color: ICE, align: "center", italic: true
  });
}

// ============ SLIDE 2: 项目背景 ============
{
  const slide = pres.addSlide();
  addBg(slide);
  addTitle(slide, "项目背景");

  // Left card
  addCard(slide, 0.5, 1.15, 4.2, 1.7);
  slide.addText("🚗 城市交通拥堵现状", { x: 0.7, y: 1.2, w: 3.8, h: 0.4, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true });
  addBullets(slide, 0.7, 1.6, 3.8, 1.1, [
    "城市化进程加速，机动车保有量持续增长",
    "交通拥堵造成巨大经济损失与时间浪费",
    "传统人工计数方式效率低、成本高"
  ]);

  // Right card
  addCard(slide, 5.3, 1.15, 4.2, 1.7);
  slide.addText("🎯 实时检测需求", { x: 5.5, y: 1.2, w: 3.8, h: 0.4, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true });
  addBullets(slide, 5.5, 1.6, 3.8, 1.1, [
    "智能交通系统需要高精度实时车流检测",
    "YOLO系列算法在实时目标检测中表现优异",
    "YOLO26带来更低延迟与更高精度"
  ]);

  // Bottom card
  addCard(slide, 0.5, 3.1, 9, 1.3);
  slide.addText("💡 项目目标", { x: 0.7, y: 3.15, w: 8.6, h: 0.4, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true });
  addBullets(slide, 0.7, 3.5, 8.6, 0.8, [
    "基于YOLO26构建实时车流量检测系统，实现多类别车辆检测、跟踪与双向计数",
    "通过优化策略提升检测准确性，提供直观的PyQt5图形界面"
  ]);
}

// ============ SLIDE 3: YOLO26技术 ============
{
  const slide = pres.addSlide();
  addBg(slide);
  addTitle(slide, "YOLO26 技术架构");

  // Architecture card
  addCard(slide, 0.5, 1.15, 4.2, 3.2);
  slide.addText("🏗 网络架构", { x: 0.7, y: 1.2, w: 3.8, h: 0.4, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true });

  // Architecture flow boxes
  const archItems = ["C3k2 (轻量化特征提取)", "SPPF (空间金字塔池化)", "C2PSA (注意力增强)"];
  archItems.forEach((item, i) => {
    slide.addShape(pres.ShapeType.roundRect, { x: 0.8, y: 1.7 + i * 0.65, w: 3.5, h: 0.5, fill: { color: MID }, rectRadius: 0.05 });
    slide.addText(item, { x: 0.8, y: 1.7 + i * 0.65, w: 3.5, h: 0.5, fontSize: 13, fontFace: "Arial", color: WHITE, align: "center", valign: "middle" });
    if (i < 2) {
      slide.addText("▼", { x: 2.3, y: 2.15 + i * 0.65, w: 0.5, h: 0.25, fontSize: 14, color: ICE, align: "center" });
    }
  });
  slide.addText("YOLO26 Backbone + Head", { x: 0.8, y: 3.65, w: 3.5, h: 0.4, fontSize: 12, fontFace: "Arial", color: ICE, align: "center", italic: true });

  // Innovations card
  addCard(slide, 5.3, 1.15, 4.2, 3.2);
  slide.addText("🚀 核心创新", { x: 5.5, y: 1.2, w: 3.8, h: 0.4, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true });

  const innovations = [
    ["No-NMS", "无需非极大值抑制后处理"],
    ["MuSGD", "混合随机梯度下降优化器"],
    ["ProgLoss", "渐进式损失函数"],
    ["STAL", "自训练主动学习策略"]
  ];
  innovations.forEach((item, i) => {
    slide.addShape(pres.ShapeType.roundRect, { x: 5.6, y: 1.7 + i * 0.62, w: 3.7, h: 0.5, fill: { color: MID }, rectRadius: 0.05 });
    slide.addText([
      { text: item[0] + "  ", options: { fontSize: 13, fontFace: "Arial", color: GREEN, bold: true } },
      { text: item[1], options: { fontSize: 12, fontFace: "Arial", color: WHITE } }
    ], { x: 5.6, y: 1.7 + i * 0.62, w: 3.7, h: 0.5, valign: "middle" });
  });

  // Bottom stat
  addCard(slide, 0.5, 4.5, 9, 0.5);
  slide.addText("⚡ CPU推理速度提升 43%   |   边缘设备部署更友好   |   端到端检测无需后处理", {
    x: 0.7, y: 4.5, w: 8.6, h: 0.5, fontSize: 14, fontFace: "Arial", color: ICE, align: "center", valign: "middle"
  });
}

// ============ SLIDE 4: 数据集与训练 ============
{
  const slide = pres.addSlide();
  addBg(slide);
  addTitle(slide, "数据集与训练配置");

  // Dataset card
  addCard(slide, 0.5, 1.15, 4.2, 1.8);
  slide.addText("📊 UA-DETRAC 数据集", { x: 0.7, y: 1.2, w: 3.8, h: 0.4, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true });
  addBullets(slide, 0.7, 1.6, 3.8, 1.2, [
    "训练集：135,914 张图像",
    "验证集：28,256 张图像",
    "5个类别：Car, Bus, Truck, Van, Others",
    "真实交通监控场景采集"
  ]);

  // Training params card
  addCard(slide, 5.3, 1.15, 4.2, 1.8);
  slide.addText("⚙ 训练参数", { x: 5.5, y: 1.2, w: 3.8, h: 0.4, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true });

  const params = [
    ["图像尺寸", "480×480"],
    ["批大小", "32"],
    ["冻结层数", "10"],
    ["早停轮数", "6 epochs"]
  ];
  params.forEach((p, i) => {
    slide.addText([
      { text: p[0] + "：  ", options: { fontSize: 13, fontFace: "Arial", color: ICE } },
      { text: p[1], options: { fontSize: 13, fontFace: "Arial", color: WHITE, bold: true } }
    ], { x: 5.7, y: 1.65 + i * 0.35, w: 3.5, h: 0.35 });
  });

  // Dataset split visual
  addCard(slide, 0.5, 3.2, 9, 1.3);
  slide.addText("数据集划分", { x: 0.7, y: 3.25, w: 2, h: 0.35, fontSize: 14, fontFace: "Arial", color: ICE });

  // Train bar
  slide.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 3.7, w: 6.5, h: 0.35, fill: { color: "2E86AB" }, rectRadius: 0.05 });
  slide.addText("训练集 135,914 (82.8%)", { x: 0.7, y: 3.7, w: 6.5, h: 0.35, fontSize: 12, fontFace: "Arial", color: WHITE, align: "center", valign: "middle" });

  // Val bar
  slide.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 4.15, w: 1.5, h: 0.35, fill: { color: "A23B72" }, rectRadius: 0.05 });
  slide.addText("验证集 28,256", { x: 0.7, y: 4.15, w: 2.5, h: 0.35, fontSize: 12, fontFace: "Arial", color: WHITE, valign: "middle" });
}

// ============ SLIDE 5: 训练结果 ============
{
  const slide = pres.addSlide();
  addBg(slide);
  addTitle(slide, "训练结果");

  // Metrics cards - 4 columns
  const metrics = [
    { label: "mAP@50", value: "0.723", color: "2ECC71" },
    { label: "mAP@50-95", value: "0.542", color: "3498DB" },
    { label: "Precision", value: "0.762", color: "E67E22" },
    { label: "Recall", value: "0.643", color: "9B59B6" }
  ];
  metrics.forEach((m, i) => {
    const x = 0.5 + i * 2.3;
    addCard(slide, x, 1.15, 2.1, 1.5);
    slide.addShape(pres.ShapeType.rect, { x: x, y: 1.15, w: 2.1, h: 0.06, fill: { color: m.color } });
    slide.addText(m.value, { x: x, y: 1.4, w: 2.1, h: 0.6, fontSize: 30, fontFace: "Arial", color: m.color, bold: true, align: "center" });
    slide.addText(m.label, { x: x, y: 2.05, w: 2.1, h: 0.4, fontSize: 14, fontFace: "Arial", color: ICE, align: "center" });
  });

  // Training notes
  addCard(slide, 0.5, 2.9, 9, 1.6);
  slide.addText("📋 训练分析", { x: 0.7, y: 2.95, w: 8.6, h: 0.4, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true });
  addBullets(slide, 0.7, 3.35, 8.6, 1.0, [
    "最佳模型在第6个epoch取得（触发早停机制）",
    "Precision较高(0.762)说明误检较少，Recall(0.643)存在漏检改进空间",
    "mAP@50达到0.723，验证了YOLO26在车辆检测任务上的有效性",
    "使用冻结前10层策略，加速收敛并防止过拟合"
  ]);
}

// ============ SLIDE 6: 系统架构 ============
{
  const slide = pres.addSlide();
  addBg(slide);
  addTitle(slide, "系统架构");

  // Pipeline flow
  const pipeline = [
    { title: "YOLO26\n目标检测", desc: "车辆检测与分类", icon: "🔍" },
    { title: "ByteTrack\n多目标跟踪", desc: "ID分配与轨迹维护", icon: "📡" },
    { title: "计数逻辑\n处理引擎", desc: "双向计数与过滤", icon: "🔢" },
    { title: "PyQt5\n图形界面", desc: "可视化与交互", icon: "🖥" }
  ];

  pipeline.forEach((p, i) => {
    const x = 0.5 + i * 2.35;
    addCard(slide, x, 1.2, 2.1, 2.0);
    slide.addText(p.icon, { x: x, y: 1.25, w: 2.1, h: 0.5, fontSize: 28, align: "center" });
    slide.addText(p.title, { x: x, y: 1.7, w: 2.1, h: 0.7, fontSize: 14, fontFace: "Arial", color: WHITE, bold: true, align: "center", valign: "middle" });
    slide.addText(p.desc, { x: x, y: 2.4, w: 2.1, h: 0.4, fontSize: 11, fontFace: "Arial", color: ICE, align: "center" });
    if (i < 3) {
      slide.addText("▶", { x: x + 2.1, y: 1.8, w: 0.25, h: 0.5, fontSize: 18, color: GREEN, align: "center", valign: "middle" });
    }
  });

  // Key modules
  addCard(slide, 0.5, 3.5, 9, 1.2);
  slide.addText("🔧 关键技术模块", { x: 0.7, y: 3.55, w: 8.6, h: 0.4, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true });
  addBullets(slide, 0.7, 3.9, 8.6, 0.7, [
    "检测模块：支持视频文件与摄像头实时输入，置信度阈值可调",
    "跟踪模块：ByteTrack实现稳定的多目标跟踪，维护目标ID一致性",
    "计数模块：可拖拽计数线，支持双向计数，轨迹可视化"
  ]);
}

// ============ SLIDE 7: 核心功能 ============
{
  const slide = pres.addSlide();
  addBg(slide);
  addTitle(slide, "核心功能 — 5大优化策略");

  const strategies = [
    { name: "ID去重", desc: "防止同一车辆被重复计数，基于跟踪ID进行唯一性判断" },
    { name: "≥5帧过滤", desc: "仅统计出现≥5帧的目标，过滤瞬时误检" },
    { name: "区域过滤", desc: "限制检测区域，排除画面边缘干扰目标" },
    { name: "类别投票", desc: "对同一ID多次检测结果进行多数投票确定类别" },
    { name: "宽高比过滤", desc: "基于车辆宽高比特征排除非车辆目标" }
  ];

  strategies.forEach((s, i) => {
    const row = Math.floor(i / 3);
    const col = i % 3;
    const x = 0.5 + col * 3.1;
    const y = 1.15 + row * 1.65;
    const w = 2.9;
    addCard(slide, x, y, w, 1.45);
    slide.addShape(pres.ShapeType.roundRect, { x: x + 0.1, y: y + 0.1, w: 0.5, h: 0.5, fill: { color: "2E86AB" }, rectRadius: 0.25 });
    slide.addText(String(i + 1), { x: x + 0.1, y: y + 0.1, w: 0.5, h: 0.5, fontSize: 18, fontFace: "Arial", color: WHITE, bold: true, align: "center", valign: "middle" });
    slide.addText(s.name, { x: x + 0.7, y: y + 0.15, w: w - 0.9, h: 0.4, fontSize: 15, fontFace: "Arial", color: GREEN, bold: true });
    slide.addText(s.desc, { x: x + 0.2, y: y + 0.65, w: w - 0.4, h: 0.7, fontSize: 11, fontFace: "Arial", color: WHITE });
  });

  // Bottom features
  addCard(slide, 0.5, 4.4, 9, 0.6);
  slide.addText("📈 双向计数   |   🎨 轨迹可视化   |   📊 实时统计显示", {
    x: 0.7, y: 4.4, w: 8.6, h: 0.6, fontSize: 15, fontFace: "Arial", color: ICE, align: "center", valign: "middle"
  });
}

// ============ SLIDE 8: 系统演示 ============
{
  const slide = pres.addSlide();
  addBg(slide);
  addTitle(slide, "系统演示 — PyQt5 图形界面");

  // Feature cards
  const features = [
    ["🎬 多源输入", "支持视频文件与摄像头实时采集输入"],
    ["🎚 置信度滑块", "实时调节检测置信度阈值"],
    ["📏 可拖拽计数线", "自由设定车辆计数触发区域"],
    ["💾 CSV导出", "检测结果与统计数据一键导出"],
    ["🌙 暗色主题", "专业暗色界面设计，护眼舒适"],
    ["🗺 轨迹可视化", "实时绘制车辆运动轨迹路径"]
  ];

  features.forEach((f, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.5 + col * 4.7;
    const y = 1.15 + row * 1.15;
    addCard(slide, x, y, 4.5, 0.95);
    slide.addText(f[0], { x: x + 0.15, y: y + 0.05, w: 4.2, h: 0.4, fontSize: 15, fontFace: "Arial", color: GREEN, bold: true });
    slide.addText(f[1], { x: x + 0.15, y: y + 0.45, w: 4.2, h: 0.4, fontSize: 12, fontFace: "Arial", color: WHITE });
  });
}

// ============ SLIDE 9: 总结与展望 ============
{
  const slide = pres.addSlide();
  addBg(slide);
  addTitle(slide, "总结与展望");

  // Achievements
  addCard(slide, 0.5, 1.15, 4.2, 1.6);
  slide.addText("✅ 项目成果", { x: 0.7, y: 1.2, w: 3.8, h: 0.4, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true });
  addBullets(slide, 0.7, 1.6, 3.8, 1.0, [
    "成功部署YOLO26车辆检测模型",
    "集成ByteTrack实现稳定跟踪",
    "5大优化策略提升计数准确率",
    "完整的PyQt5可视化界面"
  ]);

  // Limitations
  addCard(slide, 5.3, 1.15, 4.2, 1.6);
  slide.addText("⚠ 当前局限", { x: 5.5, y: 1.2, w: 3.8, h: 0.4, fontSize: 16, fontFace: "Arial", color: "E67E22", bold: true });
  addBullets(slide, 5.5, 1.6, 3.8, 1.0, [
    "Recall值偏低，存在漏检",
    "密集遮挡场景效果下降",
    "仅支持白天场景检测",
    "单摄像头部署限制"
  ]);

  // Future work
  addCard(slide, 0.5, 3.0, 9, 1.5);
  slide.addText("🔮 未来展望", { x: 0.7, y: 3.05, w: 8.6, h: 0.4, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true });
  addBullets(slide, 0.7, 3.4, 8.6, 1.0, [
    "ONNX模型导出与边缘设备部署（TensorRT加速）",
    "夜间/低光照场景检测能力增强（红外融合）",
    "多摄像头联动与全局交通态势分析",
    "结合Transformer提升遮挡场景下的检测精度"
  ]);
}

// ============ SLIDE 10: 致谢 ============
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };
  slide.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: "100%", h: 0.12, fill: { color: ICE } });
  slide.addShape(pres.ShapeType.rect, { x: 0, y: 4.38, w: "100%", h: 0.12, fill: { color: ICE } });
  slide.addShape(pres.ShapeType.rect, { x: 0, y: 0.12, w: 0.08, h: 4.26, fill: { color: GREEN } });

  slide.addText("感谢聆听", {
    x: 0.8, y: 1.2, w: 8.4, h: 1.0,
    fontSize: 44, fontFace: "Arial", color: WHITE, bold: true, align: "center"
  });

  slide.addShape(pres.ShapeType.rect, { x: 3.5, y: 2.3, w: 3, h: 0.04, fill: { color: GREEN } });

  slide.addText("欢迎各位老师批评指正", {
    x: 0.8, y: 2.6, w: 8.4, h: 0.6,
    fontSize: 22, fontFace: "Arial", color: ICE, align: "center"
  });

  slide.addText("Q & A", {
    x: 0.8, y: 3.4, w: 8.4, h: 0.8,
    fontSize: 36, fontFace: "Arial", color: GREEN, bold: true, align: "center"
  });
}

// Generate
const outPath = "/mnt/e/dev/yjwlYOLO/dev/答辩PPT.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("DONE: " + outPath);
}).catch(err => {
  console.error("ERROR:", err);
});

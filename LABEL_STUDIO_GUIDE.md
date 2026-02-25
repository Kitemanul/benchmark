# Label Studio 标注界面使用指南

## 1. 基本界面布局

```
┌─────────────────────────────────────────────────────────┐
│  Label Studio                            [User] [Help]  │
├─────────────────────────────────────────────────────────┤
│  Project: Athlete Segmentation           [Settings]     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │  任务列表        │  │     图像显示区域              │ │
│  │  ☐ Task 1       │  │                             │ │
│  │  ☐ Task 2       │  │     [图像]                  │ │
│  │  ☐ Task 3       │  │                             │ │
│  │  ...            │  │                             │ │
│  └─────────────────┘  └─────────────────────────────┘ │
│                       ┌─────────────────────────────┐ │
│                       │   标注工具栏                 │ │
│                       │   [Rectangle] [Polygon]     │ │
│                       └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 切换图片的方法

### 方法 1：使用键盘快捷键（推荐）

| 快捷键 | 功能 |
|--------|------|
| `Alt/Option + →` | 下一张图片 |
| `Alt/Option + ←` | 上一张图片 |
| `Ctrl/Cmd + Enter` | 提交当前标注并跳到下一张 |
| `Ctrl/Cmd + ←` | 上一张（无需提交） |
| `Ctrl/Cmd + →` | 下一张（无需提交） |

### 方法 2：点击导航按钮

界面顶部或底部有导航栏：

```
[<] 上一张  |  Task 5/512  |  下一张 [>]  |  [Submit]
```

- **< 按钮**：返回上一张
- **> 按钮**：跳到下一张
- **Task 数字**：点击可跳转到指定任务

### 方法 3：从任务列表选择

在左侧边栏（Data Manager）：
1. 点击任意任务行
2. 会自动加载该图像到标注界面

### 方法 4：批量标注模式

启用 **Quick View** 模式：
1. 在 Data Manager 勾选多个任务
2. 点击 **Label All Tasks**
3. 会自动连续加载，标完一张自动跳下一张

---

## 3. 标注工具使用

### Polygon 标注（分割 mask）

**基本操作**：

1. **选择工具**：点击左侧 `Polygon` 工具或按快捷键 `P`

2. **开始绘制**：
   - 在图像上点击设置第一个顶点
   - 继续点击添加更多顶点
   - 双击或点击起点完成闭合

3. **编辑顶点**：
   - **移动顶点**：点击拖动现有顶点
   - **添加顶点**：双击边缘
   - **删除顶点**：选中顶点后按 `Delete`

4. **完成标注**：
   - 按 `Enter` 完成当前多边形
   - 或点击起点闭合

**快捷键**：

| 快捷键 | 功能 |
|--------|------|
| `P` | 选择 Polygon 工具 |
| `Enter` | 完成当前多边形 |
| `Esc` | 取消当前绘制 |
| `Delete` / `Backspace` | 删除选中的标注 |
| `Ctrl/Cmd + Z` | 撤销 |
| `Ctrl/Cmd + Shift + Z` | 重做 |

### Brush 标注（画笔涂抹）

1. **选择工具**：点击 `Brush` 工具或按 `B`
2. **调整笔刷大小**：使用滑块或 `[` 减小、`]` 增大
3. **涂抹**：按住鼠标左键拖动
4. **擦除**：按住 `Alt` 拖动或切换到 `Eraser`

### Rectangle 标注（矩形框）

1. **选择工具**：点击 `Rectangle` 或按 `R`
2. **绘制**：点击并拖动创建矩形
3. **调整**：拖动边缘或角点调整大小

---

## 4. 视图控制

### 缩放和平移

| 操作 | 快捷键/鼠标 |
|------|------------|
| **放大** | `+` 或鼠标滚轮向上 |
| **缩小** | `-` 或鼠标滚轮向下 |
| **平移** | 按住 `Space` + 拖动，或鼠标中键拖动 |
| **适应窗口** | `Ctrl/Cmd + 0` |
| **100% 原始大小** | `Ctrl/Cmd + 1` |

### 图像调整

- **亮度/对比度**：工具栏 → Settings → Brightness/Contrast
- **显示网格**：Settings → Show Grid
- **显示所有标注**：Toggle All Regions（眼睛图标）

---

## 5. 标注管理

### 查看标注列表

右侧面板显示当前图像的所有标注：

```
Regions (3)
├─ person #1 [Edit] [Delete]
├─ person #2 [Edit] [Delete]
└─ person #3 [Edit] [Delete]
```

- **选择标注**：点击列表项，对应的 mask 会高亮
- **编辑标签**：点击 [Edit] 修改类别
- **删除标注**：点击 [Delete] 或选中后按 `Delete`

### 复制/粘贴标注

- `Ctrl/Cmd + C`：复制选中的标注
- `Ctrl/Cmd + V`：粘贴到当前图像

### 隐藏/显示标注

- **单个标注**：点击列表中的眼睛图标
- **所有标注**：按 `H` 键切换显示/隐藏

---

## 6. 标注状态

每个任务有不同的状态：

| 状态 | 说明 | 图标颜色 |
|------|------|---------|
| **To Do** | 未开始标注 | 灰色 |
| **In Progress** | 正在标注中 | 黄色 |
| **Completed** | 已完成 | 绿色 |
| **Skipped** | 跳过 | 蓝色 |

**操作**：

- **提交标注**：点击 `Submit` → 标记为 Completed
- **跳过任务**：点击 `Skip` → 标记为 Skipped
- **更新标注**：点击 `Update` → 保存修改但保持当前状态

---

## 7. 过滤和排序任务

在 Data Manager（任务列表）中：

### 过滤

点击顶部的 **Filters** 按钮：

```
Filters
├─ Annotations
│  ├─ Completed only
│  ├─ Skipped only
│  └─ Not started
├─ Predictions
│  └─ Has predictions
└─ Custom filters
```

### 排序

点击列标题排序：
- **ID**：按任务 ID
- **Created At**：按创建时间
- **Updated At**：按更新时间
- **Annotations**：按标注数量

---

## 8. 批量操作

### 批量标注

1. 在 Data Manager 勾选多个任务
2. 点击 **Actions** 下拉菜单：
   ```
   Actions
   ├─ Label All Tasks       # 批量标注
   ├─ Delete Tasks          # 删除任务
   ├─ Update Annotations    # 更新标注
   └─ Export                # 导出
   ```

### 批量删除

1. 勾选要删除的任务
2. Actions → Delete Tasks
3. 确认删除

---

## 9. 自动标注（与 ML Backend 集成）

如果连接了 ML Backend（如 SAM）：

### 启用自动标注

1. **打开标注界面**
2. **点击底部的 Auto-Annotation 开关**
   ```
   [✓] Auto-annotation
   ```
3. **输入 prompt**（如果是文本驱动的模型）：
   ```
   Prompt: football player
   [Add]
   ```
4. **等待预测**：自动生成 mask
5. **手动调整**：修正错误的边界
6. **提交**：保存修正后的标注

### 预标注工作流

```
原始图像
   │
   ▼
ML 模型预测（自动）
   │
   ▼
显示预测结果（半透明）
   │
   ▼
人工检查和修正
   │
   ▼
提交最终标注
```

---

## 10. 常见工作流

### 工作流 A：从头标注

```
1. 打开任务 → 2. 选择工具 → 3. 绘制标注
      ↓              ↓              ↓
   加载图像      Polygon/Brush   添加顶点
      ↓              ↓              ↓
   提交标注  ←  完成绘制  ←  设置类别
      ↓
   下一张图像
```

### 工作流 B：检查自动标注

```
1. 打开任务 → 2. 启用 Auto-annotation → 3. 等待预测
      ↓                  ↓                      ↓
   加载图像          输入 prompt             显示 mask
      ↓                  ↓                      ↓
4. 检查质量 → 5. 修正错误 → 6. 提交
      ↓                  ↓                      ↓
   是否准确         调整顶点            标记为完成
      ↓                  ↓                      ↓
   下一张  ←  提交  ←  保存
```

### 工作流 C：批量审核

```
1. Data Manager → 2. 过滤 Completed → 3. Label All Tasks
        ↓                  ↓                    ↓
   查看所有任务      只看已标注的         批量审核模式
        ↓                  ↓                    ↓
4. 快速检查  → 5. 发现错误就修正 → 6. Update
        ↓                  ↓                    ↓
   浏览标注          编辑顶点             保存修改
        ↓                  ↓                    ↓
   下一张  ←  完成  ←  继续
```

---

## 11. 快捷键速查表

### 导航

| 快捷键 | 功能 |
|--------|------|
| `Ctrl/Cmd + →` | 下一张图片 |
| `Ctrl/Cmd + ←` | 上一张图片 |
| `Ctrl/Cmd + Enter` | 提交并跳到下一张 |

### 工具

| 快捷键 | 功能 |
|--------|------|
| `P` | Polygon 工具 |
| `B` | Brush 工具 |
| `R` | Rectangle 工具 |
| `H` | 隐藏/显示所有标注 |
| `[` / `]` | 调整笔刷大小 |

### 编辑

| 快捷键 | 功能 |
|--------|------|
| `Delete` / `Backspace` | 删除选中标注 |
| `Ctrl/Cmd + Z` | 撤销 |
| `Ctrl/Cmd + Shift + Z` | 重做 |
| `Ctrl/Cmd + C` | 复制标注 |
| `Ctrl/Cmd + V` | 粘贴标注 |
| `Enter` | 完成当前多边形 |
| `Esc` | 取消当前操作 |

### 视图

| 快捷键 | 功能 |
|--------|------|
| `+` / `-` | 放大/缩小 |
| `Ctrl/Cmd + 0` | 适应窗口 |
| `Ctrl/Cmd + 1` | 100% 原始大小 |
| `Space + 拖动` | 平移视图 |

---

## 12. 提示和最佳实践

### 提高效率

1. **熟练使用快捷键**：比鼠标点击快 3-5 倍
2. **启用自动标注**：先让 AI 标，再手动修正
3. **批量标注模式**：连续标注不中断
4. **合理设置顶点**：20-40 个顶点平衡精度和速度

### 质量控制

1. **定期休息**：每标注 30 分钟休息 5 分钟
2. **放大检查细节**：边缘区域放大到 200-400%
3. **一致性检查**：同一类对象保持相似的标注风格
4. **使用网格辅助**：Settings → Show Grid

### 避免常见错误

1. **不要遗漏小目标**：放大检查是否有遗漏
2. **检查遮挡边界**：遮挡部分用直线闭合
3. **顶点不要太密集**：除非必要，避免过度细节
4. **及时保存**：定期点击 Update 保存进度

---

## 13. 常见问题

### Q: 标注丢失了怎么办？

**A**: Label Studio 有自动保存，但建议：
- 定期点击 `Update` 手动保存
- 重要标注完成后立即 `Submit`
- 检查浏览器是否允许自动保存

### Q: 图片加载很慢？

**A**: 可能的原因：
1. 图片太大 → 预先 resize 到合适尺寸
2. 网络问题 → 检查本地文件服务是否启用
3. 浏览器缓存 → 清除缓存重试

### Q: 快捷键不起作用？

**A**: 检查：
1. 输入框是否获得焦点 → 点击图像区域
2. 浏览器扩展冲突 → 暂时禁用扩展
3. 操作系统快捷键冲突 → 自定义 Label Studio 快捷键

### Q: 如何批量修改标签？

**A**:
1. Data Manager 勾选任务
2. Actions → Update Annotations
3. 选择要修改的标签
4. 批量更新

---

## 14. 进阶功能

### 自定义标注配置

修改 Labeling Interface 的 XML 配置：

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>

  <!-- Polygon 标注 -->
  <PolygonLabels name="label" toName="image" strokeWidth="3">
    <Label value="athlete" background="red"/>
    <Label value="referee" background="blue"/>
  </PolygonLabels>

  <!-- Brush 标注 -->
  <BrushLabels name="brush" toName="image">
    <Label value="athlete" background="red"/>
  </BrushLabels>
</View>
```

### 添加元数据字段

```xml
<View>
  <Image name="image" value="$image"/>

  <!-- 标注工具 -->
  <PolygonLabels name="label" toName="image">
    <Label value="athlete"/>
  </PolygonLabels>

  <!-- 元数据输入 -->
  <Choices name="difficulty" toName="image" choice="single">
    <Choice value="easy"/>
    <Choice value="medium"/>
    <Choice value="hard"/>
  </Choices>

  <TextArea name="notes" toName="image" placeholder="备注..."/>
</View>
```

### 协作标注

1. **Settings** → **Members**
2. 添加团队成员邮箱
3. 分配角色：
   - **Annotator**：只能标注
   - **Reviewer**：可以审核
   - **Manager**：完全权限

---

## 15. 导出标注

### 导出格式

1. 项目页面点击 **Export**
2. 选择格式：
   - **COCO** - instance segmentation（推荐）
   - **JSON** - Label Studio 原始格式
   - **CSV** - 表格格式
   - **YOLO** - YOLOv8 格式

### COCO 导出示例

```bash
# 导出后的文件结构
project-1-export.json
├─ images: [...]
├─ annotations: [...]
├─ categories: [
     {"id": 1, "name": "athlete"}
   ]
```

---

## 参考资源

- [Label Studio 官方文档](https://labelstud.io/guide/)
- [快捷键完整列表](https://labelstud.io/guide/keyboard_shortcuts.html)
- [标注配置模板](https://labelstud.io/templates/)
- [ML Backend 集成](https://labelstud.io/guide/ml.html)

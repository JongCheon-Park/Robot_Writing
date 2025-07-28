
# 🖋️ Hangeul Stroke Annotator & StrokeNet

A comprehensive system for collecting, visualizing, and learning Hangeul character stroke data using Python GUI tools and deep learning.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `main_2.py` | GUI tool to collect strokes manually from rendered Hangeul characters |
| `test.py` | Visualizer for displaying stroke data from `progress.json` |
| `test_code.py` | Inference and visualization using a trained StrokeNet model |
| `train.py` | Full training pipeline for stroke classification and order regression |

---

## 🧩 Features

### 🖼️ 1. Stroke Collection GUI (`main_2.py`)

A `Tkinter`-based GUI tool to manually annotate strokes of Hangeul characters.

- Automatically renders characters using a specified TTF font
- Allows mouse-based stroke drawing and point collection
- Interpolates & filters points using an external processor
- Computes stroke thickness via distance transform
- Saves structured results in `progress.json`

**Run the tool**:

```bash
python main_2.py
```

---

### 🧪 2. Stroke Visualizer (`test.py`)

A `matplotlib` tool to inspect the collected stroke data interactively.

- Loads character data from `progress.json`
- Renders strokes with thickness-based point sizes
- Displays stroke sequence numbers, start/end markers
- Use arrow keys or GUI buttons to navigate characters
- Press `s` to save the current plot as PNG

**Run the visualizer**:

```bash
python test.py
```

---

### 🧠 3. Inference with StrokeNet (`test_code.py`)

Run predictions using a trained `StrokeNet` model.

- Loads saved PyTorch weights
- Predicts stroke labels and point order for a given character
- Displays predicted strokes with color-coded order

**Example run**:

```bash
python test_code.py
```

> The script loads example data for the character `'갇'` from `progress.json`.

---

### 🏋️ 4. Training Pipeline (`train.py`)

Train the `StrokeNet` model for two tasks:
- **Stroke classification** (multi-class)
- **Order regression** (continuous)

**Model Overview**:
- Image Encoder: CNN → Linear
- Point Encoder: MLP
- Fusion Layer: Concatenates image and point features
- Outputs: Stroke logits & order values

**Training highlights**:
- Custom `StrokeDataset` using collected JSON
- Dynamic padding via `collate_fn`
- Saves models every 1000 epochs
- Visualizes sample predictions periodically

**Run training**:

```bash
python train.py
```

Models are saved to:
```
output/weights/000X/strokenet_epochXXXX.pt
```

---

## 🧷 Example Data Format: `progress.json`

```json
{
  "characters": {
    "갇": {
      "image_path": "img/U+AC07.png",
      "points": [[x1, y1], [x2, y2], ...],
      "strokes": [[0,1,2], [3,4]],
      "stroke_labels": [0,0,0,1,1],
      "thicknesses": [1.2, 1.3, ...],
      "sequence": [...],
      ...
    }
  }
}
```

---

## 🔧 Setup

```bash
pip install torch torchvision matplotlib opencv-python pillow
```

> For GPU acceleration, install PyTorch with CUDA support.

---

## 🧠 Model: StrokeNet

A lightweight neural network to jointly predict stroke clusters and point order.

- **Inputs**: image (resized), normalized 2D points
- **Outputs**:
  - Stroke label (integer)
  - Relative stroke order (float ∈ [0,1])

---

## 🚀 Future Improvements

- Integrate handwriting data input
- Expand font support
- Improve filtering algorithm for point selection
- Add attention-based encoder for better generalization

---

## 📄 License

MIT License

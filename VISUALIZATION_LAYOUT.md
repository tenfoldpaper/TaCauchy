# Visualization Layout Guide

## 📊 New Improved Layout

The visualization `trial_sim_comparison_full.png` now follows a clean, professional layout similar to the reference image.

### Layout Structure (7 rows × 3 columns)

```
┌─────────────────────────────────────────────────────────┐
│  Row 0: Radar Chart (spanning all 3 columns)           │
│         - Overall similarity metrics comparison         │
│         - All 6 trials overlaid                        │
├─────────────────────────────────────────────────────────┤
│  Row 1: Bar Chart (spanning all 3 columns)             │
│         - Detailed metrics comparison                   │
│         - SSIM, NCC, Hist Corr, Combined               │
├───────────────┬───────────────┬───────────────────────┤
│  Row 2, Col 0 │  Row 2, Col 1 │  Row 2, Col 2        │
│  MSE Chart    │  PSNR Chart   │  Pixel Diff Stats    │
├───────────────┼───────────────┼───────────────────────┤
│  Row 3: Trial 1 Comparison                            │
│  Real Image   │  Sim Image    │  Difference Heatmap  │
├───────────────┼───────────────┼───────────────────────┤
│  Row 4: Trial 2 Comparison                            │
│  Real Image   │  Sim Image    │  Difference Heatmap  │
├───────────────┼───────────────┼───────────────────────┤
│  Row 5: Trial 3 Comparison                            │
│  Real Image   │  Sim Image    │  Difference Heatmap  │
├───────────────┼───────────────┼───────────────────────┤
│  Row 6: Trials 4, 5, 6 (Compact View)                │
│  Trial 4      │  Trial 5      │  Trial 6             │
│  (3-in-1)     │  (3-in-1)     │  (3-in-1)            │
└───────────────┴───────────────┴───────────────────────┘
```

### Key Improvements

1. **Top Section (Rows 0-2)**: Summary Statistics
   - Radar chart for overall comparison
   - Bar chart for detailed metrics
   - Three focused metric charts (MSE, PSNR, Pixel Diff)

2. **Middle Section (Rows 3-5)**: Detailed Comparisons
   - First 3 trials get full-size image comparisons
   - Each trial shows: Real | Simulation | Difference Heatmap
   - Clear visual comparison with color-coded heatmaps

3. **Bottom Section (Row 6)**: Compact View
   - Last 3 trials in compact format
   - Each shows Real | Sim | Diff side-by-side
   - Saves space while maintaining clarity

### Visual Elements

- **Color Scheme**: Consistent color palette across all charts
- **Labels**: All in English (no font issues)
- **Grid Lines**: Subtle grid for better readability
- **Legends**: Clear and positioned appropriately
- **Titles**: Bold and descriptive

### Comparison with Previous Layout

**Old Layout Issues:**
- ❌ Too many rows (18 rows)
- ❌ Repetitive information
- ❌ Top 5 matches took too much space
- ❌ Hard to see overall patterns

**New Layout Benefits:**
- ✅ Compact (7 rows vs 18 rows)
- ✅ Clear hierarchy (summary → details)
- ✅ Better use of space
- ✅ Easier to compare across trials
- ✅ Professional appearance

### File Information

- **Filename**: `trial_sim_comparison_full.png`
- **Size**: 2.0 MB
- **Dimensions**: 20" × 24" (3000 × 3600 pixels at 150 DPI)
- **Format**: PNG with transparency support

### How to Read the Visualization

1. **Start with Radar Chart**: Get overall sense of similarity across all trials
2. **Check Bar Chart**: See which metrics are strongest/weakest
3. **Review Metric Charts**: Understand MSE, PSNR, and pixel differences
4. **Examine Image Comparisons**: Visual inspection of matches
5. **Look at Heatmaps**: Identify where differences occur

### Metrics Displayed

- **SSIM**: Structural Similarity Index (0-1, higher is better)
- **NCC**: Normalized Cross-Correlation (0-1, higher is better)
- **Hist Corr**: Histogram Correlation (0-1, higher is better)
- **Combined**: Weighted average of all metrics
- **MSE**: Mean Squared Error (lower is better)
- **PSNR**: Peak Signal-to-Noise Ratio (dB, higher is better)
- **MAE**: Mean Absolute Error (lower is better)

### Color Coding

- **Heatmaps**: Hot colormap (dark red = high difference, yellow = low difference)
- **Bar Charts**: Distinct colors for each metric
- **Trial Colors**: Consistent across all charts for easy tracking

---

*This layout provides a comprehensive yet compact view of all matching results.*

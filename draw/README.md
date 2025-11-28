# Professional Architecture Diagrams

This directory contains professional model architecture diagrams for Enhanced ReferSAM, designed in the style of top-tier computer vision conferences (CVPR, ICCV, NeurIPS, ECCV).

## Generated Diagrams

All diagrams are saved in `draw/img/`:

1. **overall_architecture.png** - Complete Enhanced ReferSAM architecture
2. **multiscale_fusion.png** - MultiScale Fusion module details
3. **c1c2_fusion.png** - Enhanced C1C2 Fusion module details
4. **text_attention.png** - Text Attention Aggregation module details
5. **enhanced_loss.png** - Enhanced Loss Functions module details

## Design Features

### Professional Styling
- **Color Scheme**: Inspired by top-tier CV papers (Material Design palette)
- **Typography**: Sans-serif fonts, clear hierarchy
- **Layout**: Clean, well-spaced, professional appearance
- **Resolution**: 300 DPI for publication quality

### Visual Elements
- **Boxes**: Rounded corners, subtle shadows, clear borders
- **Arrows**: Professional arrow style, proper flow direction
- **Highlights**: Red borders for enhanced/new modules
- **Annotations**: Clear labels with proper typography

## Usage

### For Papers

1. **Main Architecture Figure** (overall_architecture.png)
   - Use as Figure 1 or Figure 2 in your paper
   - Caption: "Overall architecture of Enhanced ReferSAM"

2. **Module Details** (multiscale_fusion.png, c1c2_fusion.png, etc.)
   - Use as subfigures in detailed method section
   - Can be combined into a single figure with subfigures

### For Presentations

- All diagrams are high-resolution (300 DPI)
- Suitable for slides and posters
- Can be scaled without quality loss

## Alternative: TikZ/LaTeX Version

For LaTeX users, a TikZ version is available: `tikz_architecture.tex`

### Usage in LaTeX:

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning,calc,decorations.pathreplacing}

% Include the TikZ code
\input{draw/tikz_architecture.tex}
```

### Advantages of TikZ:
- Vector graphics (infinite scalability)
- Perfect integration with LaTeX
- Easy to modify colors and styles
- Professional typography matching your paper

## Customization

### Python Script

Edit `draw_professional_architecture.py` to customize:

```python
# Change colors
COLORS = {
    'input': '#E3F2FD',      # Modify as needed
    # ...
}

# Change figure size
fig, ax = setup_figure(18, 11)  # width, height

# Change font sizes
fontsize=10, fontweight='bold'
```

Then run:
```bash
python draw/draw_professional_architecture.py
```

### TikZ Version

Edit `tikz_architecture.tex` to customize:

```latex
% Change colors
\definecolor{inputcolor}{RGB}{227,242,253}  % Modify RGB values

% Change styles
\tikzset{
    module/.style={
        rounded corners=3pt,  % Adjust corner radius
        minimum width=2.5cm,   % Adjust size
        % ...
    }
}
```

Then compile:
```bash
pdflatex tikz_architecture.tex
```

## Comparison with Previous Version

### Improvements:
1. **Professional Color Scheme**: Material Design inspired palette
2. **Better Typography**: Clear font hierarchy, proper sizing
3. **Cleaner Layout**: Better spacing, alignment
4. **Enhanced Visibility**: Text with stroke effects for better readability
5. **Consistent Styling**: Unified design language across all diagrams

### Style Reference:
- CVPR/ICCV/NeurIPS paper figures
- Material Design principles
- Professional academic publications

## File Structure

```
draw/
├── README.md                          # This file
├── draw_professional_architecture.py  # Python script for PNG generation
├── tikz_architecture.tex               # TikZ/LaTeX code for vector graphics
└── img/
    ├── overall_architecture.png
    ├── multiscale_fusion.png
    ├── c1c2_fusion.png
    ├── text_attention.png
    └── enhanced_loss.png
```

## Tips for Best Results

1. **For Papers**: Use TikZ version if writing in LaTeX (vector graphics)
2. **For Presentations**: PNG version is sufficient (300 DPI)
3. **For Posters**: Consider larger figure sizes
4. **Color Adjustments**: Match your paper's color scheme if needed
5. **Font Consistency**: Ensure fonts match your paper's style

## Dependencies

### Python Version:
- matplotlib
- numpy

### TikZ Version:
- LaTeX distribution (TeX Live, MiKTeX, etc.)
- TikZ package (usually included)

## License

These diagrams are generated for your research paper. Feel free to modify and use as needed.

## Contact

For questions or customization requests, refer to:
- Model code: `model/vit_adapter/` and `model/enhanced_criterion.py`
- Improvement analysis: `MODEL_IMPROVEMENT_ANALYSIS.md`


# MMSB-VI Experiments Directory Structure

## Directory Organization

### `core_experiments/`
Contains pure experimental code without visualization:
- `geometric_limits_validation.py` - Ultra-rigorous geometric limits validation
- `parameter_sensitivity_analysis.py` - Ultra-rigorous parameter sensitivity analysis
- Basic validation experiments and test files

### `visualization/`
Contains visualization modules for generating publication-quality figures:
- `geometric_limits_visualization.py` - Visualization for geometric limits validation
- `parameter_sensitivity_visualization.py` - Visualization for parameter sensitivity analysis

### `results/`
Contains experimental results and generated figures:
- `*.pkl` - Pickled experimental results data
- `*.png` - Publication-quality figures

### `logs/`
Contains execution logs and documentation:
- `*.log` - Detailed execution logs
- `*.md` - Experimental documentation

## Usage

### Running Experiments
```bash
# Run geometric limits validation
cd core_experiments/
python geometric_limits_validation.py

# Run parameter sensitivity analysis
cd core_experiments/
python parameter_sensitivity_analysis.py
```

### Generating Visualizations
```bash
# Generate geometric limits figures
cd visualization/
python geometric_limits_visualization.py

# Generate parameter sensitivity figures
cd visualization/
python parameter_sensitivity_visualization.py
```

## Features

- **Clean Separation**: Experiments are separate from visualization
- **Organized Results**: All outputs go to dedicated folders
- **Publication Ready**: High-quality figures with bilingual annotations
- **Ultra-Rigorous**: Comprehensive statistical validation and error analysis
- **Reproducible**: Consistent random seeds and logging
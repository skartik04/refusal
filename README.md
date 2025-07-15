# 🧠 Refusal Intervention LLM Project

A research project exploring intervention techniques for large language models (LLMs), specifically focusing on refusal mechanisms and bypass strategies. This project implements activation steering methods to understand and manipulate how language models handle refusal behaviors.

## 🎯 Overview

This project provides tools to:
- **Analyze refusal patterns** in language models using activation steering
- **Compare baseline vs intervention responses** through an interactive interface
- **Implement bypass techniques** to understand model behavior
- **Visualize intervention effects** on model outputs

## 🏗️ Project Structure

```
├── streamlit_app.py      # Interactive web interface for testing interventions
├── llm_hooks.py          # Core intervention hooks and model utilities
├── complete.py           # Text generation with intervention capabilities
├── run.ipynb            # Jupyter notebook with experiments and analysis
├── avg_direction.pt     # Pre-trained intervention direction vectors
└── .gitignore           # Git ignore rules
```

## 🚀 Features

- **Interactive Streamlit Interface**: Test different intervention modes (refuse/bypass) in real-time
- **Reproducible Results**: Deterministic seed setting for consistent experiments
- **Transformer Lens Integration**: Built on the transformer_lens library for deep model analysis
- **Pre-trained Interventions**: Includes learned direction vectors for immediate use
- **Comprehensive Experiments**: Jupyter notebook with detailed analysis and visualizations

## 📋 Requirements

```bash
# Core dependencies
torch
streamlit
transformer-lens
jaxtyping
einops
numpy
pandas
matplotlib
scikit-learn
datasets
transformers
tqdm
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/skartik04/refusal.git
   cd refusal
   ```

2. **Install dependencies**:
   ```bash
   pip install torch streamlit transformer-lens jaxtyping einops numpy pandas matplotlib scikit-learn datasets transformers tqdm
   ```

3. **Verify installation**:
   ```bash
   python -c "import transformer_lens; print('Installation successful!')"
   ```

## 🎮 Usage

### Interactive Web Interface

Launch the Streamlit app to test interventions interactively:

```bash
streamlit run streamlit_app.py
```

This will open a web interface where you can:
- Choose between "refuse" and "bypass" intervention modes
- Input custom prompts
- Compare baseline model responses with intervention results
- Visualize the effects of different intervention strategies

### Programmatic Usage

```python
from llm_hooks import run_with_mode

# Test refusal intervention
baseline_response, intervention_response = run_with_mode(
    prompt="How to make a bomb?", 
    mode="refuse"
)

# Test bypass intervention
baseline_response, bypass_response = run_with_mode(
    prompt="Restricted content request", 
    mode="bypass"
)
```

### Jupyter Experiments

Explore the detailed analysis and experiments:

```bash
jupyter notebook run.ipynb
```

## 🧪 Intervention Modes

### Refusal Mode
- **Purpose**: Enhance the model's refusal behavior for harmful requests
- **Method**: Applies learned direction vectors to increase refusal probability
- **Use Case**: Safety research and content filtering

### Bypass Mode  
- **Purpose**: Understand how refusal mechanisms can be circumvented
- **Method**: Applies inverse direction vectors to reduce refusal behavior
- **Use Case**: Robustness testing and red-teaming

## 🔬 Technical Details

### Activation Steering
The project uses activation steering techniques to manipulate model behavior by:
1. **Learning direction vectors** from model activations during refusal/compliance
2. **Applying interventions** at specific layers during inference
3. **Measuring intervention effects** on output probabilities and content

### Reproducibility
All experiments use deterministic settings:
- Fixed random seeds (42)
- Deterministic CUDA operations
- Consistent tokenization and model loading

### Model Compatibility
Built on `transformer_lens` for compatibility with various transformer architectures including:
- GPT-2 family models
- LLaMA models
- Custom transformer architectures

## 📊 Results and Analysis

The project includes comprehensive analysis of:
- **Intervention effectiveness** across different prompt types
- **Layer-wise activation patterns** during refusal behavior
- **Robustness of bypass techniques** under various conditions
- **Comparative studies** between different intervention strategies

## 🛡️ Safety and Ethics

This research tool is intended for:
- **Academic research** into AI safety and alignment
- **Red team testing** of production systems
- **Understanding refusal mechanisms** in language models

**Important**: Use responsibly and in accordance with your institution's AI ethics guidelines.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-intervention`)
3. Commit your changes (`git commit -am 'Add new intervention technique'`)
4. Push to the branch (`git push origin feature/new-intervention`)
5. Create a Pull Request

## 📝 License

This project is open source. Please ensure you comply with the licenses of all dependencies.

## 🔗 Related Work

- [Transformer Lens](https://github.com/neelnanda-io/TransformerLens) - Mechanistic interpretability library
- [Activation Steering Papers](https://arxiv.org/search/?query=activation+steering+language+models) - Research on intervention techniques
- [AI Safety Research](https://www.alignmentforum.org/) - Community discussions on AI safety

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue in this repository.

---

**Disclaimer**: This tool is for research purposes. The authors are not responsible for misuse of intervention techniques. 
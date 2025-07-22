# 🎯 PromptTune - Democratizing Prompt Engineering

PromptTune is a web-based application that democratizes prompt engineering using the Gradio framework for its user interface and the DSPy library for its optimization engine. It enables users to systematically test, evaluate, and optimize their prompts for Large Language Models (LLMs).

## 🌟 Features

- **Easy-to-use Web Interface**: Built with Gradio for intuitive interaction
- **Document Upload**: Support for PDF, TXT, and Markdown context documents
- **Ground Truth Data**: Upload CSV/Excel files with question-answer pairs
- **Multiple Evaluation Metrics**: Exact Match, F1 Score, and LLM-as-Judge
- **DSPy Optimization**: Powered by DSPy's BootstrapFewShot optimizer
- **Interactive Testing**: Test playground for validating optimized prompts
- **Export Functionality**: Copy optimized prompts for use in production

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI or Anthropic API key

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd prompttune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:7860`

### Usage

1. **Configure LLM**: Select your provider (OpenAI/Anthropic) and enter your API key
2. **Upload Context Document**: Upload a PDF, TXT, or MD file containing your knowledge base
3. **Upload Ground Truth Data**: Upload a CSV/Excel file with `question` and `ground_truth_answer` columns
4. **Set Initial Prompt**: Enter your starting prompt template using `{context}` and `{question}` placeholders
5. **Choose Evaluation Metric**: Select how you want to measure prompt performance
6. **Run Optimization**: Click "🚀 Optimize My Prompt" to start the process
7. **Test Results**: Use the playground to test your optimized prompt with new questions

## 📊 Evaluation Metrics

- **Exact Match**: Requires exact text match between prediction and ground truth
- **F1 Score**: Measures word overlap between prediction and ground truth (recommended)
- **LLM-as-Judge**: Uses a powerful LLM to grade response quality

## 📁 Data Format

### Ground Truth Data Format

Your CSV/Excel file should contain at least these columns:
- `question`: The input question
- `ground_truth_answer`: The expected answer

Example:
```csv
question,ground_truth_answer
"What is the capital of France?","Paris"
"Who wrote Romeo and Juliet?","William Shakespeare"
```

### Prompt Template Format

Use placeholders in your prompt template:
```
Use the following context to answer the question:

Context: {context}

Question: {question}

Answer:
```

## 🔧 Configuration

### Supported LLM Providers

- **OpenAI**: GPT-3.5-turbo, GPT-4
- **Anthropic**: Claude-3-haiku

### Optimization Algorithms

- **BootstrapFewShot**: DSPy's few-shot learning optimizer (default)

## 🛡️ Security

- API keys are not stored or logged
- Files are processed temporarily and deleted after sessions
- All processing happens locally in your environment

## 📈 Performance

- Optimized for datasets with <50 examples
- Context documents up to 10 pages work best
- Optimization typically completes in under 5 minutes

## 🎯 Use Cases

### For AI/ML Developers (Alex)
- Improve RAG system performance
- Fine-tune prompts for specific tasks
- Get quantitative evidence for prompt effectiveness
- Export optimized prompts for production use

### For Domain Experts/Product Managers (Priya)
- Ensure chatbot accuracy with company knowledge
- Test content generator quality
- No-code prompt optimization
- Clear, understandable results

## 🚧 Roadmap

### Version 1.1+ Features
- Support for JSONL ground truth format
- Advanced metrics (Ragas integration)
- Project save/load functionality
- Visualization charts
- Batch testing capabilities
- Multi-document context support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## �� Support

For issues and questions:
1. Check the existing issues
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce

## 🙏 Acknowledgments

- [DSPy](https://github.com/stanfordnlp/dspy) for the optimization framework
- [Gradio](https://gradio.app/) for the web interface
- The open-source community for the supporting libraries

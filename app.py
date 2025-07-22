import gradio as gr
import pandas as pd
import dspy
import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from pypdf import PdfReader
from sklearn.metrics import f1_score
import re
import traceback

class PromptTuneApp:
    def __init__(self):
        self.lm = None
        self.context_content = ""
        self.ground_truth_data = None
        self.optimization_results = []
        self.best_prompt = ""
        self.compiled_program = None
        
    def setup_llm(self, provider: str, api_key: str) -> Tuple[bool, str]:
        """Setup the language model based on provider and API key."""
        try:
            if provider == "OpenAI":
                self.lm = dspy.OpenAI(
                    model="gpt-3.5-turbo",
                    api_key=api_key,
                    max_tokens=500
                )
            elif provider == "Anthropic":
                self.lm = dspy.Claude(
                    model="claude-3-haiku-20240307",
                    api_key=api_key,
                    max_tokens=500
                )
            else:
                return False, f"Provider {provider} not yet supported"
            
            dspy.settings.configure(lm=self.lm)
            return True, "LLM configured successfully!"
        except Exception as e:
            return False, f"Error configuring LLM: {str(e)}"
    
    def process_context_document(self, file) -> Tuple[bool, str]:
        """Process uploaded context document."""
        try:
            if file is None:
                return False, "No file uploaded"
            
            # Handle both file objects and file paths
            if isinstance(file, str):
                file_path = file
            else:
                file_path = file.name
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.txt' or file_extension == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.context_content = f.read()
            elif file_extension == '.pdf':
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                self.context_content = text
            else:
                return False, f"Unsupported file format: {file_extension}"
            
            return True, f"Document processed successfully! Content length: {len(self.context_content)} characters"
        except Exception as e:
            return False, f"Error processing document: {str(e)}"
    
    def process_ground_truth_data(self, file) -> Tuple[bool, str, str]:
        """Process uploaded ground truth data."""
        try:
            if file is None:
                return False, "No file uploaded", ""
            
            # Handle both file objects and file paths
            if isinstance(file, str):
                file_path = file
            else:
                file_path = file.name
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return False, f"Unsupported file format: {file_extension}", ""
            
            # Validate required columns
            required_columns = ['question', 'ground_truth_answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}", ""
            
            self.ground_truth_data = df
            
            # Create preview of first 5 rows
            preview = df.head().to_html(index=False, classes="preview-table")
            
            return True, f"Ground truth data loaded successfully! {len(df)} examples found.", preview
        except Exception as e:
            return False, f"Error processing ground truth data: {str(e)}", ""

    def exact_match_metric(self, prediction: str, ground_truth: str) -> float:
        """Calculate exact match score."""
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0
    
    def f1_score_metric(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score between prediction and ground truth."""
        def tokenize(text):
            return re.findall(r'\w+', text.lower())
        
        pred_tokens = set(tokenize(prediction))
        truth_tokens = set(tokenize(ground_truth))
        
        if not pred_tokens and not truth_tokens:
            return 1.0
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        intersection = pred_tokens.intersection(truth_tokens)
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(truth_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def llm_as_judge_metric(self, prediction: str, ground_truth: str, question: str) -> float:
        """Use LLM to judge the quality of the prediction."""
        try:
            judge_prompt = f"""
            Question: {question}
            Ground Truth Answer: {ground_truth}
            Predicted Answer: {prediction}
            
            Please evaluate how well the predicted answer matches the ground truth answer on a scale of 0.0 to 1.0.
            Consider accuracy, completeness, and relevance.
            
            Respond with only a number between 0.0 and 1.0.
            """
            
            response = self.lm(judge_prompt)
            # Extract numeric score from response
            score_match = re.search(r'(\d+\.?\d*)', response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            return 0.0
        except:
            return 0.0

    def evaluate_prompt(self, prompt_template: str, metric: str) -> float:
        """Evaluate a prompt template using the selected metric."""
        try:
            # Create DSPy signature and program
            class QASignature(dspy.Signature):
                """Answer questions using the provided context."""
                context = dspy.InputField()
                question = dspy.InputField()
                answer = dspy.OutputField()
            
            class QAProgram(dspy.Module):
                def __init__(self, prompt_template):
                    super().__init__()
                    self.prompt_template = prompt_template
                    self.predict = dspy.Predict(QASignature)
                
                def forward(self, question):
                    # Format the prompt with context
                    formatted_context = self.prompt_template.format(
                        context=self.context_content,
                        question=question
                    )
                    return self.predict(context=formatted_context, question=question)
            
            program = QAProgram(prompt_template)
            
            # Evaluate on ground truth data
            scores = []
            for _, row in self.ground_truth_data.iterrows():
                question = row['question']
                ground_truth = row['ground_truth_answer']
                
                try:
                    result = program(question)
                    prediction = result.answer
                    
                    if metric == "Exact Match":
                        score = self.exact_match_metric(prediction, ground_truth)
                    elif metric == "F1 Score":
                        score = self.f1_score_metric(prediction, ground_truth)
                    elif metric == "LLM-as-Judge":
                        score = self.llm_as_judge_metric(prediction, ground_truth, question)
                    else:
                        score = 0.0
                    
                    scores.append(score)
                except Exception as e:
                    print(f"Error evaluating question '{question}': {e}")
                    scores.append(0.0)
            
            return np.mean(scores) if scores else 0.0
        except Exception as e:
            print(f"Error in evaluate_prompt: {e}")
            return 0.0

    def optimize_prompt(self, initial_prompt: str, metric: str, optimizer: str, progress_callback) -> Tuple[bool, str]:
        """Optimize the prompt using DSPy."""
        try:
            if self.lm is None:
                return False, "Please configure an LLM first"
            
            if not self.context_content:
                return False, "Please upload a context document first"
            
            if self.ground_truth_data is None:
                return False, "Please upload ground truth data first"
            
            progress_callback("Starting optimization...")
            
            # Create DSPy examples from ground truth data
            examples = []
            for _, row in self.ground_truth_data.iterrows():
                example = dspy.Example(
                    context=self.context_content,
                    question=row['question'],
                    answer=row['ground_truth_answer']
                ).with_inputs('context', 'question')
                examples.append(example)
            
            # Split data into train and validation
            split_idx = int(0.8 * len(examples))
            train_examples = examples[:split_idx]
            val_examples = examples[split_idx:]
            
            progress_callback("Setting up DSPy program...")
            
            # Create DSPy signature and program
            class QASignature(dspy.Signature):
                """Answer questions using the provided context."""
                context = dspy.InputField()
                question = dspy.InputField()
                answer = dspy.OutputField()
            
            class QAProgram(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.predict = dspy.ChainOfThought(QASignature)
                
                def forward(self, context, question):
                    return self.predict(context=context, question=question)
            
            program = QAProgram()
            
            progress_callback("Configuring optimizer...")
            
            # Setup teleprompter (optimizer)
            if optimizer == "BootstrapFewShot":
                teleprompter = dspy.BootstrapFewShot(metric=self._create_metric_function(metric))
            else:  # Default to BootstrapFewShot
                teleprompter = dspy.BootstrapFewShot(metric=self._create_metric_function(metric))
            
            progress_callback("Running optimization...")
            
            # Compile the program
            self.compiled_program = teleprompter.compile(program, trainset=train_examples)
            
            progress_callback("Evaluating optimized prompt...")
            
            # Evaluate the compiled program
            total_score = 0
            results = []
            
            for i, example in enumerate(val_examples):
                progress_callback(f"Evaluating example {i+1}/{len(val_examples)}...")
                
                try:
                    result = self.compiled_program(context=example.context, question=example.question)
                    prediction = result.answer
                    
                    if metric == "Exact Match":
                        score = self.exact_match_metric(prediction, example.answer)
                    elif metric == "F1 Score":
                        score = self.f1_score_metric(prediction, example.answer)
                    elif metric == "LLM-as-Judge":
                        score = self.llm_as_judge_metric(prediction, example.answer, example.question)
                    else:
                        score = 0.0
                    
                    total_score += score
                    results.append({
                        'question': example.question,
                        'prediction': prediction,
                        'ground_truth': example.answer,
                        'score': score
                    })
                except Exception as e:
                    print(f"Error evaluating example: {e}")
                    results.append({
                        'question': example.question,
                        'prediction': f"Error: {str(e)}",
                        'ground_truth': example.answer,
                        'score': 0.0
                    })
            
            avg_score = total_score / len(val_examples) if val_examples else 0.0
            
            # Store results
            self.optimization_results = results
            self.best_prompt = str(self.compiled_program.predict.signature)
            
            progress_callback("Optimization complete!")
            
            return True, f"Optimization completed! Average score: {avg_score:.3f}"
            
        except Exception as e:
            error_msg = f"Error during optimization: {str(e)}\n{traceback.format_exc()}"
            progress_callback(f"Error: {str(e)}")
            return False, error_msg

    def _create_metric_function(self, metric_name: str):
        """Create a metric function for DSPy optimization."""
        def metric(gold, pred, trace=None):
            if metric_name == "Exact Match":
                return self.exact_match_metric(pred.answer, gold.answer)
            elif metric_name == "F1 Score":
                return self.f1_score_metric(pred.answer, gold.answer)
            elif metric_name == "LLM-as-Judge":
                return self.llm_as_judge_metric(pred.answer, gold.answer, gold.question)
            else:
                return 0.0
        return metric

    def test_prompt(self, question: str) -> str:
        """Test the optimized prompt with a new question."""
        try:
            if self.compiled_program is None:
                return "Please run optimization first"
            
            if not question.strip():
                return "Please enter a question"
            
            result = self.compiled_program(context=self.context_content, question=question)
            return result.answer
        except Exception as e:
            return f"Error: {str(e)}"

    def get_results_table(self) -> str:
        """Get results table as HTML."""
        if not self.optimization_results:
            return "No results available"
        
        df = pd.DataFrame(self.optimization_results)
        return df.to_html(index=False, classes="results-table", escape=False)

# Initialize the app
app = PromptTuneApp()

# Gradio Interface
def create_interface():
    with gr.Blocks(title="PromptTune - Democratizing Prompt Engineering", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎯 PromptTune
            ### Democratizing Prompt Engineering with DSPy
            
            Systematically test, evaluate, and optimize your prompts for Large Language Models using data-driven methods.
            """
        )
        
        with gr.Row():
            # Left Panel - Setup & Inputs
            with gr.Column(scale=1):
                gr.Markdown("## 🔧 Setup & Configuration")
                
                # LLM Configuration
                with gr.Group():
                    gr.Markdown("### LLM Configuration")
                    llm_provider = gr.Dropdown(
                        choices=["OpenAI", "Anthropic"],
                        label="LLM Provider",
                        value="OpenAI"
                    )
                    api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="Enter your API key"
                    )
                    llm_setup_btn = gr.Button("Configure LLM", variant="secondary")
                    llm_status = gr.Textbox(label="LLM Status", interactive=False)
                
                # File Uploads
                with gr.Group():
                    gr.Markdown("### Data Upload")
                    context_file = gr.File(
                        label="Context Document",
                        file_types=[".txt", ".md", ".pdf"],
                        file_count="single"
                    )
                    context_status = gr.Textbox(label="Context Status", interactive=False)
                    
                    ground_truth_file = gr.File(
                        label="Ground Truth Data",
                        file_types=[".csv", ".xlsx"],
                        file_count="single"
                    )
                    ground_truth_status = gr.Textbox(label="Ground Truth Status", interactive=False)
                    ground_truth_preview = gr.HTML(label="Data Preview")
                
                # Initial Prompt
                with gr.Group():
                    gr.Markdown("### Initial Prompt Template")
                    initial_prompt = gr.Textbox(
                        label="Prompt Template",
                        lines=4,
                        placeholder="Use {context} to answer the following question: {question}",
                        value="Use the following context to answer the question:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
                    )
                
                # Optimization Controls
                with gr.Group():
                    gr.Markdown("### Optimization Settings")
                    metric_choice = gr.Radio(
                        choices=["Exact Match", "F1 Score", "LLM-as-Judge"],
                        label="Evaluation Metric",
                        value="F1 Score"
                    )
                    optimizer_choice = gr.Dropdown(
                        choices=["BootstrapFewShot"],
                        label="Optimizer",
                        value="BootstrapFewShot"
                    )
                    optimize_btn = gr.Button("🚀 Optimize My Prompt", variant="primary", size="lg")
            
            # Right Panel - Results & Outputs
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Results & Testing")
                
                # Status and Progress
                with gr.Group():
                    gr.Markdown("### Status")
                    status_output = gr.Textbox(
                        label="Progress",
                        lines=3,
                        interactive=False
                    )
                
                # Best Prompt Output
                with gr.Group():
                    gr.Markdown("### Optimized Prompt")
                    best_prompt_output = gr.Textbox(
                        label="Best Prompt",
                        lines=5,
                        interactive=False
                    )
                    copy_prompt_btn = gr.Button("📋 Copy Prompt", size="sm")
                
                # Results Table
                with gr.Group():
                    gr.Markdown("### Optimization Results")
                    results_table = gr.HTML(label="Results")
                
                # Testing Playground
                with gr.Group():
                    gr.Markdown("### 🛝 Test Playground")
                    test_question = gr.Textbox(
                        label="Test Question",
                        placeholder="Enter a question to test the optimized prompt"
                    )
                    test_btn = gr.Button("Test", variant="secondary")
                    test_answer = gr.Textbox(
                        label="Answer",
                        lines=3,
                        interactive=False
                    )
        
        # Event handlers
        def setup_llm_handler(provider, key):
            success, message = app.setup_llm(provider, key)
            return message
        
        def process_context_handler(file):
            success, message = app.process_context_document(file)
            return message
        
        def process_ground_truth_handler(file):
            success, message, preview = app.process_ground_truth_data(file)
            return message, preview
        
        def optimize_handler(prompt, metric, optimizer, progress=gr.Progress()):
            def progress_callback(msg):
                progress(0.5, desc=msg)
                return msg
            
            success, message = app.optimize_prompt(prompt, metric, optimizer, progress_callback)
            
            if success:
                results_html = app.get_results_table()
                return message, app.best_prompt, results_html
            else:
                return message, "", "No results available"
        
        def test_prompt_handler(question):
            return app.test_prompt(question)
        
        # Wire up events
        llm_setup_btn.click(
            setup_llm_handler,
            inputs=[llm_provider, api_key],
            outputs=[llm_status]
        )
        
        context_file.change(
            process_context_handler,
            inputs=[context_file],
            outputs=[context_status]
        )
        
        ground_truth_file.change(
            process_ground_truth_handler,
            inputs=[ground_truth_file],
            outputs=[ground_truth_status, ground_truth_preview]
        )
        
        optimize_btn.click(
            optimize_handler,
            inputs=[initial_prompt, metric_choice, optimizer_choice],
            outputs=[status_output, best_prompt_output, results_table]
        )
        
        test_btn.click(
            test_prompt_handler,
            inputs=[test_question],
            outputs=[test_answer]
        )
        
        # Copy button functionality
        copy_prompt_btn.click(
            lambda x: x,
            inputs=[best_prompt_output],
            outputs=[],
            js="(text) => { navigator.clipboard.writeText(text); return text; }"
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

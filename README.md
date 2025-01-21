
# High Performance LLM Model in Tamil Language

## Project Overview
This project focuses on developing a high-performance Tamil-specific Large Language Model (LLM) using cutting-edge techniques like model distillation and fine-tuning. By leveraging existing datasets and efficient training methodologies, this project aims to create a lightweight yet powerful language model tailored to the Tamil language. The goal is to enable efficient text generation, classification, and other NLP tasks, even with limited computational resources.

## Objectives
- Improve the performance of the Tamil-LLaMA model through **model distillation** and **fine-tuning**.
- Preprocess and optimize Tamil datasets for better training efficiency.
- Compare and evaluate small LLM models and distillation techniques.
- Develop a working application integrating the Tamil LLM for real-world tasks.
- Optimize speed, accuracy, and resource usage for practical deployment.

## Key Features
- **Multilingual Support:** Focus on Tamil language while considering interactions with other Indian languages.
- **Efficient Training:** Use techniques like model distillation to achieve high performance with minimal resources.
- **Dataset Optimization:** Preprocess and reduce the size of instruction-based datasets for faster training.
- **Real-World Application:** Test the model's performance in tasks like text generation and classification.

## Datasets
The following datasets are used for training and evaluation:
- [abhinand/Tamil-ARIVU-workcopy](https://huggingface.co/datasets/abhinand/Tamil-ARIVU-workcopy)
- [abhinand/tamil-alpaca](https://huggingface.co/datasets/abhinand/tamil-alpaca)
- [abhinand/tamil-alpaca-orca](https://huggingface.co/datasets/abhinand/tamil-alpaca-orca)
- [RajeevanL/tamil_squad-2.0](https://huggingface.co/datasets/RajeevanL/tamil_squad-2.0)

## Methodology
1. **Data Preprocessing:**
   - Clean and tokenize the datasets.
   - Reduce dataset size while retaining essential information.
2. **Model Training:**
   - Fine-tune the Tamil-LLaMA model.
   - Use model distillation to create a lightweight version of the model.
3. **Evaluation:**
   - Compare the performance of different models using metrics like accuracy, BLEU scores, and latency.
   - Visualize results with clustering techniques and Silhouette scores.
4. **Application Integration:**
   - Deploy the model in a working application.
   - Test and optimize for tasks like text generation and classification.

## Requirements
### Software
- Python 3.8+
- Google Colab or a local machine with 15GB RAM (minimum)
- PyTorch
- Hugging Face Transformers
- IndicNLP Suite

### Hardware
- A machine with at least 8GB GPU memory (recommended for training).

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Rajeevan25/High-Performance-Tamil-Language-Model.git
   cd tamil-llm-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and preprocess datasets:
   ```bash
   python preprocess_data.py
   ```

## Usage
### Training
Run the training script:
```bash
python train_model.py --config config/train_config.json
```

### Evaluation
Evaluate the trained model:
```bash
python evaluate_model.py --model_path models/tamil_llama.pt
```

### Application
Start the application:
```bash
python app.py
```

## Results
- The distilled Tamil-LLaMA model achieves X% accuracy on the test set.
- Latency is reduced by Y% compared to the original model.
- BLEU score for text generation: Z.

## Future Work
- Extend support to additional Indian languages.
- Explore zero-shot and few-shot learning capabilities.
- Develop a mobile-friendly version for resource-constrained devices.

## References
1. Abhinand Balachandran. "TAMIL-LLAMA: A New Tamil Language Model Based on LLaMA 2."
2. [Cheng-Yu Hsieh et al., 2023] "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes."
3. Peiyuan Zhang et al. "TinyLlama: An Open-Source Small Language Model."
4. Hoffmann et al. "Scaling Laws for Neural Language Models."
5. [IndicNLP Suite Documentation](https://github.com/anoopkunchukuttan/indic_nlp_library)

## Contributors
- **Rajeevan Yogarajah**  
  Index Number: 200501P  
  University of Moratuwa

---
Feel free to contribute or suggest improvements!

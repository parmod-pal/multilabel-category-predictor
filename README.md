# 🛍️ Multi-Label Category Predictor

An AI-powered product categorization system that uses deep learning to automatically classify products into multiple categories with high accuracy.

## Overview

This project implements an intelligent product categorizer using a fine-tuned **RoBERTa** transformer model for multi-label classification. The application provides a user-friendly Streamlit interface for real-time product categorization, complete with a feedback collection system and analytics dashboard.

### Key Features

✨ **Multi-Label Classification**: Automatically assigns multiple relevant categories to a product
🤖 **Deep Learning Powered**: Uses RoBERTa (transformer-based) model for accurate predictions
📊 **Real-time Analytics Dashboard**: Monitor model performance and accuracy metrics
💬 **Feedback Loop**: Collect user feedback to continuously improve model predictions
🔧 **Rule-Based Override**: Business logic rules for specific product types (eco-friendly, apparel gender, etc.)
⚡ **High Performance**: GPU-accelerated predictions with Torch
🎯 **Confidence Scoring**: Display prediction confidence levels for each category

## Project Structure

```
cube_multilabel_category_predictor/
├── app.py                              # Main Streamlit application
├── pages/
│   └── admin.py                        # Analytics & performance dashboard
├── saved_model/                        # Pre-trained RoBERTa model files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── special_tokens_map.json
│   └── training_args.bin
├── multi_label_category_prediction.ipynb  # Model training notebook
├── products.csv                        # Sample product data
├── feedback.csv                        # User feedback & corrections
├── requirements.txt                    # Project dependencies
└── README.md                           # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- 4GB+ RAM (GPU recommended for faster inference)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/parmod-pal/multilabel-category-predictor.git
cd multilabel-category-predictor
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Usage

### Main Application (app.py)

1. **Enter Product Information**
   - Product Name: Name of the product
   - Description: Detailed product description

2. **Get Predictions**
   - Click "Predict Categories" button
   - View predicted categories with confidence scores
   - Categories are color-coded:
     - 🟢 Green: Confident matches (confidence ≥ threshold)
     - 🔵 Blue: Rule-based predictions
     - 🟠 Orange: Low confidence predictions

3. **Provide Feedback**
   - Select "Correct" if prediction is accurate
   - Select "Incorrect" if prediction needs correction
   - Choose the correct parent and sub-category
   - Feedback is saved to `feedback.csv`

### Analytics Dashboard (pages/admin.py)

Access the dashboard by clicking "📊 Open Dashboard" in the sidebar.

Features:
- **Live Accuracy**: Overall model accuracy based on feedback
- **Success Rate**: Pie chart showing correct vs incorrect predictions
- **Total Corrections**: Number of corrections collected
- **Performance Trends**: Visualize model improvement over time
- **Category Performance**: Analyze which categories are predicted correctly

## Model Architecture

- **Base Model**: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Task**: Multi-label Sequence Classification
- **Input**: Cleaned product name + description (max 256 tokens)
- **Output**: Sigmoid probabilities for each category
- **Threshold**: 0.5 (configurable)

## Configuration

Key parameters in `app.py`:

```python
MODEL_PATH = "./saved_model"      # Path to pre-trained model
FEEDBACK_FILE = "feedback.csv"     # Feedback storage file
THRESHOLD = 0.5                    # Classification threshold
```

## Features in Detail

### Text Preprocessing
- HTML tag removal (BeautifulSoup)
- Special character removal
- Lowercase normalization
- Whitespace normalization
- Limited to first 256 tokens

### Prediction Logic
1. **Neural Network Predictions**: RoBERTa model outputs sigmoid probabilities
2. **Multi-label Selection**: All predictions above threshold are included
3. **Auto-pick**: If no predictions exceed threshold, the highest probability is selected
4. **Rule-based Overrides**:
   - Eco-friendly/Sustainable products: Auto-tag as "Sustainable"
   - Tradeshow products: Auto-tag as "Tradeshow"
   - Apparel products with gender keywords: Auto-tag with gender (Men/Women/Unisex)

### Feedback Collection
- Stores both correct and incorrect predictions
- Records: product name, description, predicted category, actual category, status
- Used for model evaluation and future retraining

## Dependencies

Key packages:
- **streamlit**: Web application framework
- **torch**: Deep learning framework
- **transformers**: Pre-trained model library
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities
- **plotly**: Interactive visualizations
- **beautifulsoup4**: HTML parsing

See `requirements.txt` for complete list.

## Performance Metrics

Monitor these in the Analytics Dashboard:
- **Accuracy**: % of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Troubleshooting

### Model not loading
```
Error: Error loading model from ./saved_model
```
**Solution**: Ensure the `saved_model/` directory contains all required files and is in the same directory as `app.py`.

### Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: The model will automatically fallback to CPU. For faster CPU inference, consider GPU setup or model quantization.

### Streamlit not found
```
zsh: command not found: streamlit
```
**Solution**: Run `pip install streamlit` or ensure virtual environment is activated.

## Future Enhancements

- [ ] Model quantization for faster inference
- [ ] Batch prediction API
- [ ] Model retraining pipeline
- [ ] Category hierarchy visualization
- [ ] Export predictions to Excel/JSON
- [ ] Multi-language support
- [ ] A/B testing framework for model variants
- [ ] Fine-tune model with collected feedback

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Feedback & Issues

Found a bug or have a suggestion? [Open an issue](https://github.com/parmod-pal/multilabel-category-predictor/issues)

## License

This project is open source and available under the MIT License.

## Contact

**Author**: Parmod Pal
**Email**: parmod.pal@example.com
**GitHub**: [@parmod-pal](https://github.com/parmod-pal)

## Acknowledgments

- Hugging Face for transformer models
- Streamlit for the web framework
- PyTorch team for deep learning framework
- All contributors who provided feedback and improvements

---

Made with ❤️ for better product categorization

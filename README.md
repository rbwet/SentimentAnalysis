# Sentiment Analysis Project

This project implements a sentiment analysis pipeline that classifies tweets as **positive** or **negative** based on their content. The analysis is conducted using a Naive Bayes classifier, preprocessed with stopword filtering, and visualized for insights.

---

## 📌 Features

* **Sentiment Analysis:** Classifies tweets into positive and negative categories.
* **Data Preprocessing:** Cleans tweets by removing stopwords and special characters.
* **Naive Bayes Classifier:** Implements Naive Bayes for sentiment classification.
* **Data Handling:** Reads from `positive_tweets.json` and `negative_tweets.json` for input.
* **Stopwords Filtering:** Uses `english_stopwords.txt` for preprocessing.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed:

```bash
sudo apt-get update
sudo apt-get install python3
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:

* `nltk`
* `numpy`

---

### OpenAI API Key

> **⚠️ Important:** Do **not** hardcode your OpenAI API key directly in the script or push it to your repository. Exposing API keys can lead to unauthorized usage and security risks.

The recommended approach is to store it in a **secure environment variable** or reference it from a **`.env` file**:

```bash
export OPENAI_API_KEY='your_openai_api_key_here'
```

Alternatively, use a `.env` file and load it using `dotenv`:

1. Install `python-dotenv`:

   ```bash
   pip install python-dotenv
   ```

2. Create a `.env` file:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Load it in your script:

   ```python
   from dotenv import load_dotenv
   import os

   load_dotenv()
   openai_api_key = os.getenv('OPENAI_API_KEY')
   ```

**Never** push `.env` files to version control. Add it to `.gitignore`:

```plaintext
.env
```

---

## 📂 Directory Structure

```
.
├── tweet_processor.py         # Script for preprocessing tweets
├── sentiment_analysis.py      # Main sentiment analysis logic
├── requirements.txt           # Dependencies
├── positive_tweets.json       # Positive tweets dataset
├── negative_tweets.json       # Negative tweets dataset
└── english_stopwords.txt      # List of stopwords
```

---

## 💡 Usage

To run the tweet processor:

```bash
python3 tweet_processor.py
```

To perform sentiment analysis:

```bash
python3 sentiment_analysis.py
```

---

### Example:

```plaintext
Input: "I love sunny days!"
Output: "Positive"

Input: "I hate waiting in traffic."
Output: "Negative"
```

---

## 🔍 Advanced Usage

* **Custom Dataset:** You can add more tweets to `positive_tweets.json` and `negative_tweets.json` for improved training.
* **Real-time Sentiment Analysis:** Integrate with the Twitter API to analyze live tweets in real-time.
* **Fine-tuning Stopwords:** Update `english_stopwords.txt` to filter out domain-specific terms.

---

## 🛠️ Error Handling and Debugging

* **File Not Found:** If JSON files are missing, ensure they are placed in the root directory.
* **Dependencies Missing:** Run `pip install -r requirements.txt` if you encounter import errors.
* **Unicode Errors:** For non-English tweets, consider adding Unicode support.

---

## 🚀 Future Improvements

* [ ] Integrate Twitter API for live analysis.
* [ ] Implement SVM and Logistic Regression for improved accuracy.
* [ ] Add visualization for sentiment trends over time.

---

## 🤝 Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License.

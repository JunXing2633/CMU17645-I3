
# Movie Review Sentiment Analysis Tool for Developers

This project is a specialized tool designed for data scientists and developers working on movie recommendation systems. It aids in model selection and testing by analyzing movie reviews logged by users, using various sentiment analysis models integrated with Streamlit, MLFlow, and language models (LLMs). This application allows for the efficient evaluation of different models to determine the best fit for incorporating user review sentiment into a movie recommendation system.

## Project Scenario

In a movie recommendation system, understanding user sentiments through their reviews is crucial for personalizing recommendations. Data scientists aim to incorporate sentiment analysis into the model in production. This tool facilitates the selection and testing of sentiment analysis models using real-world data, ensuring the chosen model accurately captures user sentiment towards movies.

## Testing Data

The data used in this project is sourced from the IMDB dataset available on Hugging Face Datasets (https://huggingface.co/datasets/imdb). This dataset comprises movie reviews from IMDB, labeled as positive or negative, which is ideal for training and testing sentiment analysis models.

## Streamlit Introduction

Streamlit is an open-source app framework for Machine Learning and Data Science projects. It enables developers to quickly create interactive web applications. In this project, Streamlit provides the interface for users to upload datasets, select models for sentiment analysis, and visualize the results and performance metrics.

## MLFlow Introduction

MLFlow is an open-source platform for managing the machine learning lifecycle, including experimentation, reproducibility, and deployment. It tracks experiments, records and compares parameters and results, and manages models. This tool uses MLFlow to track the performance of various sentiment analysis models, facilitating the selection of the most effective model for production use.

## Leveraging Language Models in Three Ways

The tool employs language models in three distinct methods to provide comprehensive support for sentiment analysis:

1. **Direct API Calls to OpenAI**: Utilizes OpenAI's API for accessing GPT models, offering cutting-edge sentiment analysis capabilities directly from OpenAI's platform.

2. **Hugging Face Pipeline**: Leverages the Hugging Face `pipeline` for easy integration of RoBERTa and other transformer-based models, enabling fast and efficient sentiment analysis without the need for extensive setup.

3. **Offline Models**: Supports the use of locally saved models, allowing for sentiment analysis in environments with limited internet access or where model performance is critical.

## Installation

Follow these steps to set up the project environment:

1. **Clone the Repository**

```bash
git clone https://yourprojectrepository.git
cd yourprojectdirectory
```

2. **Create a Virtual Environment (Optional)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Set Up MLFlow (Optional)**

#### Option 1(Please use this option in lab7): Run MLFLow Tracking Server on Localhost
1. Run `mlflow server --host 127.0.0.1 --port 6001` to launch tracking server on port 6001. Show the logs in the terminal to TA for deliverable 1.
2. Visit [http://127.0.0.1:6001](http://127.0.0.1:6001) to verify your MLFlow Tracking Server is running. Show the webpage in browser to TA for deliverable 1.

#### Option 2: Use Databricks free MLFlow Server
This option does not provide model registry. This is provided because a cloud server is better at team collaboration than local server.
1. Go to the [login page of Databricks CE](https://community.cloud.databricks.com/login.html)
2. Click on ==Sign Up== at the right bottom of the login box
3. Fill out all the necessary information. Remeber to choose community edition instead of any cloud services.
When you set tracking server, instead of running `mlflow.set_tracking_uri("<your tracking server uri>")` in the python script, you should run `mlflow.login` and provide:
- Databricks Host: https://community.cloud.databricks.com/
- Username: Your Databricks CE email address.
- Password: Your Databricks CE password.
For more details, please visit [Additional Reference](#Additional-Reference)
## Usage

1. **Start the Streamlit App**

```bash
streamlit run app.py
```

2. **Navigate to the Web Interface**

Open your browser to `http://localhost:8501` to interact with the tool.

3. **Using the Tool**

- Upload the dataset (CSV format with 'text' and 'label' columns).
- Choose the number of data points and the sentiment analysis model.
- Enable MLFlow tracking if desired.
- Analyze the sentiments and review the metrics and visualizations provided.

## Contributing

We welcome contributions! Please fork the repository and submit a pull request with your enhancements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

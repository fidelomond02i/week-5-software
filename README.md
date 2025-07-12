# week-5-software
Code: Well-commented scripts/Jupyter notebooks. (To be submitted on GitHub)

Report: A PDF with:

Answers to theoretical questions. 

Screenshots of model outputs (e.g., accuracy graphs, NER results).

Ethical reflection.

(Share this PDF as an article in the Community, Peer Group Review and include it in the GitHub Repo)

Presentation: Create 3-minute video explaining your approach. (All members in the group should participate in the video. Share the video on the Community platform)

Grading Rubric
Criteria	Weight
Theoretical Accuracy	30%
Code Functionality & Quality	40%
Ethical Analysis	15%
Creativity & Presentation	15%
Tools & Resources

Frameworks: TensorFlow, PyTorch, Scikit-learn, spaCy.

Platforms: Google Colab (free GPU), Jupyter Notebook.

Datasets: Kaggle, TensorFlow Datasets.

Why This Matters

Real-World Impact: These tools power industries from healthcare to finance.

Skill Validation: Employers seek proficiency in TensorFlow, PyTorch, and Scikit-learn.

Deadline: 7 days. Showcase your AI toolkit mastery! 🚀

Need Help?

Use official documentation: TensorFlow, PyTorch, spaCy.

Post questions on the LMS Community with #AIToolsAssignment.

Pro Tip: Test code incrementally—small wins lead to big successes! 💡






Computation Graphs:

TensorFlow (Traditional): Historically, TensorFlow used a static computation graph ("define-and-run"). You would first define the entire computational graph (the network architecture) as a static structure, then compile it, and finally execute it in a session. This offered advantages for optimization and deployment but made debugging more challenging as you couldn't inspect intermediate values easily.

PyTorch: PyTorch employs a dynamic computation graph ("define-by-run" or "imperative style"). The graph is built on the fly as operations are executed. This makes it more Pythonic, intuitive, and significantly easier to debug, as you can use standard Python debugging tools to inspect values at any point.

TensorFlow 2.x Evolution: TensorFlow 2.0 largely adopted an imperative style with Eager Execution as the default, blurring this distinction considerably. While it still supports static graphs (via tf.function), the primary user experience now mimics PyTorch's dynamism.

Ease of Use & Debugging:

PyTorch: Generally considered more "Pythonic" and easier to learn for beginners due to its imperative style and direct integration with Python's debugging tools.

TensorFlow: Historically had a steeper learning curve due to its static graph paradigm. However, TensorFlow 2.x with Keras (its high-level API) and Eager Execution has significantly improved its user-friendliness, making it much more accessible.

Deployment & Production Readiness:

TensorFlow: Has historically had a more mature and extensive ecosystem for production deployment, with tools like TensorFlow Serving, TensorFlow Lite (for mobile/edge), and TensorFlow.js (for web). It's widely adopted in large-scale enterprise environments.

PyTorch: While initially more research-focused, PyTorch has made significant strides in production readiness with tools like TorchServe (co-developed with AWS) and better ONNX export capabilities, making it increasingly viable for large-scale deployments.

Community & Adoption:

TensorFlow: Boasts a larger and more established community, particularly in industry, with vast resources, tutorials, and pre-trained models.

PyTorch: Has gained significant traction in academia and research due to its flexibility and ease of experimentation, leading to a rapidly growing and very active community. Many cutting-edge research papers often release PyTorch implementations.

When to Choose One Over the Other:

Choose PyTorch if:

You are primarily focused on research, rapid prototyping, and experimentation with novel models.

You prioritize flexibility and ease of debugging using standard Python tools.

You prefer a more "Pythonic" and imperative programming style.

You are working with generative AI (e.g., GANs, VAEs) where dynamic graphs can be particularly beneficial for real-time adjustments.

Choose TensorFlow if:

You are looking for a mature, end-to-end ecosystem for production deployment at scale, especially in large enterprise environments.

You need robust cross-platform compatibility (mobile, web, edge devices) and specialized hardware support (TPUs).

You prefer a more structured approach and are working on large-scale, highly optimized deployments.

You value a very large and established industry community with extensive production examples.

Conclusion: Both frameworks are powerful and constantly evolving, often adopting features from each other. The choice often comes down to personal preference, project requirements (research vs. production), and the specific ecosystem/tools you need.

Q2: Describe two use cases for Jupyter Notebooks in AI development.
Jupyter Notebooks are an indispensable tool in AI development due to their interactive nature and ability to combine code, output, and rich text.

Exploratory Data Analysis (EDA) and Data Preprocessing:

Use Case: Before training any AI model, data scientists need to understand their data's characteristics, identify patterns, and clean it. Jupyter Notebooks are perfect for this.

How it works: You can load datasets, calculate descriptive statistics, create various visualizations (histograms, scatter plots, box plots) using libraries like Matplotlib, Seaborn, or Plotly, and document your findings directly alongside the code. You can also iteratively write code cells to handle missing values, outliers, transform features, and encode categorical variables, seeing the results of each step immediately. This interactive feedback loop significantly speeds up the data preparation phase.

Rapid Prototyping and Model Experimentation:

Use Case: AI development often involves iterative experimentation with different model architectures, hyperparameters, and training strategies. Jupyter Notebooks provide an ideal environment for this.

How it works: You can define a model, train it on a small subset of data, evaluate its performance (e.g., plot loss curves, accuracy graphs), and then quickly modify parts of the code (e.g., change the number of layers, adjust the learning rate, try a different optimizer) and re-run only the relevant cells. This "define-by-run" execution style allows for quick iteration and comparison of different approaches, helping data scientists converge on an optimal model more efficiently than in a traditional IDE.

Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?
While basic Python string operations (like split(), find(), replace(), lower()) can perform very rudimentary text manipulation, spaCy provides a deep, linguistic, and highly efficient understanding of text that goes far beyond simple character or word-level operations.

Here's how spaCy enhances NLP tasks:

Linguistic Annotation and Contextual Understanding:

Python Strings: Treat text as just a sequence of characters. text.split(' ') might separate words, but it won't understand "don't" as two tokens ("do" and "n't") or "New York" as a single entity.

spaCy: Processes text into a Doc object, which is a rich container of linguistic annotations. It automatically performs:

Tokenization: Splits text into meaningful units (tokens), handling contractions, punctuation, and multi-word expressions intelligently.

Part-of-Speech (POS) Tagging: Identifies the grammatical role of each word (noun, verb, adjective, etc.).

Dependency Parsing: Shows the grammatical relationships between words in a sentence.

Lemmatization: Reduces words to their base form (e.g., "running," "runs," "ran" all become "run"). This is crucial for analyzing word meaning regardless of inflection.

Named Entity Recognition (NER): Identifies and categorizes named entities in text (persons, organizations, locations, dates, products, etc.). This is a complex task impossible with basic string operations.

Enhancement: These annotations provide contextual understanding. For example, knowing "Apple" is an ORG (organization) vs. a PRODUCT vs. a FRUIT based on surrounding words is powerful, whereas simple string search would treat all "Apple" instances identically.

Efficiency and Performance:

Python Strings: Operations are generally less optimized for large-scale text processing, especially when linguistic rules need to be applied iteratively.

spaCy: Written in Cython for speed, spaCy is designed for production-grade performance and can process large volumes of text very efficiently. It leverages pre-trained statistical models optimized for speed and accuracy.

Enhancement: For real-world applications dealing with gigabytes of text data, spaCy's speed is a game-changer, allowing for rapid processing that would be prohibitively slow with custom Python string logic.

Rule-Based Matching & Pattern Recognition:

Python Strings: Primarily use regular expressions (re module) for pattern matching, which can become complex and brittle for linguistic patterns.

spaCy: Offers powerful rule-based matching engines (Matcher, PhraseMatcher) that operate on tokens and their linguistic attributes (e.g., find a sequence of tokens where the first is a proper noun, the second is a verb, and the third is an adjective).

Enhancement: This allows for much more sophisticated and robust pattern matching than regex alone, enabling the extraction of specific phrases or the identification of complex structures based on their linguistic properties, not just their character sequence.

Pre-trained Models and Ecosystem:

Python Strings: Provide no inherent NLP capabilities; you'd have to build everything from scratch.

spaCy: Comes with high-quality, pre-trained statistical models for various languages, which are ready to use out-of-the-box for tasks like NER, POS tagging, and dependency parsing. It also integrates well with Transformer models (like BERT) for more advanced tasks.

Enhancement: This significantly reduces the development time and expertise required to build sophisticated NLP pipelines. You don't need to train your own models for common tasks; you can leverage spaCy's pre-trained capabilities.

In short, spaCy moves beyond mere text manipulation to provide a robust framework for understanding the meaning and structure of human language, offering speed, accuracy, and depth that basic Python string operations cannot achieve.

2. Comparative Analysis: Scikit-learn and TensorFlow
Target Applications:
Scikit-learn:

Primary Focus: Classical machine learning algorithms for structured/tabular data.

Use Cases:

Classification: Spam detection, sentiment analysis (simpler cases), credit risk assessment, medical diagnosis (binary/multi-class).

Regression: House price prediction, stock price forecasting, sales prediction, time series forecasting (linear models).

Clustering: Customer segmentation, anomaly detection, document grouping.

Dimensionality Reduction: Feature selection, data visualization (PCA, t-SNE).

Traditional NLP: Basic text classification (e.g., using TF-IDF features with SVM/Logistic Regression).

Strengths: Excellent for common, well-defined ML problems, especially when deep learning might be overkill or data is limited/tabular.

TensorFlow:

Primary Focus: Deep learning and neural networks for unstructured data (images, audio, text) and large-scale, complex problems.

Use Cases:

Computer Vision: Image recognition, object detection, image generation (GANs), semantic segmentation.

Natural Language Processing (NLP): Machine translation, complex sentiment analysis (using Transformers), chatbots, text generation, speech recognition.

Reinforcement Learning: Training agents for games, robotics, complex control systems.

Large-scale Numerical Computation: Any task requiring extensive parallel computations on GPUs/TPUs.

Strengths: Unparalleled for building and deploying complex, state-of-the-art deep learning models that can learn intricate patterns from massive datasets.

Ease of Use for Beginners:
Scikit-learn:

Ease of Use: Generally considered easier for beginners. It has a highly consistent and intuitive API (.fit(), .predict(), .transform()) across all its algorithms. This makes it quick to pick up and experiment with different models.

Learning Curve: Relatively shallow for classical ML tasks. Concepts are often more directly interpretable.

TensorFlow:

Ease of Use: Historically had a steeper learning curve, especially with its older static graph paradigm and lower-level API.

Learning Curve (TF 2.x with Keras): TensorFlow 2.x, with Keras as its high-level API and Eager Execution, has significantly improved its beginner-friendliness. Building standard neural networks is now much more straightforward. However, for highly custom models, advanced features, or distributed training, it can still be more complex than Scikit-learn.

Community Support:
Scikit-learn:

Community: Has a very large, mature, and active community. It's the de facto standard for classical ML in Python, leading to abundant tutorials, documentation, Stack Overflow answers, and examples.

Resources: Official documentation is excellent, and many online courses and books focus on Scikit-learn.

TensorFlow:

Community: Boasts an enormous and highly active global community, backed by Google. It has extensive official documentation, tutorials, research papers, and a vast ecosystem of related tools (TensorBoard, TF Serving, TF Lite).

Resources: Due to its broad adoption in both research and industry, there are countless online courses, books, and public projects. For cutting-edge deep learning, the community support for TensorFlow (and PyTorch) is unmatched.

Part 2: Practical Implementation (50%)
(Note: For this part, you would actually write and execute the code. Below are descriptions of what your code should do and example structures.)

Task 1: Classical ML with Scikit-learn
Dataset: Iris Species Dataset
Goal:

Preprocess the data (handle missing values, encode labels).

Train a decision tree classifier to predict iris species.

Evaluate using accuracy, precision, and recall.
Deliverable: Python script/Jupyter notebook with comments explaining each step.

Example Python/Jupyter Notebook Structure:

Python

# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np # For potential missing value handling, though Iris is clean

# 1. Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target) # Target is already numerical (0, 1, 2)

# Display basic info and check for missing values (Iris dataset is clean by default)
print("Dataset Info:")
X.info()
print("\nMissing values before preprocessing:")
print(X.isnull().sum())

# 2. Preprocess the data
# Handle missing values: Iris dataset typically has no missing values.
# If it did, you might use:
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy='mean')
# X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Encode labels: The target variable (species) is already numerical (0, 1, 2),
# so no explicit label encoding is strictly necessary for the Decision Tree.
# If it were strings like ['setosa', 'versicolor', 'virginica'],
# we'd use LabelEncoder:
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# print(f"\nOriginal Labels: {iris.target_names}")
# print(f"Encoded Labels: {np.unique(y_encoded)}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# 3. Train a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
print("\nDecision Tree Classifier trained successfully.")

# 4. Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# Precision and Recall for multi-class need 'average' parameter
# 'weighted' accounts for class imbalance, 'macro' gives unweighted mean
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")

# Optional: Visualize the Decision Tree (requires graphviz and pydotplus)
# from sklearn.tree import export_graphviz
# from io import StringIO
# import pydotplus
# from IPython.display import Image
#
# dot_data = StringIO()
# export_graphviz(dt_classifier, out_file=dot_data,
#                 filled=True, rounded=True,
#                 feature_names=iris.feature_names,
#                 class_names=iris.target_names,
#                 special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
Task 2: Deep Learning with TensorFlow/PyTorch
Dataset: MNIST Handwritten Digits
Goal:

Build a CNN model to classify handwritten digits.

Achieve >95% test accuracy.

Visualize the model’s predictions on 5 sample images.
Deliverable: Code with model architecture, training loop, and evaluation.

Example TensorFlow/Keras Structure:

Python

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data to include channel dimension (for CNNs: (batch, height, width, channels))
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode the target labels
y_train_one_hot = to_categorical(y_train, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

print(f"X_train shape: {X_train.shape}, y_train_one_hot shape: {y_train_one_hot.shape}")
print(f"X_test shape: {X_test.shape}, y_test_one_hot shape: {y_test_one_hot.shape}")

# 2. Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Dropout for regularization
    Dense(10, activation='softmax') # 10 classes for digits 0-9
])

# Display model summary
model.summary()

# 3. Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
print("\nTraining the CNN model...")
history = model.fit(X_train, y_train_one_hot,
                    epochs=10, # You might need more or fewer epochs
                    batch_size=128,
                    validation_split=0.1)

# 5. Evaluate the model on the test set
print("\nEvaluating the model on the test set...")
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print(f"Test Accuracy: {accuracy:.4f}")

# Check if target accuracy is achieved
if accuracy > 0.95:
    print("Achieved >95% test accuracy!")
else:
    print("Test accuracy is below 95%. Consider adjusting model architecture or training parameters.")

# 6. Visualize model's predictions on 5 sample images
print("\nVisualizing predictions on sample images:")
predictions = model.predict(X_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predicted_labels[i]}\nActual: {y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Optional: Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()
Task 3: NLP with spaCy
Text Data: User reviews from Amazon Product Reviews (you'll need to find a suitable dataset or create sample reviews).
Goal:

Perform named entity recognition (NER) to extract product names and brands.

Analyze sentiment (positive/negative) using a rule-based approach.
Deliverable: Code snippet and output showing extracted entities and sentiment.

Example Python/Jupyter Notebook Structure:

Python

import spacy
from spacy.matcher import Matcher

# Load a pre-trained English model (small model for quick testing)
# You might need to download it first: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Sample Amazon Product Reviews (replace with actual dataset loading)
# For a real project, you'd load from a CSV/JSON file.
amazon_reviews = [
    "I absolutely love my new iPhone 15 Pro Max! The camera is incredible, and the battery life is amazing. Apple really outdid themselves.",
    "This Samsung Galaxy Watch 5 is a sleek device, but its battery drains too quickly. Disappointed with the performance.",
    "The Sony WH-1000XM5 headphones offer superior noise cancellation. Best headphones I've ever owned. Highly recommend Sony products.",
    "Bought a new Dell XPS 15 laptop. Great screen, but the fan noise is a bit annoying. Overall, it's a solid machine from Dell.",
    "The product arrived broken. Terrible quality, complete waste of money. Avoid this brand."
]

print("--- Named Entity Recognition (NER) and Sentiment Analysis ---")

for i, review_text in enumerate(amazon_reviews):
    print(f"\nReview {i+1}: \"{review_text}\"")
    doc = nlp(review_text)

    # 1. Perform Named Entity Recognition (NER) to extract product names and brands
    print("  Extracted Entities:")
    found_entities = []
    for ent in doc.ents:
        # Focusing on common entity types for products/brands
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "GPE", "LOC"]: # ORG for companies, PRODUCT for specific products
            found_entities.append(f"    - {ent.text} ({ent.label_})")
    if found_entities:
        for entity in found_entities:
            print(entity)
    else:
        print("    No specific product/brand entities found by NER.")

    # --- Enhance NER with rule-based matching for specific patterns (e.g., "iPhone 15 Pro Max") ---
    # This is useful if spaCy's default NER misses specific product names
    matcher = Matcher(nlp.vocab)
    # Example pattern for "iPhone" followed by model number
    pattern_iphone = [{"LOWER": "iphone"}, {"IS_DIGIT": True, "OP": "?"}, {"LOWER": "pro", "OP": "?"}, {"LOWER": "max", "OP": "?"}]
    matcher.add("IPHONE_MODEL", [pattern_iphone])

    # Example pattern for "Galaxy Watch" followed by model number
    pattern_galaxy_watch = [{"LOWER": "galaxy"}, {"LOWER": "watch"}, {"IS_DIGIT": True, "OP": "?"}]
    matcher.add("GALAXY_WATCH", [pattern_galaxy_watch])

    matches = matcher(doc)
    rule_based_entities = []
    for match_id, start, end in matches:
        span = doc[start:end]
        rule_based_entities.append(f"    - {span.text} (Rule-based Product)")

    if rule_based_entities:
        print("  Rule-based Entities:")
        for entity in rule_based_entities:
            print(entity)

    # 2. Analyze sentiment using a rule-based approach
    # Define keywords for sentiment analysis
    positive_keywords = ["love", "incredible", "amazing", "best", "highly recommend", "great", "solid", "excellent", "superb"]
    negative_keywords = ["drains too quickly", "disappointed", "annoying", "terrible", "waste of money", "avoid", "broken", "bad", "poor"]

    sentiment_score = 0
    words = [token.text.lower() for token in doc]

    for word in words:
        if word in positive_keywords:
            sentiment_score += 1
        elif word in negative_keywords:
            sentiment_score -= 1

    sentiment = "Neutral"
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"

    print(f"  Sentiment: {sentiment} (Score: {sentiment_score})")

    # Optional: More sophisticated sentiment can use spaCy's TextCategorizer or external libraries
    # from textblob import TextBlob
    # blob = TextBlob(review_text)
    # print(f"  TextBlob Sentiment: {blob.sentiment.polarity:.2f} (Polarity)")
Part 3: Ethics & Optimization (10%)
1. Ethical Considerations
Potential Biases in your MNIST or Amazon Reviews model:

MNIST Handwritten Digits Model (Bias in Data):

Potential Bias: While MNIST is often considered a "clean" dataset, potential biases could arise if the handwritten digits primarily came from a very specific demographic (e.g., only young adults, only people from a particular region, or only certain writing styles). A model trained solely on such data might perform poorly when presented with handwriting from different demographics, ages, or cultural backgrounds (e.g., different styles of writing certain digits). This is a form of representation bias in the training data.

Mitigation with TensorFlow Fairness Indicators:

Identify Bias: To mitigate this, one would ideally need demographic metadata (age, origin, handedness etc.) associated with each MNIST sample (which isn't usually available in the standard MNIST). If such metadata were available, Fairness Indicators could be used to slice performance metrics (e.g., accuracy, false positive rate) by these demographic attributes. For example, you could check if the model's accuracy for "digit 7" written by older adults is significantly lower than for younger adults.

Actionable Insights: If a disparity is found, Fairness Indicators helps quantify it. Mitigation steps would then involve:

Data Augmentation: Augmenting the dataset with more diverse handwriting styles.

Re-weighting Samples: Giving more weight to underperforming groups during training.

Fairness-Aware Optimization: Using techniques like "constrained optimization" (which can be used with TFCO, a library designed to work with Fairness Indicators) to ensure the model performs equally well across different demographic slices.

Amazon Reviews Model (Bias in Data & NLP):

Potential Biases:

Sentiment Bias: If the training data for sentiment analysis (even rule-based, as rules are often human-derived) over-represents or under-represents certain opinions, or if keywords are implicitly associated with certain groups or products. For example, if "budget-friendly" is always associated with negative sentiment in training, but in reality, it's a positive for some consumers.

Product/Brand Bias in NER: The NER model (even en_core_web_sm and your custom rules) might be more adept at recognizing popular Western brands/products due to more extensive training data, while missing niche, regional, or non-English brands. If reviews frequently mention brands from a specific region or culture, but the model has not been adequately trained on them, it will fail to extract them. This is a form of representation bias and data collection bias.

Spam/Fake Review Bias: AI might misclassify genuine reviews as spam or vice-versa if trained on imbalanced or unrepresentative spam datasets.

Mitigation with spaCy's Rule-Based Systems (for NER bias):

Custom Rules for Undetected Entities: If the pre-trained en_core_web_sm model (or your current rule set) consistently misses specific product names or brands that are important for your context (e.g., African brands, specialized industrial equipment), you can explicitly add custom rules to spaCy's Matcher or PhraseMatcher. For instance, if "Safaricom" is often missed, you can add a rule to recognize it as an ORG. This ensures that important entities are captured regardless of their frequency in general training corpora.

Post-processing with Rules: You can use spaCy's rule-based components in the pipeline to correct misclassifications or merge entities based on domain-specific knowledge, thereby reducing statistical model bias in specific contexts.

Mitigation for Sentiment Bias (Rule-based approach):

Review and Diversify Keyword Lists: Manually inspect and expand your positive_keywords and negative_keywords to ensure they are culturally neutral and cover a wider range of expressions. Avoid words that might have positive connotations in one context but negative in another, or words that are primarily used by a specific demographic.

Contextual Rules: Introduce more sophisticated rules that consider the context of a word. For example, "slow" might be negative for performance but neutral for a "slow cooker." SpaCy's dependency parser can help with this by analyzing word relationships.

Human-in-the-Loop: Regularly review misclassified sentiment examples and update the rule sets.

2. Troubleshooting Challenge
Buggy Code: A provided TensorFlow script has errors (e.g., dimension mismatches, incorrect loss functions). Debug and fix the code.

(This task requires an actual buggy script to debug. As an AI, I cannot generate or interact with live buggy code to fix it. However, I can explain the common types of errors you'd look for and how to debug them.)

Common TensorFlow Errors and Debugging Strategies:

Dimension Mismatches (ValueError: Dimensions must be equal, Input 0 is incompatible with layer...):

Cause: This is perhaps the most frequent error in deep learning. It occurs when the shape of the data (input, output of a layer, or target labels) does not match what the TensorFlow operation or Keras layer expects. Common culprits include:

Incorrect input_shape in the first layer: Not matching the shape of your input data (e.g., (28, 28, 1) for grayscale images, but you're passing (28, 28)).

Missing Flatten layer: Between convolutional/pooling layers and dense layers.

Incorrect output shape of a layer: The number of units in a Dense layer not matching the number of classes in your softmax layer.

Batch dimension handling: Forgetting that TensorFlow often adds a batch dimension (e.g., (None, height, width, channels)).

Debugging Strategy:

model.summary(): This is your best friend. It shows the output shape of each layer. Compare the Output Shape of one layer to the Input Shape of the next.

Print tensor.shape: Insert print(tensor.shape) statements at various points in your data preprocessing or model forward pass to explicitly see the dimensions of your tensors.

Review tf.reshape() or np.reshape(): Ensure reshape operations are correctly applied and maintain the total number of elements.

Check to_categorical(): Ensure your labels are correctly one-hot encoded and match the number of output classes.

Incorrect Loss Functions (InvalidArgumentError: Loss calculation failed..., "Expected target of type int32 but received float32"):

Cause: Using a loss function that is incompatible with your model's output activation or your target label format.

categorical_crossentropy vs. sparse_categorical_crossentropy:

categorical_crossentropy expects one-hot encoded labels (e.g., [0, 0, 1, 0, 0] for class 2). Your model's final layer usually has a softmax activation.

sparse_categorical_crossentropy expects integer labels (e.g., 2 for class 2). Your model's final layer usually has a softmax activation.

binary_crossentropy: Used for binary classification (two classes) with a sigmoid activation in the output layer.

Regression Loss Functions (e.g., MSE, MAE): Accidentally using these for a classification task, or vice versa.

Debugging Strategy:

Match Loss to Output Activation: Verify the activation function of your model's final layer and choose the appropriate loss function.

Match Loss to Label Format: Confirm whether your y_train and y_test are one-hot encoded or integer labels, and select the corresponding loss function.

Check Data Types: Ensure input data and labels have the correct data types (tf.float32, tf.int32, etc.).

General Debugging Tips:

Read Error Messages Carefully: TensorFlow's error messages are often verbose but provide crucial information about the exact layer or operation where the error occurred, and what the expected vs. actual shapes/types were.

Incremental Testing: Run your code in small blocks. Test data loading, then preprocessing, then model definition, then compilation, then training. This helps isolate where the error occurs.

Small Dataset: Start with a very small subset of your data to quickly test the pipeline without long training times.

TensorBoard: For more complex debugging and visualization of graph operations, use TensorBoard.

Official Documentation: Refer to the TensorFlow/Keras documentation for specific layer expectations and function arguments.

Bonus Task (Extra 10%)
Deploy Your Model: Use Streamlit or Flask to create a web interface for your MNIST classifier.
Deliverable: Screenshot and a live demo link.

(This requires actual coding and deployment. Here's an outline for a Streamlit app.)

Example Streamlit App (app.py):

Python

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image # Pillow library for image manipulation
import io

# Load the pre-trained MNIST model
# Make sure 'mnist_cnn_model.h5' is in the same directory as this script,
# or provide the full path to your saved model.
# You would save your model after training in Task 2 using:
# model.save('mnist_cnn_model.h5')
try:
    model = load_model('mnist_cnn_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure 'mnist_cnn_model.h5' is in the correct path.")
    st.stop() # Stop the app if model can't be loaded

st.title("MNIST Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) and the model will classify it.")

# Option 1: File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert("L") # Convert to grayscale

    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the model
    # Resize to 28x28 pixels
    image = image.resize((28, 28))
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{predicted_digit}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("All Probabilities:")
    # Display probabilities for all digits
    prob_df = pd.DataFrame(prediction.flatten(), columns=['Probability'], index=range(10))
    prob_df = prob_df.sort_values(by='Probability', ascending=False)
    st.bar_chart(prob_df)

    # Optional: Allow user to draw (more complex, requires frontend drawing library)
    # For a simpler approach, stick to file upload.

st.caption("Note: For best results, use clear, centered handwritten digits similar to the MNIST dataset.")

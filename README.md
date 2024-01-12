# Predicting Political Orientation Based on Language Use
## Paper details
Python (3.11.7) and R (4.3.1) are used to produce these result.

Data used:
- Reddit data from the subreddit R\AskEurope.
- Profanity data of Europe from https://github.com/matteofabbri/EuropeanGreatAtlasOfProfanities

## tl;dr
- XGBoost model emerged as the best-performing classifier compared to Random Forest Classifier and Logistic Regression, achieving the highest accuracy at 51 percent. 
- It is an improvement over the baseline model (OLS) with an accuracy of 44%.
- Language use variables that seems significant predictors from the politeness package are Hedges, Negation, Informal.Title, First.Person.Plural (positive effect) and Impersonal.Pronoun, Swearing, Filler.Pause, Goodbye, For.Me, Ask.Agency with an negative effect.

## Reproduction
1. Make sure to install the dependencies listed in the section below. This can be done by pip install -r requirements.tx 
2. Make sure to have the right versions for Python (3.11.7) and R (4.3.1). 
3. Download the profanity data.
4. Turn the json file of the profanity data to a csv file with the following code in swearwords.ipynb
```python
with open('build.json', 'r', encoding='utf-8') as json_file:
   json_data = json.load(json_file)


csv_data = []
for lang, words in json_data.items():
 for i, word in enumerate(words):
     csv_data.append([lang, i, word])
with open('output.csv', 'w', encoding="utf-8", newline='') as csv_file:
 writer = csv.writer(csv_file)
 writer.writerow(['Language', 'Index', 'Word']) # write the header
 for row in csv_data:
     writer.writerow(row)

csv_data
```

5. Run in data_cleaning.ipynb the first cell looking like this:
```python
    # RUN this cell if you run this code for the first time
    # !pip install -r requirements.txt
    # only run following if you have not yet downloaded the stopwords
    # nltk.download('stopwords') 
    # nltk.download('punkt')
    # nltk.download('wordnet')
```
6. Run the data_cleaning.ipynb file.
7. Install in R the package politeness and run politeness_extraction on a R version of 4.3.1
8. Run data_exploration.ipynb

## Dependencies
The libraries and its versions that are required in Python:
- pandas == 2.1.4
- tqdm == 4.66.1
- nltk == 3.8.1
- matplotlib == 3.8.2
- scipy == 1.11.4
- scikit-learn == 1.3.2
- statsmodels == 0.14.1
- seaborn == 0.13.0
- wordcloud == 1.9.3
- xgboost

The libraries and its versions that are required in R:
- politeness == 0.9.3

## Resources required
The GPU used is Intel (R) UHD Graphics 630. 
The CPU used is Intel(R) Core(TM) i7-9750H CPU.

Running the following files took:
- swearwords.ipynb: 12 min
- data_cleaning.ipynb: 57 min
- data_exploration.ipynb: 45 min


## Experimental manipulation
Several elements can be changed in the experiment.

### Add data cleaning methods
Extra data cleaning methods can be added by adding a function among these functions:
```python
def preprocess(text):
    text = text.lower()

    text  = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]',"", text)

    tokens = nltk.word_tokenize(text)

    return tokens

def remove_stopwords (tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def lemmatization(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized

```

After defining the function, incorporate the function in clean_text and apply the function to the variable text.
```python
def clean_text(text):
    tokens = preprocess(text)
    filtered_tokens = remove_stopwords(tokens)
    lemmatizer = lemmatization(filtered_tokens)
    clean = " ".join(lemmatizer)
    return clean

```
### Adding label options
If three labels is too much or too less to predict, the amount of labels can be adjusted here by modifying the if statement:

```python
df_political_leaning['political_leaning_id'] = df_political_leaning['political_leaning'].apply(lambda x: -1 if x == 'left' else 0 if x == 'center' else 1)
```
### Changing the cutoff for the cursewords
Changing the cutoff can be done by modifying the following line in data_exploration.ipynb:

```python
df_politics_cleaned = df_politics[df_politics['amount_of_cursewords'] <= 52]
```
### Adding another classification model
Copy the following lines to copy the dependent and independent variables:
```python
# Split the data into features (X) and target (Y)
X = df_politics_super_cleaned[super_significant]
Y = df_politics_super_cleaned['political_leaning_id']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

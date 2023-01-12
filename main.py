# Flask imports
from flask import Blueprint, send_file, request, render_template
from flask_login import login_required, current_user
# App imports
from app import db
from create_insert_sqlite_tocsv import create_prediction, create_connection, save_db_to_csv
# Importing Python libraries
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os
import keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
# Downloading nltk dependencies
for dependency in (
    "omw-1.4",
    "stopwords",
    "wordnet",
    "punkt",
    "averaged_perceptron_tagger"
):
    nltk.download(dependency)

# F1-SCORE


def f1_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# LOAD MODEL


MODEL_VERSION = 'model.h5'  # modèle
MODEL_PATH = os.path.join(os.getcwd(), 'models',
                          MODEL_VERSION)  # path vers le modèle
model = load_model(MODEL_PATH, custom_objects={
                   'f1_score': f1_score}, compile=False)  # chargement du modèle

# LOAD TOKENIZER

TOKENIZER_VERSION = 'tokenizer.pickle'
TOKENIZER_PATH = os.path.join(os.getcwd(), 'models',
                              TOKENIZER_VERSION)  # path vers le tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# PREPROCESS TEXT

stop_words = stopwords.words('english')


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN


def cleaning(data):
    # 1. Tokenize
    text_tokens = word_tokenize(data.replace("'", "").lower())
    # 2. Remove Puncs
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]
    # 3. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    # 4. Lemmatize
    POS_tagging = pos_tag(tokens_without_sw)
    wordnet_pos_tag = []
    wordnet_pos_tag = [(word, get_wordnet_pos(pos_tag))
                       for (word, pos_tag) in POS_tagging]
    wnl = WordNetLemmatizer()
    lemma = [wnl.lemmatize(word, tag) for word, tag in wordnet_pos_tag]
    return " ".join(lemma)


# PREDICT

def my_predict(text):
    # Tokenize text
    text_pad_sequences = pad_sequences(tokenizer.texts_to_sequences(
        [text]), maxlen=300)
    # Predict
    predict_val = float(model.predict([text_pad_sequences]))
    recommandation = "Recommandé" if predict_val > 0.5 else "Non Recommandé"
    score = int(predict_val*100)
    return score, recommandation

# MAIN


main = Blueprint('main', __name__)

# HOMEPAGE ROUTE


@main.route('/')
def index():
    return render_template('index.html')


# RECOMMANDATION ROUTE (GET and POST)


@main.route('/recommandation', methods=['GET', 'POST'])
@login_required
def predict():

    if request.method == 'POST':
        if request.form['customer_feedback']:
            customer_feedback = str(request.form['customer_feedback'])
            clean_comment = cleaning(customer_feedback)

            score, recommandation = my_predict(clean_comment)

            # Save into sqlite DB
            # DB connection
            conn = create_connection(os.path.join(
                os.getcwd(), 'prediction_extractions', 'predictions.db'))
            c = conn.cursor()

            # DB and tables creation
            c.execute('''CREATE TABLE IF NOT EXISTS predictions (predictions_id INTEGER PRIMARY KEY AUTOINCREMENT, predictions_review_text TEXT, predictions_recommended VARCHAR(50), predictions_review_score INTEGER)''')
            create_prediction(conn, (customer_feedback, recommandation, score))
            save_db_to_csv(conn)

            return render_template('recommandation.html', name=current_user.name, text=customer_feedback, recommandation=recommandation, score=f"Note estimée : {score}/100")

    else:
        return render_template('recommandation.html', name=current_user.name)

# DOWNLOAD DATABASE ROUTE (GET and POST)


@main.route('/getRepVetCSV', methods=['GET', 'POST'])
@login_required
def repvet_csv():
    return send_file(os.path.join(
        os.getcwd(), 'prediction_extractions', 'repvet.csv'),
        mimetype='text/csv',
        download_name='repvet.csv',
        as_attachment=True
    )

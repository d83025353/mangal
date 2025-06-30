# === IMPORTS ===
import os
import time
import subprocess
import threading
import sqlite3
from datetime import datetime
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, session
import pytz
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import ta
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Flatten, Concatenate, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping

# === CONFIGURATION ===
app = Flask(__name__)
app.secret_key = "supersecurekey"
UPLOAD_FOLDER = 'scripts'
MODEL_PATH = "super_model_online.keras"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Globals
model = None
prediction = {"ohlc": None, "signal": "None"}
enabled_scripts = {"CALL": None, "PUT": None}
timezone = pytz.timezone("Asia/Kolkata")
API_KEY = 'a07464cf9439493fb8b86b6d9e80f5a2'
TD_URL = "https://api.twelvedata.com/time_series"
SYMBOL = "EUR/USD"
INTERVAL = "5min"
LOOKBACK = 10
target_times = [f"{i:02d}:01" for i in range(0, 60, 5)]

# === DATABASE INIT ===
def init_db():
    with sqlite3.connect("users.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            );
        """)
init_db()

# === MODEL DEFINITION ===
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation='relu'), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

def build_super_model(input_shape):
    inputs = Input(shape=input_shape)
    x1 = TransformerBlock(embed_dim=input_shape[-1], num_heads=4, ff_dim=64)(inputs)
    x2 = LSTM(64, return_sequences=False)(inputs)
    x = Concatenate()([Flatten()(x1), x2])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(4, activation="linear")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def load_or_build_model(input_shape):
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, custom_objects={'TransformerBlock': TransformerBlock})
    else:
        return build_super_model(input_shape)

# === DATA FETCHING AND PREPARATION ===
def fetch_data():
    try:
        response = requests.get(TD_URL, params={"symbol": SYMBOL, "interval": INTERVAL, "outputsize": 100, "apikey": API_KEY})
        data = response.json()
        if "values" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df[['open', 'high', 'low', 'close']].astype(float).sort_index()
        return df
    except:
        return pd.DataFrame()

def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=14).ema_indicator()
    df['sma'] = ta.trend.SMAIndicator(df['close'], window=14).sma_indicator()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['williams'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
    df.dropna(inplace=True)
    return df

def prepare_data(df, lookback=LOOKBACK):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(lookback, len(scaled) - 1):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i + 1][:4])
    return np.array(X), np.array(y), scaler

def retrain_and_predict(model, df):
    df = add_indicators(df)
    X, y, scaler = prepare_data(df)
    if len(X) == 0:
        return model, None
    #model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    model.fit(X, y, epochs=5, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])

    latest_input = np.expand_dims(X[-1], axis=0)
    prediction_scaled = model.predict(latest_input, verbose=0)[0]
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, :4] = prediction_scaled
    inv_transformed = scaler.inverse_transform(dummy)
    pred_ohlc = inv_transformed[0][:4]
    model.save(MODEL_PATH)
    update_signal(pred_ohlc)
    return model, pred_ohlc

def update_signal(pred_ohlc):
    prediction["ohlc"] = pred_ohlc.tolist()
    if pred_ohlc[0] < pred_ohlc[3]:
        signal = "CALL"
    elif pred_ohlc[0] > pred_ohlc[3]:
        signal = "PUT"
    else:
        signal = "⚖️ NO SIGNAL"
    prediction["signal"] = signal
    if signal in enabled_scripts and enabled_scripts[signal]:
        subprocess.Popen(["python", os.path.join(UPLOAD_FOLDER, enabled_scripts[signal])])

# === SCHEDULER ===
def time_based_scheduler():
    global model
    while True:
        timezone = pytz.timezone("Asia/Kolkata")
        now = datetime.now(timezone)
        current_time = now.strftime("%M:%S")
        if current_time in target_times:
            df = fetch_data()
            if df.empty or len(df) < 80:
                time.sleep(60)
                continue
            df = add_indicators(df)
            X, _, _ = prepare_data(df)
            if model is None and len(X) > 0:
                model = load_or_build_model(input_shape=X.shape[1:])
            if model:
                model, _ = retrain_and_predict(model, df)
            #time.sleep(60)
        time.sleep(1)

threading.Thread(target=time_based_scheduler, daemon=True).start()

# === HTML Templates ===
ADMIN_TEMPLATE = '''
<h2>EUR/USD Admin Panel</h2>
<p><strong>Prediction:</strong> {{ prediction['ohlc'] or "Waiting..." }}</p>
<p><strong>Signal:</strong> {{ prediction['signal'] }}</p>
<p><a href="/logout">Logout</a> | <a href="/download-model">Download Model</a></p>
<h3>Upload Script</h3>
<form method="POST" enctype="multipart/form-data" action="/upload">
    <input type="file" name="script" required>
    <select name="signal"><option value="CALL">CALL</option><option value="PUT">PUT</option></select>
    <input type="submit" value="Upload & Enable">
</form>
<h3>Enabled Scripts</h3>
<table border="1">
    <tr><th>Signal</th><th>Script</th><th>Action</th></tr>
    {% for sig, script in enabled_scripts.items() %}
    <tr>
        <td>{{ sig }}</td>
        <td>{{ script or "None" }}</td>
        <td>
            {% if script %}
                <a href="/disable/{{ sig }}">Disable</a> |
                <a href="/delete/{{ sig }}">Delete</a>
            {% endif %}
        </td>
    </tr>
    {% endfor %}
</table>
'''

LOGIN_TEMPLATE = '''
<h2>Login</h2>
<form method="POST">
    <input name="username" placeholder="Username" required><br><br>
    <input type="password" name="password" placeholder="Password" required><br><br>
    <input type="submit" value="Login">
</form>
<a href="/register">Register</a> | <a href="/change-password">Change Password</a>
'''

REGISTER_TEMPLATE = '''
<h2>Register</h2>
<form method="POST">
    <input name="username" placeholder="Username" required><br><br>
    <input type="password" name="password" placeholder="Password" required><br><br>
    <input type="submit" value="Register">
</form>
<a href="/">Login</a>
'''

CHANGE_PASSWORD_TEMPLATE = '''
<h2>Change Password</h2>
<form method="POST">
    <input name="username" placeholder="Username" required><br><br>
    <input type="password" name="old_password" placeholder="Old Password" required><br><br>
    <input type="password" name="new_password" placeholder="New Password" required><br><br>
    <input type="submit" value="Change">
</form>
<a href="/">Back</a>
'''

# === ROUTES ===
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        pwd = request.form["password"]
        with sqlite3.connect("users.db") as conn:
            row = conn.execute("SELECT password FROM users WHERE username=?", (user,)).fetchone()
        if row and check_password_hash(row[0], pwd):
            session["logged_in"] = True
            return redirect(url_for("admin_panel"))
        return "❌ Invalid credentials"
    return render_template_string(LOGIN_TEMPLATE)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user = request.form["username"]
        pwd = generate_password_hash(request.form["password"])
        try:
            with sqlite3.connect("users.db") as conn:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user, pwd))
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return "❌ Username already exists"
    return render_template_string(REGISTER_TEMPLATE)

@app.route("/change-password", methods=["GET", "POST"])
def change_password():
    if request.method == "POST":
        user = request.form["username"]
        old = request.form["old_password"]
        new = request.form["new_password"]
        with sqlite3.connect("users.db") as conn:
            row = conn.execute("SELECT password FROM users WHERE username=?", (user,)).fetchone()
            if row and check_password_hash(row[0], old):
                conn.execute("UPDATE users SET password=? WHERE username=?", (generate_password_hash(new), user))
                return redirect(url_for("login"))
            return "❌ Incorrect old password"
    return render_template_string(CHANGE_PASSWORD_TEMPLATE)

@app.route("/admin")
def admin_panel():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template_string(ADMIN_TEMPLATE, prediction=prediction, enabled_scripts=enabled_scripts)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/upload", methods=["POST"])
def upload():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    file = request.files["script"]
    signal = request.form["signal"]
    filename = f"{signal.lower()}_{file.filename}"
    file.save(os.path.join(UPLOAD_FOLDER, filename))
    enabled_scripts[signal] = filename
    return redirect(url_for("admin_panel"))

@app.route("/disable/<signal>")
def disable(signal):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    enabled_scripts[signal] = None
    return redirect(url_for("admin_panel"))

@app.route("/delete/<signal>")
def delete(signal):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    fname = enabled_scripts.get(signal)
    if fname:
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, fname))
        except:
            pass
    enabled_scripts[signal] = None
    return redirect(url_for("admin_panel"))

@app.route("/download-model")
def download_model():
    if not os.path.exists(MODEL_PATH):
        return "Model not found."
    directory, filename = os.path.split(MODEL_PATH)
    return send_from_directory(directory, filename, as_attachment=True)

# === LAUNCH ===
if __name__ == "__main__":
    print("🔥 EUR/USD Forecast App Starting...")
    app.run(debug=False, port=5000)

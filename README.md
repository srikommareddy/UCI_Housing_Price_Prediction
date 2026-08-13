# 🏠 Real Estate Valuation — EDA & Linear Regression (Streamlit App)

A beginner-friendly, interactive Streamlit app for exploring the **UCI Real
Estate Valuation** dataset and training a scikit-learn **Linear Regression**
model — no coding required to use it.

Built as a companion to a Jupyter/Colab notebook covering the same workflow
(EDA → cleaning → linear regression → evaluation), reimagined as a deployable
web app for students to explore interactively.

## Live features

- **Overview tab** — dataset preview, column descriptions, basic stats
- **EDA tab** — summary statistics, missing values, target/feature
  distributions, correlation heatmap, scatter plots
- **Model & Evaluation tab** — trains a Linear Regression model live,
  shows coefficients, R² / RMSE / MAE, predicted-vs-actual plot, and a
  residual plot
- **Try a Prediction tab** — interactive sliders to predict a price for a
  hypothetical property
- Sidebar controls to: upload your own CSV instead of the bundled dataset,
  change the train/test split size and random state, and choose which
  columns to drop before modeling

## Dataset

The bundled dataset (`data/real_estate_valuation.csv`) is the **UCI Real
Estate Valuation Data Set** (Sindian District, New Taipei City, Taiwan):

| Column | Description |
|---|---|
| `No` | Row ID (dropped by default — not predictive) |
| `X1 transaction date` | Sale date, decimal-year format |
| `X2 house age` | Age of the house in years |
| `X3 distance to the nearest MRT station` | Distance in meters |
| `X4 number of convenience stores` | Nearby convenience stores |
| `X5 latitude` | Geographic latitude |
| `X6 longitude` | Geographic longitude |
| `Y house price of unit area` | **Target** — price per unit area |

You can also upload your own CSV with a similar structure from the sidebar.

## Project structure

```
.
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── data/
│   └── real_estate_valuation.csv   # Bundled sample dataset
└── README.md
```

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this project to a **public GitHub repository** (structure above,
   with `app.py` in the repo root).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, then select:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. Streamlit Cloud will read `requirements.txt`
   automatically and install everything.
5. Your app will be live at a URL like:
   `https://<your-app-name>.streamlit.app`

That's it — no server setup needed.

## Notes for instructors

- Students can compare their own manual EDA/model results against this
  app's outputs by uploading the same dataset and matching the
  **random state** and **test size** settings in the sidebar.
- The "columns to drop" control lets students test how removing features
  like `X1 transaction date` affects R² / RMSE, without editing code.
- This is a teaching tool, not a production valuation model.

## Credits

Dataset: Yeh, I. C., & Hsu, T. K. (2018). *Building real estate valuation
models with comparative approach through case-based reasoning.* UCI Machine
Learning Repository.

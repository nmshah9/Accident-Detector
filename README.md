# Accident Detection from CCTV Footage — Complete Project

One project, one environment, one `requirements.txt`. Everything is flat at the project
root so it drops straight into VS Code.

```
accident-detection-project/
├── requirements.txt                # ONE file — covers the notebook AND the app
├── Accident_Detection_CCTV.ipynb   # EDA + 3 models (scratch CNN, transfer learning, +augmentation) + comparison
├── app.py                          # Streamlit app (run this directly)
├── src/
│   └── model_utils.py              # preprocessing, prediction, Grad-CAM — used by app.py
├── models/
│   └── accident_mobilenetv2.weights.h5  # trained weights — same model evaluated as "Model 2" in the notebook
├── data/
│   ├── train/{Accident, Non Accident}/
│   ├── val/{Accident, Non Accident}/
│   └── test/{Accident, Non Accident}/
└── sample_test_images/
    ├── Accident/            (3 sample frames to try the app with)
    └── Non Accident/
```

## Setup (one time)

Open this folder in VS Code, open a terminal in it, and run:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

That single `requirements.txt` installs everything needed for **both** the notebook and
the Streamlit app — no separate environments.

## Run the notebook

In VS Code: open `Accident_Detection_CCTV.ipynb`, select the `venv` you just created as
the kernel (top-right kernel picker → Python Environments → venv), then **Run All**.

- It will re-train all 3 models on the included `data/` folder and reproduce the results
  documented in the notebook (Scratch CNN ~53% test accuracy, MobileNetV2 Transfer
  Learning ~92%, Transfer Learning + Augmentation ~73%).
- On a normal laptop CPU this takes roughly **15–20 minutes** end to end (the scratch CNN
  is the slowest part). No GPU required.
- The notebook already contains the original run's outputs/plots too, so you can read
  through it immediately without re-running if you just want to review the analysis.

## Run the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload any image from `sample_test_images/` to try it
immediately, or use your own CCTV frame.

This uses the **already-trained** weights in
`models/accident_mobilenetv2.weights.h5` — you do **not** need to run the notebook
first. If you re-run the notebook and want the app to reflect a newly retrained
model, add `model.save_weights("../models/accident_mobilenetv2.weights.h5")`
(weights-only, matching what `src/model_utils.py` loads) after training Model 2.

## Troubleshooting

**Already fixed, no action needed:** the app previously loaded a full serialized
`.keras` model file, which broke on any Keras version older than the one the model
was saved with (`GlorotUniform.__init__() got an unexpected keyword argument
'input_axes'` is that failure). This is now fixed properly at the source — the app
rebuilds the model **architecture from code** and loads only the raw trained
**weights** (`models/accident_mobilenetv2.weights.h5`), which sidesteps Keras
version-compatibility issues entirely.

This was verified, not assumed: I reproduced the original error with an old
Keras version, confirmed the new weights-only approach loads and predicts
correctly in that same old environment, and confirmed the app boots (`HTTP 200`)
there too — as well as with current TensorFlow/Keras. So Python 3.12 (or your
current version) should work regardless; you no longer need an exact version pin.

## Notes

- Verified before delivery: `requirements.txt` was installed into a **clean virtual
  environment** (pip resolved TensorFlow to 2.21.0, matching what the model was saved
  with), and `streamlit run app.py` was booted from this exact folder layout and
  confirmed to serve correctly (`HTTP 200`, no errors).
- This is a demo/research model trained on ~890 images — treat predictions as indicative,
  not as a validated safety system.
- For deploying the app publicly (Streamlit Community Cloud), push this whole folder to a
  GitHub repo and point Streamlit Cloud at `app.py`. If the free tier's 1GB RAM is tight
  with `tensorflow-cpu`, converting the model to `.tflite` is the usual fix — ask if you
  want that done.

# Census Income Classification API

Submission repository URL: **Pending — replace with your public fork URL before submission.**

This project trains a Census income random forest, evaluates held-out categorical slices, and serves predictions through FastAPI. All 12 tests passed locally, and flake8 reported no errors. Actual metrics are in model/metrics.json and model_card.md.

## Run (Python 3.12)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python -m pytest -v
python -m flake8 .
python -m uvicorn main:app --reload
```

In a second terminal with the same environment, run `python local_api.py`. GET / returns a greeting and POST /predict returns a JSON prediction of <=50K or >50K. Interactive API documentation is at http://127.0.0.1:8000/docs. Hyphenated feature names follow the CSV schema. Unknown categories are supported; missing fields and invalid numeric ranges return HTTP 422.

## Files and evidence

The data, trained model, categorical encoder, label binarizer, metrics, model card, complete categorical slice report, tests, and Python CI workflow are included. The workflow runs pytest and flake8 on pushes and pull requests to main/master using Python 3.12. unit_test_output.txt and local_api_output.txt contain actual execution logs. The screenshot PNGs are still pending. Run the commands in your terminal and capture screenshots/unit_test.png and screenshots/local_api.png. Capture screenshots/continuous_integration.png only after the genuine GitHub workflow passes.

## Finish GitHub submission

1. Fork https://github.com/udacity/Deploying-a-Scalable-ML-Pipeline-with-FastAPI and ensure your fork is public.
2. Copy this project's files into your cloned fork, including .github/workflows/ci.yml and .flake8. Remove the starter manual.yml workflow if still present.
3. Replace the pending submission URL above with the actual public repository link.
4. Commit and push the files to main or master, then open the Actions tab and verify Python CI passes.
5. Capture the genuine passing GitHub Actions page as screenshots/continuous_integration.png, commit it, and push again. This screenshot is deliberately absent until an actual remote workflow passes.
6. Submit the public repository URL and put the same URL in Udacity's Submission Details box.

Do not submit this ZIP alone: the rubric requires a public GitHub repository URL. The rubric's GitHub publication and CI screenshot remain pending.

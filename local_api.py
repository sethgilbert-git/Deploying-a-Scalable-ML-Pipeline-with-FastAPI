"""Exercise GET and POST against a running local Uvicorn server."""
import argparse

import requests

EXAMPLE = {
    "age": 39, "workclass": "State-gov", "fnlgt": 77516,
    "education": "Bachelors", "education-num": 13,
    "marital-status": "Never-married", "occupation": "Adm-clerical",
    "relationship": "Not-in-family", "race": "White", "sex": "Male",
    "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
    "native-country": "United-States",
}


def main():
    """Print actual HTTP statuses and JSON for both required methods."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    url = parser.parse_args().url.rstrip("/")
    for method, path, payload in [("GET", "/", None),
                                  ("POST", "/predict", EXAMPLE)]:
        response = requests.request(method, url + path, json=payload, timeout=10)
        print(f"{method} {path} | Status Code: {response.status_code}")
        print(f"Result: {response.json()}")
        response.raise_for_status()


if __name__ == "__main__":
    main()

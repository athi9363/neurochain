## Week 1 — Quickstart Checklist

Use this checklist to get the project compiled, tested and the toy models trained locally.

- [ ] Install Node.js (recommended v18+) and npm
- [ ] Install Python 3.10 and create a virtual environment
- [ ] Install repo dev dependencies:
  - npm install
  - .venv\Scripts\python.exe -m pip install -r requirements.txt
- [ ] Compile and test contracts
  - npx hardhat compile
  - npx hardhat test
- [ ] Deploy locally (hardhat local network)
  - npx hardhat node
  - npx hardhat run scripts/deploy.js --network localhost
- [ ] Train and validate models
  - python models/train_image.py
  - python models/test_onnx.py
- [ ] Create and commit model artifacts only if small (or push to release storage)
- [ ] Add `.venv/` and data folders (e.g. `MNIST/`) to `.gitignore`

Notes
- If you need GPU builds for PyTorch, follow the official PyTorch installer: https://pytorch.org/
- Use `python -m pip freeze > requirements.txt` in the venv to pin exact package versions.

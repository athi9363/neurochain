import { ethers } from "https://cdn.jsdelivr.net/npm/ethers@6.15.0/dist/ethers.esm.min.js";

const connectBtn = document.getElementById("connect");
const accountDiv = document.getElementById("account");
const loadTokenBtn = document.getElementById("loadToken");
const tokenUriDiv = document.getElementById("tokenUri");
const contractAddressInput = document.getElementById("contractAddress");
const tokenIdInput = document.getElementById("tokenId");
const loadModelBtn = document.getElementById("loadModel");
const modelStatus = document.getElementById("modelStatus");
const runInferenceBtn = document.getElementById("runInference");
const inferenceResult = document.getElementById("inferenceResult");
const recordUsageBtn = document.getElementById("recordUsage");
const txStatus = document.getElementById("txStatus");

let provider, signer, contract, ortSession;

const ABI = [
  "function tokenURI(uint256) view returns (string)",
  "function recordUsage(uint256)",
];

connectBtn.onclick = async () => {
  if (!window.ethereum) return alert("Install MetaMask");
  provider = new ethers.BrowserProvider(window.ethereum);
  await provider.send("eth_requestAccounts", []);
  signer = await provider.getSigner();
  const addr = await signer.getAddress();
  accountDiv.textContent = addr;
};

loadTokenBtn.onclick = async () => {
  const addr = contractAddressInput.value.trim();
  const id = parseInt(tokenIdInput.value || "0");
  if (!addr) return alert("enter contract address");
  if (!provider) provider = ethers.getDefaultProvider();
  contract = new ethers.Contract(addr, ABI, provider);
  try {
    const uri = await contract.tokenURI(id);
    tokenUriDiv.textContent = uri;
  } catch (e) {
    tokenUriDiv.textContent = "Error: " + e;
  }
};

loadModelBtn.onclick = async () => {
  let uri = tokenUriDiv.textContent || '';
  if (!uri || uri.startsWith('Error')) uri = '../models/tiny_mnist.onnx';
  modelStatus.textContent = 'Loading model from ' + uri;
  try {
    ortSession = await ort.InferenceSession.create(uri);
    modelStatus.textContent = 'Model loaded';
  } catch (e) {
    modelStatus.textContent = 'Load error: ' + e;
  }
};

runInferenceBtn.onclick = async () => {
  if (!ortSession) return alert('Load model first');
  // random input 1x1x28x28
  const input = new Float32Array(1 * 1 * 28 * 28).map(() => Math.random());
  const tensor = new ort.Tensor('float32', input, [1,1,28,28]);
  try {
    const out = await ortSession.run({ input: tensor });
    const key = Object.keys(out)[0];
    inferenceResult.textContent = JSON.stringify(out[key].data.slice(0,10));
  } catch (e) {
    inferenceResult.textContent = 'Run error: ' + e;
  }
};

recordUsageBtn.onclick = async () => {
  if (!signer) return alert('Connect wallet first');
  const addr = contractAddressInput.value.trim();
  const id = parseInt(tokenIdInput.value || '0');
  contract = new ethers.Contract(addr, ABI, signer);
  try {
    const tx = await contract.recordUsage(id);
    txStatus.textContent = 'Sent tx: ' + tx.hash;
    await tx.wait();
    txStatus.textContent = 'Tx mined: ' + tx.hash;
  } catch (e) {
    txStatus.textContent = 'Error: ' + e;
  }
};

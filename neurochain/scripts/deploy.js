const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  const BrainNFT = await hre.ethers.getContractFactory("BrainNFT");
  const brain = await BrainNFT.deploy();

  // Support both ethers v6 and v5:
  if (typeof brain.waitForDeployment === "function") {
    // ethers v6
    await brain.waitForDeployment();
  } else if (typeof brain.deployed === "function") {
    // ethers v5
    await brain.deployed();
  } else {
    // fallback: mine a block (Hardhat) or small delay
    await hre.network.provider.send("evm_mine");
  }

  console.log("BrainNFT deployed:", brain.target ?? brain.address);

  // Mint a token (call returns a tx in v5/v6)
  const tx = await brain.mintBrain(deployer.address, "ipfs://placeholderCID");

  // Wait for transaction to be mined:
  if (tx.wait) {
    // ethers v5/v6 tx.wait exists
    await tx.wait();
  }

  // Read tokenURI
  const tokenUri = await brain.tokenURI(0);
  console.log("Token 0 URI:", tokenUri);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

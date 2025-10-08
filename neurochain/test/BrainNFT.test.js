const { expect } = require("chai");
const { ethers } = require("hardhat");
// anyValue helper for event arg matching
const { anyValue } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");

describe("BrainNFT", function () {
  let BrainNFT, brain, owner, addr1;

  beforeEach(async function () {
    BrainNFT = await ethers.getContractFactory("BrainNFT");
    [owner, addr1] = await ethers.getSigners();
  brain = await BrainNFT.deploy();
  await brain.waitForDeployment();
  });

  it("mints and returns tokenURI", async function () {
    const tokenURI = "ipfs://testCID/manifest.json";
    await brain.mintBrain(owner.address, tokenURI);
    expect(await brain.tokenURI(0)).to.equal(tokenURI);
  });

  it("emits ModelUsed event on recordUsage", async function () {
    await brain.mintBrain(owner.address, "ipfs://x");
    await expect(brain.connect(addr1).recordUsage(0))
      .to.emit(brain, 'ModelUsed')
      .withArgs(0, addr1.address, anyValue); // anyValue is provided by waffle/matchers
  });
});

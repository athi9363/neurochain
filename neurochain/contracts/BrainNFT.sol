// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract BrainNFT is ERC721, Ownable {
    uint256 public nextId;
    mapping(uint256 => string) private _tokenURIs;

    event ModelUsed(uint256 indexed tokenId, address indexed user, uint256 timestamp);

    constructor() ERC721("NeuroChainBrain", "NBRAIN") {}

    function mintBrain(address to, string memory tokenURI_) external onlyOwner returns (uint256) {
        uint256 id = nextId++;
        _safeMint(to, id);
        _tokenURIs[id] = tokenURI_;
        return id;
    }

    function tokenURI(uint256 tokenId) public view override returns (string memory) {
        require(_exists(tokenId), "no token");
        return _tokenURIs[tokenId];
    }

    function recordUsage(uint256 tokenId) external {
        require(_exists(tokenId), "no token");
        emit ModelUsed(tokenId, msg.sender, block.timestamp);
    }
}

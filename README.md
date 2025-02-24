Distributed AI Mining Software

🏗 Overview

This software is designed for decentralized AI training using GPU-based matrix multiplication. By participating in the network, users can contribute their GPU power to assist in large-scale AI model training while earning rewards for their computations.

✨ Features

🚀 Decentralized Network: Uses blockchain technology to distribute and verify computations.

🧮 Matrix Multiplication Optimization: Efficiently performs large-scale matrix operations for deep learning models.

🎮 GPU Acceleration: Fully utilizes GPU resources for high-performance computing.

💰 Reward System: Users receive tokens based on their contribution to AI model training.

🔍 Fault Tolerance: Ensures computational integrity with error detection and redundancy mechanisms.

📌 Requirements

Operating System: Windows (Linux and macOS support coming soon)

GPU: NVIDIA with CUDA support (AMD and other platforms will be supported in future updates)

RAM: Minimum 8GB recommended

Software Dependencies:

.NET 8.0+

CUDA Toolkit (for NVIDIA GPUs)

🛠 Installation

# Clone the Repository
git clone [https://github.com/dait-dev/DAITMiner.git](https://github.com/dait-dev/DAITMiner.git)
or
download DAITCore.zip

cd bin/Debug/net8.0
DAITCore.exe

# Install Dependencies
https://dotnet.microsoft.com/en-us/download/dotnet/8.0

# Run the Miner
DAITCore.exe

⚙ Configuration

Edit the PubKey.txt file to set up:

Wallet Address: Where mining rewards will be sent.

(later) GPU Settings: Specify GPU usage limits.

(later) Network Node: Connect to the distributed AI training network.

🔄 How It Works

The miner receives AI training tasks from the network.

It performs large-scale matrix multiplications using GPU power.

Computation results are validated and submitted to the blockchain.

Users receive rewards based on their contribution.

🔮 Future Development

✅ Support for AMD GPUs and OpenCL

✅ Cross-platform compatibility (Linux, macOS)

✅ Enhanced network optimizations and security features

✅ Support for other types of algorithms

✅ Network traffic optimization

✅ Publication of the source code and interaction mechanism for the network nodes


🤝 Contributing

Contributions are welcome! Submit a pull request or report issues on GitHub.

📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

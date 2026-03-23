# 📑 Tiny ImageNet Portfolio: Setup & Infrastructure Guide
**Project Stage:** Phase 1 - Environment & Baseline
**Hardware:** Ubuntu 20.04 (Laptop) -> Windows 11 (Remote GPU Desktop)
**Stack:** Keras 3 (Multi-backend), TensorFlow, PyTorch, Miniconda

---

## 🛠 Step 1: VS Code (The Command Center)
Install the official Microsoft repository to enable Remote Tunnels.

```bash
# Update and install dependencies
sudo apt update && sudo apt install software-properties-common apt-transport-https wget -y

# Import GPG key and add repository
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"

# Install VS Code
sudo apt update && sudo apt install code -y
```

---

## 🛠 Step 2: Miniconda Installation
Miniconda is preferred over venv for AI projects because it manages non-Python dependencies (like CUDA/cuDNN) much more reliably across different OS (Linux vs Windows).

```bash
# Download Linux installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run installer (Follow prompts, type 'yes' to initialize)
bash Miniconda3-latest-Linux-x86_64.sh

# RESTART YOUR TERMINAL NOW
```

---

## 🛠 Step 3: Environment Setup
We will create a 'mirrored' environment.

```bash
# Create the environment
conda create -n vision_portfolio python=3.10 -y
conda activate vision_portfolio

# Install Keras 3 and both backends
pip install --upgrade pip
pip install tensorflow keras torch torchvision matplotlib numpy pandas tqdm
```

---

## 🛠 Step 4: Project Structure & Data Fix
1. **Create Folders:**
   ```bash
   mkdir -p ~/TinyImageNet_Project/{data,models,notebooks,src}
   cd ~/TinyImageNet_Project
   touch src/data_loader.py src/models.py src/train.py README.md .gitignore
   ```

2. **Download Tiny ImageNet:**
   ```bash
   cd data
   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
   unzip tiny-imagenet-200.zip
   rm tiny-imagenet-200.zip
   ```

3. **The Validation Script (Run this in src/data_loader.py):**
   Paste the reorganization logic provided in the chat to move validation images into class-named subfolders.

---

## 🛠 Step 5: Remote Bridge (For Day 8)
When you reach your Windows Desktop:
1. Open PowerShell: `code tunnel`
2. On Laptop VS Code: Install "Remote - Tunnels" extension.
3. Click Blue Icon (Bottom Left) -> Connect to Tunnel.

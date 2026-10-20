## 1. Prerequisites (Ubuntu 24.04 — the XRT debs are `_24.04`)

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv libboost-filesystem1.74.0 dkms
sudo apt install libportaudio2     # only needed for live_transcribe.py (sounddevice)
sudo apt install ffmpeg            # optional, for converting non-WAV input
```

## 2. NPU driver + XRT

Download `RAI_1.8_Linux_NPU_XRT.zip`: https://download.amd.com/opendownload/RyzenAI/Driver/RAI_1.8_Linux_NPU_XRT.zip

unzip RAI_1.8_Linux_NPU_XRT.zip && cd <extracted-dir>
sudo apt install --fix-broken -y ./xrt_202620.2.25.37_24.04-amd64-base.deb
sudo apt install --fix-broken -y ./xrt_202620.2.25.37_24.04-amd64-base-dev.deb
sudo apt install --fix-broken -y ./xrt_202620.2.25.37_24.04-amd64-npu.deb
sudo apt install --fix-broken -y ./xrt_plugin.2.25.260102.56.release_24.04-amd64-amdxdna.deb

source /opt/xilinx/xrt/setup.sh
xrt-smi examine        # should list an "NPU Strix"-type device (aie2p)
Reboot if `xrt-smi examine` shows no device.

## 3. Ryzen AI 1.8.0 software

Download `ryzen_ai-1.8.0.tgz` (EULA-gated): https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=ryzen_ai-1.8.0.tgz

mkdir ryzen_ai-1.8.0 && cp ryzen_ai-1.8.0.tgz ryzen_ai-1.8.0 && cd ryzen_ai-1.8.0
tar -xvzf ryzen_ai-1.8.0.tgz
./install_ryzen_ai.sh -a yes -p ~/ryzenai/venv
source ~/ryzenai/venv/bin/activate
source /opt/xilinx/xrt/setup.sh
echo $RYZEN_AI_INSTALLATION_PATH
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/lib/:$LD_LIBRARY_PATH

# sanity check
cd ~/ryzenai/venv/quicktest && python quicktest.py     # expect "Test Finished"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# expect 'VitisAIExecutionProvider' in the list (DmlExecutionProvider will NOT appear on Linux)
```

Do not `pip install onnxruntime` into this venv — it would overwrite the VitisAI-enabled build.
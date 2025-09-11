2025-09-11 06:14:56,176 - realtime_streaming - INFO - Starting Voxtral Conversational Streaming UI Server with VAD
INFO:     Started server process [3033]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
🔗 Starting TCP streaming server on port 8766...
📋 Note: Model initialization optimized for conversation with VAD

🔍 Verifying service startup...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Installing netcat for service checking...
Hit:1 http://security.ubuntu.com/ubuntu jammy-security InRelease
Hit:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
Hit:3 http://archive.ubuntu.com/ubuntu jammy InRelease
Hit:4 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease
Hit:5 http://archive.ubuntu.com/ubuntu jammy-updates InRelease
Hit:6 http://archive.ubuntu.com/ubuntu jammy-backports InRelease
Reading package lists... Done
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following NEW packages will be installed:
  netcat-openbsd
0 upgraded, 1 newly installed, 0 to remove and 99 not upgraded.
Need to get 39.4 kB of archives.
After this operation, 109 kB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy/main amd64 netcat-openbsd amd64 1.218-4ubuntu1 [39.4 kB]
Fetched 39.4 kB in 0s (218 kB/s)
debconf: delaying package configuration, since apt-utils is not installed
Selecting previously unselected package netcat-openbsd.
(Reading database ... 26968 files and directories currently installed.)
Preparing to unpack .../netcat-openbsd_1.218-4ubuntu1_amd64.deb ...
Unpacking netcat-openbsd (1.218-4ubuntu1) ...
Setting up netcat-openbsd (1.218-4ubuntu1) ...
update-alternatives: using /bin/nc.openbsd to provide /bin/nc (nc) in auto mode
update-alternatives: warning: skip creation of /usr/share/man/man1/nc.1.gz because associated file /usr/share/man/man1/nc_openbsd.1.gz (of link group nc) doesn't exist
update-alternatives: warning: skip creation of /usr/share/man/man1/netcat.1.gz because associated file /usr/share/man/man1/nc_openbsd.1.gz (of link group nc) doesn't exist
✅ Health Check Server is running on port 8005
✅ Conversational UI Server is running on port 8000
⏳ Waiting for TCP Streaming Server on port 8766...
2025-09-11 06:15:05,490 - voxtral_realtime - INFO - VoxtralModel initialized for cuda with torch.bfloat16
2025-09-11 06:15:05,628 - root - INFO - Successfully imported models for TCP server
2025-09-11 06:15:05,629 - root - INFO - TCP server configured for 0.0.0.0:8766
2025-09-11 06:15:05,630 - root - INFO - 🚀 Starting TCP server on 0.0.0.0:8766
2025-09-11 06:15:05,639 - realtime_audio - INFO - 🔊 AudioProcessor initialized for PRODUCTION real-time streaming:
2025-09-11 06:15:05,640 - realtime_audio - INFO -    📊 Sample rate: 16000 Hz
2025-09-11 06:15:05,641 - realtime_audio - INFO -    🎵 Mel bins: 128
2025-09-11 06:15:05,641 - realtime_audio - INFO -    📐 FFT size: 1024
2025-09-11 06:15:05,642 - realtime_audio - INFO -    ⏱️  Hop length: 160
2025-09-11 06:15:05,642 - realtime_audio - INFO -    🪟 Window length: 400
2025-09-11 06:15:05,643 - realtime_audio - INFO -    🎙️  VAD threshold: 0.015
2025-09-11 06:15:05,643 - realtime_audio - INFO -    🔇 Energy threshold: 3e-06
2025-09-11 06:15:05,644 - root - INFO - ✅ TCP server audio processor initialized with VAD
2025-09-11 06:15:05,644 - root - INFO - 🚀 Initializing Voxtral model for TCP server...
2025-09-11 06:15:05,645 - voxtral_realtime - INFO - 🚀 Starting Voxtral model initialization for conversational streaming...
2025-09-11 06:15:05,645 - voxtral_realtime - INFO - 📥 Loading AutoProcessor from mistralai/Voxtral-Mini-3B-2507
preprocessor_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 357/357 [00:00<00:00, 2.06MB/s]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.41it/s]
Fetching 1 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 584.73it/s]
tekken.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 14.9M/14.9M [00:00<00:00, 15.8MB/s]
2025-09-11 06:15:08,903 - mistral_common.tokens.tokenizers.tekken - INFO - Vocab size: 150000
2025-09-11 06:15:08,908 - mistral_common.tokens.tokenizers.tekken - INFO - Cutting vocab to first 130072 tokens.
2025-09-11 06:15:10,519 - mistral_common.tokens.tokenizers.tekken - INFO - Vocab size: 150000
2025-09-11 06:15:10,521 - mistral_common.tokens.tokenizers.tekken - INFO - Cutting vocab to first 130072 tokens.
2025-09-11 06:15:11,029 - voxtral_realtime - INFO - ✅ AutoProcessor loaded successfully
2025-09-11 06:15:11,033 - voxtral_realtime - INFO - 💡 FlashAttention2 not installed, using eager attention
2025-09-11 06:15:11,034 - voxtral_realtime - INFO - 💡 To install: pip install flash-attn --no-build-isolation
2025-09-11 06:15:11,034 - voxtral_realtime - INFO - 🔧 Using attention implementation: eager
2025-09-11 06:15:11,035 - voxtral_realtime - INFO - 📥 Loading Voxtral model from mistralai/Voxtral-Mini-3B-2507
`torch_dtype` is deprecated! Use `dtype` instead!
config.json: 1.36kB [00:00, 4.29MB/s]
model.safetensors.index.json: 68.1kB [00:00, 57.2MB/s]
Fetching 2 files:   0%|                                                                                                                              | 0/2 [00:00<?, ?it/s]INFO:     100.64.0.28:33962 - "GET / HTTP/1.1" 200 OK

INFO:     100.64.0.25:48788 - "GET /favicon.ico HTTP/1.1" 404 Not Found                                                                 | 613M/4.98G [00:03<00:11, 375MB/s]
model-00001-of-00002.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 4.98G/4.98G [00:19<00:00, 261MB/s]
model-00002-of-00002.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 4.38G/4.38G [00:19<00:00, 223MB/s]
Fetching 2 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:20<00:00, 10.03s/it]
2025-09-11 06:15:32,615 - accelerate.utils.modeling - INFO - We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You canset `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards:  50%|██████████████████████████████████████████████████████▌                                                      | 1/2 [00:01<00:01,  1.72s/it]❌ TCP Streaming Server failed to start on port 8766 after 30 seconds

🛑 Shutting down Conversational Streaming Server...
root@1ead7f6fcad2:/workspace/Voxtral-Final# INFO:     Shutting down
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [3033]
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [3019]

root@1ead7f6fcad2:/workspace/Voxtral-Final#
root@1ead7f6fcad2:/workspace/Voxtral-Final# ./run_realtime.sh
=== Starting Voxtral CONVERSATIONAL Streaming Server (PRODUCTION FIXED) ===
🚀 Version 3.0 - Production Ready with VAD and Silence Detection

🧹 Cleaning up existing processes...
=== Cleaning up existing Voxtral processes ===
Killing existing processes by name pattern...
Killing processes using specific ports...
Checking port 8000 for UI Server...
No processes found using port 8000
Checking port 8005 for Health Check Server...
No processes found using port 8005
Checking port 8766 for TCP Server...
No processes found using port 8766
Checking port 8765 for Old WebSocket Server...
No processes found using port 8765
Cleaning up any remaining uvicorn processes...
Final port availability check...
✅ Port 8000 is available
✅ Port 8005 is available
✅ Port 8766 is available

✅ Cleanup completed!
Available ports:
  - 8000: UI Server + WebSocket
  - 8005: Health Check
  - 8766: TCP Server

You can now run ./run.sh to start the servers
🔧 Environment variables and Python path set for conversational performance
📁 PYTHONPATH: /workspace/Voxtral-Final:
📁 Current directory: /workspace/Voxtral-Final
🔍 Checking FlashAttention2 availability...
💡 FlashAttention2 not detected - using eager attention (still fast!)
📝 Note: This is normal and the system will work perfectly.
🩺 Starting health check server on port 8005...
⏳ Waiting for health server to initialize...
INFO:     Started server process [3817]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8005 (Press CTRL+C to quit)
🌐 Starting CONVERSATIONAL UI Server on port 8000...
📋 Using optimized conversational streaming components with VAD
💡 Using eager attention - performance is still excellent
⏳ Waiting for UI server to start...
2025-09-11 06:16:37,265 - realtime_streaming - INFO - Starting Voxtral Conversational Streaming UI Server with VAD
INFO:     Started server process [3831]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
🔗 Starting TCP streaming server on port 8766...
📋 Note: Model initialization optimized for conversation with VAD

🔍 Verifying service startup...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Health Check Server is running on port 8005
✅ Conversational UI Server is running on port 8000
⏳ Waiting for TCP Streaming Server on port 8766...
2025-09-11 06:16:46,545 - voxtral_realtime - INFO - VoxtralModel initialized for cuda with torch.bfloat16
2025-09-11 06:16:46,645 - root - INFO - Successfully imported models for TCP server
2025-09-11 06:16:46,647 - root - INFO - TCP server configured for 0.0.0.0:8766
2025-09-11 06:16:46,648 - root - INFO - 🚀 Starting TCP server on 0.0.0.0:8766
2025-09-11 06:16:46,651 - realtime_audio - INFO - 🔊 AudioProcessor initialized for PRODUCTION real-time streaming:
2025-09-11 06:16:46,652 - realtime_audio - INFO -    📊 Sample rate: 16000 Hz
2025-09-11 06:16:46,653 - realtime_audio - INFO -    🎵 Mel bins: 128
2025-09-11 06:16:46,653 - realtime_audio - INFO -    📐 FFT size: 1024
2025-09-11 06:16:46,654 - realtime_audio - INFO -    ⏱️  Hop length: 160
2025-09-11 06:16:46,654 - realtime_audio - INFO -    🪟 Window length: 400
2025-09-11 06:16:46,655 - realtime_audio - INFO -    🎙️  VAD threshold: 0.015
2025-09-11 06:16:46,656 - realtime_audio - INFO -    🔇 Energy threshold: 3e-06
2025-09-11 06:16:46,656 - root - INFO - ✅ TCP server audio processor initialized with VAD
2025-09-11 06:16:46,657 - root - INFO - 🚀 Initializing Voxtral model for TCP server...
2025-09-11 06:16:46,657 - voxtral_realtime - INFO - 🚀 Starting Voxtral model initialization for conversational streaming...
2025-09-11 06:16:46,658 - voxtral_realtime - INFO - 📥 Loading AutoProcessor from mistralai/Voxtral-Mini-3B-2507
Fetching 1 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 432.85it/s]
Fetching 1 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 730.46it/s]
2025-09-11 06:16:48,399 - mistral_common.tokens.tokenizers.tekken - INFO - Vocab size: 150000
2025-09-11 06:16:48,423 - mistral_common.tokens.tokenizers.tekken - INFO - Cutting vocab to first 130072 tokens.
2025-09-11 06:16:50,051 - mistral_common.tokens.tokenizers.tekken - INFO - Vocab size: 150000
2025-09-11 06:16:50,053 - mistral_common.tokens.tokenizers.tekken - INFO - Cutting vocab to first 130072 tokens.
2025-09-11 06:16:50,559 - voxtral_realtime - INFO - ✅ AutoProcessor loaded successfully
2025-09-11 06:16:50,563 - voxtral_realtime - INFO - 💡 FlashAttention2 not installed, using eager attention
2025-09-11 06:16:50,564 - voxtral_realtime - INFO - 💡 To install: pip install flash-attn --no-build-isolation
2025-09-11 06:16:50,564 - voxtral_realtime - INFO - 🔧 Using attention implementation: eager
2025-09-11 06:16:50,565 - voxtral_realtime - INFO - 📥 Loading Voxtral model from mistralai/Voxtral-Mini-3B-2507
`torch_dtype` is deprecated! Use `dtype` instead!
2025-09-11 06:16:51,324 - accelerate.utils.modeling - INFO - We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You canset `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:18<00:00,  9.26s/it]
generation_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 108/108 [00:00<00:00, 431kB/s]
2025-09-11 06:17:10,421 - voxtral_realtime - INFO - ✅ Voxtral model loaded successfully with eager attention
2025-09-11 06:17:10,425 - voxtral_realtime - INFO - 🔧 Model set to evaluation mode
2025-09-11 06:17:10,426 - voxtral_realtime - INFO - 💡 Skipping torch.compile (disabled for stability)
2025-09-11 06:17:10,427 - voxtral_realtime - INFO - 🎉 Voxtral model fully initialized in 23.77s and ready for conversation!
2025-09-11 06:17:10,428 - root - INFO - ✅ Voxtral model initialized for TCP server
2025-09-11 06:17:10,428 - root - INFO - 🎉 TCP server components fully initialized
2025-09-11 06:17:10,429 - root - INFO - ✅ TCP server running on 0.0.0.0:8766
2025-09-11 06:17:10,430 - root - INFO - 🎙️ VAD-enabled streaming server ready for production
❌ TCP Streaming Server failed to start on port 8766 after 30 seconds

🛑 Shutting down Conversational Streaming Server...
root@1ead7f6fcad2:/workspace/Voxtral-Final# INFO:     Shutting down
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [3831]
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [3817]

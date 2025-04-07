# arc_cuda

An approximate / very arcade-like particle heat simulation, based on discussions with an LLM about how plasma cutting works.


# Demonstration Video

https://youtu.be/TWe5m8YjZJM

## Requirements
- CUDA (anything should work, tested with 12.x)
- libwebsockets
- libdatachannel
- nlohmann's json library


## Usage

```bash
mkdir build
cd build
cmake ..
make
./arc_cuda
```

Then serve the frontend (e.g. `python -m http.server`) and open index.html. Make sure to tailor the hardcoded IP to your network setup depending on where the server is being run.

## LICENSE

MIT

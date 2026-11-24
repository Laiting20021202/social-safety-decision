# Runtime configuration

The Compose launcher mounts this directory at `/runtime-config` read-only.
Place measured hardware parameters in an external directory and set
`OPENARM_CONFIG_DIR` to that directory. The Isaac defaults are versioned in
`src/openarm_sim/config`.

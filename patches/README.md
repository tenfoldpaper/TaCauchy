# Local libuipc patches

TaCauchy pins `source/tacex_uipc/libuipc` to `DH-Ng/libuipc@837808a361398f18e9526e36a0764b1ec0b741ac` and stores the project-specific local changes in this directory.

Apply them after cloning submodules and before building `source/tacex_uipc`:

```bash
./scripts/setup_libuipc_patches.sh
```

Patch contents:

- `libuipc-tacauchy.patch`: changes in the pinned `libuipc` checkout, including CUDA/CMake and Cauchy stress extraction hooks.
- `muda-tacauchy.patch`: local change in `libuipc/external/muda`.

# Vulcan

Portable, self-hosted code generation. Vulcan lowers a small SSA IR to native machine code
for several targets, with no LLVM dependency: its own register allocators, linker, and JIT.

## Targets

- **AArch64** (A64 + NEON): native JIT, ELF objects, linker, runnable ELF executables
- **RISC-V** (RV64 + RVV): ELF objects, linker, JIT
- **x86-64** (SSE/SSE2 + AVX) and **x86-32**: ELF objects, linker, JIT
- **NVIDIA SASS** (compute and graphics): encoder and instruction selection
- Container formats: in-memory W^X JIT, ELF objects and executables, baremetal flat binaries,
  UEFI PE32+/COFF

## Frontends

- **Zig** (`vulcan-zig`): `build-exe` and `build-obj` emit static ELF executables and
  relocatable objects; executables can link `.o` inputs and use `-T` linker scripts.
  `-Mname=path` maps package imports to source files.
- **WebAssembly** (`vulcan-wasm`): Wasm to IR to JIT, as a host CLI or a UEFI boot application
- **GLSL** (`vulcan-glsl`): a GLSL to SPIR-V shader compiler
- **SPIR-V** (`vulcan-spirv`): bidirectional SPIR-V and IR

## Optimization (`vulcan-opt`)

A pass manager with cached analyses (dominators, CFG, natural loops) driving constant folding,
GVN/CSE, LICM, DCE, inter-procedural inlining, SLP auto-vectorization, and division lowering,
plus link-time optimization (LTO) and profile-guided optimization (PGO).

## Build

```
zig build                        # host CLIs: vulcan-wasm, vulcan-glsl, vulcan-zig
zig build test                   # run the test suite
zig build run-glsl  -- in.glsl   # compile GLSL to SPIR-V
zig build run-zig -- build-exe app.zig
zig build run-zig -- build-obj -target riscv64-freestanding-none app.zig
zig build run-wasm  -- in.wasm   # JIT and run a Wasm module
zig build -Dtarget=x86_64-uefi   # build the UEFI boot application
```

Requires Zig 0.16. The Nix flake provides a dev shell with the toolchain (`nix develop`).

//! vulcan-zig: a Zig frontend for Vulcan. Parses Zig source with `std.zig.Ast`, lowers it to
//! Vulcan IR, and (through the target backends) executes or emits it. This root module
//! re-exports the lowering stage and the top-level `compile` entry point.

const std = @import("std");

pub const lower = @import("vulcan-zig/lower.zig");

/// Compile Zig source for the host source configuration. Source-level target builtins such as
/// `builtin.cpu.arch` use the host architecture; backend emission remains separate.
pub const compile = lower.compile;
/// Compile Zig source with an explicit source-level target architecture. This controls only
/// target-dependent frontend folding, not the architecture selected by an IR backend.
pub const compileForTarget = lower.compileForTarget;
/// Compile a Zig source FILE, resolving real `@import` declarations on disk (relative to
/// `path`, or `std_dir` for `@import("std")`). The only entry point that touches a filesystem.
pub const compileFile = lower.compileFile;
pub const compileFileOptions = lower.compileFileOptions;
pub const FileCompileOptions = lower.FileCompileOptions;
pub const ModuleMapping = lower.ModuleMapping;
/// Architectures recognized by source-level target builtin folding.
pub const TargetArch = lower.TargetArch;
pub const Module = lower.Module;
pub const NamedFunction = lower.NamedFunction;
pub const Data = lower.Data;
pub const DataKind = lower.DataKind;
/// A direct source-level module declaration for a later resolver.
pub const Import = lower.Import;
/// The error set `compile` can return (parse and lowering failures).
pub const Error = lower.Error;

test {
    std.testing.refAllDecls(@This());
}

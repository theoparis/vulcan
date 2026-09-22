//! Differential oracle tests: the real `zig` compiler runs a synthesized `main` that prints
//! the result of calling into the same extended-subset source, and this frontend's own
//! lower -> IR -> host JIT path is required to print the identical line. `zig_exe` comes
//! through `build_options`, the same `zig` that builds this test, so it does not depend on
//! the PATH. A test SKIPS, and never fails, when that toolchain cannot run (mirrors
//! `libs/vulcan-wasm/tests/toolchain.zig`'s precedent).

const std = @import("std");
const zigc = @import("vulcan-zig");
const target = @import("vulcan-target");
const build_options = @import("build_options");

/// Run `body` (the extended-subset source) through the real `zig` compiler, appending a
/// `main` that prints `call_expr`'s result with `fmt`, and return its stdout. Returns
/// `error.NoToolchain` when `zig` itself cannot be run, so the caller skips.
fn oracleRun(allocator: std.mem.Allocator, io: std.Io, tmp: *std.testing.TmpDir, body: []const u8, call_expr: []const u8, comptime fmt: []const u8) ![]u8 {
    const source = try std.fmt.allocPrint(allocator,
        \\{s}
        \\const std = @import("std");
        \\pub fn main(init: std.process.Init) !void {{
        \\    var buf: [256]u8 = undefined;
        \\    var w = std.Io.File.stdout().writer(init.io, &buf);
        \\    try w.interface.print("{s}\n", .{{{s}}});
        \\    try w.interface.flush();
        \\}}
        \\
    , .{ body, fmt, call_expr });
    defer allocator.free(source);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.zig", .data = source });

    const ran = std.process.run(allocator, io, .{
        .argv = &.{ build_options.zig_exe, "run", "m.zig" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |err| switch (err) {
        error.FileNotFound => return error.NoToolchain,
        else => return err,
    };
    defer allocator.free(ran.stderr);
    if (ran.term != .exited or ran.term.exited != 0) {
        allocator.free(ran.stdout);
        std.debug.print("zig run failed:\n{s}\n--- source ---\n{s}\n", .{ ran.stderr, source });
        return error.CompileFailed;
    }
    return ran.stdout;
}

test "oracle: recursive fib matches the real zig compiler" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const body =
        \\fn fib(n: i64) i64 {
        \\    if (n < 2) return n;
        \\    return fib(n - 1) + fib(n - 2);
        \\}
    ;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const oracle_out = oracleRun(a, io, &tmp, body, "fib(10)", "{d}") catch |err| switch (err) {
        error.NoToolchain => return error.SkipZigTest,
        else => return err,
    };
    defer a.free(oracle_out);

    var mod = try zigc.compile(a, body);
    defer mod.deinit(a);
    const f_ir = mod.find("fib") orelse return error.MissingFunction;
    var jitted = try target.native.jitModule(a, &.{.{ .name = "fib", .func = f_ir }});
    defer jitted.deinit();
    const fib = jitted.entry(*const fn (i64) callconv(.c) i64, "fib") orelse return error.NoEntry;

    const ours = try std.fmt.allocPrint(a, "{d}\n", .{fib(10)});
    defer a.free(ours);
    try std.testing.expectEqualStrings(oracle_out, ours);
}

test "oracle: struct-and-array-summing function matches the real zig compiler" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const body =
        \\const Point = struct { x: i64, y: i64 };
        \\fn weighted(p: *Point, weights: *[2]i64) i64 {
        \\    var arr: [2]i64 = undefined;
        \\    arr[0] = p.x;
        \\    arr[1] = p.y;
        \\    var total: i64 = 0;
        \\    var i: usize = 0;
        \\    while (i < 2) : (i += 1) {
        \\        total += arr[i] * weights.*[i];
        \\    }
        \\    return total;
        \\}
        \\fn f(a: i64, b: i64) i64 {
        \\    var pt = Point{ .x = a, .y = b };
        \\    var w: [2]i64 = undefined;
        \\    w[0] = 3;
        \\    w[1] = 5;
        \\    return weighted(&pt, &w);
        \\}
    ;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const oracle_out = oracleRun(a, io, &tmp, body, "f(4, 7)", "{d}") catch |err| switch (err) {
        error.NoToolchain => return error.SkipZigTest,
        else => return err,
    };
    defer a.free(oracle_out);

    var mod = try zigc.compile(a, body);
    defer mod.deinit(a);
    var mod_funcs = try a.alloc(target.native.ModuleFunction, mod.funcs.len);
    defer a.free(mod_funcs);
    for (mod.funcs, 0..) |*nf, i| mod_funcs[i] = .{ .name = nf.name, .func = &nf.func };
    var jitted = try target.native.jitModule(a, mod_funcs);
    defer jitted.deinit();
    const f = jitted.entry(*const fn (i64, i64) callconv(.c) i64, "f") orelse return error.NoEntry;

    const ours = try std.fmt.allocPrint(a, "{d}\n", .{f(4, 7)});
    defer a.free(ours);
    try std.testing.expectEqualStrings(oracle_out, ours);
}

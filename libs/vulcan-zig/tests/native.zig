//! Execution tests for the Zig frontend: compile the extended subset to Vulcan IR, JIT it
//! for the host via `vulcan-target.native`, and run it. One test per subset feature
//! (arithmetic, if/while/for, pointers, arrays, structs, recursive calls), plus one test per
//! documented `Error` case.

const std = @import("std");
const ir = @import("vulcan-ir");
const zigc = @import("vulcan-zig");
const target = @import("vulcan-target");

fn jitFn(a: std.mem.Allocator, mod: *const zigc.Module, name: []const u8, comptime Fn: type) !struct { j: target.native.JittedModule, f: Fn } {
    var mod_funcs = try a.alloc(target.native.ModuleFunction, mod.funcs.len);
    defer a.free(mod_funcs);
    for (mod.funcs, 0..) |*nf, i| mod_funcs[i] = .{ .name = nf.name, .func = &nf.func };
    var mod_data = try a.alloc(target.native.ModuleData, mod.data.len);
    defer a.free(mod_data);
    for (mod.data, 0..) |d, i| mod_data[i] = .{
        .name = d.name,
        .bytes = d.bytes,
        .kind = switch (d.kind) {
            .rodata => .rodata,
            .data => .data,
            .bss => .bss,
        },
        .size = d.size,
    };
    var jitted = try target.native.jitModuleData(a, mod_funcs, mod_data);
    const f = jitted.entry(Fn, name) orelse return error.NoEntry;
    return .{ .j = jitted, .f = f };
}

const FormatWriter = extern struct {
    vtable: *const FormatVTable,
    buffer: [*]u8,
    buffer_len: usize,
    end: usize,
};

const FormatVTable = extern struct {
    drain: *const anyopaque,
};

var format_output: [256]u8 = undefined;
var format_output_len: usize = 0;

fn captureFormatDrain(_: *FormatWriter, data: [*]const []const u8, data_len: usize, splat: usize) callconv(.c) usize {
    if (data_len != 1) return 0;
    const pattern = data[0];
    const length = pattern.len * splat;
    if (length > format_output.len - format_output_len) return 0;
    for (0..splat) |_| {
        @memcpy(format_output[format_output_len..][0..pattern.len], pattern);
        format_output_len += pattern.len;
    }
    return 0;
}

const format_vtable = FormatVTable{ .drain = @ptrCast(&captureFormatDrain) };

test "Writer.print formats integers and drains formatted bytes" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\const VTable = extern struct { drain: *const anyopaque };
        \\const Writer = extern struct { vtable: *const VTable, buffer: [*]u8, buffer_len: usize, end: usize };
        \\pub fn emit(writer: *Writer, value: u64, signed: i64, bytes: *const [2]u8) void {
        \\    writer.print("dec={d} hex={x} upper={X} padded={x:0>4}\n", .{ value, value, value, value }) catch {};
        \\    writer.print("signed={d:0>4}\n", .{signed}) catch {};
        \\    writer.print("bytes={s}\n", .{bytes}) catch {};
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "emit", *const fn (*FormatWriter, u64, i64, *const [2]u8) callconv(.c) void);
    defer r.j.deinit();
    var mock_buffer: [1]u8 = undefined;
    var writer = FormatWriter{ .vtable = &format_vtable, .buffer = &mock_buffer, .buffer_len = 0, .end = 0 };
    format_output_len = 0;
    const bytes = [_]u8{ 'o', 'k' };
    r.f(&writer, 42, -42, &bytes);
    try std.testing.expectEqualStrings("dec=42 hex=2a upper=2A padded=002a\nsigned=-042\nbytes=ok\n", format_output[0..format_output_len]);
}

test "unused comptime-only declaration is not emitted or evaluated" {
    const a = std.testing.allocator;
    const source =
        \\const hook = opaqueInitializer();
        \\fn f() i64 { return 42; }
    ;
    var mod = try zigc.compile(a, source);
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "f", *const fn () callconv(.c) i64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(i64, 42), r.f());

    try std.testing.expectError(zigc.Error.UnsupportedType, zigc.compile(a,
        \\const hook = opaqueInitializer();
        \\fn f() i64 { return hook; }
    ));
}

test "forward comptime constants resolve on demand without runtime storage" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\const answer = part + 2;
        \\const selected = alias;
        \\const alias = true;
        \\const part = 40;
        \\fn f() i64 { if (selected) return answer; return 0; }
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "f", *const fn () callconv(.c) i64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(i64, 42), r.f());
    try std.testing.expectError(zigc.Error.UnsupportedType, zigc.compile(a,
        \\const a = b;
        \\const b = a;
        \\fn f() i64 { return a; }
    ));
}

test "void error-union catch runs its handler only on errors" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn succeed() error{Oops}!void { return; }
        \\fn fail() error{Oops}!void { return error.Oops; }
        \\fn exercise() u64 {
        \\    succeed() catch { return 1; };
        \\    fail() catch { return 42; };
        \\    return 0;
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "exercise", *const fn () callconv(.c) u64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(u64, 42), r.f());
}

test "plain imported struct fields work across source files" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "shape.zig", .data =
        \\pub const Shape = struct {
        \\    const unused = 99;
        \\    left: u64,
        \\    right: u64,
        \\    ctx: ?*anyopaque,
        \\    read_fn: *const fn (ctx: ?*anyopaque, off: usize) u64,
        \\    pub fn unusedMethod() void {}
        \\};
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "root.zig", .data =
        \\const shape = @import("shape.zig");
        \\fn sum() u64 {
        \\    var item = shape.Shape{ .left = 7, .right = 35 };
        \\    return item.left + item.right;
        \\}
    });
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path, "root.zig" });
    defer a.free(path);
    var mod = try zigc.compileFile(a, io, path, .x86_64, null);
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "sum", *const fn () callconv(.c) u64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(u64, 42), r.f());
}

test "imported mutable scalar globals resolve to linked data" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "state.zig", .data = "pub var ready: bool = false;" });
    try tmp.dir.writeFile(io, .{ .sub_path = "root.zig", .data =
        \\const state = @import("state.zig");
        \\fn check() u64 {
        \\    if (state.ready) return 1;
        \\    return 0;
        \\}
    });
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path, "root.zig" });
    defer a.free(path);
    const state_path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path, "state.zig" });
    defer a.free(state_path);
    var root = try zigc.compileFile(a, io, path, .x86_64, null);
    defer root.deinit(a);
    var state = try zigc.compileFile(a, io, state_path, .x86_64, null);
    defer state.deinit(a);

    const functions = try a.alloc(target.native.ModuleFunction, root.funcs.len);
    defer a.free(functions);
    for (root.funcs, 0..) |*nf, i| functions[i] = .{ .name = nf.name, .func = &nf.func };
    const data = try a.alloc(target.native.ModuleData, state.data.len);
    defer a.free(data);
    for (state.data, 0..) |d, i| data[i] = .{
        .name = d.name,
        .bytes = d.bytes,
        .kind = switch (d.kind) {
            .rodata => .rodata,
            .data => .data,
            .bss => .bss,
        },
        .size = d.size,
    };
    var jitted = try target.native.jitModuleData(a, functions, data);
    defer jitted.deinit();
    const check = jitted.entry(*const fn () callconv(.c) u64, "check") orelse return error.NoEntry;
    try std.testing.expectEqual(@as(u64, 0), check());
    jitted.dataAddr("ready").?[0] = 1;
    try std.testing.expectEqual(@as(u64, 1), check());
}

test "imported source file itself is a struct type" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "packet.zig", .data =
        \\const This = @This();
        \\value: u64,
        \\pub fn unused() void {}
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "root.zig", .data =
        \\const Packet = @import("packet.zig");
        \\fn read() u64 {
        \\    var p = Packet.This{ .value = 42 };
        \\    return p.value;
        \\}
    });
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path, "root.zig" });
    defer a.free(path);
    var mod = try zigc.compileFile(a, io, path, .x86_64, null);
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "read", *const fn () callconv(.c) u64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(u64, 42), r.f());
}

test "opaque pointer round-trip preserves its address without allowing dereference" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn bounce(p: *anyopaque) *anyopaque { return p; }
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "bounce", *const fn (*anyopaque) callconv(.c) *anyopaque);
    defer r.j.deinit();
    var byte: u8 = 0;
    try std.testing.expectEqual(@intFromPtr(&byte), @intFromPtr(r.f(&byte)));
    try std.testing.expectError(zigc.Error.UnsupportedType, zigc.compile(a,
        \\fn bad(p: *anyopaque) void { p.* = undefined; }
    ));
}

test "extern global: address resolves against another module's data" {
    const a = std.testing.allocator;
    var consumer = try zigc.compile(a,
        \\extern var counter: i64;
        \\fn readCounter() i64 {
        \\    return counter;
        \\}
    );
    defer consumer.deinit(a);
    var provider = try zigc.compile(a,
        \\var counter: i64 = 42;
    );
    defer provider.deinit(a);

    try std.testing.expectEqual(@as(usize, 0), consumer.data.len);
    const functions = [_]target.native.ModuleFunction{
        .{ .name = consumer.funcs[0].name, .func = &consumer.funcs[0].func },
    };
    const data = [_]target.native.ModuleData{
        .{
            .name = provider.data[0].name,
            .bytes = provider.data[0].bytes,
            .kind = .data,
            .size = provider.data[0].size,
        },
    };
    var jitted = try target.native.jitModuleData(a, &functions, &data);
    defer jitted.deinit();
    const read_counter = jitted.entry(*const fn () callconv(.c) i64, "readCounter") orelse return error.NoEntry;
    try std.testing.expectEqual(@as(i64, 42), read_counter());
}

test "array length: named constants and shifts lower in pointer types" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\const MAX_HARTS: usize = 1 << 3;
        \\fn acceptStacks(stacks: *[MAX_HARTS][1 << 20]u8) void {}
    );
    defer mod.deinit(a);
    try std.testing.expect(mod.find("acceptStacks") != null);
}

test "noreturn functions terminate their own and calling control flow" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn halt() noreturn {
        \\    while (true) {}
        \\}
        \\fn caller() void {
        \\    halt();
        \\}
    );
    defer mod.deinit(a);
    try std.testing.expect(mod.find("halt") != null);
    try std.testing.expect(mod.find("caller") != null);
}

test "arithmetic: +, -, *, /, % over i64" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn f(x: i64, y: i64) i64 {
        \\    return (x + y) * (x - y) - (x / y) + (x % y);
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "f", *const fn (i64, i64) callconv(.c) i64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(i64, (17 + 5) * (17 - 5) - (17 / 5) + (17 % 5)), r.f(17, 5));
}

test "comparisons and bool operators" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn f(x: i64, y: i64) bool {
        \\    return (x < y) or (x == y and y != 0);
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "f", *const fn (i64, i64) callconv(.c) bool);
    defer r.j.deinit();
    try std.testing.expect(r.f(1, 2));
    try std.testing.expect(!r.f(3, 2));
    try std.testing.expect(r.f(2, 2));
}

test "if/return and recursive fib" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn fib(n: i64) i64 {
        \\    if (n < 2) return n;
        \\    return fib(n - 1) + fib(n - 2);
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "fib", *const fn (i64) callconv(.c) i64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(i64, 55), r.f(10));
}

test "while loop with continue expression" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn sum(n: i64) i64 {
        \\    var total: i64 = 0;
        \\    var i: i64 = 0;
        \\    while (i < n) : (i += 1) {
        \\        total += i;
        \\    }
        \\    return total;
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "sum", *const fn (i64) callconv(.c) i64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(i64, 45), r.f(10));
}

test "for loop over an int range, with break/continue" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn sumEven(n: i64) i64 {
        \\    var total: i64 = 0;
        \\    for (0..n) |i| {
        \\        if (i == 7) break;
        \\        if (i % 2 == 1) continue;
        \\        total += i;
        \\    }
        \\    return total;
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "sumEven", *const fn (i64) callconv(.c) i64);
    defer r.j.deinit();
    // i = 0,2,4,6 sum before hitting the break at i==7.
    try std.testing.expectEqual(@as(i64, 0 + 2 + 4 + 6), r.f(20));
}

test "pointers: address-of, deref, and a pointer parameter that mutates the caller's local" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn addOne(p: *i64) void {
        \\    p.* = p.* + 1;
        \\}
        \\fn f(x: i64) i64 {
        \\    var v: i64 = x;
        \\    addOne(&v);
        \\    addOne(&v);
        \\    return v;
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "f", *const fn (i64) callconv(.c) i64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(i64, 9), r.f(7));
}

test "fixed-size arrays: local array, indexed read/write with a runtime index" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn f(n: i64) i64 {
        \\    var arr: [4]i64 = undefined;
        \\    var i: i64 = 0;
        \\    while (i < 4) : (i += 1) {
        \\        arr[i] = i * n;
        \\    }
        \\    return arr[0] + arr[1] + arr[2] + arr[3];
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "f", *const fn (i64) callconv(.c) i64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(i64, (0 + 1 + 2 + 3) * 5), r.f(5));
}

test "structs: field init and access through a pointer" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\const Point = struct {
        \\    x: i64,
        \\    y: i64,
        \\};
        \\fn dist(p: *Point) i64 {
        \\    return p.x * p.x + p.y * p.y;
        \\}
        \\fn f(a: i64, b: i64) i64 {
        \\    var pt = Point{ .x = a, .y = b };
        \\    pt.x = pt.x + 1;
        \\    return dist(&pt);
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "f", *const fn (i64, i64) callconv(.c) i64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(i64, (3 + 1) * (3 + 1) + 4 * 4), r.f(3, 4));
}

test "extern struct: field offsets match a host ABI record" {
    const Wire = extern struct { byte: u8, wide: u64 };
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\const Wire = extern struct { byte: u8, wide: u64 };
        \\fn fill(wire: *Wire) void {
        \\    wire.byte = 7;
        \\    wire.wide = 1234;
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "fill", *const fn (*Wire) callconv(.c) void);
    defer r.j.deinit();
    var wire: Wire = undefined;
    r.f(&wire);
    try std.testing.expectEqual(@as(u8, 7), wire.byte);
    try std.testing.expectEqual(@as(u64, 1234), wire.wide);
}

test "nested struct field access and float arithmetic" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\const Inner = struct { v: f64 };
        \\const Outer = struct { inner: Inner, scale: f64 };
        \\fn f(x: f64) f64 {
        \\    var o = Outer{ .inner = .{ .v = x }, .scale = 2.0 };
        \\    return o.inner.v * o.scale;
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "f", *const fn (f64) callconv(.c) f64);
    defer r.j.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), r.f(3.5), 1e-9);
}

test "optional aggregate storage, capture, and global initialization" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\const Shape = struct { first: u64, value: u64, last: u64 };
        \\var some: ?Shape = Shape{ .first = 2, .value = 19, .last = 3 };
        \\fn none() u64 {
        \\    var maybe: ?Shape = null;
        \\    if (maybe) |shape| return shape.value;
        \\    return 7;
        \\}
        \\fn present() u64 {
        \\    var maybe: ?Shape = some;
        \\    if (maybe) |shape| return shape.value;
        \\    return 0;
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "none", *const fn () callconv(.c) u64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(u64, 7), r.f());
    var present = try jitFn(a, &mod, "present", *const fn () callconv(.c) u64);
    defer present.j.deinit();
    try std.testing.expectEqual(@as(u64, 19), present.f());
}

test "aarch64 optional large aggregate returns use hidden result storage" {
    const a = std.testing.allocator;
    var mod = try zigc.compileForTarget(a,
        \\const Shape = struct { first: u64, value: u64, last: u64 };
        \\fn empty() ?Shape { return null; }
        \\fn read() u64 {
        \\    if (empty()) |shape| return shape.value;
        \\    return 0;
        \\}
    , .aarch64);
    defer mod.deinit(a);
    for (mod.funcs) |*nf| {
        const code = try target.aarch64.isel.selectFunction(a, &nf.func);
        defer a.free(code);
    }
}

test "aarch64 simple inline assembly emits exact opcodes" {
    const a = std.testing.allocator;
    var mod = try zigc.compileForTarget(a,
        \\pub fn wait() void {
        \\    asm volatile ("wfe");
        \\    asm volatile ("sev");
        \\    asm volatile ("isb");
        \\    asm volatile ("dsb sy");
        \\    asm volatile ("dsb ish");
        \\    asm volatile ("eret");
        \\    asm volatile ("hvc #0");
        \\    asm volatile ("msr daifclr, #2");
        \\    asm volatile ("msr daifset, #0xf");
        \\    asm volatile ("ic iallu");
        \\    asm volatile ("tlbi vmalle1");
        \\}
    , .aarch64);
    defer mod.deinit(a);
    const code = try target.aarch64.isel.selectFunction(a, &mod.funcs[0].func);
    defer a.free(code);
    for ([_]u32{
        0xd503205f, 0xd503209f, 0xd5033fdf, 0xd5033f9f, 0xd5033b9f,
        0xd69f03e0, 0xd4000002, 0xd50342ff, 0xd5034fdf, 0xd508751f,
        0xd508871f,
    }) |opcode| try std.testing.expect(std.mem.indexOfScalar(u32, code, opcode) != null);
}

test "global fixed arrays support repeated initializer patterns" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\var values: [4]u64 = [_]u64{9} ** 4;
        \\fn last() u64 { return values[3]; }
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "last", *const fn () callconv(.c) u64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(u64, 9), r.f());
}

test "slice parameters lower to pointer and length ABI values" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn length(bytes: []const u8) usize { return bytes.len; }
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "length", *const fn ([*]const u8, usize) callconv(.c) usize);
    defer r.j.deinit();
    const bytes = [_]u8{ 4, 8, 15, 16 };
    try std.testing.expectEqual(@as(usize, bytes.len), r.f(&bytes, bytes.len));
}

test "import declarations: unused namespaces do not enter generated IR" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\const std = @import("std");
        \\const mem = std.mem;
        \\fn f() i64 {
        \\    return 9;
        \\}
    );
    defer mod.deinit(a);
    try std.testing.expectEqual(@as(usize, 1), mod.imports.len);
    try std.testing.expectEqualStrings("std", mod.imports[0].name);
    try std.testing.expectEqualStrings("std", mod.imports[0].path);
    var r = try jitFn(a, &mod, "f", *const fn () callconv(.c) i64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(i64, 9), r.f());
}

test "builtin target: cpu arch equality folds for the requested target" {
    const a = std.testing.allocator;
    const source =
        \\const builtin = @import("builtin");
        \\const is_riscv = builtin.cpu.arch == .riscv64;
        \\fn selected() i64 {
        \\    if (is_riscv) {
        \\        return 1;
        \\    }
        \\    return 0;
        \\}
    ;
    var riscv = try zigc.compileForTarget(a, source, .riscv64);
    defer riscv.deinit(a);
    var riscv_run = try jitFn(a, &riscv, "selected", *const fn () callconv(.c) i64);
    defer riscv_run.j.deinit();
    try std.testing.expectEqual(@as(i64, 1), riscv_run.f());

    var aarch64 = try zigc.compileForTarget(a, source, .aarch64);
    defer aarch64.deinit(a);
    var aarch64_run = try jitFn(a, &aarch64, "selected", *const fn () callconv(.c) i64);
    defer aarch64_run.j.deinit();
    try std.testing.expectEqual(@as(i64, 0), aarch64_run.f());
}

test "top-level constants: booleans and integers are available in function bodies" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\const count = 3;
        \\const enabled = true;
        \\fn selected() i64 {
        \\    if (enabled) return count + 1;
        \\    return 0;
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "selected", *const fn () callconv(.c) i64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(i64, 4), r.f());
}

test "top-level constants: utf8ToUtf16LeStringLiteral idiom initializes an inferred-type array global" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\const std = @import("std");
        \\var vendor = std.unicode.utf8ToUtf16LeStringLiteral("Hi").*;
        \\fn firstUnit() u64 {
        \\    return vendor[0];
        \\}
        \\fn secondUnit() u64 {
        \\    return vendor[1];
        \\}
        \\fn terminator() u64 {
        \\    return vendor[2];
        \\}
    );
    defer mod.deinit(a);
    var r0 = try jitFn(a, &mod, "firstUnit", *const fn () callconv(.c) u64);
    defer r0.j.deinit();
    try std.testing.expectEqual(@as(u64, 'H'), r0.f());
    var r1 = try jitFn(a, &mod, "secondUnit", *const fn () callconv(.c) u64);
    defer r1.j.deinit();
    try std.testing.expectEqual(@as(u64, 'i'), r1.f());
    var r2 = try jitFn(a, &mod, "terminator", *const fn () callconv(.c) u64);
    defer r2.j.deinit();
    try std.testing.expectEqual(@as(u64, 0), r2.f());
}

test "error: parse error fails closed" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.ParseError, zigc.compile(a, "fn f(x: i64 i64 { return x; }"));
}

test "error: undeclared identifier" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.UndeclaredIdentifier, zigc.compile(a, "fn f() i64 { return y; }"));
}

test "error: call arity mismatch" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.CallArityMismatch, zigc.compile(a,
        \\fn g(a: i64, b: i64) i64 { return a + b; }
        \\fn f() i64 { return g(1); }
    ));
}

test "error: type mismatch on return" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.TypeMismatch, zigc.compile(a, "fn f() bool { return 1; }"));
}

test "error: duplicate top-level declaration" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.DuplicateDecl, zigc.compile(a,
        \\fn f() i64 { return 1; }
        \\fn f() i64 { return 2; }
    ));
}

test "error: unknown struct field" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.UnknownField, zigc.compile(a,
        \\const P = struct { x: i64 };
        \\fn f(p: *P) i64 { return p.y; }
    ));
}

test "error: unsupported type (comptime generic parameter)" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.UnsupportedType, zigc.compile(a, "fn f(comptime T: type) T { return undefined; }"));
}

test "error: unsupported type (slice return)" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.UnsupportedType, zigc.compile(a, "fn f(s: []i64) []i64 { return s; }"));
}

test "globals: direct data, read-only data, and zero-initialized storage persist across calls" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\var counter: i64 = 40;
        \\const delta: i64 = 2;
        \\var zero: i64 = 0;
        \\fn next() i64 {
        \\    counter += delta;
        \\    return counter + zero;
        \\}
    );
    defer mod.deinit(a);
    try std.testing.expectEqual(@as(usize, 3), mod.data.len);
    try std.testing.expectEqual(zigc.DataKind.data, mod.data[0].kind);
    try std.testing.expectEqual(zigc.DataKind.rodata, mod.data[1].kind);
    try std.testing.expectEqual(zigc.DataKind.bss, mod.data[2].kind);

    var r = try jitFn(a, &mod, "next", *const fn () callconv(.c) i64);
    defer r.j.deinit();
    try std.testing.expectEqual(@as(i64, 42), r.f());
    try std.testing.expectEqual(@as(i64, 44), r.f());
}

test "nullable pointer payload: if binding exposes the non-null pointer only in its branch" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn unwrapOrZero(value: ?*i64) i64 {
        \\    if (value) |pointer| return pointer.*;
        \\    return 0;
        \\}
    );
    defer mod.deinit(a);
    var r = try jitFn(a, &mod, "unwrapOrZero", *const fn (?*i64) callconv(.c) i64);
    defer r.j.deinit();
    var value: i64 = 19;
    try std.testing.expectEqual(@as(i64, 19), r.f(&value));
    try std.testing.expectEqual(@as(i64, 0), r.f(null));
}

test "variadic nullable pointer: orelse break terminates the list and defer closes it" {
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn sumUntilNull(...) callconv(.c) i64 {
        \\    var va = @cVaStart();
        \\    defer @cVaEnd(&va);
        \\    var total: i64 = 0;
        \\    while (true) {
        \\        const value = @cVaArg(&va, ?*i64) orelse break;
        \\        total += value.*;
        \\    }
        \\    return total;
        \\}
    );
    defer mod.deinit(a);
    const callee = mod.find("sumUntilNull") orelse return error.MissingFunction;
    try std.testing.expect(callee.is_variadic);
    try std.testing.expectEqual(@as(u32, 0), callee.num_fixed_params);

    var caller = ir.function.Function.init(a);
    defer caller.deinit();
    const entry = try caller.appendBlock();
    const i64t = try caller.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptrt = try caller.types.ptrGlobal();
    const first = try caller.appendInst(entry, ptrt, .{ .alloca = .{ .elem = i64t } });
    const second = try caller.appendInst(entry, ptrt, .{ .alloca = .{ .elem = i64t } });
    const five = try caller.appendInst(entry, i64t, .{ .iconst = 5 });
    const eleven = try caller.appendInst(entry, i64t, .{ .iconst = 11 });
    try caller.appendStore(entry, five, first);
    try caller.appendStore(entry, eleven, second);
    const null_ptr = try caller.appendInst(entry, ptrt, .{ .iconst = 0 });
    const result = try caller.appendCallV(entry, i64t, "sumUntilNull", &.{ first, second, null_ptr }, 0);
    caller.setTerminator(entry, .{ .ret = ir.function.Ret.one(result) });

    var jitted = try target.native.jitModule(a, &.{
        .{ .name = "sumUntilNull", .func = callee },
        .{ .name = "callSumUntilNull", .func = &caller },
    });
    defer jitted.deinit();
    const call_sum = jitted.entry(*const fn () callconv(.c) i64, "callSumUntilNull") orelse return error.MissingEntry;
    try std.testing.expectEqual(@as(i64, 16), call_sum());
}
test "variadic function definition: @cVaStart/@cVaArg/@cVaEnd sums trailing i64 args" {
    // Real Zig's own `VaList`/`@cVaStart` is `@compileError`'d for `stage2_llvm` on
    // aarch64-uefi/windows (the exact bug this frontend exists to route around - see
    // `../lower.zig`'s header doc and the project's long-term goal). There is also no safe
    // way to call INTO this callee from a *host*-compiled variadic caller: Apple's aarch64
    // ABI (what a macOS test host's own Zig would use) puts every variadic argument on the
    // stack, never in x1-x7, while this callee's `@cVaStart`/`@cVaArg` lowering
    // (`vulcan-target/aarch64/isel.zig`'s `emitVaStart`/`emitVaArg`) follows the STANDARD
    // AAPCS64 convention (named args then variadic args share the x1-x7/v0-v7 save area,
    // register-then-stack). So the caller here is hand-built `vulcan-ir`, not host Zig: both
    // sides then agree on the one calling convention `vulcan-target`'s own backend defines,
    // which is what this frontend and its callers actually use end to end.
    const a = std.testing.allocator;
    var mod = try zigc.compile(a,
        \\fn sumVarargs(count: i64, ...) callconv(.c) i64 {
        \\    var va = @cVaStart();
        \\    var total: i64 = 0;
        \\    var i: i64 = 0;
        \\    while (i < count) : (i += 1) {
        \\        total += @cVaArg(&va, i64);
        \\    }
        \\    @cVaEnd(&va);
        \\    return total;
        \\}
    );
    defer mod.deinit(a);
    const callee = mod.find("sumVarargs") orelse return error.MissingFunction;
    try std.testing.expect(callee.is_variadic);
    try std.testing.expectEqual(@as(u32, 1), callee.num_fixed_params);

    var caller = ir.function.Function.init(a);
    defer caller.deinit();
    const entry = try caller.appendBlock();
    const i64t = try caller.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const count_v = try caller.appendInst(entry, i64t, .{ .iconst = 3 });
    const a_v = try caller.appendInst(entry, i64t, .{ .iconst = 10 });
    const b_v = try caller.appendInst(entry, i64t, .{ .iconst = 20 });
    const c_v = try caller.appendInst(entry, i64t, .{ .iconst = 30 });
    const result = try caller.appendCallV(entry, i64t, "sumVarargs", &.{ count_v, a_v, b_v, c_v }, 1);
    caller.setTerminator(entry, .{ .ret = ir.function.Ret.one(result) });

    var jitted = try target.native.jitModule(a, &.{
        .{ .name = "sumVarargs", .func = callee },
        .{ .name = "caller", .func = &caller },
    });
    defer jitted.deinit();
    const run = jitted.entry(*const fn () callconv(.c) i64, "caller") orelse return error.NoEntry;
    try std.testing.expectEqual(@as(i64, 60), run());
}

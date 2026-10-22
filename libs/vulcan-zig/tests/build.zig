//! End-to-end build-system coverage: Maker configures a normal std.Build graph,
//! while Vulcan compiles its application artifacts and imported Zig modules.
const std = @import("std");
const builtin = @import("builtin");
const options = @import("build_options");

fn runExpect(
    allocator: std.mem.Allocator,
    io: std.Io,
    tmp: *std.testing.TmpDir,
    argv: []const []const u8,
    expected: u8,
) !void {
    const result = try std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .dir = tmp.dir },
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (result.term != .exited or result.term.exited != expected) {
        std.debug.print("command {s}: {any}\nstdout:\n{s}\nstderr:\n{s}\n", .{
            argv[0], result.term, result.stdout, result.stderr,
        });
        return error.UnexpectedProcessResult;
    }
}

fn writeStartup(allocator: std.mem.Allocator, io: std.Io, tmp: *std.testing.TmpDir) !void {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const source = switch (builtin.cpu.arch) {
        .x86_64 =>
        \\.text
        \\.globl _start
        \\_start:
        \\    xor %ebp, %ebp
        \\    and $-16, %rsp
        \\    call mainCode
        \\    mov %rax, %rdi
        \\    mov $60, %rax
        \\    syscall
        ,
        .aarch64 =>
        \\.text
        \\.globl _start
        \\_start:
        \\    bl mainCode
        \\    mov x8, #93
        \\    svc #0
        ,
        .riscv64 =>
        \\.text
        \\.globl _start
        \\_start:
        \\    call mainCode
        \\    li a7, 93
        \\    ecall
        ,
        .x86 =>
        \\.text
        \\.globl _start
        \\_start:
        \\    call mainCode
        \\    mov %eax, %ebx
        \\    mov $1, %eax
        \\    int $0x80
        ,
        else => return error.SkipZigTest,
    };
    try tmp.dir.writeFile(io, .{ .sub_path = "start.s", .data = source });
    try runExpect(allocator, io, tmp, &.{ options.zig_exe, "cc", "-c", "start.s", "-o", "start.o" }, 0);
}

const logic_source =
    \\const config = @import("config");
    \\pub fn compute(value: u64) u64 { return privateIdentity(value) + config.addend; }
    \\fn privateIdentity(value: u64) u64 { return value; }
    \\pub fn unused(comptime T: type, value: T) T { return value; }
;
const changed_logic_source =
    \\const config = @import("config");
    \\pub fn compute(value: u64) u64 { return privateIdentity(value) + config.addend; }
    \\fn privateIdentity(value: u64) u64 { return value + 1; }
    \\pub fn unused(comptime T: type, value: T) T { return value; }
;

test "Maker builds installs and rebuilds normal Zig artifacts without an application fallback" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeStartup(a, io, &tmp);
    try tmp.dir.writeFile(io, .{ .sub_path = "build.zig", .data =
        \\const std = @import("std");
        \\pub fn build(b: *std.Build) void {
        \\    const target = b.standardTargetOptions(.{});
        \\    const optimize = b.standardOptimizeOption(.{});
        \\    const config = b.addOptions();
        \\    config.addOption(u64, "addend", 29);
        \\    const logic = b.createModule(.{
        \\        .root_source_file = b.path("logic.zig"), .target = target, .optimize = optimize,
        \\    });
        \\    logic.addOptions("config", config);
        \\    const object = b.addObject(.{
        \\        .name = "contract",
        \\        .root_module = b.createModule(.{
        \\            .root_source_file = b.path("contract.zig"), .target = target, .optimize = optimize,
        \\        }),
        \\    });
        \\    const exe = b.addExecutable(.{
        \\        .name = "maker-demo",
        \\        .root_module = b.createModule(.{
        \\            .root_source_file = b.path("app.zig"), .target = target, .optimize = optimize,
        \\            .imports = &.{.{ .name = "logic", .module = logic }},
        \\        }),
        \\    });
        \\    exe.root_module.addObject(object);
        \\    exe.root_module.addObjectFile(b.path("start.o"));
        \\    b.installArtifact(exe);
        \\    const run = b.addRunArtifact(exe);
        \\    run.expectExitCode(42);
        \\    b.step("check", "Run the built artifact").dependOn(&run.step);
        \\}
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "logic.zig", .data = logic_source });
    try tmp.dir.writeFile(io, .{ .sub_path = "contract.zig", .data = "export fn contractValue() u64 { return 7; }" });
    try tmp.dir.writeFile(io, .{ .sub_path = "app.zig", .data =
        \\const logic = @import("logic");
        \\extern fn contractValue() u64;
        \\export fn mainCode() u64 { return logic.compute(6) + contractValue(); }
    });
    const compiler = try std.fs.path.resolve(a, &.{ options.build_root, options.vulcan_zig_exe });
    defer a.free(compiler);
    try runExpect(a, io, &tmp, &.{ compiler, "build", "check", "install", "--summary", "none" }, 0);
    try runExpect(a, io, &tmp, &.{"./zig-out/bin/maker-demo"}, 42);

    try tmp.dir.writeFile(io, .{ .sub_path = "logic.zig", .data = changed_logic_source });
    try runExpect(a, io, &tmp, &.{ compiler, "build", "install", "--summary", "none" }, 0);
    try runExpect(a, io, &tmp, &.{"./zig-out/bin/maker-demo"}, 43);

    // This is valid Zig that upstream can compile but this frontend cannot. A
    // successful build here would conceal an application fallback to upstream.
    try tmp.dir.writeFile(io, .{ .sub_path = "contract.zig", .data =
        \\export fn contractValue() u64 {
        \\    const Choice = enum { chosen, ignored };
        \\    const choice: Choice = .chosen;
        \\    return switch (choice) { .chosen => 7, .ignored => 0 };
        \\}
    });
    const rejected = try std.process.run(a, io, .{
        .argv = &.{ compiler, "build", "install", "--summary", "none" },
        .cwd = .{ .dir = tmp.dir },
    });
    defer a.free(rejected.stdout);
    defer a.free(rejected.stderr);
    try std.testing.expect(rejected.term == .exited and rejected.term.exited != 0);

    try tmp.dir.writeFile(io, .{ .sub_path = "contract.zig", .data = "export fn contractValue() u64 { return 7; }" });
    try runExpect(a, io, &tmp, &.{ compiler, "build", "install", "--summary", "none" }, 0);
    try runExpect(a, io, &tmp, &.{"./zig-out/bin/maker-demo"}, 43);
}

const UpdateResult = struct {
    digest: ?std.Build.Cache.BinDigest = null,
    errors: u32 = 0,
};

fn receiveUpdate(allocator: std.mem.Allocator, client: *std.zig.Client) !UpdateResult {
    var result: UpdateResult = .{};
    while (true) {
        const header = try client.receiveMessage();
        const body = try client.in.take(header.bytes_len);
        var body_reader: std.Io.Reader = .fixed(body);
        switch (header.tag) {
            .zig_version => try std.testing.expectEqualStrings(builtin.zig_version_string, body),
            .config, .file_system_inputs => {},
            .emit_digest => {
                _ = try body_reader.takeStruct(std.zig.Server.Message.EmitDigest, .little);
                result.digest = (try body_reader.takeArray(std.Build.Cache.bin_digest_len)).*;
            },
            .error_bundle => {
                var errors = try std.zig.Server.allocErrorBundle(allocator, body);
                defer errors.deinit(allocator);
                result.errors = errors.errorMessageCount();
                return result;
            },
            else => return error.UnexpectedCompilerMessage,
        }
    }
}

fn runServerArtifact(
    allocator: std.mem.Allocator,
    io: std.Io,
    tmp: *std.testing.TmpDir,
    digest: std.Build.Cache.BinDigest,
    expected: u8,
) !void {
    const hex = std.Build.Cache.binToHex(digest);
    const object_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "o", &hex, "server.test.o" });
    defer allocator.free(object_path);
    try runExpect(allocator, io, tmp, &.{
        options.zig_exe, "build-exe", object_path, "start.o", "-femit-bin=server-app",
    }, 0);
    try runExpect(allocator, io, tmp, &.{"./server-app"}, expected);
}

test "compiler server recompiles updates and recovers after an invalid source" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeStartup(a, io, &tmp);
    try tmp.dir.writeFile(io, .{
        .sub_path = "server.zig",
        .data = "export fn mainCode() u64 { return 42; }",
    });
    const compiler = try std.fs.path.resolve(a, &.{ options.build_root, options.vulcan_zig_exe });
    defer a.free(compiler);
    var child = try std.process.spawn(io, .{
        .argv = &.{
            compiler,            "build-obj",  "--cache-dir", ".zig-cache", "--name", "server.test",
            "-Mroot=server.zig", "--listen=-",
        },
        .cwd = .{ .dir = tmp.dir },
        .stdin = .pipe,
        .stdout = .pipe,
    });
    defer child.kill(io);
    var input_buffer: [4096]u8 = undefined;
    var input = child.stdout.?.readerStreaming(io, &input_buffer);
    var output_buffer: [128]u8 = undefined;
    var output_writer = child.stdin.?.writerStreaming(io, &output_buffer);
    var client: std.zig.Client = .{ .in = &input.interface, .out = &output_writer.interface };

    try client.serveBodylessMessage(.update);
    const first = try receiveUpdate(a, &client);
    try std.testing.expectEqual(@as(u32, 0), first.errors);
    try runServerArtifact(a, io, &tmp, first.digest orelse return error.MissingArtifactDigest, 42);

    try tmp.dir.writeFile(io, .{
        .sub_path = "server.zig",
        .data = "export fn mainCode() u64 { return 43; }",
    });
    try client.serveBodylessMessage(.update);
    const second = try receiveUpdate(a, &client);
    try std.testing.expectEqual(@as(u32, 0), second.errors);
    try runServerArtifact(a, io, &tmp, second.digest orelse return error.MissingArtifactDigest, 43);

    try tmp.dir.writeFile(io, .{ .sub_path = "server.zig", .data = "export fn mainCode( {" });
    try client.serveBodylessMessage(.update);
    const broken = try receiveUpdate(a, &client);
    try std.testing.expect(broken.errors > 0);
    try std.testing.expect(broken.digest == null);

    try tmp.dir.writeFile(io, .{
        .sub_path = "server.zig",
        .data = "export fn mainCode() u64 { return 44; }",
    });
    try client.serveBodylessMessage(.update);
    const recovered = try receiveUpdate(a, &client);
    try std.testing.expectEqual(@as(u32, 0), recovered.errors);
    try runServerArtifact(a, io, &tmp, recovered.digest orelse return error.MissingArtifactDigest, 44);
    try client.serveBodylessMessage(.exit);
    const term = try child.wait(io);
    try std.testing.expect(term == .exited and term.exited == 0);
}

test "Maker compiles and runs supported unit tests through the frontend" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "build.zig", .data =
        \\const std = @import("std");
        \\pub fn build(b: *std.Build) void {
        \\    const target = b.standardTargetOptions(.{});
        \\    const optimize = b.standardOptimizeOption(.{});
        \\    const tests = b.addTest(.{ .root_module = b.createModule(.{
        \\        .root_source_file = b.path("lib.zig"), .target = target, .optimize = optimize,
        \\    }) });
        \\    b.step("test", "Run tests").dependOn(&b.addRunArtifact(tests).step);
        \\}
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "lib.zig", .data =
        \\fn add(a: u64, b: u64) u64 { return a + b; }
        \\test "adds numbers" { _ = add(2, 3); }
        \\test "keeps locals" { const answer: u64 = 42; _ = answer; }
    });
    const compiler = try std.fs.path.resolve(a, &.{ options.build_root, options.vulcan_zig_exe });
    defer a.free(compiler);
    try runExpect(a, io, &tmp, &.{ compiler, "build", "test", "--summary", "none" }, 0);

    // A test body outside the subset must fail the build with a source location,
    // not silently report the test as passed.
    try tmp.dir.writeFile(io, .{ .sub_path = "lib.zig", .data =
        \\test "unsupported" { const value: c_int = 0; _ = value; }
    });
    const rejected = try std.process.run(a, io, .{
        .argv = &.{ compiler, "build", "test", "--summary", "none" },
        .cwd = .{ .dir = tmp.dir },
    });
    defer a.free(rejected.stdout);
    defer a.free(rejected.stderr);
    try std.testing.expect(rejected.term == .exited and rejected.term.exited != 0);
    try std.testing.expect(std.mem.find(u8, rejected.stderr, "lib.zig:1:") != null);

    try tmp.dir.writeFile(io, .{ .sub_path = "lib.zig", .data =
        \\fn add(a: u64, b: u64) u64 { return a + b; }
        \\test "adds numbers" { _ = add(2, 3); }
    });
    try runExpect(a, io, &tmp, &.{ compiler, "build", "test", "--summary", "none" }, 0);
}

test "Maker analysis-only test steps work without an emitted artifact" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "build.zig", .data =
        \\const std = @import("std");
        \\pub fn build(b: *std.Build) void {
        \\    const target = b.standardTargetOptions(.{});
        \\    const optimize = b.standardOptimizeOption(.{});
        \\    const tests = b.addTest(.{ .root_module = b.createModule(.{
        \\        .root_source_file = b.path("lib.zig"), .target = target, .optimize = optimize,
        \\    }) });
        \\    tests.generated_bin = .none;
        \\    b.step("analyze", "Analyze tests").dependOn(&tests.step);
        \\}
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "lib.zig", .data =
        \\fn add(a: u64, b: u64) u64 { return a + b; }
        \\test "analyzed" { _ = add(4, 5); }
    });
    const compiler = try std.fs.path.resolve(a, &.{ options.build_root, options.vulcan_zig_exe });
    defer a.free(compiler);
    try runExpect(a, io, &tmp, &.{ compiler, "build", "analyze", "--summary", "none" }, 0);

    try tmp.dir.writeFile(io, .{ .sub_path = "lib.zig", .data =
        \\test "unsupported" { const value: c_int = 0; _ = value; }
    });
    const rejected = try std.process.run(a, io, .{
        .argv = &.{ compiler, "build", "analyze", "--summary", "none" },
        .cwd = .{ .dir = tmp.dir },
    });
    defer a.free(rejected.stdout);
    defer a.free(rejected.stderr);
    try std.testing.expect(rejected.term == .exited and rejected.term.exited != 0);
}

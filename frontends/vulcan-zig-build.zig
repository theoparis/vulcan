const std = @import("std");
const options = @import("build_options");

pub fn run(init: std.process.Init, args: []const []const u8) !void {
    const a = init.arena.allocator();
    const executable = try std.process.executablePathAlloc(init.io, a);
    const directory = std.fs.path.dirname(executable) orelse return error.InvalidExecutablePath;
    const argv = try a.alloc([]const u8, args.len + 6);
    argv[0] = try std.fs.path.join(a, &.{ directory, "vulcan-zig-maker" });
    argv[1] = "build";
    argv[2] = try std.fmt.allocPrint(a, "--zig-lib={s}", .{options.zig_lib_dir});
    argv[3] = try std.fmt.allocPrint(a, "--zig={s}", .{executable});
    argv[4] = try std.fmt.allocPrint(a, "--global-cache={s}", .{init.environ_map.get("ZIG_GLOBAL_CACHE_DIR") orelse options.global_cache_dir});
    argv[5] = "--seed=0";
    @memcpy(argv[6..], args);
    try spawnWait(init.io, argv);
}

pub fn isConfigure(init: std.process.Init, args: []const []const u8) !bool {
    if (args.len == 0 or !std.mem.eql(u8, args[0], "build-exe")) return false;
    var root: ?[]const u8 = null;
    var lib: []const u8 = options.zig_lib_dir;
    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.startsWith(u8, arg, "-Mroot=")) {
            root = arg["-Mroot=".len..];
        } else if (std.mem.eql(u8, arg, "--zig-lib-dir")) {
            index += 1;
            if (index == args.len) return false;
            lib = args[index];
        } else if (std.mem.startsWith(u8, arg, "--zig-lib-dir=")) {
            lib = arg["--zig-lib-dir=".len..];
        }
    }
    const path = root orelse return false;
    if (!std.mem.eql(u8, std.fs.path.basename(path), "configurer.zig")) return false;
    const a = init.arena.allocator();
    const cwd = try std.process.currentPathAlloc(init.io, a);
    const actual = try std.fs.path.resolve(a, &.{ cwd, path });
    const expected = try std.fs.path.resolve(a, &.{ cwd, lib, "compiler/configurer.zig" });
    return std.mem.eql(u8, actual, expected);
}

pub fn configure(init: std.process.Init, args: []const []const u8) !void {
    const a = init.arena.allocator();
    const cwd = try std.process.currentPathAlloc(init.io, a);
    const upstream = try std.fs.path.resolve(a, &.{ cwd, options.zig_exe });
    const executable = try std.process.executablePathAlloc(init.io, a);
    if (std.mem.eql(u8, upstream, executable)) return error.RecursiveBootstrap;
    const argv = try a.alloc([]const u8, args.len + 1);
    argv[0] = options.zig_exe;
    @memcpy(argv[1..], args);
    try spawnWait(init.io, argv);
}

fn spawnWait(io: std.Io, argv: []const []const u8) !void {
    var child = try std.process.spawn(io, .{ .argv = argv });
    defer child.kill(io);
    const term = try child.wait(io);
    switch (term) {
        .exited => |code| if (code != 0) std.process.exit(code),
        else => std.process.exit(1),
    }
}

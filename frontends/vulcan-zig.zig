//! vulcan-zig: compile the supported Zig subset to ELF objects and executables.
//!
//! Usage:
//!   vulcan-zig build-exe [options] <input.zig> [object.o ...]
//!   vulcan-zig build-obj [options] <input.zig>
//! Options: -target <triple>, -T <script>, -femit-bin[=path], --entry <symbol>, -Mname=path

const std = @import("std");
const builtin = @import("builtin");
const zigc = @import("vulcan-zig");
const target = @import("vulcan-target");
const link = @import("vulcan-link");
const build_options = @import("build_options");

const Command = enum { exe, obj };
const CodeTarget = struct { source: zigc.TargetArch, arch: link.Arch };

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    var errbuf: [1024]u8 = undefined;
    var errfile = std.Io.File.stderr().writer(io, &errbuf);
    const errw = &errfile.interface;

    var it = try init.minimal.args.iterateAllocator(allocator);
    defer it.deinit();
    _ = it.skip(); // argv0
    const command_arg = it.next() orelse {
        try printUsage(errw);
        return error.Usage;
    };
    const command: Command = if (std.mem.eql(u8, command_arg, "build-exe"))
        .exe
    else if (std.mem.eql(u8, command_arg, "build-obj"))
        .obj
    else if (std.mem.eql(u8, command_arg, "--help") or std.mem.eql(u8, command_arg, "-h")) {
        try printUsage(errw);
        return;
    } else {
        errw.print("error: unknown command '{s}'\n", .{command_arg}) catch {};
        try printUsage(errw);
        return error.Usage;
    };

    var source_path_arg: ?[]const u8 = null;
    var object_paths: std.ArrayList([]const u8) = .empty;
    defer object_paths.deinit(allocator);
    var output: ?[]const u8 = null;
    var entry: []const u8 = "_start";
    var entry_explicit = false;
    var script_path: ?[]const u8 = null;
    var target_triple: ?[]const u8 = null;
    var module_mappings: std.ArrayList(zigc.ModuleMapping) = .empty;
    defer module_mappings.deinit(allocator);
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "-target") or std.mem.eql(u8, arg, "--target")) {
            target_triple = it.next() orelse {
                try printUsage(errw);
                return error.Usage;
            };
        } else if (std.mem.startsWith(u8, arg, "-target=")) {
            target_triple = arg["-target=".len..];
        } else if (std.mem.eql(u8, arg, "-femit-bin")) {
            // Without an explicit path, use the command's normal output name.
        } else if (std.mem.startsWith(u8, arg, "-femit-bin=")) {
            output = arg["-femit-bin=".len..];
            if (output.?.len == 0) {
                errw.writeAll("error: -femit-bin requires a non-empty path\n") catch {};
                errw.flush() catch {};
                return error.Usage;
            }
        } else if (std.mem.eql(u8, arg, "-T")) {
            script_path = it.next() orelse {
                try printUsage(errw);
                return error.Usage;
            };
        } else if (std.mem.startsWith(u8, arg, "-T") and arg.len > 2) {
            script_path = arg[2..];
        } else if (std.mem.eql(u8, arg, "--entry") or std.mem.eql(u8, arg, "-e")) {
            entry = it.next() orelse {
                try printUsage(errw);
                return error.Usage;
            };
            entry_explicit = true;
        } else if (std.mem.startsWith(u8, arg, "-fentry=")) {
            entry = arg["-fentry=".len..];
            entry_explicit = true;
        } else if (std.mem.startsWith(u8, arg, "-M")) {
            const spec = arg[2..];
            const equals = std.mem.indexOfScalar(u8, spec, '=') orelse {
                errw.print("error: module option '{s}' must be -Mname=path\n", .{arg}) catch {};
                errw.flush() catch {};
                return error.Usage;
            };
            const name = spec[0..equals];
            const path = spec[equals + 1 ..];
            if (name.len == 0 or path.len == 0) {
                errw.print("error: module option '{s}' must have a non-empty name and path\n", .{arg}) catch {};
                errw.flush() catch {};
                return error.Usage;
            }
            for (module_mappings.items) |module| {
                if (std.mem.eql(u8, module.name, name)) {
                    errw.print("error: duplicate module name '{s}'\n", .{name}) catch {};
                    errw.flush() catch {};
                    return error.Usage;
                }
            }
            try module_mappings.append(allocator, .{ .name = name, .path = path });
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printUsage(errw);
            return;
        } else if (std.mem.startsWith(u8, arg, "-")) {
            errw.print("error: unsupported option '{s}'\n", .{arg}) catch {};
            errw.flush() catch {};
            return error.Usage;
        } else if (std.mem.endsWith(u8, arg, ".o")) {
            if (command != .exe) {
                errw.writeAll("error: build-obj accepts one Zig source file, not object inputs\n") catch {};
                errw.flush() catch {};
                return error.Usage;
            }
            try object_paths.append(allocator, arg);
        } else if (source_path_arg == null) {
            source_path_arg = arg;
        } else {
            try printUsage(errw);
            return error.Usage;
        }
    }
    const source_path = source_path_arg orelse {
        try printUsage(errw);
        return error.Usage;
    };
    if (script_path != null and command != .exe) {
        errw.writeAll("error: -T requires build-exe\n") catch {};
        errw.flush() catch {};
        return error.Usage;
    }

    const selected_target = if (target_triple) |triple|
        targetFromTriple(triple) orelse {
            errw.print("error: unsupported target '{s}' (supported architectures: riscv64, aarch64, x86_64, x86)\n", .{triple}) catch {};
            errw.flush() catch {};
            return error.Usage;
        }
    else
        hostTarget() orelse {
            errw.print("error: unsupported host architecture: {t}\n", .{builtin.cpu.arch}) catch {};
            errw.flush() catch {};
            return error.Usage;
        };
    const std_dir = try std.fmt.allocPrint(allocator, "{s}/std", .{build_options.zig_lib_dir});

    var mod = zigc.compileFileOptions(allocator, io, source_path, selected_target.source, .{
        .std_dir = std_dir,
        .modules = module_mappings.items,
    }) catch |err| {
        errw.print("error: {s}: failed to compile: {t}\n", .{ source_path, err }) catch {};
        errw.flush() catch {};
        return err;
    };
    defer mod.deinit(allocator);

    if (command == .exe and script_path == null and mod.find(entry) == null and object_paths.items.len == 0) {
        errw.print("error: no entry function named '{s}' in {s}\n", .{ entry, source_path }) catch {};
        errw.flush() catch {};
        return error.Usage;
    }

    const funcs = try allocator.alloc(target.native.ModuleFunction, mod.funcs.len);
    defer allocator.free(funcs);
    for (mod.funcs, 0..) |*function, i| funcs[i] = .{ .name = function.name, .func = &function.func };
    const data = try allocator.alloc(target.native.ObjData, mod.data.len);
    defer allocator.free(data);
    for (mod.data, 0..) |object, i| data[i] = .{
        .name = object.name,
        .bytes = object.bytes,
        .kind = switch (object.kind) {
            .rodata => .rodata,
            .data => .data,
            .bss => .bss,
        },
        .size = object.size,
    };

    const object_bytes = target.native.writeObjectDataForWithIo(allocator, selected_target.arch, funcs, data, io) catch |err| {
        errw.print("error: {s}: failed to emit object: {s}\n", .{ source_path, @errorName(err) }) catch {};
        errw.flush() catch {};
        return err;
    };
    defer allocator.free(object_bytes);

    const output_path = output orelse try defaultOutputName(allocator, source_path, command);
    if (command == .obj) {
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = output_path, .data = object_bytes }) catch |err| {
            errw.print("error: cannot write '{s}': {s}\n", .{ output_path, @errorName(err) }) catch {};
            errw.flush() catch {};
            return err;
        };
        return;
    }

    const objects = try allocator.alloc([]const u8, object_paths.items.len + 1);
    defer allocator.free(objects);
    objects[0] = object_bytes;
    for (object_paths.items, 1..) |path, i| {
        objects[i] = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024)) catch |err| {
            errw.print("error: cannot read object '{s}': {s}\n", .{ path, @errorName(err) }) catch {};
            errw.flush() catch {};
            return err;
        };
    }
    if (script_path) |path| {
        const script_bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024)) catch |err| {
            errw.print("error: cannot read linker script '{s}': {s}\n", .{ path, @errorName(err) }) catch {};
            errw.flush() catch {};
            return err;
        };
        defer allocator.free(script_bytes);

        var diagnostic: link.ScriptDiagnostic = .{ .line = 0, .col = 0, .msg = "" };
        var script = link.parseScript(allocator, script_bytes, &diagnostic) catch |err| {
            if (err == error.OutOfMemory) return err;
            errw.print("error: {s}:{d}:{d}: {s}\n", .{ path, diagnostic.line, diagnostic.col, diagnostic.msg }) catch {};
            errw.flush() catch {};
            return err;
        };
        defer script.deinit();

        const inputs = try allocator.alloc(link.Input, objects.len);
        defer allocator.free(inputs);
        for (objects, 0..) |object, i| inputs[i] = .{ .object = object };
        var linked = link.linkInputsScript(allocator, inputs, &script, null) catch |err| {
            errw.print("error: script link failed: {s}\n", .{@errorName(err)}) catch {};
            errw.flush() catch {};
            return err;
        };
        defer linked.deinit(allocator);

        const entry_addr = if (entry_explicit)
            link.elf.findSymbol(linked.placement.symbols, entry) orelse {
                errw.print("error: undefined entry symbol '{s}'\n", .{entry}) catch {};
                errw.flush() catch {};
                return error.Usage;
            }
        else
            linked.entry;
        const executable = try link.writeElfSegments(linked.arch, allocator, &linked.placement, entry_addr);
        defer allocator.free(executable);
        std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = output_path,
            .data = executable,
            .flags = .{ .permissions = .executable_file },
        }) catch |err| {
            errw.print("error: cannot write '{s}': {s}\n", .{ output_path, @errorName(err) }) catch {};
            errw.flush() catch {};
            return err;
        };
        return;
    }

    const base: u64 = if (selected_target.arch == .x86) 0x08048000 else 0x400000;
    var image = link.linkObjects(allocator, objects, base) catch |err| {
        errw.print("error: failed to link: {s}\n", .{@errorName(err)}) catch {};
        errw.flush() catch {};
        return err;
    };
    defer image.deinit(allocator);

    const executable = link.writeExecutable(selected_target.arch, allocator, &image, entry) catch |err| {
        errw.print("error: failed to write executable: {s}\n", .{@errorName(err)}) catch {};
        errw.flush() catch {};
        return err;
    };
    defer allocator.free(executable);

    std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = output_path,
        .data = executable,
        .flags = .{ .permissions = .executable_file },
    }) catch |err| {
        errw.print("error: cannot write '{s}': {s}\n", .{ output_path, @errorName(err) }) catch {};
        errw.flush() catch {};
        return err;
    };
}

fn hostTarget() ?CodeTarget {
    return switch (builtin.cpu.arch) {
        .aarch64 => .{ .source = .aarch64, .arch = .aarch64 },
        .x86_64 => .{ .source = .x86_64, .arch = .x86_64 },
        .x86 => .{ .source = .x86, .arch = .x86 },
        .riscv64 => .{ .source = .riscv64, .arch = .riscv64 },
        else => null,
    };
}

fn targetFromTriple(triple: []const u8) ?CodeTarget {
    const arch_name = if (std.mem.indexOfScalar(u8, triple, '-')) |dash| triple[0..dash] else triple;
    if (std.mem.eql(u8, arch_name, "riscv64")) return .{ .source = .riscv64, .arch = .riscv64 };
    if (std.mem.eql(u8, arch_name, "aarch64")) return .{ .source = .aarch64, .arch = .aarch64 };
    if (std.mem.eql(u8, arch_name, "x86_64")) return .{ .source = .x86_64, .arch = .x86_64 };
    if (std.mem.eql(u8, arch_name, "x86")) return .{ .source = .x86, .arch = .x86 };
    return null;
}

fn printUsage(errw: *std.Io.Writer) !void {
    try errw.writeAll(
        \\usage: vulcan-zig build-exe [options] <input.zig> [object.o ...]
        \\       vulcan-zig build-obj [options] <input.zig>
        \\  -target <triple>    target architecture (default: host)
        \\  -T <script>        use a linker script for executable layout
        \\  -Mname=path        map @import("name") to a Zig source file
        \\  -femit-bin[=path]  output path (default: input basename)
        \\  --entry <symbol>   executable entry symbol (default: _start)
        \\
    );
    try errw.flush();
}

fn defaultOutputName(allocator: std.mem.Allocator, input: []const u8, command: Command) ![]const u8 {
    const base = std.fs.path.basename(input);
    const stem = if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| base[0..dot] else base;
    return if (command == .obj)
        std.fmt.allocPrint(allocator, "{s}.o", .{stem})
    else
        allocator.dupe(u8, stem);
}

//! vulcan-zig: compile the supported Zig subset to ELF objects and executables.
//!
//!   vulcan-zig build-exe [options] <input.zig> [object.o ...]
//!   vulcan-zig build-obj [options] <input.zig>
//!   vulcan-zig build-exe [options] [-Mname=path ...]
//!   vulcan-zig build-obj [options] [-Mname=path ...]
//! Options: --dep name|alias=module applies to the next -M; first -M is root.
//! Other options: -target <triple>, -T <script>, -femit-bin[=path], --entry <symbol>

const std = @import("std");
const builtin = @import("builtin");
const zigc = @import("vulcan-zig");
const target = @import("vulcan-target");
const link = @import("vulcan-link");
const build_options = @import("build_options");
const build_driver = @import("vulcan-zig-build");

const Command = enum { exe, obj, tests };
const CodeTarget = struct { source: zigc.TargetArch, arch: link.Arch };

/// Distinguishes producing an artifact from executing the tests it describes.
const Mode = enum { build, test_run };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len > 1) {
        if (std.mem.eql(u8, args[1], "__test-run")) return compileOnce(init, null, null, .test_run);
        if (std.mem.eql(u8, args[1], "build")) return build_driver.run(init, args[2..]);
        if (try build_driver.isConfigure(init, args[1..])) return build_driver.configure(init, args[1..]);
        for (args[2..]) |arg| {
            if (std.mem.eql(u8, arg, "--listen=-")) return serveUpdates(init, args[1]);
        }
    }
    return compileOnce(init, null, null, .build);
}

const ServerResult = struct {
    digest: std.Build.Cache.BinDigest,
    sources: []const []const u8,
    emit: bool = true,
};

fn compileOnce(init: std.process.Init, result: ?*ServerResult, diagnostics: ?*std.Io.Writer, mode: Mode) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    var errbuf: [1024]u8 = undefined;
    var errfile = std.Io.File.stderr().writer(io, &errbuf);
    const errw = diagnostics orelse &errfile.interface;

    var it = try init.minimal.args.iterateAllocator(allocator);
    defer it.deinit();
    _ = it.skip(); // argv0
    var command_arg = it.next() orelse {
        try printUsage(errw);
        return error.Usage;
    };
    if (mode == .test_run) {
        if (!std.mem.eql(u8, command_arg, "__test-run")) return error.InvalidTestRunnerInvocation;
        command_arg = it.next() orelse return error.Usage;
    }
    const command: Command = if (std.mem.eql(u8, command_arg, "build-exe"))
        .exe
    else if (std.mem.eql(u8, command_arg, "build-obj"))
        .obj
    else if (std.mem.eql(u8, command_arg, "test"))
        .tests
    else if (std.mem.eql(u8, command_arg, "test-obj")) {
        errw.writeAll("error: test-obj is not supported; use object tests via `test` in a separate module\n") catch {};
        errw.flush() catch {};
        return error.Usage;
    } else if (std.mem.eql(u8, command_arg, "--help") or std.mem.eql(u8, command_arg, "-h")) {
        if (result != null) return error.UnsupportedServerOption;
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
    var depfile_path: ?[]const u8 = null;
    var entry: []const u8 = "_start";
    var entry_explicit = false;
    var script_path: ?[]const u8 = null;
    var target_triple: ?[]const u8 = null;
    var artifact_name: ?[]const u8 = null;
    var zig_lib_dir: []const u8 = build_options.zig_lib_dir;
    var include_paths: std.ArrayList([]const u8) = .empty;
    defer include_paths.deinit(allocator);
    var module_mappings: std.ArrayList(zigc.ModuleMapping) = .empty;
    defer module_mappings.deinit(allocator);
    var pending_dependencies: std.ArrayList(zigc.ModuleDependency) = .empty;
    defer pending_dependencies.deinit(allocator);
    var cache_dir: ?[]const u8 = null;
    var global_cache_dir: ?[]const u8 = null;
    var build_root: ?[]const u8 = null;
    var server_mode = false;
    var emit_bin = true;
    var test_filters: std.ArrayList([]const u8) = .empty;
    defer test_filters.deinit(allocator);
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "-target") or std.mem.eql(u8, arg, "--target")) {
            const next_target = it.next() orelse return error.Usage;
            if (target_triple) |previous| {
                if (!std.mem.eql(u8, previous, next_target)) return error.Usage;
            }
            target_triple = next_target;
        } else if (std.mem.eql(u8, arg, "-I")) {
            const path = it.next() orelse return error.Usage;
            if (path.len == 0) return error.Usage;
            try include_paths.append(allocator, path);
        } else if (std.mem.startsWith(u8, arg, "-I") and arg.len > 2) {
            try include_paths.append(allocator, arg[2..]);
        } else if (std.mem.eql(u8, arg, "-Odebug") or std.mem.eql(u8, arg, "-ODebug")) {} else if (std.mem.eql(u8, arg, "-O") or std.mem.startsWith(u8, arg, "-O")) {
            errw.print("error: unsupported optimization mode '{s}' (supported: -Odebug)\n", .{arg}) catch {};
            errw.flush() catch {};
            return error.Usage;
        } else if (std.mem.eql(u8, arg, "--name") or std.mem.startsWith(u8, arg, "--name=")) {
            artifact_name = if (std.mem.eql(u8, arg, "--name")) it.next() else arg["--name=".len..];
            if (artifact_name == null or artifact_name.?.len == 0) return error.Usage;
        } else if (std.mem.eql(u8, arg, "--zig-lib-dir") or std.mem.startsWith(u8, arg, "--zig-lib-dir=")) {
            zig_lib_dir = if (std.mem.eql(u8, arg, "--zig-lib-dir")) it.next() orelse return error.Usage else arg["--zig-lib-dir=".len..];
            if (zig_lib_dir.len == 0) return error.Usage;
        } else if (std.mem.eql(u8, arg, "--cache-dir") or std.mem.startsWith(u8, arg, "--cache-dir=")) {
            cache_dir = if (std.mem.eql(u8, arg, "--cache-dir")) it.next() orelse return error.Usage else arg["--cache-dir=".len..];
            if (cache_dir.?.len == 0) return error.Usage;
        } else if (std.mem.eql(u8, arg, "--global-cache-dir") or std.mem.startsWith(u8, arg, "--global-cache-dir=")) {
            global_cache_dir = if (std.mem.eql(u8, arg, "--global-cache-dir")) it.next() orelse return error.Usage else arg["--global-cache-dir=".len..];
            if (global_cache_dir.?.len == 0) return error.Usage;
        } else if (std.mem.eql(u8, arg, "--build-root") or std.mem.startsWith(u8, arg, "--build-root=")) {
            build_root = if (std.mem.eql(u8, arg, "--build-root")) it.next() orelse return error.Usage else arg["--build-root=".len..];
            if (build_root.?.len == 0) return error.Usage;
        } else if (std.mem.eql(u8, arg, "-fno-emit-bin")) {
            emit_bin = false;
        } else if (std.mem.eql(u8, arg, "--test-filter") or std.mem.startsWith(u8, arg, "--test-filter=")) {
            const filter = if (std.mem.eql(u8, arg, "--test-filter"))
                it.next() orelse return error.Usage
            else
                arg["--test-filter=".len..];
            if (filter.len == 0) return error.Usage;
            try test_filters.append(allocator, filter);
        } else if (std.mem.eql(u8, arg, "--test-runner") or std.mem.startsWith(u8, arg, "--test-runner=")) {
            errw.writeAll("error: custom test runners are not supported\n") catch {};
            errw.flush() catch {};
            return error.Usage;
        } else if (std.mem.eql(u8, arg, "--listen=-")) {
            // In test-runner mode this argument is the test protocol itself, which is
            // served separately; it does not select the compiler server.
            if (mode == .build) server_mode = true;
        } else if (std.mem.eql(u8, arg, "--seed") or std.mem.startsWith(u8, arg, "--seed=")) {
            if (std.mem.eql(u8, arg, "--seed")) _ = it.next() orelse return error.Usage;
        } else if (std.mem.startsWith(u8, arg, "-target=")) {
            const next_target = arg["-target=".len..];
            if (target_triple) |previous| {
                if (!std.mem.eql(u8, previous, next_target)) {
                    errw.writeAll("error: modules with differing target triples are unsupported\n") catch {};
                    errw.flush() catch {};
                    return error.Usage;
                }
            }
            target_triple = next_target;
        } else if (std.mem.eql(u8, arg, "-femit-bin")) {
            // Without an explicit path, use the command's normal output name.
        } else if (std.mem.startsWith(u8, arg, "-femit-bin=")) {
            output = arg["-femit-bin=".len..];
            if (output.?.len == 0) {
                errw.writeAll("error: -femit-bin requires a non-empty path\n") catch {};
                errw.flush() catch {};
                return error.Usage;
            }
        } else if (std.mem.eql(u8, arg, "--depfile") or std.mem.startsWith(u8, arg, "--depfile=")) {
            const path = if (std.mem.eql(u8, arg, "--depfile"))
                it.next() orelse {
                    errw.writeAll("error: --depfile requires a path\n") catch {};
                    errw.flush() catch {};
                    return error.Usage;
                }
            else
                arg["--depfile=".len..];
            if (path.len == 0) {
                errw.writeAll("error: --depfile requires a non-empty path\n") catch {};
                errw.flush() catch {};
                return error.Usage;
            }
            depfile_path = path;
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
        } else if (std.mem.eql(u8, arg, "--dep")) {
            const spec = it.next() orelse {
                errw.writeAll("error: --dep requires name or alias=module\n") catch {};
                errw.flush() catch {};
                return error.Usage;
            };
            const equals = std.mem.indexOfScalar(u8, spec, '=');
            const alias = if (equals) |at| spec[0..at] else spec;
            const module = if (equals) |at| spec[at + 1 ..] else spec;
            if (alias.len == 0 or module.len == 0 or std.mem.indexOfScalar(u8, module, '=') != null) {
                errw.print("error: malformed dependency '{s}'; expected name or alias=module\n", .{spec}) catch {};
                errw.flush() catch {};
                return error.Usage;
            }
            for (pending_dependencies.items) |dependency| {
                if (std.mem.eql(u8, dependency.name, alias)) {
                    errw.print("error: duplicate dependency alias '{s}'\n", .{alias}) catch {};
                    errw.flush() catch {};
                    return error.Usage;
                }
            }
            try pending_dependencies.append(allocator, .{ .name = alias, .module = module });
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
            const dependencies = try allocator.dupe(zigc.ModuleDependency, pending_dependencies.items);
            try module_mappings.append(allocator, .{
                .name = name,
                .path = path,
                .dependencies = dependencies,
            });
            pending_dependencies.clearRetainingCapacity();
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            if (result != null) return error.UnsupportedServerOption;
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
    if (pending_dependencies.items.len != 0) {
        errw.writeAll("error: --dep dependencies must be followed by a -M module\n") catch {};
        errw.flush() catch {};
        return error.Usage;
    }
    if (module_mappings.items.len != 0 and source_path_arg != null) {
        errw.writeAll("error: positional Zig source cannot be combined with -M modules\n") catch {};
        errw.flush() catch {};
        return error.Usage;
    }
    const source_path = if (module_mappings.items.len != 0)
        module_mappings.items[0].path
    else
        source_path_arg orelse {
            try printUsage(errw);
            return error.Usage;
        };
    if (result) |server_result| {
        server_result.sources = &.{source_path};
    }
    if (module_mappings.items.len != 0) {
        for (module_mappings.items) |module| {
            for (module.dependencies) |dependency| {
                var found = false;
                for (module_mappings.items) |candidate| {
                    if (std.mem.eql(u8, candidate.name, dependency.module)) found = true;
                }
                if (!found) {
                    errw.print("error: module '{s}' depends on undefined module '{s}'\n", .{ module.name, dependency.module }) catch {};
                    errw.flush() catch {};
                    return error.Usage;
                }
            }
        }
    }
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
    const std_dir = try std.fmt.allocPrint(allocator, "{s}/std", .{zig_lib_dir});
    var diagnostic_path: [std.fs.max_path_bytes]u8 = undefined;
    var compile_diagnostic: zigc.Diagnostic = .{ .path_buffer = &diagnostic_path };
    var mod = zigc.compileFileOptions(allocator, io, source_path, selected_target.source, .{
        .std_dir = std_dir,
        .modules = module_mappings.items,
        .root_module = if (module_mappings.items.len != 0) module_mappings.items[0].name else null,
        .collect_source_files = depfile_path != null or server_mode,
        .diagnostic = &compile_diagnostic,
        .tests = command == .tests,
        .test_filters = test_filters.items,
    }) catch |err| {
        if (compile_diagnostic.line != 0) {
            errw.print("error: {s}:{d}:{d}: failed to compile: {t}\n", .{
                compile_diagnostic.path(), compile_diagnostic.line, compile_diagnostic.column, err,
            }) catch {};
        } else {
            errw.print("error: {s}: failed to compile: {t}\n", .{ source_path, err }) catch {};
        }
        errw.flush() catch {};
        return err;
    };
    defer mod.deinit(allocator);

    if (command == .tests) {
        if (mode == .test_run) return serveTestProtocol(init, allocator, &mod);
        if (script_path != null or object_paths.items.len != 0) {
            errw.writeAll("error: test accepts one Zig module graph, not linker inputs\n") catch {};
            errw.flush() catch {};
            return error.Usage;
        }
        const test_name = if (artifact_name) |name| blk: {
            if (std.mem.findAny(u8, name, "/\\") != null or
                std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return error.InvalidArtifactName;
            break :blk name;
        } else try defaultOutputName(allocator, source_path, command);
        if (!emit_bin) {
            if (depfile_path) |path| try writeDepfile(io, path, test_name, mod.source_files, &.{}, null);
            if (result) |server_result| {
                server_result.emit = false;
                server_result.sources = try serverSources(allocator, &mod, object_paths.items, script_path);
            }
            return;
        }
        const script = try testRunnerScript(allocator, io, try init.minimal.args.toSlice(allocator));
        const test_output_path = if (server_mode) blk: {
            if (output != null) return error.UnsupportedServerOutput;
            const cache_output = try cacheOutputPath(
                allocator,
                io,
                cache_dir orelse return error.MissingCacheDir,
                test_name,
                mod.source_files,
                object_paths.items,
                script_path,
                target_triple,
                entry,
                entry_explicit,
                command,
                module_mappings.items,
                global_cache_dir,
                build_root,
                zig_lib_dir,
                include_paths.items,
            );
            if (result) |server_result| {
                server_result.digest = cache_output.digest;
                server_result.sources = try serverSources(allocator, &mod, object_paths.items, script_path);
            }
            break :blk cache_output.path;
        } else output orelse test_name;
        std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = test_output_path,
            .data = script,
            .flags = .{ .permissions = .executable_file },
        }) catch |err| {
            errw.print("error: cannot write '{s}': {s}\n", .{ test_output_path, @errorName(err) }) catch {};
            errw.flush() catch {};
            return err;
        };
        if (depfile_path) |path| try writeDepfile(io, path, test_output_path, mod.source_files, &.{}, null);
        return;
    }

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

    const object_bytes = target.native.writeObjectDataFor(allocator, selected_target.arch, funcs, data) catch |err| {
        errw.print("error: {s}: failed to emit object: {s}\n", .{ source_path, @errorName(err) }) catch {};
        errw.flush() catch {};
        return err;
    };
    defer allocator.free(object_bytes);

    const default_name = if (artifact_name) |name| blk: {
        if (std.mem.findAny(u8, name, "/\\") != null or
            std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return error.InvalidArtifactName;
        break :blk if (command == .obj)
            try std.fmt.allocPrint(allocator, "{s}.o", .{name})
        else
            name;
    } else try defaultOutputName(allocator, source_path, command);
    const output_path = if (server_mode) blk: {
        if (output != null) return error.UnsupportedServerOutput;
        const cache_output = try cacheOutputPath(
            allocator,
            io,
            cache_dir orelse return error.MissingCacheDir,
            default_name,
            mod.source_files,
            object_paths.items,
            script_path,
            target_triple,
            entry,
            entry_explicit,
            command,
            module_mappings.items,
            global_cache_dir,
            build_root,
            zig_lib_dir,
            include_paths.items,
        );
        if (result) |server_result| {
            server_result.digest = cache_output.digest;
            server_result.sources = try serverSources(allocator, &mod, object_paths.items, script_path);
        }
        break :blk cache_output.path;
    } else output orelse default_name;
    if (command == .obj) {
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = output_path, .data = object_bytes }) catch |err| {
            errw.print("error: cannot write '{s}': {s}\n", .{ output_path, @errorName(err) }) catch {};
            errw.flush() catch {};
            return err;
        };
        if (depfile_path) |path| try writeDepfile(io, path, output_path, mod.source_files, object_paths.items, script_path);
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
        if (depfile_path) |dependency_path| try writeDepfile(io, dependency_path, output_path, mod.source_files, object_paths.items, script_path);
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
    if (depfile_path) |path| try writeDepfile(io, path, output_path, mod.source_files, object_paths.items, script_path);
}

fn serveUpdates(init: std.process.Init, command: []const u8) !void {
    const io = init.io;
    var in_buffer: [4096]u8 = undefined;
    var input = std.Io.File.stdin().readerStreaming(io, &in_buffer);
    var out_buffer: [4096]u8 = undefined;
    var output = std.Io.File.stdout().writerStreaming(io, &out_buffer);
    var server: std.zig.Server = .{ .in = &input.interface, .out = &output.interface };
    try server.serveStringMessage(.zig_version, builtin.zig_version_string);
    try server.serveConfig(.{ .flags = .{
        .output_mode = if (std.mem.eql(u8, command, "build-exe") or std.mem.eql(u8, command, "test")) .Exe else .Obj,
        .link_mode = .static,
        .link_libc = false,
        .link_libcpp = false,
        .link_libunwind = false,
        .pie = false,
    } });

    while (true) {
        _ = server.in.peekByte() catch |err| switch (err) {
            error.EndOfStream => return,
            else => return err,
        };
        const message = try server.receiveMessage();
        if (message.bytes_len != 0) return error.InvalidServerMessage;
        switch (message.tag) {
            .exit => return,
            .update => {
                var update_arena = std.heap.ArenaAllocator.init(init.gpa);
                defer update_arena.deinit();
                var update_init = init;
                update_init.arena = &update_arena;
                var diagnostics: std.Io.Writer.Allocating = .init(update_arena.allocator());
                defer diagnostics.deinit();
                var result: ServerResult = .{ .digest = undefined, .sources = &.{}, .emit = true };
                compileOnce(update_init, &result, &diagnostics.writer, .build) catch |err| {
                    var wip = try std.zig.ErrorBundle.Wip.init(init.gpa);
                    defer wip.deinit();
                    var message_text = std.mem.trimEnd(u8, diagnostics.written(), "\n");
                    if (std.mem.startsWith(u8, message_text, "error: ")) message_text = message_text["error: ".len..];
                    const message_index = if (message_text.len == 0)
                        try wip.printString("compilation failed: {s}", .{@errorName(err)})
                    else
                        try wip.printString("{s}", .{message_text});
                    try wip.addRootErrorMessage(.{ .msg = message_index });
                    var bundle = try wip.toOwnedBundle("");
                    defer bundle.deinit(init.gpa);
                    try server.serveErrorBundle(.error_bundle, bundle);
                    continue;
                };
                if (result.emit) try server.serveEmitDigest(&result.digest, .{ .flags = .{ .cache_hit = false } });
                try serveBuildInputs(&server, result);
                try server.serveErrorBundle(.error_bundle, .empty);
            },
            else => return error.InvalidServerMessage,
        }
    }
}

/// The executable a `test` compile emits is a launcher: tests run through the JIT of
/// this same compiler, so the test bodies are still the IR lowered from the project's
/// source, not upstream output.
fn testRunnerScript(allocator: std.mem.Allocator, io: std.Io, argv: []const [:0]const u8) ![]u8 {
    const compiler = try std.process.executablePathAlloc(io, allocator);
    const cwd_path = try std.process.currentPathAlloc(io, allocator);
    var script: std.Io.Writer.Allocating = .init(allocator);
    errdefer script.deinit();
    const writer = &script.writer;
    try writer.writeAll("#!/bin/sh\ncd ");
    try writeShellQuoted(writer, cwd_path);
    try writer.writeAll(" || exit 1\nexec ");
    try writeShellQuoted(writer, compiler);
    try writer.writeAll(" __test-run");
    for (argv[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--listen=-")) continue;
        try writer.writeByte(' ');
        try writeShellQuoted(writer, arg);
    }
    try writer.writeAll(" \"$@\"\n");
    return script.toOwnedSlice();
}

fn writeShellQuoted(writer: *std.Io.Writer, text: []const u8) !void {
    try writer.writeByte('\'');
    for (text) |byte| {
        if (byte == '\'') {
            try writer.writeAll("'\\''");
        } else {
            try writer.writeByte(byte);
        }
    }
    try writer.writeByte('\'');
}

fn serverSources(
    allocator: std.mem.Allocator,
    mod: *const zigc.Module,
    objects: []const []const u8,
    script: ?[]const u8,
) ![]const []const u8 {
    const sources = try allocator.alloc([]const u8, mod.source_files.len + objects.len + @intFromBool(script != null));
    var index: usize = 0;
    // `Module.deinit` poisons its arena-owned path strings before the server
    // finishes writing the response, so keep a copy of each.
    for (mod.source_files) |source| {
        sources[index] = try allocator.dupe(u8, source);
        index += 1;
    }
    for (objects) |object| {
        sources[index] = object;
        index += 1;
    }
    if (script) |path| sources[index] = path;
    return sources;
}

/// Serves the Zig unit-test protocol used by the build runner: metadata, then one
/// `run_test` at a time, each answered with `test_started` and `test_results`.
fn serveTestProtocol(init: std.process.Init, allocator: std.mem.Allocator, mod: *const zigc.Module) !void {
    const io = init.io;
    const funcs = try allocator.alloc(target.native.ModuleFunction, mod.funcs.len);
    for (mod.funcs, 0..) |*function, index| funcs[index] = .{ .name = function.name, .func = &function.func };
    const data = try allocator.alloc(target.native.ModuleData, mod.data.len);
    for (mod.data, 0..) |object, index| data[index] = .{
        .name = object.name,
        .bytes = object.bytes,
        .kind = switch (object.kind) {
            .rodata => .rodata,
            .data => .data,
            .bss => .bss,
        },
        .size = object.size,
    };
    var jitted = try target.native.jitModuleData(allocator, funcs, data);
    defer jitted.deinit();

    var string_bytes: std.ArrayList(u8) = .empty;
    defer string_bytes.deinit(allocator);
    const names = try allocator.alloc(u32, mod.tests.len);
    defer allocator.free(names);
    const expected_panic_msgs = try allocator.alloc(u32, mod.tests.len);
    defer allocator.free(expected_panic_msgs);
    for (mod.tests, 0..) |test_entry, index| {
        names[index] = @intCast(string_bytes.items.len);
        try string_bytes.appendSlice(allocator, test_entry.name);
        try string_bytes.append(allocator, 0);
        expected_panic_msgs[index] = 0;
    }

    var in_buffer: [4096]u8 = undefined;
    var input = std.Io.File.stdin().readerStreaming(io, &in_buffer);
    var out_buffer: [4096]u8 = undefined;
    var output = std.Io.File.stdout().writerStreaming(io, &out_buffer);
    var server: std.zig.Server = .{ .in = &input.interface, .out = &output.interface };
    try server.serveStringMessage(.zig_version, builtin.zig_version_string);

    while (true) {
        const message = server.receiveMessage() catch |err| switch (err) {
            error.EndOfStream => return,
            else => return err,
        };
        const body = try server.in.take(message.bytes_len);
        switch (message.tag) {
            .query_test_metadata => {
                if (body.len != 0) return error.InvalidTestProtocolMessage;
                try server.serveTestMetadata(.{
                    .names = names,
                    .expected_panic_msgs = expected_panic_msgs,
                    .string_bytes = string_bytes.items,
                });
            },
            .run_test => {
                if (body.len != @sizeOf(u32)) return error.InvalidTestProtocolMessage;
                const index = std.mem.readInt(u32, body[0..@sizeOf(u32)], .little);
                if (index >= mod.tests.len) return error.InvalidTestProtocolMessage;
                try server.serveBodylessMessage(.test_started);
                const test_entry = mod.tests[index];
                const test_fn = jitted.entry(*const fn () callconv(.c) void, test_entry.link_name);
                const status: std.zig.Server.Message.TestResults.Status = if (test_fn) |func| blk: {
                    func();
                    break :blk .pass;
                } else .fail;
                try server.serveTestResults(.{
                    .index = index,
                    .flags = .{ .status = status, .fuzz = false, .log_err_count = 0, .leak_count = 0 },
                });
            },
            .exit => {
                if (body.len != 0) return error.InvalidTestProtocolMessage;
                return;
            },
            else => return error.InvalidTestProtocolMessage,
        }
    }
}

fn serveBuildInputs(server: *std.zig.Server, result: ServerResult) !void {
    if (result.sources.len == 0) return;
    var bytes_len = result.sources.len * 2 - 1;
    for (result.sources) |source| bytes_len += source.len;
    try server.serveMessageHeader(.{
        .tag = .file_system_inputs,
        .bytes_len = std.math.cast(u32, bytes_len) orelse return error.TooManySourceInputs,
    });
    for (result.sources, 0..) |source, index| {
        if (index != 0) try server.out.writeByte(0);
        try server.out.writeByte(@backingInt(std.zig.Server.Message.PathPrefix.cwd) + 1);
        try server.out.writeAll(source);
    }
    try server.out.flush();
}

fn writeDepfile(
    io: std.Io,
    path: []const u8,
    output: []const u8,
    sources: []const []u8,
    objects: []const []const u8,
    script: ?[]const u8,
) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(io, &buffer);
    const writer = &file_writer.interface;
    try writeMakePath(writer, output);
    try writer.writeByte(':');
    for (sources) |source| {
        try writer.writeByte(' ');
        try writeMakePath(writer, source);
    }
    for (objects) |object| {
        try writer.writeByte(' ');
        try writeMakePath(writer, object);
    }
    if (script) |source| {
        try writer.writeByte(' ');
        try writeMakePath(writer, source);
    }
    try writer.writeByte('\n');
    try writer.flush();
}

fn writeMakePath(writer: *std.Io.Writer, path: []const u8) !void {
    var index: usize = 0;
    while (index < path.len) : (index += 1) {
        const byte = path[index];
        if (byte == '\\') {
            const start = index;
            while (index < path.len and path[index] == '\\') : (index += 1) {}
            const count = index - start;
            const escape = index == path.len or switch (path[index]) {
                ' ', '\t', '#', ':' => true,
                else => false,
            };
            try writer.splatByteAll('\\', count * @as(usize, if (escape) 2 else 1));
            if (index == path.len) break;
        }
        switch (path[index]) {
            '\n', '\r' => return error.InvalidDepfilePath,
            '$' => try writer.writeAll("$$"),
            ' ', '\t', '#', ':' => {
                try writer.writeByte('\\');
                try writer.writeByte(path[index]);
            },
            else => try writer.writeByte(path[index]),
        }
    }
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
        \\usage: vulcan-zig build [zig build options] [steps]
        \\       vulcan-zig build-exe [options] <input.zig> [object.o ...]
        \\       vulcan-zig build-obj [options] <input.zig>
        \\       vulcan-zig test [options] <input.zig>
        \\       vulcan-zig build-exe [options] -Mname=path ... [object.o ...]
        \\       vulcan-zig build-obj [options] -Mname=path ...
        \\       vulcan-zig test [options] -Mname=path ...
        \\  --dep name|alias=module dependency for the next -M; first -M is root
        \\  -Mname=path        define a named module source
        \\  -target <triple>   target architecture (default: host)
        \\  -T <script>       use a linker script for executable layout
        \\  -femit-bin[=path] output path (default: input basename)
        \\  -fno-emit-bin      compile tests without producing an artifact
        \\  --test-filter <s>  keep only tests whose name contains <s>
        \\  --depfile[=]<path> write Make dependencies, including transitive imports
        \\  --entry <symbol> executable entry symbol (default: _start)
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

const CacheOutput = struct {
    path: []const u8,
    digest: std.Build.Cache.BinDigest,
};

fn cacheOutputPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    cache_dir: []const u8,
    name: []const u8,
    sources: []const []u8,
    objects: []const []const u8,
    script: ?[]const u8,
    target_triple: ?[]const u8,
    entry: []const u8,
    entry_explicit: bool,
    command: Command,
    modules: []const zigc.ModuleMapping,
    global_cache_dir: ?[]const u8,
    build_root: ?[]const u8,
    zig_lib_dir: []const u8,
    include_paths: []const []const u8,
) !CacheOutput {
    var hash: std.Build.Cache.HashHelper = .{};
    hash.addBytes(builtin.zig_version_string);
    hash.add(command);
    hash.addBytes(target_triple orelse "");
    hash.addBytes(global_cache_dir orelse "");
    hash.addBytes(build_root orelse "");
    hash.addBytes(zig_lib_dir);
    hash.addBytes(entry);
    hash.add(entry_explicit);
    hash.addBytes(name);
    hash.addBytes(script orelse "");
    for (include_paths) |path| hash.addBytes(path);
    for (modules) |module| {
        hash.addBytes(module.name);
        hash.addBytes(module.path);
        for (module.dependencies) |dependency| {
            hash.addBytes(dependency.name);
            hash.addBytes(dependency.module);
        }
    }
    for (sources) |path| try hashFile(allocator, io, &hash, path, 64 * 1024 * 1024);
    for (objects) |path| try hashFile(allocator, io, &hash, path, 64 * 1024 * 1024);
    if (script) |path| try hashFile(allocator, io, &hash, path, 16 * 1024 * 1024);
    var digest: std.Build.Cache.BinDigest = undefined;
    hash.hasher.final(&digest);
    const hex = std.Build.Cache.binToHex(digest);
    const directory = try std.fmt.allocPrint(allocator, "{s}/o/{s}", .{ cache_dir, hex });
    try std.Io.Dir.cwd().createDirPath(io, directory);
    return .{
        .path = try std.fs.path.join(allocator, &.{ directory, name }),
        .digest = digest,
    };
}

fn hashFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    hash: *std.Build.Cache.HashHelper,
    path: []const u8,
    limit: usize,
) !void {
    const contents = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(limit));
    hash.addBytes(path);
    hash.addBytes(contents);
}

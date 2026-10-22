//! Lower parsed Zig source (via `std.zig.Ast`) to Vulcan IR. Supports top-level
//! functions/data, primitive scalars, pointers, optionals, fixed-size arrays,
//! simple structs, slices, and the supported error-union operations.
//! File compilation emits reached imported top-level functions/data on demand;
//! root function bodies remain eager. General generics/comptime, tagged unions,
//! struct method bodies, and general `defer` remain unsupported (`defer @cVaEnd`
//! is the narrow defer exception).
//!
//! `vulcan-ir`'s `TypeKind.@"struct"`/`Extract`/`StructNew` opcodes are High-profile-only
//! and, as of this frontend, not lowered for a non-vector aggregate by any native backend
//! (`Extract`/`StructNew` codegen accepts SIMD vectors only). So neither a struct nor an
//! array is ever represented as a single aggregate-typed IR `Value` here. Instead, every
//! struct/array local is a byte-addressed `alloca` slot (an `array` of its element type, or
//! for a struct a byte blob sized by this frontend's own field layout), and every
//! field/element access computes a `ptr`-typed address via `arith .add` (an ordinary pointer
//! + byte-offset add, which the IR verifier allows for mismatched ptr/int operand widths).
//! This mirrors `vulcan-cc`'s own struct/array addressing (`byteOffset`/scaled-index helpers
//! in `libs/vulcan-cc/lower.zig`), which uses the same pointer-arithmetic strategy for the
//! same reason. A struct or array therefore only ever appears BEHIND a pointer in this
//! frontend: passing or returning one by value (not through `*T`) is `error.UnsupportedType`.
//!
//! Struct fields use their declared order with each field and final size rounded to the
//! field's natural alignment. That matches Zig's ordinary and `extern struct` layout for this
//! subset, so pointer-based calls can share a record with source compiled by Zig. Aggregate
//! values still remain behind pointers because the IR lacks native non-vector aggregate values.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const Ast = std.zig.Ast;

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = ir.types.Type;
const FloatKind = ir.types.FloatKind;

/// A lowered, named function: its link name and its IR. Caller owns both.
pub const NamedFunction = struct { name: []u8, func: Function };

/// The linker's storage class for a top-level Zig declaration.
pub const DataKind = enum { rodata, data, bss };

/// A named top-level data object. `bytes` is empty only for `.bss`; its `size` remains the
/// object's storage size because the loader, not the source program, zero-initializes `.bss`.
pub const Data = struct {
    name: []u8,
    bytes: []u8,
    kind: DataKind,
    size: u64,
};

/// The target facts the Zig frontend can fold without importing the target's full standard
/// library namespace. More ABI-sensitive target facts remain in the backend.
pub const TargetArch = enum { aarch64, x86_64, x86, riscv64 };

/// A direct `@import` declaration preserved for a later module-resolution stage.
pub const Import = struct { name: []u8, path: []u8 };

/// One `test` declaration compiled into a callable IR function. `name` is what a
/// caller displays; `link_name` is the function's name in `Module.funcs`.
pub const Test = struct {
    name: []u8,
    link_name: []u8,
};

/// Root definitions and reached imported top-level definitions, plus the root's direct
/// module declarations. Caller owns all slices and strings.
pub const Module = struct {
    funcs: []NamedFunction,
    data: []Data,
    imports: []Import,
    /// Tests lowered when `FileCompileOptions.tests` was set.
    tests: []Test = &.{},
    /// Normalized absolute source paths, populated only when requested by file compilation.
    source_files: [][]u8 = &.{},

    pub fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        for (self.funcs) |*nf| {
            nf.func.deinit();
            allocator.free(nf.name);
        }
        allocator.free(self.funcs);
        self.funcs = &.{};
        for (self.data) |d| {
            allocator.free(d.name);
            allocator.free(d.bytes);
        }
        allocator.free(self.data);
        self.data = &.{};
        for (self.imports) |import| {
            allocator.free(import.name);
            allocator.free(import.path);
        }
        allocator.free(self.imports);
        self.imports = &.{};
        for (self.tests) |test_entry| {
            allocator.free(test_entry.name);
            allocator.free(test_entry.link_name);
        }
        allocator.free(self.tests);
        self.tests = &.{};
        for (self.source_files) |path| allocator.free(path);
        allocator.free(self.source_files);
        self.source_files = &.{};
    }

    /// Find a function's IR by its link name, or `null` if the module has none by that name.
    pub fn find(self: *const Module, name: []const u8) ?*const Function {
        for (self.funcs) |*nf| if (std.mem.eql(u8, nf.name, name)) return &nf.func;
        return null;
    }
};

pub const Error = error{
    /// A malformed or unresolved named-module graph.
    InvalidModuleGraph,
    /// `std.zig.Ast.parse` reported one or more syntax errors.
    ParseError,
    /// An AST node whose shape is outside the extended subset (e.g. a tagged union, a
    /// generic function, `defer`, a method inside a `struct`, an unsupported statement).
    UnsupportedNode,
    /// A type expression outside int/bool/float/single-item-pointer/array/struct-name, or a
    /// struct/array type used somewhere only a scalar or pointer is supported (by-value
    /// struct/array parameter or return, or as an operand of arithmetic).
    UnsupportedType,
    /// A name referenced that is not a local, a parameter, or a declared function.
    UndeclaredIdentifier,
    /// A call's argument count does not match the callee's declared parameter count.
    CallArityMismatch,
    /// An operand's type does not match what an operator, assignment, return, or call
    /// argument requires, and this subset's narrow (widening-only) conversion rule does not
    /// bridge it.
    TypeMismatch,
    /// Two top-level declarations (`fn` or `struct`) share a name.
    DuplicateDecl,
    /// A `.field` access or struct-literal field name that the struct type does not declare.
    UnknownField,
    /// An integer literal does not fit the target integer type.
    IntegerOverflow,
    /// The build host is outside the frontend's supported native backends.
    UnsupportedTarget,
} || std.mem.Allocator.Error;

/// Compile for the host architecture. Call `compileForTarget` when source-level `@import("builtin")`
/// needs the target architecture rather than the host's.
pub fn compile(allocator: std.mem.Allocator, source: []const u8) Error!Module {
    const target = switch (builtin.cpu.arch) {
        .aarch64 => TargetArch.aarch64,
        .x86_64 => TargetArch.x86_64,
        .x86 => TargetArch.x86,
        .riscv64 => TargetArch.riscv64,
        else => return Error.UnsupportedTarget,
    };
    return compileForTarget(allocator, source, target);
}

/// Compile Zig source for `target`, including the small target-dependent builtin subset.
pub fn compileForTarget(allocator: std.mem.Allocator, source: []const u8, target: TargetArch) Error!Module {
    const source_z = try allocator.dupeSentinel(u8, source, 0);
    defer allocator.free(source_z);
    var tree = try Ast.parse(allocator, source_z, .{});
    defer tree.deinit(allocator);
    if (tree.errors.len != 0) return Error.ParseError;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    return lowerRoot(allocator, arena_state.allocator(), &tree, target, ".", null, false, &.{});
}

pub const ModuleDependency = struct {
    /// Local import alias visible in this module.
    name: []const u8,
    /// Identifier of the dependency module.
    module: []const u8,
};

pub const ModuleMapping = struct {
    name: []const u8,
    path: []const u8,
    dependencies: []const ModuleDependency = &.{},
};

/// Where lowering stopped, for callers that want to report a source location.
/// `path_buffer` is caller-owned; a diagnostic never allocates.
pub const Diagnostic = struct {
    path_buffer: []u8,
    path_len: usize = 0,
    /// 1-based, or 0 when no source node was recorded.
    line: u32 = 0,
    column: u32 = 0,

    pub fn path(self: *const Diagnostic) []const u8 {
        return self.path_buffer[0..self.path_len];
    }
};

pub const FileCompileOptions = struct {
    std_dir: ?[]const u8 = null,
    modules: []const ModuleMapping = &.{},
    /// Selects the entry module; its mapped path must resolve to `compileFileOptions`'s `path`.
    root_module: ?[]const u8 = null,
    /// Retain every source read during lowering for build-system dependency tracking.
    collect_source_files: bool = false,
    /// On failure, receives the deepest source location lowering visited.
    diagnostic: ?*Diagnostic = null,
    /// Lower `test` declarations in the root module's files into `Module.tests`.
    tests: bool = false,
    /// Keep only tests whose name contains one of these strings; empty keeps all.
    test_filters: []const []const u8 = &.{},
};

/// Compile a Zig source file, resolving relative imports beside the importing file,
/// `@import("std")` through `std_dir`, and package imports through `modules`.
pub fn compileFileOptions(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    target: TargetArch,
    options: FileCompileOptions,
) Error!Module {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var modules = Modules{
        .io = io,
        .arena = arena,
        .std_dir = options.std_dir,
        .named = options.modules,
        .root_module = options.root_module,
        .diagnostic = options.diagnostic,
    };
    try modules.validate();
    var cwd_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const cwd_len = std.Io.Dir.cwd().realPath(io, &cwd_buffer) catch return Error.UnsupportedNode;
    modules.cwd = cwd_buffer[0..cwd_len];
    const root_module_index = if (options.root_module) |name| modules.moduleIndex(name) orelse return Error.InvalidModuleGraph else null;
    const source_path = std.fs.path.resolve(arena, &.{ modules.cwd, path }) catch return Error.UnsupportedNode;
    if (root_module_index) |index| {
        const mapped_path = std.fs.path.resolve(arena, &.{ modules.cwd, modules.named[index].path }) catch return Error.InvalidModuleGraph;
        if (!std.mem.eql(u8, source_path, mapped_path)) return Error.InvalidModuleGraph;
    }
    const source = std.Io.Dir.cwd().readFileAllocOptions(io, path, arena, .limited(16 * 1024 * 1024), .of(u8), 0) catch return Error.UnsupportedNode;
    var tree = try Ast.parse(arena, source, .{});
    if (tree.errors.len != 0) return Error.ParseError;
    const source_dir = std.fs.path.dirname(source_path) orelse ".";
    modules.root = .{ .tree = &tree, .dir = source_dir, .module_index = root_module_index, .path = source_path };
    if (options.tests) modules.loadTestImports() catch |err| {
        if (options.diagnostic) |diagnostic| modules.fillDiagnostic(diagnostic);
        return err;
    };
    var result = lowerRoot(allocator, arena, &tree, target, source_dir, &modules, options.tests, options.test_filters) catch |err| {
        if (options.diagnostic) |diagnostic| modules.fillDiagnostic(diagnostic);
        return err;
    };
    errdefer result.deinit(allocator);
    if (options.collect_source_files) {
        var paths: std.ArrayList([]u8) = .empty;
        errdefer {
            for (paths.items) |loaded_path| allocator.free(loaded_path);
            paths.deinit(allocator);
        }
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        for (0..modules.files.items.len + 1) |index| {
            const loaded_path = if (index == 0) source_path else modules.files.items[index - 1].path;
            const entry = try seen.getOrPut(arena, loaded_path);
            if (entry.found_existing) continue;
            const owned_path = try allocator.dupe(u8, loaded_path);
            errdefer allocator.free(owned_path);
            try paths.append(allocator, owned_path);
        }
        result.source_files = try paths.toOwnedSlice(allocator);
    }
    return result;
}

/// Compatibility entry point for callers that only need relative imports and `std`.
pub fn compileFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, target: TargetArch, std_dir: ?[]const u8) Error!Module {
    return compileFileOptions(allocator, io, path, target, .{ .std_dir = std_dir });
}

test "lower and JIT-run a constant return" {
    const target = @import("vulcan-target");
    const allocator = std.testing.allocator;
    var mod = try compile(allocator, "fn f() i64 { return -7; }");
    defer mod.deinit(allocator);
    const f_ir = mod.find("f") orelse return error.NoFunc;
    var jitted = try target.native.jitModule(allocator, &.{.{ .name = "f", .func = f_ir }});
    defer jitted.deinit();
    const f = jitted.entry(*const fn () callconv(.c) i64, "f") orelse return error.NoEntry;
    try std.testing.expectEqual(@as(i64, -7), f());
}

test "optional integer calls preserve null, payload, and adjacent stack slots" {
    const target = @import("vulcan-target");
    const allocator = std.testing.allocator;
    var mod = try compile(allocator,
        \\fn maybe(n: usize) ?usize {
        \\    if (n == 0) return null;
        \\    return n;
        \\}
        \\fn next(value: ?usize) usize {
        \\    if (value) |n| return n + 1;
        \\    return 5;
        \\}
        \\fn check(n: usize) usize {
        \\    const value: ?usize = maybe(n);
        \\    return next(value);
        \\}
        \\fn small(n: u16) ?u16 {
        \\    if (n == 0) return null;
        \\    return n;
        \\}
        \\fn shortCheck(n: u16) u16 {
        \\    return small(n) orelse 9;
        \\}
    );
    defer mod.deinit(allocator);
    const funcs = try allocator.alloc(target.native.ModuleFunction, mod.funcs.len);
    defer allocator.free(funcs);
    for (mod.funcs, 0..) |*nf, i| funcs[i] = .{ .name = nf.name, .func = &nf.func };
    var jitted = try target.native.jitModule(allocator, funcs);
    defer jitted.deinit();
    const check = jitted.entry(*const fn (usize) callconv(.c) usize, "check") orelse return error.NoEntry;
    const short_check = jitted.entry(*const fn (u16) callconv(.c) u16, "shortCheck") orelse return error.NoEntry;
    try std.testing.expectEqual(@as(usize, 5), check(0));
    try std.testing.expectEqual(@as(usize, 42), check(41));
    try std.testing.expectEqual(@as(u16, 9), short_check(0));
    try std.testing.expectEqual(@as(u16, 7), short_check(7));
}

// ---------------------------------------------------------------------------------------
// Frontend-local type representation. Never `ir.types.Type` directly for a pointer/array/
// struct: the IR's own `ptr` carries no pointee, and (per this file's header doc) a struct
// never becomes an aggregate IR type at all. `ZType.irType` is the only place a `ZType`
// turns into a real (per-`Function`-interned) `ir.types.Type`, used solely for a scalar
// SSA value's type or an `alloca`'s storage-sizing element type.
// ---------------------------------------------------------------------------------------

const ZInt = struct { signed: bool, bits: u16 };
const ZErrorSet = struct { names: []const []const u8 };
const ZErrorUnion = struct { payload: *const ZType, errors: *const ZErrorSet };

const ZType = union(enum) {
    error_void,
    error_set: *const ZErrorSet,
    /// Non-void error unions store payload bytes first, then a u16 tag.
    error_union: ZErrorUnion,
    boolean,
    opaque_type,
    int: ZInt,
    float: FloatKind,
    /// The pointee type. Arena-owned (see `Ctx.arena`).
    ptr: *const ZType,
    /// Nullable pointers have a spare zero bit pattern and remain scalar IR values.
    optional_ptr: *const ZType,
    /// A non-pointer optional stores its payload followed by a presence byte, with
    /// each field and the whole value rounded to the payload's alignment. This is
    /// a frontend-internal representation: Zig does not guarantee an extern ABI
    /// for non-pointer optionals. Only calls between functions built here share it.
    optional_value: *const ZType,
    /// A Zig slice is stored as a pointer/length pair; function parameters flatten to two ABI values.
    slice: *const ZType,
    array: ZArray,
    strct: *const StructDef,
    /// Nominal: `ZType.enumeration` carries its identity, never just the backing width.
    enumeration: *const EnumDef,
    /// A `@cVaStart()`-initialized `va_list` object. Backend-opaque: this frontend never reads
    /// or writes its bytes itself (`ir.function.Opcode.va_start`/`va_arg`/`va_end` do, at
    /// fixed per-target offsets (AAPCS64 needs 32 bytes), only needs a slot it can address,
    /// exactly like any other local. See `lowerStmt`'s `@cVaStart()` var-decl special case.
    va_list,
    const ZArray = struct { len: u64, elem: *const ZType };

    fn isAggregate(self: ZType) bool {
        return self == .array or self == .strct or self == .slice or self == .optional_value or self == .error_union or self == .opaque_type;
    }

    fn optionalPayloadIsAggregate(self: ZType) bool {
        return self == .optional_value and self.optional_value.*.isAggregate();
    }

    fn errorTagOffset(self: ZType) u64 {
        return switch (self) {
            .error_union => |e| alignForward(e.payload.byteSize(), 2),
            else => unreachable,
        };
    }
    fn alignment(self: ZType) u64 {
        return switch (self) {
            .error_void => 2,
            .boolean, .opaque_type => 1,
            .int, .float => self.byteSize(),
            .enumeration => |e| enumByteSize(e.backing),
            .ptr, .optional_ptr, .slice, .va_list => 8,
            .error_set => 2,
            .error_union => |e| @max(e.payload.alignment(), 2),
            .optional_value => |p| p.alignment(),
            .array => |a| a.elem.alignment(),
            .strct => |s| s.alignment,
        };
    }

    /// This type's storage size in bytes, used for `alloca` sizing (via `irType`) and field
    /// offsets / array strides. Struct sizes include their trailing ABI alignment padding.
    fn byteSize(self: ZType) u64 {
        return switch (self) {
            .error_void, .error_set => 2,
            .boolean => 1,
            .int => |i| (@as(u64, i.bits) + 7) / 8,
            .enumeration => |e| enumByteSize(e.backing),
            .opaque_type => 0,
            .float => |f| switch (f) {
                .f16 => 2,
                .f32 => 4,
                .f64 => 8,
                .f128 => 16,
            },
            .ptr, .optional_ptr => 8,
            .slice => 16,
            .error_union => |e| alignForward(alignForward(e.payload.byteSize(), 2) + 2, @max(e.payload.alignment(), 2)),
            .optional_value => |p| alignForward(p.byteSize() + 1, p.alignment()),
            .array => |a| a.len * a.elem.byteSize(),
            .strct => |s| s.byte_size,
            // 32 bytes: the largest of the three per-target `va_list` layouts this frontend's
            // backends read/write (AAPCS64; x86_64 needs 24, riscv64 needs 8).
            .va_list => 32,
        };
    }

    /// The `ir.types.Type` handle for this type, interned in `func`'s own type table. Only
    /// ever used for a scalar value's type, or as an `alloca`'s `elem` (in which case an
    /// aggregate (`array`/`strct`) yields a byte-sized shape: an array yields a REAL `.array`
    /// (the aarch64 backend already sizes that correctly: `len * typeSize(elem)`), and a
    /// struct yields a `.array` of `u8` sized to its own byte layout (structs have no IR
    /// aggregate type at all here, see this file's header doc).
    fn irType(self: ZType, func: *Function) std.mem.Allocator.Error!Type {
        return switch (self) {
            .error_void => func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } }),
            .error_set => func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } }),
            .boolean => func.types.intern(.bool),
            .opaque_type => unreachable, // Zig never materializes an anyopaque value.
            .int => |i| func.types.intern(.{ .int = .{ .signedness = if (i.signed) .signed else .unsigned, .bits = i.bits } }),
            .enumeration => |e| func.types.intern(.{ .int = .{ .signedness = if (e.backing.signed) .signed else .unsigned, .bits = e.backing.bits } }),
            .float => |f| func.types.intern(.{ .float = f }),
            .ptr, .optional_ptr => func.types.ptrGlobal(),
            .slice => blk: {
                const word = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
                break :blk func.types.intern(.{ .array = .{ .len = 2, .elem = word } });
            },
            .optional_value => |p| blk: {
                const size = alignForward(p.byteSize() + 1, p.alignment());
                if (p.isAggregate()) {
                    const stride = p.alignment();
                    const word = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = @intCast(stride * 8) } });
                    break :blk func.types.intern(.{ .array = .{ .len = size / stride, .elem = word } });
                }
                const bits: u16 = if (size == 2) 16 else if (size == 4) 32 else 64;
                const word = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = bits } });
                break :blk func.types.intern(.{ .array = .{ .len = if (size == 16) 2 else 1, .elem = word } });
            },
            .array => |a| blk: {
                const e = try a.elem.irType(func);
                break :blk func.types.intern(.{ .array = .{ .len = a.len, .elem = e } });
            },
            .strct => |s| blk: {
                const u8t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
                break :blk func.types.intern(.{ .array = .{ .len = s.byte_size, .elem = u8t } });
            },
            .error_union => blk: {
                const u8t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
                break :blk func.types.intern(.{ .array = .{ .len = self.byteSize(), .elem = u8t } });
            },
            .va_list => blk: {
                const i64t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
                break :blk func.types.intern(.{ .array = .{ .len = 4, .elem = i64t } });
            },
        };
    }

    fn eql(a: ZType, b: ZType) bool {
        if (@as(std.meta.Tag(ZType), a) != @as(std.meta.Tag(ZType), b)) return false;
        return switch (a) {
            .error_void, .boolean, .opaque_type => true,
            .int => |ai| ai.signed == b.int.signed and ai.bits == b.int.bits,
            .float => |af| af == b.float,
            .ptr => |ap| ap.eql(b.ptr.*),
            .optional_ptr => |ap| ap.eql(b.optional_ptr.*),
            .optional_value => |ap| ap.eql(b.optional_value.*),
            .slice => |ap| ap.eql(b.slice.*),
            .array => |aa| aa.len == b.array.len and aa.elem.eql(b.array.elem.*),
            .strct => |as_| as_ == b.strct,
            .enumeration => |ae| ae == b.enumeration,
            .error_set => |ae| errorSetsEqual(ae.*, b.error_set.*),
            .error_union => |ae| ae.payload.eql(b.error_union.payload.*) and errorSetsEqual(ae.errors.*, b.error_union.errors.*),
            .va_list => true,
        };
    }
};

const StructField = struct { name: []const u8, ty: ZType, offset: u64 };
const StructDef = struct { name: []const u8, fields: []const StructField, byte_size: u64, alignment: u64 };
/// One named enumeration tag. Tags are stored as `EnumDef.backing`; an enum value is a
/// nominal type, so two enums with the same backing width do not interchange.
const EnumField = struct { name: []const u8, value: u64 };
const EnumDef = struct {
    name: []const u8,
    /// The integral backing type. Only integer widths are modeled.
    backing: ZInt,
    fields: []const EnumField,
    /// A trailing `_` tag makes the enum non-exhaustive: a value outside `fields` is
    /// still a valid enum, only its name is not known at compile time.
    non_exhaustive: bool,

    fn field(self: *const EnumDef, name: []const u8) ?EnumField {
        if (std.mem.eql(u8, name, "_")) return null;
        for (self.fields) |f| if (std.mem.eql(u8, f.name, name)) return f;
        return null;
    }

    /// Does any named tag hold `value`?
    fn hasValue(self: *const EnumDef, value: u64) bool {
        for (self.fields) |f| if (f.value == value) return true;
        return false;
    }
};
/// A top-level `const` whose value was evaluated, with the type it evaluated at.
const ConstEntry = struct { zty: ZType, value: ConstValue };
const GlobalDef = struct { zty: ZType, symbol: []const u8 };
fn enumByteSize(backing: ZInt) u64 {
    return (@as(u64, backing.bits) + 7) / 8;
}
const FnDeclInfo = struct { decl_node: Ast.Node.Index, proto_node: Ast.Node.Index, body_node: ?Ast.Node.Index };
const GlobalDeclInfo = struct { decl_node: Ast.Node.Index };
/// A named container (struct or enum): its declaring node and, when it is a `fn`
/// declaration's member of a container, `parent` is that container node.
const TypeDeclInfo = struct { decl_node: Ast.Node.Index, parent: ?Ast.Node.Index = null };
const DeclKey = struct { tree: *const Ast, node: Ast.Node.Index };
const ConstValue = union(enum) {
    boolean: bool,
    /// The integer's bit pattern; its width and signedness come from the expression that
    /// produced it (`EvalValue.zty`), because Zig integers are typed rather than `i64`.
    int: i64,
};
/// A comptime value with the type it was evaluated at.
const EvalValue = struct { zty: ZType, value: ConstValue };
/// Evaluation answer plus whether the expression depended on a runtime value (an enum
/// literal without an expected type, or a non-comptime parameter). A dependent result may
/// be used for its type, but never folded into data or a constant.
const Eval = struct { value: EvalValue, dependent: bool = false };
const ConstBinding = struct { name: []const u8, value: i64 };

fn errorSetsEqual(a: ZErrorSet, b: ZErrorSet) bool {
    if (a.names.len != b.names.len) return false;
    for (a.names, b.names) |left, right| if (!std.mem.eql(u8, left, right)) return false;
    return true;
}
fn alignForward(value: u64, alignment: u64) u64 {
    std.debug.assert(alignment != 0 and std.math.isPowerOfTwo(alignment));
    return (value + alignment - 1) & ~(alignment - 1);
}

fn findField(sd: *const StructDef, name: []const u8) ?StructField {
    for (sd.fields) |f| if (std.mem.eql(u8, f.name, name)) return f;
    return null;
}

/// A parsed file kept alive for the whole `compileFile` compilation (its `Ast` is
/// `Modules.arena`-owned), together with the directory a relative `@import` reached FROM it
/// resolves against.
const LoadedFile = struct { tree: *const Ast, dir: []const u8, module_index: ?usize, path: []const u8, is_std: bool = false };
const LoadedEntry = struct { path: []const u8, file: LoadedFile };

const Modules = struct {
    io: std.Io,
    arena: std.mem.Allocator,
    cwd: []const u8 = ".",
    std_dir: ?[]const u8,
    named: []const ModuleMapping = &.{},
    root_module: ?[]const u8 = null,
    root: ?LoadedFile = null,
    files: std.ArrayList(LoadedEntry) = .empty,
    diagnostic: ?*Diagnostic = null,
    last_tree: ?*const Ast = null,
    last_node: ?Ast.Node.Index = null,

    /// Record the node most recently being lowered, so a failure after it can name it.
    fn mark(self: *Modules, tree: *const Ast, node: Ast.Node.Index) void {
        if (self.diagnostic == null) return;
        self.last_tree = tree;
        self.last_node = node;
    }

    fn fillDiagnostic(self: *const Modules, diagnostic: *Diagnostic) void {
        const tree = self.last_tree orelse return;
        const node = self.last_node orelse return;
        const loaded = self.owner(tree) orelse return;
        const path = loaded.path[0..@min(loaded.path.len, diagnostic.path_buffer.len)];
        @memcpy(diagnostic.path_buffer[0..path.len], path);
        diagnostic.path_len = path.len;
        const location = tree.tokenLocation(0, tree.nodeMainToken(node));
        diagnostic.line = @intCast(location.line + 1);
        diagnostic.column = @intCast(location.column + 1);
    }

    fn moduleIndex(self: *const Modules, name: []const u8) ?usize {
        for (self.named, 0..) |module, index| if (std.mem.eql(u8, module.name, name)) return index;
        return null;
    }

    fn validate(self: *const Modules) Error!void {
        for (self.named, 0..) |module, i| {
            if (module.name.len == 0 or module.path.len == 0) return Error.InvalidModuleGraph;
            for (self.named[0..i]) |prior| if (std.mem.eql(u8, module.name, prior.name)) return Error.InvalidModuleGraph;
            for (module.dependencies, 0..) |dep, j| {
                if (dep.name.len == 0 or dep.module.len == 0 or self.moduleIndex(dep.module) == null) return Error.InvalidModuleGraph;
                for (module.dependencies[0..j]) |prior| if (std.mem.eql(u8, dep.name, prior.name)) return Error.InvalidModuleGraph;
            }
        }
        if (self.root_module) |root| if (self.moduleIndex(root) == null) return Error.InvalidModuleGraph;
    }

    fn owner(self: *const Modules, tree: *const Ast) ?LoadedFile {
        if (self.root) |root| if (root.tree == tree) return root;
        for (self.files.items) |entry| if (entry.file.tree == tree) return entry.file;
        return null;
    }

    fn load(self: *Modules, from_tree: *const Ast, import_path: []const u8) Error!LoadedFile {
        if (std.mem.eql(u8, import_path, "builtin")) return Error.UnsupportedType;
        const from = self.owner(from_tree) orelse return Error.UnsupportedType;
        if (std.mem.eql(u8, import_path, "root")) return self.root orelse return Error.UnsupportedType;
        var target_module_index = from.module_index;
        var from_std = false;
        const abs = if (std.mem.eql(u8, import_path, "std")) blk: {
            target_module_index = null;
            from_std = true;
            break :blk try std.fs.path.resolve(self.arena, &.{ self.cwd, self.std_dir orelse return Error.UnsupportedType, "std.zig" });
        } else blk: {
            if (from.module_index) |module_index| {
                const module = self.named[module_index];
                for (module.dependencies) |dep| if (std.mem.eql(u8, dep.name, import_path)) {
                    const target_index = self.moduleIndex(dep.module) orelse return Error.InvalidModuleGraph;
                    target_module_index = target_index;
                    break :blk try std.fs.path.resolve(self.arena, &.{ self.cwd, self.named[target_index].path });
                };
            }
            if (!std.mem.endsWith(u8, import_path, ".zig")) return Error.UnsupportedType;
            break :blk try std.fs.path.resolve(self.arena, &.{ from.dir, import_path });
        };
        return self.loadAbs(abs, target_module_index, from_std);
    }

    fn loadAbs(self: *Modules, abs: []const u8, module_index: ?usize, is_std: bool) Error!LoadedFile {
        if (self.root) |root| if (root.module_index == module_index and std.mem.eql(u8, root.path, abs)) return root;
        for (self.files.items) |entry| if (entry.file.module_index == module_index and std.mem.eql(u8, entry.path, abs)) return entry.file;
        const source = std.Io.Dir.cwd().readFileAllocOptions(self.io, abs, self.arena, .limited(16 * 1024 * 1024), .of(u8), 0) catch return Error.UnsupportedType;
        const tree = try self.arena.create(Ast);
        tree.* = Ast.parse(self.arena, source, .{}) catch return Error.ParseError;
        if (tree.errors.len != 0) return Error.ParseError;
        const loaded = LoadedFile{ .tree = tree, .dir = std.fs.path.dirname(abs) orelse ".", .module_index = module_index, .path = abs, .is_std = is_std };
        try self.files.append(self.arena, .{ .path = abs, .file = loaded });
        return loaded;
    }

    /// In test mode, every file of the root module is part of the test binary, not just
    /// the files some function happened to reach. Walk the root module's relative
    /// imports eagerly so their `test` declarations are discovered. Relative imports
    /// inherit the importing file's module identity; named dependencies and `std` are
    /// different modules and are not walked.
    fn loadTestImports(self: *Modules) Error!void {
        const root = self.root orelse return;
        var index: usize = 0;
        while (true) {
            const file = if (index == 0)
                root
            else if (index - 1 < self.files.items.len)
                self.files.items[index - 1].file
            else
                break;
            index += 1;
            if (file.is_std) continue;
            if (file.module_index != root.module_index) continue;
            for (file.tree.rootDecls()) |decl| {
                const vd = file.tree.fullVarDecl(decl) orelse continue;
                if (file.tree.tokenTag(vd.ast.mut_token) != .keyword_const) continue;
                const init_node = vd.ast.init_node.unwrap() orelse continue;
                if (!isImportDecl(file.tree, init_node)) continue;
                var buf: [2]Ast.Node.Index = undefined;
                const args = builtinArgs(file.tree, init_node, &buf);
                const import_path = std.zig.string_literal.parseAlloc(self.arena, file.tree.tokenSlice(file.tree.nodeMainToken(args[0]))) catch return Error.UnsupportedNode;
                if (std.mem.eql(u8, import_path, "std") or
                    std.mem.eql(u8, import_path, "builtin") or
                    std.mem.eql(u8, import_path, "root")) continue;
                // A name a module dependency owns crosses a module boundary.
                var is_dependency = false;
                if (file.module_index) |module_index| {
                    for (self.named[module_index].dependencies) |dependency| {
                        if (std.mem.eql(u8, dependency.name, import_path)) {
                            is_dependency = true;
                            break;
                        }
                    }
                }
                if (is_dependency) continue;
                _ = try self.load(file.tree, import_path);
            }
        }
    }
};

/// Memo key for a struct reached through a qualified name (`uefi.Guid`): its defining
/// `container_decl` node together with the file it lives in (the same node index can appear
/// in many different trees).
const ForeignKey = struct { tree: *const Ast, node: Ast.Node.Index };
/// One file's source-level namespace. All contexts in a compilation share declaration
/// worklists, link names, literal numbering, and canonical struct identities in `Compiler`.
const Ctx = struct {
    tree: *const Ast,
    /// `tree`'s own directory, used to resolve a relative `@import(...)` reached from it.
    /// `"."` (inert) whenever `modules` is `null`.
    dir: []const u8 = ".",
    arena: std.mem.Allocator,
    target: TargetArch,
    /// Non-`null` only under `compileFile`: lets a qualified type name (`uefi.Guid`) or a
    /// struct-literal's type expression chase a real `@import` to another file on disk. `null`
    /// under `compile`/`compileForTarget` (an in-memory source string, no filesystem access),
    /// where any such chase fails closed with `Error.UnsupportedType` - unchanged from before
    /// this file supported multiple files at all.
    modules: ?*Modules = null,
    nodes: std.StringHashMapUnmanaged(Ast.Node.Index) = .empty,
    defs: std.StringHashMapUnmanaged(*const StructDef) = .empty,
    resolving: std.StringHashMapUnmanaged(void) = .empty,
    globals: std.StringHashMapUnmanaged(GlobalDef) = .empty,
    constants: std.StringHashMapUnmanaged(ConstEntry) = .empty,
    deferred_constants: std.StringHashMapUnmanaged(Ast.Node.Index) = .empty,
    compiler: *Compiler,
    is_root: bool = false,
    function_decls: std.StringHashMapUnmanaged(FnDeclInfo) = .empty,
    /// Every enum or struct container named by a top-level declaration, including those
    /// reached through a qualified name in another file.
    type_decls: std.StringHashMapUnmanaged(TypeDeclInfo) = .empty,
    global_decls: std.StringHashMapUnmanaged(GlobalDeclInfo) = .empty,
    literal_data: std.ArrayList(Data) = .empty,
    inferred_error_names: std.ArrayList([]const u8) = .empty,
    /// Local `const Name = enum/struct {...}` type declarations of the function being
    /// lowered, by name. Cleared per function.
    local_types: std.StringHashMapUnmanaged(ZType) = .empty,

    fn addInferredError(self: *Ctx, name: []const u8) Error!void {
        for (self.inferred_error_names.items) |existing| {
            if (std.mem.eql(u8, existing, name)) return;
        }
        try self.inferred_error_names.append(self.arena, name);
    }

    fn defaultErrorSet(self: *Ctx) Error!*const ZErrorSet {
        const result = try self.arena.create(ZErrorSet);
        result.* = .{ .names = self.inferred_error_names.items };
        return result;
    }

    fn resolve(self: *Ctx, name: []const u8) Error!*const StructDef {
        if (self.defs.get(name)) |d| return d;
        if (self.resolving.contains(name)) return Error.UnsupportedType;
        const node = self.nodes.get(name) orelse return Error.UnsupportedType;
        const key = ForeignKey{ .tree = self.tree, .node = node };
        if (self.compiler.foreign_defs.get(key)) |def| return def;
        try self.resolving.put(self.arena, name, {});

        var buf: [2]Ast.Node.Index = undefined;
        const cd = self.tree.fullContainerDecl(&buf, node).?;
        var fields: std.ArrayList(StructField) = .empty;
        var offset: u64 = 0;
        var alignment: u64 = 1;
        for (cd.ast.members) |m| {
            const cf = self.tree.fullContainerField(m) orelse return Error.UnsupportedNode;
            const te = cf.ast.type_expr.unwrap() orelse return Error.UnsupportedType;
            const fty = try resolveType(self, te);
            const fname = self.tree.tokenSlice(cf.ast.main_token);
            const field_alignment = fty.alignment();
            offset = alignForward(offset, field_alignment);
            try fields.append(self.arena, .{ .name = fname, .ty = fty, .offset = offset });
            offset += fty.byteSize();
            alignment = @max(alignment, field_alignment);
        }
        const def = try self.arena.create(StructDef);
        def.* = .{
            .name = name,
            .fields = try fields.toOwnedSlice(self.arena),
            .byte_size = alignForward(offset, alignment),
            .alignment = alignment,
        };
        try self.defs.put(self.arena, name, def);
        try self.compiler.foreign_defs.put(self.arena, key, def);
        _ = self.resolving.remove(name);
        return def;
    }
};

const FunctionEntry = struct {
    ctx: *Ctx,
    info: FnDeclInfo,
    symbol: []const u8,
    sig: FnSig,
    is_local: bool,
};
const Compiler = struct {
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    target: TargetArch,
    modules: ?*Modules,
    root_ctx: *Ctx,
    contexts: std.ArrayList(*Ctx) = .empty,
    context_by_tree: std.AutoHashMapUnmanaged(*const Ast, *Ctx) = .empty,
    foreign_defs: std.AutoHashMapUnmanaged(ForeignKey, *const StructDef) = .empty,
    foreign_resolving: std.AutoHashMapUnmanaged(ForeignKey, void) = .empty,
    functions: std.AutoHashMapUnmanaged(DeclKey, FunctionEntry) = .empty,
    /// Canonical enum definitions, keyed by the defining container in its own tree, so
    /// every module instance and imported reference agrees on tags and identity.
    enum_defs: std.AutoHashMapUnmanaged(ForeignKey, *const EnumDef) = .empty,
    enum_resolving: std.AutoHashMapUnmanaged(ForeignKey, void) = .empty,
    function_queue: std.ArrayList(DeclKey) = .empty,
    data: std.ArrayList(Data) = .empty,
    next_link_id: u64 = 0,
    next_literal_id: u64 = 0,

    fn linkName(self: *Compiler, source_name: []const u8, external: bool) Error![]const u8 {
        if (external) return source_name;
        const name = try std.fmt.allocPrint(self.arena, ".vulcan.{d}.{s}", .{ self.next_link_id, source_name });
        self.next_link_id += 1;
        return name;
    }

    fn contextFor(self: *Compiler, tree: *const Ast) Error!*Ctx {
        if (tree == self.root_ctx.tree) return self.root_ctx;
        if (self.context_by_tree.get(tree)) |ctx| return ctx;
        const owner = (self.modules orelse return Error.UnsupportedType).owner(tree) orelse return Error.UnsupportedType;
        const ctx = try self.arena.create(Ctx);
        ctx.* = .{
            .tree = tree,
            .dir = owner.dir,
            .arena = self.arena,
            .target = self.target,
            .modules = self.modules,
            .compiler = self,
        };
        for (0..tree.nodes.len) |node_idx| {
            if (tree.nodes.items(.tag)[node_idx] == .error_value) {
                const node: Ast.Node.Index = @fromBackingInt(@intCast(node_idx));
                try ctx.addInferredError(tree.tokenSlice(tree.nodeMainToken(node) + 2));
            }
        }
        for (tree.rootDecls()) |decl| {
            if (self.modules) |modules| modules.mark(tree, decl);
            const is_definition = tree.nodeTag(decl) == .fn_decl;
            const proto_node = if (is_definition) tree.nodeData(decl).node_and_node[0] else decl;
            var proto_buf: [1]Ast.Node.Index = undefined;
            if (tree.fullFnProto(&proto_buf, proto_node)) |proto| {
                const token = proto.name_token orelse continue;
                try ctx.function_decls.put(self.arena, tree.tokenSlice(token), .{
                    .decl_node = decl,
                    .proto_node = proto_node,
                    .body_node = if (is_definition) tree.nodeData(decl).node_and_node[1] else null,
                });
            } else if (tree.fullVarDecl(decl)) |vd| {
                const name = tree.tokenSlice(vd.ast.mut_token + 1);
                if (vd.ast.init_node.unwrap()) |init_node| {
                    var container_buf: [2]Ast.Node.Index = undefined;
                    if (tree.fullContainerDecl(&container_buf, init_node)) |container| {
                        switch (tree.tokenTag(container.ast.main_token)) {
                            .keyword_struct => try ctx.nodes.put(self.arena, name, init_node),
                            .keyword_enum => try ctx.type_decls.put(self.arena, name, .{ .decl_node = init_node }),
                            else => {},
                        }
                        continue;
                    }
                    if (tree.tokenTag(vd.ast.mut_token) == .keyword_const and vd.ast.type_node.unwrap() == null)
                        try ctx.deferred_constants.put(self.arena, name, init_node);
                }
                try ctx.global_decls.put(self.arena, name, .{ .decl_node = decl });
            }
        }
        try self.context_by_tree.put(self.arena, tree, ctx);
        try self.contexts.append(self.arena, ctx);
        return ctx;
    }
};

/// A resolved qualified-name chain: `tree`/`dir` is the file the chain landed in, and `node`
/// is either a specific declaration's init expression (an import alias or a real value/type
/// resolved further, e.g. a `container_decl`), or `null` for "the whole file's top-level
/// scope" (the chain ended at an `@import(...)` with nothing dereferenced yet).
const Resolved = struct { tree: *const Ast, dir: []const u8, node: ?Ast.Node.Index };

/// The first top-level (or, given `container`, struct-member) `const`/`var` declaration in
/// `tree` named `name`, or `null`. Order-independent (Zig top-level declarations may appear
/// in any order), so this also serves a struct declared later in the same file.
fn findRootConstInit(tree: *const Ast, name: []const u8) ?Ast.Node.Index {
    for (tree.rootDecls()) |decl| {
        const vd = tree.fullVarDecl(decl) orelse continue;
        if (std.mem.eql(u8, tree.tokenSlice(vd.ast.mut_token + 1), name)) return vd.ast.init_node.unwrap();
    }
    return null;
}

const RootGlobalType = struct { type_node: ?Ast.Node.Index, init_node: ?Ast.Node.Index, is_const: bool };

fn findRootGlobalType(tree: *const Ast, name: []const u8) ?RootGlobalType {
    for (tree.rootDecls()) |decl| {
        const vd = tree.fullVarDecl(decl) orelse continue;
        if (std.mem.eql(u8, tree.tokenSlice(vd.ast.mut_token + 1), name))
            return .{
                .type_node = vd.ast.type_node.unwrap(),
                .init_node = vd.ast.init_node.unwrap(),
                .is_const = tree.tokenTag(vd.ast.mut_token) == .keyword_const,
            };
    }
    return null;
}
fn findMemberConstInit(tree: *const Ast, container: Ast.Node.Index, name: []const u8) ?Ast.Node.Index {
    var buf: [2]Ast.Node.Index = undefined;
    const cd = tree.fullContainerDecl(&buf, container) orelse return null;
    for (cd.ast.members) |member| {
        const vd = tree.fullVarDecl(member) orelse continue;
        if (std.mem.eql(u8, tree.tokenSlice(vd.ast.mut_token + 1), name)) return vd.ast.init_node.unwrap();
    }
    return null;
}

/// Follow ONE declaration's initializer past every `@import(...)`/plain-alias hop
/// (`const uefi = std.os.uefi;`), landing on either a fresh file's root scope or a genuine
/// value/type expression (whichever comes first).
fn followInit(ctx: *Ctx, tree: *const Ast, dir: []const u8, init_node: Ast.Node.Index) Error!Resolved {
    if (tree.nodeTag(init_node) == .identifier) {
        const token = tree.tokenSlice(tree.nodeMainToken(init_node));
        if (std.mem.eql(u8, token, "true") or std.mem.eql(u8, token, "false") or
            std.mem.eql(u8, token, "bool") or std.mem.eql(u8, token, "anyopaque") or
            std.mem.eql(u8, token, "usize") or std.mem.eql(u8, token, "isize") or
            std.mem.eql(u8, token, "f32") or std.mem.eql(u8, token, "f64") or
            (token.len >= 2 and (token[0] == 'i' or token[0] == 'u') and
                (std.fmt.parseInt(u16, token[1..], 10) catch 0) > 0))
            return .{ .tree = tree, .dir = dir, .node = init_node };
    }
    if (isBuiltinCallTag(tree.nodeTag(init_node)) and std.mem.eql(u8, builtinName(tree, init_node), "@This"))
        return .{ .tree = tree, .dir = dir, .node = null };
    if (isImportDecl(tree, init_node)) {
        var buf: [2]Ast.Node.Index = undefined;
        const args = builtinArgs(tree, init_node, &buf);
        const path = std.zig.string_literal.parseAlloc(ctx.arena, tree.tokenSlice(tree.nodeMainToken(args[0]))) catch return Error.UnsupportedNode;
        const modules = ctx.modules orelse return Error.UnsupportedType;
        const loaded = try modules.load(tree, path);
        return .{ .tree = loaded.tree, .dir = loaded.dir, .node = null };
    }
    return switch (tree.nodeTag(init_node)) {
        .identifier, .field_access => resolveQualifiedName(ctx, tree, dir, init_node),
        else => .{ .tree = tree, .dir = dir, .node = init_node },
    };
}

/// Resolve an identifier or `field_access` chain (`uefi.Guid`, `std.os.uefi.tables.SystemTable`)
/// to whatever real declaration it ultimately names, crossing into another file on disk at
/// every `@import(...)` hop via `ctx.modules`. Used only for a qualified TYPE name or a
/// struct-literal's own type expression - a plain in-scope local/param/global identifier
/// never reaches this (see `lowerAddr`/`findBinding` for that).
fn resolveQualifiedName(ctx: *Ctx, tree: *const Ast, dir: []const u8, node: Ast.Node.Index) Error!Resolved {
    if (ctx.modules) |modules| modules.mark(tree, node);
    if (isImportDecl(tree, node) or
        (isBuiltinCallTag(tree.nodeTag(node)) and std.mem.eql(u8, builtinName(tree, node), "@This")))
        return followInit(ctx, tree, dir, node);
    switch (tree.nodeTag(node)) {
        .identifier => {
            const name = tree.tokenSlice(tree.nodeMainToken(node));
            const init_node = findRootConstInit(tree, name) orelse return Error.UnsupportedType;
            return followInit(ctx, tree, dir, init_node);
        },
        .field_access => {
            const d = tree.nodeData(node).node_and_token;
            const base = try resolveQualifiedName(ctx, tree, dir, d[0]);
            const field_name = tree.tokenSlice(d[1]);
            const init_node = if (base.node) |container|
                findMemberConstInit(base.tree, container, field_name) orelse return Error.UnsupportedType
            else
                findRootConstInit(base.tree, field_name) orelse return Error.UnsupportedType;
            return followInit(ctx, base.tree, base.dir, init_node);
        },
        else => return Error.UnsupportedType,
    }
}

/// Build a `StructDef` for a struct reached through a qualified name, memoized by
/// `ForeignKey` (never `ctx.defs`, which is keyed by plain name within `ctx.tree` only - see
/// `Ctx.resolve`). Unlike `Ctx.resolve` (which requires every member be a field, matching this
/// subset's own struct declarations), a REAL stdlib/foreign struct routinely mixes fields with
/// `comptime {}` blocks, nested `pub const`s, and methods; every non-field member is skipped
/// rather than rejected. `extern struct` uses declared-order, natural alignment
/// for interop; a plain `struct` uses the same layout as this frontend's local
/// structs. Zig deliberately does not promise an external ABI for plain structs,
/// so their fields need only agree across files compiled by this frontend.
/// Packed structs remain unsupported; explicit field alignment is not modeled.
fn resolveForeignStruct(ctx: *Ctx, tree: *const Ast, dir: []const u8, node: Ast.Node.Index) Error!*const StructDef {
    if (tree == ctx.tree) {
        var local_it = ctx.nodes.iterator();
        while (local_it.next()) |entry| if (entry.value_ptr.* == node) return ctx.resolve(entry.key_ptr.*);
    }
    const key = ForeignKey{ .tree = tree, .node = node };
    if (ctx.compiler.foreign_defs.get(key)) |d| return d;
    if (ctx.compiler.foreign_resolving.contains(key)) return Error.UnsupportedType;
    try ctx.compiler.foreign_resolving.put(ctx.arena, key, {});

    var buf: [2]Ast.Node.Index = undefined;
    const cd = tree.fullContainerDecl(&buf, node) orelse return Error.UnsupportedType;
    if (tree.nodeTag(node) != .root and tree.tokenTag(cd.ast.main_token) != .keyword_struct) return Error.UnsupportedType;
    if (cd.layout_token) |layout| {
        if (tree.tokenTag(layout) != .keyword_extern) return Error.UnsupportedType;
    }

    var fields: std.ArrayList(StructField) = .empty;
    var offset: u64 = 0;
    var alignment: u64 = 1;
    for (cd.ast.members) |member| {
        if (tree.fullVarDecl(member)) |vd| {
            const name = tree.tokenSlice(vd.ast.mut_token + 1);
            if (vd.ast.init_node.unwrap()) |member_init| {
                if (tree.fullContainerDecl(undefined, member_init)) |inner| {
                    switch (tree.tokenTag(inner.ast.main_token)) {
                        .keyword_struct, .keyword_enum => try ctx.type_decls.put(ctx.arena, name, .{ .decl_node = member_init, .parent = node }),
                        else => {},
                    }
                }
            }
            continue;
        }
        if (tree.fullFnProto(undefined, member) != null) continue;
        const cf = tree.fullContainerField(member) orelse continue;
        const te = cf.ast.type_expr.unwrap() orelse return Error.UnsupportedType;
        const fty = try resolveTypeIn(ctx, tree, dir, te);
        const fname = tree.tokenSlice(cf.ast.main_token);
        const field_alignment = fty.alignment();
        offset = alignForward(offset, field_alignment);
        try fields.append(ctx.arena, .{ .name = fname, .ty = fty, .offset = offset });
        offset += fty.byteSize();
        alignment = @max(alignment, field_alignment);
    }
    const def = try ctx.arena.create(StructDef);
    def.* = .{
        .name = "",
        .fields = try fields.toOwnedSlice(ctx.arena),
        .byte_size = alignForward(offset, alignment),
        .alignment = alignment,
    };
    try ctx.compiler.foreign_defs.put(ctx.arena, key, def);
    _ = ctx.compiler.foreign_resolving.remove(key);
    return def;
}
fn resolveContainerTypeIn(ctx: *Ctx, tree: *const Ast, dir: []const u8, node: Ast.Node.Index) Error!ZType {
    var buf: [2]Ast.Node.Index = undefined;
    const decl = tree.fullContainerDecl(&buf, node) orelse return Error.UnsupportedType;
    return switch (tree.tokenTag(decl.ast.main_token)) {
        .keyword_opaque => .opaque_type,
        .keyword_enum => ZType{ .enumeration = try resolveEnum(ctx, tree, dir, node, decl) },
        .keyword_struct => ZType{ .strct = try resolveForeignStruct(ctx, tree, dir, node) },
        else => Error.UnsupportedType,
    };
}

/// Resolve a container that names an enum, or `null` when it names something else.
/// Used by the constant evaluator, which reaches an enum through `Enum.field` without
/// knowing the container's kind.
fn resolveEnumTypeIn(ctx: *Ctx, tree: *const Ast, dir: []const u8, node: Ast.Node.Index) ?ZType {
    var buf: [2]Ast.Node.Index = undefined;
    const decl = tree.fullContainerDecl(&buf, node) orelse return null;
    if (tree.tokenTag(decl.ast.main_token) != .keyword_enum) return null;
    return ZType{ .enumeration = resolveEnum(ctx, tree, dir, node, decl) catch return null };
}

/// Build (once per defining container) the named-tag table of an enum declaration.
/// Tags keep declaration order and each value is the previous value plus one unless the
/// field names one explicitly. The backing type defaults to `isize`/`i32`, and a trailing
/// `_` makes the enum non-exhaustive.
fn resolveEnum(ctx: *Ctx, tree: *const Ast, dir: []const u8, node: Ast.Node.Index, decl: Ast.full.ContainerDecl) Error!*const EnumDef {
    const key = ForeignKey{ .tree = tree, .node = node };
    if (ctx.compiler.enum_defs.get(key)) |def| return def;
    if (ctx.compiler.enum_resolving.contains(key)) return Error.UnsupportedType;
    try ctx.compiler.enum_resolving.put(ctx.arena, key, {});

    const backing: ZInt = if (decl.ast.arg.unwrap()) |arg_node| blk: {
        const ty = try resolveTypeIn(ctx, tree, dir, arg_node);
        if (ty != .int) return Error.UnsupportedType;
        if (ty.int.bits == 0 or ty.int.bits > 64) return Error.UnsupportedType;
        break :blk ty.int;
    } else .{ .signed = true, .bits = if (ctx.target == .x86) 32 else 64 };

    var tags: std.ArrayList(EnumField) = .empty;
    var next: ?u64 = null;
    var non_exhaustive = false;
    for (decl.ast.members) |member| {
        const cf = tree.fullContainerField(member) orelse continue;
        const name = tree.tokenSlice(cf.ast.main_token);
        if (std.mem.eql(u8, name, "_")) {
            if (cf.ast.value_expr.unwrap() != null) return Error.UnsupportedNode;
            non_exhaustive = true;
            continue;
        }
        const value: u64 = if (cf.ast.value_expr.unwrap()) |value_node| blk: {
            const evaluated = evalInt(ctx, tree, value_node, &.{}, 0, null) orelse return Error.UnsupportedType;
            if (evaluated.zty != .int and evaluated.zty != .enumeration) return Error.UnsupportedType;
            break :blk @bitCast(evaluated.value.int);
        } else next orelse 0;
        if (!enumValueFits(backing, value)) return Error.IntegerOverflow;
        for (tags.items) |existing| {
            const width: u6 = @intCast(@max(backing.bits, 1) - 1);
            const mask: u64 = if (backing.bits >= 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(backing.bits)) - 1;
            _ = width;
            if ((existing.value & mask) == (value & mask)) return Error.DuplicateDecl;
        }
        try tags.append(ctx.arena, .{ .name = name, .value = value });
        next = if (value == std.math.maxInt(u64)) null else value + 1;
    }
    if (tags.items.len == 0 and !non_exhaustive) {
        // A zero-tag enum has no valid value in Zig; it cannot be stored here.
        return Error.UnsupportedType;
    }
    if (tags.items.len != 0 and !non_exhaustive and next == null) return Error.IntegerOverflow;

    const def = try ctx.arena.create(EnumDef);
    def.* = .{
        .name = "",
        .backing = backing,
        .fields = try tags.toOwnedSlice(ctx.arena),
        .non_exhaustive = non_exhaustive,
    };
    try ctx.compiler.enum_defs.put(ctx.arena, key, def);
    _ = ctx.compiler.enum_resolving.remove(key);
    return def;
}

fn enumValueFits(backing: ZInt, value: u64) bool {
    if (backing.bits >= 64) return true;
    if (backing.signed) {
        const sign_bit: u64 = @as(u64, 1) << @intCast(backing.bits - 1);
        const mask: u64 = (@as(u64, 1) << @intCast(backing.bits)) - 1;
        _ = sign_bit;
        return value & ~mask == 0;
    }
    return value >> @intCast(backing.bits) == 0;
}

/// Resolve a type expression node to a `ZType`: `bool`, `void` is NEVER valid here (callers
/// handle `void` themselves before calling this, only in a return-type position), `iN`/`uN`
/// (plus `usize`/`isize` as 64-bit special cases), `f32`/`f64`, `*T` (single-item pointer
/// only - `fullPtrType`'s `.size != .one` is `Error.UnsupportedType`), nullable `?*T`,
/// `[N]T` (`N` must be a literal), or a declared struct's name.
fn resolveType(ctx: *Ctx, node: Ast.Node.Index) Error!ZType {
    return resolveTypeIn(ctx, ctx.tree, ctx.dir, node);
}

fn resolveTypeIn(ctx: *Ctx, tree: *const Ast, dir: []const u8, node: Ast.Node.Index) Error!ZType {
    if (ctx.modules) |modules| modules.mark(tree, node);
    switch (tree.nodeTag(node)) {
        .identifier => {
            const name = tree.tokenSlice(tree.nodeMainToken(node));
            if (std.mem.eql(u8, name, "bool")) return .boolean;
            if (std.mem.eql(u8, name, "anyopaque")) return .opaque_type;
            if (std.mem.eql(u8, name, "usize")) return ZType{ .int = .{ .signed = false, .bits = 64 } };
            if (std.mem.eql(u8, name, "isize")) return ZType{ .int = .{ .signed = true, .bits = 64 } };
            if (std.mem.eql(u8, name, "f32")) return ZType{ .float = .f32 };
            if (std.mem.eql(u8, name, "f64")) return ZType{ .float = .f64 };
            if (name.len >= 2 and (name[0] == 'i' or name[0] == 'u')) {
                if (std.fmt.parseInt(u16, name[1..], 10)) |bits| {
                    if (bits > 0) return ZType{ .int = .{ .signed = name[0] == 'i', .bits = bits } };
                } else |_| {}
            }
            // `ctx.tree` keeps its own well-tested by-name struct table (`ctx.resolve`,
            // forward-reference tolerant); any OTHER file's local identifier is looked up
            // fresh and built via the general foreign-struct path instead.
            if (tree == ctx.tree and ctx.nodes.contains(name)) return ZType{ .strct = try ctx.resolve(name) };
            if (tree == ctx.tree) {
                if (ctx.local_types.get(name)) |zty| return zty;
            }
            if (tree == ctx.tree) {
                if (ctx.type_decls.contains(name)) {
                    const local = try ctx.compiler.contextFor(tree);
                    if (local.type_decls.get(name)) |local_decl| {
                        if (resolveEnumTypeIn(local, tree, ctx.dir, local_decl.decl_node)) |zty| return zty;
                    }
                }
            }
            const resolved = try resolveQualifiedName(ctx, tree, dir, node);
            if (resolved.node) |decl| {
                return switch (resolved.tree.nodeTag(decl)) {
                    .container_decl, .container_decl_trailing, .container_decl_two, .container_decl_two_trailing, .container_decl_arg, .container_decl_arg_trailing => try resolveContainerTypeIn(ctx, resolved.tree, resolved.dir, decl),
                    else => try resolveTypeIn(ctx, resolved.tree, resolved.dir, decl),
                };
            }
            return ZType{ .strct = try resolveForeignStruct(ctx, resolved.tree, resolved.dir, .root) };
        },
        .ptr_type_aligned, .ptr_type_sentinel, .ptr_type, .ptr_type_bit_range => {
            const pt = tree.fullPtrType(node) orelse return Error.UnsupportedType;
            const pointee = resolveTypeIn(ctx, tree, dir, pt.ast.child_type) catch if (tree != ctx.tree) .opaque_type else return Error.UnsupportedType;
            const p = try ctx.arena.create(ZType);
            p.* = pointee;
            if (pt.size == .slice) return ZType{ .slice = p };
            if (pt.size != .one and pt.size != .many) return Error.UnsupportedType;
            return ZType{ .ptr = p };
        },
        .error_union => {
            const nodes = tree.nodeData(node).node_and_node;
            const errors = try resolveErrorSetIn(ctx, tree, dir, nodes[0]);
            const payload_node = nodes[1];
            if (isVoidType(tree, payload_node)) return .error_void;
            const payload = try ctx.arena.create(ZType);
            payload.* = try resolveTypeIn(ctx, tree, dir, payload_node);
            return ZType{ .error_union = .{ .payload = payload, .errors = errors } };
        },
        .bool_not => {
            // In type context, !T is an inferred error union `!T`
            const child_node = tree.nodeData(node).node;
            if (isVoidType(tree, child_node)) return .error_void;
            const payload = try ctx.arena.create(ZType);
            payload.* = try resolveTypeIn(ctx, tree, dir, child_node);
            return ZType{ .error_union = .{ .payload = payload, .errors = try ctx.defaultErrorSet() } };
        },

        .optional_type => {
            const payload = try resolveTypeIn(ctx, tree, dir, tree.nodeData(node).node);
            const p = try ctx.arena.create(ZType);
            if (payload == .ptr) {
                p.* = payload.ptr.*;
                return ZType{ .optional_ptr = p };
            }
            if (payload.isAggregate()) {
                if (payload == .opaque_type) return Error.UnsupportedType;
            } else if (payload != .int or
                (payload.int.bits != 8 and payload.int.bits != 16 and payload.int.bits != 32 and payload.int.bits != 64))
            {
                return Error.UnsupportedType;
            }
            if (ctx.target == .x86 and payload == .int and payload.byteSize() == 8) return Error.UnsupportedType;
            p.* = payload;
            return ZType{ .optional_value = p };
        },
        .array_type, .array_type_sentinel => {
            const at = tree.fullArrayType(node) orelse return Error.UnsupportedType;
            const elem = try resolveTypeIn(ctx, tree, dir, at.ast.elem_type);
            const e = try ctx.arena.create(ZType);
            e.* = elem;
            const len = evalArrayLength(ctx, tree, at.ast.elem_count, 0) orelse return ZType{ .array = .{ .len = 0, .elem = e } };
            return ZType{ .array = .{ .len = len, .elem = e } };
        },
        .field_access => {
            const resolved = try resolveQualifiedName(ctx, tree, dir, node);
            if (resolved.node) |decl| {
                return switch (resolved.tree.nodeTag(decl)) {
                    .container_decl, .container_decl_trailing, .container_decl_two, .container_decl_two_trailing, .container_decl_arg, .container_decl_arg_trailing => try resolveContainerTypeIn(ctx, resolved.tree, resolved.dir, decl),
                    else => try resolveTypeIn(ctx, resolved.tree, resolved.dir, decl),
                };
            }
            return ZType{ .strct = try resolveForeignStruct(ctx, resolved.tree, resolved.dir, .root) };
        },
        .container_decl, .container_decl_trailing, .container_decl_two, .container_decl_two_trailing, .container_decl_arg, .container_decl_arg_trailing => return resolveContainerTypeIn(ctx, tree, dir, node),
        else => return Error.UnsupportedType,
    }
}
fn resolveErrorSetIn(ctx: *Ctx, tree: *const Ast, dir: []const u8, node: Ast.Node.Index) Error!*const ZErrorSet {
    var set_tree = tree;
    var set_node = node;
    if (tree.nodeTag(node) != .error_set_decl) {
        const resolved = try resolveQualifiedName(ctx, tree, dir, node);
        set_tree = resolved.tree;
        set_node = resolved.node orelse return Error.UnsupportedType;
    }
    if (set_tree.nodeTag(set_node) != .error_set_decl) return Error.UnsupportedType;
    const tokens = set_tree.nodeData(set_node).token_and_token;
    const first = tokens[0] + 1;
    const end = tokens[1];
    var names: std.ArrayList([]const u8) = .empty;
    var token = first;
    while (token < end) : (token += 1) {
        if (set_tree.tokenTag(token) == .identifier)
            try names.append(ctx.arena, set_tree.tokenSlice(token));
    }
    const result = try ctx.arena.create(ZErrorSet);
    result.* = .{ .names = try names.toOwnedSlice(ctx.arena) };
    return result;
}

fn resolveFnReturnType(ctx: *Ctx, tree: *const Ast, dir: []const u8, ret_node: Ast.Node.Index) Error!?ZType {
    if (isNoReturnType(tree, ret_node)) return null;
    const first = tree.firstToken(ret_node);
    const has_bang = first > 0 and tree.tokenTag(first - 1) == .bang;
    if (has_bang) {
        if (isVoidType(tree, ret_node)) return .error_void;
        const payload = try ctx.arena.create(ZType);
        payload.* = try resolveTypeIn(ctx, tree, dir, ret_node);
        return ZType{ .error_union = .{ .payload = payload, .errors = try ctx.defaultErrorSet() } };
    }
    if (isVoidType(tree, ret_node)) return null;
    return try resolveTypeIn(ctx, tree, dir, ret_node);
}

fn isVoidType(tree: *const Ast, node: Ast.Node.Index) bool {
    return tree.nodeTag(node) == .identifier and std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "void");
}

fn isNoReturnType(tree: *const Ast, node: Ast.Node.Index) bool {
    return tree.nodeTag(node) == .identifier and std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "noreturn");
}

fn isUndefinedLiteral(tree: *const Ast, node: Ast.Node.Index) bool {
    return tree.nodeTag(node) == .identifier and std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "undefined");
}

fn allZero(bytes: []const u8) bool {
    for (bytes) |byte| if (byte != 0) return false;
    return true;
}

fn evalUnsignedIntegerExpr(tree: *const Ast, node: Ast.Node.Index, depth: u8) ?u64 {
    if (depth == 32) return null;
    switch (tree.nodeTag(node)) {
        .number_literal => return std.fmt.parseInt(u64, tree.tokenSlice(tree.nodeMainToken(node)), 0) catch null,
        .identifier => {
            const name = tree.tokenSlice(tree.nodeMainToken(node));
            for (tree.rootDecls()) |decl| {
                const vd = tree.fullVarDecl(decl) orelse continue;
                if (tree.tokenTag(vd.ast.mut_token) != .keyword_const) continue;
                if (!std.mem.eql(u8, tree.tokenSlice(vd.ast.mut_token + 1), name)) continue;
                const init = vd.ast.init_node.unwrap() orelse return null;
                return evalUnsignedIntegerExpr(tree, init, depth + 1);
            }
            return null;
        },
        .grouped_expression => return evalUnsignedIntegerExpr(tree, tree.nodeData(node).node_and_token[0], depth + 1),
        .add, .sub, .mul, .shl => {
            const operands = tree.nodeData(node).node_and_node;
            const lhs = evalUnsignedIntegerExpr(tree, operands[0], depth + 1) orelse return null;
            const rhs = evalUnsignedIntegerExpr(tree, operands[1], depth + 1) orelse return null;
            return switch (tree.nodeTag(node)) {
                .add => blk: {
                    const sum = @addWithOverflow(lhs, rhs);
                    break :blk if (sum[1] == 0) sum[0] else null;
                },
                .sub => if (lhs >= rhs) lhs - rhs else null,
                .mul => if (lhs == 0 or rhs <= std.math.maxInt(u64) / lhs) lhs * rhs else null,
                .shl => blk: {
                    if (rhs >= 64) break :blk null;
                    const shift: u6 = @intCast(rhs);
                    const factor: u64 = @as(u64, 1) << shift;
                    break :blk if (lhs <= std.math.maxInt(u64) / factor) lhs * factor else null;
                },
                else => unreachable,
            };
        },
        else => return null,
    }
}

fn integerLiteralBits(tree: *const Ast, node: Ast.Node.Index, zint: ZInt) Error!struct { bits: u64, negative: bool } {
    const tag = tree.nodeTag(node);
    const expr = if (tag == .negation) tree.nodeData(node).node else node;
    const magnitude = evalUnsignedIntegerExpr(tree, expr, 0) orelse return Error.UnsupportedNode;
    const negative = tag == .negation;
    if (negative and !zint.signed) return Error.IntegerOverflow;
    if (zint.bits < 64) {
        const width: u6 = @intCast(if (negative) zint.bits - 1 else if (zint.signed) zint.bits - 1 else zint.bits);
        const limit = @as(u64, 1) << width;
        if (magnitude >= limit and (!negative or magnitude > limit)) return Error.IntegerOverflow;
    }
    return .{ .bits = if (negative) 0 -% magnitude else magnitude, .negative = negative };
}

/// Fold the scalar subset's top-level initializer to its target-endian bytes. This is kept
/// separate from function lowering because a global must be materialized before any function
/// can take its address.
fn encodeGlobalScalar(allocator: std.mem.Allocator, tree: *const Ast, zty: ZType, node: Ast.Node.Index) Error![]u8 {
    const bytes = try allocator.alloc(u8, @intCast(zty.byteSize()));
    errdefer allocator.free(bytes);
    @memset(bytes, 0);
    switch (zty) {
        .boolean => {
            const name = if (tree.nodeTag(node) == .identifier) tree.tokenSlice(tree.nodeMainToken(node)) else return Error.UnsupportedNode;
            if (std.mem.eql(u8, name, "true")) bytes[0] = 1 else if (!std.mem.eql(u8, name, "false")) return Error.UnsupportedNode;
        },
        .int => |zint| {
            const value = try integerLiteralBits(tree, node, zint);
            if (value.negative) @memset(bytes, 0xff);
            var bits = value.bits;
            for (bytes[0..@min(bytes.len, 8)]) |*byte| {
                byte.* = @truncate(bits);
                bits >>= 8;
            }
        },
        .float => return Error.UnsupportedNode,
        else => return Error.UnsupportedType,
    }
    return bytes;
}

fn encodeConstValue(allocator: std.mem.Allocator, zty: ZType, value: ConstValue) Error![]u8 {
    const bytes = try allocator.alloc(u8, @intCast(zty.byteSize()));
    errdefer allocator.free(bytes);
    @memset(bytes, 0);
    switch (value) {
        .boolean => |boolean| {
            if (zty != .boolean) return Error.TypeMismatch;
            bytes[0] = @intFromBool(boolean);
        },
        .int => |integer| {
            if (zty != .int) return Error.TypeMismatch;
            var bits: u64 = @bitCast(integer);
            for (bytes[0..@min(bytes.len, 8)]) |*byte| {
                byte.* = @truncate(bits);
                bits >>= 8;
            }
        },
    }
    return bytes;
}
/// this only ever produces static bytes, never IR.
fn encodeGlobalValue(allocator: std.mem.Allocator, tree: *const Ast, zty: ZType, node: Ast.Node.Index) Error![]u8 {
    if (zty == .array and isBuiltinCallTag(tree.nodeTag(node)) and std.mem.eql(u8, builtinName(tree, node), "@splat")) {
        const za = zty.array;
        var buf: [2]Ast.Node.Index = undefined;
        const args = builtinArgs(tree, node, &buf);
        if (args.len != 1) return Error.CallArityMismatch;
        const elem_size = za.elem.byteSize();
        const elem_bytes = try encodeGlobalValue(allocator, tree, za.elem.*, args[0]);
        defer allocator.free(elem_bytes);
        if (elem_bytes.len != elem_size) return Error.UnsupportedType;

        const bytes = try allocator.alloc(u8, @intCast(zty.byteSize()));
        errdefer allocator.free(bytes);
        for (0..za.len) |i| {
            @memcpy(bytes[i * elem_size ..][0..elem_size], elem_bytes);
        }
        return bytes;
    }
    switch (zty) {
        .optional_value => |payload| {
            const bytes = try allocator.alloc(u8, @intCast(zty.byteSize()));
            errdefer allocator.free(bytes);
            @memset(bytes, 0);
            if (tree.nodeTag(node) == .identifier and std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "null")) return bytes;
            if (payload.isAggregate()) {
                const value = try encodeGlobalValue(allocator, tree, payload.*, node);
                defer allocator.free(value);
                @memcpy(bytes[0..value.len], value);
                bytes[payload.byteSize()] = 1;
            } else {
                const value = try encodeGlobalScalar(allocator, tree, payload.*, node);
                defer allocator.free(value);
                @memcpy(bytes[0..value.len], value);
                bytes[payload.byteSize()] = 1;
            }

            return bytes;
        },
        .boolean, .int, .float => return encodeGlobalScalar(allocator, tree, zty, node),
        .array => |za| {
            const bytes = try allocator.alloc(u8, @intCast(zty.byteSize()));
            errdefer allocator.free(bytes);
            @memset(bytes, 0);
            var buf: [2]Ast.Node.Index = undefined;
            const ai = tree.fullArrayInit(&buf, node) orelse return Error.UnsupportedNode;
            if (ai.ast.elements.len > za.len) return Error.UnsupportedNode;
            const elem_size = za.elem.byteSize();
            for (ai.ast.elements, 0..) |ev, i| {
                const elem_bytes = try encodeGlobalValue(allocator, tree, za.elem.*, ev);
                defer allocator.free(elem_bytes);
                @memcpy(bytes[i * elem_size ..][0..elem_bytes.len], elem_bytes);
            }
            return bytes;
        },
        .strct => |sd| {
            const bytes = try allocator.alloc(u8, @intCast(zty.byteSize()));
            errdefer allocator.free(bytes);
            @memset(bytes, 0);
            var buf: [2]Ast.Node.Index = undefined;
            const si = tree.fullStructInit(&buf, node) orelse return Error.UnsupportedNode;
            for (si.ast.fields) |fv| {
                const eq_tok = tree.firstToken(fv) - 1;
                const name_tok = eq_tok - 1;
                const fname = tree.tokenSlice(name_tok);
                const field = findField(sd, fname) orelse return Error.UnknownField;
                const field_bytes = try encodeGlobalValue(allocator, tree, field.ty, fv);
                defer allocator.free(field_bytes);
                @memcpy(bytes[@intCast(field.offset)..][0..field_bytes.len], field_bytes);
            }
            return bytes;
        },
        else => return Error.UnsupportedType,
    }
}

fn lowerGlobalData(
    allocator: std.mem.Allocator,
    tree: *const Ast,
    name: []const u8,
    zty: ZType,
    is_const: bool,
    init_node: Ast.Node.Index,
    constant: ?ConstValue,
) Error!Data {
    const owned_name = try allocator.dupe(u8, name);
    errdefer allocator.free(owned_name);
    if (isUndefinedLiteral(tree, init_node)) {
        if (is_const) return Error.UnsupportedNode;
        return .{ .name = owned_name, .bytes = try allocator.alloc(u8, 0), .kind = .bss, .size = zty.byteSize() };
    }
    const bytes = if (constant) |value|
        try encodeConstValue(allocator, zty, value)
    else
        try encodeGlobalValue(allocator, tree, zty, init_node);
    const kind: DataKind = if (is_const) .rodata else if (allZero(bytes)) .bss else .data;
    if (kind == .bss) {
        allocator.free(bytes);
        return .{ .name = owned_name, .bytes = try allocator.alloc(u8, 0), .kind = .bss, .size = zty.byteSize() };
    }
    return .{ .name = owned_name, .bytes = bytes, .kind = kind, .size = bytes.len };
}

fn ensureGlobal(ctx: *Ctx, name: []const u8) Error!?GlobalDef {
    if (ctx.globals.get(name)) |global| return global;
    const global_info = ctx.global_decls.get(name) orelse return null;
    const tree = ctx.tree;
    const decl = global_info.decl_node;
    const vd = tree.fullVarDecl(decl) orelse return Error.UnsupportedNode;
    const init_node = vd.ast.init_node.unwrap();
    const is_const = tree.tokenTag(vd.ast.mut_token) == .keyword_const;
    const is_extern = if (vd.extern_export_token) |token|
        tree.tokenTag(token) == .keyword_extern
    else
        false;
    const is_export = declarationHasModifier(tree, decl, .keyword_export);
    const symbol = try ctx.compiler.linkName(name, ctx.is_root or is_export or is_extern);
    if (init_node == null and !is_extern) return Error.UnsupportedNode;
    const type_node = vd.ast.type_node.unwrap();
    if (type_node == null and init_node != null) {
        if (utf16LiteralArg(tree, init_node.?)) |string_node| {
            const encoded = try encodeUtf16LiteralGlobal(ctx.compiler.gpa, ctx.arena, tree, string_node);
            const kind: DataKind = if (is_const) .rodata else if (allZero(encoded.bytes)) .bss else .data;
            const bytes = if (kind == .bss) blk: {
                ctx.compiler.gpa.free(encoded.bytes);
                break :blk try ctx.compiler.gpa.alloc(u8, 0);
            } else encoded.bytes;
            errdefer ctx.compiler.gpa.free(bytes);
            const data_name = try ctx.compiler.gpa.dupe(u8, symbol);
            errdefer ctx.compiler.gpa.free(data_name);
            const global = GlobalDef{ .zty = encoded.zty, .symbol = symbol };
            try ctx.globals.put(ctx.arena, name, global);
            try ctx.compiler.data.append(ctx.arena, .{
                .name = data_name,
                .bytes = bytes,
                .kind = kind,
                .size = encoded.zty.byteSize(),
            });
            return global;
        }
    }
    const zty = if (type_node) |node|
        try resolveType(ctx, node)
    else if (init_node) |node| blk: {
        if (!isStructInitTag(tree.nodeTag(node))) return Error.UnsupportedType;
        var init_buf: [2]Ast.Node.Index = undefined;
        const init = tree.fullStructInit(&init_buf, node) orelse return Error.UnsupportedNode;
        const type_expr = init.ast.type_expr.unwrap() orelse return Error.UnsupportedType;
        break :blk try resolveType(ctx, type_expr);
    } else return Error.UnsupportedType;
    const global = GlobalDef{ .zty = zty, .symbol = symbol };
    try ctx.globals.put(ctx.arena, name, global);
    if (is_extern) return global;
    const init = init_node orelse return Error.UnsupportedNode;
    const constant = evalTopConst(ctx, init);
    const data = try lowerGlobalData(ctx.compiler.gpa, tree, symbol, zty, is_const, init, if (constant) |entry| entry.value else null);
    errdefer {
        ctx.compiler.gpa.free(data.name);
        ctx.compiler.gpa.free(data.bytes);
    }
    try ctx.compiler.data.append(ctx.arena, data);
    return global;
}

fn isStructInitTag(tag: Ast.Node.Tag) bool {
    return switch (tag) {
        .struct_init_one, .struct_init_one_comma, .struct_init_dot_two, .struct_init_dot_two_comma, .struct_init_dot, .struct_init_dot_comma, .struct_init, .struct_init_comma => true,
        else => false,
    };
}

fn isArrayInitTag(tag: Ast.Node.Tag) bool {
    return switch (tag) {
        .array_init_one, .array_init_one_comma, .array_init_dot_two, .array_init_dot_two_comma, .array_init_dot, .array_init_dot_comma, .array_init, .array_init_comma => true,
        else => false,
    };
}

fn isBuiltinCallTag(tag: Ast.Node.Tag) bool {
    return switch (tag) {
        .builtin_call_two, .builtin_call_two_comma, .builtin_call, .builtin_call_comma => true,
        else => false,
    };
}

/// A builtin call's argument nodes, uniform over its two-shape family: `@a(b, c)` (0-2 args,
/// `.opt_node_and_opt_node`) or `@a(b, c, d, ...)` (any count, `.extra_range`). Mirrors
/// `blockStatements`'s two-shape handling for the same reason (no `fullX` helper exists for
/// a builtin call, unlike `fullCall`).
fn builtinArgs(tree: *const Ast, node: Ast.Node.Index, buf: *[2]Ast.Node.Index) []const Ast.Node.Index {
    switch (tree.nodeTag(node)) {
        .builtin_call_two, .builtin_call_two_comma => {
            const d = tree.nodeData(node).opt_node_and_opt_node;
            var n: usize = 0;
            if (d[0].unwrap()) |a| {
                buf[0] = a;
                n = 1;
            }
            if (d[1].unwrap()) |b| {
                buf[n] = b;
                n += 1;
            }
            return buf[0..n];
        },
        .builtin_call, .builtin_call_comma => return tree.extraDataSlice(tree.nodeData(node).extra_range, Ast.Node.Index),
        else => return &.{},
    }
}

fn builtinName(tree: *const Ast, node: Ast.Node.Index) []const u8 {
    return tree.tokenSlice(tree.nodeMainToken(node));
}

/// `@cVaStart()` used as anything but a var-decl initializer (see `lowerStmt`'s special case):
/// unlike every other builtin, it has no destination-free lowering (it must initialize a
/// specific `va_list` slot's memory in place, exactly like a struct literal - see this file's
/// header doc on why structs/aggregates are never free-floating IR values here either).
fn isCVaStartCall(tree: *const Ast, node: Ast.Node.Index) bool {
    if (!isBuiltinCallTag(tree.nodeTag(node))) return false;
    if (!std.mem.eql(u8, builtinName(tree, node), "@cVaStart")) return false;
    var buf: [2]Ast.Node.Index = undefined;
    return builtinArgs(tree, node, &buf).len == 0;
}

fn isBuiltinCpuArch(tree: *const Ast, node: Ast.Node.Index) bool {
    if (tree.nodeTag(node) != .field_access) return false;
    const arch = tree.nodeData(node).node_and_token;
    if (!std.mem.eql(u8, tree.tokenSlice(arch[1]), "arch") or tree.nodeTag(arch[0]) != .field_access) return false;
    const cpu = tree.nodeData(arch[0]).node_and_token;
    return std.mem.eql(u8, tree.tokenSlice(cpu[1]), "cpu") and tree.nodeTag(cpu[0]) == .identifier and std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(cpu[0])), "builtin");
}

/// Recognizes the `<ns>.utf8ToUtf16LeStringLiteral(<string-literal>).*` idiom real Zig code
/// uses to build a UTF-16LE array constant from a UTF-8 string literal at comptime. This
/// frontend performs no general comptime evaluation (see this file's header); this one
/// well-known stdlib pattern is recognized by shape and computed directly, the same way
/// `evalBuiltinArchEqual`/`isCVaStartCall` recognize other single named idioms. Returns the
/// string-literal argument node, or `null` if `node` does not match the shape.
fn utf16LiteralArg(tree: *const Ast, node: Ast.Node.Index) ?Ast.Node.Index {
    if (tree.nodeTag(node) != .deref) return null;
    const call_node = tree.nodeData(node).node;
    var buf: [1]Ast.Node.Index = undefined;
    const c = tree.fullCall(&buf, call_node) orelse return null;
    if (tree.nodeTag(c.ast.fn_expr) != .field_access) return null;
    const fa = tree.nodeData(c.ast.fn_expr).node_and_token;
    if (!std.mem.eql(u8, tree.tokenSlice(fa[1]), "utf8ToUtf16LeStringLiteral")) return null;
    if (c.ast.params.len != 1 or tree.nodeTag(c.ast.params[0]) != .string_literal) return null;
    return c.ast.params[0];
}

/// Encodes `string_node`'s UTF-8 bytes as a NUL-terminated UTF-16LE array (matching
/// `std.unicode.utf8ToUtf16LeStringLiteral`'s `[N:0]u16` result - modeled here as a plain
/// `[N+1]u16` array, an identical byte layout since this frontend's `ZType.array` has no
/// sentinel concept). `bytes` is `gpa`-allocated (it outlives the lowering arena as part of
/// the returned `Module`); `zty`'s `elem` is `arena`-allocated like every other `ZType`.
fn encodeUtf16LiteralGlobal(gpa: std.mem.Allocator, arena: std.mem.Allocator, tree: *const Ast, string_node: Ast.Node.Index) Error!struct { zty: ZType, bytes: []u8 } {
    const utf8 = std.zig.string_literal.parseAlloc(gpa, tree.tokenSlice(tree.nodeMainToken(string_node))) catch return Error.UnsupportedNode;
    defer gpa.free(utf8);
    const len = std.unicode.calcUtf16LeLen(utf8) catch return Error.UnsupportedNode;
    const units = try gpa.alloc(u16, len);
    defer gpa.free(units);
    const written = std.unicode.utf8ToUtf16Le(units, utf8) catch return Error.UnsupportedNode;
    std.debug.assert(written == len);
    const bytes = try gpa.alloc(u8, (len + 1) * 2);
    errdefer gpa.free(bytes);
    for (units, 0..) |unit, i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], unit, .little);
    bytes[len * 2] = 0;
    bytes[len * 2 + 1] = 0;
    const elem = try arena.create(ZType);
    elem.* = ZType{ .int = .{ .signed = false, .bits = 16 } };
    return .{ .zty = ZType{ .array = .{ .len = len + 1, .elem = elem } }, .bytes = bytes };
}

/// A top-level `const` visible from the current file. The root file keeps its own
/// evaluated table; another file's constant is evaluated in that file's context.
fn localConstEntry(l: *L, name: []const u8) Error!?ConstEntry {
    if (l.r.constants.get(name)) |entry| return entry;
    if (!l.r.deferred_constants.contains(name)) return null;
    const init = l.r.deferred_constants.get(name).?;
    const value = evalInt(l.r, l.r.tree, init, &.{}, 0, null) orelse return null;
    return .{ .zty = value.zty, .value = value.value };
}

/// Materialize an evaluated constant at the type the use site needs. Booleans and
/// integers convert; enum tags keep their nominal type unless coerced to another enum
/// with an equal-shaped backing.
fn constEntryValue(l: *L, entry: ConstEntry, expected: ?ZType) Error!TypedValue {
    if (expected) |zty| {
        if (zty == .enumeration) {
            const value = entry.value.int;
            if (!entry.zty.eql(zty)) {
                if (!comptimeCoerces(enumBackingOf(entry.zty), ZType{ .int = zty.enumeration.backing })) return Error.TypeMismatch;
                if (!zty.enumeration.non_exhaustive and !zty.enumeration.hasValue(@bitCast(value))) return Error.TypeMismatch;
            }
            const irty = try zty.irType(l.func);
            return .{ .value = try l.func.appendInst(l.block, irty, .{ .iconst = value }), .zty = zty };
        }
        if (zty != .int and zty != .boolean) return Error.TypeMismatch;
    }
    switch (entry.value) {
        .boolean => |value| return .{ .value = try l.func.appendInst(l.block, try boolType(l.func), .{ .iconst = @intFromBool(value) }), .zty = .boolean },
        .int => |value| {
            const zty = expected orelse entry.zty;
            if (zty != .int) return Error.TypeMismatch;
            if (!fitsInt(zty, value)) return Error.IntegerOverflow;
            return .{ .value = try l.func.appendInst(l.block, try zty.irType(l.func), .{ .iconst = value }), .zty = zty };
        },
    }
}

fn lowerEnumLiteral(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!TypedValue {
    const zty = expected orelse return Error.TypeMismatch;
    const def = switch (zty) {
        .enumeration => |d| d,
        else => return Error.TypeMismatch,
    };
    const entry = def.field(l.r.tree.tokenSlice(l.r.tree.nodeMainToken(node))) orelse return Error.TypeMismatch;
    return .{ .value = try l.func.appendInst(l.block, try zty.irType(l.func), .{ .iconst = std.math.cast(i64, entry.value) orelse @as(i64, @bitCast(entry.value)) }), .zty = zty };
}

fn enumLike(zty: ZType) bool {
    return zty == .enumeration or zty == .int;
}

fn enumBackingOf(zty: ZType) ZType {
    return switch (zty) {
        .enumeration => |e| ZType{ .int = e.backing },
        .int => zty,
        else => unreachable,
    };
}

fn evalBuiltinArchEqual(ctx: *const Ctx, node: Ast.Node.Index) ?bool {
    if (ctx.tree.nodeTag(node) != .equal_equal) return null;
    const operands = ctx.tree.nodeData(node).node_and_node;
    if (!isBuiltinCpuArch(ctx.tree, operands[0]) or ctx.tree.nodeTag(operands[1]) != .enum_literal) return null;
    const name = ctx.tree.tokenSlice(ctx.tree.nodeMainToken(operands[1]));
    const arch = std.meta.stringToEnum(TargetArch, name) orelse return null;
    return arch == ctx.target;
}

fn evalConstInt(ctx: *Ctx, tree: *const Ast, node: Ast.Node.Index, bindings: []const ConstBinding, depth: u8) ?i64 {
    const result = evalTypedInt(ctx, tree, node, bindings, depth) orelse return null;
    return result.value.int;
}

/// Evaluate an integer expression at its native comptime type. `Enum(usize)` tags and
/// `usize` limits stay unsigned; nothing is silently reinterpreted as `i64`.
fn evalTypedInt(ctx: *Ctx, tree: *const Ast, node: Ast.Node.Index, bindings: []const ConstBinding, depth: u8) ?EvalValue {
    return evalInt(ctx, tree, node, bindings, depth, null);
}

/// Evaluate a typed comptime integer expression. `expected` supplies the type of an
/// untyped literal or enum literal; without one a number keeps its literal value and the
/// evaluator cannot resolve enum fields. Results always carry a real type so `usize`
/// limits and enum backings are never reinterpreted as `i64`.
fn evalInt(
    ctx: *Ctx,
    tree: *const Ast,
    node: Ast.Node.Index,
    bindings: []const ConstBinding,
    depth: u8,
    expected: ?ZType,
) ?EvalValue {
    if (depth == 32) return null;
    switch (tree.nodeTag(node)) {
        .number_literal => {
            const zty = expected orelse ZType{ .int = .{ .signed = true, .bits = 64 } };
            const value: i64 = switch (std.zig.number_literal.parseNumberLiteral(tree.tokenSlice(tree.nodeMainToken(node)))) {
                .int => |bits| @bitCast(bits),
                else => return null,
            };
            if (!fitsInt(zty, value)) return null;
            return .{ .zty = zty, .value = .{ .int = value } };
        },
        .identifier => {
            const name = tree.tokenSlice(tree.nodeMainToken(node));
            for (bindings) |binding| {
                if (std.mem.eql(u8, binding.name, name)) return .{ .zty = expected orelse ZType{ .int = .{ .signed = true, .bits = 64 } }, .value = .{ .int = binding.value } };
            }
            if (tree == ctx.tree) {
                if (ctx.constants.get(name)) |value| {
                    const zty = expected orelse value.zty;
                    if (!zty.eql(value.zty) and !comptimeCoerces(value.zty, zty)) return null;
                    return .{ .zty = zty, .value = value.value };
                }
                // A constant declared later in this file: evaluate it here instead of
                // requiring the caller to have registered it already.
                if (findRootConstInit(tree, name)) |init| {
                    const found = evalInt(ctx, tree, init, &.{}, depth + 1, expected) orelse return null;
                    return found;
                }
            }
            const global = findRootGlobalType(tree, name) orelse return null;
            if (!global.is_const) return null;
            const init = global.init_node orelse return null;
            const body = evalInt(ctx, tree, init, &.{}, depth + 1, null) orelse return null;
            if (global.type_node) |type_node| {
                const declared = resolveTypeIn(ctx, tree, ownerDir(ctx, tree), type_node) catch return null;
                if (!declared.eql(body.zty) and !comptimeCoerces(body.zty, declared)) return null;
                return .{ .zty = declared, .value = body.value };
            }
            return body;
        },
        .field_access => {
            const access = tree.nodeData(node).node_and_token;
            const dir = ownerDir(ctx, tree);
            const resolved = resolveQualifiedName(ctx, tree, dir, access[0]) catch return null;
            const field_name = tree.tokenSlice(access[1]);
            if (resolved.node) |container| {
                const zty = resolveEnumTypeIn(ctx, resolved.tree, resolved.dir, container) orelse return null;
                const entry = zty.enumeration.field(field_name) orelse return null;
                return .{ .zty = zty, .value = .{ .int = @bitCast(entry.value) } };
            }
            const global = findRootGlobalType(resolved.tree, field_name) orelse return null;
            if (!global.is_const) return null;
            const init = global.init_node orelse return null;
            const body = evalInt(ctx, resolved.tree, init, &.{}, depth + 1, null) orelse return null;
            if (global.type_node) |type_node| {
                const declared = resolveTypeIn(ctx, resolved.tree, resolved.dir, type_node) catch return null;
                if (!declared.eql(body.zty) and !comptimeCoerces(body.zty, declared)) return null;
                return .{ .zty = declared, .value = body.value };
            }
            return body;
        },
        .enum_literal => {
            const zty = expected orelse return null;
            const def = switch (zty) {
                .enumeration => |d| d,
                else => return null,
            };
            const entry = def.field(tree.tokenSlice(tree.nodeMainToken(node))) orelse return null;
            return .{ .zty = zty, .value = .{ .int = @bitCast(entry.value) } };
        },
        .grouped_expression => return evalInt(ctx, tree, tree.nodeData(node).node_and_token[0], bindings, depth + 1, expected),
        .negation => {
            const value = evalInt(ctx, tree, tree.nodeData(node).node, bindings, depth + 1, expected) orelse return null;
            if (value.zty != .int and value.zty != .enumeration) return null;
            return .{ .zty = value.zty, .value = .{ .int = 0 -% value.value.int } };
        },
        .add, .sub, .mul, .bit_and, .bit_or, .bit_xor, .shl, .shr => {
            const operands = tree.nodeData(node).node_and_node;
            const lhs = evalInt(ctx, tree, operands[0], bindings, depth + 1, expected) orelse return null;
            const rhs = evalInt(ctx, tree, operands[1], bindings, depth + 1, if (lhs.zty == .enumeration) lhs.zty else expected) orelse return null;
            if (!lhs.zty.eql(rhs.zty)) return null;
            const a = lhs.value.int;
            const b = rhs.value.int;
            const result = switch (tree.nodeTag(node)) {
                .add => a +% b,
                .sub => a -% b,
                .mul => a *% b,
                .bit_and => a & b,
                .bit_or => a | b,
                .bit_xor => a ^ b,
                .shl => if (b >= 0 and b < 64) @as(i64, @bitCast(@as(u64, @bitCast(a)) << @as(u6, @intCast(b)))) else return null,
                .shr => if (b >= 0 and b < 64) a >> @as(u6, @intCast(b)) else return null,
                else => unreachable,
            };
            return .{ .zty = lhs.zty, .value = .{ .int = result } };
        },
        .equal_equal, .bang_equal, .less_than, .greater_than, .less_or_equal, .greater_or_equal => {
            const operands = tree.nodeData(node).node_and_node;
            const lhs = evalInt(ctx, tree, operands[0], bindings, depth + 1, expected) orelse return null;
            const rhs = evalInt(ctx, tree, operands[1], bindings, depth + 1, lhs.zty) orelse return null;
            if (!lhs.zty.eql(rhs.zty)) return null;
            const a = lhs.value.int;
            const b = rhs.value.int;
            const answer = switch (tree.nodeTag(node)) {
                .equal_equal => a == b,
                .bang_equal => a != b,
                .less_than => a < b,
                .greater_than => a > b,
                .less_or_equal => a <= b,
                .greater_or_equal => a >= b,
                else => unreachable,
            };
            return .{ .zty = .boolean, .value = .{ .boolean = answer } };
        },
        .builtin_call, .builtin_call_comma, .builtin_call_two, .builtin_call_two_comma => {
            var buf: [2]Ast.Node.Index = undefined;
            const args = builtinArgs(tree, node, &buf);
            const name = builtinName(tree, node);
            if (std.mem.eql(u8, name, "@as")) {
                if (args.len != 2) return null;
                const target = resolveTypeIn(ctx, tree, ownerDir(ctx, tree), args[0]) catch return null;
                return evalInt(ctx, tree, args[1], bindings, depth + 1, target);
            }
            if (std.mem.eql(u8, name, "@bitCast") or std.mem.eql(u8, name, "@intCast") or std.mem.eql(u8, name, "@truncate")) {
                if (args.len != 1) return null;
                return evalInt(ctx, tree, args[0], bindings, depth + 1, expected);
            }
            if (std.mem.eql(u8, name, "@intFromEnum")) {
                if (args.len != 1) return null;
                const source = evalInt(ctx, tree, args[0], bindings, depth + 1, expected) orelse return null;
                const def = switch (source.zty) {
                    .enumeration => |d| d,
                    else => return null,
                };
                const zty = ZType{ .int = .{ .signed = def.backing.signed, .bits = def.backing.bits } };
                return .{ .zty = zty, .value = source.value };
            }
            if (std.mem.eql(u8, name, "@enumFromInt")) {
                if (args.len != 1) return null;
                const target = expected orelse return null;
                if (target != .enumeration) return null;
                const source = evalInt(ctx, tree, args[0], bindings, depth + 1, null) orelse return null;
                if (source.zty != .int) return null;
                return .{ .zty = target, .value = source.value };
            }
            return null;
        },
        .call_one, .call_one_comma, .call, .call_comma => return evalIntCall(ctx, tree, node, bindings, depth, expected),
        else => return null,
    }
}

/// Evaluate a call to another function at comptime. Only the shape a standard-library
/// integer helper has is supported: fixed parameters, local `const`s bound to evaluated
/// arguments or literals, and a final `return`. Anything else declines, so the caller
/// falls back to ordinary lowering rather than guessing.
fn evalIntCall(
    ctx: *Ctx,
    tree: *const Ast,
    node: Ast.Node.Index,
    bindings: []const ConstBinding,
    depth: u8,
    expected: ?ZType,
) ?EvalValue {
    var call_buf: [1]Ast.Node.Index = undefined;
    const call = tree.fullCall(&call_buf, node) orelse return null;
    if (tree.nodeTag(call.ast.fn_expr) != .identifier) return null;
    const name = tree.tokenSlice(tree.nodeMainToken(call.ast.fn_expr));
    var proto_node: ?Ast.Node.Index = null;
    var body_node: ?Ast.Node.Index = null;
    for (tree.rootDecls()) |decl| {
        if (tree.nodeTag(decl) != .fn_decl) continue;
        const data = tree.nodeData(decl).node_and_node;
        var proto_buf: [1]Ast.Node.Index = undefined;
        const proto = tree.fullFnProto(&proto_buf, data[0]) orelse continue;
        const name_token = proto.name_token orelse continue;
        if (std.mem.eql(u8, tree.tokenSlice(name_token), name)) {
            proto_node = data[0];
            body_node = data[1];
            break;
        }
    }
    const proto = tree.fullFnProto(&call_buf, proto_node orelse return null) orelse return null;
    const body = body_node orelse return null;
    var local_bindings: [8]ConstBinding = undefined;
    var binding_count: usize = 0;
    var param_index: usize = 0;
    var params = proto.iterate(tree);
    while (params.next()) |param| {
        const name_token = param.name_token orelse return null;
        // A `comptime T: type` parameter is bound from the argument's type, not value.
        if (param.comptime_noalias) |token| {
            if (tree.tokenTag(token) == .keyword_comptime) {
                if (param_index >= call.ast.params.len) return null;
                const zty = resolveTypeIn(ctx, tree, ownerDir(ctx, tree), call.ast.params[param_index]) catch return null;
                if (zty != .int or zty.int.bits == 0) return null;
                local_bindings[binding_count] = .{ .name = tree.tokenSlice(name_token), .value = @intCast(zty.int.bits) };
                binding_count += 1;
                param_index += 1;
                continue;
            }
        }
        if (param_index >= call.ast.params.len or binding_count >= local_bindings.len) return null;
        const param_type = resolveTypeIn(ctx, tree, ownerDir(ctx, tree), param.type_expr orelse return null) catch return null;
        const value = evalInt(ctx, tree, call.ast.params[param_index], bindings, depth + 1, if (param_type == .int) param_type else null) orelse return null;
        local_bindings[binding_count] = .{ .name = tree.tokenSlice(name_token), .value = value.value.int };
        binding_count += 1;
        param_index += 1;
    }
    if (param_index != call.ast.params.len) return null;
    var stmt_buf: [2]Ast.Node.Index = undefined;
    const statements = blockStatements(tree, body, &stmt_buf);
    for (statements) |statement| {
        switch (tree.nodeTag(statement)) {
            .simple_var_decl, .local_var_decl => {
                const vd = tree.fullVarDecl(statement) orelse return null;
                if (tree.tokenTag(vd.ast.mut_token) != .keyword_const) return null;
                const var_name = tree.tokenSlice(vd.ast.mut_token + 1);
                const declared = if (vd.ast.type_node.unwrap()) |type_node|
                    resolveTypeIn(ctx, tree, ownerDir(ctx, tree), type_node) catch return null
                else
                    null;
                const init_node = vd.ast.init_node.unwrap() orelse return null;
                const value = evalInt(ctx, tree, init_node, local_bindings[0..binding_count], depth + 1, declared) orelse return null;
                if (binding_count == local_bindings.len) return null;
                local_bindings[binding_count] = .{ .name = var_name, .value = value.value.int };
                binding_count += 1;
            },
            .@"return" => {
                const return_node = tree.nodeData(statement).opt_node.unwrap() orelse return null;
                return evalInt(ctx, tree, return_node, local_bindings[0..binding_count], depth + 1, expected);
            },
            else => return null,
        }
    }
    return null;
}

/// Does `value` fit `zty` as an integer bit pattern?
fn fitsInt(zty: ZType, value: i64) bool {
    const int = switch (zty) {
        .int => |i| i,
        .enumeration => |e| e.backing,
        else => return false,
    };
    if (int.bits >= 64) return true;
    const magnitude: u64 = @bitCast(value);
    return magnitude >> @intCast(int.bits) == 0;
}

/// An implicit coercion allowed during comptime evaluation only (`usize` to `u64`,
/// `comptime_int` to a named integer type). Runtime coercion stays in `convertTo`.
fn comptimeCoerces(from: ZType, to: ZType) bool {
    return switch (from) {
        .enumeration => |a| switch (to) {
            .enumeration => |b| a.backing.signed == b.backing.signed and a.backing.bits <= b.backing.bits,
            else => false,
        },
        .int => |a| switch (to) {
            .int => |b| a.signed == b.signed and a.bits <= b.bits,
            else => false,
        },
        else => false,
    };
}

fn ownerDir(ctx: *Ctx, tree: *const Ast) []const u8 {
    if (ctx.tree == tree) return ctx.dir;
    if (ctx.modules) |modules| if (modules.owner(tree)) |file| return file.dir;
    return ctx.dir;
}
fn evalTopConst(ctx: *Ctx, node: Ast.Node.Index) ?ConstEntry {
    if (evalBuiltinArchEqual(ctx, node)) |value| return .{ .zty = .boolean, .value = .{ .boolean = value } };
    if (ctx.tree.nodeTag(node) == .identifier) {
        const name = ctx.tree.tokenSlice(ctx.tree.nodeMainToken(node));
        if (std.mem.eql(u8, name, "true")) return .{ .zty = .boolean, .value = .{ .boolean = true } };
        if (std.mem.eql(u8, name, "false")) return .{ .zty = .boolean, .value = .{ .boolean = false } };
    }
    const value = evalInt(ctx, ctx.tree, node, &.{}, 0, null) orelse return null;
    return .{ .zty = value.zty, .value = value.value };
}

fn evalDeferredConst(ctx: *Ctx, name: []const u8, depth: u8) ?ConstEntry {
    if (depth == 32) return null;
    if (ctx.constants.get(name)) |value| return value;
    const init = ctx.deferred_constants.get(name) orelse return null;
    if (ctx.tree.nodeTag(init) == .identifier) {
        const alias = ctx.tree.tokenSlice(ctx.tree.nodeMainToken(init));
        if (std.mem.eql(u8, alias, "true")) return .{ .zty = .boolean, .value = .{ .boolean = true } };
        if (std.mem.eql(u8, alias, "false")) return .{ .zty = .boolean, .value = .{ .boolean = false } };
        if (evalDeferredConst(ctx, alias, depth + 1)) |value| return value;
    }
    if (evalBuiltinArchEqual(ctx, init)) |value| return .{ .zty = .boolean, .value = .{ .boolean = value } };
    const value = evalInt(ctx, ctx.tree, init, &.{}, depth + 1, null) orelse return null;
    return .{ .zty = value.zty, .value = value.value };
}

/// Resolve the integer constant expressions used as fixed-array lengths. Imported Zig
/// modules commonly put the length in a named constant; resolve those in their owning tree,
/// rather than accidentally restricting array types to declarations in the entry file.
fn evalArrayLength(ctx: *Ctx, tree: *const Ast, node: Ast.Node.Index, depth: u8) ?u64 {
    if (depth == 32) return null;
    switch (tree.nodeTag(node)) {
        .number_literal => return std.fmt.parseInt(u64, tree.tokenSlice(tree.nodeMainToken(node)), 0) catch null,
        .identifier => {
            const name = tree.tokenSlice(tree.nodeMainToken(node));
            if (tree == ctx.tree) {
                if (ctx.constants.get(name)) |value| switch (value.value) {
                    .int => |integer| return if (integer >= 0) @intCast(integer) else null,
                    .boolean => {},
                };
            }
            for (tree.rootDecls()) |decl| {
                const vd = tree.fullVarDecl(decl) orelse continue;
                if (tree.tokenTag(vd.ast.mut_token) != .keyword_const) continue;
                if (!std.mem.eql(u8, tree.tokenSlice(vd.ast.mut_token + 1), name)) continue;
                const init_node = vd.ast.init_node.unwrap() orelse return null;
                return evalArrayLength(ctx, tree, init_node, depth + 1);
            }
            return null;
        },
        .grouped_expression => return evalArrayLength(ctx, tree, tree.nodeData(node).node_and_token[0], depth + 1),
        .field_access => {
            const resolved = resolveQualifiedName(ctx, tree, ctx.dir, node) catch return null;
            const value_node = resolved.node orelse return null;
            return evalArrayLength(ctx, resolved.tree, value_node, depth + 1);
        },
        .add => {
            const operands = tree.nodeData(node).node_and_node;
            const lhs = evalArrayLength(ctx, tree, operands[0], depth + 1) orelse return null;
            const rhs = evalArrayLength(ctx, tree, operands[1], depth + 1) orelse return null;
            const sum = @addWithOverflow(lhs, rhs);
            return if (sum[1] == 0) sum[0] else null;
        },
        .mul => {
            const operands = tree.nodeData(node).node_and_node;
            const lhs = evalArrayLength(ctx, tree, operands[0], depth + 1) orelse return null;
            const rhs = evalArrayLength(ctx, tree, operands[1], depth + 1) orelse return null;
            if (lhs != 0 and rhs > (std.math.maxInt(u64) / lhs)) return null;
            return lhs * rhs;
        },
        .shl => {
            const operands = tree.nodeData(node).node_and_node;
            const lhs = evalArrayLength(ctx, tree, operands[0], depth + 1) orelse return null;
            const rhs = evalArrayLength(ctx, tree, operands[1], depth + 1) orelse return null;
            if (rhs >= 64) return null;
            const shift: u6 = @intCast(rhs);
            const factor: u64 = @as(u64, 1) << shift;
            if (lhs > (std.math.maxInt(u64) / factor)) return null;
            return lhs * factor;
        },
        .builtin_call, .builtin_call_comma, .builtin_call_two, .builtin_call_two_comma => {
            if (!std.mem.eql(u8, builtinName(tree, node), "@max")) return null;
            var buf: [2]Ast.Node.Index = undefined;
            const args = builtinArgs(tree, node, &buf);
            if (args.len != 2) return null;
            const lhs = evalArrayLength(ctx, tree, args[0], depth + 1) orelse return null;
            const rhs = evalArrayLength(ctx, tree, args[1], depth + 1) orelse return null;
            return @max(lhs, rhs);
        },
        else => return null,
    }
}

fn isImportDecl(tree: *const Ast, node: Ast.Node.Index) bool {
    if (!isBuiltinCallTag(tree.nodeTag(node)) or !std.mem.eql(u8, builtinName(tree, node), "@import")) return false;
    var buf: [2]Ast.Node.Index = undefined;
    const args = builtinArgs(tree, node, &buf);
    return args.len == 1 and tree.nodeTag(args[0]) == .string_literal;
}

fn collectImports(allocator: std.mem.Allocator, tree: *const Ast) Error![]Import {
    var result: std.ArrayList(Import) = .empty;
    errdefer {
        for (result.items) |import| {
            allocator.free(import.name);
            allocator.free(import.path);
        }
        result.deinit(allocator);
    }
    try result.ensureTotalCapacity(allocator, tree.rootDecls().len);
    for (tree.rootDecls()) |decl| {
        const vd = tree.fullVarDecl(decl) orelse continue;
        if (tree.tokenTag(vd.ast.mut_token) != .keyword_const) continue;
        const init_node = vd.ast.init_node.unwrap() orelse continue;
        if (!isImportDecl(tree, init_node)) continue;

        var buf: [2]Ast.Node.Index = undefined;
        const args = builtinArgs(tree, init_node, &buf);
        const name = try allocator.dupe(u8, tree.tokenSlice(vd.ast.mut_token + 1));
        errdefer allocator.free(name);
        const path = std.zig.string_literal.parseAlloc(allocator, tree.tokenSlice(tree.nodeMainToken(args[0]))) catch return Error.UnsupportedNode;
        errdefer allocator.free(path);
        result.appendAssumeCapacity(.{ .name = name, .path = path });
    }
    return result.toOwnedSlice(allocator);
}

/// Import namespace aliases have no runtime storage. File compilation resolves the
/// declarations reached through them and emits supported runtime definitions on demand.
fn isImportNamespace(tree: *const Ast, node: Ast.Node.Index, imports: *const std.StringHashMapUnmanaged(void)) bool {
    if (isImportDecl(tree, node)) return true;
    if (tree.nodeTag(node) != .field_access) return false;
    const base = tree.nodeData(node).node_and_token[0];
    if (tree.nodeTag(base) == .identifier) return imports.contains(tree.tokenSlice(tree.nodeMainToken(base)));
    return isImportNamespace(tree, base, imports);
}

/// A named top-level function's resolved signature, built before any body is lowered so
/// every call site (forward or backward reference) resolves.
const FnSig = struct {
    name: []const u8,
    params: []const ZType,
    ret: ?ZType,
    is_variadic: bool = false,
    is_noreturn: bool = false,
};

/// `Enum.tag` used as a value (the field lives in the container, not in a variable).
fn enumFieldOrDefault(l: *L, fn_expr: Ast.Node.Index) Error!?TypedValue {
    if (l.r.tree.nodeTag(fn_expr) != .field_access) return null;
    const access = l.r.tree.nodeData(fn_expr).node_and_token;
    const resolved = resolveQualifiedName(l.r, l.r.tree, l.r.dir, access[0]) catch return null;
    const container = resolved.node orelse return null;
    const zty = resolveEnumTypeIn(l.r, resolved.tree, resolved.dir, container) orelse return null;
    const field_name = l.r.tree.tokenSlice(access[1]);
    const entry = zty.enumeration.field(field_name) orelse return null;
    return .{ .value = try l.func.appendInst(l.block, try zty.irType(l.func), .{ .iconst = std.math.cast(i64, entry.value) orelse @as(i64, @bitCast(entry.value)) }), .zty = zty };
}

fn findFnSig(sigs: []const FnSig, name: []const u8) ?FnSig {
    for (sigs) |s| if (std.mem.eql(u8, s.name, name)) return s;
    return null;
}

fn declarationHasModifier(tree: *const Ast, decl: Ast.Node.Index, keyword: std.zig.Token.Tag) bool {
    var proto_buf: [1]Ast.Node.Index = undefined;
    if (tree.fullFnProto(&proto_buf, decl)) |proto| {
        const token = proto.extern_export_inline_token orelse return false;
        return tree.tokenTag(token) == keyword;
    }
    if (tree.fullVarDecl(decl)) |vd| {
        const token = vd.extern_export_token orelse return false;
        return tree.tokenTag(token) == keyword;
    }
    return false;
}

fn buildFunctionSig(ctx: *Ctx, proto_node: Ast.Node.Index, symbol: []const u8) Error!FnSig {
    var proto_buf: [1]Ast.Node.Index = undefined;
    const proto = ctx.tree.fullFnProto(&proto_buf, proto_node) orelse return Error.UnsupportedNode;
    const ret_node = proto.ast.return_type.unwrap() orelse return Error.UnsupportedType;
    const ret: ?ZType = try resolveFnReturnType(ctx, ctx.tree, ctx.dir, ret_node);
    if (ret) |rz| {
        if ((rz.isAggregate() and rz != .optional_value and rz != .slice and rz != .error_union) or (rz.optionalPayloadIsAggregate() and (ctx.target != .aarch64 or rz.byteSize() <= 16))) return Error.UnsupportedType;
        if (rz == .error_union and rz.byteSize() > 16 and ctx.target != .aarch64) return Error.UnsupportedType;
    }
    var params: std.ArrayList(ZType) = .empty;
    var variadic = false;
    var iter = proto.iterate(ctx.tree);
    while (iter.next()) |param| {
        if (param.comptime_noalias) |token| {
            if (ctx.tree.tokenTag(token) == .keyword_comptime) return Error.UnsupportedType;
        }
        if (param.anytype_ellipsis3) |tok| {
            if (ctx.tree.tokenTag(tok) != .ellipsis3) return Error.UnsupportedType;
            variadic = true;
            break;
        }
        const type_node = param.type_expr orelse return Error.UnsupportedType;
        const zty = try resolveType(ctx, type_node);
        if ((zty.isAggregate() and zty != .optional_value and zty != .slice) or zty.optionalPayloadIsAggregate()) return Error.UnsupportedType;
        try params.append(ctx.arena, zty);
    }
    return .{
        .name = symbol,
        .params = try params.toOwnedSlice(ctx.arena),
        .ret = ret,
        .is_variadic = variadic,
        .is_noreturn = isNoReturnType(ctx.tree, ret_node),
    };
}

fn ensureFunction(ctx: *Ctx, name: []const u8) Error!?FnSig {
    const info = ctx.function_decls.get(name) orelse return null;
    const key = DeclKey{ .tree = ctx.tree, .node = info.proto_node };
    if (ctx.compiler.functions.get(key)) |entry| return entry.sig;
    const is_extern = info.body_node == null;
    const is_export = declarationHasModifier(ctx.tree, info.decl_node, .keyword_export);
    const external = is_extern or is_export;
    const symbol = try ctx.compiler.linkName(name, ctx.is_root or external);
    const sig = try buildFunctionSig(ctx, info.proto_node, symbol);
    const entry = FunctionEntry{
        .ctx = ctx,
        .info = info,
        .symbol = symbol,
        .sig = sig,
        .is_local = !ctx.is_root and !external,
    };
    try ctx.compiler.functions.put(ctx.arena, key, entry);
    if (!ctx.is_root and !is_extern) {
        try ctx.compiler.function_queue.append(ctx.arena, key);
    }
    return sig;
}

fn ensureLocalFnSig(l: *L, name: []const u8) Error!?FnSig {
    if (findFnSig(l.fn_sigs, name)) |sig| return sig;
    return ensureFunction(l.r, name);
}

fn importedFnSig(l: *L, fn_expr: Ast.Node.Index) Error!?FnSig {
    if (l.r.tree.nodeTag(fn_expr) != .field_access) return null;
    const access = l.r.tree.nodeData(fn_expr).node_and_token;
    const resolved = resolveQualifiedName(l.r, l.r.tree, l.r.dir, access[0]) catch return null;
    if (resolved.node != null) return null;
    const name = l.r.tree.tokenSlice(access[1]);
    const imported_ctx = try l.r.compiler.contextFor(resolved.tree);
    return ensureFunction(imported_ctx, name);
}

fn foreignMethodSig(l: *L, receiver: ZType, method_name: []const u8) Error!?FnSig {
    const struct_def = switch (receiver) {
        .strct => |s| s,
        .ptr => |p| if (p.* == .strct) p.strct else return null,
        else => return null,
    };
    var found: ?ForeignKey = null;
    var foreign_it = l.r.compiler.foreign_defs.iterator();
    while (foreign_it.next()) |entry| {
        if (entry.value_ptr.* == struct_def) {
            found = entry.key_ptr.*;
            break;
        }
    }
    const key = found orelse return null;
    const modules = l.r.modules orelse return null;
    const owner = modules.owner(key.tree) orelse return null;
    const dir = owner.dir;
    var decl_buf: [2]Ast.Node.Index = undefined;
    const container = key.tree.fullContainerDecl(&decl_buf, key.node) orelse return null;
    for (container.ast.members) |member| {
        var proto_buf: [1]Ast.Node.Index = undefined;
        const proto = key.tree.fullFnProto(&proto_buf, member) orelse continue;
        const name_token = proto.name_token orelse continue;
        if (!std.mem.eql(u8, key.tree.tokenSlice(name_token), method_name)) continue;
        const ret_node = proto.ast.return_type.unwrap() orelse return Error.UnsupportedType;
        const is_noreturn = isNoReturnType(key.tree, ret_node);
        const ret: ?ZType = try resolveFnReturnType(l.r, key.tree, dir, ret_node);
        var params: std.ArrayList(ZType) = .empty;
        var iter = proto.iterate(key.tree);
        while (iter.next()) |param| {
            if (param.anytype_ellipsis3 != null) return Error.UnsupportedType;
            const type_node = param.type_expr orelse return Error.UnsupportedType;
            try params.append(l.r.arena, try resolveTypeIn(l.r, key.tree, dir, type_node));
        }
        return FnSig{
            .name = try std.fmt.allocPrint(l.r.arena, "{s}.{s}", .{ struct_def.name, method_name }),
            .params = try params.toOwnedSlice(l.r.arena),
            .ret = ret,
            .is_noreturn = is_noreturn,
        };
    }
    return null;
}

/// Intern (in `func`'s own type table) the IR `bool` type. A free function, not
/// `ZType.boolean.irType(func)`, because `ZType.boolean` (a fieldless union tag accessed via
/// the type name) resolves to the TAG, not a constructed union value, and has no methods.
fn boolType(func: *Function) std.mem.Allocator.Error!Type {
    return func.types.intern(.bool);
}

fn binOpFor(tag: Ast.Node.Tag) ?ir.function.BinOp {
    return switch (tag) {
        .add, .add_wrap, .assign_add, .assign_add_wrap => .add,
        .sub, .sub_wrap, .assign_sub, .assign_sub_wrap => .sub,
        .mul, .mul_wrap, .assign_mul, .assign_mul_wrap => .mul,
        .div, .assign_div => .div,
        .mod, .assign_mod => .rem,
        .bit_and, .assign_bit_and => .bit_and,
        .bit_or, .assign_bit_or => .bit_or,
        .bit_xor, .assign_bit_xor => .bit_xor,
        .shl, .assign_shl => .shl,
        .shr, .assign_shr => .shr,
        else => null,
    };
}

fn cmpOpFor(tag: Ast.Node.Tag) ?ir.function.CmpOp {
    return switch (tag) {
        .equal_equal => .eq,
        .bang_equal => .ne,
        .less_than => .lt,
        .greater_than => .gt,
        .less_or_equal => .le,
        .greater_or_equal => .ge,
        else => null,
    };
}

/// A lowered expression's IR value together with its Zig-subset type.
const TypedValue = struct { value: Value, zty: ZType };

/// An in-scope name bound to its address (an `alloca` slot - every local, param, or `for`
/// index variable gets one, even when never reassigned; this sacrifices mem2reg-level
/// optimality at the frontend, matching `vulcan-cc`'s own precedent) and its declared type.
const Binding = struct { name: []const u8, zty: ZType, addr: Value };

/// The mutable lowering context for one function body.
const L = struct {
    r: *Ctx,
    func: *Function,
    block: Block,
    ret_zty: ?ZType,
    env: std.ArrayList(Binding),
    sret_ptr: ?Value = null,
    fn_sigs: []const FnSig,
    is_naked: bool = false,
    // This is deliberately narrow: `@cVaEnd` is the only deferred operation this subset
    // lowers. It must run on every explicit return, including a return from inside the
    // variadic reader's loop; emitting it at the `defer` site would end the list too early.
    deferred_va_ends: std.ArrayList(Value) = .empty,
    break_target: ?Block = null,
    continue_target: ?Block = null,
};

fn findBinding(l: *L, name: []const u8) ?Binding {
    var i = l.env.items.len;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(u8, l.env.items[i].name, name)) return l.env.items[i];
    }
    return null;
}

/// Allocate a fresh stack slot at the declaration site, so its address is
/// defined on every control-flow path that can use the bound name.
fn declareLocal(l: *L, name: []const u8, zty: ZType, init_value: ?Value) Error!Value {
    const irty = try zty.irType(l.func);
    const ptrt = try l.func.types.ptrGlobal();
    const addr = try l.func.appendInst(l.block, ptrt, .{ .alloca = .{ .elem = irty } });
    if (init_value) |v| try l.func.appendStore(l.block, v, addr);
    try l.env.append(l.func.allocator, .{ .name = name, .zty = zty, .addr = addr });
    return addr;
}

fn emitDeferredVaEnds(l: *L) Error!void {
    var i = l.deferred_va_ends.items.len;
    while (i > 0) {
        i -= 1;
        try l.func.appendVaEnd(l.block, l.deferred_va_ends.items[i]);
    }
}

/// `ptr + off` (a `ptr`-typed `arith add`, byte offset). A no-op (`ptr` unchanged) when
/// `off == 0`, so a first field's address is the struct's own address.
fn byteOffsetPtr(l: *L, ptr: Value, off: u64) Error!Value {
    if (off == 0) return ptr;
    const ptrt = try l.func.types.ptrGlobal();
    const i64t = try l.func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const off_v = try l.func.appendInst(l.block, i64t, .{ .iconst = @intCast(off) });
    return l.func.appendInst(l.block, ptrt, .{ .arith = .{ .op = .add, .lhs = ptr, .rhs = off_v } });
}

/// The optional's in-memory bytes are passed as one or two integer register
/// chunks. Keep sub-word loads/stores narrow: a `?u16` field occupies four
/// bytes, not the eight bytes of its enclosing register.
fn optionalWordType(l: *L, size: u64) Error!Type {
    const bits: u16 = if (size == 2) 16 else if (size == 4) 32 else 64;
    return l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = bits } });
}

fn optionalSlot(l: *L, zty: ZType) Error!Value {
    const ptrt = try l.func.types.ptrGlobal();
    return l.func.appendInst(l.block, ptrt, .{ .alloca = .{ .elem = try zty.irType(l.func) } });
}

fn optionalChunks(l: *L, zty: ZType, addr: Value, out: *[2]Value) Error![]const Value {
    const size = zty.byteSize();
    const ty = try optionalWordType(l, size);
    out[0] = try l.func.appendInst(l.block, ty, .{ .load = .{ .ptr = addr } });
    if (size == 16) {
        const hi = try byteOffsetPtr(l, addr, 8);
        out[1] = try l.func.appendInst(l.block, ty, .{ .load = .{ .ptr = hi } });
        return out[0..2];
    }
    return out[0..1];
}

fn storeOptionalChunks(l: *L, dest: Value, chunks: []const Value) Error!void {
    try l.func.appendStore(l.block, chunks[0], dest);
    if (chunks.len == 2) try l.func.appendStore(l.block, chunks[1], try byteOffsetPtr(l, dest, 8));
}

fn copyOptional(l: *L, zty: ZType, dest: Value, source: Value) Error!void {
    if (zty.optionalPayloadIsAggregate()) {
        const size = zty.byteSize();
        const chunk_bytes: u64 = if (zty.optional_value.alignment() >= 4 and size % 4 == 0) 4 else 1;
        const chunk_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = @intCast(chunk_bytes * 8) } });
        var i: u64 = 0;
        while (i < size) : (i += chunk_bytes) {
            const chunk = try l.func.appendInst(l.block, chunk_ty, .{ .load = .{ .ptr = try byteOffsetPtr(l, source, i) } });
            try l.func.appendStore(l.block, chunk, try byteOffsetPtr(l, dest, i));
        }
        return;
    }
    var chunks: [2]Value = undefined;
    const words = try optionalChunks(l, zty, source, &chunks);
    try storeOptionalChunks(l, dest, words);
}
fn optionalPresence(l: *L, optional: TypedValue) Error!Value {
    const payload = optional.zty.optional_value.*;
    const tag_addr = try byteOffsetPtr(l, optional.value, payload.byteSize());
    const byte_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const tag = try l.func.appendInst(l.block, byte_ty, .{ .load = .{ .ptr = tag_addr } });
    const zero = try l.func.appendInst(l.block, byte_ty, .{ .iconst = 0 });
    return l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .ne, .lhs = tag, .rhs = zero } });
}

fn optionalPayload(l: *L, optional: TypedValue) Error!TypedValue {
    const payload = optional.zty.optional_value.*;
    if (payload.isAggregate()) return .{ .value = optional.value, .zty = payload };
    const value = try l.func.appendInst(l.block, try payload.irType(l.func), .{ .load = .{ .ptr = optional.value } });
    return .{ .value = value, .zty = payload };
}

fn packOptional(l: *L, zty: ZType, payload: ?TypedValue) Error!TypedValue {
    const dest = try optionalSlot(l, zty);
    if (zty.optionalPayloadIsAggregate()) {
        const byte_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
        const zero = try l.func.appendInst(l.block, byte_ty, .{ .iconst = 0 });
        var i: u64 = 0;
        while (i < zty.byteSize()) : (i += 1) try l.func.appendStore(l.block, zero, try byteOffsetPtr(l, dest, i));
        if (payload) |p| {
            if (!p.zty.eql(zty.optional_value.*)) return Error.TypeMismatch;
            i = 0;
            while (i < p.zty.byteSize()) : (i += 1) {
                const byte = try l.func.appendInst(l.block, byte_ty, .{ .load = .{ .ptr = try byteOffsetPtr(l, p.value, i) } });
                try l.func.appendStore(l.block, byte, try byteOffsetPtr(l, dest, i));
            }
            const one = try l.func.appendInst(l.block, byte_ty, .{ .iconst = 1 });
            try l.func.appendStore(l.block, one, try byteOffsetPtr(l, dest, zty.optional_value.byteSize()));
        }
        return .{ .value = dest, .zty = zty };
    }
    const word_ty = try optionalWordType(l, zty.byteSize());
    const zero = try l.func.appendInst(l.block, word_ty, .{ .iconst = 0 });
    try l.func.appendStore(l.block, zero, dest);
    if (zty.byteSize() == 16) try l.func.appendStore(l.block, zero, try byteOffsetPtr(l, dest, 8));
    if (payload) |p| {
        const coerced = try convertTo(l, p, zty.optional_value.*);
        try l.func.appendStore(l.block, coerced.value, dest);
        const byte_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
        const one = try l.func.appendInst(l.block, byte_ty, .{ .iconst = 1 });
        try l.func.appendStore(l.block, one, try byteOffsetPtr(l, dest, zty.optional_value.byteSize()));
    }
    return .{ .value = dest, .zty = zty };
}

/// `base + idx * elem_size` (an element address for a runtime array index).
fn indexPtr(l: *L, base: Value, idx: Value, elem_size: u64) Error!Value {
    const ptrt = try l.func.types.ptrGlobal();
    const i64t = try l.func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const size_v = try l.func.appendInst(l.block, i64t, .{ .iconst = @intCast(elem_size) });
    const scaled = try l.func.appendInst(l.block, i64t, .{ .arith = .{ .op = .mul, .lhs = idx, .rhs = size_v } });
    return l.func.appendInst(l.block, ptrt, .{ .arith = .{ .op = .add, .lhs = base, .rhs = scaled } });
}

/// Convert an integer value to signed 64-bit, for use as an index-scaling operand. Not a
/// source-level implicit conversion (see `convertTo`): any int width/signedness converts,
/// since this is purely an internal address-arithmetic mechanism.
fn toI64(l: *L, tv: TypedValue) Error!Value {
    const i64zty = ZType{ .int = .{ .signed = true, .bits = 64 } };
    if (tv.zty.eql(i64zty)) return tv.value;
    if (tv.zty != .int) return Error.TypeMismatch;
    const irty = try i64zty.irType(l.func);
    return l.func.appendInst(l.block, irty, .{ .convert = .{ .value = tv.value } });
}

/// Convert `tv` to `target`'s representation for an assignment/return/call-argument
/// boundary, emitting an IR `convert` only when needed. Same type is a no-op. Otherwise,
/// this subset allows ONLY same-signedness integer widening or same-float-kind (a no-op
/// already handled above); anything else (narrowing, signedness change, int<->float,
/// pointer mismatch, any aggregate) is `Error.TypeMismatch` - fail closed rather than
/// silently truncating, matching this subset's "no comptime_int peer resolution" simplicity.
fn convertTo(l: *L, tv: TypedValue, target: ZType) Error!TypedValue {
    if (tv.zty.eql(target)) return tv;
    if (target == .enumeration) {
        // A named enum tag may be assigned from another enum with the same backing
        // shape (implicit literals stay nominal via `lowerEnumLiteral`), but never from
        // a plain integer without an explicit cast.
        if (tv.zty != .enumeration) return Error.TypeMismatch;
        if (!comptimeCoerces(ZType{ .int = tv.zty.enumeration.backing }, ZType{ .int = target.enumeration.backing })) return Error.TypeMismatch;
        const irty = try target.irType(l.func);
        const value = try l.func.appendInst(l.block, irty, .{ .convert = .{ .value = tv.value } });
        return .{ .value = value, .zty = target };
    }
    if (tv.zty == .enumeration and target == .int) return Error.TypeMismatch;
    switch (tv.zty) {
        .int => |si| switch (target) {
            .int => |ti| if (ti.signed != si.signed or ti.bits < si.bits) return Error.TypeMismatch,
            else => return Error.TypeMismatch,
        },
        .ptr => |source| switch (target) {
            .optional_ptr => |dest| if (source.eql(dest.*)) return .{ .value = tv.value, .zty = target },
            .slice => |dest_elem| if (source.* == .array and source.array.elem.eql(dest_elem.*)) {
                return packSlice(l, dest_elem, tv.value, @as(u64, source.array.len));
            },
            else => return Error.TypeMismatch,
        },
        .optional_ptr => |source| switch (target) {
            .ptr => |dest| if (source.eql(dest.*)) return .{ .value = tv.value, .zty = target },
            else => return Error.TypeMismatch,
        },
        else => return Error.TypeMismatch,
    }
    const irty = try target.irType(l.func);
    const v = try l.func.appendInst(l.block, irty, .{ .convert = .{ .value = tv.value } });
    return .{ .value = v, .zty = target };
}

/// Load a scalar lvalue, or decay a struct/array lvalue to its own address (never loaded as
/// a whole aggregate value - see this file's header doc).
fn loadOrDecay(l: *L, a: TypedValue) Error!TypedValue {
    if (a.zty.isAggregate()) return a;
    const irty = try a.zty.irType(l.func);
    const v = try l.func.appendInst(l.block, irty, .{ .load = .{ .ptr = a.value } });
    return .{ .value = v, .zty = a.zty };
}

/// Resolve an lvalue node to its address and pointee type: a local/param name, `expr.*`, an
/// array element (`base[idx]`, `base` must itself resolve to an `array`), or a struct field
/// (`base.field`, `base` a struct value/pointer). Any other node shape is not addressable in
/// this subset.
fn lowerAddr(l: *L, node: Ast.Node.Index) Error!TypedValue {
    const tree = l.r.tree;
    switch (tree.nodeTag(node)) {
        .grouped_expression => return lowerAddr(l, tree.nodeData(node).node_and_token[0]),
        .identifier => {
            const name = tree.tokenSlice(tree.nodeMainToken(node));
            if (findBinding(l, name)) |binding| return .{ .value = binding.addr, .zty = binding.zty };
            const global = if (l.r.globals.get(name)) |known|
                known
            else if (try ensureGlobal(l.r, name)) |reached|
                reached
            else {
                const sig = try ensureLocalFnSig(l, name) orelse {
                    if (l.r.deferred_constants.contains(name)) return Error.UnsupportedType;
                    return Error.UndeclaredIdentifier;
                };
                const ptrt = try l.func.types.ptrGlobal();
                const fn_ptr = try l.r.arena.create(ZType);
                fn_ptr.* = .opaque_type;
                return .{ .value = try l.func.appendGlobalAddr(l.block, ptrt, sig.name), .zty = .{ .ptr = fn_ptr } };
            };
            const ptrt = try l.func.types.ptrGlobal();
            return .{ .value = try l.func.appendGlobalAddr(l.block, ptrt, global.symbol), .zty = global.zty };
        },
        .deref => {
            const inner_node = tree.nodeData(node).node;
            const inner = try lowerExpr(l, inner_node, null);
            if (inner.zty != .ptr) return Error.TypeMismatch;
            if (inner.zty.ptr.* == .opaque_type) return Error.UnsupportedType;
            return .{ .value = inner.value, .zty = inner.zty.ptr.* };
        },
        .array_access => {
            const d = tree.nodeData(node).node_and_node;
            const base = try lowerExpr(l, d[0], null);
            const elem_zty: ZType = switch (base.zty) {
                .array => |a| a.elem.*,
                .slice => |p| p.*,
                else => return Error.UnsupportedNode,
            };
            const idx = try lowerExpr(l, d[1], ZType{ .int = .{ .signed = true, .bits = 64 } });
            const idx64 = try toI64(l, idx);
            const base_ptr = if (base.zty == .slice) blk: {
                const ptr_ty = try l.func.types.ptrGlobal();
                break :blk try l.func.appendInst(l.block, ptr_ty, .{ .load = .{ .ptr = base.value } });
            } else base.value;
            const addr = try indexPtr(l, base_ptr, idx64, elem_zty.byteSize());
            return .{ .value = addr, .zty = elem_zty };
        },
        .field_access => {
            const d = tree.nodeData(node).node_and_token;
            // `Enum.tag` where `Enum` is a top-level declaration in this file.
            if (resolveQualifiedName(l.r, tree, l.r.dir, d[0])) |resolved| {
                if (resolved.node) |container| {
                    if (resolveEnumTypeIn(l.r, resolved.tree, resolved.dir, container)) |zty| {
                        const entry = zty.enumeration.field(tree.tokenSlice(d[1])) orelse return Error.UnknownField;
                        return .{ .value = try l.func.appendInst(l.block, try zty.irType(l.func), .{ .iconst = std.math.cast(i64, entry.value) orelse @as(i64, @bitCast(entry.value)) }), .zty = zty };
                    }
                }
            } else |_| {}
            if (tree.nodeTag(d[0]) == .identifier) {
                const base_name = tree.tokenSlice(tree.nodeMainToken(d[0]));
                if (l.r.local_types.get(base_name)) |zty| {
                    if (zty != .enumeration) return Error.UnsupportedNode;
                    const entry = zty.enumeration.field(tree.tokenSlice(d[1])) orelse return Error.UnknownField;
                    _ = try declareLocal(l, base_name, zty, null);
                    return .{ .value = try l.func.appendInst(l.block, try zty.irType(l.func), .{ .iconst = std.math.cast(i64, entry.value) orelse @as(i64, @bitCast(entry.value)) }), .zty = zty };
                }
            }
            const qualified_base = resolveQualifiedName(l.r, tree, l.r.dir, d[0]) catch null;
            if (qualified_base) |resolved| if (resolved.node == null) {
                const field_name = tree.tokenSlice(d[1]);
                const imported_ctx = try l.r.compiler.contextFor(resolved.tree);
                const ptrt = try l.func.types.ptrGlobal();
                if (try ensureGlobal(imported_ctx, field_name)) |global| {
                    return .{ .value = try l.func.appendGlobalAddr(l.block, ptrt, global.symbol), .zty = global.zty };
                }
                const sig = (try ensureFunction(imported_ctx, field_name)) orelse return Error.UnsupportedType;
                const fn_ptr = try l.r.arena.create(ZType);
                fn_ptr.* = .opaque_type;
                return .{ .value = try l.func.appendGlobalAddr(l.block, ptrt, sig.name), .zty = .{ .ptr = fn_ptr } };
            };
            const base = try lowerExpr(l, d[0], null);
            if (base.zty == .slice) {
                const fname = tree.tokenSlice(d[1]);
                const field_zty: ZType = if (std.mem.eql(u8, fname, "ptr"))
                    ZType{ .ptr = base.zty.slice }
                else if (std.mem.eql(u8, fname, "len"))
                    ZType{ .int = .{ .signed = false, .bits = 64 } }
                else
                    return Error.UnknownField;
                const offset: u64 = if (std.mem.eql(u8, fname, "ptr")) 0 else 8;
                return .{ .value = try byteOffsetPtr(l, base.value, offset), .zty = field_zty };
            }
            const sd: *const StructDef = switch (base.zty) {
                .strct => |s| s,
                .ptr => |p| switch (p.*) {
                    .strct => |s| s,
                    else => return Error.TypeMismatch,
                },
                else => return Error.TypeMismatch,
            };
            const fname = tree.tokenSlice(d[1]);
            const field = findField(sd, fname) orelse return Error.UnknownField;
            const addr = try byteOffsetPtr(l, base.value, field.offset);
            return .{ .value = addr, .zty = field.ty };
        },
        else => return Error.UnsupportedNode,
    }
}

/// Lower a struct-literal (`.struct_init*`) directly into memory at `dest`: each field
/// initializer stores through `dest + field.offset`, recursing for a nested struct field.
/// There is no whole-struct IR value to build (see this file's header doc), so this is the
/// only way a struct literal is ever lowered.
fn lowerStructInitInto(l: *L, node: Ast.Node.Index, dest: Value, sd: *const StructDef) Error!void {
    const tree = l.r.tree;
    var buf: [2]Ast.Node.Index = undefined;
    const si = tree.fullStructInit(&buf, node) orelse return Error.UnsupportedNode;
    for (si.ast.fields) |fv| {
        const eq_tok = tree.firstToken(fv) - 1;
        const name_tok = eq_tok - 1;
        const fname = tree.tokenSlice(name_tok);
        const field = findField(sd, fname) orelse return Error.UnknownField;
        const addr = try byteOffsetPtr(l, dest, field.offset);
        if (field.ty == .strct) {
            try lowerStructInitInto(l, fv, addr, field.ty.strct);
        } else if (field.ty == .array) {
            try lowerArrayInitInto(l, fv, addr, field.ty.array);
        } else if (field.ty == .optional_value) {
            const value = try lowerExpr(l, fv, field.ty);
            try copyOptional(l, field.ty, addr, value.value);
        } else {
            const val = try lowerExpr(l, fv, field.ty);
            const conv = try convertTo(l, val, field.ty);
            try l.func.appendStore(l.block, conv.value, addr);
        }
    }
}

fn lowerArrayInitInto(l: *L, node: Ast.Node.Index, dest: Value, az: ZType.ZArray) Error!void {
    const tree = l.r.tree;
    var buf: [2]Ast.Node.Index = undefined;
    const ai = tree.fullArrayInit(&buf, node) orelse return Error.UnsupportedNode;
    if (ai.ast.elements.len != az.len) return Error.TypeMismatch;
    for (ai.ast.elements, 0..) |elem_node, i| {
        const offset = i * az.elem.byteSize();
        const addr = try byteOffsetPtr(l, dest, offset);
        if (az.elem.* == .strct) {
            try lowerStructInitInto(l, elem_node, addr, az.elem.strct);
        } else if (az.elem.* == .array) {
            try lowerArrayInitInto(l, elem_node, addr, az.elem.array);
        } else if (az.elem.* == .optional_value) {
            const val = try lowerExpr(l, elem_node, az.elem.*);
            try copyOptional(l, az.elem.*, addr, val.value);
        } else {
            const val = try lowerExpr(l, elem_node, az.elem.*);
            const conv = try convertTo(l, val, az.elem.*);
            try l.func.appendStore(l.block, conv.value, addr);
        }
    }
}

fn packSlice(l: *L, elem_zty: *const ZType, ptr_val: Value, len_val: anytype) Error!TypedValue {
    const slice_zty = ZType{ .slice = elem_zty };
    const ptr_ty = try l.func.types.ptrGlobal();
    const len_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const slot = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = try slice_zty.irType(l.func) } });
    const len: Value = if (@TypeOf(len_val) == Value)
        len_val
    else
        try l.func.appendInst(l.block, len_ty, .{ .iconst = @as(i64, @intCast(len_val)) });
    const ptr_conv = try l.func.appendInst(l.block, ptr_ty, .{ .convert = .{ .value = ptr_val } });
    try l.func.appendStore(l.block, ptr_conv, slot);
    try l.func.appendStore(l.block, len, try byteOffsetPtr(l, slot, 8));
    return .{ .value = slot, .zty = slice_zty };
}

fn lowerWriterSlice(l: *L, node: Ast.Node.Index) Error!TypedValue {
    const tree = l.r.tree;
    if (tree.nodeTag(node) != .string_literal) {
        const value = try lowerExpr(l, node, null);
        if (value.zty == .slice) {
            if (value.zty.slice.* != .int or value.zty.slice.int.bits != 8)
                return Error.UnsupportedType;
            return value;
        }
        if (value.zty == .ptr and value.zty.ptr.* == .array and
            value.zty.ptr.array.elem.* == .int and value.zty.ptr.array.elem.int.bits == 8)
        {
            return packSlice(l, value.zty.ptr.array.elem, value.value, value.zty.ptr.array.len);
        }
        return Error.UnsupportedType;
    }

    const bytes = std.zig.string_literal.parseAlloc(l.r.arena, tree.tokenSlice(tree.nodeMainToken(node))) catch return Error.UnsupportedNode;
    const byte_zty = ZType{ .int = .{ .signed = false, .bits = 8 } };
    const byte_ty = try byte_zty.irType(l.func);
    const elem = try l.r.arena.create(ZType);
    elem.* = byte_zty;
    const array_zty = ZType{ .array = .{ .len = bytes.len, .elem = elem } };
    const ptr_ty = try l.func.types.ptrGlobal();
    const array = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = try array_zty.irType(l.func) } });
    for (bytes, 0..) |byte, i| {
        const value = try l.func.appendInst(l.block, byte_ty, .{ .iconst = byte });
        try l.func.appendStore(l.block, value, try byteOffsetPtr(l, array, i));
    }
    const slice_zty = ZType{ .slice = elem };
    const slice = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = try slice_zty.irType(l.func) } });
    const data_ptr = try l.func.appendInst(l.block, ptr_ty, .{ .convert = .{ .value = array } });
    const usize_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const length = try l.func.appendInst(l.block, usize_ty, .{ .iconst = @intCast(bytes.len) });
    try l.func.appendStore(l.block, data_ptr, slice);
    try l.func.appendStore(l.block, length, try byteOffsetPtr(l, slice, 8));
    return .{ .value = slice, .zty = slice_zty };
}
fn lowerWriterDrain(l: *L, writer: TypedValue, data_ptr: Value, data_len: Value) Error!TypedValue {
    const writer_struct = switch (writer.zty) {
        .ptr => |p| if (p.* == .strct) p.strct else return Error.TypeMismatch,
        else => return Error.TypeMismatch,
    };
    const vtable_field = findField(writer_struct, "vtable") orelse return Error.UnknownField;
    const vtable_addr = try byteOffsetPtr(l, writer.value, vtable_field.offset);
    const ptr_ty = try l.func.types.ptrGlobal();
    const vtable = try l.func.appendInst(l.block, ptr_ty, .{ .load = .{ .ptr = vtable_addr } });
    const vtable_struct = switch (vtable_field.ty) {
        .ptr => |p| if (p.* == .strct) p.strct else return Error.TypeMismatch,
        else => return Error.TypeMismatch,
    };
    const drain_field = findField(vtable_struct, "drain") orelse return Error.UnknownField;
    const drain_addr = try byteOffsetPtr(l, vtable, drain_field.offset);
    const drain = try l.func.appendInst(l.block, ptr_ty, .{ .load = .{ .ptr = drain_addr } });
    const u64_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const descriptor_type = try l.func.types.intern(.{ .array = .{ .len = 2, .elem = u64_ty } });
    const descriptors = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = descriptor_type } });
    try l.func.appendStore(l.block, data_ptr, descriptors);
    try l.func.appendStore(l.block, data_len, try byteOffsetPtr(l, descriptors, 8));
    const one = try l.func.appendInst(l.block, u64_ty, .{ .iconst = 1 });
    const args = [_]Value{ writer.value, descriptors, one, one };
    const result_type = try l.func.types.intern(.{ .array = .{ .len = 2, .elem = u64_ty } });
    const result = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = result_type } });
    try l.func.appendCallIndirectStructRet(l.block, drain, &args, result, &.{ .{ .offset = 0 }, .{ .offset = 8 } });
    const error_void = ZType{ .error_void = {} };
    const err_type = try error_void.irType(l.func);
    const tag_word = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = result } });
    const err_tag = try l.func.appendInst(l.block, err_type, .{ .convert = .{ .value = tag_word } });
    return .{ .value = err_tag, .zty = error_void };
}

fn lowerWriterWriteAll(l: *L, writer_node: Ast.Node.Index, bytes_node: Ast.Node.Index) Error!TypedValue {
    const writer = try lowerExpr(l, writer_node, null);
    const tree = l.r.tree;
    if (tree.nodeTag(bytes_node) == .string_literal) {
        const literal = std.zig.string_literal.parseAlloc(l.r.arena, tree.tokenSlice(tree.nodeMainToken(bytes_node))) catch return Error.UnsupportedNode;
        if (literal.len == 0) {
            const error_void = ZType{ .error_void = {} };
            return .{ .value = try l.func.appendInst(l.block, try error_void.irType(l.func), .{ .iconst = 0 }), .zty = error_void };
        }
    }
    const bytes = try lowerWriterSlice(l, bytes_node);
    const ptr_ty = try l.func.types.ptrGlobal();
    const u64_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const data_ptr = try l.func.appendInst(l.block, ptr_ty, .{ .load = .{ .ptr = bytes.value } });
    const data_len = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = try byteOffsetPtr(l, bytes.value, 8) } });
    return lowerWriterDrain(l, writer, data_ptr, data_len);
}

fn lowerWriterPrintDrain(l: *L, writer: TypedValue, data_ptr: Value, data_len: Value, status: Value) Error!void {
    const err_ty = try (ZType{ .error_void = {} }).irType(l.func);
    const zero = try l.func.appendInst(l.block, err_ty, .{ .iconst = 0 });
    const current = try l.func.appendInst(l.block, err_ty, .{ .load = .{ .ptr = status } });
    const ok = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .eq, .lhs = current, .rhs = zero } });
    const emit = try l.func.appendBlock();
    const merge = try l.func.appendBlock();
    try l.func.appendIf(l.block, ok, .{ .target = emit }, .{ .target = merge });
    l.block = emit;
    const result = try lowerWriterDrain(l, writer, data_ptr, data_len);
    try l.func.appendStore(l.block, result.value, status);
    try l.func.setJump(l.block, merge, &.{});
    l.block = merge;
}

fn lowerWriterPrintLiteral(l: *L, writer: TypedValue, bytes: []const u8, status: Value) Error!void {
    if (bytes.len == 0) return;
    const byte_zty = ZType{ .int = .{ .signed = false, .bits = 8 } };
    const byte_ty = try byte_zty.irType(l.func);
    const elem = try l.r.arena.create(ZType);
    elem.* = byte_zty;
    const array_zty = ZType{ .array = .{ .len = bytes.len, .elem = elem } };
    const ptr_ty = try l.func.types.ptrGlobal();
    const array = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = try array_zty.irType(l.func) } });
    for (bytes, 0..) |byte, i| {
        const value = try l.func.appendInst(l.block, byte_ty, .{ .iconst = byte });
        try l.func.appendStore(l.block, value, try byteOffsetPtr(l, array, i));
    }
    const len_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const len = try l.func.appendInst(l.block, len_ty, .{ .iconst = @intCast(bytes.len) });
    try lowerWriterPrintDrain(l, writer, array, len, status);
}

fn lowerWriterFormatInteger(
    l: *L,
    writer: TypedValue,
    value: TypedValue,
    base: u64,
    uppercase: bool,
    width: u64,
    status: Value,
) Error!void {
    if (value.zty != .int) return Error.UnsupportedType;
    const u64_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const index_zty = ZType{ .int = .{ .signed = false, .bits = 64 } };
    const byte_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const ptr_ty = try l.func.types.ptrGlobal();
    const buffer_ty = try l.func.types.intern(.{ .array = .{ .len = 128, .elem = byte_ty } });
    const buffer = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = buffer_ty } });
    const idx_addr = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = u64_ty } });
    const rem_addr = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = u64_ty } });
    const idx_start = try l.func.appendInst(l.block, u64_ty, .{ .iconst = 127 });
    const signed_decimal = value.zty.int.signed and base == 10;
    const negative: ?Value = if (signed_decimal) blk: {
        const signed_ty = try value.zty.irType(l.func);
        const signed_zero = try l.func.appendInst(l.block, signed_ty, .{ .iconst = 0 });
        break :blk try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .lt, .lhs = value.value, .rhs = signed_zero } });
    } else null;
    const zero = try l.func.appendInst(l.block, u64_ty, .{ .iconst = 0 });
    const converted = if (value.zty.int.bits == 64 and !value.zty.int.signed)
        value.value
    else
        try l.func.appendInst(l.block, u64_ty, .{ .convert = .{ .value = value.value } });
    const raw = if (negative) |is_negative| blk: {
        const magnitude = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = converted } });
        break :blk try l.func.appendInst(l.block, u64_ty, .{ .select = .{ .cond = is_negative, .then = magnitude, .@"else" = converted } });
    } else converted;
    try l.func.appendStore(l.block, idx_start, idx_addr);
    try l.func.appendStore(l.block, raw, rem_addr);
    const is_zero = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .eq, .lhs = raw, .rhs = zero } });
    const zero_block = try l.func.appendBlock();
    const loop_block = try l.func.appendBlock();
    const cond_block = try l.func.appendBlock();
    const merge = try l.func.appendBlock();
    try l.func.appendIf(l.block, is_zero, .{ .target = zero_block }, .{ .target = loop_block });
    l.block = zero_block;
    const char_zero = try l.func.appendInst(l.block, byte_ty, .{ .iconst = '0' });
    const zero_idx = try l.func.appendInst(l.block, u64_ty, .{ .iconst = 127 });
    const zero_idx_i64 = try toI64(l, .{ .value = zero_idx, .zty = index_zty });
    try l.func.appendStore(l.block, char_zero, try indexPtr(l, buffer, zero_idx_i64, 1));
    const idx_126 = try l.func.appendInst(l.block, u64_ty, .{ .iconst = 126 });
    try l.func.appendStore(l.block, idx_126, idx_addr);
    try l.func.setJump(l.block, merge, &.{});

    l.block = loop_block;
    try l.func.setJump(l.block, cond_block, &.{});
    l.block = cond_block;
    const remaining = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = rem_addr } });
    const done = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .eq, .lhs = remaining, .rhs = zero } });
    const body = try l.func.appendBlock();
    try l.func.appendIf(l.block, done, .{ .target = merge }, .{ .target = body });
    l.block = body;
    const base_value = try l.func.appendInst(l.block, u64_ty, .{ .iconst = @intCast(base) });
    const remaining_body = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = rem_addr } });
    const digit = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .rem, .lhs = remaining_body, .rhs = base_value } });
    const offset = try l.func.appendInst(l.block, u64_ty, .{ .iconst = if (uppercase) 'A' - 10 else 'a' - 10 });
    const threshold = try l.func.appendInst(l.block, u64_ty, .{ .iconst = 10 });
    const under_ten = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .lt, .lhs = digit, .rhs = threshold } });
    const numeric = try l.func.appendInst(l.block, byte_ty, .{ .convert = .{ .value = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .add, .lhs = digit, .rhs = try l.func.appendInst(l.block, u64_ty, .{ .iconst = '0' }) } }) } });
    const alpha = try l.func.appendInst(l.block, byte_ty, .{ .convert = .{ .value = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .add, .lhs = digit, .rhs = offset } }) } });
    const char = try l.func.appendInst(l.block, byte_ty, .{ .select = .{ .cond = under_ten, .then = numeric, .@"else" = alpha } });
    const idx = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = idx_addr } });
    const idx_i64 = try toI64(l, .{ .value = idx, .zty = index_zty });
    try l.func.appendStore(l.block, char, try indexPtr(l, buffer, idx_i64, 1));
    const one = try l.func.appendInst(l.block, u64_ty, .{ .iconst = 1 });
    const prev_idx = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .sub, .lhs = idx, .rhs = one } });
    try l.func.appendStore(l.block, prev_idx, idx_addr);
    const next = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .div, .lhs = remaining_body, .rhs = base_value } });
    try l.func.appendStore(l.block, next, rem_addr);
    try l.func.setJump(l.block, cond_block, &.{});
    l.block = merge;
    if (width != 0) {
        const idx_for_width = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = idx_addr } });
        const digit_count = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .sub, .lhs = idx_start, .rhs = idx_for_width } });
        const one_width = try l.func.appendInst(l.block, u64_ty, .{ .iconst = 1 });
        const signed_count = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .add, .lhs = digit_count, .rhs = one_width } });
        const padded_count = if (negative) |is_negative|
            try l.func.appendInst(l.block, u64_ty, .{ .select = .{ .cond = is_negative, .then = signed_count, .@"else" = digit_count } })
        else
            digit_count;
        const width_value = try l.func.appendInst(l.block, u64_ty, .{ .iconst = @intCast(width) });
        const pad_needed = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .lt, .lhs = padded_count, .rhs = width_value } });
        const pad_entry = try l.func.appendBlock();
        const pad_cond = try l.func.appendBlock();
        const pad_body = try l.func.appendBlock();
        const pad_done = try l.func.appendBlock();
        try l.func.appendIf(l.block, pad_needed, .{ .target = pad_entry }, .{ .target = pad_done });
        l.block = pad_entry;
        const one_pad = try l.func.appendInst(l.block, u64_ty, .{ .iconst = 1 });
        const pad_addr = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = u64_ty } });
        const pad_count = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .sub, .lhs = width_value, .rhs = padded_count } });
        try l.func.appendStore(l.block, pad_count, pad_addr);
        try l.func.setJump(l.block, pad_cond, &.{});
        l.block = pad_cond;
        const current_pad = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = pad_addr } });
        const no_padding = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .eq, .lhs = current_pad, .rhs = zero } });
        try l.func.appendIf(l.block, no_padding, .{ .target = pad_done }, .{ .target = pad_body });
        l.block = pad_body;
        const pad_index = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = idx_addr } });
        const pad_index_i64 = try toI64(l, .{ .value = pad_index, .zty = index_zty });
        const pad_byte = try l.func.appendInst(l.block, byte_ty, .{ .iconst = '0' });
        try l.func.appendStore(l.block, pad_byte, try indexPtr(l, buffer, pad_index_i64, 1));
        const pad_index_next = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .sub, .lhs = pad_index, .rhs = one_pad } });
        try l.func.appendStore(l.block, pad_index_next, idx_addr);
        const pad_remaining = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = pad_addr } });
        const pad_remaining_next = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .sub, .lhs = pad_remaining, .rhs = one_pad } });
        try l.func.appendStore(l.block, pad_remaining_next, pad_addr);
        try l.func.setJump(l.block, pad_cond, &.{});
        l.block = pad_done;
    }
    if (negative) |is_negative| {
        const sign = try l.func.appendBlock();
        const after_sign = try l.func.appendBlock();
        try l.func.appendIf(l.block, is_negative, .{ .target = sign }, .{ .target = after_sign });
        l.block = sign;
        const sign_index = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = idx_addr } });
        const sign_index_i64 = try toI64(l, .{ .value = sign_index, .zty = index_zty });
        const minus = try l.func.appendInst(l.block, byte_ty, .{ .iconst = '-' });
        try l.func.appendStore(l.block, minus, try indexPtr(l, buffer, sign_index_i64, 1));
        const one_sign = try l.func.appendInst(l.block, u64_ty, .{ .iconst = 1 });
        const prev_sign_index = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .sub, .lhs = sign_index, .rhs = one_sign } });
        try l.func.appendStore(l.block, prev_sign_index, idx_addr);
        try l.func.setJump(l.block, after_sign, &.{});
        l.block = after_sign;
    }
    const one_after = try l.func.appendInst(l.block, u64_ty, .{ .iconst = 1 });
    const idx_final = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = idx_addr } });
    const start = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .add, .lhs = idx_final, .rhs = one_after } });
    const length = try l.func.appendInst(l.block, u64_ty, .{ .arith = .{ .op = .sub, .lhs = idx_start, .rhs = idx_final } });
    const start_i64 = try toI64(l, .{ .value = start, .zty = index_zty });
    const ptr = try indexPtr(l, buffer, start_i64, 1);
    try lowerWriterPrintDrain(l, writer, ptr, length, status);
}

fn lowerWriterPrint(l: *L, writer_node: Ast.Node.Index, fmt_node: Ast.Node.Index, tuple_node: Ast.Node.Index) Error!TypedValue {
    const tree = l.r.tree;
    if (tree.nodeTag(fmt_node) != .string_literal) return Error.UnsupportedNode;
    const fmt = std.zig.string_literal.parseAlloc(l.r.arena, tree.tokenSlice(tree.nodeMainToken(fmt_node))) catch return Error.UnsupportedNode;
    var tuple_buf: [2]Ast.Node.Index = undefined;
    const tuple_elements: []const Ast.Node.Index = switch (tree.nodeTag(tuple_node)) {
        .array_init_one, .array_init_one_comma, .array_init_dot_two, .array_init_dot_two_comma, .array_init_dot, .array_init_dot_comma, .array_init, .array_init_comma => (tree.fullArrayInit(&tuple_buf, tuple_node) orelse return Error.UnsupportedNode).ast.elements,
        .struct_init_one, .struct_init_one_comma, .struct_init_dot_two, .struct_init_dot_two_comma, .struct_init_dot, .struct_init_dot_comma, .struct_init, .struct_init_comma => (tree.fullStructInit(&tuple_buf, tuple_node) orelse return Error.UnsupportedNode).ast.fields,
        else => return Error.UnsupportedNode,
    };

    const writer = try lowerExpr(l, writer_node, null);
    const error_void = ZType{ .error_void = {} };
    const err_ty = try error_void.irType(l.func);
    const ptr_ty = try l.func.types.ptrGlobal();
    const status = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = err_ty } });
    const zero = try l.func.appendInst(l.block, err_ty, .{ .iconst = 0 });
    try l.func.appendStore(l.block, zero, status);

    var arg_index: usize = 0;
    var literal_start: usize = 0;
    var i: usize = 0;
    while (i < fmt.len) : (i += 1) {
        if (fmt[i] != '{') continue;
        if (i + 1 < fmt.len and fmt[i + 1] == '{') {
            try lowerWriterPrintLiteral(l, writer, fmt[literal_start..i], status);
            try lowerWriterPrintLiteral(l, writer, "{", status);
            i += 1;
            literal_start = i + 1;
            continue;
        }
        const close = std.mem.indexOfScalarPos(u8, fmt, i + 1, '}') orelse return Error.UnsupportedNode;
        try lowerWriterPrintLiteral(l, writer, fmt[literal_start..i], status);
        const raw_spec = fmt[i + 1 .. close];
        const colon = std.mem.indexOfScalar(u8, raw_spec, ':');
        const spec = if (colon) |sep| raw_spec[0..sep] else raw_spec;
        const width: u64 = if (colon) |sep| blk: {
            const layout = raw_spec[sep + 1 ..];
            if (layout.len < 3 or layout[0] != '0' or layout[1] != '>') return Error.UnsupportedNode;
            break :blk std.fmt.parseInt(u64, layout[2..], 10) catch return Error.UnsupportedNode;
        } else 0;
        if (width > 127) return Error.UnsupportedNode;
        if (arg_index >= tuple_elements.len) return Error.CallArityMismatch;
        const arg_node = tuple_elements[arg_index];
        arg_index += 1;
        if (std.mem.eql(u8, spec, "s")) {
            const string = try lowerWriterSlice(l, arg_node);
            const ptr = try l.func.appendInst(l.block, ptr_ty, .{ .load = .{ .ptr = string.value } });
            const len_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
            const len = try l.func.appendInst(l.block, len_ty, .{ .load = .{ .ptr = try byteOffsetPtr(l, string.value, 8) } });
            try lowerWriterPrintDrain(l, writer, ptr, len, status);
        } else if (std.mem.eql(u8, spec, "d") or std.mem.eql(u8, spec, "") or std.mem.eql(u8, spec, "x") or std.mem.eql(u8, spec, "X") or std.mem.eql(u8, spec, "?x")) {
            const value = try lowerExpr(l, arg_node, null);
            if (std.mem.eql(u8, spec, "?x")) {
                const present = switch (value.zty) {
                    .optional_ptr => blk: {
                        const zero_ptr = try l.func.appendInst(l.block, try value.zty.irType(l.func), .{ .iconst = 0 });
                        break :blk try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .ne, .lhs = value.value, .rhs = zero_ptr } });
                    },
                    .optional_value => try optionalPresence(l, value),
                    else => return Error.TypeMismatch,
                };
                const yes = try l.func.appendBlock();
                const no = try l.func.appendBlock();
                const merge = try l.func.appendBlock();
                try l.func.appendIf(l.block, present, .{ .target = yes }, .{ .target = no });
                l.block = yes;
                const payload = if (value.zty == .optional_value)
                    try optionalPayload(l, value)
                else
                    TypedValue{ .value = value.value, .zty = value.zty.optional_ptr.* };
                try lowerWriterFormatInteger(l, writer, payload, 16, false, 0, status);
                try l.func.setJump(l.block, merge, &.{});
                l.block = no;
                try lowerWriterPrintLiteral(l, writer, "null", status);
                try l.func.setJump(l.block, merge, &.{});
                l.block = merge;
            } else if (value.zty == .boolean) {
                const yes = try l.func.appendBlock();
                const no = try l.func.appendBlock();
                const merge = try l.func.appendBlock();
                try l.func.appendIf(l.block, value.value, .{ .target = yes }, .{ .target = no });
                l.block = yes;
                try lowerWriterPrintLiteral(l, writer, "true", status);
                try l.func.setJump(l.block, merge, &.{});
                l.block = no;
                try lowerWriterPrintLiteral(l, writer, "false", status);
                try l.func.setJump(l.block, merge, &.{});
                l.block = merge;
            } else {
                const base: u64 = if (std.mem.eql(u8, spec, "x") or std.mem.eql(u8, spec, "X") or std.mem.eql(u8, spec, "?x")) 16 else 10;
                try lowerWriterFormatInteger(l, writer, value, base, std.mem.eql(u8, spec, "X"), width, status);
            }
        } else {
            return Error.UnsupportedNode;
        }
        i = close;
        literal_start = close + 1;
    }
    try lowerWriterPrintLiteral(l, writer, fmt[literal_start..], status);
    if (arg_index != tuple_elements.len) return Error.CallArityMismatch;
    const result = try l.func.appendInst(l.block, err_ty, .{ .load = .{ .ptr = status } });
    return .{ .value = result, .zty = error_void };
}

/// Resolve a call's callee-by-name and lower its arguments, converting each to the callee's
/// declared parameter type. Returns `null` for a void-returning callee (only valid as a bare
/// expression statement, see `lowerStmt`'s default arm).
fn lowerArrayEql(l: *L, params: []const Ast.Node.Index) Error!TypedValue {
    if (params.len != 3) return Error.CallArityMismatch;
    const elem_ty = try resolveType(l.r, params[0]);
    if (elem_ty != .int and elem_ty != .boolean) return Error.UnsupportedType;
    const lhs = try lowerExpr(l, params[1], null);
    const rhs = try lowerExpr(l, params[2], null);
    if (lhs.zty != .ptr or rhs.zty != .ptr or lhs.zty.ptr.* != .array or rhs.zty.ptr.* != .array)
        return Error.UnsupportedType;
    const lhs_array = lhs.zty.ptr.array;
    const rhs_array = rhs.zty.ptr.array;
    if (!lhs_array.elem.eql(elem_ty) or !rhs_array.elem.eql(elem_ty)) return Error.TypeMismatch;
    if (lhs_array.len != rhs_array.len)
        return .{ .value = try l.func.appendInst(l.block, try boolType(l.func), .{ .iconst = 0 }), .zty = .boolean };

    const index_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const zero = try l.func.appendInst(l.block, index_ty, .{ .iconst = 0 });
    const ptr_ty = try l.func.types.ptrGlobal();
    const index_addr = try l.func.appendInst(l.block, ptr_ty, .{ .alloca = .{ .elem = index_ty } });
    try l.func.appendStore(l.block, zero, index_addr);
    const cond_block = try l.func.appendBlock();
    const body_block = try l.func.appendBlock();
    const equal_block = try l.func.appendBlock();
    const unequal_block = try l.func.appendBlock();
    const merge_block = try l.func.appendBlock();
    const bool_ty = try boolType(l.func);
    const result = try l.func.appendBlockParam(merge_block, bool_ty);
    try l.func.setJump(l.block, cond_block, &.{});

    l.block = cond_block;
    const index = try l.func.appendInst(l.block, index_ty, .{ .load = .{ .ptr = index_addr } });
    const limit = try l.func.appendInst(l.block, index_ty, .{ .iconst = @intCast(lhs_array.len) });
    const in_range = try l.func.appendInst(l.block, bool_ty, .{ .icmp = .{ .op = .lt, .lhs = index, .rhs = limit } });
    try l.func.appendIf(l.block, in_range, .{ .target = body_block }, .{ .target = equal_block });

    l.block = body_block;
    const lhs_addr = try indexPtr(l, lhs.value, try toI64(l, .{ .value = index, .zty = .{ .int = .{ .signed = false, .bits = 64 } } }), elem_ty.byteSize());
    const rhs_addr = try indexPtr(l, rhs.value, try toI64(l, .{ .value = index, .zty = .{ .int = .{ .signed = false, .bits = 64 } } }), elem_ty.byteSize());
    const lhs_value = try l.func.appendInst(l.block, try elem_ty.irType(l.func), .{ .load = .{ .ptr = lhs_addr } });
    const rhs_value = try l.func.appendInst(l.block, try elem_ty.irType(l.func), .{ .load = .{ .ptr = rhs_addr } });
    const matches = try l.func.appendInst(l.block, bool_ty, .{ .icmp = .{ .op = .eq, .lhs = lhs_value, .rhs = rhs_value } });
    const next_block = try l.func.appendBlock();
    try l.func.appendIf(l.block, matches, .{ .target = next_block }, .{ .target = unequal_block });

    l.block = unequal_block;
    const no = try l.func.appendInst(l.block, bool_ty, .{ .iconst = 0 });
    try l.func.setJump(l.block, merge_block, &.{no});

    l.block = next_block;
    const current = try l.func.appendInst(l.block, index_ty, .{ .load = .{ .ptr = index_addr } });
    const one = try l.func.appendInst(l.block, index_ty, .{ .iconst = 1 });
    const following = try l.func.appendInst(l.block, index_ty, .{ .arith = .{ .op = .add, .lhs = current, .rhs = one } });
    try l.func.appendStore(l.block, following, index_addr);
    try l.func.setJump(l.block, cond_block, &.{});

    l.block = equal_block;
    const yes = try l.func.appendInst(l.block, bool_ty, .{ .iconst = 1 });
    try l.func.setJump(l.block, merge_block, &.{yes});
    l.block = merge_block;
    return .{ .value = result, .zty = .boolean };
}

fn lowerCall(l: *L, node: Ast.Node.Index) Error!?TypedValue {
    const tree = l.r.tree;
    var buf: [1]Ast.Node.Index = undefined;
    const c = tree.fullCall(&buf, node) orelse return Error.UnsupportedNode;
    var fname: []const u8 = undefined;
    var sig: FnSig = undefined;
    var method_receiver: ?TypedValue = null;
    if (tree.nodeTag(c.ast.fn_expr) == .field_access) {
        const method = tree.nodeData(c.ast.fn_expr).node_and_token;
        const method_name = tree.tokenSlice(method[1]);
        if (std.mem.eql(u8, method_name, "eql") and c.ast.params.len == 3) {
            const base_span = tree.nodeToSpan(method[0]);
            if (std.mem.eql(u8, tree.source[base_span.start..base_span.end], "std.mem"))
                return try lowerArrayEql(l, c.ast.params);
        }
        if (std.mem.eql(u8, method_name, "doNotOptimizeAway") and c.ast.params.len == 1) {
            const base_span = tree.nodeToSpan(method[0]);
            if (std.mem.eql(u8, tree.source[base_span.start..base_span.end], "std.mem")) {
                _ = try lowerExpr(l, c.ast.params[0], null);
                return null;
            }
        }
        if (std.mem.eql(u8, method_name, "emergencyWriteAll") and c.ast.params.len == 1) {
            const bytes = try lowerWriterSlice(l, c.ast.params[0]);
            const ptr_ty = try l.func.types.ptrGlobal();
            const u64_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
            const data_ptr = try l.func.appendInst(l.block, ptr_ty, .{ .load = .{ .ptr = bytes.value } });
            const len = try l.func.appendInst(l.block, u64_ty, .{ .load = .{ .ptr = try byteOffsetPtr(l, bytes.value, 8) } });
            try l.func.appendVoidCall(l.block, "emergencyWriteAll", &.{ data_ptr, len });
            return null;
        }
        if (std.mem.eql(u8, method_name, "writeAll") and c.ast.params.len == 1)
            return try lowerWriterWriteAll(l, method[0], c.ast.params[0]);
        if (std.mem.eql(u8, method_name, "print") and c.ast.params.len == 2)
            return try lowerWriterPrint(l, method[0], c.ast.params[0], c.ast.params[1]);
        if (std.mem.eql(u8, method_name, "emergencyWriteHex") and c.ast.params.len == 1) {
            const value = try lowerExpr(l, c.ast.params[0], null);
            const u64_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
            const converted = if (value.zty == .int and value.zty.int.bits == 64 and !value.zty.int.signed)
                value.value
            else
                try l.func.appendInst(l.block, u64_ty, .{ .convert = .{ .value = value.value } });
            try l.func.appendVoidCall(l.block, "emergencyWriteHex", &.{converted});
            return null;
        }
        if ((std.mem.eql(u8, method_name, "halt") or std.mem.eql(u8, method_name, "reset") or std.mem.eql(u8, method_name, "powerOff")) and c.ast.params.len == 0) {
            try l.func.appendVoidCall(l.block, method_name, &.{});
            return null;
        }
        const foreign_sig = try importedFnSig(l, c.ast.fn_expr);
        if (foreign_sig) |resolved_sig| {
            fname = resolved_sig.name;
            sig = resolved_sig;
        } else if (try enumFieldOrDefault(l, c.ast.fn_expr)) |constant| {
            return constant;
        } else {
            const receiver = try lowerExpr(l, method[0], null);
            const method_sig = (try foreignMethodSig(l, receiver.zty, method_name)) orelse return Error.UnsupportedNode;
            fname = method_sig.name;
            sig = method_sig;
            method_receiver = receiver;
        }
    } else {
        fname = tree.tokenSlice(tree.nodeMainToken(c.ast.fn_expr));
        if (l.r.deferred_constants.contains(fname)) return Error.UnsupportedType;
        sig = try ensureLocalFnSig(l, fname) orelse return Error.UndeclaredIdentifier;
        fname = sig.name;
    }
    const params = if (method_receiver != null) sig.params[1..] else sig.params;
    if (method_receiver != null and sig.params.len == 0) return Error.CallArityMismatch;
    if (c.ast.params.len != params.len) return Error.CallArityMismatch;

    var args: std.ArrayList(Value) = .empty;
    defer args.deinit(l.r.arena);
    if (method_receiver) |receiver| {
        const receiver_param = sig.params[0];
        if (receiver_param.isAggregate() and receiver_param != .optional_value and receiver_param != .slice) {
            if (l.r.target != .aarch64 or !receiver.zty.eql(receiver_param)) return Error.UnsupportedType;
            var offset: u64 = 0;
            while (offset < receiver_param.byteSize()) {
                const bytes: u64 = if (receiver_param.byteSize() - offset >= 8 and offset % 8 == 0) 8 else if (receiver_param.byteSize() - offset >= 4 and offset % 4 == 0) 4 else if (receiver_param.byteSize() - offset >= 2 and offset % 2 == 0) 2 else 1;
                const chunk_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = @intCast(bytes * 8) } });
                const chunk = try l.func.appendInst(l.block, chunk_ty, .{ .load = .{ .ptr = try byteOffsetPtr(l, receiver.value, offset) } });
                try args.append(l.r.arena, chunk);
                offset += bytes;
            }
        } else {
            const coerced = try convertTo(l, receiver, receiver_param);
            try args.append(l.r.arena, coerced.value);
        }
    }
    for (c.ast.params, params) |an, pty| {
        const av = try lowerExpr(l, an, pty);
        if (pty == .optional_value) {
            var chunks: [2]Value = undefined;
            const words = try optionalChunks(l, pty, av.value, &chunks);
            try args.appendSlice(l.r.arena, words);
        } else if (pty == .slice) {
            const ptr_ty = try l.func.types.ptrGlobal();
            const len_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
            const ptr = try l.func.appendInst(l.block, ptr_ty, .{ .load = .{ .ptr = av.value } });
            const len = try l.func.appendInst(l.block, len_ty, .{ .load = .{ .ptr = try byteOffsetPtr(l, av.value, 8) } });
            try args.append(l.r.arena, ptr);
            try args.append(l.r.arena, len);
        } else {
            const cv = try convertTo(l, av, pty);
            try args.append(l.r.arena, cv.value);
        }
    }

    if (sig.ret) |rz| if (rz == .optional_value or rz == .slice) {
        const slot = try optionalSlot(l, rz);
        if (rz.optionalPayloadIsAggregate() and rz.byteSize() > 16) {
            if (l.r.target != .aarch64) return Error.UnsupportedType;
            try args.insert(l.r.arena, 0, slot);
            try l.func.appendCallSret(l.block, fname, args.items);
        } else if (rz.byteSize() == 16) {
            try l.func.appendCallStructRet(l.block, fname, args.items, slot, &.{
                .{ .offset = 0 }, .{ .offset = 8 },
            });
        } else {
            const word_ty = try optionalWordType(l, rz.byteSize());
            const word = try l.func.appendCall(l.block, word_ty, fname, args.items);
            try l.func.appendStore(l.block, word, slot);
        }
        return TypedValue{ .value = slot, .zty = rz };
    };
    if (sig.ret) |rz| if (rz == .error_union) {
        const slot = try optionalSlot(l, rz);
        if (rz.byteSize() > 16) {
            if (l.r.target != .aarch64) return Error.UnsupportedType;
            try args.insert(l.r.arena, 0, slot);
            try l.func.appendCallSret(l.block, fname, args.items);
        } else {
            try l.func.appendCallStructRet(l.block, fname, args.items, slot, &.{
                .{ .offset = 0 }, .{ .offset = 8 },
            });
        }
        return TypedValue{ .value = slot, .zty = rz };
    };
    if (sig.ret) |rz| {
        const irty = try rz.irType(l.func);
        const v = try l.func.appendCall(l.block, irty, fname, args.items);
        return TypedValue{ .value = v, .zty = rz };
    }
    try l.func.appendVoidCall(l.block, fname, args.items);
    return null;
}
fn lowerAggregateCatch(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!TypedValue {
    const tree = l.r.tree;
    const d = tree.nodeData(node).node_and_node;
    const result = try lowerExpr(l, d[0], null);
    if (result.zty != .error_union) return Error.UnsupportedType;
    const payload_type = result.zty.error_union.payload.*;
    const optional_type = expected orelse return Error.UnsupportedType;
    if (optional_type != .optional_value or !optional_type.optional_value.eql(payload_type))
        return Error.TypeMismatch;

    const tag_zty = ZType{ .error_void = {} };
    const tag = try l.func.appendInst(l.block, try tag_zty.irType(l.func), .{ .load = .{
        .ptr = try byteOffsetPtr(l, result.value, result.zty.errorTagOffset()),
    } });
    const zero = try l.func.appendInst(l.block, try tag_zty.irType(l.func), .{ .iconst = 0 });
    const failed = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .ne, .lhs = tag, .rhs = zero } });
    const handler = try l.func.appendBlock();
    const success = try l.func.appendBlock();
    try l.func.appendIf(l.block, failed, .{ .target = handler }, .{ .target = success });

    l.block = handler;
    const env_len = l.env.items.len;
    const catch_token = tree.nodeMainToken(node);
    if (tree.tokenTag(catch_token + 1) == .pipe) {
        const name_token = catch_token + 2;
        if (tree.tokenTag(name_token) != .identifier) return Error.UnsupportedNode;
        _ = try declareLocal(l, tree.tokenSlice(name_token), .{ .error_set = result.zty.error_union.errors }, tag);
    }
    if (!try lowerStmt(l, d[1])) return Error.UnsupportedNode;
    l.env.items.len = env_len;

    l.block = success;
    const value = try packOptional(l, optional_type, .{ .value = result.value, .zty = payload_type });
    return .{ .value = value.value, .zty = optional_type };
}

fn lowerCatchExpr(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!TypedValue {
    const tree = l.r.tree;
    const d = tree.nodeData(node).node_and_node;
    if (expected) |exp| {
        if (exp == .optional_value) {
            return lowerAggregateCatch(l, node, expected);
        }
    }
    const result = try lowerExpr(l, d[0], null);
    if (result.zty != .error_union) return Error.UnsupportedType;
    const payload_type = result.zty.error_union.payload.*;

    const tag_zty = ZType{ .error_void = {} };
    const tag = try l.func.appendInst(l.block, try tag_zty.irType(l.func), .{ .load = .{
        .ptr = try byteOffsetPtr(l, result.value, result.zty.errorTagOffset()),
    } });
    const zero = try l.func.appendInst(l.block, try tag_zty.irType(l.func), .{ .iconst = 0 });
    const failed = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .ne, .lhs = tag, .rhs = zero } });
    const handler = try l.func.appendBlock();
    const success = try l.func.appendBlock();
    try l.func.appendIf(l.block, failed, .{ .target = handler }, .{ .target = success });

    if (payload_type.isAggregate()) {
        const slot = try optionalSlot(l, payload_type);
        const merge = try l.func.appendBlock();

        l.block = success;
        try copyAggregateBytes(l, slot, result.value, payload_type.byteSize());
        try l.func.setJump(l.block, merge, &.{});

        l.block = handler;
        const env_len = l.env.items.len;
        const catch_token = tree.nodeMainToken(node);
        if (tree.tokenTag(catch_token + 1) == .pipe) {
            const name_token = catch_token + 2;
            if (tree.tokenTag(name_token) != .identifier) return Error.UnsupportedNode;
            _ = try declareLocal(l, tree.tokenSlice(name_token), .{ .error_set = result.zty.error_union.errors }, tag);
        }
        if (tree.nodeTag(d[1]) == .@"return") {
            _ = try lowerStmt(l, d[1]);
        } else {
            const fallback = try lowerExpr(l, d[1], payload_type);
            if (!fallback.zty.eql(payload_type)) return Error.TypeMismatch;
            try copyAggregateBytes(l, slot, fallback.value, payload_type.byteSize());
            try l.func.setJump(l.block, merge, &.{});
        }
        l.env.items.len = env_len;

        l.block = merge;
        return .{ .value = slot, .zty = payload_type };
    }

    const res_irty = try payload_type.irType(l.func);
    const merge = try l.func.appendBlock();
    const phi = try l.func.appendBlockParam(merge, res_irty);

    l.block = success;
    const ok_val = try l.func.appendInst(l.block, res_irty, .{ .load = .{ .ptr = result.value } });
    try l.func.setJump(l.block, merge, &.{ok_val});

    l.block = handler;
    const env_len = l.env.items.len;
    const catch_token = tree.nodeMainToken(node);
    if (tree.tokenTag(catch_token + 1) == .pipe) {
        const name_token = catch_token + 2;
        if (tree.tokenTag(name_token) != .identifier) return Error.UnsupportedNode;
        _ = try declareLocal(l, tree.tokenSlice(name_token), .{ .error_set = result.zty.error_union.errors }, tag);
    }
    if (tree.nodeTag(d[1]) == .@"return") {
        _ = try lowerStmt(l, d[1]);
    } else {
        const fallback = try lowerExpr(l, d[1], payload_type);
        const conv = try convertTo(l, fallback, payload_type);
        try l.func.setJump(l.block, merge, &.{conv.value});
    }
    l.env.items.len = env_len;

    l.block = merge;
    return .{ .value = phi, .zty = payload_type };
}

fn lowerVoidCatch(l: *L, node: Ast.Node.Index) Error!bool {
    const tree = l.r.tree;
    const d = tree.nodeData(node).node_and_node;
    const result = try lowerExpr(l, d[0], null);
    if (result.zty != .error_void) return Error.UnsupportedType;
    const result_ty = try result.zty.irType(l.func);
    const zero = try l.func.appendInst(l.block, result_ty, .{ .iconst = 0 });
    const failed = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .ne, .lhs = result.value, .rhs = zero } });
    const handler = try l.func.appendBlock();
    const merge = try l.func.appendBlock();
    try l.func.appendIf(l.block, failed, .{ .target = handler }, .{ .target = merge });
    l.block = handler;
    const handler_terminates = try lowerStmt(l, d[1]);
    if (!handler_terminates) try l.func.setJump(l.block, merge, &.{});
    l.block = merge;
    return false;
}

/// Return the final field name from an enum literal or member-access expression.
fn atomicEnumName(tree: *const Ast, node: Ast.Node.Index) ?[]const u8 {
    const span = tree.nodeToSpan(node);
    const text = tree.source[span.start..span.end];
    const dot = std.mem.lastIndexOfScalar(u8, text, '.') orelse return null;
    return std.mem.trim(u8, text[dot + 1 ..], " \t\r\n)");
}

fn lowerAtomicOrdering(tree: *const Ast, node: Ast.Node.Index) Error!ir.function.AtomicOrdering {
    const name = atomicEnumName(tree, node) orelse return Error.UnsupportedNode;
    if (std.mem.eql(u8, name, "unordered") or std.mem.eql(u8, name, "monotonic")) return .relaxed;
    if (std.mem.eql(u8, name, "acquire")) return .acquire;
    if (std.mem.eql(u8, name, "release")) return .release;
    if (std.mem.eql(u8, name, "acq_rel")) return .acq_rel;
    if (std.mem.eql(u8, name, "seq_cst")) return .seq_cst;
    return Error.UnsupportedNode;
}

fn lowerBuiltinCall(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!?TypedValue {
    const tree = l.r.tree;
    var buf: [2]Ast.Node.Index = undefined;
    const args = builtinArgs(tree, node, &buf);
    const name = builtinName(tree, node);
    if (std.mem.eql(u8, name, "@memcpy")) {
        if (args.len != 2) return Error.CallArityMismatch;
        const dest = try lowerExpr(l, args[0], null);
        const source = try lowerExpr(l, args[1], null);
        if (dest.zty != .ptr or source.zty != .ptr) return Error.TypeMismatch;
        const dest_type = dest.zty.ptr.*;
        const source_type = source.zty.ptr.*;
        if (dest_type != .array or source_type != .array or !dest_type.eql(source_type))
            return Error.UnsupportedType;
        try copyAggregateBytes(l, dest.value, source.value, dest_type.byteSize());
        return null;
    }
    if (std.mem.eql(u8, name, "@atomicStore")) {
        if (args.len != 4) return Error.CallArityMismatch;
        const zty = try resolveType(l.r, args[0]);
        const ptr = try lowerExpr(l, args[1], null);
        if (ptr.zty != .ptr or !ptr.zty.ptr.eql(zty)) return Error.TypeMismatch;
        const value = try lowerExpr(l, args[2], zty);
        if (!value.zty.eql(zty)) return Error.TypeMismatch;
        const atomic_value = if (zty == .boolean) blk: {
            const byte_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
            break :blk try l.func.appendInst(l.block, byte_ty, .{ .convert = .{ .value = value.value } });
        } else value.value;
        const ordering = try lowerAtomicOrdering(tree, args[3]);
        try l.func.appendAtomicRmwStmt(l.block, .{
            .op = .exchange,
            .ptr = ptr.value,
            .value = atomic_value,
            .ordering = ordering,
            .scope = .device,
        });
        return null;
    }
    if (std.mem.eql(u8, name, "@atomicRmw")) {
        if (args.len != 5) return Error.CallArityMismatch;
        const zty = try resolveType(l.r, args[0]);
        const ptr = try lowerExpr(l, args[1], null);
        if (ptr.zty != .ptr or !ptr.zty.ptr.eql(zty)) return Error.TypeMismatch;
        const op_name = atomicEnumName(tree, args[2]) orelse return Error.UnsupportedNode;
        const op: ir.function.AtomicOp = if (std.mem.eql(u8, op_name, "Add"))
            .add
        else if (std.mem.eql(u8, op_name, "Xchg"))
            .exchange
        else
            return Error.UnsupportedNode;
        const value = try lowerExpr(l, args[3], zty);
        if (!value.zty.eql(zty)) return Error.TypeMismatch;
        const result = try l.func.appendAtomicRmw(l.block, .{
            .op = op,
            .ptr = ptr.value,
            .value = value.value,
            .ordering = try lowerAtomicOrdering(tree, args[4]),
            .scope = .device,
        });
        return .{ .value = result, .zty = zty };
    }
    if (std.mem.eql(u8, name, "@cVaEnd")) {
        if (args.len != 1) return Error.CallArityMismatch;
        const list = try lowerVaListAddr(l, args[0]);
        try l.func.appendVaEnd(l.block, list);
        return null;
    }
    if (std.mem.eql(u8, name, "@cVaArg")) {
        if (args.len != 2) return Error.CallArityMismatch;
        const list = try lowerVaListAddr(l, args[0]);
        const zty = try resolveType(l.r, args[1]);
        if (zty.isAggregate() or zty == .va_list) return Error.UnsupportedType;
        const irty = try zty.irType(l.func);
        const v = try l.func.appendVaArg(l.block, list, irty);
        return TypedValue{ .value = v, .zty = zty };
    }
    if (std.mem.eql(u8, name, "@intFromPtr")) {
        if (args.len != 1) return Error.CallArityMismatch;
        const ptr = try lowerExpr(l, args[0], null);
        if (ptr.zty != .ptr and ptr.zty != .optional_ptr) return Error.TypeMismatch;
        const int_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
        const value = try l.func.appendInst(l.block, int_ty, .{ .convert = .{ .value = ptr.value } });
        return TypedValue{ .value = value, .zty = .{ .int = .{ .signed = false, .bits = 64 } } };
    }
    if (std.mem.eql(u8, name, "@intCast") or std.mem.eql(u8, name, "@truncate")) {
        if (args.len != 1) return Error.CallArityMismatch;
        const target = expected orelse return Error.TypeMismatch;
        if (target != .int) return Error.TypeMismatch;
        const source = try lowerExpr(l, args[0], null);
        if (source.zty != .int) return Error.TypeMismatch;
        const value = try l.func.appendInst(l.block, try target.irType(l.func), .{ .convert = .{ .value = source.value } });
        return .{ .value = value, .zty = target };
    }
    if (std.mem.eql(u8, name, "@min") or std.mem.eql(u8, name, "@max")) {
        if (args.len != 2) return Error.CallArityMismatch;
        const lhs = try lowerExpr(l, args[0], null);
        if (lhs.zty != .int) return Error.UnsupportedType;
        const rhs = try lowerExpr(l, args[1], lhs.zty);
        if (!rhs.zty.eql(lhs.zty)) return Error.TypeMismatch;
        const op: ir.function.CmpOp = if (std.mem.eql(u8, name, "@min")) .lt else .gt;
        const cond = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = op, .lhs = lhs.value, .rhs = rhs.value } });
        const value = try l.func.appendInst(l.block, try lhs.zty.irType(l.func), .{ .select = .{ .cond = cond, .then = lhs.value, .@"else" = rhs.value } });
        return .{ .value = value, .zty = lhs.zty };
    }
    if (std.mem.eql(u8, name, "@intFromEnum")) {
        if (args.len != 1) return Error.CallArityMismatch;
        const value = try lowerExpr(l, args[0], null);
        if (value.zty != .enumeration) return Error.TypeMismatch;
        const int = ZType{ .int = .{ .signed = value.zty.enumeration.backing.signed, .bits = value.zty.enumeration.backing.bits } };
        const converted = try l.func.appendInst(l.block, try int.irType(l.func), .{ .convert = .{ .value = value.value } });
        return .{ .value = converted, .zty = int };
    }
    if (std.mem.eql(u8, name, "@enumFromInt")) {
        if (args.len != 1) return Error.CallArityMismatch;
        const target = expected orelse return Error.TypeMismatch;
        if (target != .enumeration) return Error.TypeMismatch;
        const source = try lowerExpr(l, args[0], null);
        if (source.zty != .int) return Error.TypeMismatch;
        const converted = try l.func.appendInst(l.block, try target.irType(l.func), .{ .convert = .{ .value = source.value } });
        return .{ .value = converted, .zty = target };
    }
    // `@fromBackingInt`/`@backingInt` (toolchain spellings of the conversions above) and
    // `@intCast` are typed by their expected type where one exists.
    if (std.mem.eql(u8, name, "@fromBackingInt") and expected != null and expected.? == .enumeration) {
        if (args.len != 1) return Error.CallArityMismatch;
        const target = expected.?;
        const source = try lowerExpr(l, args[0], null);
        if (source.zty != .int) return Error.TypeMismatch;
        const converted = try l.func.appendInst(l.block, try target.irType(l.func), .{ .convert = .{ .value = source.value } });
        return .{ .value = converted, .zty = target };
    }
    if (std.mem.eql(u8, name, "@as")) {
        if (args.len != 2) return Error.CallArityMismatch;
        const target = try resolveType(l.r, args[0]);
        const value = try lowerExpr(l, args[1], target);
        return try convertTo(l, value, target);
    }
    if (std.mem.eql(u8, name, "@backingInt")) {
        if (args.len != 1) return Error.CallArityMismatch;
        const value = try lowerExpr(l, args[0], null);
        if (value.zty != .enumeration) return Error.TypeMismatch;
        const int = ZType{ .int = .{ .signed = value.zty.enumeration.backing.signed, .bits = value.zty.enumeration.backing.bits } };
        const converted = try l.func.appendInst(l.block, try int.irType(l.func), .{ .convert = .{ .value = value.value } });
        return .{ .value = converted, .zty = int };
    }
    if (std.mem.eql(u8, name, "@errorName")) {
        if (args.len != 1) return Error.CallArityMismatch;
        const value = try lowerExpr(l, args[0], null);
        if (value.zty != .error_set) return Error.UnsupportedType;
        return try lowerErrorName(l, value, value.zty.error_set.*);
    }
    return Error.UnsupportedNode;
}

/// `&va`'s address for a `va_list` local, the only shape `@cVaEnd`/`@cVaArg`'s first argument
/// takes: lower it as an rvalue (an `.address_of` node yields a `ptr` wrapping `.va_list`) and
/// unwrap to the raw address `appendVaEnd`/`appendVaArg` want.
fn lowerVaListAddr(l: *L, node: Ast.Node.Index) Error!Value {
    const v = try lowerExpr(l, node, null);
    if (v.zty != .ptr or v.zty.ptr.* != .va_list) return Error.TypeMismatch;
    return v.value;
}

fn lowerCharLiteral(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!TypedValue {
    const text = l.r.tree.tokenSlice(l.r.tree.nodeMainToken(node));
    const res = std.zig.string_literal.parseCharLiteral(text);
    const cp: u32 = switch (res) {
        .success => |c| c,
        .failure => return Error.UnsupportedNode,
    };
    const zty = expected orelse ZType{ .int = .{ .signed = false, .bits = 8 } };
    if (zty != .int) return Error.TypeMismatch;
    if (zty.int.bits < 32) {
        const limit = (@as(u64, 1) << @intCast(zty.int.bits));
        if (cp >= limit) return Error.IntegerOverflow;
    }
    const irty = try zty.irType(l.func);
    return .{ .value = try l.func.appendInst(l.block, irty, .{ .iconst = cp }), .zty = zty };
}

fn lowerNumberLiteral(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!TypedValue {
    const text = l.r.tree.tokenSlice(l.r.tree.nodeMainToken(node));
    const res = std.zig.number_literal.parseNumberLiteral(text);
    switch (res) {
        .int => |v| {
            const zty = expected orelse ZType{ .int = .{ .signed = true, .bits = 64 } };
            switch (zty) {
                .int => |iz| {
                    if (iz.bits < 64) {
                        const limit_bits: u6 = @intCast(if (iz.signed) iz.bits - 1 else iz.bits);
                        const max: u64 = (@as(u64, 1) << limit_bits) - 1;
                        if (v > max) return Error.IntegerOverflow;
                    }
                    const irty = try zty.irType(l.func);
                    const val: i64 = @bitCast(v);
                    return .{ .value = try l.func.appendInst(l.block, irty, .{ .iconst = val }), .zty = zty };
                },
                .float => {
                    const irty = try zty.irType(l.func);
                    const fv: f64 = @floatFromInt(v);
                    return .{ .value = try l.func.appendInst(l.block, irty, .{ .fconst = fv }), .zty = zty };
                },
                else => return Error.TypeMismatch,
            }
        },
        .float => {
            var clean: [128]u8 = undefined;
            var n: usize = 0;
            for (text) |c| {
                if (c != '_') {
                    clean[n] = c;
                    n += 1;
                }
            }
            const fv = std.fmt.parseFloat(f64, clean[0..n]) catch return Error.UnsupportedType;
            const zty = expected orelse ZType{ .float = .f64 };
            if (zty != .float) return Error.TypeMismatch;
            const irty = try zty.irType(l.func);
            return .{ .value = try l.func.appendInst(l.block, irty, .{ .fconst = fv }), .zty = zty };
        },
        .big_int, .failure => return Error.UnsupportedType,
    }
}

/// Lower `optional orelse fallback` for nullable pointers. A null `@cVaArg` often exits a
/// loop, so the bare-`break` form branches directly to the loop's existing merge block rather
/// than synthesizing a value that the source program never observes.
fn callReturnType(l: *L, node: Ast.Node.Index) Error!?ZType {
    var buf: [1]Ast.Node.Index = undefined;
    const call = l.r.tree.fullCall(&buf, node) orelse return null;
    if (l.r.tree.nodeTag(call.ast.fn_expr) == .identifier) {
        const name = l.r.tree.tokenSlice(l.r.tree.nodeMainToken(call.ast.fn_expr));
        const sig = try ensureLocalFnSig(l, name) orelse return null;
        return sig.ret;
    }
    if (l.r.tree.nodeTag(call.ast.fn_expr) == .field_access) {
        const sig = try importedFnSig(l, call.ast.fn_expr) orelse return null;
        return sig.ret;
    }
    return null;
}
fn copyAggregateBytes(l: *L, dest: Value, source: Value, size: u64) Error!void {
    const byte_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    var offset: u64 = 0;
    while (offset < size) : (offset += 1) {
        const byte = try l.func.appendInst(l.block, byte_ty, .{ .load = .{ .ptr = try byteOffsetPtr(l, source, offset) } });
        try l.func.appendStore(l.block, byte, try byteOffsetPtr(l, dest, offset));
    }
}
fn emitReturnErrorUnion(l: *L, ret_zty: ZType, tag: Value) Error!void {
    if (ret_zty == .error_void) {
        l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.one(tag) });
        return;
    }
    if (ret_zty != .error_union) return Error.TypeMismatch;
    if (l.sret_ptr) |dest| {
        const byte_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
        const zero_byte = try l.func.appendInst(l.block, byte_ty, .{ .iconst = 0 });
        var offset: u64 = 0;
        while (offset < ret_zty.byteSize()) : (offset += 1)
            try l.func.appendStore(l.block, zero_byte, try byteOffsetPtr(l, dest, offset));
        try l.func.appendStore(l.block, tag, try byteOffsetPtr(l, dest, ret_zty.errorTagOffset()));
        l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.one(dest) });
    } else {
        const slot = try optionalSlot(l, ret_zty);
        const byte_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
        const zero_byte = try l.func.appendInst(l.block, byte_ty, .{ .iconst = 0 });
        var offset: u64 = 0;
        while (offset < ret_zty.byteSize()) : (offset += 1)
            try l.func.appendStore(l.block, zero_byte, try byteOffsetPtr(l, slot, offset));
        try l.func.appendStore(l.block, tag, try byteOffsetPtr(l, slot, ret_zty.errorTagOffset()));
        var chunks: [2]Value = undefined;
        const words = try optionalChunks(l, ret_zty, slot, &chunks);
        l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.many(words) });
    }
}

fn lowerTryExpr(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!TypedValue {
    const tree = l.r.tree;
    const child_node = tree.nodeData(node).node;
    const ret_zty = l.ret_zty orelse return Error.TypeMismatch;
    const result = try lowerExpr(l, child_node, null);

    if (result.zty == .error_void) {
        const zero = try l.func.appendInst(l.block, try result.zty.irType(l.func), .{ .iconst = 0 });
        const failed = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .ne, .lhs = result.value, .rhs = zero } });
        const ret_block = try l.func.appendBlock();
        const next_block = try l.func.appendBlock();
        try l.func.appendIf(l.block, failed, .{ .target = ret_block }, .{ .target = next_block });

        l.block = ret_block;
        try emitDeferredVaEnds(l);
        try emitReturnErrorUnion(l, ret_zty, result.value);

        l.block = next_block;
        return result;
    }

    if (result.zty != .error_union) return Error.TypeMismatch;
    const payload_type = result.zty.error_union.payload.*;

    const tag_zty = ZType{ .error_void = {} };
    const tag = try l.func.appendInst(l.block, try tag_zty.irType(l.func), .{ .load = .{
        .ptr = try byteOffsetPtr(l, result.value, result.zty.errorTagOffset()),
    } });
    const zero = try l.func.appendInst(l.block, try tag_zty.irType(l.func), .{ .iconst = 0 });
    const failed = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .ne, .lhs = tag, .rhs = zero } });
    const ret_block = try l.func.appendBlock();
    const next_block = try l.func.appendBlock();
    try l.func.appendIf(l.block, failed, .{ .target = ret_block }, .{ .target = next_block });

    l.block = ret_block;
    try emitDeferredVaEnds(l);
    try emitReturnErrorUnion(l, ret_zty, tag);

    l.block = next_block;
    if (payload_type.isAggregate()) {
        const slot = try optionalSlot(l, payload_type);
        try copyAggregateBytes(l, slot, result.value, payload_type.byteSize());
        return .{ .value = slot, .zty = payload_type };
    }
    const val_irty = try payload_type.irType(l.func);
    const val = try l.func.appendInst(l.block, val_irty, .{ .load = .{ .ptr = result.value } });
    const tv = TypedValue{ .value = val, .zty = payload_type };
    if (expected) |exp| return convertTo(l, tv, exp);
    return tv;
}

fn lowerOrelse(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!TypedValue {
    const tree = l.r.tree;
    const d = tree.nodeData(node).node_and_node;
    const optional = try lowerExpr(l, d[0], null);
    const fallback_ret = try callReturnType(l, d[1]);
    if (optional.zty.optionalPayloadIsAggregate()) if (expected) |zty| if (zty.eql(optional.zty) and fallback_ret != null and fallback_ret.?.eql(zty)) {
        const result = try optionalSlot(l, zty);
        const present = try optionalPresence(l, optional);
        const present_block = try l.func.appendBlock();
        const fallback_block = try l.func.appendBlock();
        const merge_block = try l.func.appendBlock();
        try l.func.appendIf(l.block, present, .{ .target = present_block }, .{ .target = fallback_block });
        l.block = present_block;
        try copyOptional(l, zty, result, optional.value);
        try l.func.setJump(l.block, merge_block, &.{});
        l.block = fallback_block;
        const fallback = try lowerExpr(l, d[1], zty);
        if (!fallback.zty.eql(zty)) return Error.TypeMismatch;
        try copyOptional(l, zty, result, fallback.value);
        try l.func.setJump(l.block, merge_block, &.{});
        l.block = merge_block;
        return .{ .value = result, .zty = zty };
    };
    if (optional.zty.optionalPayloadIsAggregate()) {
        const payload_type = optional.zty.optional_value.*;
        const result = try optionalSlot(l, payload_type);
        const present = try optionalPresence(l, optional);
        const present_block = try l.func.appendBlock();
        const fallback_block = try l.func.appendBlock();
        const merge_block = try l.func.appendBlock();
        try l.func.appendIf(l.block, present, .{ .target = present_block }, .{ .target = fallback_block });
        l.block = present_block;
        try copyAggregateBytes(l, result, optional.value, payload_type.byteSize());
        try l.func.setJump(l.block, merge_block, &.{});
        l.block = fallback_block;
        if (tree.nodeTag(d[1]) == .@"return") {
            _ = try lowerStmt(l, d[1]);
            l.block = merge_block;
            return .{ .value = result, .zty = payload_type };
        }
        const fallback = try lowerExpr(l, d[1], payload_type);
        if (!fallback.zty.eql(payload_type)) return Error.TypeMismatch;
        try copyAggregateBytes(l, result, fallback.value, payload_type.byteSize());
        try l.func.setJump(l.block, merge_block, &.{});
        l.block = merge_block;
        return .{ .value = result, .zty = payload_type };
    }
    const payload_type: ZType = switch (optional.zty) {
        .optional_ptr => |p| .{ .ptr = p },
        .optional_value => |p| p.*,
        else => return Error.TypeMismatch,
    };
    if (optional.zty.optionalPayloadIsAggregate()) return Error.UnsupportedType;
    const result_type = try payload_type.irType(l.func);
    const present = if (optional.zty == .optional_value)
        try optionalPresence(l, optional)
    else blk: {
        const zero = try l.func.appendInst(l.block, result_type, .{ .iconst = 0 });
        break :blk try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .ne, .lhs = optional.value, .rhs = zero } });
    };
    if (tree.nodeTag(d[1]) == .@"break") {
        const target = l.break_target orelse return Error.UnsupportedNode;
        const present_block = try l.func.appendBlock();
        try l.func.appendIf(l.block, present, .{ .target = present_block }, .{ .target = target });
        l.block = present_block;
        return if (optional.zty == .optional_value) optionalPayload(l, optional) else .{ .value = optional.value, .zty = payload_type };
    }

    const present_block = try l.func.appendBlock();
    const fallback_block = try l.func.appendBlock();
    const merge_block = try l.func.appendBlock();
    const result = try l.func.appendBlockParam(merge_block, result_type);
    try l.func.appendIf(l.block, present, .{ .target = present_block }, .{ .target = fallback_block });
    l.block = present_block;
    const payload = if (optional.zty == .optional_value) try optionalPayload(l, optional) else TypedValue{ .value = optional.value, .zty = payload_type };
    try l.func.setJump(l.block, merge_block, &.{payload.value});
    l.block = fallback_block;
    if (tree.nodeTag(d[1]) == .@"return") {
        _ = try lowerStmt(l, d[1]);
        l.block = merge_block;
        return .{ .value = result, .zty = payload_type };
    }
    const fallback = try lowerExpr(l, d[1], payload_type);
    const converted = try convertTo(l, fallback, payload_type);
    try l.func.setJump(l.block, merge_block, &.{converted.value});
    l.block = merge_block;
    return .{ .value = result, .zty = payload_type };
}

/// Lower an rvalue expression. `expected`, when present, resolves an untyped literal's width
/// (the only "peer type resolution" this subset implements - see the number-literal arm).
fn importedIntegerConstant(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!?TypedValue {
    const tree = l.r.tree;
    if (tree.nodeTag(node) != .field_access) return null;
    const access = tree.nodeData(node).node_and_token;
    const resolved = resolveQualifiedName(l.r, tree, l.r.dir, access[0]) catch return null;
    if (resolved.node != null) return null;
    const field_name = tree.tokenSlice(access[1]);
    const global = findRootGlobalType(resolved.tree, field_name) orelse return null;
    if (!global.is_const) return null;
    const init_node = global.init_node orelse return null;
    const value = evalConstInt(l.r, resolved.tree, init_node, &.{}, 0) orelse return null;
    const zty = if (expected) |ty|
        if (ty == .int) ty else return null
    else if (global.type_node) |type_node| blk: {
        const ty = try resolveTypeIn(l.r, resolved.tree, resolved.dir, type_node);
        if (ty != .int) return null;
        break :blk ty;
    } else ZType{ .int = .{ .signed = true, .bits = 64 } };
    if (zty.int.signed) {
        if (zty.int.bits < 64) {
            const limit: i64 = @as(i64, 1) << @intCast(zty.int.bits - 1);
            if (value < -limit or value >= limit) return Error.TypeMismatch;
        }
    } else {
        if (value < 0) return Error.TypeMismatch;
        if (zty.int.bits < 63 and value >= (@as(i64, 1) << @intCast(zty.int.bits))) return Error.TypeMismatch;
    }
    return .{ .value = try l.func.appendInst(l.block, try zty.irType(l.func), .{ .iconst = value }), .zty = zty };
}
fn lowerSliceExpr(l: *L, node: Ast.Node.Index) Error!TypedValue {
    const tree = l.r.tree;
    const slice_info: Ast.full.Slice = switch (tree.nodeTag(node)) {
        .slice_open => tree.sliceOpen(node),
        .slice => tree.slice(node),
        .slice_sentinel => tree.sliceSentinel(node),
        else => return Error.UnsupportedNode,
    };

    const base = try lowerExpr(l, slice_info.ast.sliced, null);
    const elem_zty: *const ZType = switch (base.zty) {
        .slice => |s| s,
        .array => |a| a.elem,
        .ptr => |p| switch (p.*) {
            .array => |a| a.elem,
            else => p,
        },
        else => return Error.UnsupportedNode,
    };

    const usize_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const ptr_ty = try l.func.types.ptrGlobal();

    // Determine the base pointer and total length (if known/slice)
    const base_ptr: Value = if (base.zty == .slice)
        try l.func.appendInst(l.block, ptr_ty, .{ .load = .{ .ptr = base.value } })
    else
        base.value;

    const start_val = try lowerExpr(l, slice_info.ast.start, ZType{ .int = .{ .signed = false, .bits = 64 } });
    const start_i64 = try toI64(l, start_val);
    const start_usize = if (start_val.zty.int.bits == 64 and !start_val.zty.int.signed)
        start_val.value
    else
        try l.func.appendInst(l.block, usize_ty, .{ .convert = .{ .value = start_val.value } });

    const sub_ptr = try indexPtr(l, base_ptr, start_i64, elem_zty.byteSize());

    const sub_len: Value = if (slice_info.ast.end.unwrap()) |end_node| blk: {
        const end_val = try lowerExpr(l, end_node, ZType{ .int = .{ .signed = false, .bits = 64 } });
        const end_usize = if (end_val.zty.int.bits == 64 and !end_val.zty.int.signed)
            end_val.value
        else
            try l.func.appendInst(l.block, usize_ty, .{ .convert = .{ .value = end_val.value } });
        break :blk try l.func.appendInst(l.block, usize_ty, .{ .arith = .{ .op = .sub, .lhs = end_usize, .rhs = start_usize } });
    } else blk: {
        // [start..] open slice: length is total_len - start
        const total_len: Value = switch (base.zty) {
            .slice => try l.func.appendInst(l.block, usize_ty, .{ .load = .{ .ptr = try byteOffsetPtr(l, base.value, 8) } }),
            .array => |a| try l.func.appendInst(l.block, usize_ty, .{ .iconst = @intCast(a.len) }),
            .ptr => |p| switch (p.*) {
                .array => |a| try l.func.appendInst(l.block, usize_ty, .{ .iconst = @intCast(a.len) }),
                else => return Error.UnsupportedNode,
            },
            else => return Error.UnsupportedNode,
        };
        break :blk try l.func.appendInst(l.block, usize_ty, .{ .arith = .{ .op = .sub, .lhs = total_len, .rhs = start_usize } });
    };

    return packSlice(l, elem_zty, sub_ptr, sub_len);
}

fn lowerStringLiteral(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!TypedValue {
    const tree = l.r.tree;
    const slice_type = if (expected) |ty| ty else blk: {
        const elem = try l.r.arena.create(ZType);
        elem.* = .{ .int = .{ .signed = false, .bits = 8 } };
        break :blk ZType{ .slice = elem };
    };
    if (slice_type != .slice or slice_type.slice.* != .int or
        slice_type.slice.int.signed or slice_type.slice.int.bits != 8)
        return Error.UnsupportedType;
    const text = std.zig.string_literal.parseAlloc(l.r.arena, tree.tokenSlice(tree.nodeMainToken(node))) catch return Error.UnsupportedNode;
    const name = try std.fmt.allocPrint(l.r.arena, ".vulcan.literal.{d}", .{l.r.compiler.next_literal_id});
    l.r.compiler.next_literal_id += 1;
    const bytes = try l.r.arena.alloc(u8, text.len + 1);
    @memcpy(bytes[0..text.len], text);
    bytes[text.len] = 0;
    try l.r.literal_data.append(l.r.arena, .{ .name = name, .bytes = bytes, .kind = .rodata, .size = bytes.len });

    const slot = try optionalSlot(l, slice_type);
    const ptr_ty = try l.func.types.ptrGlobal();
    const data_ptr = try l.func.appendGlobalAddr(l.block, ptr_ty, name);
    const len_ty = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const len = try l.func.appendInst(l.block, len_ty, .{ .iconst = @intCast(text.len) });
    try l.func.appendStore(l.block, data_ptr, slot);
    try l.func.appendStore(l.block, len, try byteOffsetPtr(l, slot, 8));
    return .{ .value = slot, .zty = slice_type };
}
fn appendLiteralSlice(l: *L, text: []const u8) Error!TypedValue {
    const name = try std.fmt.allocPrint(l.r.arena, ".vulcan.literal.{d}", .{l.r.compiler.next_literal_id});
    l.r.compiler.next_literal_id += 1;
    const bytes = try l.r.arena.alloc(u8, text.len + 1);
    @memcpy(bytes[0..text.len], text);
    bytes[text.len] = 0;
    try l.r.literal_data.append(l.r.arena, .{ .name = name, .bytes = bytes, .kind = .rodata, .size = bytes.len });
    const byte_type = try l.r.arena.create(ZType);
    byte_type.* = .{ .int = .{ .signed = false, .bits = 8 } };
    const slice_type = ZType{ .slice = byte_type };
    const slot = try optionalSlot(l, slice_type);
    const pointer_type = try l.func.types.ptrGlobal();
    const pointer = try l.func.appendGlobalAddr(l.block, pointer_type, name);
    const usize_type = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const length = try l.func.appendInst(l.block, usize_type, .{ .iconst = @intCast(text.len) });
    try l.func.appendStore(l.block, pointer, slot);
    try l.func.appendStore(l.block, length, try byteOffsetPtr(l, slot, 8));
    return .{ .value = slot, .zty = slice_type };
}

fn lowerErrorName(l: *L, tag: TypedValue, errors: ZErrorSet) Error!TypedValue {
    const slice_elem = try l.r.arena.create(ZType);
    slice_elem.* = .{ .int = .{ .signed = false, .bits = 8 } };
    const slice_type = ZType{ .slice = slice_elem };
    const slot = try optionalSlot(l, slice_type);
    const ptr_type = try l.func.types.ptrGlobal();
    const len_type = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const tag_type = try tag.zty.irType(l.func);
    const merge = try l.func.appendBlock();
    for (errors.names, 0..) |error_name, index| {
        const expected_tag = try l.func.appendInst(l.block, tag_type, .{ .iconst = @intCast(index + 1) });
        const matches = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .eq, .lhs = tag.value, .rhs = expected_tag } });
        const yes = try l.func.appendBlock();
        const next = try l.func.appendBlock();
        try l.func.appendIf(l.block, matches, .{ .target = yes }, .{ .target = next });
        l.block = yes;
        const literal = try appendLiteralSlice(l, error_name);
        const pointer = try l.func.appendInst(l.block, ptr_type, .{ .load = .{ .ptr = literal.value } });
        const length = try l.func.appendInst(l.block, len_type, .{ .load = .{ .ptr = try byteOffsetPtr(l, literal.value, 8) } });
        try l.func.appendStore(l.block, pointer, slot);
        try l.func.appendStore(l.block, length, try byteOffsetPtr(l, slot, 8));
        try l.func.setJump(l.block, merge, &.{});
        l.block = next;
    }
    const fallback = try appendLiteralSlice(l, "unknown");
    const fallback_ptr = try l.func.appendInst(l.block, ptr_type, .{ .load = .{ .ptr = fallback.value } });
    const fallback_len = try l.func.appendInst(l.block, len_type, .{ .load = .{ .ptr = try byteOffsetPtr(l, fallback.value, 8) } });
    try l.func.appendStore(l.block, fallback_ptr, slot);
    try l.func.appendStore(l.block, fallback_len, try byteOffsetPtr(l, slot, 8));
    try l.func.setJump(l.block, merge, &.{});
    l.block = merge;
    return .{ .value = slot, .zty = slice_type };
}

fn errorTagForName(errors: ZErrorSet, name: []const u8) ?u16 {
    for (errors.names, 0..) |error_name, index| {
        if (std.mem.eql(u8, error_name, name)) return @intCast(index + 1);
    }
    return null;
}

fn lowerErrorValue(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!TypedValue {
    const name = l.r.tree.tokenSlice(l.r.tree.nodeMainToken(node) + 2);
    const zty = expected orelse blk: {
        try l.r.addInferredError(name);
        const err_set = try l.r.defaultErrorSet();
        break :blk ZType{ .error_set = err_set };
    };
    if (zty == .error_void) {
        const tag_type = try zty.irType(l.func);
        return .{ .value = try l.func.appendInst(l.block, tag_type, .{ .iconst = 1 }), .zty = zty };
    }
    const errors = switch (zty) {
        .error_set => |set| set.*,
        .error_union => |union_type| union_type.errors.*,
        else => return Error.TypeMismatch,
    };
    const tag = errorTagForName(errors, name) orelse blk: {
        try l.r.addInferredError(name);
        break :blk errorTagForName(errors, name) orelse return Error.TypeMismatch;
    };
    const error_type: ZType = .{ .error_set = switch (zty) {
        .error_set => |set| set,
        .error_union => |union_type| union_type.errors,
        else => unreachable,
    } };
    const tag_type = try error_type.irType(l.func);
    const tag_value = try l.func.appendInst(l.block, tag_type, .{ .iconst = tag });
    if (zty == .error_set) return .{ .value = tag_value, .zty = zty };
    const slot = try optionalSlot(l, zty);
    const byte_type = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const zero_byte = try l.func.appendInst(l.block, byte_type, .{ .iconst = 0 });
    var offset: u64 = 0;
    while (offset < zty.byteSize()) : (offset += 1)
        try l.func.appendStore(l.block, zero_byte, try byteOffsetPtr(l, slot, offset));
    try l.func.appendStore(l.block, tag_value, try byteOffsetPtr(l, slot, zty.errorTagOffset()));
    return .{ .value = slot, .zty = zty };
}

fn packErrorUnion(l: *L, payload: TypedValue, zty: ZType) Error!TypedValue {
    if (zty != .error_union or !payload.zty.eql(zty.error_union.payload.*)) return Error.TypeMismatch;
    const slot = try optionalSlot(l, zty);
    if (payload.zty.isAggregate()) {
        try copyAggregateBytes(l, slot, payload.value, payload.zty.byteSize());
    } else {
        try l.func.appendStore(l.block, payload.value, slot);
    }
    const tag_type = try (ZType{ .error_set = zty.error_union.errors }).irType(l.func);
    const zero = try l.func.appendInst(l.block, tag_type, .{ .iconst = 0 });
    try l.func.appendStore(l.block, zero, try byteOffsetPtr(l, slot, zty.errorTagOffset()));
    return .{ .value = slot, .zty = zty };
}

fn lowerExpectedErrorUnion(l: *L, node: Ast.Node.Index, zty: ZType) Error!TypedValue {
    if (l.r.tree.nodeTag(node) == .error_value) return lowerErrorValue(l, node, zty);
    if (try callReturnType(l, node)) |return_type| {
        if (return_type.eql(zty)) return lowerExpr(l, node, null);
    }
    const payload = try lowerExpr(l, node, zty.error_union.payload.*);
    if (payload.zty.eql(zty)) return payload;
    return packErrorUnion(l, payload, zty);
}
fn lowerExpr(l: *L, node: Ast.Node.Index, expected: ?ZType) Error!TypedValue {
    const tree = l.r.tree;
    if (l.r.modules) |modules| modules.mark(tree, node);
    // A qualified enum tag (`Limit.unlimited`) is a value, not an address; resolve it
    // before the general field-access path treats the name as storage.
    if (tree.nodeTag(node) == .field_access) {
        if (try enumFieldOrDefault(l, node)) |constant| return constant;
    }
    if (try importedIntegerConstant(l, node, expected)) |constant| return constant;
    if (expected != null and tree.nodeTag(node) == .@"orelse") return lowerOrelse(l, node, expected);
    if (tree.nodeTag(node) == .@"catch") return lowerCatchExpr(l, node, expected);
    if (tree.nodeTag(node) == .@"try") return lowerTryExpr(l, node, expected);
    if (expected) |zty| if (zty == .error_union) return lowerExpectedErrorUnion(l, node, zty);
    if (expected) |zty| if (zty == .optional_value) {
        if (tree.nodeTag(node) == .identifier and std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "null"))
            return packOptional(l, zty, null);
        const value = if (tree.nodeTag(node) == .number_literal)
            try lowerExpr(l, node, zty.optional_value.*)
        else
            try lowerExpr(l, node, null);
        if (value.zty.eql(zty)) return value;
        return packOptional(l, zty, value);
    };
    switch (tree.nodeTag(node)) {
        .char_literal => return lowerCharLiteral(l, node, expected),
        .string_literal => return lowerStringLiteral(l, node, expected),
        .number_literal => return lowerNumberLiteral(l, node, expected),
        .error_value => return lowerErrorValue(l, node, expected),
        .enum_literal => return lowerEnumLiteral(l, node, expected),
        .grouped_expression => return lowerExpr(l, tree.nodeData(node).node_and_token[0], expected),
        .identifier => {
            const name = tree.tokenSlice(tree.nodeMainToken(node));
            if (std.mem.eql(u8, name, "true")) {
                return .{ .value = try l.func.appendInst(l.block, try boolType(l.func), .{ .iconst = 1 }), .zty = .boolean };
            }
            if (std.mem.eql(u8, name, "false")) {
                return .{ .value = try l.func.appendInst(l.block, try boolType(l.func), .{ .iconst = 0 }), .zty = .boolean };
            }
            if (std.mem.eql(u8, name, "null")) {
                const zty = expected orelse return Error.TypeMismatch;
                if (zty != .optional_ptr) return Error.TypeMismatch;
                return .{ .value = try l.func.appendInst(l.block, try zty.irType(l.func), .{ .iconst = 0 }), .zty = zty };
            }
            if (try localConstEntry(l, name)) |constant| return try constEntryValue(l, constant, expected);
            return loadOrDecay(l, try lowerAddr(l, node));
        },
        .deref, .array_access, .field_access => return loadOrDecay(l, try lowerAddr(l, node)),
        .slice_open, .slice, .slice_sentinel => return lowerSliceExpr(l, node),
        .unwrap_optional => {
            const optional = try lowerExpr(l, tree.nodeData(node).node_and_token[0], null);
            if (optional.zty != .optional_ptr and optional.zty != .optional_value) return Error.TypeMismatch;
            return optionalPayload(l, optional);
        },
        .call_one, .call_one_comma, .call, .call_comma => return try lowerCall(l, node) orelse Error.TypeMismatch,
        .builtin_call_two, .builtin_call_two_comma, .builtin_call, .builtin_call_comma => return try lowerBuiltinCall(l, node, expected) orelse Error.TypeMismatch,
        .@"orelse" => return lowerOrelse(l, node, expected),
        .@"catch" => return lowerCatchExpr(l, node, expected),
        .@"try" => return lowerTryExpr(l, node, expected),
        .add, .add_wrap, .sub, .sub_wrap, .mul, .mul_wrap, .div, .mod, .bit_and, .bit_or, .bit_xor, .shl, .shr => {
            const d = tree.nodeData(node).node_and_node;
            const lhs = try lowerExpr(l, d[0], expected);
            const rhs = try lowerExpr(l, d[1], lhs.zty);
            if (!lhs.zty.eql(rhs.zty)) return Error.TypeMismatch;
            if (lhs.zty != .int and lhs.zty != .float) return Error.TypeMismatch;
            const irty = try lhs.zty.irType(l.func);
            const r = try l.func.appendInst(l.block, irty, .{ .arith = .{ .op = binOpFor(tree.nodeTag(node)).?, .lhs = lhs.value, .rhs = rhs.value } });
            return .{ .value = r, .zty = lhs.zty };
        },
        .equal_equal, .bang_equal, .less_than, .greater_than, .less_or_equal, .greater_or_equal => {
            const d = tree.nodeData(node).node_and_node;
            const lhs = try lowerExpr(l, d[0], null);
            const rhs = try lowerExpr(l, d[1], lhs.zty);
            const lhs_ok = lhs.zty.eql(rhs.zty) or enumLike(lhs.zty) and enumLike(rhs.zty) and
                comptimeCoerces(enumBackingOf(lhs.zty), enumBackingOf(rhs.zty));
            if (!lhs_ok) return Error.TypeMismatch;
            const boolt = try boolType(l.func);
            const r = try l.func.appendInst(l.block, boolt, .{ .icmp = .{ .op = cmpOpFor(tree.nodeTag(node)).?, .lhs = lhs.value, .rhs = rhs.value } });
            return .{ .value = r, .zty = .boolean };
        },
        .bool_and, .bool_or => {
            const d = tree.nodeData(node).node_and_node;
            const lhs = try lowerExpr(l, d[0], .boolean);
            if (!lhs.zty.eql(.boolean)) return Error.TypeMismatch;
            const boolt = try boolType(l.func);
            const rhs_b = try l.func.appendBlock();
            const merge_b = try l.func.appendBlock();
            const p = try l.func.appendBlockParam(merge_b, boolt);
            if (tree.nodeTag(node) == .bool_and) {
                try l.func.appendIf(l.block, lhs.value, .{ .target = rhs_b }, .{ .target = merge_b, .args = &.{lhs.value} });
            } else {
                try l.func.appendIf(l.block, lhs.value, .{ .target = merge_b, .args = &.{lhs.value} }, .{ .target = rhs_b });
            }
            l.block = rhs_b;
            const rhs = try lowerExpr(l, d[1], .boolean);
            if (!rhs.zty.eql(.boolean)) return Error.TypeMismatch;
            try l.func.setJump(l.block, merge_b, &.{rhs.value});
            l.block = merge_b;
            return .{ .value = p, .zty = .boolean };
        },
        .bool_not => {
            const d = tree.nodeData(node).node;
            const v = try lowerExpr(l, d, .boolean);
            if (!v.zty.eql(.boolean)) return Error.TypeMismatch;
            const boolt = try boolType(l.func);
            const zero = try l.func.appendInst(l.block, boolt, .{ .iconst = 0 });
            const r = try l.func.appendInst(l.block, boolt, .{ .icmp = .{ .op = .eq, .lhs = v.value, .rhs = zero } });
            return .{ .value = r, .zty = .boolean };
        },
        .negation => {
            const d = tree.nodeData(node).node;
            const v = try lowerExpr(l, d, expected);
            const irty = try v.zty.irType(l.func);
            switch (v.zty) {
                .int => {
                    const zero = try l.func.appendInst(l.block, irty, .{ .iconst = 0 });
                    const r = try l.func.appendInst(l.block, irty, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = v.value } });
                    return .{ .value = r, .zty = v.zty };
                },
                .float => {
                    const zero = try l.func.appendInst(l.block, irty, .{ .fconst = 0 });
                    const r = try l.func.appendInst(l.block, irty, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = v.value } });
                    return .{ .value = r, .zty = v.zty };
                },
                else => return Error.TypeMismatch,
            }
        },
        .bit_not => {
            const d = tree.nodeData(node).node;
            const v = try lowerExpr(l, d, expected);
            if (v.zty != .int) return Error.TypeMismatch;
            const irty = try v.zty.irType(l.func);
            const allones = try l.func.appendInst(l.block, irty, .{ .iconst = -1 });
            const r = try l.func.appendInst(l.block, irty, .{ .arith = .{ .op = .bit_xor, .lhs = v.value, .rhs = allones } });
            return .{ .value = r, .zty = v.zty };
        },
        .address_of => {
            const d = tree.nodeData(node).node;
            const a = try lowerAddr(l, d);
            const p = try l.r.arena.create(ZType);
            p.* = a.zty;
            const ptr_val = TypedValue{ .value = a.value, .zty = ZType{ .ptr = p } };
            if (expected) |exp| {
                if (exp == .slice and a.zty == .array and a.zty.array.elem.eql(exp.slice.*)) {
                    return convertTo(l, ptr_val, exp);
                }
            }
            return ptr_val;
        },
        else => return Error.UnsupportedNode,
    }
}

fn blockStatements(tree: *const Ast, node: Ast.Node.Index, buf: *[2]Ast.Node.Index) []const Ast.Node.Index {
    switch (tree.nodeTag(node)) {
        .block_two, .block_two_semicolon => {
            const d = tree.nodeData(node).opt_node_and_opt_node;
            var n: usize = 0;
            if (d[0].unwrap()) |a| {
                buf[0] = a;
                n = 1;
            }
            if (d[1].unwrap()) |b| {
                buf[n] = b;
                n += 1;
            }
            return buf[0..n];
        },
        .block, .block_semicolon => return tree.extraDataSlice(tree.nodeData(node).extra_range, Ast.Node.Index),
        else => return &.{},
    }
}

fn lowerBlock(l: *L, node: Ast.Node.Index) Error!bool {
    var buf: [2]Ast.Node.Index = undefined;
    const stmts = blockStatements(l.r.tree, node, &buf);
    for (stmts) |s| {
        if (try lowerStmt(l, s)) return true;
    }
    return false;
}

/// Lower one statement (or, for `if`/`while`/`for` bodies and `{}`, a nested block). Returns
/// `true` when the statement unconditionally transfers control out of `l.block` (a `return`,
/// `break`, `continue`, or an `if`/block whose every path already terminated).
fn asmTemplate(l: *L, node: Ast.Node.Index) Error![]const u8 {
    const tree = l.r.tree;
    if (tree.nodeTag(node) == .string_literal)
        return std.zig.string_literal.parseAlloc(l.r.arena, tree.tokenSlice(tree.nodeMainToken(node))) catch Error.UnsupportedNode;
    if (tree.nodeTag(node) != .multiline_string_literal) return Error.UnsupportedNode;
    const tokens = tree.nodeData(node).token_and_token;
    var lines: std.ArrayList(u8) = .empty;
    var token = tokens[0];
    while (token <= tokens[1]) : (token += 1) {
        if (tree.tokenTag(token) != .multiline_string_literal_line) continue;
        const raw = tree.tokenSlice(token);
        if (!std.mem.startsWith(u8, raw, "\\\\")) return Error.UnsupportedNode;
        try lines.appendSlice(l.r.arena, std.mem.trimStart(u8, raw[2..], " \t"));
        try lines.append(l.r.arena, '\n');
    }
    if (lines.items.len == 0) return Error.UnsupportedNode;
    return lines.toOwnedSlice(l.r.arena);
}
fn lowerSwitchStmt(l: *L, node: Ast.Node.Index) Error!bool {
    const tree = l.r.tree;
    const sw = tree.fullSwitch(node) orelse return Error.UnsupportedNode;
    const condition = try lowerExpr(l, sw.ast.condition, null);
    if (condition.zty != .int or sw.ast.cases.len == 0) return Error.UnsupportedType;
    const cases = try l.r.arena.alloc(Ast.full.SwitchCase, sw.ast.cases.len);
    const blocks = try l.r.arena.alloc(@TypeOf(l.block), sw.ast.cases.len);
    var else_block: ?@TypeOf(l.block) = null;
    for (sw.ast.cases, 0..) |case_node, i| {
        cases[i] = tree.fullSwitchCase(case_node) orelse return Error.UnsupportedNode;
        blocks[i] = try l.func.appendBlock();
        if (cases[i].ast.values.len == 0) {
            if (else_block != null) return Error.UnsupportedNode;
            else_block = blocks[i];
        }
    }
    const merge = try l.func.appendBlock();
    var test_block = l.block;
    for (cases, 0..) |case, i| {
        for (case.ast.values) |value_node| {
            l.block = test_block;
            const value = try lowerExpr(l, value_node, condition.zty);
            if (!value.zty.eql(condition.zty)) return Error.TypeMismatch;
            const equal = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .eq, .lhs = condition.value, .rhs = value.value } });
            const next_test = try l.func.appendBlock();
            try l.func.appendIf(l.block, equal, .{ .target = blocks[i] }, .{ .target = next_test });
            test_block = next_test;
        }
    }
    const fallback = else_block orelse return Error.UnsupportedNode;
    l.block = test_block;
    try l.func.setJump(l.block, fallback, &.{});

    var all_terminated = true;
    for (cases, 0..) |case, i| {
        l.block = blocks[i];
        if (!try lowerStmt(l, case.ast.target_expr)) {
            all_terminated = false;
            try l.func.setJump(l.block, merge, &.{});
        }
    }
    l.block = merge;
    return all_terminated;
}

fn lowerPointerArrayFor(l: *L, ff: Ast.full.For) Error!bool {
    const tree = l.r.tree;
    if (ff.ast.inputs.len != 2) return Error.UnsupportedNode;
    const source = try lowerExpr(l, ff.ast.inputs[0], null);
    if (source.zty != .ptr or source.zty.ptr.* != .array) return Error.UnsupportedType;
    const array = source.zty.ptr.array;

    const range = ff.ast.inputs[1];
    if (tree.nodeTag(range) != .for_range) return Error.UnsupportedNode;
    const range_data = tree.nodeData(range).node_and_opt_node;
    if (range_data[1].unwrap() != null) return Error.UnsupportedNode;

    var capture = ff.payload_token;
    const pointer_capture = tree.tokenTag(capture) == .asterisk;
    if (pointer_capture) capture += 1;
    if (tree.tokenTag(capture) != .identifier) return Error.UnsupportedNode;
    const item_name = tree.tokenSlice(capture);
    capture += 1;
    if (tree.tokenTag(capture) != .comma) return Error.UnsupportedNode;
    capture += 1;
    if (tree.tokenTag(capture) != .identifier) return Error.UnsupportedNode;
    const index_name = tree.tokenSlice(capture);
    const index_ty = ZType{ .int = .{ .signed = false, .bits = 64 } };
    const start = try lowerExpr(l, range_data[0], index_ty);
    if (!start.zty.eql(index_ty)) return Error.TypeMismatch;
    const end = try l.func.appendInst(l.block, try index_ty.irType(l.func), .{ .iconst = @intCast(array.len) });

    const index_addr = try declareLocal(l, index_name, index_ty, start.value);
    var element_ptr: ?ZType = null;
    var item_addr: ?Value = null;
    if (pointer_capture) {
        const pointee = try l.r.arena.create(ZType);
        pointee.* = array.elem.*;
        const ptr_ty = ZType{ .ptr = pointee };
        element_ptr = ptr_ty;
        item_addr = try declareLocal(l, item_name, ptr_ty, source.value);
    } else {
        return Error.UnsupportedNode;
    }

    const cond_b = try l.func.appendBlock();
    const body_b = try l.func.appendBlock();
    const inc_b = try l.func.appendBlock();
    const merge_b = try l.func.appendBlock();
    try l.func.setJump(l.block, cond_b, &.{});

    l.block = cond_b;
    const current = try l.func.appendInst(l.block, try index_ty.irType(l.func), .{ .load = .{ .ptr = index_addr } });
    const cond = try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .lt, .lhs = current, .rhs = end } });
    try l.func.appendIf(l.block, cond, .{ .target = body_b }, .{ .target = merge_b });

    l.block = body_b;
    const current_index = try l.func.appendInst(l.block, try index_ty.irType(l.func), .{ .load = .{ .ptr = index_addr } });
    const offset = try indexPtr(l, source.value, try toI64(l, .{ .value = current_index, .zty = index_ty }), array.elem.byteSize());
    _ = element_ptr.?;
    try l.func.appendStore(l.block, offset, item_addr.?);
    const saved_break = l.break_target;
    const saved_continue = l.continue_target;
    l.break_target = merge_b;
    l.continue_target = inc_b;
    const body_term = try lowerStmt(l, ff.ast.then_expr);
    l.break_target = saved_break;
    l.continue_target = saved_continue;
    if (!body_term) try l.func.setJump(l.block, inc_b, &.{});

    l.block = inc_b;
    const index = try l.func.appendInst(l.block, try index_ty.irType(l.func), .{ .load = .{ .ptr = index_addr } });
    const one = try l.func.appendInst(l.block, try index_ty.irType(l.func), .{ .iconst = 1 });
    const next = try l.func.appendInst(l.block, try index_ty.irType(l.func), .{ .arith = .{ .op = .add, .lhs = index, .rhs = one } });
    try l.func.appendStore(l.block, next, index_addr);
    try l.func.setJump(l.block, cond_b, &.{});
    l.block = merge_b;
    return false;
}

fn lowerStmt(l: *L, stmt: Ast.Node.Index) Error!bool {
    const tree = l.r.tree;
    if (l.r.modules) |modules| modules.mark(tree, stmt);
    if (l.is_naked and tree.nodeTag(stmt) != .asm_simple) return Error.UnsupportedNode;
    switch (tree.nodeTag(stmt)) {
        .block, .block_semicolon, .block_two, .block_two_semicolon => return lowerBlock(l, stmt),
        .@"switch", .switch_comma => return lowerSwitchStmt(l, stmt),

        .simple_var_decl, .local_var_decl, .aligned_var_decl => {
            const vd = tree.fullVarDecl(stmt) orelse return Error.UnsupportedNode;
            const type_node = vd.ast.type_node.unwrap();
            const init_node = vd.ast.init_node.unwrap() orelse return Error.UnsupportedNode;
            const name = tree.tokenSlice(vd.ast.mut_token + 1);

            if (isCVaStartCall(tree, init_node)) {
                const addr = try declareLocal(l, name, .va_list, null);
                try l.func.appendVaStart(l.block, addr);
                return false;
            }
            if (isUndefinedLiteral(tree, init_node)) {
                const tn = type_node orelse return Error.UnsupportedType;
                const zty = try resolveType(l.r, tn);
                _ = try declareLocal(l, name, zty, null);
                return false;
            }

            if (isStructInitTag(tree.nodeTag(init_node))) {
                const zty = if (type_node) |tn| try resolveType(l.r, tn) else blk: {
                    var buf2: [2]Ast.Node.Index = undefined;
                    const si = tree.fullStructInit(&buf2, init_node) orelse return Error.UnsupportedNode;
                    const te = si.ast.type_expr.unwrap() orelse return Error.UnsupportedType;
                    break :blk try resolveType(l.r, te);
                };
                if (zty != .strct) return Error.UnsupportedType;
                const addr = try declareLocal(l, name, zty, null);
                try lowerStructInitInto(l, init_node, addr, zty.strct);
                return false;
            }

            if (isArrayInitTag(tree.nodeTag(init_node))) {
                var buf2: [2]Ast.Node.Index = undefined;
                const ai = tree.fullArrayInit(&buf2, init_node) orelse return Error.UnsupportedNode;
                const zty = if (type_node) |tn| try resolveType(l.r, tn) else blk: {
                    const te = ai.ast.type_expr.unwrap() orelse return Error.UnsupportedType;
                    var parsed_ty = try resolveType(l.r, te);
                    if (parsed_ty == .array and parsed_ty.array.len == 0) {
                        parsed_ty.array.len = ai.ast.elements.len;
                    }
                    break :blk parsed_ty;
                };
                if (zty != .array) return Error.UnsupportedType;
                const addr = try declareLocal(l, name, zty, null);
                try lowerArrayInitInto(l, init_node, addr, zty.array);
                return false;
            }

            // A local `const Name = enum {...};` (or struct) declares a type, not a value.
            var local_container_buf: [2]Ast.Node.Index = undefined;
            if (tree.fullContainerDecl(&local_container_buf, init_node)) |container| {
                const tag = tree.tokenTag(container.ast.main_token);
                if (tag != .keyword_struct and tag != .keyword_enum) return Error.UnsupportedNode;
                const zty = try resolveContainerTypeIn(l.r, tree, l.r.dir, init_node);
                try l.r.local_types.put(l.r.arena, name, zty);
                return false;
            }

            if (type_node) |tn| {
                const zty = try resolveType(l.r, tn);
                const v = try lowerExpr(l, init_node, zty);
                if (zty == .optional_value) {
                    const dest = try declareLocal(l, name, zty, null);
                    try copyOptional(l, zty, dest, v.value);
                } else if (zty == .strct or zty == .array or zty == .slice) {
                    if (!v.zty.eql(zty)) return Error.TypeMismatch;
                    const dest = try declareLocal(l, name, zty, null);
                    try copyAggregateBytes(l, dest, v.value, zty.byteSize());
                } else {
                    const cv = try convertTo(l, v, zty);
                    _ = try declareLocal(l, name, zty, cv.value);
                }
                return false;
            }

            const v = try lowerExpr(l, init_node, null);
            if (v.zty == .strct or v.zty == .array or v.zty == .slice) {
                const dest = try declareLocal(l, name, v.zty, null);
                try copyAggregateBytes(l, dest, v.value, v.zty.byteSize());
            } else {
                const dest = try declareLocal(l, name, v.zty, if (v.zty == .optional_value) null else v.value);
                if (v.zty == .optional_value) try copyOptional(l, v.zty, dest, v.value);
            }
            return false;
        },

        .@"catch" => return lowerVoidCatch(l, stmt),
        .@"try" => {
            _ = try lowerTryExpr(l, stmt, null);
            return false;
        },
        .@"defer" => {
            const deferred = tree.nodeData(stmt).node;
            if (!isBuiltinCallTag(tree.nodeTag(deferred)) or !std.mem.eql(u8, builtinName(tree, deferred), "@cVaEnd")) return Error.UnsupportedNode;
            var buf: [2]Ast.Node.Index = undefined;
            const args = builtinArgs(tree, deferred, &buf);
            if (args.len != 1) return Error.CallArityMismatch;
            try l.deferred_va_ends.append(l.func.allocator, try lowerVaListAddr(l, args[0]));
            return false;
        },

        .if_simple, .@"if" => {
            const fi = tree.fullIf(stmt).?;
            if (fi.error_token != null) return Error.UnsupportedNode;
            const Payload = struct { name: []const u8, zty: ZType, value: Value, indirect: bool = false };
            var payload: ?Payload = null;
            const condition = if (fi.payload_token) |payload_token| blk: {
                if (tree.tokenTag(payload_token) != .identifier) return Error.UnsupportedNode;
                const optional = try lowerExpr(l, fi.ast.cond_expr, null);
                switch (optional.zty) {
                    .optional_ptr => |p| {
                        const ptr_zty = ZType{ .ptr = p };
                        const ptrt = try ptr_zty.irType(l.func);
                        const zero = try l.func.appendInst(l.block, ptrt, .{ .iconst = 0 });
                        payload = .{ .name = tree.tokenSlice(payload_token), .zty = ptr_zty, .value = optional.value };
                        break :blk try l.func.appendInst(l.block, try boolType(l.func), .{ .icmp = .{ .op = .ne, .lhs = optional.value, .rhs = zero } });
                    },
                    .optional_value => |p| {
                        payload = .{ .name = tree.tokenSlice(payload_token), .zty = p.*, .value = optional.value, .indirect = true };
                        break :blk try optionalPresence(l, optional);
                    },
                    else => return Error.TypeMismatch,
                }
            } else blk: {
                const value = try lowerExpr(l, fi.ast.cond_expr, .boolean);
                if (!value.zty.eql(.boolean)) return Error.TypeMismatch;
                break :blk value.value;
            };

            const then_b = try l.func.appendBlock();
            const else_node = fi.ast.else_expr.unwrap();
            const else_b = if (else_node != null) try l.func.appendBlock() else null;
            const merge_b = try l.func.appendBlock();
            try l.func.appendIf(l.block, condition, .{ .target = then_b }, .{ .target = else_b orelse merge_b });

            const scope_mark = l.env.items.len;
            l.block = then_b;
            if (payload) |p| {
                if (p.indirect) {
                    try l.env.append(l.func.allocator, .{ .name = p.name, .zty = p.zty, .addr = p.value });
                } else {
                    _ = try declareLocal(l, p.name, p.zty, p.value);
                }
            }
            const then_term = try lowerStmt(l, fi.ast.then_expr);
            if (!then_term) try l.func.setJump(l.block, merge_b, &.{});
            l.env.shrinkRetainingCapacity(scope_mark);

            var all_term = false;
            if (else_node) |en| {
                l.block = else_b.?;
                const else_term = try lowerStmt(l, en);
                if (!else_term) try l.func.setJump(l.block, merge_b, &.{});
                l.env.shrinkRetainingCapacity(scope_mark);
                all_term = then_term and else_term;
            }

            l.block = merge_b;
            return all_term;
        },

        .while_simple, .while_cont, .@"while" => {
            const fw = tree.fullWhile(stmt).?;
            if (fw.payload_token != null or fw.label_token != null or fw.inline_token != null or fw.ast.else_expr.unwrap() != null) return Error.UnsupportedNode;

            const cond_b = try l.func.appendBlock();
            const body_b = try l.func.appendBlock();
            const merge_b = try l.func.appendBlock();
            const cont_node = fw.ast.cont_expr.unwrap();
            const back_target = if (cont_node != null) try l.func.appendBlock() else cond_b;

            try l.func.setJump(l.block, cond_b, &.{});
            l.block = cond_b;
            const cond = try lowerExpr(l, fw.ast.cond_expr, .boolean);
            if (!cond.zty.eql(.boolean)) return Error.TypeMismatch;
            try l.func.appendIf(l.block, cond.value, .{ .target = body_b }, .{ .target = merge_b });

            l.block = body_b;
            const saved_break = l.break_target;
            const saved_cont = l.continue_target;
            l.break_target = merge_b;
            l.continue_target = back_target;
            const body_term = try lowerStmt(l, fw.ast.then_expr);
            if (!body_term) try l.func.setJump(l.block, back_target, &.{});
            l.break_target = saved_break;
            l.continue_target = saved_cont;

            if (cont_node) |ce| {
                l.block = back_target;
                _ = try lowerStmt(l, ce);
                try l.func.setJump(l.block, cond_b, &.{});
            }

            l.block = merge_b;
            return false;
        },

        .for_simple, .@"for" => {
            const ff = tree.fullFor(stmt).?;
            if (ff.inline_token != null or ff.label_token != null or ff.ast.else_expr.unwrap() != null) return Error.UnsupportedNode;
            if (ff.ast.inputs.len != 1) return try lowerPointerArrayFor(l, ff);
            const range_node = ff.ast.inputs[0];
            if (tree.nodeTag(range_node) != .for_range) return Error.UnsupportedNode;
            const rd = tree.nodeData(range_node).node_and_opt_node;
            const end_node = rd[1].unwrap() orelse return Error.UnsupportedNode;
            if (tree.tokenTag(ff.payload_token) != .identifier) return Error.UnsupportedNode;
            const idx_name = tree.tokenSlice(ff.payload_token);

            const end_v = try lowerExpr(l, end_node, null);
            if (end_v.zty != .int) return Error.TypeMismatch;
            const index_zty = end_v.zty;
            const start_v = try lowerExpr(l, rd[0], index_zty);
            const irty = try index_zty.irType(l.func);
            const idx_addr = try declareLocal(l, idx_name, index_zty, start_v.value);

            const cond_b = try l.func.appendBlock();
            const body_b = try l.func.appendBlock();
            const inc_b = try l.func.appendBlock();
            const merge_b = try l.func.appendBlock();
            try l.func.setJump(l.block, cond_b, &.{});

            l.block = cond_b;
            const cur = try l.func.appendInst(l.block, irty, .{ .load = .{ .ptr = idx_addr } });
            const boolt = try boolType(l.func);
            const cmp = try l.func.appendInst(l.block, boolt, .{ .icmp = .{ .op = .lt, .lhs = cur, .rhs = end_v.value } });
            try l.func.appendIf(l.block, cmp, .{ .target = body_b }, .{ .target = merge_b });

            l.block = body_b;
            const saved_break = l.break_target;
            const saved_cont = l.continue_target;
            l.break_target = merge_b;
            l.continue_target = inc_b;
            const body_term = try lowerStmt(l, ff.ast.then_expr);
            if (!body_term) try l.func.setJump(l.block, inc_b, &.{});
            l.break_target = saved_break;
            l.continue_target = saved_cont;

            l.block = inc_b;
            const cur2 = try l.func.appendInst(l.block, irty, .{ .load = .{ .ptr = idx_addr } });
            const one = try l.func.appendInst(l.block, irty, .{ .iconst = 1 });
            const next = try l.func.appendInst(l.block, irty, .{ .arith = .{ .op = .add, .lhs = cur2, .rhs = one } });
            try l.func.appendStore(l.block, next, idx_addr);
            try l.func.setJump(l.block, cond_b, &.{});

            l.block = merge_b;
            return false;
        },

        .@"break" => {
            const d = tree.nodeData(stmt).opt_token_and_opt_node;
            if (d[0] != .none or d[1].unwrap() != null) return Error.UnsupportedNode;
            const target = l.break_target orelse return Error.UnsupportedNode;
            try l.func.setJump(l.block, target, &.{});
            return true;
        },
        .@"continue" => {
            const d = tree.nodeData(stmt).opt_token_and_opt_node;
            if (d[0] != .none or d[1].unwrap() != null) return Error.UnsupportedNode;
            const target = l.continue_target orelse return Error.UnsupportedNode;
            try l.func.setJump(l.block, target, &.{});
            return true;
        },
        .unreachable_literal => {
            try l.func.setJump(l.block, l.block, &.{});
            return true;
        },
        .@"return" => {
            const ret_node = tree.nodeData(stmt).opt_node.unwrap();
            if (ret_node == null) {
                if (l.ret_zty) |rz| {
                    if (rz != .error_void) return Error.TypeMismatch;
                    try emitDeferredVaEnds(l);
                    const zero = try l.func.appendInst(l.block, try rz.irType(l.func), .{ .iconst = 0 });
                    l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.one(zero) });
                    return true;
                }
                try emitDeferredVaEnds(l);
                l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.none() });
                return true;
            }
            const rz = l.ret_zty orelse return Error.TypeMismatch;
            const v = try lowerExpr(l, ret_node.?, rz);
            try emitDeferredVaEnds(l);
            if (rz == .optional_value or rz == .slice) {
                if (l.sret_ptr) |dest| {
                    if (rz == .optional_value) {
                        try copyOptional(l, rz, dest, v.value);
                    } else {
                        try copyAggregateBytes(l, dest, v.value, rz.byteSize());
                    }
                    l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.one(dest) });
                } else {
                    var chunks: [2]Value = undefined;
                    const words = try optionalChunks(l, rz, v.value, &chunks);
                    l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.many(words) });
                }
            } else if (rz == .error_union) {
                if (l.sret_ptr) |dest| {
                    try copyAggregateBytes(l, dest, v.value, rz.byteSize());
                    l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.one(dest) });
                } else {
                    var chunks: [2]Value = undefined;
                    const words = try optionalChunks(l, rz, v.value, &chunks);
                    l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.many(words) });
                }
            } else {
                const cv = try convertTo(l, v, rz);
                l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.one(cv.value) });
            }
            return true;
        },

        .assign => {
            const d = tree.nodeData(stmt).node_and_node;
            if (tree.nodeTag(d[0]) == .identifier and std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(d[0])), "_")) {
                _ = try lowerExpr(l, d[1], null);
                return false;
            }
            const addr = try lowerAddr(l, d[0]);
            if (addr.zty == .strct) {
                if (!isStructInitTag(tree.nodeTag(d[1]))) return Error.UnsupportedNode;
                try lowerStructInitInto(l, d[1], addr.value, addr.zty.strct);
                return false;
            }
            if (addr.zty == .array) {
                if (!isArrayInitTag(tree.nodeTag(d[1]))) return Error.UnsupportedNode;
                try lowerArrayInitInto(l, d[1], addr.value, addr.zty.array);
                return false;
            }
            if (addr.zty == .optional_value) {
                const v = try lowerExpr(l, d[1], addr.zty);
                try copyOptional(l, addr.zty, addr.value, v.value);
                return false;
            }
            const v = try lowerExpr(l, d[1], addr.zty);
            const cv = try convertTo(l, v, addr.zty);
            try l.func.appendStore(l.block, cv.value, addr.value);
            return false;
        },
        .assign_add, .assign_sub, .assign_mul, .assign_div, .assign_mod, .assign_bit_and, .assign_bit_or, .assign_bit_xor, .assign_shl, .assign_shr => {
            const d = tree.nodeData(stmt).node_and_node;
            const addr = try lowerAddr(l, d[0]);
            if (addr.zty != .int and addr.zty != .float) return Error.UnsupportedNode;
            const cur = try loadOrDecay(l, addr);
            const rhs = try lowerExpr(l, d[1], addr.zty);
            if (!rhs.zty.eql(addr.zty)) return Error.TypeMismatch;
            const irty = try addr.zty.irType(l.func);
            const nv = try l.func.appendInst(l.block, irty, .{ .arith = .{ .op = binOpFor(tree.nodeTag(stmt)).?, .lhs = cur.value, .rhs = rhs.value } });
            try l.func.appendStore(l.block, nv, addr.value);
            return false;
        },

        .asm_simple, .@"asm" => {
            if (l.r.target != .aarch64) return Error.UnsupportedNode;
            const asm_node = tree.fullAsm(stmt) orelse return Error.UnsupportedNode;
            if (l.is_naked) {
                if (asm_node.outputs.len != 0 or asm_node.inputs.len != 0 or l.func.naked_asm != null) return Error.UnsupportedNode;
                const body = try asmTemplate(l, asm_node.ast.template);
                l.func.naked_asm = try l.func.allocator.dupe(u8, body);
                return false;
            }
            if (asm_node.outputs.len != 0) return Error.UnsupportedNode;
            if (asm_node.inputs.len != 0) {
                const expected_names = [_][]const u8{ "handle", "table", "entry", "after" };
                if (asm_node.inputs.len != expected_names.len) return Error.UnsupportedNode;
                const template = std.mem.trim(u8, try asmTemplate(l, asm_node.ast.template), " \t\r\n");
                const expected_template =
                    "mov x0, %[handle]\n" ++
                    "mov x1, %[table]\n" ++
                    "mov x30, %[after]\n" ++
                    "br %[entry]";
                if (!std.mem.eql(u8, template, expected_template)) return Error.UnsupportedNode;
                var args: [4]Value = undefined;
                for (asm_node.inputs, 0..) |input, i| {
                    if (tree.nodeTag(input) != .asm_input or
                        !std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(input)), expected_names[i]))
                        return Error.UnsupportedNode;
                    const expression = tree.nodeData(input).node_and_token[0];
                    const value = try lowerExpr(l, expression, null);
                    args[i] = value.value;
                }
                try l.func.appendVoidCall(l.block, "__vulcan_aarch64_asm_uefi_transfer", &args);
                return true;
            }
            if (asm_node.outputs.len != 0 or asm_node.inputs.len != 0) return Error.UnsupportedNode;
            const raw_template: []const u8 = switch (tree.nodeTag(asm_node.ast.template)) {
                .string_literal => std.zig.string_literal.parseAlloc(l.r.arena, tree.tokenSlice(tree.nodeMainToken(asm_node.ast.template))) catch return Error.UnsupportedNode,
                .multiline_string_literal => blk: {
                    const range = tree.nodeData(asm_node.ast.template).token_and_token;
                    if (range[0] != range[1]) return Error.UnsupportedNode;
                    const raw = tree.tokenSlice(range[0]);
                    if (!std.mem.startsWith(u8, raw, "\\\\")) return Error.UnsupportedNode;
                    break :blk std.mem.trim(u8, raw[2..], " \t\r\n");
                },
                else => return Error.UnsupportedNode,
            };
            const template = std.mem.trim(u8, raw_template, " \t\r\n");
            // A reserved call keeps this side effect visible to IR passes. The AArch64
            // selector recognizes these names and emits the instruction, not a BL relocation.
            const name = if (std.mem.eql(u8, template, "wfe"))
                "__vulcan_aarch64_asm_wfe"
            else if (std.mem.eql(u8, template, "sev"))
                "__vulcan_aarch64_asm_sev"
            else if (std.mem.eql(u8, template, "isb"))
                "__vulcan_aarch64_asm_isb"
            else if (std.mem.eql(u8, template, "dsb sy"))
                "__vulcan_aarch64_asm_dsb_sy"
            else if (std.mem.eql(u8, template, "dsb ish"))
                "__vulcan_aarch64_asm_dsb_ish"
            else if (std.mem.eql(u8, template, "eret"))
                "__vulcan_aarch64_asm_eret"
            else if (std.mem.eql(u8, template, "hvc #0"))
                "__vulcan_aarch64_asm_hvc_0"
            else if (std.mem.eql(u8, template, "msr daifclr, #2"))
                "__vulcan_aarch64_asm_daifclr_2"
            else if (std.mem.eql(u8, template, "msr daifset, #0xf"))
                "__vulcan_aarch64_asm_daifset_f"
            else if (std.mem.eql(u8, template, "ic iallu"))
                "__vulcan_aarch64_asm_ic_iallu"
            else if (std.mem.eql(u8, template, "tlbi vmalle1"))
                "__vulcan_aarch64_asm_tlbi_vmalle1"
            else
                return Error.UnsupportedNode;
            try l.func.appendVoidCall(l.block, name, &.{});
            return false;
        },

        else => {
            if (isBuiltinCallTag(tree.nodeTag(stmt))) {
                _ = try lowerBuiltinCall(l, stmt, null);
                return false;
            }
            _ = try lowerCall(l, stmt);
            var call_buf: [1]Ast.Node.Index = undefined;
            const call = tree.fullCall(&call_buf, stmt) orelse return false;
            if (tree.nodeTag(call.ast.fn_expr) == .identifier) {
                const name = tree.tokenSlice(tree.nodeMainToken(call.ast.fn_expr));
                const sig = try ensureLocalFnSig(l, name);
                if (sig) |resolved_sig| if (resolved_sig.is_noreturn) {
                    try l.func.setJump(l.block, l.block, l.func.blockParams(l.block));
                    return true;
                };
            }
            if (tree.nodeTag(call.ast.fn_expr) == .field_access) {
                if (try importedFnSig(l, call.ast.fn_expr)) |sig| if (sig.is_noreturn) {
                    try l.func.setJump(l.block, l.block, l.func.blockParams(l.block));
                    return true;
                };
            }
            return false;
        },
    }
}

fn isNakedCallconv(tree: *const Ast, fn_proto: Ast.full.FnProto) bool {
    const callconv_node = fn_proto.ast.callconv_expr.unwrap() orelse return false;
    if (tree.nodeTag(callconv_node) == .enum_literal) {
        return std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(callconv_node)), "naked");
    }
    var call_buf: [1]Ast.Node.Index = undefined;
    const call = tree.fullCall(&call_buf, callconv_node) orelse return false;
    if (call.ast.params.len != 1 or tree.nodeTag(call.ast.params[0]) != .enum_literal) return false;
    return std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(call.ast.params[0])), "naked");
}

fn lowerFunction(gpa: std.mem.Allocator, ctx: *Ctx, proto_node: Ast.Node.Index, body: Ast.Node.Index, fn_sigs: []const FnSig) Error!Function {
    const tree = ctx.tree;
    var func = Function.init(gpa);
    errdefer func.deinit();
    const entry = try func.appendBlock();

    // `fullFnProto`'s one-param path points its `ast.params` slice at `buf1`: keep `buf1`
    // and every use of the `FnProto` it produces within this same call frame (never store
    // the `FnProto` itself past this function, see `lowerRoot`'s matching comment).
    var buf1: [1]Ast.Node.Index = undefined;
    const fn_proto = tree.fullFnProto(&buf1, proto_node) orelse return Error.UnsupportedNode;
    if (fn_proto.ast.section_expr.unwrap()) |section_node| {
        if (tree.nodeTag(section_node) != .string_literal) return Error.UnsupportedNode;
        const section_name = std.zig.string_literal.parseAlloc(ctx.arena, tree.tokenSlice(tree.nodeMainToken(section_node))) catch return Error.UnsupportedNode;
        func.link_section = try gpa.dupe(u8, section_name);
    }

    const ret_node = fn_proto.ast.return_type.unwrap() orelse return Error.UnsupportedNode;
    const is_noreturn = isNoReturnType(tree, ret_node);
    const ret_zty: ?ZType = try resolveFnReturnType(ctx, tree, ctx.dir, ret_node);
    if (ret_zty) |rz| {
        if ((rz.isAggregate() and rz != .optional_value and rz != .slice and rz != .error_union) or (rz.optionalPayloadIsAggregate() and (ctx.target != .aarch64 or rz.byteSize() <= 16))) return Error.UnsupportedType;
        if (rz == .error_union and rz.byteSize() > 16 and ctx.target != .aarch64) return Error.UnsupportedType;
    }

    const use_sret = if (ret_zty) |rz| (rz == .error_union and rz.byteSize() > 16) or (rz.optionalPayloadIsAggregate() and rz.byteSize() > 16) else false;
    if (use_sret) func.sret = true;
    var l: L = .{ .r = ctx, .func = &func, .block = entry, .ret_zty = ret_zty, .env = .empty, .fn_sigs = fn_sigs, .is_naked = isNakedCallconv(tree, fn_proto) };
    if (use_sret) l.sret_ptr = try func.appendBlockParam(entry, try func.types.ptrGlobal());
    defer l.env.deinit(gpa);
    defer l.deferred_va_ends.deinit(gpa);

    var pit = fn_proto.iterate(tree);
    var fixed_count: u32 = 0;
    while (pit.next()) |p| {
        if (p.anytype_ellipsis3) |tok| {
            if (tree.tokenTag(tok) != .ellipsis3) return Error.UnsupportedType;
            func.is_variadic = true;
            func.num_fixed_params = fixed_count;
            break;
        }
        const te = p.type_expr orelse return Error.UnsupportedType;
        const name_tok = p.name_token orelse return Error.UnsupportedNode;
        const pzty = try resolveType(ctx, te);
        if ((pzty.isAggregate() and pzty != .optional_value and pzty != .slice) or pzty.optionalPayloadIsAggregate()) return Error.UnsupportedType;
        if (pzty == .slice) {
            const slot = try declareLocal(&l, tree.tokenSlice(name_tok), pzty, null);
            const ptr_ty = try func.types.ptrGlobal();
            const len_ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
            const ptr = try func.appendBlockParam(entry, ptr_ty);
            const len = try func.appendBlockParam(entry, len_ty);
            try func.appendStore(entry, ptr, slot);
            try func.appendStore(entry, len, try byteOffsetPtr(&l, slot, 8));
            fixed_count += 1;
            continue;
        }
        if (pzty == .optional_value) {
            const slot = try declareLocal(&l, tree.tokenSlice(name_tok), pzty, null);
            const word_ty = try optionalWordType(&l, pzty.byteSize());
            const lo = try func.appendBlockParam(entry, word_ty);
            try func.appendStore(entry, lo, slot);
            if (pzty.byteSize() == 16) {
                const hi = try func.appendBlockParam(entry, word_ty);
                try func.appendStore(entry, hi, try byteOffsetPtr(&l, slot, 8));
            }
            fixed_count += 1;
            continue;
        }
        const irty = try pzty.irType(&func);
        const pv = try func.appendBlockParam(entry, irty);
        _ = try declareLocal(&l, tree.tokenSlice(name_tok), pzty, pv);
        fixed_count += 1;
    }

    const terminated = try lowerBlock(&l, body);
    if (l.is_naked) {
        if (func.naked_asm == null) return Error.UnsupportedNode;
        return func;
    }
    if (!terminated) {
        if (l.ret_zty) |rz| {
            if (rz != .error_void) return Error.UnsupportedNode;
            try emitDeferredVaEnds(&l);
            const zero = try func.appendInst(l.block, try rz.irType(&func), .{ .iconst = 0 });
            func.setTerminator(l.block, .{ .ret = ir.function.Ret.one(zero) });
        } else {
            try emitDeferredVaEnds(&l);
            if (is_noreturn)
                try func.setJump(l.block, l.block, &.{})
            else
                func.setTerminator(l.block, .{ .ret = ir.function.Ret.none() });
        }
    }
    return func;
}

/// A `test` body has no parameters and no return value. Failure in this subset is a
/// runtime trap (or an unsupported construct), not an error value.
fn lowerTestFunction(gpa: std.mem.Allocator, ctx: *Ctx, body: Ast.Node.Index, fn_sigs: []const FnSig) Error!Function {
    var func = Function.init(gpa);
    errdefer func.deinit();
    const entry = try func.appendBlock();
    var l: L = .{
        .r = ctx,
        .func = &func,
        .block = entry,
        .ret_zty = null,
        .env = .empty,
        .fn_sigs = fn_sigs,
    };
    defer l.env.deinit(gpa);
    defer l.deferred_va_ends.deinit(gpa);
    const terminated = try lowerBlock(&l, body);
    if (!terminated) {
        try emitDeferredVaEnds(&l);
        func.setTerminator(l.block, .{ .ret = ir.function.Ret.none() });
    }
    return func;
}

/// Lower every `test` declaration in the root module's files. `ModuleMapping` module
/// boundaries and `std` belong to other compilations, so their tests are not included.
fn lowerTests(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    compiler: *Compiler,
    root_ctx: *Ctx,
    root_tree: *const Ast,
    root_sigs: []const FnSig,
    filters: []const []const u8,
    funcs: *std.ArrayList(NamedFunction),
    tests: *std.ArrayList(Test),
) Error!void {
    const modules = root_ctx.modules;
    var test_index: usize = 0;
    var file_index: usize = 0;
    while (true) : (file_index += 1) {
        const tree: *const Ast = if (file_index == 0) root_tree else blk: {
            const module_files = if (modules) |m| m.files.items else break;
            const i = file_index - 1;
            if (i >= module_files.len) break;
            const file = module_files[i].file;
            const root_module_index = modules.?.root.?.module_index;
            if (file.is_std or file.module_index != root_module_index) continue;
            break :blk file.tree;
        };
        const ctx = if (tree == root_tree) root_ctx else try compiler.contextFor(tree);
        const sigs: []const FnSig = if (tree == root_tree) root_sigs else &.{};
        const loaded = if (modules) |m| m.owner(tree) else null;
        for (tree.rootDecls()) |decl| {
            if (tree.nodeTag(decl) != .test_decl) continue;
            const data = tree.nodeData(decl).opt_token_and_node;
            const body = data[1];
            const display = if (data[0].unwrap()) |name_token| blk: {
                if (tree.tokenTag(name_token) == .string_literal)
                    break :blk std.zig.string_literal.parseAlloc(arena, tree.tokenSlice(name_token)) catch return Error.UnsupportedNode;
                break :blk try arena.dupe(u8, tree.tokenSlice(name_token));
            } else blk: {
                const location = tree.tokenLocation(0, tree.nodeMainToken(decl));
                const file_name = if (loaded) |file| std.fs.path.basename(file.path) else "<source>";
                break :blk try std.fmt.allocPrint(arena, "{s}:{d}", .{ file_name, location.line + 1 });
            };
            if (filters.len != 0) {
                var matched = false;
                for (filters) |filter| {
                    if (std.mem.find(u8, display, filter) != null) {
                        matched = true;
                        break;
                    }
                }
                if (!matched) continue;
            }
            const link_name = try std.fmt.allocPrint(arena, "__vulcan_test_{d}", .{test_index});
            test_index += 1;
            var func = try lowerTestFunction(gpa, ctx, body, sigs);
            errdefer func.deinit();
            const owned_link = try gpa.dupe(u8, link_name);
            errdefer gpa.free(owned_link);
            try funcs.append(arena, .{ .name = owned_link, .func = func });
            const owned_name = try gpa.dupe(u8, display);
            errdefer gpa.free(owned_name);
            const owned_link_copy = try gpa.dupe(u8, link_name);
            errdefer gpa.free(owned_link_copy);
            try tests.append(arena, .{ .name = owned_name, .link_name = owned_link_copy });
        }
    }
}

fn lowerRoot(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    tree: *const Ast,
    target: TargetArch,
    dir: []const u8,
    modules: ?*Modules,
    include_tests: bool,
    test_filters: []const []const u8,
) Error!Module {
    var ctx: Ctx = undefined;
    var compiler = Compiler{ .gpa = gpa, .arena = arena, .target = target, .modules = modules, .root_ctx = &ctx };
    ctx = .{ .tree = tree, .dir = dir, .arena = arena, .target = target, .modules = modules, .compiler = &compiler, .is_root = true };
    try compiler.contexts.append(arena, &ctx);
    errdefer for (compiler.data.items) |d| {
        gpa.free(d.name);
        gpa.free(d.bytes);
    };
    var names: std.StringHashMapUnmanaged(void) = .empty;
    var imports: std.StringHashMapUnmanaged(void) = .empty;

    // Only the raw proto node is kept, never a `fullFnProto`-derived `FnProto`: its one-param
    // path returns an `ast.params` slice pointing at a CALLER-supplied stack buffer, which
    // cannot survive past the call that filled it. Every later use (the signature pass below,
    // and `lowerFunction`) re-derives the `FnProto` fresh with its own local buffer.
    const FnNode = struct { name: []const u8, proto_node: Ast.Node.Index, body: Ast.Node.Index };
    const GlobalNode = struct {
        name: []const u8,
        /// `null` when `zty` is already known (a struct-literal's own type expression, e.g.
        /// `uefi.Guid{...}` - see below); otherwise resolved from this pass.
        type_node: ?Ast.Node.Index,
        zty: ?ZType = null,
        init_node: ?Ast.Node.Index,
        is_const: bool,
        is_extern: bool = false,
        constant: ?ConstValue = null,
    };
    var fn_nodes: std.ArrayList(FnNode) = .empty;
    var global_nodes: std.ArrayList(GlobalNode) = .empty;
    // Globals inferred via `utf16LiteralArg` (no `resolveType`-derived type_node exists):
    // their `Data` is computed directly, so they merge into `data` alongside `global_nodes`'
    // output rather than going through `lowerGlobalData`. Backing array is `arena`-owned;
    // each entry's `name`/`bytes` are `gpa`-owned and transferred into `data` below.
    var inferred_data: std.ArrayList(Data) = .empty;
    errdefer for (inferred_data.items) |d| {
        gpa.free(d.name);
        gpa.free(d.bytes);
    };

    for (0..tree.nodes.len) |node_idx| {
        if (tree.nodes.items(.tag)[node_idx] == .error_value) {
            const err_name = tree.tokenSlice(tree.nodeMainToken(@fromBackingInt(@intCast(node_idx))) + 2);
            try ctx.addInferredError(err_name);
        }
    }

    for (tree.rootDecls()) |decl| {
        if (modules) |module_graph| module_graph.mark(tree, decl);
        switch (tree.nodeTag(decl)) {
            .fn_decl => {
                const dd = tree.nodeData(decl).node_and_node;
                var buf1: [1]Ast.Node.Index = undefined;
                const proto = tree.fullFnProto(&buf1, dd[0]) orelse return Error.UnsupportedNode;
                const name_tok = proto.name_token orelse return Error.UnsupportedNode;
                const name = tree.tokenSlice(name_tok);
                if (names.contains(name)) return Error.DuplicateDecl;
                try names.put(arena, name, {});
                try fn_nodes.append(arena, .{ .name = name, .proto_node = dd[0], .body = dd[1] });
                try ctx.function_decls.put(arena, name, .{ .decl_node = decl, .proto_node = dd[0], .body_node = dd[1] });
            },
            .simple_var_decl, .local_var_decl, .aligned_var_decl, .global_var_decl => {
                const vd = tree.fullVarDecl(decl) orelse return Error.UnsupportedNode;
                const init_opt = vd.ast.init_node.unwrap();
                const is_extern = if (vd.extern_export_token) |token|
                    tree.tokenTag(token) == .keyword_extern
                else
                    false;
                const name = tree.tokenSlice(vd.ast.mut_token + 1);
                if (names.contains(name)) return Error.DuplicateDecl;
                try names.put(arena, name, {});

                if (init_opt == null) {
                    if (!is_extern) return Error.UnsupportedNode;
                    const type_node = vd.ast.type_node.unwrap() orelse return Error.UnsupportedType;
                    try global_nodes.append(arena, .{
                        .name = name,
                        .type_node = type_node,
                        .init_node = null,
                        .is_const = tree.tokenTag(vd.ast.mut_token) == .keyword_const,
                        .is_extern = true,
                    });
                    continue;
                }
                if (is_extern) return Error.UnsupportedNode;
                const init_node = init_opt.?;

                var buf2: [2]Ast.Node.Index = undefined;
                if (tree.fullContainerDecl(&buf2, init_node)) |container| {
                    switch (tree.tokenTag(container.ast.main_token)) {
                        .keyword_struct => try ctx.nodes.put(arena, name, init_node),
                        .keyword_enum => try ctx.type_decls.put(arena, name, .{ .decl_node = init_node }),
                        else => {},
                    }
                    continue;
                }
                if (tree.tokenTag(vd.ast.mut_token) == .keyword_const and isImportNamespace(tree, init_node, &imports)) {
                    try imports.put(arena, name, {});
                    continue;
                }
                if (vd.ast.type_node.unwrap() == null and isStructInitTag(tree.nodeTag(init_node))) {
                    var buf3: [2]Ast.Node.Index = undefined;
                    const si = tree.fullStructInit(&buf3, init_node) orelse return Error.UnsupportedNode;
                    const te = si.ast.type_expr.unwrap() orelse return Error.UnsupportedType;
                    const zty = try resolveType(&ctx, te);
                    if (zty != .strct) return Error.UnsupportedType;
                    try global_nodes.append(arena, .{
                        .name = name,
                        .type_node = null,
                        .zty = zty,
                        .init_node = init_node,
                        .is_const = tree.tokenTag(vd.ast.mut_token) == .keyword_const,
                    });
                    continue;
                }
                if (tree.tokenTag(vd.ast.mut_token) == .keyword_const and vd.ast.type_node.unwrap() == null) {
                    if (evalTopConst(&ctx, init_node)) |value| {
                        try ctx.constants.put(arena, name, value);
                    } else {
                        try ctx.deferred_constants.put(arena, name, init_node);
                    }
                    continue;
                }
                if (vd.ast.type_node.unwrap() == null) {
                    if (utf16LiteralArg(tree, init_node)) |string_node| {
                        const encoded = try encodeUtf16LiteralGlobal(gpa, arena, tree, string_node);
                        errdefer gpa.free(encoded.bytes);
                        try ctx.globals.put(arena, name, .{ .zty = encoded.zty, .symbol = name });
                        const owned_name = try gpa.dupe(u8, name);
                        errdefer gpa.free(owned_name);
                        const is_const_decl = tree.tokenTag(vd.ast.mut_token) == .keyword_const;
                        if (!is_const_decl and allZero(encoded.bytes)) {
                            gpa.free(encoded.bytes);
                            try inferred_data.append(arena, .{ .name = owned_name, .bytes = try gpa.alloc(u8, 0), .kind = .bss, .size = encoded.zty.byteSize() });
                        } else {
                            try inferred_data.append(arena, .{ .name = owned_name, .bytes = encoded.bytes, .kind = if (is_const_decl) .rodata else .data, .size = encoded.bytes.len });
                        }
                        continue;
                    }
                }
                var constant: ?ConstValue = null;
                if (tree.tokenTag(vd.ast.mut_token) == .keyword_const) {
                    if (evalTopConst(&ctx, init_node)) |entry| {
                        constant = entry.value;
                        try ctx.constants.put(arena, name, entry);
                    }
                }
                const type_node = vd.ast.type_node.unwrap() orelse return Error.UnsupportedType;
                try global_nodes.append(arena, .{
                    .name = name,
                    .type_node = type_node,
                    .init_node = init_node,
                    .is_const = tree.tokenTag(vd.ast.mut_token) == .keyword_const,
                    .constant = constant,
                });
            },
            .test_decl => {},
            else => {
                var proto_buf: [1]Ast.Node.Index = undefined;
                const proto = tree.fullFnProto(&proto_buf, decl) orelse return Error.UnsupportedNode;
                const token = proto.name_token orelse return Error.UnsupportedNode;
                const name = tree.tokenSlice(token);
                if (names.contains(name)) return Error.DuplicateDecl;
                try names.put(arena, name, {});
                try ctx.function_decls.put(arena, name, .{ .decl_node = decl, .proto_node = decl, .body_node = null });
            },
        }
    }

    // Structs are resolved on demand after every declaration is registered, preserving
    // forward references without rejecting unused type declarations.

    // Register every global before lowering functions. A function may use a later source
    // declaration, and the direct `global_addr` relocation resolves only after the module's
    // complete data table reaches the linker.
    for (global_nodes.items) |global| {
        const zty = global.zty orelse try resolveType(&ctx, global.type_node.?);
        try ctx.globals.put(arena, global.name, .{ .zty = zty, .symbol = global.name });
    }
    var defined_global_count: usize = 0;
    for (global_nodes.items) |global| if (!global.is_extern) {
        defined_global_count += 1;
    };
    var data = try gpa.alloc(Data, defined_global_count + inferred_data.items.len);
    var data_built: usize = 0;
    errdefer {
        for (data[0..data_built]) |d| {
            gpa.free(d.name);
            gpa.free(d.bytes);
        }
        gpa.free(data);
    }
    for (global_nodes.items) |global| {
        if (global.is_extern) continue;
        const def = ctx.globals.get(global.name).?;
        const constant = global.constant orelse if (evalTopConst(&ctx, global.init_node.?)) |entry| entry.value else null;
        data[data_built] = try lowerGlobalData(gpa, tree, global.name, def.zty, global.is_const, global.init_node.?, constant);
        data_built += 1;
    }
    for (inferred_data.items) |d| {
        data[data_built] = d;
        data_built += 1;
    }
    // Ownership of every `inferred_data` entry's `name`/`bytes` just moved into `data`;
    // clear its view so the still-armed `errdefer` above does not double-free them on a
    // later error (its backing array remains `arena`-owned, so no separate deinit is due).
    inferred_data.items.len = 0;

    var fn_sigs: std.ArrayList(FnSig) = .empty;
    for (fn_nodes.items) |fd| {
        try fn_sigs.append(arena, (try ensureFunction(&ctx, fd.name)).?);
    }

    var funcs: std.ArrayList(NamedFunction) = .empty;
    errdefer {
        for (funcs.items) |*nf| {
            nf.func.deinit();
            gpa.free(nf.name);
        }
    }
    for (fn_nodes.items) |fd| {
        var func = try lowerFunction(gpa, &ctx, fd.proto_node, fd.body, fn_sigs.items);
        errdefer func.deinit();
        const name = try gpa.dupe(u8, fd.name);
        errdefer gpa.free(name);
        try funcs.append(arena, .{ .name = name, .func = func });
    }
    var tests: std.ArrayList(Test) = .empty;
    errdefer for (tests.items) |test_entry| {
        gpa.free(test_entry.name);
        gpa.free(test_entry.link_name);
    };
    if (include_tests) try lowerTests(gpa, arena, &compiler, &ctx, tree, fn_sigs.items, test_filters, &funcs, &tests);
    // Signatures publish their symbols before a body enters this queue. A reached
    // declaration can therefore recurse, or be reached through a diamond, only once.
    var next_function: usize = 0;
    while (next_function < compiler.function_queue.items.len) : (next_function += 1) {
        const key = compiler.function_queue.items[next_function];
        const entry = compiler.functions.get(key).?;
        var func = try lowerFunction(gpa, entry.ctx, entry.info.proto_node, entry.info.body_node.?, &.{});
        errdefer func.deinit();
        func.is_local = entry.is_local;
        const name = try gpa.dupe(u8, entry.symbol);
        errdefer gpa.free(name);
        try funcs.append(arena, .{ .name = name, .func = func });
    }
    var literal_count: usize = 0;
    for (compiler.contexts.items) |file_ctx| literal_count += file_ctx.literal_data.items.len;
    data = try gpa.realloc(data, data_built + compiler.data.items.len + literal_count);
    for (compiler.data.items) |d| {
        data[data_built] = d;
        data_built += 1;
    }
    compiler.data.items.len = 0;
    for (compiler.contexts.items) |file_ctx| for (file_ctx.literal_data.items) |literal| {
        const name = try gpa.dupe(u8, literal.name);
        const bytes = gpa.dupe(u8, literal.bytes) catch |err| {
            gpa.free(name);
            return err;
        };
        data[data_built] = .{ .name = name, .bytes = bytes, .kind = literal.kind, .size = literal.size };
        data_built += 1;
    };
    const imports_result = try collectImports(gpa, tree);
    errdefer {
        for (imports_result) |import| {
            gpa.free(import.name);
            gpa.free(import.path);
        }
        gpa.free(imports_result);
    }
    return Module{
        .funcs = try gpa.dupe(NamedFunction, funcs.items),
        .data = data,
        .imports = imports_result,
        .tests = try gpa.dupe(Test, tests.items),
    };
}

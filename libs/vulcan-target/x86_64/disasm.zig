//! x86-64 disassembler for the subset `encode.zig` emits. Unlike the fixed-width RISC
//! backends, x86 is variable length, so the decoder parses the real instruction structure
//! (legacy/mandatory prefix, REX or 3-byte VEX, 0F/0F38/0F3A opcode escape, ModRM+SIB+disp,
//! immediate) to both render the instruction AND compute its length, which is how it steps
//! through a byte stream. Anything outside the emitted subset prints as `.byte 0x<hex>`.
//!
//! The parsing lives in `decode.zig`, which yields a typed instruction; this file is the
//! rendering of that type.
//!
//! `format` renders a whole code buffer; `disasmInst` decodes one instruction and reports
//! its length. Validated by round-tripping every encoder function.

const std = @import("std");
const encode = @import("encode.zig");
const decoder = @import("decode.zig");

const Size = decoder.Size;

pub fn format(allocator: std.mem.Allocator, code: []const u8) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var pos: usize = 0;
    while (pos < code.len) {
        try out.print(allocator, "{x:0>4}: ", .{pos});
        const len = try decode(allocator, &out, code, pos);
        try out.append(allocator, '\n');
        pos += len;
    }
    return out.toOwnedSlice(allocator);
}

/// Decode one instruction; append its text and return its byte length.
pub fn disasmInst(allocator: std.mem.Allocator, out: *std.ArrayList(u8), code: []const u8, pos: usize) std.mem.Allocator.Error!usize {
    return decode(allocator, out, code, pos);
}

/// A function's byte offset in a linked image.
pub const Sym = struct { name: []const u8, offset: usize };

/// Render a whole linked module: each function gets a `name:` label at its byte offset, and
/// every resolved `call rel32` that lands on a function is annotated with its name, so a
/// linked image (from link.compileModule) reads as a symbolized listing. Caller owns it.
/// A decoded source-line row keyed by byte offset (from a `.debug_line` program), for objdump
/// `-S`-style annotation. In a relocatable object `.text` sits at address 0, so address == offset.
pub const AddrLine = struct { addr: u64, line: u32 };

pub fn formatModule(allocator: std.mem.Allocator, code: []const u8, syms: []const Sym) std.mem.Allocator.Error![]u8 {
    return formatModuleImpl(allocator, code, syms, &.{});
}

/// Like `formatModule`, but interleaves `; line N` markers from `lines` (sorted by address).
pub fn formatModuleWithLines(allocator: std.mem.Allocator, code: []const u8, syms: []const Sym, lines: []const AddrLine) std.mem.Allocator.Error![]u8 {
    return formatModuleImpl(allocator, code, syms, lines);
}

fn formatModuleImpl(allocator: std.mem.Allocator, code: []const u8, syms: []const Sym, lines: []const AddrLine) std.mem.Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var pos: usize = 0;
    var li: usize = 0;
    while (pos < code.len) {
        for (syms) |s| if (s.offset == pos) {
            if (pos != 0) try buf.append(allocator, '\n');
            try buf.print(allocator, "{s}:\n", .{s.name});
        };
        while (li < lines.len and lines[li].addr < pos) : (li += 1) {}
        while (li < lines.len and lines[li].addr == pos) : (li += 1) {
            try buf.print(allocator, "; line {d}\n", .{lines[li].line});
        }
        try buf.print(allocator, "{x:0>4}: ", .{pos});
        const len = try decode(allocator, &buf, code, pos);
        // A `call rel32` (0xE8): its target is `end + rel32`; annotate a hit symbol.
        // `decode` reports len 5 even for a truncated call (its cursor reads past the
        // end as 0), so guard the 4-byte read against the real buffer length.
        if (code[pos] == 0xE8 and len == 5 and code.len - pos >= 5) {
            const rel: i32 = @bitCast(std.mem.readInt(u32, code[pos + 1 ..][0..4], .little));
            const target = @as(i64, @intCast(pos)) + @as(i64, @intCast(len)) + rel;
            if (target >= 0) for (syms) |s| {
                if (s.offset == @as(usize, @intCast(target))) {
                    try buf.print(allocator, "  <{s}>", .{s.name});
                    break;
                }
            };
        }
        try buf.append(allocator, '\n');
        pos += len;
    }
    return buf.toOwnedSlice(allocator);
}

const gpr = [_][]const u8{
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
};
const gpr32 = [_][]const u8{
    "eax", "ecx", "edx",  "ebx",  "esp",  "ebp",  "esi",  "edi",
    "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
};
const gpr16 = [_][]const u8{
    "ax",  "cx",  "dx",   "bx",   "sp",   "bp",   "si",   "di",
    "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w",
};
const gpr8 = [_][]const u8{
    "al",  "cl",  "dl",   "bl",   "spl",  "bpl",  "sil",  "dil",
    "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
};

/// A general register at the given operand size. Addresses always use the 64-bit names.
fn regName(size: Size, n: u8) []const u8 {
    return switch (size) {
        .byte => gpr8[n & 15],
        .word => gpr16[n & 15],
        .dword => gpr32[n & 15],
        else => gpr[n & 15],
    };
}

/// A general register that is 64-bit with REX.W else 32-bit, ignoring a 0x66 that is a
/// mandatory SSE prefix (movd/movq, cvtsi2ss, cvttss2si) rather than an operand-size
/// override.
fn regNameWide(wide: bool, n: u8) []const u8 {
    return if (wide) gpr[n & 15] else gpr32[n & 15];
}

/// The `<size> ptr ` hint llvm prints before a memory operand.
fn hintText(size: Size) []const u8 {
    return switch (size) {
        .byte => "byte ptr ",
        .word => "word ptr ",
        .dword => "dword ptr ",
        .qword => "qword ptr ",
        .xmmword => "xmmword ptr ",
        .ymmword => "ymmword ptr ",
    };
}

/// Render a memory operand `<hint>[base + index*scale + disp]`. Address registers are 64-bit.
fn memText(a: std.mem.Allocator, b: *std.ArrayList(u8), m: decoder.Mem, hint: ?Size) !void {
    if (hint) |h| try b.appendSlice(a, hintText(h));
    if (m.base == null and m.index == null and !m.rip) {
        // A displacement with no base and no index: an absolute address.
        try b.print(a, "[{d}]", .{m.disp});
        return;
    }
    try b.append(a, '[');
    if (m.rip) {
        try b.appendSlice(a, "rip");
    } else if (m.base) |r| {
        try b.appendSlice(a, gpr[@backingInt(r)]);
    }
    if (m.index) |ix| {
        try b.appendSlice(a, " + ");
        if (m.scale != 1) try b.print(a, "{d}*", .{m.scale});
        try b.appendSlice(a, gpr[@backingInt(ix)]);
    }
    if (m.disp != 0) try b.print(a, " {s} {d}", .{ if (m.disp < 0) "-" else "+", @abs(m.disp) });
    try b.append(a, ']');
}

/// Render an immediate, signed when the processor sign-extends it.
fn immText(a: std.mem.Allocator, b: *std.ArrayList(u8), imm: decoder.Imm) !void {
    if (!imm.sign_extended) {
        try b.print(a, "{d}", .{imm.value});
        return;
    }
    const v: i64 = switch (imm.bits) {
        8 => @as(i8, @bitCast(@as(u8, @truncate(imm.value)))),
        16 => @as(i16, @bitCast(@as(u16, @truncate(imm.value)))),
        32 => @as(i32, @bitCast(@as(u32, @truncate(imm.value)))),
        else => @bitCast(imm.value),
    };
    try b.print(a, "{d}", .{v});
}

fn destText(a: std.mem.Allocator, b: *std.ArrayList(u8), d: decoder.Dest, size: Size) !void {
    switch (d) {
        .reg => |r| try b.appendSlice(a, regName(size, @backingInt(r))),
        .mem => |m| try memText(a, b, m, size),
    }
}

fn srcText(a: std.mem.Allocator, b: *std.ArrayList(u8), s: decoder.Src, size: Size) !void {
    switch (s) {
        .reg => |r| try b.appendSlice(a, regName(size, @backingInt(r))),
        .mem => |m| try memText(a, b, m, size),
        .imm => |i| try immText(a, b, i),
    }
}

/// A branch target: a displacement from the end of the instruction, a register, or a
/// memory operand.
fn targetText(a: std.mem.Allocator, b: *std.ArrayList(u8), t: decoder.Target) !void {
    switch (t) {
        .rel => |rel| try b.print(a, ".{s}{d}", .{ relSign(rel), @abs(rel) }),
        .reg => |r| try b.appendSlice(a, gpr[@backingInt(r)]),
        .mem => |m| try memText(a, b, m, .qword),
    }
}

/// The ALU mnemonic for an operation.
fn aluName(op: decoder.AluOp) []const u8 {
    return switch (op) {
        .add => "add",
        .@"or" => "or",
        .adc => "adc",
        .sbb => "sbb",
        .@"and" => "and",
        .sub => "sub",
        .xor => "xor",
        .cmp => "cmp",
        .@"test" => "test",
    };
}

fn unaryName(op: decoder.UnaryOp) []const u8 {
    return switch (op) {
        .not => "not",
        .neg => "neg",
        .mul => "mul",
        .imul => "imul",
        .div => "div",
        .idiv => "idiv",
    };
}

fn shiftName(op: decoder.ShiftOp) []const u8 {
    return switch (op) {
        .rol => "rol",
        .ror => "ror",
        .rcl => "rcl",
        .rcr => "rcr",
        .shl => "shl",
        .shr => "shr",
        .sar => "sar",
    };
}

fn condName(cc: decoder.Cond) []const u8 {
    return switch (cc) {
        .o => "o",
        .no => "no",
        .b => "b",
        .ae => "ae",
        .e => "e",
        .ne => "ne",
        .be => "be",
        .a => "a",
        .s => "s",
        .ns => "ns",
        .p => "p",
        .np => "np",
        .l => "l",
        .ge => "ge",
        .le => "le",
        .g => "g",
    };
}

fn relSign(v: i32) []const u8 {
    return if (v < 0) "-" else "+";
}

/// Render one decoded instruction.
fn render(a: std.mem.Allocator, b: *std.ArrayList(u8), inst: decoder.Instruction) !void {
    switch (inst) {
        .nop => try b.appendSlice(a, "nop"),
        .pause => try b.appendSlice(a, "pause"),
        .hlt => try b.appendSlice(a, "hlt"),
        .cli => try b.appendSlice(a, "cli"),
        .sti => try b.appendSlice(a, "sti"),
        .cld => try b.appendSlice(a, "cld"),
        .std => try b.appendSlice(a, "std"),
        .ret => try b.appendSlice(a, "ret"),
        .syscall => try b.appendSlice(a, "syscall"),
        .endbr64 => try b.appendSlice(a, "endbr64"),
        .widen => |w| try b.appendSlice(a, switch (w) {
            .cdq => "cdq",
            .cqo => "cqo",
            .cwde => "cwde",
            .cdqe => "cdqe",
        }),

        .mov => |m| {
            try b.appendSlice(a, "mov ");
            try destText(a, b, m.dest, m.size);
            try b.appendSlice(a, ", ");
            try srcText(a, b, m.src, m.size);
        },
        .movzx => |m| {
            try b.appendSlice(a, "movzx ");
            try b.appendSlice(a, regName(m.dest_size, @backingInt(m.dest)));
            try b.appendSlice(a, ", ");
            switch (m.src) {
                .reg => |r| try b.appendSlice(a, regName(m.src_size, @backingInt(r))),
                .mem => |mem| try memText(a, b, mem, m.src_size),
                .imm => |i| try immText(a, b, i),
            }
        },
        .movsx => |m| {
            try b.appendSlice(a, "movsx ");
            try b.appendSlice(a, regName(m.dest_size, @backingInt(m.dest)));
            try b.appendSlice(a, ", ");
            switch (m.src) {
                .reg => |r| try b.appendSlice(a, regName(m.src_size, @backingInt(r))),
                .mem => |mem| try memText(a, b, mem, m.src_size),
                .imm => |i| try immText(a, b, i),
            }
        },
        .movsxd => |m| {
            try b.appendSlice(a, "movsxd ");
            try b.appendSlice(a, gpr[@backingInt(m.dest)]);
            try b.appendSlice(a, ", ");
            switch (m.src) {
                .reg => |r| try b.appendSlice(a, gpr32[@backingInt(r)]),
                .mem => |mem| try memText(a, b, mem, .dword),
                .imm => |i| try immText(a, b, i),
            }
        },
        .lea => |m| {
            try b.appendSlice(a, "lea ");
            try b.appendSlice(a, gpr[@backingInt(m.dest)]);
            try b.appendSlice(a, ", ");
            try memText(a, b, m.mem, null);
        },

        .push => |m| {
            try b.appendSlice(a, "push ");
            try srcText(a, b, m.src, m.size);
        },
        .pop => |m| {
            try b.appendSlice(a, "pop ");
            try destText(a, b, m.dest, m.size);
        },

        .alu => |m| {
            try b.print(a, "{s} ", .{aluName(m.op)});
            try destText(a, b, m.dest, m.size);
            try b.appendSlice(a, ", ");
            try srcText(a, b, m.src, m.size);
        },
        .unary => |m| {
            try b.print(a, "{s} ", .{unaryName(m.op)});
            try destText(a, b, m.dest, m.size);
        },
        .incdec => |m| {
            try b.appendSlice(a, if (m.op == .inc) "inc " else "dec ");
            try destText(a, b, m.dest, m.size);
        },
        .shift => |m| {
            try b.print(a, "{s} ", .{shiftName(m.op)});
            try destText(a, b, m.dest, m.size);
            try b.appendSlice(a, ", ");
            switch (m.count) {
                .imm => |n| try b.print(a, "{d}", .{n}),
                .cl => try b.appendSlice(a, "cl"),
            }
        },
        .imul => |m| {
            try b.appendSlice(a, "imul ");
            try b.appendSlice(a, regName(m.size, @backingInt(m.dest)));
            try b.appendSlice(a, ", ");
            switch (m.src) {
                .reg => |r| try b.appendSlice(a, regName(m.size, @backingInt(r))),
                .mem => |mem| try memText(a, b, mem, m.size),
                .imm => |i| try immText(a, b, i),
            }
            if (m.imm) |i| {
                try b.appendSlice(a, ", ");
                try immText(a, b, i);
            }
        },

        .jmp => |m| {
            try b.appendSlice(a, "jmp ");
            try targetText(a, b, m.target);
        },
        .jcc => |m| {
            try b.print(a, "j{s} ", .{condName(m.cond)});
            try b.print(a, ".{s}{d}", .{ relSign(m.rel), @abs(m.rel) });
        },
        .call => |m| {
            try b.appendSlice(a, "call ");
            try targetText(a, b, m.target);
        },
        .setcc => |m| {
            try b.print(a, "set{s} ", .{condName(m.cond)});
            try destText(a, b, m.dest, .byte);
        },
        .cmov => |m| {
            try b.print(a, "cmov{s} ", .{condName(m.cond)});
            try b.appendSlice(a, regName(m.size, @backingInt(m.dest)));
            try b.appendSlice(a, ", ");
            switch (m.src) {
                .reg => |r| try b.appendSlice(a, regName(m.size, @backingInt(r))),
                .mem => |mem| try memText(a, b, mem, m.size),
                .imm => |i| try immText(a, b, i),
            }
        },

        .io_in => |m| {
            try b.appendSlice(a, "in ");
            try b.appendSlice(a, regName(m.size, 0));
            try b.appendSlice(a, ", ");
            try portText(a, b, m.port);
        },
        .io_out => |m| {
            try b.appendSlice(a, "out ");
            try portText(a, b, m.port);
            try b.appendSlice(a, ", ");
            try b.appendSlice(a, regName(m.size, 0));
        },

        .movd => |m| {
            try b.appendSlice(a, if (m.wide) "movq " else "movd ");
            if (m.to_xmm) {
                try b.print(a, "xmm{d}, ", .{@backingInt(m.xmm)});
                try b.appendSlice(a, regNameWide(m.wide, @backingInt(m.gpr)));
            } else {
                try b.print(a, "{s}, ", .{regNameWide(m.wide, @backingInt(m.gpr))});
                try b.print(a, "xmm{d}", .{@backingInt(m.xmm)});
            }
        },
        .cvt_int_to_float => |m| {
            try b.appendSlice(a, if (m.double) "cvtsi2sd " else "cvtsi2ss ");
            try b.print(a, "xmm{d}, ", .{@backingInt(m.xmm)});
            try b.appendSlice(a, regNameWide(m.wide, @backingInt(m.gpr)));
        },
        .cvt_float_to_int => |m| {
            try b.appendSlice(a, if (m.double) "cvttsd2si " else "cvttss2si ");
            try b.appendSlice(a, regNameWide(m.wide, @backingInt(m.gpr)));
            try b.print(a, ", xmm{d}", .{@backingInt(m.xmm)});
        },
        .pinsrd => |m| {
            try b.appendSlice(a, "pinsrd ");
            try b.print(a, "xmm{d}, ", .{@backingInt(m.xmm)});
            try b.appendSlice(a, regNameWide(m.wide, @backingInt(m.gpr)));
            try b.print(a, ", {d}", .{m.imm});
        },
        .sse_shift_imm => |m| {
            try b.print(a, "{s} xmm{d}, {d}", .{ decoder.sseShiftName(m.digit), @backingInt(m.xmm), m.imm });
        },
        .vinsertf128 => |m| {
            try b.print(a, "vinsertf128 ymm{d}, ymm{d}, xmm{d}, {d}", .{
                @backingInt(m.dst),
                @backingInt(m.src1),
                @backingInt(m.src2),
                m.imm,
            });
        },
        .vextractf128 => |m| {
            try b.print(a, "vextractf128 xmm{d}, ymm{d}, {d}", .{ @backingInt(m.dst), @backingInt(m.src), m.imm });
        },

        .sse => |m| {
            try b.print(a, "{s} ", .{sseMnemonic(m)});
            if (m.store) {
                try sseRmText(a, b, m.rm, m.hint);
                try b.appendSlice(a, ", ");
                try b.print(a, "xmm{d}", .{@backingInt(m.reg)});
            } else {
                try b.print(a, "xmm{d}, ", .{@backingInt(m.reg)});
                try sseRmText(a, b, m.rm, m.hint);
            }
            if (m.imm) |imm| try b.print(a, ", {d}", .{imm});
        },
        .vex => |m| {
            try b.print(a, "{s} ", .{decoder.vexName(m.op)});
            if (m.store) {
                try vexRmText(a, b, m.src2);
                try b.appendSlice(a, ", ");
                try b.print(a, "ymm{d}", .{@backingInt(m.dst)});
            } else {
                try b.print(a, "ymm{d}", .{@backingInt(m.dst)});
                // The three-operand forms carry their first source in vvvv; the two-operand
                // vmovups does not.
                if (m.op != 0x10 and m.op != 0x11) {
                    try b.appendSlice(a, ", ");
                    try b.print(a, "ymm{d}", .{@backingInt(m.src1.?)});
                }
                try b.appendSlice(a, ", ");
                try vexRmText(a, b, m.src2);
            }
        },

        .unknown => |u| {
            try b.appendSlice(a, ".byte 0x");
            for (u.bytes[0..u.len]) |byte| try b.print(a, "{x:0>2}", .{byte});
        },
    }
}

fn sseMnemonic(m: decoder.Sse) []const u8 {
    return switch (m.esc) {
        0x3A => switch (m.op) {
            0x21 => "insertps",
            0x11 => if (m.mand == 0xF2) "roundsd" else "roundss",
            else => "sse?",
        },
        else => decoder.sseName(m.mand, m.op) orelse "sse?",
    };
}

fn sseRmText(a: std.mem.Allocator, b: *std.ArrayList(u8), rm: decoder.XmmOrMem, hint: ?Size) !void {
    switch (rm) {
        .xmm => |x| try b.print(a, "xmm{d}", .{@backingInt(x)}),
        .mem => |m| try memText(a, b, m, hint),
    }
}

fn vexRmText(a: std.mem.Allocator, b: *std.ArrayList(u8), rm: decoder.XmmOrMem) !void {
    switch (rm) {
        .xmm => |x| try b.print(a, "ymm{d}", .{@backingInt(x)}),
        .mem => |m| try memText(a, b, m, null),
    }
}

fn portText(a: std.mem.Allocator, b: *std.ArrayList(u8), port: decoder.Port) !void {
    switch (port) {
        .dx => try b.appendSlice(a, "dx"),
        .imm => |n| try b.print(a, "{d}", .{n}),
    }
}

/// Decode one instruction; append its text and return its byte length.
fn decode(a: std.mem.Allocator, b: *std.ArrayList(u8), code: []const u8, pos: usize) !usize {
    const d = decoder.decode(code, pos);
    try render(a, b, d.inst);
    return d.len;
}

fn expectOne(inst: encode.Inst, expected: []const u8) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    _ = try decode(std.testing.allocator, &out, inst.slice(), 0);
    try std.testing.expectEqualStrings(expected, out.items);
}

test "round-trips integer ops" {
    // These round-trip the 64-bit (REX.W) forms, so each width-parameterized encoder is called
    // with w = true to match the 64-bit register names in the expected disassembly.
    try expectOne(encode.movImm(.rax, 42, true), "mov rax, 42");
    try expectOne(encode.movReg(.rax, .rbx), "mov rax, rbx");
    try expectOne(encode.add(.rax, .rbx, true), "add rax, rbx");
    try expectOne(encode.sub(.rcx, .rdx, true), "sub rcx, rdx");
    try expectOne(encode.xorr(.rax, .rax, true), "xor rax, rax");
    try expectOne(encode.imul(.rax, .rbx, true), "imul rax, rbx");
    try expectOne(encode.cmp(.rsi, .rdi, true), "cmp rsi, rdi");
    try expectOne(encode.aluImm(0, .rax, 5, true), "add rax, 5");
    try expectOne(encode.aluImm(5, .rbx, 7, true), "sub rbx, 7");
    try expectOne(encode.imulImm(.rax, .rbx, 3, true), "imul rax, rbx, 3");
    try expectOne(encode.shiftImm(4, .rax, 2, true), "shl rax, 2");
    try expectOne(encode.shrCl(.rbx, true), "shr rbx, cl");
    try expectOne(encode.idiv(.rcx, true), "idiv rcx");
    try expectOne(encode.cqo(), "cqo");
    try expectOne(encode.movImm64(.r8, 0x1234), "mov r8, 4660");
    try expectOne(encode.ret(), "ret");
    try expectOne(encode.syscall(), "syscall");
}

test "round-trips flags, branches, memory" {
    try expectOne(encode.setcc(.rax, .l), "setl al");
    try expectOne(encode.cmovcc(.rax, .rbx, .ne), "cmovne rax, rbx");
    try expectOne(encode.movzxByte(.rax, .rbx), "movzx rax, bl");
    try expectOne(encode.jcc(.e, 16), "je .+16");
    try expectOne(encode.jmp(-8), "jmp .-8");
    try expectOne(encode.callRel(32), "call .+32");
    try expectOne(encode.movFromStack(.rax, 8), "mov rax, qword ptr [rsp + 8]");
    try expectOne(encode.movToStack(16, .rbx), "mov qword ptr [rsp + 16], rbx");
    try expectOne(encode.movFromMem(.rax, .rbx, 4), "mov rax, qword ptr [rbx + 4]");
    try expectOne(encode.leaFromStack(.rax, 24), "lea rax, [rsp + 24]");
}

test "round-trips SSE and AVX" {
    try expectOne(encode.addss(.xmm0, .xmm1), "addss xmm0, xmm1");
    try expectOne(encode.divsd(.xmm2, .xmm3), "divsd xmm2, xmm3");
    try expectOne(encode.movdToXmm(.xmm0, .rax), "movd xmm0, eax");
    try expectOne(encode.movdFromXmm(.rax, .xmm0), "movd eax, xmm0");
    try expectOne(encode.cvtsi2ss(.xmm0, .rax), "cvtsi2ss xmm0, eax");
    try expectOne(encode.cvttss2si(.rax, .xmm0), "cvttss2si eax, xmm0");
    try expectOne(encode.sqrtss(.xmm0, .xmm1), "sqrtss xmm0, xmm1");
    try expectOne(encode.addps(.xmm0, .xmm1), "addps xmm0, xmm1");
    try expectOne(encode.ucomiss(.xmm0, .xmm1), "ucomiss xmm0, xmm1");
    try expectOne(encode.pshufd(.xmm0, .xmm1, 2), "pshufd xmm0, xmm1, 2");
    try expectOne(encode.insertps(.xmm0, .xmm1, 0x10), "insertps xmm0, xmm1, 16");
    try expectOne(encode.pinsrd(.xmm0, .rax, 1), "pinsrd xmm0, eax, 1");
    try expectOne(encode.movssStore(8, .xmm1), "movss dword ptr [rsp + 8], xmm1");
    try expectOne(encode.vaddps(.xmm0, .xmm1, .xmm2), "vaddps ymm0, ymm1, ymm2");
    try expectOne(encode.vmovupsRR(.xmm0, .xmm1), "vmovups ymm0, ymm1");
    try expectOne(encode.vextractf128(.xmm0, .xmm1, 1), "vextractf128 xmm0, ymm1, 1");
}

test "formatModuleWithLines interleaves source-line markers before instructions" {
    const a = std.testing.allocator;
    // nop (0x90) at offset 0, ret (0xC3) at offset 1; line 5 then line 6.
    const code = [_]u8{ 0x90, 0xC3 };
    const lines = [_]AddrLine{ .{ .addr = 0, .line = 5 }, .{ .addr = 1, .line = 6 } };
    const out = try formatModuleWithLines(a, &code, &.{}, &lines);
    defer a.free(out);
    const at5 = std.mem.indexOf(u8, out, "; line 5") orelse return error.NoLine5;
    const at6 = std.mem.indexOf(u8, out, "; line 6") orelse return error.NoLine6;
    try std.testing.expect(at5 < at6); // line 5 marks the nop, line 6 marks the ret
    // plain formatModule (no lines) omits the markers.
    const plain = try formatModule(a, &code, &.{});
    defer a.free(plain);
    try std.testing.expect(std.mem.indexOf(u8, plain, "; line") == null);
}

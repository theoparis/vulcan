//! x86-64 instruction decoder: the typed form of the instruction stream.
//!
//! Unlike the fixed-width RISC backends, x86 is variable length, so the decoder parses
//! the real instruction structure (legacy/mandatory prefix, REX or 3-byte VEX,
//! 0F/0F38/0F3A opcode escape, ModRM+SIB+displacement, immediate) into a typed
//! instruction. A consumer can then render the instruction (`disasm.zig`) or execute it
//! without re-parsing the encoding.
//!
//! An opcode the decoder does not model decodes to `unknown`, which carries the raw
//! bytes so a disassembler can still print them and a consumer can refuse the
//! instruction.

const std = @import("std");
const encode = @import("encode.zig");

/// A 64-bit general register.
pub const Reg = encode.Reg;
/// A vector register (xmm0..xmm15).
pub const Xmm = encode.Xmm;

pub const Error = error{TruncatedInstruction};

/// The size of an operand, in bytes.
pub const Size = enum(u8) {
    byte = 1,
    word = 2,
    dword = 4,
    qword = 8,
    xmmword = 16,
    ymmword = 32,
};

/// A condition code, as encoded in the low nibble of a condition opcode.
pub const Cond = enum(u4) {
    o = 0,
    no = 1,
    b = 2,
    ae = 3,
    e = 4,
    ne = 5,
    be = 6,
    a = 7,
    s = 8,
    ns = 9,
    p = 10,
    np = 11,
    l = 12,
    ge = 13,
    le = 14,
    g = 15,
};

/// The arithmetic and logic operations, as the `/digit` of an opcode group selects them.
pub const AluOp = enum {
    add,
    @"or",
    adc,
    sbb,
    @"and",
    sub,
    xor,
    cmp,
    @"test",
};

/// The one-operand operations of the F6/F7 group.
pub const UnaryOp = enum {
    not,
    neg,
    mul,
    imul,
    div,
    idiv,
};

/// The increment and decrement of the FE/FF group.
pub const IncDecOp = enum { inc, dec };

/// The shifts and rotates.
pub const ShiftOp = enum { rol, ror, rcl, rcr, shl, shr, sar };

/// A memory operand: a base register, an optional index register with a scale, and a
/// displacement.
pub const Mem = struct {
    base: ?Reg = null,
    index: ?Reg = null,
    scale: u8 = 1,
    disp: i32 = 0,
    /// true when the address is relative to the next instruction (mod == 0, rm == 5).
    rip: bool = false,
};

/// The destination of an instruction: a register or a memory operand.
pub const Dest = union(enum) {
    reg: Reg,
    mem: Mem,
};

/// A source operand: a register, a memory operand, or an immediate.
pub const Src = union(enum) {
    reg: Reg,
    mem: Mem,
    imm: Imm,
};

/// An immediate operand: the raw value as encoded, the width it occupies, and whether
/// the processor sign-extends it to the operand size.
pub const Imm = struct {
    value: u64,
    /// the number of bits the immediate occupies in the encoding.
    bits: u8,
    /// true when the processor sign-extends the immediate to the operand size.
    sign_extended: bool,

    /// The value as the processor sees it, sign- or zero-extended to 64 bits.
    pub fn extended(self: Imm) u64 {
        if (!self.sign_extended) return self.value;
        return switch (self.bits) {
            8 => @as(u64, @bitCast(@as(i8, @truncate(self.value)))),
            16 => @as(u64, @bitCast(@as(i16, @truncate(self.value)))),
            32 => @as(u64, @bitCast(@as(i32, @truncate(self.value)))),
            else => self.value,
        };
    }
};

/// Where a branch goes: a displacement from the end of the instruction, a register, or
/// a memory operand.
pub const Target = union(enum) {
    rel: i32,
    reg: Reg,
    mem: Mem,
};

/// A shift count: an immediate or the count register.
pub const Count = union(enum) {
    imm: u8,
    cl,
};

/// An I/O port: a port number or the port register.
pub const Port = union(enum) {
    dx,
    imm: u8,
};

/// The rm operand of a vector instruction: a vector register or a memory operand.
pub const XmmOrMem = union(enum) {
    xmm: Xmm,
    mem: Mem,
};

/// A vector instruction of the SSE family. The mnemonic depends on the mandatory
/// prefix, so it is looked up from the opcode rather than stored.
pub const Sse = struct {
    /// the opcode escape the instruction came from: 0x0F or 0x3A.
    esc: u8,
    /// the opcode below the escape.
    op: u8,
    /// the mandatory prefix: 0x66, 0xF2, 0xF3, or 0 for none.
    mand: u8,
    /// the destination (ModRM.reg).
    reg: Xmm,
    /// the source (ModRM.rm): a vector register or a memory operand.
    rm: XmmOrMem,
    /// true when the memory form stores: the memory operand comes first in the text.
    store: bool = false,
    /// the size of a memory operand, or null when the operand carries no size hint.
    hint: ?Size = null,
    /// the immediate, when the instruction carries one.
    imm: ?u8 = null,
};

/// A VEX-encoded instruction of the 0F map: the arithmetic and move forms.
pub const Vex = struct {
    /// the opcode below the VEX escape.
    op: u8,
    /// the destination (ModRM.reg).
    dst: Xmm,
    /// the first source (VEX.vvvv), when the instruction has one.
    src1: ?Xmm,
    /// the second source (ModRM.rm): a vector register or a memory operand.
    src2: XmmOrMem,
    /// true when the memory form stores: the memory operand comes first in the text.
    store: bool = false,
};

/// A decoded instruction.
pub const Instruction = union(enum) {
    nop,
    pause,
    hlt,
    cli,
    sti,
    cld,
    std,
    ret,
    syscall,
    /// `endbr64` and the other reserved-nop hints.
    endbr64,
    /// Widen the accumulator: `cdq`, `cqo`, `cwde` or `cdqe`.
    widen: enum { cdq, cqo, cwde, cdqe },

    mov: struct { dest: Dest, src: Src, size: Size },
    movzx: struct { dest: Reg, src: Src, src_size: Size, dest_size: Size },
    movsx: struct { dest: Reg, src: Src, src_size: Size, dest_size: Size },
    /// `movsxd r64, r/m32`.
    movsxd: struct { dest: Reg, src: Src },
    lea: struct { dest: Reg, mem: Mem },

    push: struct { src: Src, size: Size },
    pop: struct { dest: Dest, size: Size },

    alu: struct { op: AluOp, dest: Dest, src: Src, size: Size },
    unary: struct { op: UnaryOp, dest: Dest, size: Size },
    incdec: struct { op: IncDecOp, dest: Dest, size: Size },
    shift: struct { op: ShiftOp, dest: Dest, count: Count, size: Size },
    imul: struct { dest: Reg, src: Src, imm: ?Imm, size: Size },

    jmp: struct { target: Target },
    jcc: struct { cond: Cond, rel: i32 },
    call: struct { target: Target },
    setcc: struct { cond: Cond, dest: Dest },
    cmov: struct { cond: Cond, dest: Reg, src: Src, size: Size },

    /// An I/O port read or write.
    io_in: struct { port: Port, size: Size },
    io_out: struct { port: Port, size: Size },

    /// `movd`/`movq` between a vector register and a general register.
    movd: struct { to_xmm: bool, xmm: Xmm, gpr: Reg, wide: bool },
    /// `cvtsi2ss`/`cvtsi2sd`: an integer in a general register to a scalar float.
    cvt_int_to_float: struct { xmm: Xmm, gpr: Reg, double: bool, wide: bool },
    /// `cvttss2si`/`cvttsd2si`: a scalar float to an integer in a general register.
    cvt_float_to_int: struct { gpr: Reg, xmm: Xmm, double: bool, wide: bool },
    /// `pinsrd`: a general register into a lane of a vector register.
    pinsrd: struct { xmm: Xmm, gpr: Reg, imm: u8, wide: bool },
    /// The shift-by-immediate group: `psrlq xmm, imm8` and its siblings.
    sse_shift_imm: struct { digit: u8, xmm: Xmm, imm: u8 },
    /// `vinsertf128`: two 128-bit lanes into one 256-bit register.
    vinsertf128: struct { dst: Xmm, src1: Xmm, src2: Xmm, imm: u8 },
    /// `vextractf128`: one 128-bit lane out of a 256-bit register.
    vextractf128: struct { dst: Xmm, src: Xmm, imm: u8 },

    sse: Sse,
    vex: Vex,

    /// An opcode the decoder does not model: the raw bytes, so a disassembler can print
    /// them and a consumer can refuse the instruction.
    unknown: struct { bytes: [4]u8, len: u3 },
};

/// A decoded instruction and its length in bytes.
pub const Decoded = struct {
    inst: Instruction,
    /// the length of the instruction in bytes.
    len: usize,
    /// true when the instruction ran off the end of the buffer.
    truncated: bool = false,
};

/// The decoder cursor and the prefix state it has collected.
const Cursor = struct {
    code: []const u8,
    i: usize,
    /// the mandatory prefix: 0x66, 0xF2 or 0xF3, or 0 for none.
    mand: u8 = 0,
    /// true when the REX prefix asks for 64-bit operands.
    rexW: bool = false,
    /// the extension bits of the ModRM.reg, SIB.index and ModRM.rm fields.
    R: u8 = 0,
    X: u8 = 0,
    B: u8 = 0,
    /// true when the instruction is VEX-encoded.
    vex: bool = false,
    /// the VEX source-1 register.
    vvvv: u8 = 0,
    /// the opcode map: 0 (one byte), 0x0F, 0x38 or 0x3A.
    esc: u16 = 0,
    /// true when the instruction ran off the end of the buffer.
    truncated: bool = false,

    /// Read one byte. Past the end of the buffer the byte is zero, which keeps the
    /// length of a truncated instruction the same as the length it would have had.
    fn byte(c: *Cursor) u8 {
        if (c.i < c.code.len) {
            const b = c.code[c.i];
            c.i += 1;
            return b;
        }
        c.truncated = true;
        c.i += 1;
        return 0;
    }

    fn peek(c: *const Cursor) u8 {
        return if (c.i < c.code.len) c.code[c.i] else 0;
    }

    fn imm8(c: *Cursor) u8 {
        return c.byte();
    }

    fn imm16(c: *Cursor) u16 {
        const lo = c.byte();
        const hi = c.byte();
        return @as(u16, lo) | (@as(u16, hi) << 8);
    }

    fn imm32(c: *Cursor) u32 {
        var v: u32 = 0;
        inline for (0..4) |k| v |= @as(u32, c.byte()) << (k * 8);
        return v;
    }

    fn imm64(c: *Cursor) u64 {
        var v: u64 = 0;
        inline for (0..8) |k| v |= @as(u64, c.byte()) << (k * 8);
        return v;
    }

    /// The operand size: 64-bit with REX.W, 16-bit with a 0x66 prefix, else 32-bit
    /// (the 64-bit-mode default).
    fn opSize(c: *const Cursor) Size {
        if (c.rexW) return .qword;
        if (c.mand == 0x66) return .word;
        return .dword;
    }
};

/// The ModRM operands: the register field, the rm field, and the memory operand when
/// the rm field is not a register.
const Operands = struct {
    reg: u8,
    rm: u8,
    mem: bool,
    mem_op: Mem,
};

fn modrm(c: *Cursor) Operands {
    const m = c.byte();
    const mod = m >> 6;
    const reg = (c.R << 3) | ((m >> 3) & 7);
    const rm_low = m & 7;

    if (mod == 3) {
        return .{ .reg = reg, .rm = (c.B << 3) | rm_low, .mem = false, .mem_op = .{} };
    }

    var mem = Mem{};
    if (rm_low == 4) {
        // A SIB byte: a base register, an index register with a scale, and a
        // displacement.
        const sib = c.byte();
        const base = (c.B << 3) | (sib & 7);
        const idx = (c.X << 3) | ((sib >> 3) & 7);
        if (idx != 4) {
            // An index of 100 without REX.X means "no index register".
            mem.index = @fromBackingInt(@intCast(idx & 0xF));
            mem.scale = @as(u8, 1) << @intCast(sib >> 6);
        }
        if (mod == 0 and (sib & 7) == 5) {
            // A base of 101 with no base register: the displacement alone.
            mem.disp = @bitCast(c.imm32());
        } else {
            mem.base = @fromBackingInt(@intCast(base & 0xF));
        }
    } else if (mod == 0 and rm_low == 5) {
        // No base register: the displacement is relative to the next instruction.
        mem.rip = true;
        mem.disp = @bitCast(c.imm32());
    } else {
        mem.base = @fromBackingInt(@intCast((c.B << 3) | rm_low));
    }

    switch (mod) {
        1 => mem.disp = @as(i8, @bitCast(c.byte())),
        2 => mem.disp = @bitCast(c.imm32()),
        else => {},
    }

    return .{ .reg = reg, .rm = (c.B << 3) | rm_low, .mem = true, .mem_op = mem };
}

fn destOf(o: Operands) Dest {
    if (o.mem) return .{ .mem = o.mem_op };
    return .{ .reg = @fromBackingInt(@intCast(o.rm & 0xF)) };
}

fn srcOf(o: Operands) Src {
    if (o.mem) return .{ .mem = o.mem_op };
    return .{ .reg = @fromBackingInt(@intCast(o.rm & 0xF)) };
}

fn xmmOf(o: Operands) Xmm {
    return @fromBackingInt(@intCast(o.reg & 0xF));
}

fn rmXmmOrMem(o: Operands) XmmOrMem {
    if (o.mem) return .{ .mem = o.mem_op };
    return .{ .xmm = @fromBackingInt(@intCast(o.rm & 0xF)) };
}

/// The arithmetic and logic operation an opcode group digit selects.
fn aluOp(digit: u8) AluOp {
    return switch (digit & 7) {
        0 => .add,
        1 => .@"or",
        2 => .adc,
        3 => .sbb,
        4 => .@"and",
        5 => .sub,
        6 => .xor,
        else => .cmp,
    };
}

/// The shift an opcode group digit selects.
fn shiftOp(digit: u8) ShiftOp {
    return switch (digit & 7) {
        0 => .rol,
        1 => .ror,
        2 => .rcl,
        3 => .rcr,
        4 => .shl,
        5 => .shr,
        else => .sar,
    };
}

/// Decode the instruction at `pos` in `bytes`.
///
/// Reading past the end of the buffer yields zero bytes and sets `truncated`, so a
/// disassembler can still render a partial instruction.
pub fn decode(bytes: []const u8, pos: usize) Decoded {
    var c = Cursor{ .code = bytes, .i = pos };
    const inst = decodeAt(&c);
    return .{ .inst = inst, .len = c.i - pos, .truncated = c.truncated };
}

/// Decode the instruction at `pos` in `bytes`, refusing one that runs off the end.
pub fn decodeStrict(bytes: []const u8, pos: usize) Error!Decoded {
    const d = decode(bytes, pos);
    if (d.truncated) return error.TruncatedInstruction;
    return d;
}

fn decodeAt(c: *Cursor) Instruction {
    // Legacy prefixes: 0x66/0xF2/0xF3 are mandatory or operand-size prefixes (kept);
    // segment overrides, address-size and lock are consumed so the stream stays in sync.
    while (true) switch (c.peek()) {
        0x66, 0xF2, 0xF3 => c.mand = c.byte(),
        0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65, 0x67, 0xF0 => _ = c.byte(),
        else => break,
    };

    if (c.peek() == 0xC4) {
        // A 3-byte VEX prefix.
        _ = c.byte();
        const byte2 = c.byte();
        const byte3 = c.byte();
        c.vex = true;
        c.R = ((byte2 >> 7) & 1) ^ 1;
        c.X = ((byte2 >> 6) & 1) ^ 1;
        c.B = ((byte2 >> 5) & 1) ^ 1;
        c.esc = switch (byte2 & 0x1F) {
            1 => 0x0F,
            2 => 0x38,
            3 => 0x3A,
            else => 0x0F,
        };
        c.rexW = (byte3 >> 7) & 1 != 0;
        c.vvvv = (~(byte3 >> 3)) & 0xF;
        c.mand = switch (byte3 & 3) {
            1 => 0x66,
            2 => 0xF3,
            3 => 0xF2,
            else => 0,
        };
    } else {
        if (c.peek() & 0xF0 == 0x40) {
            const rex = c.byte();
            c.rexW = rex & 8 != 0;
            c.R = (rex >> 2) & 1;
            c.X = (rex >> 1) & 1;
            c.B = rex & 1;
        }
        if (c.peek() == 0x0F) {
            _ = c.byte();
            c.esc = 0x0F;
            if (c.peek() == 0x38 or c.peek() == 0x3A) c.esc = c.byte();
        }
    }

    const op = c.byte();
    if (c.vex) return decodeVex(c, op);
    if (c.esc == 0x0F) return decodeTwoByte(c, op);
    if (c.esc == 0x3A) return decodeThreeByte(c, op);
    if (c.esc == 0x38) return .{ .unknown = .{ .bytes = .{ 0x0F, 0x38, op, 0 }, .len = 3 } };
    return decodeOneByte(c, op);
}

fn decodeOneByte(c: *Cursor, op: u8) Instruction {
    switch (op) {
        0x90 => return if (c.mand == 0xF3) .pause else .nop,
        0xF4 => return .hlt,
        0xFA => return .cli,
        0xFB => return .sti,
        0xFC => return .cld,
        0xFD => return .std,
        0xC3 => return .ret,

        // The I/O instructions: the port is an immediate or the port register.
        0xE4 => return .{ .io_in = .{ .port = .{ .imm = c.byte() }, .size = .byte } },
        0xE5 => return .{ .io_in = .{ .port = .{ .imm = c.byte() }, .size = c.opSize() } },
        0xEC => return .{ .io_in = .{ .port = .dx, .size = .byte } },
        0xED => return .{ .io_in = .{ .port = .dx, .size = c.opSize() } },
        0xE6 => return .{ .io_out = .{ .port = .{ .imm = c.byte() }, .size = .byte } },
        0xE7 => return .{ .io_out = .{ .port = .{ .imm = c.byte() }, .size = c.opSize() } },
        0xEE => return .{ .io_out = .{ .port = .dx, .size = .byte } },
        0xEF => return .{ .io_out = .{ .port = .dx, .size = c.opSize() } },

        // The arithmetic and logic group, in its four shapes.
        0x01, 0x09, 0x21, 0x29, 0x31, 0x39 => { // r/m, reg
            const o = modrm(c);
            return .{ .alu = .{ .op = aluOp(op >> 3), .dest = destOf(o), .src = .{ .reg = @fromBackingInt(@intCast(o.reg & 0xF)) }, .size = c.opSize() } };
        },
        0x03, 0x0B, 0x13, 0x1B, 0x23, 0x2B, 0x33, 0x3B => { // reg, r/m
            const o = modrm(c);
            return .{ .alu = .{ .op = aluOp(op >> 3), .dest = .{ .reg = @fromBackingInt(@intCast(o.reg & 0xF)) }, .src = srcOf(o), .size = c.opSize() } };
        },
        0x05, 0x0D, 0x15, 0x1D, 0x25, 0x2D, 0x35, 0x3D => { // al/eax/rax, imm32
            const imm = c.imm32();
            return .{ .alu = .{ .op = aluOp(op >> 3), .dest = .{ .reg = .rax }, .src = .{ .imm = .{ .value = imm, .bits = 32, .sign_extended = true } }, .size = c.opSize() } };
        },
        0x04, 0x0C, 0x14, 0x1C, 0x24, 0x2C, 0x34, 0x3C => { // al, imm8
            const imm = c.byte();
            return .{ .alu = .{ .op = aluOp(op >> 3), .dest = .{ .reg = .rax }, .src = .{ .imm = .{ .value = imm, .bits = 8, .sign_extended = true } }, .size = .byte } };
        },
        0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38 => { // r/m8, r8
            const o = modrm(c);
            return .{ .alu = .{ .op = aluOp(op >> 3), .dest = destOf(o), .src = .{ .reg = @fromBackingInt(@intCast(o.reg & 0xF)) }, .size = .byte } };
        },
        0x02, 0x0A, 0x12, 0x1A, 0x22, 0x2A, 0x32, 0x3A => { // r8, r/m8
            const o = modrm(c);
            return .{ .alu = .{ .op = aluOp(op >> 3), .dest = .{ .reg = @fromBackingInt(@intCast(o.reg & 0xF)) }, .src = srcOf(o), .size = .byte } };
        },
        0x85 => { // test r/m, reg
            const o = modrm(c);
            return .{ .alu = .{ .op = .@"test", .dest = destOf(o), .src = .{ .reg = @fromBackingInt(@intCast(o.reg & 0xF)) }, .size = c.opSize() } };
        },
        0x84 => { // test r/m8, r8
            const o = modrm(c);
            return .{ .alu = .{ .op = .@"test", .dest = destOf(o), .src = .{ .reg = @fromBackingInt(@intCast(o.reg & 0xF)) }, .size = .byte } };
        },
        0x89 => { // mov r/m, reg
            const o = modrm(c);
            return .{ .mov = .{ .dest = destOf(o), .src = .{ .reg = @fromBackingInt(@intCast(o.reg & 0xF)) }, .size = c.opSize() } };
        },
        0x8B => { // mov reg, r/m
            const o = modrm(c);
            return .{ .mov = .{ .dest = .{ .reg = @fromBackingInt(@intCast(o.reg & 0xF)) }, .src = srcOf(o), .size = c.opSize() } };
        },
        0x8A => { // mov r8, r/m8
            const o = modrm(c);
            return .{ .mov = .{ .dest = .{ .reg = @fromBackingInt(@intCast(o.reg & 0xF)) }, .src = srcOf(o), .size = .byte } };
        },
        0x88 => { // mov r/m8, r8
            const o = modrm(c);
            return .{ .mov = .{ .dest = destOf(o), .src = .{ .reg = @fromBackingInt(@intCast(o.reg & 0xF)) }, .size = .byte } };
        },
        0xC7 => { // mov r/m, imm32
            const o = modrm(c);
            const imm = c.imm32();
            return .{ .mov = .{ .dest = destOf(o), .src = .{ .imm = .{ .value = imm, .bits = 32, .sign_extended = false } }, .size = c.opSize() } };
        },
        0xC6 => { // mov r/m8, imm8
            const o = modrm(c);
            const imm = c.byte();
            return .{ .mov = .{ .dest = destOf(o), .src = .{ .imm = .{ .value = imm, .bits = 8, .sign_extended = false } }, .size = .byte } };
        },
        0x8D => { // lea reg, mem
            const o = modrm(c);
            if (!o.mem) return .{ .unknown = .{ .bytes = .{ op, 0, 0, 0 }, .len = 1 } };
            return .{ .lea = .{ .dest = @fromBackingInt(@intCast(o.reg & 0xF)), .mem = o.mem_op } };
        },
        0x63 => { // movsxd reg, r/m32
            const o = modrm(c);
            return .{ .movsxd = .{ .dest = @fromBackingInt(@intCast(o.reg & 0xF)), .src = srcOf(o) } };
        },

        // The moves of an immediate into a register.
        0xB8...0xBF => {
            const r = (c.B << 3) | (op & 7);
            if (c.rexW) {
                const imm = c.imm64();
                return .{ .mov = .{ .dest = .{ .reg = @fromBackingInt(@intCast(r & 0xF)) }, .src = .{ .imm = .{ .value = imm, .bits = 64, .sign_extended = false } }, .size = .qword } };
            }
            const imm = c.imm32();
            return .{ .mov = .{ .dest = .{ .reg = @fromBackingInt(@intCast(r & 0xF)) }, .src = .{ .imm = .{ .value = imm, .bits = 32, .sign_extended = false } }, .size = .dword } };
        },

        0x50...0x57 => { // push reg
            const r = (c.B << 3) | (op & 7);
            return .{ .push = .{ .src = .{ .reg = @fromBackingInt(@intCast(r & 0xF)) }, .size = .qword } };
        },
        0x58...0x5F => { // pop reg
            const r = (c.B << 3) | (op & 7);
            return .{ .pop = .{ .dest = .{ .reg = @fromBackingInt(@intCast(r & 0xF)) }, .size = .qword } };
        },
        0x68 => { // push imm32
            const imm = c.imm32();
            return .{ .push = .{ .src = .{ .imm = .{ .value = imm, .bits = 32, .sign_extended = false } }, .size = .qword } };
        },
        0x6A => { // push imm8
            const imm = c.byte();
            return .{ .push = .{ .src = .{ .imm = .{ .value = imm, .bits = 8, .sign_extended = true } }, .size = .qword } };
        },

        // The arithmetic and logic group with an immediate.
        0x81 => { // r/m, imm32
            const o = modrm(c);
            const imm = c.imm32();
            return .{ .alu = .{ .op = aluOp(o.reg), .dest = destOf(o), .src = .{ .imm = .{ .value = imm, .bits = 32, .sign_extended = true } }, .size = c.opSize() } };
        },
        0x83 => { // r/m, imm8 (sign-extended)
            const o = modrm(c);
            const imm = c.byte();
            return .{ .alu = .{ .op = aluOp(o.reg), .dest = destOf(o), .src = .{ .imm = .{ .value = imm, .bits = 8, .sign_extended = true } }, .size = c.opSize() } };
        },
        0x80 => { // r/m8, imm8
            const o = modrm(c);
            const imm = c.byte();
            return .{ .alu = .{ .op = aluOp(o.reg), .dest = destOf(o), .src = .{ .imm = .{ .value = imm, .bits = 8, .sign_extended = false } }, .size = .byte } };
        },

        0x69 => { // imul reg, r/m, imm32
            const o = modrm(c);
            const imm = c.imm32();
            return .{ .imul = .{ .dest = @fromBackingInt(@intCast(o.reg & 0xF)), .src = srcOf(o), .imm = .{ .value = imm, .bits = 32, .sign_extended = true }, .size = c.opSize() } };
        },
        0x6B => { // imul reg, r/m, imm8
            const o = modrm(c);
            const imm = c.byte();
            return .{ .imul = .{ .dest = @fromBackingInt(@intCast(o.reg & 0xF)), .src = srcOf(o), .imm = .{ .value = imm, .bits = 8, .sign_extended = true }, .size = c.opSize() } };
        },

        // The shifts and rotates.
        0xC0, 0xC1 => { // r/m, imm8
            const o = modrm(c);
            const imm = c.byte();
            const size: Size = if (op == 0xC0) .byte else c.opSize();
            return .{ .shift = .{ .op = shiftOp(o.reg), .dest = destOf(o), .count = .{ .imm = imm }, .size = size } };
        },
        0xD0, 0xD1 => { // r/m, 1
            const o = modrm(c);
            const size: Size = if (op == 0xD0) .byte else c.opSize();
            return .{ .shift = .{ .op = shiftOp(o.reg), .dest = destOf(o), .count = .{ .imm = 1 }, .size = size } };
        },
        0xD2, 0xD3 => { // r/m, cl
            const o = modrm(c);
            const size: Size = if (op == 0xD2) .byte else c.opSize();
            return .{ .shift = .{ .op = shiftOp(o.reg), .dest = destOf(o), .count = .cl, .size = size } };
        },

        // The F6/F7 group: test, not, neg, mul, imul, div and idiv.
        0xF6, 0xF7 => {
            const o = modrm(c);
            const size: Size = if (op == 0xF6) .byte else c.opSize();
            switch (o.reg & 7) {
                0, 1 => {
                    const imm = c.imm32();
                    return .{ .alu = .{ .op = .@"test", .dest = destOf(o), .src = .{ .imm = .{ .value = imm, .bits = 32, .sign_extended = false } }, .size = size } };
                },
                2 => return .{ .unary = .{ .op = .not, .dest = destOf(o), .size = size } },
                3 => return .{ .unary = .{ .op = .neg, .dest = destOf(o), .size = size } },
                4 => return .{ .unary = .{ .op = .mul, .dest = destOf(o), .size = size } },
                5 => return .{ .unary = .{ .op = .imul, .dest = destOf(o), .size = size } },
                6 => return .{ .unary = .{ .op = .div, .dest = destOf(o), .size = size } },
                else => return .{ .unary = .{ .op = .idiv, .dest = destOf(o), .size = size } },
            }
        },

        // The FE group: inc and dec of a byte.
        0xFE => {
            const o = modrm(c);
            const op_name: IncDecOp = if (o.reg & 7 == 0) .inc else .dec;
            return .{ .incdec = .{ .op = op_name, .dest = destOf(o), .size = .byte } };
        },

        // The FF group: inc, dec, call, jmp and push.
        0xFF => {
            const o = modrm(c);
            switch (o.reg & 7) {
                0 => return .{ .incdec = .{ .op = .inc, .dest = destOf(o), .size = c.opSize() } },
                1 => return .{ .incdec = .{ .op = .dec, .dest = destOf(o), .size = c.opSize() } },
                2 => return .{ .call = .{ .target = targetOf(o) } },
                4 => return .{ .jmp = .{ .target = targetOf(o) } },
                6 => return .{ .push = .{ .src = srcOf(o), .size = .qword } },
                else => return .{ .unknown = .{ .bytes = .{ op, 0, 0, 0 }, .len = 1 } },
            }
        },

        0x98 => return .{ .widen = if (c.rexW) .cdqe else .cwde },
        0x99 => return .{ .widen = if (c.rexW) .cqo else .cdq },

        0xE8 => { // call rel32
            const rel = c.imm32();
            return .{ .call = .{ .target = .{ .rel = @bitCast(rel) } } };
        },
        0xE9 => { // jmp rel32
            const rel = c.imm32();
            return .{ .jmp = .{ .target = .{ .rel = @bitCast(rel) } } };
        },
        0xEB => { // jmp rel8
            const rel = c.byte();
            return .{ .jmp = .{ .target = .{ .rel = @as(i8, @bitCast(rel)) } } };
        },
        0x70...0x7F => { // jcc rel8
            const rel = c.byte();
            return .{ .jcc = .{ .cond = @fromBackingInt(@intCast(op & 0xF)), .rel = @as(i8, @bitCast(rel)) } };
        },

        0xA8 => { // test al, imm8
            const imm = c.byte();
            return .{ .alu = .{ .op = .@"test", .dest = .{ .reg = .rax }, .src = .{ .imm = .{ .value = imm, .bits = 8, .sign_extended = false } }, .size = .byte } };
        },
        0xA9 => { // test r/m, imm32
            const o = modrm(c);
            const imm = c.imm32();
            return .{ .alu = .{ .op = .@"test", .dest = destOf(o), .src = .{ .imm = .{ .value = imm, .bits = 32, .sign_extended = true } }, .size = c.opSize() } };
        },

        else => return .{ .unknown = .{ .bytes = .{ op, 0, 0, 0 }, .len = 1 } },
    }
}

/// The branch target of a register or memory operand.
fn targetOf(o: Operands) Target {
    if (o.mem) return .{ .mem = o.mem_op };
    return .{ .reg = @fromBackingInt(@intCast(o.rm & 0xF)) };
}

fn decodeTwoByte(c: *Cursor, op: u8) Instruction {
    switch (op) {
        0x05 => return .syscall,
        0x1F => { // A multi-byte nop: consume the ModRM so the stream stays in sync.
            _ = modrm(c);
            return .nop;
        },
        0x1E => { // endbr64 and the other reserved-nop hints.
            _ = c.byte();
            return .endbr64;
        },
        0x2A => { // cvtsi2ss / cvtsi2sd
            const o = modrm(c);
            return .{ .cvt_int_to_float = .{ .xmm = xmmOf(o), .gpr = @fromBackingInt(@intCast(o.rm & 0xF)), .double = c.mand == 0xF2, .wide = c.rexW } };
        },
        0x2C => { // cvttss2si / cvttsd2si
            const o = modrm(c);
            return .{ .cvt_float_to_int = .{ .gpr = @fromBackingInt(@intCast(o.reg & 0xF)), .xmm = @fromBackingInt(@intCast(o.rm & 0xF)), .double = c.mand == 0xF2, .wide = c.rexW } };
        },
        0x40...0x4F => { // cmovcc reg, r/m
            const o = modrm(c);
            return .{ .cmov = .{ .cond = @fromBackingInt(@intCast(op & 0xF)), .dest = @fromBackingInt(@intCast(o.reg & 0xF)), .src = srcOf(o), .size = c.opSize() } };
        },
        0x6E => { // movd / movq to a vector register
            const o = modrm(c);
            return .{ .movd = .{ .to_xmm = true, .xmm = xmmOf(o), .gpr = @fromBackingInt(@intCast(o.rm & 0xF)), .wide = c.rexW } };
        },
        0x7E => { // movd / movq from a vector register
            const o = modrm(c);
            return .{ .movd = .{ .to_xmm = false, .xmm = xmmOf(o), .gpr = @fromBackingInt(@intCast(o.rm & 0xF)), .wide = c.rexW } };
        },
        0x70 => { // pshufd
            const o = modrm(c);
            const imm = c.byte();
            return .{ .sse = .{ .esc = 0x0F, .op = op, .mand = c.mand, .reg = xmmOf(o), .rm = rmXmmOrMem(o), .hint = null, .imm = imm } };
        },
        0x73 => { // psrlq, psrldq, psllq and pslldq
            const o = modrm(c);
            const imm = c.byte();
            return .{ .sse_shift_imm = .{ .digit = o.reg & 7, .xmm = @fromBackingInt(@intCast(o.rm & 0xF)), .imm = imm } };
        },
        0x80...0x8F => { // jcc rel32
            const rel = c.imm32();
            return .{ .jcc = .{ .cond = @fromBackingInt(@intCast(op & 0xF)), .rel = @bitCast(rel) } };
        },
        0x90...0x9F => { // setcc r/m8
            const o = modrm(c);
            return .{ .setcc = .{ .cond = @fromBackingInt(@intCast(op & 0xF)), .dest = destOf(o) } };
        },
        0xAF => { // imul reg, r/m
            const o = modrm(c);
            return .{ .imul = .{ .dest = @fromBackingInt(@intCast(o.reg & 0xF)), .src = srcOf(o), .imm = null, .size = c.opSize() } };
        },
        0xB6, 0xB7 => { // movzx
            const o = modrm(c);
            return .{ .movzx = .{ .dest = @fromBackingInt(@intCast(o.reg & 0xF)), .src = srcOf(o), .src_size = if (op == 0xB7) .word else .byte, .dest_size = c.opSize() } };
        },
        0xBE, 0xBF => { // movsx
            const o = modrm(c);
            return .{ .movsx = .{ .dest = @fromBackingInt(@intCast(o.reg & 0xF)), .src = srcOf(o), .src_size = if (op == 0xBF) .word else .byte, .dest_size = c.opSize() } };
        },
        else => {},
    }

    // The SSE family: the mnemonic depends on the mandatory prefix.
    if (sseName(c.mand, op)) |_| {
        const o = modrm(c);
        return .{ .sse = .{
            .esc = 0x0F,
            .op = op,
            .mand = c.mand,
            .reg = xmmOf(o),
            .rm = rmXmmOrMem(o),
            .store = sseStores(op),
            .hint = sseHint(c.mand, op),
        } };
    }

    return .{ .unknown = .{ .bytes = .{ 0x0F, op, 0, 0 }, .len = 2 } };
}

fn decodeThreeByte(c: *Cursor, op: u8) Instruction {
    const o = modrm(c);
    switch (op) {
        0x22 => { // pinsrd xmm, gpr32, imm8
            const imm = c.byte();
            return .{ .pinsrd = .{ .xmm = xmmOf(o), .gpr = @fromBackingInt(@intCast(o.rm & 0xF)), .imm = imm, .wide = c.rexW } };
        },
        0x21 => { // insertps
            const imm = c.byte();
            return .{ .sse = .{ .esc = 0x3A, .op = op, .mand = c.mand, .reg = xmmOf(o), .rm = rmXmmOrMem(o), .hint = null, .imm = imm } };
        },
        0x11 => { // roundss / roundsd
            const imm = c.byte();
            return .{ .sse = .{ .esc = 0x3A, .op = op, .mand = c.mand, .reg = xmmOf(o), .rm = rmXmmOrMem(o), .hint = null, .imm = imm } };
        },
        else => return .{ .unknown = .{ .bytes = .{ 0x0F, 0x3A, op, 0 }, .len = 3 } },
    }
}

fn decodeVex(c: *Cursor, op: u8) Instruction {
    if (c.esc == 0x3A) {
        const o = modrm(c);
        const imm = c.byte();
        switch (op) {
            0x18 => return .{ .vinsertf128 = .{ .dst = xmmOf(o), .src1 = @fromBackingInt(@intCast(c.vvvv & 0xF)), .src2 = @fromBackingInt(@intCast(o.rm & 0xF)), .imm = imm } },
            0x19 => return .{ .vextractf128 = .{ .dst = @fromBackingInt(@intCast(o.rm & 0xF)), .src = xmmOf(o), .imm = imm } },
            else => return .{ .unknown = .{ .bytes = .{ 0xC4, op, 0, 0 }, .len = 2 } },
        }
    }
    const o = modrm(c);
    return .{ .vex = .{
        .op = op,
        .dst = xmmOf(o),
        .src1 = @fromBackingInt(@intCast(c.vvvv & 0xF)),
        .src2 = rmXmmOrMem(o),
        .store = op == 0x11,
    } };
}

/// The mnemonic of an SSE instruction, or null when the opcode is not one of the SSE
/// forms.
pub fn sseName(mand: u8, op: u8) ?[]const u8 {
    return switch (mand) {
        0xF3 => switch (op) {
            0x10, 0x11 => "movss",
            0x58 => "addss",
            0x59 => "mulss",
            0x5C => "subss",
            0x5E => "divss",
            0x51 => "sqrtss",
            0x5A => "cvtss2sd",
            0x6F, 0x7F => "movdqu",
            else => null,
        },
        0xF2 => switch (op) {
            0x10, 0x11 => "movsd",
            0x58 => "addsd",
            0x59 => "mulsd",
            0x5C => "subsd",
            0x5E => "divsd",
            0x51 => "sqrtsd",
            0x5A => "cvtsd2ss",
            else => null,
        },
        0x66 => switch (op) {
            0x2E => "ucomisd",
            0x70 => "pshufd",
            0x28, 0x29 => "movapd",
            0x6F, 0x7F => "movdqa",
            0xEF => "pxor",
            0xDB => "pand",
            0xEB => "por",
            0xDF => "pandn",
            0xD4 => "paddq",
            0xFB => "psubq",
            0xFE => "paddd",
            0xFA => "psubd",
            0xF4 => "pmuludq",
            0xD5 => "pmullw",
            0xFC => "paddb",
            0xFD => "paddw",
            0xEC => "paddsb",
            0xED => "paddsw",
            0xE2 => "psrad",
            0xD6 => "movq",
            else => null,
        },
        else => switch (op) { // no mandatory prefix: packed / other
            0x10, 0x11 => "movups",
            0x58 => "addps",
            0x59 => "mulps",
            0x5C => "subps",
            0x5E => "divps",
            0x2E => "ucomiss",
            0x28, 0x29 => "movaps",
            else => null,
        },
    };
}

/// True when the memory form of an SSE opcode stores: the memory operand comes first
/// in the text.
pub fn sseStores(op: u8) bool {
    return op == 0x11 or op == 0x29 or op == 0x7F or op == 0xD6;
}

/// The size of an SSE memory operand: a scalar single is a dword, a scalar double is a
/// qword, and everything packed or integer is an xmmword.
pub fn sseHint(mand: u8, op: u8) Size {
    const scalar = switch (op) {
        0x10, 0x11, 0x51, 0x58, 0x59, 0x5A, 0x5C, 0x5E, 0x2A, 0x2C, 0x2E => true,
        else => false,
    };
    if (scalar and mand == 0xF3) return .dword;
    if (scalar and mand == 0xF2) return .qword;
    return .xmmword;
}

/// The mnemonic of the shift-by-immediate group.
pub fn sseShiftName(digit: u8) []const u8 {
    return switch (digit & 7) {
        2 => "psrlq",
        3 => "psrldq",
        6 => "psllq",
        else => "pslldq",
    };
}

/// The mnemonic of a VEX-encoded arithmetic or move instruction.
pub fn vexName(op: u8) []const u8 {
    return switch (op) {
        0x58 => "vaddps",
        0x5C => "vsubps",
        0x59 => "vmulps",
        0x5E => "vdivps",
        0x10, 0x11 => "vmovups",
        else => "vex?",
    };
}

fn expect(inst: Instruction, code: []const u8) !void {
    const d = try decodeStrict(code, 0);
    try std.testing.expectEqual(inst, d.inst);
    try std.testing.expectEqual(code.len, d.len);
}

test "decodes a register move" {
    // 48 89 d8: mov rax, rbx
    try expect(.{ .mov = .{ .dest = .{ .reg = .rax }, .src = .{ .reg = .rbx }, .size = .qword } }, &.{ 0x48, 0x89, 0xD8 });
    // 89 d8: mov eax, ebx (the 64-bit-mode default operand size is 32 bits)
    try expect(.{ .mov = .{ .dest = .{ .reg = .rax }, .src = .{ .reg = .rbx }, .size = .dword } }, &.{ 0x89, 0xD8 });
}

test "decodes a memory operand with a SIB byte" {
    // 48 8b 44 24 08: mov rax, [rsp + 8]
    try expect(.{ .mov = .{ .dest = .{ .reg = .rax }, .src = .{ .mem = .{ .base = .rsp, .disp = 8 } }, .size = .qword } }, &.{ 0x48, 0x8B, 0x44, 0x24, 0x08 });
    // 4a 8b 44 24 10: mov rax, [rsp + r12*1 + 0x10] (REX.X selects r12 as the index)
    try expect(.{ .mov = .{ .dest = .{ .reg = .rax }, .src = .{ .mem = .{ .base = .rsp, .index = .r12, .scale = 1, .disp = 0x10 } }, .size = .qword } }, &.{ 0x4A, 0x8B, 0x44, 0x24, 0x10 });
}

test "decodes a RIP-relative operand" {
    // 48 8b 05 10 00 00 00: mov rax, [rip + 0x10]
    try expect(.{ .mov = .{ .dest = .{ .reg = .rax }, .src = .{ .mem = .{ .rip = true, .disp = 0x10 } }, .size = .qword } }, &.{ 0x48, 0x8B, 0x05, 0x10, 0x00, 0x00, 0x00 });
}

test "decodes an immediate and its extension" {
    // 83 c0 ff: add rax, -1 (the immediate is sign-extended)
    try expect(.{ .alu = .{ .op = .add, .dest = .{ .reg = .rax }, .src = .{ .imm = .{ .value = 0xFF, .bits = 8, .sign_extended = true } }, .size = .dword } }, &.{ 0x83, 0xC0, 0xFF });
    // 48 b8 2a 00 00 00 00 00 00 00 00: mov rax, 42
    try expect(.{ .mov = .{ .dest = .{ .reg = .rax }, .src = .{ .imm = .{ .value = 42, .bits = 64, .sign_extended = false } }, .size = .qword } }, &.{ 0x48, 0xB8, 0x2A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 });
}

test "decodes the system and branch instructions" {
    try expect(.syscall, &.{ 0x0F, 0x05 });
    try expect(.nop, &.{0x90});
    try expect(.pause, &.{ 0xF3, 0x90 });
    // ff d0: call rax
    try expect(.{ .call = .{ .target = .{ .reg = .rax } } }, &.{ 0xFF, 0xD0 });
    // e8 10 00 00 00: call .+0x10
    try expect(.{ .call = .{ .target = .{ .rel = 0x10 } } }, &.{ 0xE8, 0x10, 0x00, 0x00, 0x00 });
    // 74 08: je .+8
    try expect(.{ .jcc = .{ .cond = .e, .rel = 8 } }, &.{ 0x74, 0x08 });
}

test "decodes an unmodelled opcode as unknown" {
    const d = try decodeStrict(&.{ 0x0F, 0x0B }, 0);
    try std.testing.expectEqual(Instruction{ .unknown = .{ .bytes = .{ 0x0F, 0x0B, 0, 0 }, .len = 2 } }, d.inst);
}

test "reports a truncated instruction" {
    const d = decode(&.{ 0x48, 0x89 }, 0);
    try std.testing.expect(d.truncated);
    try std.testing.expectError(error.TruncatedInstruction, decodeStrict(&.{ 0x48, 0x89 }, 0));
}

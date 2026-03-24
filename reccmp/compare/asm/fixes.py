import re
from typing import Sequence

from reccmp.compare.asm.parse import AsmExcerpt
from reccmp.compare.pinned_sequences import DiffOpcode

REG_FIND = re.compile(r"(?: |\[)(e?[a-d]x|e?[s,d]i|[a-d][l,h]|e?[b,s]p)")

ALLOWED_JUMP_SWAPS = (
    ("ja", "jb"),
    ("jae", "jbe"),
    ("jb", "ja"),
    ("jbe", "jae"),
    ("jg", "jl"),
    ("jge", "jle"),
    ("jl", "jg"),
    ("jle", "jge"),
    ("je", "je"),
    ("jne", "jne"),
)


def jump_swap_ok(a: str, b: str) -> bool:
    """For the instructions a,b, are they both jump instructions
    that are compatible with a swapped cmp operand order?"""
    # Grab the mnemonic
    jmp_a, _, __ = a.partition(" ")
    jmp_b, _, __ = b.partition(" ")

    return (jmp_a, jmp_b) in ALLOWED_JUMP_SWAPS


def _mnemonic(inst: str) -> str:
    if not inst:
        return ""
    return inst.split(" ", 1)[0].lower()


def _split_operands(inst: str) -> list[str]:
    _, _, operand_str = inst.partition(" ")
    if not operand_str:
        return []
    return [operand.strip() for operand in operand_str.split(",") if operand.strip()]


def is_operand_swap(a: str, b: str) -> bool:
    """This is a hack to avoid parsing the operands. It's not as simple as
    breaking on the comma because templates or string literals interfere
    with this. Instead we check:
        1. Do both strings use the exact same set of characters?
        2. If we do break on ', ', is the first token of each different?
    2 is needed to catch an edge case like:
        cmp eax, dword ptr [ecx + 0x1234]
        cmp ecx, dword ptr [eax + 0x1234]
    """
    return a.partition(", ")[0] != b.partition(", ")[0] and sorted(a) == sorted(b)


def get_patched_jump(a: str, b: str) -> str:
    """For jump instructions a, b, return `(mnemonic_a) (operand_b)`.
    The reason to do it this way (instead of just returning `a`) is that
    the jump instructions might use different displacement offsets
    or labels. If we just replace `b` with `a`, this diff would be
    incorrectly eliminated."""
    mnemonic_a, _, __ = a.partition(" ")
    _, __, operand_b = b.partition(" ")

    return mnemonic_a + " " + operand_b


def patch_mov_cmp_jmp(orig: list[str], recomp: list[str]) -> set[int]:
    return patch_mov_compare_jmp(orig, recomp, "cmp")


def patch_mov_test_jmp(orig: list[str], recomp: list[str]) -> set[int]:
    return patch_mov_compare_jmp(orig, recomp, "test")


def patch_mov_compare_jmp(
    orig: list[str], recomp: list[str], cmp_instruction: str
) -> set[int]:
    """Can we resolve the diffs between orig and recomp by patching
    swapped cmp instructions?
    For example:
        mov eax, dword ptr [ebp - 0x4]  mov eax, dword ptr [ebp - 0x8]
        cmp dword ptr [ebp - 0x8]       cmp dword ptr [ebp - 0x4]
        ja .label                       jb .label

    Returns set of fixed lines
    """

    # find the first "cmp"/"test" instruction
    cmp_index = next(
        (i for i, s in enumerate(orig) if s.startswith(cmp_instruction)), -1
    )

    # return if not found, or only found on first or last line
    if (
        cmp_index in (-1, 0, len(orig) - 1)
        or
        # recomp should also have a cmp in the same line
        not recomp[cmp_index].startswith(cmp_instruction)
        or
        # line before cmp must be a mov
        not orig[cmp_index - 1].startswith("mov")
        or not recomp[cmp_index - 1].startswith("mov")
        or
        # if the last lines are not a compatible jump difference
        not jump_swap_ok(orig[cmp_index + 1], recomp[cmp_index + 1])
    ):
        return set()

    # Checking if the combination of mov + cmp include the same set of characters
    # - that is, the set of operands are the same although switched in order
    if sorted(orig[cmp_index - 1] + orig[cmp_index]) == sorted(
        recomp[cmp_index - 1] + recomp[cmp_index]
    ):
        # We only register the fix if the jmp actually matches
        if orig[cmp_index + 1] == get_patched_jump(
            orig[cmp_index + 1], recomp[cmp_index + 1]
        ):
            return {0, 1, 2}
    return set()


def patch_mov_commutative(orig: list[str], recomp: list[str]) -> set[int]:
    """Can we resolve the diffs between orig and recomp by patching
        swapped operands in mov + commutative ops (add, and, or, xor, imul).
    For example:
        mov eax, dword ptr [ebp - 0x4]      mov eax, dword ptr [ebp - 0x8]
        add eax, dword ptr [ebp - 0x8]      add eax, dword ptr [ebp - 0x4]

    Returns set of fixed lines
    """

    valid_mnemonics = ("add", "and", "or", "xor", "imul")
    inst_index = next(
        (i for i, s in enumerate(orig) if _mnemonic(s) in valid_mnemonics), -1
    )

    # commutative op must exist and have a preceding line in both slices
    if inst_index in (-1, 0) or inst_index >= len(recomp):
        return set()

    # this pattern only handles mov + {valid_mnemonics}
    if (
        _mnemonic(recomp[inst_index]) != _mnemonic(orig[inst_index])
        or _mnemonic(orig[inst_index - 1]) != "mov"
        or _mnemonic(recomp[inst_index - 1]) != "mov"
    ):
        return set()

    orig_mov_ops = _split_operands(orig[inst_index - 1])
    recomp_mov_ops = _split_operands(recomp[inst_index - 1])
    orig_ops = _split_operands(orig[inst_index])
    recomp_ops = _split_operands(recomp[inst_index])

    # We expect these instructions to all have two operands.
    if any(
        len(operands) != 2
        for operands in (orig_mov_ops, recomp_mov_ops, orig_ops, recomp_ops)
    ):
        return set()

    # MOV destination must be the same register in both versions.
    mov_dest_norm = orig_mov_ops[0].lower()
    if mov_dest_norm != recomp_mov_ops[0].lower() or mov_dest_norm not in REGISTER_SET:
        return set()

    # Must target the same register and swap sources exactly.
    op_layout_ok = (
        len(orig_ops) == 2
        and len(recomp_ops) == 2
        and orig_ops[0].lower() == mov_dest_norm
        and recomp_ops[0].lower() == mov_dest_norm
    )
    swap_ok = orig_ops[1] == recomp_mov_ops[1] and recomp_ops[1] == orig_mov_ops[1]

    if op_layout_ok and swap_ok:
        return {inst_index - 1, inst_index}

    return set()


def patch_cmp_jmp(orig: list[str], recomp: list[str]) -> set[int]:
    return patch_compare_jmp(orig, recomp, "cmp")


def patch_test_jmp(orig: list[str], recomp: list[str]) -> set[int]:
    return patch_compare_jmp(orig, recomp, "test")


def patch_compare_jmp(
    orig: list[str], recomp: list[str], cmp_instruction: str
) -> set[int]:
    """Can we resolve the diffs between orig and recomp by patching
    swapped cmp instructions?
    For example:
        cmp eax, ebx                    cmp ebx, eax
        je .label                       je .label

        cmp eax, ebx                    cmp ebx, eax
        ja .label                       jb .label

    Returns set of fixed lines
    """

    # find the first "cmp"/"test" instruction
    cmp_index = next(
        (i for i, s in enumerate(orig) if s.startswith(cmp_instruction)), -1
    )
    # return if not found, or only found on the last line
    if (
        cmp_index in (-1, len(orig) - 1)
        or
        # recomp should also have a cmp in the same line
        not recomp[cmp_index].startswith(cmp_instruction)
        or
        # if the last lines are not a compatible jump difference
        not jump_swap_ok(orig[cmp_index + 1], recomp[cmp_index + 1])
    ):
        return set()

    # Checking two things:
    # Are the cmp operands flipped?
    # Is the jump instruction compatible with a flip?
    if is_operand_swap(orig[cmp_index], recomp[cmp_index]):
        if orig[cmp_index + 1] == get_patched_jump(
            orig[cmp_index + 1], recomp[cmp_index + 1]
        ):
            return {cmp_index, cmp_index + 1}
    return set()


def patch_fld_fmul(orig: list[str], recomp: list[str]) -> set[int]:
    """Can we resolve the diffs between orig and recomp by patching
    swapped fld/fmul instructions?
    For example:
        fld [ebp - 4]                   fmul [ebp - 8]
        fld [ebp - 8]                   fmul [ebp - 4]

    Returns set of fixed lines
    """

    valid_following_ops = ["fmul", "fadd"]

    # find the first "fld" instruction
    fld_index = next((i for i, s in enumerate(orig) if s.startswith("fld")), -1)
    # return if not found, or only found on the last line
    if (
        fld_index in (-1, len(orig) - 1)
        or
        # recomp should also have a fld in the same line
        not recomp[fld_index].startswith("fld")
    ):
        return set()

    _, _, orig_operand_a = orig[fld_index].partition(" ")
    orig_mnemonic_b, _, orig_operand_b = orig[fld_index + 1].partition(" ")

    _, _, recomp_operand_a = recomp[fld_index].partition(" ")
    recomp_mnemonic_b, _, recomp_operand_b = recomp[fld_index + 1].partition(" ")

    # fld must be followed by fmul/fadd and orig and recomp must have the same mnenomic
    # and the operands must be swapped
    if (
        orig_mnemonic_b in valid_following_ops
        and orig_mnemonic_b == recomp_mnemonic_b
        and orig_operand_a == recomp_operand_b
        and orig_operand_b == recomp_operand_a
    ):
        return {fld_index, fld_index + 1}

    return set()


def patch_cmp_swaps(
    codes: Sequence[DiffOpcode], orig_asm: list[str], recomp_asm: list[str]
) -> tuple[set[int], set[int], int]:
    """Can we resolve the diffs between orig and recomp by patching
    swapped cmp instructions?
    """

    # number of additional lines to send to the patcher when considering each diff
    additonal_lines_to_include = 3

    fixed_lines: set[int] = set()
    mov_cmp_jmp_fixed_lines: set[int] = set()
    mov_cmp_jmp_patterns: set[tuple[int, ...]] = set()

    patch_fns = [
        patch_cmp_jmp,
        patch_test_jmp,
        patch_mov_cmp_jmp,
        patch_mov_test_jmp,
        patch_fld_fmul,
        patch_mov_commutative,
    ]

    for code, i1, i2, j1, j2 in codes:
        # To save us the trouble of finding "compatible" cmp instructions
        # use the diff information we already have.
        if code != "replace":
            continue

        # If the ranges in orig and recomp are not equal, use the shorter one
        for i, j in zip(range(i1, i2), range(j1, j2)):
            # Keep track of mov+cmp/test+jmp fixes specifically.
            # We use this as a gate for fallback jump tolerance logic.
            mov_cmp_patch_lines = patch_mov_cmp_jmp(
                orig_asm[i : i + additonal_lines_to_include],
                recomp_asm[j : j + additonal_lines_to_include],
            )
            if len(mov_cmp_patch_lines) > 0:
                absolute_lines = tuple(sorted(j + x for x in mov_cmp_patch_lines))
                mov_cmp_jmp_patterns.add(absolute_lines)
                mov_cmp_jmp_fixed_lines.update(absolute_lines)

            mov_test_patch_lines = patch_mov_test_jmp(
                orig_asm[i : i + additonal_lines_to_include],
                recomp_asm[j : j + additonal_lines_to_include],
            )
            if len(mov_test_patch_lines) > 0:
                absolute_lines = tuple(sorted(j + x for x in mov_test_patch_lines))
                mov_cmp_jmp_patterns.add(absolute_lines)
                mov_cmp_jmp_fixed_lines.update(absolute_lines)

            for fn in patch_fns:
                this_patch_lines = fn(
                    orig_asm[i : i + additonal_lines_to_include],
                    recomp_asm[j : j + additonal_lines_to_include],
                )
                # if we have fixed lines by this patcher, add them to the combined `fixed_lines`
                if len(this_patch_lines) > 0:
                    fixed_lines.update([j + x for x in this_patch_lines])
                    # now that we've fixed these lines, no need to check the other patch strategies for fixing
                    break

    return fixed_lines, mov_cmp_jmp_fixed_lines, len(mov_cmp_jmp_patterns)


def effective_match_possible(orig_asm: list[str], recomp_asm: list[str]) -> bool:
    # We can only declare an effective match based on text.
    # Generally this requires equal instruction counts, but we allow
    # a single trailing terminal instruction to account for minor codegen
    # shape differences (e.g. explicit `ret`).
    if len(orig_asm) != len(recomp_asm):
        len_diff = abs(len(orig_asm) - len(recomp_asm))
        if len_diff > 2:
            return False

        longer_asm = orig_asm if len(orig_asm) > len(recomp_asm) else recomp_asm
        trailing_mnemonics = [
            line.split(" ", 1)[0].lower() for line in longer_asm[-len_diff:]
        ]
        if any(mnemonic not in TERMINAL_MNEMONICS for mnemonic in trailing_mnemonics):
            return False

    # mnemonic_orig = [inst.partition(" ")[0] for inst in orig_asm]
    # mnemonic_recomp = [inst.partition(" ")[0] for inst in recomp_asm]

    # Cannot change mnemonics. Must be same starting list
    # TODO: Fine idea but this will exclude jump swaps for cmp operand order
    # if sorted(mnemonic_orig) != sorted(mnemonic_recomp):
    #    return False

    return True


def find_regs_used(inst: str) -> list[str]:
    return REG_FIND.findall(inst)


def find_regs_changed(a: str, b: str) -> list[tuple[str, str]]:
    """For instructions a, b, return the pairs of registers that were used.
    This is not a very precise way to compare the instructions, so it depends
    on the input being two instructions that would match *except* for
    the register choice."""
    return list(zip(REG_FIND.findall(a), REG_FIND.findall(b)))


def bad_register_swaps(
    swaps: set[int], orig_asm: list[str], recomp_asm: list[str]
) -> set[int]:
    """The list of recomp indices in `swaps` tells which instructions are
    a match for orig except for the registers used. From that list, check
    whether a register swap should not be allowed.
    For now, this means checking for `push` instructions where the register
    was not used in any other register swaps on previous instructions."""
    rejects = set()

    # Foreach `push` instruction where we have excused the diff
    pushes = [j for j in swaps if recomp_asm[j].startswith("push")]

    for j in pushes:
        okay = False
        # Get the operands in each
        reg = (orig_asm[j].partition(" ")[2], recomp_asm[j].partition(" ")[2])
        # If this isn't a register at all, ignore it
        try:
            int(reg[0], 16)
            continue
        except ValueError:
            pass

        # For every other excused diff that is *not* a push:
        # Assumes same index in orig as in recomp, but so does our naive match
        for k in swaps.difference(pushes):
            changed_regs = find_regs_changed(orig_asm[k], recomp_asm[k])
            if reg in changed_regs or reg[::-1] in changed_regs:
                okay = True
                break

        if not okay:
            rejects.add(j)

    return rejects


# Instructions that result in a change to the first operand
MODIFIER_INSTRUCTIONS = ("adc", "add", "lea", "mov", "neg", "sbb", "sub", "pop", "xor")


def instruction_alters_regs(inst: str, regs: set[str]) -> bool:
    mnemonic, _, op_str = inst.partition(" ")
    first_operand, _, __ = op_str.partition(", ")

    return (mnemonic in MODIFIER_INSTRUCTIONS and first_operand in regs) or (
        mnemonic == "call" and "eax" in regs
    )


def _is_relocatable(instr: str) -> bool:
    """
    Excludes certain instructions whose relocation will always change the logic
    to be considered for an effective match.
    """
    # Any memory operand can alias with nearby loads/stores, and this matcher
    # does not model memory dependencies. Keep relocation register/immediate-only.
    if "[" in instr:
        return False
    if instr.startswith("start +"):
        # Do not relocate jump table entries (this most likely influences the behaviour)
        return False
    if instr.startswith("0x"):
        # Do not relocate data table entries (this most likely influences the behaviour)
        return False
    return True


def relocate_instructions(
    codes: Sequence[DiffOpcode], orig_asm: list[str], recomp_asm: list[str]
) -> set[int]:
    """Collect the list of instructions deleted from orig and inserted
    into recomp, according to the diff opcodes. Using this list, match up
    any pairs of instructions that we assume to be relocated and return
    the indices in recomp where this has occurred.
    This function has many limitations and could be improved. See: GH #324.
    """
    deletes = {
        i for code, i1, i2, _, __ in codes for i in range(i1, i2) if code == "delete"
    }
    # Using list instead of set to preserve ordering.
    # `i1` is the index of the orig_asm list where this line will be inserted.
    # This is not necessarily equal to `j1`, the index of the inserted line in recomp_asm.
    # Therefore we need to save `i1` so that we verify each line between the start and end of the move. (GH #332)
    inserts = [
        (i1, j)
        for code, i1, __, j1, j2 in codes
        for j in range(j1, j2)
        if code == "insert"
    ]

    relocated = set()

    for orig_dest, j in inserts:
        line = recomp_asm[j]
        if not _is_relocatable(line):
            continue
        recomp_regs_used = set(find_regs_used(line))
        for i in deletes:
            # Check for exact match.
            if orig_asm[i] == line:
                # To account for a move in either direction:
                # the deleted line can precede or follow the inserted line.
                reloc_start = min(i, orig_dest)
                reloc_end = max(i, orig_dest)

                if not any(
                    instruction_alters_regs(orig_asm[k], recomp_regs_used)
                    for k in range(reloc_start, reloc_end)
                ):
                    relocated.add(j)
                    deletes.remove(i)
                    break

    return relocated


DWORD_REGS = ("eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp")
WORD_REGS = ("ax", "bx", "cx", "dx", "si", "di", "bp", "sp")
BYTE_REGS = ("ah", "al", "bh", "bl", "ch", "cl", "dh", "dl")
REGISTER_SET = set(reg for reg in (DWORD_REGS + WORD_REGS + BYTE_REGS))


def naive_register_replacement(orig_asm: list[str], recomp_asm: list[str]) -> set[int]:
    """Replace all registers of the same size with a placeholder string.
    After doing that, compare orig and recomp again.
    Return indices from recomp that are now equal to the same index in orig.
    This requires orig and recomp to have the same number of instructions,
    but this is already a requirement for effective match."""
    orig_raw = "\n".join(orig_asm)
    recomp_raw = "\n".join(recomp_asm)

    # TODO: hardly the most elegant way to do this.
    for rdw in DWORD_REGS:
        orig_raw = orig_raw.replace(rdw, "~reg4")
        recomp_raw = recomp_raw.replace(rdw, "~reg4")

    for rw in WORD_REGS:
        orig_raw = orig_raw.replace(rw, "~reg2")
        recomp_raw = recomp_raw.replace(rw, "~reg2")

    for rb in BYTE_REGS:
        orig_raw = orig_raw.replace(rb, "~reg1")
        recomp_raw = recomp_raw.replace(rb, "~reg1")

    orig_scrubbed = orig_raw.split("\n")
    recomp_scrubbed = recomp_raw.split("\n")

    compare_len = min(len(orig_scrubbed), len(recomp_scrubbed))
    return {j for j in range(compare_len) if orig_scrubbed[j] == recomp_scrubbed[j]}


def find_effective_match(
    codes: Sequence[DiffOpcode], orig_asm: list[str], recomp_asm: list[str]
) -> bool:
    """Check whether the two sequences of instructions are an effective match.
    Meaning: do they differ only by instruction order or register selection?"""
    # pylint: disable=too-many-return-statements
    if not effective_match_possible(orig_asm, recomp_asm):
        return False

    already_equal = {
        j for code, _, __, j1, j2 in codes for j in range(j1, j2) if code == "equal"
    }

    # We need to come up with some answer for each of these lines
    recomp_lines_disputed = {
        j
        for code, _, __, j1, j2 in codes
        for j in range(j1, j2)
        if code in ("insert", "replace")
    }

    cmp_swaps, mov_cmp_jmp_swaps, mov_cmp_jmp_swap_count = patch_cmp_swaps(
        codes, orig_asm, recomp_asm
    )
    # This naive result includes lines that already match, so remove those
    naive_swaps = naive_register_replacement(orig_asm, recomp_asm).difference(
        already_equal
    )
    relocates = relocate_instructions(codes, orig_asm, recomp_asm)

    bad_swaps = bad_register_swaps(naive_swaps, orig_asm, recomp_asm)

    corrections = set().union(
        naive_swaps.difference(bad_swaps),
        cmp_swaps,
        relocates,
    )

    if corrections.issuperset(recomp_lines_disputed):
        return True

    # Fallback: if we found at least one mov+cmp/test+jmp fix, allow unresolved
    # jump displacement differences anywhere in the function when they are all
    # direct jumps with the same mnemonic and within +/-1.
    if len(mov_cmp_jmp_swaps) == 0:
        return False

    # Allow jump displacement drift proportional to the number of recognized
    # mov+cmp/test+jmp reversals, but keep an upper bound to avoid widening
    # effective-match acceptance too much.
    jump_delta_tolerance = min(2, max(1, mov_cmp_jmp_swap_count))

    unresolved = recomp_lines_disputed.difference(corrections)
    if len(unresolved) == 0:
        return True

    trailing_terminal_unresolved = set()

    for code, i1, i2, j1, j2 in codes:
        unresolved_here = unresolved.intersection(range(j1, j2))
        if len(unresolved_here) == 0:
            continue

        if code == "insert":
            if not _is_terminal_suffix(
                unresolved_here, len(recomp_asm), recomp_asm, jump_delta_tolerance
            ):
                return False
            trailing_terminal_unresolved.update(unresolved_here)
            continue

        if code != "replace":
            return False

        orig_span = i2 - i1
        recomp_span = j2 - j1
        paired_span = min(orig_span, recomp_span)

        # If this replace has unmatched trailing recomp lines, they are allowed
        # only as a terminal suffix of leave/ret/nop instructions.
        if recomp_span > orig_span:
            extra_recomp = set(range(j1 + orig_span, j2))
            unresolved_extra = unresolved_here.intersection(extra_recomp)
            if unresolved_extra != extra_recomp:
                return False
            if not _is_terminal_suffix(
                unresolved_extra, len(recomp_asm), recomp_asm, jump_delta_tolerance
            ):
                return False
            trailing_terminal_unresolved.update(unresolved_extra)
            unresolved_here = unresolved_here.difference(unresolved_extra)
        elif orig_span != recomp_span:
            return False

        for j in unresolved_here:
            if j >= j1 + paired_span:
                return False
            i = i1 + (j - j1)
            if not _near_direct_jump_match(
                orig_asm[i], recomp_asm[j], jump_delta_tolerance
            ):
                return False

    if len(trailing_terminal_unresolved) > jump_delta_tolerance:
        return False

    return True


_DIRECT_JUMP_HEX_RE = re.compile(r"^-?0x[0-9a-f]+$", re.IGNORECASE)
TERMINAL_MNEMONICS = {"ret", "nop", "leave"}


def _near_direct_jump_match(orig_line: str, recomp_line: str, max_delta: int) -> bool:
    orig_mnemonic, _, orig_operand = orig_line.partition(" ")
    recomp_mnemonic, _, recomp_operand = recomp_line.partition(" ")

    if (
        not orig_mnemonic.startswith("j")
        or not recomp_mnemonic.startswith("j")
        or orig_mnemonic != recomp_mnemonic
    ):
        return False

    if not _DIRECT_JUMP_HEX_RE.fullmatch(orig_operand.strip()):
        return False

    if not _DIRECT_JUMP_HEX_RE.fullmatch(recomp_operand.strip()):
        return False

    orig_jump = int(orig_operand, 16)
    recomp_jump = int(recomp_operand, 16)
    return abs(orig_jump - recomp_jump) <= max_delta


def _is_terminal_suffix(
    unresolved_here: set[int],
    recomp_len: int,
    recomp_asm: list[str],
    max_count: int,
) -> bool:
    if len(unresolved_here) == 0 or len(unresolved_here) > max_count:
        return False

    trailing_expected = set(range(recomp_len - len(unresolved_here), recomp_len))
    if unresolved_here != trailing_expected:
        return False

    for j in unresolved_here:
        mnemonic = recomp_asm[j].split(" ", 1)[0].lower()
        if mnemonic not in TERMINAL_MNEMONICS:
            return False

    return True


def assert_fixup(asm: AsmExcerpt):
    """Detect assert calls and replace the code filename and line number
    values with macros (from assert.h)."""
    for i, (_, line) in enumerate(asm):
        if "_assert" in line and line.startswith("call"):
            try:
                asm[i - 3] = (asm[i - 3][0], "push __LINE__")
                asm[i - 2] = (asm[i - 2][0], "push __FILE__")
            except IndexError:
                continue

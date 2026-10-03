"""Expand the official compact Netlib MPS encoding to standard MPS text."""
import re
from typing import List, Optional, Sequence

from .base import ModelParseError


_TRANSLATION = '!"#$%&\'()*+,-./0123456789;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~'
_TRANSLATION_INDEX = {character: index for index, character in enumerate(_TRANSLATION)}
_CHECKSUM_LINES = 71
_BOUND_TYPES = ("UP", "LO", "FX", "FR", "MI", "PL")


def is_netlib_compressed_mps(content: str) -> bool:
    """Identify Netlib's NAME + fixed-width statistics envelope."""
    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 4 or not lines[0].lstrip().upper().startswith("NAME"):
        return False
    return _fixed_integers(lines[1], 8) is not None and _fixed_integers(lines[2], 3) is not None


def expand_netlib_compressed_mps(
    content: str,
    *,
    file_path: Optional[str] = None,
) -> str:
    """Expand one Netlib ``emps`` stream while validating its checksums/counts."""
    reader = _NetlibLineReader(content, file_path)
    try:
        name_line = reader.readline()
        while not name_line.upper().startswith("NAME"):
            name_line = reader.readline()
        name_tokens = name_line.split(maxsplit=1)
        if len(name_tokens) != 2 or not name_tokens[1].strip():
            raise reader.error("Netlib MPS NAME record has no model name.")
        model_name = name_tokens[1].strip()

        row_stats = _fixed_integers(reader.readline(), 8)
        if row_stats is None:
            raise reader.error("Malformed Netlib MPS first statistics record.")
        bound_stats = _fixed_integers(reader.readline(), 3)
        if bound_stats is None:
            raise reader.error("Malformed Netlib MPS second statistics record.")
        (
            num_rows,
            num_columns,
            _,
            num_coefficients,
            _,
            num_rhs_values,
            num_ranges,
            num_range_values,
        ) = row_stats
        num_bound_sets, num_bounds, num_table_values = bound_stats
        if min(row_stats + bound_stats) < 0 or num_rows == 0 or num_columns == 0:
            raise reader.error("Netlib MPS statistics contain invalid dimensions or counts.")
        if num_ranges or num_range_values:
            raise reader.error("Netlib MPS RANGES sections are not supported.")

        reader.reset_checksum()
        cursor = _CompressedCursor(reader)
        number_table: List[str] = []
        for _ in range(num_table_values):
            number_table.append(_read_number(cursor, number_table, allow_reference=False))

        cursor.discard_current_line()
        rows: List[tuple[str, str]] = []
        for _ in range(num_rows):
            line = reader.readline()
            row_type = line[:1]
            row_name = line[1:9].strip()
            if row_type not in {"N", "L", "E", "G"} or not row_name:
                raise reader.error(f"Malformed Netlib MPS row record '{line}'.")
            rows.append((row_type, row_name))
        row_names = [name for _, name in rows]

        output = [f"NAME          {model_name}", "ROWS"]
        output.extend(f" {row_type}  {row_name}" for row_type, row_name in rows)

        output.append("COLUMNS")
        column_names, column_records = _read_records(
            cursor,
            reader,
            num_coefficients,
            rows,
            number_table,
            block="COLUMNS",
        )
        if len(column_names) != num_columns:
            raise reader.error(
                f"Netlib MPS declares {num_columns} columns but encodes {len(column_names)}."
            )
        output.extend(
            f"    {column}  {row}  {value}"
            for column, row, value in column_records
        )

        output.append("RHS")
        _, rhs_records = _read_records(
            cursor,
            reader,
            num_rhs_values,
            rows,
            number_table,
            block="RHS",
            variable_count=num_columns,
        )
        output.extend(
            f"    {set_name}  {row}  {value}"
            for set_name, row, value in rhs_records
        )

        if num_bounds:
            output.append("BOUNDS")
            bound_set_names, bound_records = _read_bound_records(
                cursor,
                reader,
                num_bounds,
                number_table,
                column_names,
            )
            if len(bound_set_names) != num_bound_sets:
                raise reader.error("Netlib BOUNDS set count does not match its statistics record.")
            output.extend(bound_records)
        elif num_bound_sets:
            raise reader.error("Netlib MPS declares bound sets without bound records.")

        reader.finish()
        output.append("ENDATA")
        return "\n".join(output) + "\n"
    except ModelParseError:
        raise
    except (IndexError, OverflowError, ValueError) as error:
        raise reader.error(f"Malformed compressed Netlib MPS data: {error}") from error


class _NetlibLineReader:
    def __init__(self, content: str, file_path: Optional[str]) -> None:
        if not content.isascii():
            raise ModelParseError(
                "Compressed Netlib MPS must contain ASCII data.", file_path=file_path
            )
        self.lines = content.splitlines()
        self.file_path = file_path
        self.position = 0
        self.checksum: List[str] = []

    def readline(self) -> str:
        if self.position >= len(self.lines):
            raise self.error("Unexpected end of compressed Netlib MPS input.")
        line = self.lines[self.position]
        self.position += 1
        if line.startswith(":"):
            raise self.error("Netlib MPS mystery records are not supported.")
        self.checksum.append(_checksum_character(line))
        if len(self.checksum) == _CHECKSUM_LINES:
            self._verify_checksum()
        return line

    def reset_checksum(self) -> None:
        self.checksum.clear()

    def finish(self) -> None:
        if self.checksum:
            self._verify_checksum()
        if any(line.strip() for line in self.lines[self.position :]):
            raise self.error("Multiple Netlib MPS problems or trailing data are not supported.")

    def error(self, message: str) -> ModelParseError:
        return ModelParseError(message, file_path=self.file_path)

    def _verify_checksum(self) -> None:
        if self.position >= len(self.lines):
            raise self.error("Compressed Netlib MPS is missing a checksum record.")
        expected = " " + "".join(self.checksum)
        actual = self.lines[self.position]
        self.position += 1
        if actual != expected:
            raise self.error(
                f"Compressed Netlib MPS checksum mismatch at physical line {self.position}."
            )
        self.checksum.clear()


class _CompressedCursor:
    def __init__(self, reader: _NetlibLineReader) -> None:
        self.reader = reader
        self.line = ""
        self.position = 0

    def ensure_line(self) -> None:
        while self.position >= len(self.line):
            self.line = self.reader.readline()
            self.position = 0
            if self.line:
                return

    def code(self) -> int:
        if self.position >= len(self.line):
            raise self.reader.error("Compressed Netlib token crosses a physical line boundary.")
        character = self.line[self.position]
        self.position += 1
        return _TRANSLATION_INDEX.get(character, 92)

    def index(self, first_code: Optional[int] = None) -> int:
        if first_code is None:
            self.ensure_line()
            code = self.code()
        else:
            code = first_code
        if code >= 46:
            raise self.reader.error("Invalid encoded Netlib index.")
        if code >= 23:
            return code - 23
        value = code
        for _ in range(16):
            code = self.code_or_end()
            value = value * 46 + code
            if code >= 46:
                return value - 46
        raise self.reader.error("Encoded Netlib index is too long.")

    def code_or_end(self) -> int:
        return 92 if self.position >= len(self.line) else self.code()

    def take_name(self) -> str:
        return self.take_field_name(allow_empty=False)

    def take_field_name(self, *, allow_empty: bool) -> str:
        name = self.line[self.position : self.position + 8].strip()
        if not name and not allow_empty:
            raise self.reader.error("Netlib column/set marker has no following name.")
        self.discard_current_line()
        return name

    def discard_current_line(self) -> None:
        self.line = ""
        self.position = 0


def _read_number(
    cursor: _CompressedCursor,
    number_table: Sequence[str],
    *,
    allow_reference: bool,
) -> str:
    cursor.ensure_line()
    code = cursor.code()
    if code < 46:
        index = cursor.index(code)
        if not allow_reference:
            raise cursor.reader.error("Netlib number table cannot reference itself.")
        if not 1 <= index <= len(number_table):
            raise cursor.reader.error(f"Invalid Netlib number-table index {index}.")
        return number_table[index - 1]

    code -= 46
    negative = code >= 23
    if negative:
        code -= 23
    digits_limit = 11 if negative else 12
    sign = "-" if negative else ""

    if code >= 11:
        code -= 11
        value = code - 6 if code >= 6 else _read_base46(cursor, code)
        return sign + str(value) + "."

    exponent = cursor.code() - 50
    mantissa = cursor.code()
    for _ in range(code):
        mantissa = mantissa * 92 + cursor.code()
    digits = list(reversed(str(mantissa)))
    count = len(digits)
    decimal_position = count + exponent

    if exponent > 0 and (decimal_position < digits_limit or exponent < 3):
        number = "".join(reversed(digits)) + ("0" * exponent) + "."
    elif exponent > 0:
        number = _scientific_number(digits, count, exponent)
    elif decimal_position >= 0:
        integer_digits = decimal_position
        number = "".join(digits.pop() for _ in range(integer_digits)) + "."
        number += "".join(reversed(digits))
    elif exponent > -digits_limit:
        zero_count = -decimal_position
        number = "." + ("0" * zero_count) + "".join(reversed(digits))
    else:
        number = _scientific_number(digits, count, exponent)
    return sign + number


def _read_base46(cursor: _CompressedCursor, initial: int) -> int:
    value = initial
    for _ in range(16):
        code = cursor.code_or_end()
        value = value * 46 + code
        if code >= 46:
            return value - 46
    raise cursor.reader.error("Encoded Netlib number is too long.")


def _scientific_number(digits: List[str], count: int, exponent: int) -> str:
    exponent += count - 1
    output: List[str] = []
    if exponent == -10:
        exponent = -9
    else:
        if exponent > 9 and exponent <= count + 8:
            while exponent > 10:
                output.append(digits.pop())
                exponent -= 1
        output.append(digits.pop())
    output.append(".")
    output.extend(reversed(digits))
    output.append("E")
    if exponent < 0:
        output.append("-")
        exponent = -exponent
    output.extend(reversed(str(exponent)))
    return "".join(output)


def _read_records(
    cursor: _CompressedCursor,
    reader: _NetlibLineReader,
    record_count: int,
    rows: Sequence[tuple[str, str]],
    number_table: Sequence[str],
    *,
    block: str,
    variable_count: Optional[int] = None,
) -> tuple[List[str], List[tuple[str, str, str]]]:
    cursor.discard_current_line()
    names: List[str] = []
    records: List[tuple[str, str, str]] = []
    current_name: Optional[str] = None
    for _ in range(record_count):
        index = cursor.index()
        while index == 0:
            current_name = cursor.take_field_name(allow_empty=block == "RHS")
            if block == "COLUMNS":
                names.append(current_name)
            index = cursor.index()
        if current_name is None:
            raise reader.error(f"Netlib {block} data appears before its set/column name.")
        if block == "COLUMNS":
            if not 1 <= index <= len(rows):
                raise reader.error(f"Invalid Netlib row index {index} in COLUMNS.")
            record_name = rows[index - 1][1]
        else:
            if not 1 <= index <= len(rows):
                raise reader.error(f"Invalid Netlib row index {index} in RHS.")
            record_name = rows[index - 1][1]
        value = _read_number(cursor, number_table, allow_reference=True)
        records.append((current_name, record_name, value))
    if block == "COLUMNS" and variable_count is not None and len(names) != variable_count:
        raise reader.error("Netlib COLUMNS count does not match its statistics record.")
    return names, records


def _read_bound_records(
    cursor: _CompressedCursor,
    reader: _NetlibLineReader,
    record_count: int,
    number_table: Sequence[str],
    variable_names: Sequence[str],
) -> tuple[List[str], List[str]]:
    cursor.discard_current_line()
    names: List[str] = []
    records: List[str] = []
    current_name: Optional[str] = None
    for _ in range(record_count):
        bound_code = cursor.index()
        while bound_code == 0:
            current_name = cursor.take_name()
            names.append(current_name)
            bound_code = cursor.index()
        if current_name is None or not 1 <= bound_code <= 6:
            raise reader.error(f"Invalid Netlib bound type index {bound_code}.")
        variable_index = cursor.index() - 1
        if not 0 <= variable_index < len(variable_names):
            raise reader.error(f"Invalid Netlib bound variable index {variable_index + 1}.")
        variable_name = variable_names[variable_index]
        if bound_code >= 4:
            records.append(
                f" { _BOUND_TYPES[bound_code - 1] } {current_name} {variable_name}"
            )
        else:
            value = _read_number(cursor, number_table, allow_reference=True)
            records.append(
                f" { _BOUND_TYPES[bound_code - 1] } {current_name} {variable_name} {value}"
            )
    return names, records


def _fixed_integers(line: str, count: int) -> Optional[tuple[int, ...]]:
    if len(line) < count * 9 or line[count * 9 :].strip():
        return None
    fields = [line[index : index + 9].strip() for index in range(0, count * 9, 9)]
    if any(not re.fullmatch(r"[+-]?\d+", field) for field in fields):
        return None
    return tuple(int(field) for field in fields)


def _checksum_character(line: str) -> str:
    value = 0
    for character in line:
        code = _TRANSLATION_INDEX.get(character, 92)
        value = (value >> 1) + code + (16384 if value & 1 else 0)
    return _TRANSLATION[value % 92]
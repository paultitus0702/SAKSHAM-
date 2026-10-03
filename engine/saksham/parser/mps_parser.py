"""
Standard and Free MPS Format Parser for SAKSHAM.

Parses standard MPS (.mps) mathematical programming files into CanonicalModel.
Supports NAME, ROWS (N, L, E, G), COLUMNS (with integer markers), RHS, BOUNDS (LO, UP, FX, FR, BV, LI, UI), and ENDATA.
"""
from pathlib import Path
from typing import Union, Optional, List, Dict, Set, Tuple
import numpy as np
from scipy import sparse

from .base import AbstractModelParser, ModelParseError
from .netlib_mps import expand_netlib_compressed_mps, is_netlib_compressed_mps
from ..canonical.model import CanonicalModel, ObjectiveSense, ConstraintSense, VarType
from ..canonical.validator import ModelValidator


class MPSParser(AbstractModelParser):
    """
    Parser for standard fixed-format and free-format MPS mathematical optimization models.
    """

    def parse_file(self, file_path: Union[str, Path]) -> CanonicalModel:
        """Parse MPS file from filesystem."""
        path = Path(file_path)
        if not path.is_file():
            raise ModelParseError(f"MPS file not found: {file_path}", file_path=str(file_path))
        
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ModelParseError(f"Failed to read MPS file: {str(e)}", file_path=str(file_path))

        model_name = path.stem
        return self.parse_string(content, model_name=model_name, file_path=str(file_path))

    def parse_string(
        self,
        content: str,
        model_name: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> CanonicalModel:
        """Parse MPS formulation from raw text string."""
        if is_netlib_compressed_mps(content):
            content = expand_netlib_compressed_mps(content, file_path=file_path)
        raw_lines = content.splitlines()
        
        # 1. Clean comment lines and empty lines
        cleaned: List[Tuple[int, str]] = []
        for i, line in enumerate(raw_lines, start=1):
            s = line.strip()
            # In standard MPS, comments start with '*'
            if s and not s.startswith("*"):
                cleaned.append((i, line))

        if not cleaned:
            raise ModelParseError("MPS content is empty or contains only comments.", file_path=file_path)

        # 2. Split into major sections
        sections = self._split_sections(cleaned, file_path)
        for unsupported_section in ("RANGES", "OBJSENSE"):
            if unsupported_section in sections:
                line_num, line = sections[unsupported_section][0]
                raise ModelParseError(
                    f"MPS {unsupported_section} sections are not supported.",
                    line_num=line_num,
                    line_text=line,
                    file_path=file_path,
                )

        # 3. Process NAME
        parsed_name = self._parse_name(sections.get("NAME", []))
        if model_name is None:
            model_name = parsed_name or "mps_model"

        # 4. Process ROWS
        obj_row_name, row_types, constr_names = self._parse_rows(sections.get("ROWS", []), file_path)
        num_constrs = len(constr_names)
        constr_name_to_idx = {name: idx for idx, name in enumerate(constr_names)}

        # 5. Process COLUMNS
        # Returns column ordered list, coefficients map: col_name -> {row_name: float}, integer vars set
        col_names, matrix_entries, marker_integers = self._parse_columns(
            sections.get("COLUMNS", []),
            obj_row_name,
            set(row_types.keys()),
            file_path
        )
        num_vars = len(col_names)
        if num_vars == 0:
            raise ModelParseError("No variables found in COLUMNS section of MPS model.", file_path=file_path)
        var_name_to_idx = {name: idx for idx, name in enumerate(col_names)}

        # 6. Process RHS
        rhs = self._parse_rhs(sections.get("RHS", []), constr_names, constr_name_to_idx, file_path)

        # 7. Process BOUNDS
        l_var, u_var, bound_var_types = self._parse_bounds(
            sections.get("BOUNDS", []),
            col_names,
            var_name_to_idx,
            marker_integers,
            file_path
        )

        # 8. Build Objective vector c
        c = np.zeros(num_vars, dtype=np.float64)
        for col_idx, cname in enumerate(col_names):
            row_dict = matrix_entries.get(cname, {})
            if obj_row_name and obj_row_name in row_dict:
                c[col_idx] = row_dict[obj_row_name]

        # 9. Build Constraint Matrix A
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for col_idx, cname in enumerate(col_names):
            row_dict = matrix_entries.get(cname, {})
            for rname, val in row_dict.items():
                if rname in constr_name_to_idx:
                    row_idx = constr_name_to_idx[rname]
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(val)

        A = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(num_constrs, num_vars),
            dtype=np.float64
        )

        # Constraint senses
        constraint_senses: List[ConstraintSense] = []
        for rname in constr_names:
            rtype = row_types[rname]
            if rtype == "L":
                constraint_senses.append(ConstraintSense.LESS_EQUAL)
            elif rtype == "E":
                constraint_senses.append(ConstraintSense.EQUAL)
            elif rtype == "G":
                constraint_senses.append(ConstraintSense.GREATER_EQUAL)
            else:
                raise ModelParseError(f"Unexpected constraint row type: {rtype}", file_path=file_path)

        # Standard MPS assumes minimization by default
        sense = ObjectiveSense.MINIMIZE

        model = CanonicalModel(
            name=model_name,
            sense=sense,
            c0=0.0,
            c=c,
            A=A,
            rhs=rhs,
            constraint_senses=constraint_senses,
            l_var=l_var,
            u_var=u_var,
            var_types=bound_var_types,
            var_names=col_names,
            constr_names=constr_names,
            var_name_to_idx=var_name_to_idx,
            constr_name_to_idx=constr_name_to_idx
        )

        ModelValidator.validate(model, raise_on_error=True)
        return model

    def _split_sections(
        self,
        cleaned_lines: List[Tuple[int, str]],
        file_path: Optional[str]
    ) -> Dict[str, List[Tuple[int, str]]]:
        """Splits MPS lines into standard sections."""
        sections: Dict[str, List[Tuple[int, str]]] = {}
        current_section: Optional[str] = None

        known_headers = {
            "NAME",
            "OBJSENSE",
            "ROWS",
            "COLUMNS",
            "RHS",
            "RANGES",
            "BOUNDS",
            "ENDATA",
        }

        for line_num, raw_line in cleaned_lines:
            tokens = raw_line.split()
            first_token = tokens[0].upper() if tokens else ""

            # Check if this line is a section header (first token without leading whitespace)
            # In standard MPS, section headers start in column 1
            is_header = (not raw_line.startswith(" ") and not raw_line.startswith("\t")) and (first_token in known_headers)

            if is_header:
                current_section = first_token
                sections[current_section] = [(line_num, raw_line)]
            else:
                if current_section is None:
                    raise ModelParseError(
                        f"Expected MPS section header (e.g. NAME, ROWS), got: '{raw_line}'",
                        line_num=line_num,
                        file_path=file_path
                    )
                sections[current_section].append((line_num, raw_line))

        if "ENDATA" not in sections:
            # Tolerant: some files omit ENDATA, but standard has it
            pass

        return sections

    def _parse_name(self, lines: List[Tuple[int, str]]) -> Optional[str]:
        """Extracts model name from NAME line."""
        if not lines:
            return None
        _, line = lines[0]
        tokens = line.split()
        if len(tokens) >= 2:
            return tokens[1]
        return None

    def _parse_rows(
        self,
        lines: List[Tuple[int, str]],
        file_path: Optional[str]
    ) -> Tuple[Optional[str], Dict[str, str], List[str]]:
        """
        Parses ROWS section.
        Returns (objective_row_name, {row_name: type}, ordered_constraint_names)
        """
        if not lines or len(lines) <= 1:
            raise ModelParseError("ROWS section is missing or empty in MPS file.", file_path=file_path)

        obj_row_name: Optional[str] = None
        row_types: Dict[str, str] = {}
        constr_names: List[str] = []

        for line_num, line in lines[1:]:  # Skip header line
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) < 2:
                raise ModelParseError(f"Malformed ROWS entry: '{line}'", line_num=line_num, file_path=file_path)

            rtype = tokens[0].upper()
            rname = tokens[1]

            if rtype not in ["N", "L", "E", "G"]:
                raise ModelParseError(f"Unknown row type '{rtype}' in ROWS entry.", line_num=line_num, file_path=file_path)

            if rname in row_types:
                raise ModelParseError(f"Duplicate row name '{rname}' in ROWS section.", line_num=line_num, file_path=file_path)

            row_types[rname] = rtype

            if rtype == "N":
                if obj_row_name is None:
                    obj_row_name = rname
            else:
                constr_names.append(rname)

        if not constr_names and obj_row_name is None:
            raise ModelParseError("No valid rows parsed from ROWS section.", file_path=file_path)

        return obj_row_name, row_types, constr_names

    def _parse_columns(
        self,
        lines: List[Tuple[int, str]],
        obj_row_name: Optional[str],
        all_row_names: Set[str],
        file_path: Optional[str]
    ) -> Tuple[List[str], Dict[str, Dict[str, float]], Set[str]]:
        """
        Parses COLUMNS section.
        Extracts ordered column names, coefficients mapping {col_name: {row_name: val}},
        and integer variables marked by 'MARKER' / 'INTORG'.
        """
        if not lines or len(lines) <= 1:
            raise ModelParseError("COLUMNS section is missing or empty in MPS file.", file_path=file_path)

        col_order: List[str] = []
        matrix_entries: Dict[str, Dict[str, float]] = {}
        marker_integers: Set[str] = set()

        in_integer_block = False

        for line_num, line in lines[1:]:
            tokens = line.split()
            if not tokens:
                continue

            col_name = tokens[0]

            # Check for integer marker
            # Pattern: col_name 'MARKER' 'INTORG' or col_name 'MARKER' 'INTEND'
            if "'MARKER'" in tokens or "MARKER" in tokens:
                if "'INTORG'" in tokens or "INTORG" in tokens:
                    in_integer_block = True
                    continue
                elif "'INTEND'" in tokens or "INTEND" in tokens:
                    in_integer_block = False
                    continue

            if col_name not in matrix_entries:
                matrix_entries[col_name] = {}
                col_order.append(col_name)

            if in_integer_block:
                marker_integers.add(col_name)

            # Entries follow in pairs: row_name1 val1 [row_name2 val2]
            pair_tokens = tokens[1:]
            if len(pair_tokens) % 2 != 0:
                raise ModelParseError(
                    f"Malformed COLUMNS entry (odd number of row/value tokens): '{line}'",
                    line_num=line_num,
                    file_path=file_path
                )

            for k in range(0, len(pair_tokens), 2):
                rname = pair_tokens[k]
                val_str = pair_tokens[k + 1]

                if rname not in all_row_names:
                    raise ModelParseError(
                        f"Unknown row '{rname}' referenced in COLUMNS section.",
                        line_num=line_num,
                        file_path=file_path
                    )

                try:
                    val = float(val_str)
                except ValueError:
                    raise ModelParseError(
                        f"Invalid numeric value '{val_str}' for column '{col_name}', row '{rname}'.",
                        line_num=line_num,
                        file_path=file_path
                    )

                matrix_entries[col_name][rname] = matrix_entries[col_name].get(rname, 0.0) + val

        return col_order, matrix_entries, marker_integers

    def _parse_rhs(
        self,
        lines: List[Tuple[int, str]],
        constr_names: List[str],
        constr_name_to_idx: Dict[str, int],
        file_path: Optional[str]
    ) -> np.ndarray:
        """Parses RHS section into a 1D numpy array."""
        num_constrs = len(constr_names)
        rhs = np.zeros(num_constrs, dtype=np.float64)

        if not lines or len(lines) <= 1:
            return rhs

        for line_num, line in lines[1:]:
            tokens = line.split()
            if not tokens:
                continue

            # Standard MPS RHS has: [rhs_set_name] row1 val1 [row2 val2]
            # If the line starts with an entry where 2nd token is a known row name, 1st token is rhs_set_name
            # Otherwise 1st token could be row name
            if len(tokens) >= 3 and tokens[1] in constr_name_to_idx:
                pair_tokens = tokens[1:]
            elif len(tokens) >= 2 and tokens[0] in constr_name_to_idx:
                pair_tokens = tokens
            else:
                pair_tokens = tokens[1:] if len(tokens) > 1 else []

            if len(pair_tokens) % 2 != 0:
                raise ModelParseError(f"Malformed RHS entry: '{line}'", line_num=line_num, file_path=file_path)

            for k in range(0, len(pair_tokens), 2):
                rname = pair_tokens[k]
                val_str = pair_tokens[k + 1]
                if rname in constr_name_to_idx:
                    row_idx = constr_name_to_idx[rname]
                    try:
                        rhs[row_idx] = float(val_str)
                    except ValueError:
                        raise ModelParseError(
                            f"Invalid numeric RHS value '{val_str}' for row '{rname}'",
                            line_num=line_num,
                            file_path=file_path
                        )

        return rhs

    def _parse_bounds(
        self,
        lines: List[Tuple[int, str]],
        col_names: List[str],
        var_name_to_idx: Dict[str, int],
        marker_integers: Set[str],
        file_path: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[VarType]]:
        """
        Parses BOUNDS section.
        Default bounds in MPS:
          - Non-negative: 0 <= x < +inf
        Bound types supported:
          - LO: Lower bound
          - UP: Upper bound (if LO not specified, lower bound remains 0)
          - FX: Fixed variable (l = u = val)
          - FR: Free variable (-inf <= x <= +inf)
          - BV: Binary variable (l = 0, u = 1, VarType.BINARY)
          - LI: Integer lower bound (VarType.INTEGER)
          - UI: Integer upper bound (VarType.INTEGER)
        """
        num_vars = len(col_names)
        l_var = np.zeros(num_vars, dtype=np.float64)
        u_var = np.full(num_vars, np.inf, dtype=np.float64)

        var_types: List[VarType] = [
            VarType.INTEGER if name in marker_integers else VarType.CONTINUOUS
            for name in col_names
        ]

        if not lines or len(lines) <= 1:
            return l_var, u_var, var_types

        for line_num, line in lines[1:]:
            tokens = line.split()
            if not tokens:
                continue

            btype = tokens[0].upper()
            
            # Format could be:
            # TYPE [BOUND_NAME] COL_NAME [VALUE]
            if len(tokens) == 3:
                # TYPE COL_NAME VALUE or TYPE BOUND_NAME COL_NAME (e.g. for FR/BV)
                if tokens[1] in var_name_to_idx:
                    col_name = tokens[1]
                    val_str = tokens[2]
                elif tokens[2] in var_name_to_idx:
                    col_name = tokens[2]
                    val_str = "0"
                else:
                    col_name = tokens[1]
                    val_str = tokens[2]
            elif len(tokens) >= 4:
                col_name = tokens[2]
                val_str = tokens[3]
            elif len(tokens) == 2:
                # FR COL_NAME or BV COL_NAME
                col_name = tokens[1]
                val_str = "0"
            else:
                raise ModelParseError(f"Malformed BOUNDS entry: '{line}'", line_num=line_num, file_path=file_path)

            if col_name not in var_name_to_idx:
                raise ModelParseError(
                    f"Unknown variable '{col_name}' in BOUNDS section.",
                    line_num=line_num,
                    file_path=file_path
                )

            idx = var_name_to_idx[col_name]

            if btype == "LO":
                l_var[idx] = float(val_str)
            elif btype == "UP":
                u_var[idx] = float(val_str)
            elif btype == "FX":
                val = float(val_str)
                l_var[idx] = val
                u_var[idx] = val
            elif btype == "FR":
                l_var[idx] = -np.inf
                u_var[idx] = np.inf
            elif btype == "BV":
                l_var[idx] = 0.0
                u_var[idx] = 1.0
                var_types[idx] = VarType.BINARY
            elif btype == "LI":
                l_var[idx] = float(val_str)
                var_types[idx] = VarType.INTEGER
            elif btype == "UI":
                u_var[idx] = float(val_str)
                var_types[idx] = VarType.INTEGER
            else:
                raise ModelParseError(f"Unsupported bound type '{btype}' in BOUNDS section.", line_num=line_num, file_path=file_path)

        return l_var, u_var, var_types

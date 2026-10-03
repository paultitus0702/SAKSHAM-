"""
CPLEX LP Format Parser for SAKSHAM.

Parses standard CPLEX LP format (.lp) files and text strings into CanonicalModel.
Handles MIN/MAX, linear constraints (<=, =, >=), variable bounds, binaries, and integers.
"""
from pathlib import Path
import re
from typing import Union, Optional, List, Dict, Tuple, Set
import numpy as np
from scipy import sparse

from .base import AbstractModelParser, ModelParseError
from ..canonical.model import CanonicalModel, ObjectiveSense, ConstraintSense, VarType
from ..canonical.validator import ModelValidator


class LPParser(AbstractModelParser):
    """
    Parser for CPLEX-style LP formulation files.
    """

    def parse_file(self, file_path: Union[str, Path]) -> CanonicalModel:
        """Parse LP file from filesystem."""
        path = Path(file_path)
        if not path.is_file():
            raise ModelParseError(f"LP file not found: {file_path}", file_path=str(file_path))
        
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ModelParseError(f"Failed to read LP file: {str(e)}", file_path=str(file_path))

        model_name = path.stem
        return self.parse_string(content, model_name=model_name, file_path=str(file_path))

    def parse_string(
        self,
        content: str,
        model_name: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> CanonicalModel:
        """Parse LP formulation from raw text string."""
        raw_lines = content.splitlines()
        
        # 1. Clean comments and strip trailing whitespace
        cleaned_lines: List[Tuple[int, str]] = []  # (1-based line_num, cleaned_text)
        for i, line in enumerate(raw_lines, start=1):
            # Strip comments starting with \
            comment_idx = line.find("\\")
            if comment_idx != -1:
                line = line[:comment_idx]
            cleaned = line.strip()
            if cleaned:
                cleaned_lines.append((i, cleaned))

        if not cleaned_lines:
            raise ModelParseError("LP content is empty or contains only comments.", file_path=file_path)

        # 2. Partition into sections
        sections = self._partition_sections(cleaned_lines, file_path)

        # 3. Parse Objective
        sense, obj_name, obj_terms, obj_c0 = self._parse_objective(sections.get("OBJECTIVE", []), file_path)
        if model_name is None:
            model_name = obj_name or "lp_model"

        # 4. Parse Constraints
        constrs = self._parse_constraints(sections.get("CONSTRAINTS", []), file_path)

        # Collect all variable names encountered in objective and constraints
        encountered_vars: Set[str] = set(obj_terms.keys())
        for _, terms, _, _ in constrs:
            encountered_vars.update(terms.keys())

        # 5. Parse Binaries and Integers to assign variable types
        binaries = self._parse_var_list(sections.get("BINARIES", []), file_path)
        integers = self._parse_var_list(sections.get("GENERALS", []), file_path)
        encountered_vars.update(binaries)
        encountered_vars.update(integers)

        # Deterministic variable ordering
        var_names: List[str] = sorted(list(encountered_vars))
        num_vars = len(var_names)
        if num_vars == 0:
            raise ModelParseError("No variables found in LP model.", file_path=file_path)
        var_name_to_idx = {name: idx for idx, name in enumerate(var_names)}

        # Assign VarTypes
        var_types: List[VarType] = []
        for name in var_names:
            if name in binaries:
                var_types.append(VarType.BINARY)
            elif name in integers:
                var_types.append(VarType.INTEGER)
            else:
                var_types.append(VarType.CONTINUOUS)

        # 6. Parse Bounds
        l_var, u_var = self._parse_bounds(
            sections.get("BOUNDS", []),
            var_names,
            var_types,
            var_name_to_idx,
            file_path
        )

        # 7. Construct Objective vector c
        c = np.zeros(num_vars, dtype=np.float64)
        for var_name, coeff in obj_terms.items():
            idx = var_name_to_idx[var_name]
            c[idx] += coeff

        # 8. Construct Constraint Matrix A, senses, and rhs
        num_constrs = len(constrs)
        constr_names: List[str] = []
        constraint_senses: List[ConstraintSense] = []
        rhs = np.zeros(num_constrs, dtype=np.float64)

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        seen_constr_names: Set[str] = set()
        for row_idx, (cname, terms, csense, crhs) in enumerate(constrs):
            if cname in seen_constr_names:
                raise ModelParseError(f"Duplicate constraint name '{cname}' in LP model.", file_path=file_path)
            seen_constr_names.add(cname)
            constr_names.append(cname)
            constraint_senses.append(csense)
            rhs[row_idx] = crhs

            for var_name, coeff in terms.items():
                col_idx = var_name_to_idx[var_name]
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(coeff)

        A = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(num_constrs, num_vars),
            dtype=np.float64
        )

        model = CanonicalModel(
            name=model_name,
            sense=sense,
            c0=obj_c0,
            c=c,
            A=A,
            rhs=rhs,
            constraint_senses=constraint_senses,
            l_var=l_var,
            u_var=u_var,
            var_types=var_types,
            var_names=var_names,
            constr_names=constr_names,
            var_name_to_idx=var_name_to_idx,
            constr_name_to_idx={name: idx for idx, name in enumerate(constr_names)}
        )

        # Validate resulting CanonicalModel
        ModelValidator.validate(model, raise_on_error=True)
        return model

    def _partition_sections(
        self,
        cleaned_lines: List[Tuple[int, str]],
        file_path: Optional[str]
    ) -> Dict[str, List[Tuple[int, str]]]:
        """Partitions lines into sections: OBJECTIVE, CONSTRAINTS, BOUNDS, BINARIES, GENERALS."""
        sections: Dict[str, List[Tuple[int, str]]] = {}
        current_section: Optional[str] = None

        for line_num, line in cleaned_lines:
            upper = line.upper()

            # Check section keywords
            if upper in ["MINIMIZE", "MIN", "MAXIMIZE", "MAX"] or upper.startswith("MIN") or upper.startswith("MAX"):
                current_section = "OBJECTIVE"
                sections[current_section] = [(line_num, line)]
            elif upper in ["SUBJECT TO", "SUCH THAT", "ST", "S.T.", "ST."]:
                current_section = "CONSTRAINTS"
                sections[current_section] = []
            elif upper in ["BOUNDS", "BOUND"]:
                current_section = "BOUNDS"
                sections[current_section] = []
            elif upper in ["BINARY", "BINARIES", "BIN"]:
                current_section = "BINARIES"
                sections[current_section] = []
            elif upper in ["GENERAL", "GENERALS", "GEN", "INTEGER", "INTEGERS", "INT"]:
                current_section = "GENERALS"
                sections[current_section] = []
            elif upper == "END":
                break
            else:
                if current_section is None:
                    # If section header was missing on first line, check if it's objective
                    raise ModelParseError(
                        f"Expected section header (MINIMIZE/MAXIMIZE), got: '{line}'",
                        line_num=line_num,
                        line_text=line,
                        file_path=file_path
                    )
                sections[current_section].append((line_num, line))

        return sections

    def _parse_objective(
        self,
        lines: List[Tuple[int, str]],
        file_path: Optional[str]
    ) -> Tuple[ObjectiveSense, Optional[str], Dict[str, float], float]:
        """Parses objective direction and expression terms."""
        if not lines:
            raise ModelParseError("Missing objective section in LP file.", file_path=file_path)

        first_line_num, first_line = lines[0]
        header = first_line.upper().split()[0]
        
        if header in ["MINIMIZE", "MIN"]:
            sense = ObjectiveSense.MINIMIZE
        elif header in ["MAXIMIZE", "MAX"]:
            sense = ObjectiveSense.MAXIMIZE
        else:
            raise ModelParseError(f"Invalid objective header: '{header}'", line_num=first_line_num, file_path=file_path)

        # The rest of first_line plus remaining lines form the objective expression
        expr_parts = [first_line[len(header):].strip()]
        for _, line in lines[1:]:
            expr_parts.append(line)
        full_expr = " ".join(p for p in expr_parts if p).strip()

        # Check for optional objective label (e.g., "cost: 2 x1 + 3 x2")
        obj_name: Optional[str] = None
        if ":" in full_expr:
            colon_idx = full_expr.find(":")
            candidate_name = full_expr[:colon_idx].strip()
            # Ensure label does not contain operators
            if candidate_name and not any(op in candidate_name for op in ["+", "-", "<", ">", "="]):
                obj_name = candidate_name
                full_expr = full_expr[colon_idx + 1:].strip()

        terms, constant = self._parse_linear_expression(full_expr, file_path, context="objective")
        return sense, obj_name, terms, constant

    def _parse_constraints(
        self,
        lines: List[Tuple[int, str]],
        file_path: Optional[str]
    ) -> List[Tuple[str, Dict[str, float], ConstraintSense, float]]:
        """Parses list of constraints into (name, terms, sense, rhs)."""
        constrs: List[Tuple[str, Dict[str, float], ConstraintSense, float]] = []
        if not lines:
            return constrs

        # Merge lines into single stream while keeping track of constraint boundaries
        full_text = " ".join(line for _, line in lines)
        
        # Split constraints: each constraint ends with (<=, =, >=, <, >) followed by a number
        # Pattern matches: [optional_name:] LHS (<=|= |>=|<|>) RHS
        # We can tokenize or scan
        pattern = re.compile(
            r'\s*(?:(?P<name>[a-zA-Z0-9_#\.\-]+)\s*:)?'  # Optional label
            r'(?P<lhs>[^<>=]+)'                       # LHS expression
            r'(?P<sense><=|>=|==|=|<|>)'             # Relational operator
            r'(?P<rhs>\s*[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)' # RHS value
        )

        matches = list(pattern.finditer(full_text))
        if not matches:
            raise ModelParseError("No valid constraints found in SUBJECT TO section.", file_path=file_path)

        for i, match in enumerate(matches):
            raw_name = match.group("name")
            lhs = match.group("lhs").strip()
            raw_sense = match.group("sense").strip()
            raw_rhs = match.group("rhs").strip()

            cname = raw_name if raw_name else f"c_{i+1}"
            
            # Map sense
            if raw_sense in ["<=", "<"]:
                csense = ConstraintSense.LESS_EQUAL
            elif raw_sense in ["==", "="]:
                csense = ConstraintSense.EQUAL
            elif raw_sense in [">=", ">"]:
                csense = ConstraintSense.GREATER_EQUAL
            else:
                raise ModelParseError(f"Unknown constraint sense: '{raw_sense}'", file_path=file_path)

            try:
                crhs = float(raw_rhs)
            except ValueError:
                raise ModelParseError(f"Invalid constraint RHS value: '{raw_rhs}' in constraint '{cname}'", file_path=file_path)

            terms, lhs_const = self._parse_linear_expression(lhs, file_path, context=f"constraint '{cname}'")
            # If LHS has a constant term, move it to RHS: LHS_terms + lhs_const <= crhs  =>  LHS_terms <= crhs - lhs_const
            crhs -= lhs_const

            if not terms:
                raise ModelParseError(f"Constraint '{cname}' has no variables in LHS.", file_path=file_path)

            constrs.append((cname, terms, csense, crhs))

        return constrs

    def _parse_bounds(
        self,
        lines: List[Tuple[int, str]],
        var_names: List[str],
        var_types: List[VarType],
        var_name_to_idx: Dict[str, int],
        file_path: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parses variable bounds section.
        Default bounds in CPLEX LP:
          - Continuous: [0, +inf)
          - Binary: [0, 1]
          - Integer: [0, +inf)
        """
        num_vars = len(var_names)
        l_var = np.zeros(num_vars, dtype=np.float64)
        u_var = np.full(num_vars, np.inf, dtype=np.float64)

        # Set binary defaults
        for idx, vt in enumerate(var_types):
            if vt == VarType.BINARY:
                l_var[idx] = 0.0
                u_var[idx] = 1.0

        for line_num, line in lines:
            line = line.strip()
            if not line:
                continue

            # Free variable: "x1 free"
            free_match = re.match(r'^([a-zA-Z0-9_#\.\-]+)\s+(?:FREE|free)$', line)
            if free_match:
                vname = free_match.group(1)
                if vname not in var_name_to_idx:
                    raise ModelParseError(f"Bound declared for unknown variable '{vname}'", line_num=line_num, file_path=file_path)
                idx = var_name_to_idx[vname]
                l_var[idx] = -np.inf
                u_var[idx] = np.inf
                continue

            # Double-sided bound: "2 <= x1 <= 10" or "-inf <= x1 <= 5"
            range_match = re.match(
                r'^([-+]?[0-9\.]+(?:[eE][-+]?[0-9]+)?|-inf|-infinity)\s*<=\s*([a-zA-Z0-9_#\.\-]+)\s*<=\s*([-+]?[0-9\.]+(?:[eE][-+]?[0-9]+)?|\+?inf|\+?infinity)$',
                line,
                re.IGNORECASE
            )
            if range_match:
                lb_str, vname, ub_str = range_match.groups()
                if vname not in var_name_to_idx:
                    raise ModelParseError(f"Bound declared for unknown variable '{vname}'", line_num=line_num, file_path=file_path)
                idx = var_name_to_idx[vname]
                l_var[idx] = -np.inf if "INF" in lb_str.upper() else float(lb_str)
                u_var[idx] = np.inf if "INF" in ub_str.upper() else float(ub_str)
                continue

            # Equality bound: "x1 = 5"
            eq_match = re.match(r'^([a-zA-Z0-9_#\.\-]+)\s*=\s*([-+]?[0-9\.]+(?:[eE][-+]?[0-9]+)?)$', line)
            if eq_match:
                vname, val_str = eq_match.groups()
                if vname not in var_name_to_idx:
                    raise ModelParseError(f"Bound declared for unknown variable '{vname}'", line_num=line_num, file_path=file_path)
                idx = var_name_to_idx[vname]
                val = float(val_str)
                l_var[idx] = val
                u_var[idx] = val
                continue

            # Upper bound: "x1 <= 10" or "x1 < 10"
            ub_match = re.match(r'^([a-zA-Z0-9_#\.\-]+)\s*<=?\s*([-+]?[0-9\.]+(?:[eE][-+]?[0-9]+)?|\+?inf|\+?infinity)$', line, re.IGNORECASE)
            if ub_match:
                vname, val_str = ub_match.groups()
                if vname not in var_name_to_idx:
                    raise ModelParseError(f"Bound declared for unknown variable '{vname}'", line_num=line_num, file_path=file_path)
                idx = var_name_to_idx[vname]
                u_var[idx] = np.inf if "INF" in val_str.upper() else float(val_str)
                continue

            # Lower bound: "x1 >= 5" or "5 <= x1"
            lb_match1 = re.match(r'^([a-zA-Z0-9_#\.\-]+)\s*>=?\s*([-+]?[0-9\.]+(?:[eE][-+]?[0-9]+)?|-inf|-infinity)$', line, re.IGNORECASE)
            if lb_match1:
                vname, val_str = lb_match1.groups()
                if vname not in var_name_to_idx:
                    raise ModelParseError(f"Bound declared for unknown variable '{vname}'", line_num=line_num, file_path=file_path)
                idx = var_name_to_idx[vname]
                l_var[idx] = -np.inf if "INF" in val_str.upper() else float(val_str)
                continue

            lb_match2 = re.match(r'^([-+]?[0-9\.]+(?:[eE][-+]?[0-9]+)?|-inf|-infinity)\s*<=?\s*([a-zA-Z0-9_#\.\-]+)$', line, re.IGNORECASE)
            if lb_match2:
                val_str, vname = lb_match2.groups()
                if vname not in var_name_to_idx:
                    raise ModelParseError(f"Bound declared for unknown variable '{vname}'", line_num=line_num, file_path=file_path)
                idx = var_name_to_idx[vname]
                l_var[idx] = -np.inf if "INF" in val_str.upper() else float(val_str)
                continue

            raise ModelParseError(f"Unrecognized bound specification: '{line}'", line_num=line_num, file_path=file_path)

        return l_var, u_var

    def _parse_var_list(self, lines: List[Tuple[int, str]], file_path: Optional[str]) -> Set[str]:
        """Parses space-separated list of variable names (for BINARY or GENERAL sections)."""
        vars_set: Set[str] = set()
        for _, line in lines:
            for token in line.split():
                vars_set.add(token.strip())
        return vars_set

    def _parse_linear_expression(
        self,
        expr_str: str,
        file_path: Optional[str],
        context: str = "expression"
    ) -> Tuple[Dict[str, float], float]:
        """
        Parses a linear expression string into {variable_name: coefficient} and constant offset.
        Examples:
            "3 x1 - 4.5 x2 + x3 - 2"  ->  ({"x1": 3.0, "x2": -4.5, "x3": 1.0}, -2.0)
        """
        terms: Dict[str, float] = {}
        constant: float = 0.0

        if not expr_str.strip():
            return terms, constant

        # Tokenize by finding all signs, numbers, and identifiers
        # Regex captures terms with leading sign, number, variable
        token_regex = re.compile(r'([+-]?)\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)?\s*([a-zA-Z0-9_#\.\-]+)?')
        
        pos = 0
        s = expr_str.strip()
        while pos < len(s):
            # Skip spaces
            while pos < len(s) and s[pos].isspace():
                pos += 1
            if pos >= len(s):
                break

            m = token_regex.match(s, pos)
            if not m or m.end() == pos:
                raise ModelParseError(
                    f"Syntax error parsing linear {context} near '{s[pos:pos+20]}'",
                    file_path=file_path
                )

            sign_str = m.group(1) or "+"
            num_str = m.group(2)
            var_name = m.group(3)

            sign = -1.0 if sign_str == "-" else 1.0

            if var_name:
                coeff = (float(num_str) if num_str else 1.0) * sign
                terms[var_name] = terms.get(var_name, 0.0) + coeff
            elif num_str:
                constant += float(num_str) * sign

            pos = m.end()

        return terms, constant

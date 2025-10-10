import datetime as dt
import ast
import re
import polars as pl
from typing import Any, Optional
from arcticdb import Arctic, LibraryOptions, QueryBuilder, LazyDataFrame, OutputFormat
from arcticdb.version_store.library import Library
from typing import Iterator

class PolarsToArcticDBTranslator:
    """
    Translates Polars expressions to ArcticDB QueryBuilder operations.
    
    Usage:
        translator = PolarsToArcticDBTranslator()
        qb = translator.translate(polars_expr, query_builder)
    """
    
    def __init__(self):
        self.qb = None
        
    def translate(self, polars_expr: str, query_builder: Any) -> Any:
        """
        Translate a Polars expression string to ArcticDB QueryBuilder operations.
        
        Args:
            polars_expr: String representation of Polars expression
            query_builder: ArcticDB QueryBuilder instance
            
        Returns:
            Modified QueryBuilder instance
        """
        self.qb = query_builder
        
        # Clean the expression - remove surrounding brackets if present
        expr = polars_expr.strip()
        if expr.startswith('[') and expr.endswith(']'):
            expr = expr[1:-1].strip()
        
        # Preprocess to handle Polars-specific notation like [dyn int: 2]
        expr = self._preprocess_expression(expr)
        
        # Parse the expression
        try:
            tree = ast.parse(expr, mode='eval')
            self._process_node(tree.body)
        except SyntaxError as e:
            raise ValueError(f"Invalid Polars expression: {polars_expr}") from e
        
        return self.qb
    
    def _preprocess_expression(self, expr: str) -> str:
        """
        Preprocess Polars expression to handle special notation.
        
        Converts patterns like [dyn int: 2] to just the value (2).
        """
        # Pattern to match [dyn type: value] or [lit type: value]
        pattern = r'\[(dyn|lit)\s+\w+:\s*([^\]]+)\]'
        
        def replace_dynamic(match):
            return match.group(2).strip()
        
        return re.sub(pattern, replace_dynamic, expr)
    
    def _process_node(self, node: ast.AST) -> Any:
        """Process an AST node and apply corresponding ArcticDB operation."""
        
        if isinstance(node, ast.Call):
            return self._process_call(node)
        elif isinstance(node, ast.Attribute):
            return self._process_attribute(node)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Compare):
            return self._process_compare(node)
        elif isinstance(node, ast.BinOp):
            return self._process_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._process_unaryop(node)
        elif isinstance(node, ast.List):
            return [self._process_node(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._process_node(elt) for elt in node.elts)
        else:
            raise NotImplementedError(f"Node type {type(node).__name__} not supported")
    
    def _process_call(self, node: ast.Call) -> Any:
        """Process function calls (e.g., pl.col(), methods)."""
        
        if isinstance(node.func, ast.Attribute):
            # Method call like pl.col('x').sum()
            obj = self._process_node(node.func.value)
            method = node.func.attr
            args = [self._process_node(arg) for arg in node.args]
            kwargs = {kw.arg: self._process_node(kw.value) for kw in node.keywords}
            
            return self._apply_method(obj, method, args, kwargs)
        elif isinstance(node.func, ast.Name):
            # Function call like col('x')
            func_name = node.func.id
            args = [self._process_node(arg) for arg in node.args]
            
            if func_name == 'col':
                return args[0] if args else None
            
        return None
    
    def _process_attribute(self, node: ast.Attribute) -> Any:
        """Process attribute access like pl.col or obj.attr."""
        
        obj = self._process_node(node.value)
        attr = node.attr
        
        # Handle pl.col pattern
        if obj == 'pl' and attr == 'col':
            return 'col'
        
        return f"{obj}.{attr}"
    
    def _process_compare(self, node: ast.Compare) -> Any:
        """Process comparison operations and apply filters."""
        
        left = self._process_node(node.left)
        
        # Handle multiple comparisons
        for op, comparator in zip(node.ops, node.comparators):
            right = self._process_node(comparator)
            
            if isinstance(op, ast.Eq):
                self.qb = self.qb.filter(left, '==', right)
            elif isinstance(op, ast.NotEq):
                self.qb = self.qb.filter(left, '!=', right)
            elif isinstance(op, ast.Lt):
                self.qb = self.qb.filter(left, '<', right)
            elif isinstance(op, ast.LtE):
                self.qb = self.qb.filter(left, '<=', right)
            elif isinstance(op, ast.Gt):
                self.qb = self.qb.filter(left, '>', right)
            elif isinstance(op, ast.GtE):
                self.qb = self.qb.filter(left, '>=', right)
            elif isinstance(op, ast.In):
                self.qb = self.qb.filter(left, 'isin', right)
            elif isinstance(op, ast.NotIn):
                self.qb = self.qb.filter(left, 'isnotin', right)
        
        return self.qb
    
    def _process_binop(self, node: ast.BinOp) -> Any:
        """Process binary operations."""
        
        left = self._process_node(node.left)
        right = self._process_node(node.right)
        op = node.op
        
        # For column operations, apply them via apply method
        if isinstance(op, ast.Add):
            self.qb = self.qb.apply(f"{left}_plus_{right}", f"{left} + {right}")
        elif isinstance(op, ast.Sub):
            self.qb = self.qb.apply(f"{left}_minus_{right}", f"{left} - {right}")
        elif isinstance(op, ast.Mult):
            self.qb = self.qb.apply(f"{left}_times_{right}", f"{left} * {right}")
        elif isinstance(op, ast.Div):
            self.qb = self.qb.apply(f"{left}_div_{right}", f"{left} / {right}")
        elif isinstance(op, ast.Mod):
            self.qb = self.qb.apply(f"{left}_mod_{right}", f"{left} % {right}")
        elif isinstance(op, ast.Pow):
            self.qb = self.qb.apply(f"{left}_pow_{right}", f"{left} ** {right}")
        
        return self.qb
    
    def _process_unaryop(self, node: ast.UnaryOp) -> Any:
        """Process unary operations."""
        
        operand = self._process_node(node.operand)
        op = node.op
        
        if isinstance(op, ast.Not):
            # Invert filter condition
            pass
        elif isinstance(op, ast.USub):
            self.qb = self.qb.apply(f"neg_{operand}", f"-{operand}")
        
        return self.qb
    
    def _apply_method(self, obj: Any, method: str, args: list, kwargs: dict) -> Any:
        """Apply Polars methods as ArcticDB operations."""
        
        col_name = obj if isinstance(obj, str) else None
        
        # Math methods
        if method == 'pow':
            exponent = args[0] if args else 2
            self.qb = self.qb.apply(f"{col_name}_pow_{exponent}", f"{col_name} ** {exponent}")
        elif method == 'sqrt':
            self.qb = self.qb.apply(f"{col_name}_sqrt", f"{col_name} ** 0.5")
        elif method == 'abs':
            self.qb = self.qb.apply(f"{col_name}_abs", f"abs({col_name})")
        elif method == 'round':
            decimals = args[0] if args else 0
            self.qb = self.qb.apply(f"{col_name}_round", f"round({col_name}, {decimals})")
        
        # Aggregation methods
        elif method == 'sum':
            self.qb = self.qb.groupby(args[0] if args else None).agg({col_name: 'sum'}) if args else self.qb.agg({col_name: 'sum'})
        elif method == 'mean':
            self.qb = self.qb.groupby(args[0] if args else None).agg({col_name: 'mean'}) if args else self.qb.agg({col_name: 'mean'})
        elif method == 'min':
            self.qb = self.qb.groupby(args[0] if args else None).agg({col_name: 'min'}) if args else self.qb.agg({col_name: 'min'})
        elif method == 'max':
            self.qb = self.qb.groupby(args[0] if args else None).agg({col_name: 'max'}) if args else self.qb.agg({col_name: 'max'})
        elif method == 'count':
            self.qb = self.qb.groupby(args[0] if args else None).agg({col_name: 'count'}) if args else self.qb.agg({col_name: 'count'})
        elif method == 'std':
            self.qb = self.qb.groupby(args[0] if args else None).agg({col_name: 'std'}) if args else self.qb.agg({col_name: 'std'})
        elif method == 'var':
            self.qb = self.qb.groupby(args[0] if args else None).agg({col_name: 'var'}) if args else self.qb.agg({col_name: 'var'})
        
        # String methods
        elif method == 'str':
            # String namespace - return for chaining
            return col_name
        elif method in ['upper', 'lower', 'strip', 'lstrip', 'rstrip']:
            self.qb = self.qb.apply(f"{col_name}_{method}", f"{col_name}.str.{method}()")
        elif method == 'contains':
            pattern = args[0] if args else ''
            self.qb = self.qb.filter(col_name, 'contains', pattern)
        
        # Date/time methods
        elif method == 'dt':
            # Datetime namespace - return for chaining
            return col_name
        elif method in ['year', 'month', 'day', 'hour', 'minute', 'second']:
            self.qb = self.qb.apply(f"{col_name}_{method}", f"{col_name}.dt.{method}")
        
        # Casting
        elif method == 'cast':
            dtype = args[0] if args else None
            self.qb = self.qb.apply(f"{col_name}_cast", f"{col_name}.astype('{dtype}')")
        
        # Null handling
        elif method == 'is_null':
            self.qb = self.qb.filter(col_name, 'isnull')
        elif method == 'is_not_null':
            self.qb = self.qb.filter(col_name, 'notnull')
        elif method == 'fill_null':
            value = args[0] if args else None
            self.qb = self.qb.apply(f"{col_name}_filled", f"{col_name}.fillna({value})")
        
        # Alias
        elif method == 'alias':
            new_name = args[0] if args else col_name
            self.qb = self.qb.apply(new_name, col_name)
        
        return self.qb

def parse_schema(
    lib: Library,
    symbol: str,
    as_of: int | str | dt.datetime | None = None
) -> pl.Schema:
    arrow_df = lib.read(symbol, as_of=as_of, output_format=OutputFormat.EXPERIMENTAL_ARROW, row_range=((0,1))).data
    return pl.Schema(arrow_df.schema) 

def scan_arcticdb(
    uri: str,
    lib_name: str,
    symbol: str,
    as_of: int | str | dt.datetime | None = None
) -> pl.LazyFrame:

    ac = Arctic(uri)
    lib = ac.get_library(lib_name)

    schema = parse_schema(lib, symbol, as_of)

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None
    ) -> Iterator[pl.DataFrame]:

        if predicate is not None:
            print(str(predicate))
        else:
            print("No Predicate")

        if with_columns:
            print(with_columns)
        else:
            print("No column")
    
        # TODO: convert predicate to QueryBuilder and pass it to read
        lazy_df = lib.read(symbol, as_of = as_of, columns = with_columns, lazy = True, output_format=OutputFormat.EXPERIMENTAL_ARROW)

        if batch_size is None:
            batch_size = 1000

        if n_rows is not None:
            batch_size = min(batch_size, n_rows)
        
        read_idx = 0
        while n_rows is None or n_rows > 0:
            lazy_df_slice = lazy_df.row_range((read_idx, read_idx + batch_size))
            read_idx += batch_size
            arrow_df = lazy_df_slice.collect().data
            if n_rows is not None:
                n_rows -= arrow_df.num_rows
            elif arrow_df.num_rows < batch_size:
                n_rows = 0

            yield pl.from_arrow(arrow_df)

    return pl.io.plugins.register_io_source(io_source=source_generator, schema = schema)    
